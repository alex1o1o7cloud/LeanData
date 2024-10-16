import Mathlib

namespace NUMINAMATH_CALUDE_a_formula_T_formula_l3194_319469

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := sorry

-- Define the sum of the first n terms
def S (n : ℕ) : ℚ := sorry

-- Define the conditions
axiom S3_eq_0 : S 3 = 0
axiom S5_eq_neg5 : S 5 = -5

-- Theorem 1: General formula for a_n
theorem a_formula (n : ℕ) : a n = -n + 2 := sorry

-- Define the sequence 1 / (a_{2n-1} * a_{2n+1})
def b (n : ℕ) : ℚ := 1 / (a (2*n - 1) * a (2*n + 1))

-- Define the sum of the first n terms of b
def T (n : ℕ) : ℚ := sorry

-- Theorem 2: Sum of the first n terms of b
theorem T_formula (n : ℕ) : T n = n / (1 - 2*n) := sorry

end NUMINAMATH_CALUDE_a_formula_T_formula_l3194_319469


namespace NUMINAMATH_CALUDE_max_sum_with_length_constraint_l3194_319418

-- Define the length of an integer as the number of prime factors
def length (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem max_sum_with_length_constraint :
  ∀ x y : ℕ,
    x > 1 →
    y > 1 →
    length x + length y ≤ 16 →
    x + 3 * y ≤ 98306 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_length_constraint_l3194_319418


namespace NUMINAMATH_CALUDE_pentadecagon_diagonals_l3194_319428

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentadecagon is a polygon with 15 sides -/
def pentadecagon_sides : ℕ := 15

theorem pentadecagon_diagonals :
  num_diagonals pentadecagon_sides = 90 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_diagonals_l3194_319428


namespace NUMINAMATH_CALUDE_percentage_relationships_l3194_319405

theorem percentage_relationships (a b c d e f g : ℝ) 
  (h1 : d = 0.22 * b) 
  (h2 : d = 0.35 * f) 
  (h3 : e = 0.27 * a) 
  (h4 : e = 0.60 * f) 
  (h5 : c = 0.14 * a) 
  (h6 : c = 0.40 * b) 
  (h7 : d = 2 * c) 
  (h8 : g = 3 * e) : 
  g = 0.81 * a ∧ b = 0.7 * a ∧ f = 0.45 * a := by
  sorry


end NUMINAMATH_CALUDE_percentage_relationships_l3194_319405


namespace NUMINAMATH_CALUDE_rectangle_width_l3194_319463

theorem rectangle_width (area : ℝ) (perimeter : ℝ) (width : ℝ) (length : ℝ) :
  area = 50 →
  perimeter = 30 →
  area = length * width →
  perimeter = 2 * (length + width) →
  width = 5 ∨ width = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3194_319463


namespace NUMINAMATH_CALUDE_average_songs_theorem_l3194_319477

/-- Represents a band's performance schedule --/
structure BandPerformance where
  repertoire : ℕ
  first_set : ℕ
  second_set : ℕ
  encores : ℕ

/-- Calculates the average number of songs for the remaining sets --/
def average_remaining_songs (b : BandPerformance) : ℚ :=
  let songs_played := b.first_set + b.second_set + b.encores
  let remaining_songs := b.repertoire - songs_played
  let remaining_sets := 3
  (remaining_songs : ℚ) / remaining_sets

/-- Theorem stating the average number of songs for the remaining sets --/
theorem average_songs_theorem (b : BandPerformance) 
  (h1 : b.repertoire = 50)
  (h2 : b.first_set = 8)
  (h3 : b.second_set = 12)
  (h4 : b.encores = 4) :
  average_remaining_songs b = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_songs_theorem_l3194_319477


namespace NUMINAMATH_CALUDE_books_sold_in_garage_sale_l3194_319434

theorem books_sold_in_garage_sale 
  (initial_books : ℕ) 
  (books_given_to_friend : ℕ) 
  (remaining_books : ℕ) 
  (h1 : initial_books = 108) 
  (h2 : books_given_to_friend = 35) 
  (h3 : remaining_books = 62) :
  initial_books - books_given_to_friend - remaining_books = 11 := by
sorry

end NUMINAMATH_CALUDE_books_sold_in_garage_sale_l3194_319434


namespace NUMINAMATH_CALUDE_common_tangents_possible_values_l3194_319445

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- The number of common tangent lines between two circles -/
def num_common_tangents (c1 c2 : Circle) : ℕ := sorry

/-- Theorem stating the possible values for the number of common tangents -/
theorem common_tangents_possible_values (c1 c2 : Circle) (h : c1 ≠ c2) :
  ∃ n : ℕ, num_common_tangents c1 c2 = n ∧ n ∈ ({0, 1, 2, 3, 4} : Set ℕ) := by sorry

end NUMINAMATH_CALUDE_common_tangents_possible_values_l3194_319445


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3194_319450

theorem inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | x^2 - (a^2 + a)*x + a^3 < 0}
  (a = 0 ∨ a = 1 → solution_set = ∅) ∧
  (0 < a ∧ a < 1 → solution_set = {x : ℝ | a^2 < x ∧ x < a}) ∧
  ((a < 0 ∨ a > 1) → solution_set = {x : ℝ | a < x ∧ x < a^2}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3194_319450


namespace NUMINAMATH_CALUDE_inequality_always_true_l3194_319476

theorem inequality_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l3194_319476


namespace NUMINAMATH_CALUDE_jaylen_vegetables_l3194_319479

theorem jaylen_vegetables (x y z g k h : ℕ) : 
  x = 5 → 
  y = 2 → 
  z = 2 * k → 
  g = (h / 2) - 3 → 
  k = 2 → 
  h = 20 → 
  x + y + z + g = 18 := by
sorry

end NUMINAMATH_CALUDE_jaylen_vegetables_l3194_319479


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l3194_319427

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line l
structure Line where
  k : ℝ
  b : ℝ

-- Define the problem conditions
def circle_conditions (C : Circle) : Prop :=
  let (x, y) := C.center
  x > 0 ∧ y > 0 ∧  -- Center is in the first quadrant
  3 * x = y ∧      -- Center lies on the line 3x - y = 0
  C.radius = y ∧   -- Circle is tangent to x-axis
  (2 * Real.sqrt 7) ^ 2 = 4 * (C.radius ^ 2 - x ^ 2)  -- Chord length condition

def line_intersects_circle (l : Line) (C : Circle) : Prop :=
  ∃ (x y : ℝ), l.k * x - y - 2 * l.k + 5 = 0 ∧
                (x - C.center.1) ^ 2 + (y - C.center.2) ^ 2 = C.radius ^ 2

-- Theorem statement
theorem circle_and_line_properties :
  ∀ (C : Circle) (l : Line),
    circle_conditions C →
    line_intersects_circle l C →
    (∀ (x y : ℝ), (x - 1) ^ 2 + (y - 3) ^ 2 = 9 ↔ (x - C.center.1) ^ 2 + (y - C.center.2) ^ 2 = C.radius ^ 2) ∧
    (∃ (k : ℝ), l.k = k ∧ l.b = 5 - 2 * k) ∧
    (∃ (l_shortest : Line), 
      l_shortest.k = -1/2 ∧ 
      l_shortest.b = 6 ∧
      ∀ (l' : Line), l'.k ≠ -1/2 → 
        ∃ (d d' : ℝ), 
          d = (abs (l_shortest.k * C.center.1 - C.center.2 + l_shortest.b)) / Real.sqrt (l_shortest.k ^ 2 + 1) ∧
          d' = (abs (l'.k * C.center.1 - C.center.2 + l'.b)) / Real.sqrt (l'.k ^ 2 + 1) ∧
          d < d') ∧
    (∃ (shortest_chord : ℝ), shortest_chord = 4 ∧
      ∀ (l' : Line), l'.k ≠ -1/2 → 
        ∃ (chord : ℝ), 
          chord = 2 * Real.sqrt (C.radius ^ 2 - ((abs (l'.k * C.center.1 - C.center.2 + l'.b)) / Real.sqrt (l'.k ^ 2 + 1)) ^ 2) ∧
          chord > shortest_chord) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l3194_319427


namespace NUMINAMATH_CALUDE_chocolate_division_l3194_319464

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (friend_fraction : ℚ) :
  total_chocolate = 72 / 7 →
  num_piles = 8 →
  friend_fraction = 1 / 3 →
  friend_fraction * (total_chocolate / num_piles) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l3194_319464


namespace NUMINAMATH_CALUDE_right_triangular_pyramid_relation_l3194_319444

/-- Right triangular pyramid with pairwise perpendicular side edges -/
structure RightTriangularPyramid where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  h_pos : 0 < h

/-- The relationship between side edges and altitude in a right triangular pyramid -/
theorem right_triangular_pyramid_relation (p : RightTriangularPyramid) :
  1 / p.a ^ 2 + 1 / p.b ^ 2 + 1 / p.c ^ 2 = 1 / p.h ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_pyramid_relation_l3194_319444


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3194_319475

def a : Fin 2 → ℝ := ![3, 4]
def b : Fin 2 → ℝ := ![2, -1]

theorem perpendicular_vectors (x : ℝ) : 
  (∀ i : Fin 2, (a + x • b) i * (-b i) = 0) → x = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3194_319475


namespace NUMINAMATH_CALUDE_set_intersections_l3194_319478

-- Define the sets M, N, and P
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {y | ∃ x, y = 2 * x^2}
def P : Set (ℝ × ℝ) := {p | p.2 = p.1 - 1}

-- State the theorem
theorem set_intersections :
  ((Set.univ \ M) ∩ N) = {x | 0 ≤ x ∧ x ≤ 1} ∧
  (M.prod Set.univ ∩ P) = ∅ := by sorry

end NUMINAMATH_CALUDE_set_intersections_l3194_319478


namespace NUMINAMATH_CALUDE_money_division_l3194_319437

/-- 
Given an amount of money divided between three people in the ratio 3:7:12,
where the difference between the first two shares is 4000,
prove that the difference between the second and third shares is 5000.
-/
theorem money_division (total : ℝ) : 
  let p := (3 / 22) * total
  let q := (7 / 22) * total
  let r := (12 / 22) * total
  q - p = 4000 → r - q = 5000 := by
sorry

end NUMINAMATH_CALUDE_money_division_l3194_319437


namespace NUMINAMATH_CALUDE_fraction_five_times_seven_over_ten_l3194_319458

theorem fraction_five_times_seven_over_ten : (5 * 7) / 10 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_five_times_seven_over_ten_l3194_319458


namespace NUMINAMATH_CALUDE_tangent_line_at_2_range_of_m_for_three_roots_l3194_319423

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Theorem for the tangent line equation
theorem tangent_line_at_2 :
  ∃ (A B C : ℝ), A ≠ 0 ∧ 
  (∀ x y : ℝ, y = f x → (A * x + B * y + C = 0) ↔ x = 2) ∧
  A = 12 ∧ B = -1 ∧ C = -17 :=
sorry

-- Theorem for the range of m
theorem range_of_m_for_three_roots :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) ↔ 
  -3 < m ∧ m < -2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_range_of_m_for_three_roots_l3194_319423


namespace NUMINAMATH_CALUDE_factorization_mn_minus_mn_cubed_l3194_319425

theorem factorization_mn_minus_mn_cubed (m n : ℝ) : 
  m * n - m * n^3 = m * n * (1 + n) * (1 - n) := by sorry

end NUMINAMATH_CALUDE_factorization_mn_minus_mn_cubed_l3194_319425


namespace NUMINAMATH_CALUDE_complex_product_equals_369_l3194_319442

theorem complex_product_equals_369 (x : ℂ) : 
  x = Complex.exp (2 * Real.pi * I / 9) →
  (3 * x + x^3) * (3 * x^3 + x^9) * (3 * x^6 + x^18) = 369 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_369_l3194_319442


namespace NUMINAMATH_CALUDE_molecular_weight_3_moles_Fe2SO43_l3194_319431

/-- The atomic weight of Iron in g/mol -/
def atomic_weight_Fe : ℝ := 55.845

/-- The atomic weight of Sulfur in g/mol -/
def atomic_weight_S : ℝ := 32.065

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- The molecular weight of one mole of Iron(III) sulfate in g/mol -/
def molecular_weight_Fe2SO43 : ℝ :=
  2 * atomic_weight_Fe + 3 * (atomic_weight_S + 4 * atomic_weight_O)

/-- The number of moles of Iron(III) sulfate -/
def moles_Fe2SO43 : ℝ := 3

theorem molecular_weight_3_moles_Fe2SO43 :
  moles_Fe2SO43 * molecular_weight_Fe2SO43 = 1199.619 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_3_moles_Fe2SO43_l3194_319431


namespace NUMINAMATH_CALUDE_f_composition_l3194_319455

def f (x : ℝ) := 2 * x + 1

theorem f_composition (x : ℝ) : f (2 * x - 1) = 4 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_l3194_319455


namespace NUMINAMATH_CALUDE_sum_of_integers_l3194_319416

theorem sum_of_integers (a b c d : ℤ) 
  (eq1 : 2 * (a - b + c) = 10)
  (eq2 : 2 * (b - c + d) = 12)
  (eq3 : 2 * (c - d + a) = 6)
  (eq4 : 2 * (d - a + b) = 4) :
  a + b + c + d = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3194_319416


namespace NUMINAMATH_CALUDE_percentage_of_number_l3194_319454

theorem percentage_of_number (n : ℝ) : n * 0.001 = 0.24 → n = 240 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_number_l3194_319454


namespace NUMINAMATH_CALUDE_alexanders_paintings_l3194_319498

/-- The number of paintings at each new gallery given Alexander's drawing conditions -/
theorem alexanders_paintings (first_gallery_paintings : ℕ) (new_galleries : ℕ) 
  (pencils_per_painting : ℕ) (signing_pencils_per_gallery : ℕ) (total_pencils : ℕ) :
  first_gallery_paintings = 9 →
  new_galleries = 5 →
  pencils_per_painting = 4 →
  signing_pencils_per_gallery = 2 →
  total_pencils = 88 →
  ∃ (paintings_per_new_gallery : ℕ),
    paintings_per_new_gallery = 2 ∧
    total_pencils = 
      first_gallery_paintings * pencils_per_painting + 
      new_galleries * paintings_per_new_gallery * pencils_per_painting +
      (new_galleries + 1) * signing_pencils_per_gallery :=
by sorry

end NUMINAMATH_CALUDE_alexanders_paintings_l3194_319498


namespace NUMINAMATH_CALUDE_production_exceeds_target_in_seventh_year_l3194_319414

-- Define the initial production and growth rate
def initial_production : ℝ := 40000
def growth_rate : ℝ := 1.2

-- Define the target production
def target_production : ℝ := 120000

-- Define the function to calculate production after n years
def production (n : ℕ) : ℝ := initial_production * growth_rate ^ n

-- Theorem statement
theorem production_exceeds_target_in_seventh_year :
  ∀ n : ℕ, n < 7 → production n ≤ target_production ∧
  production 7 > target_production :=
sorry

end NUMINAMATH_CALUDE_production_exceeds_target_in_seventh_year_l3194_319414


namespace NUMINAMATH_CALUDE_village_population_growth_l3194_319491

theorem village_population_growth (c d : ℕ) : 
  c^3 + 180 = d^3 + 10 →                     -- Population condition for 2001
  (d + 1)^3 = d^3 + 180 →                    -- Population condition for 2011
  (((d + 1)^3 - c^3) * 100) / c^3 = 101 :=   -- Percent growth over 20 years
by
  sorry

end NUMINAMATH_CALUDE_village_population_growth_l3194_319491


namespace NUMINAMATH_CALUDE_square_root_squared_specific_square_root_squared_l3194_319404

theorem square_root_squared (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by sorry

theorem specific_square_root_squared : (Real.sqrt 978121) ^ 2 = 978121 := by
  apply square_root_squared
  norm_num


end NUMINAMATH_CALUDE_square_root_squared_specific_square_root_squared_l3194_319404


namespace NUMINAMATH_CALUDE_intersection_point_l3194_319429

/-- The x-coordinate of the intersection point of y = 2x - 1 and y = x + 1 -/
def x : ℝ := 2

/-- The y-coordinate of the intersection point of y = 2x - 1 and y = x + 1 -/
def y : ℝ := 3

/-- The first linear function -/
def f (x : ℝ) : ℝ := 2 * x - 1

/-- The second linear function -/
def g (x : ℝ) : ℝ := x + 1

theorem intersection_point :
  f x = y ∧ g x = y ∧ f x = g x :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l3194_319429


namespace NUMINAMATH_CALUDE_prob_at_least_one_six_given_different_outcomes_prob_at_least_one_six_is_one_third_l3194_319470

/-- The probability of rolling at least one 6 given two fair dice with different outcomes -/
theorem prob_at_least_one_six_given_different_outcomes : ℝ :=
let total_outcomes := 30  -- 6 * 5, as outcomes are different
let favorable_outcomes := 10  -- 5 (first die is 6) + 5 (second die is 6)
favorable_outcomes / total_outcomes

/-- Proof that the probability is 1/3 -/
theorem prob_at_least_one_six_is_one_third :
  prob_at_least_one_six_given_different_outcomes = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_six_given_different_outcomes_prob_at_least_one_six_is_one_third_l3194_319470


namespace NUMINAMATH_CALUDE_students_answering_both_correctly_l3194_319417

theorem students_answering_both_correctly 
  (total_students : ℕ) 
  (answered_q1 : ℕ) 
  (answered_q2 : ℕ) 
  (not_taken : ℕ) 
  (h1 : total_students = 30) 
  (h2 : answered_q1 = 25) 
  (h3 : answered_q2 = 22) 
  (h4 : not_taken = 5) :
  answered_q1 + answered_q2 - (total_students - not_taken) = 22 := by
  sorry

end NUMINAMATH_CALUDE_students_answering_both_correctly_l3194_319417


namespace NUMINAMATH_CALUDE_original_eq_hyperbola_and_ellipse_l3194_319452

-- Define the original equation
def original_equation (x y : ℝ) : Prop := y^4 - 4*x^4 = 2*y^2 - 1

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := y^2 - 2*x^2 = 1

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := y^2 + 2*x^2 = 1

-- Theorem stating that the original equation is equivalent to the union of a hyperbola and an ellipse
theorem original_eq_hyperbola_and_ellipse :
  ∀ x y : ℝ, original_equation x y ↔ (hyperbola_equation x y ∨ ellipse_equation x y) :=
by sorry

end NUMINAMATH_CALUDE_original_eq_hyperbola_and_ellipse_l3194_319452


namespace NUMINAMATH_CALUDE_division_remainder_problem_solution_l3194_319406

theorem division_remainder (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = divisor * quotient + remainder →
  remainder < divisor →
  dividend / divisor = quotient →
  dividend % divisor = remainder :=
by sorry

theorem problem_solution :
  14 / 3 = 4 →
  14 % 3 = 2 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_solution_l3194_319406


namespace NUMINAMATH_CALUDE_manager_employee_ratio_l3194_319471

theorem manager_employee_ratio (total_employees : ℕ) (female_managers : ℕ) 
  (h1 : total_employees = 750) (h2 : female_managers = 300) :
  (female_managers : ℚ) / total_employees = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_manager_employee_ratio_l3194_319471


namespace NUMINAMATH_CALUDE_bank_teller_coin_rolls_l3194_319472

theorem bank_teller_coin_rolls 
  (total_coins : ℕ) 
  (num_tellers : ℕ) 
  (coins_per_roll : ℕ) 
  (h1 : total_coins = 1000) 
  (h2 : num_tellers = 4) 
  (h3 : coins_per_roll = 25) : 
  (total_coins / num_tellers) / coins_per_roll = 10 := by
sorry

end NUMINAMATH_CALUDE_bank_teller_coin_rolls_l3194_319472


namespace NUMINAMATH_CALUDE_probability_zero_or_one_excellent_equals_formula_l3194_319492

def total_people : ℕ := 12
def excellent_students : ℕ := 5
def selected_people : ℕ := 5

def probability_zero_or_one_excellent : ℚ :=
  (Nat.choose (total_people - excellent_students) selected_people +
   Nat.choose excellent_students 1 * Nat.choose (total_people - excellent_students) (selected_people - 1)) /
  Nat.choose total_people selected_people

theorem probability_zero_or_one_excellent_equals_formula :
  probability_zero_or_one_excellent = 
  (Nat.choose (total_people - excellent_students) selected_people +
   Nat.choose excellent_students 1 * Nat.choose (total_people - excellent_students) (selected_people - 1)) /
  Nat.choose total_people selected_people :=
by sorry

end NUMINAMATH_CALUDE_probability_zero_or_one_excellent_equals_formula_l3194_319492


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l3194_319446

theorem quadratic_root_implies_k (k : ℝ) : 
  (3 * ((-15 - Real.sqrt 229) / 4)^2 + 15 * ((-15 - Real.sqrt 229) / 4) + k = 0) → 
  k = -1/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l3194_319446


namespace NUMINAMATH_CALUDE_summit_academy_contestants_l3194_319481

theorem summit_academy_contestants (s j : ℕ) (h : s / 3 = j * 3 / 4) : s = 4 * j := by
  sorry

end NUMINAMATH_CALUDE_summit_academy_contestants_l3194_319481


namespace NUMINAMATH_CALUDE_original_price_correct_l3194_319489

/-- The original price of a bag of mini peanut butter cups before discount -/
def original_price : ℝ := 6

/-- The discount percentage applied to the bags -/
def discount_percentage : ℝ := 0.75

/-- The number of bags purchased -/
def num_bags : ℕ := 2

/-- The total amount spent on the bags after discount -/
def total_spent : ℝ := 3

/-- Theorem stating that the original price is correct given the conditions -/
theorem original_price_correct : 
  (1 - discount_percentage) * (num_bags * original_price) = total_spent := by
  sorry

end NUMINAMATH_CALUDE_original_price_correct_l3194_319489


namespace NUMINAMATH_CALUDE_squirrel_acorns_count_l3194_319487

/-- Represents the number of acorns hidden per hole by each animal -/
structure AcornsPerHole where
  chipmunk : ℕ
  squirrel : ℕ
  rabbit : ℕ

/-- Represents the number of holes dug by each animal -/
structure HolesCounts where
  chipmunk : ℕ
  squirrel : ℕ
  rabbit : ℕ

/-- The main theorem stating the number of acorns hidden by the squirrel -/
theorem squirrel_acorns_count 
  (aph : AcornsPerHole) 
  (hc : HolesCounts) 
  (h1 : aph.chipmunk = 4) 
  (h2 : aph.squirrel = 5) 
  (h3 : aph.rabbit = 2) 
  (h4 : aph.chipmunk * hc.chipmunk = aph.squirrel * hc.squirrel) 
  (h5 : hc.squirrel = hc.chipmunk - 5) 
  (h6 : aph.rabbit * hc.rabbit = aph.squirrel * hc.squirrel) 
  (h7 : hc.rabbit = hc.squirrel + 10) : 
  aph.squirrel * hc.squirrel = 100 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorns_count_l3194_319487


namespace NUMINAMATH_CALUDE_expected_string_length_l3194_319409

/-- Represents the states of Clayton's progress -/
inductive State
  | S0  -- No letters written
  | S1  -- M written
  | S2  -- M and A written
  | S3  -- M, A, and T written
  | S4  -- M, A, T, and H written (final state)

/-- The hexagon with vertices M, M, A, T, H, S -/
def Hexagon : Type := Unit

/-- Clayton's starting position (M adjacent to M and A) -/
def start_position : Hexagon := Unit.unit

/-- Probability of moving to an adjacent vertex -/
def move_probability : ℚ := 1/2

/-- Expected number of steps to reach the final state from a given state -/
noncomputable def expected_steps : State → ℚ
  | State.S0 => 5
  | State.S1 => 4
  | State.S2 => 3
  | State.S3 => 2
  | State.S4 => 0

/-- The main theorem: Expected length of Clayton's string is 6 -/
theorem expected_string_length :
  expected_steps State.S0 + 1 = 6 := by sorry

end NUMINAMATH_CALUDE_expected_string_length_l3194_319409


namespace NUMINAMATH_CALUDE_tangent_to_ln_curve_l3194_319449

theorem tangent_to_ln_curve (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = (Real.log x) / x) →
  (k * 0 = Real.log 0) →
  k = 1 / Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_to_ln_curve_l3194_319449


namespace NUMINAMATH_CALUDE_min_value_of_y_l3194_319433

theorem min_value_of_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → 1/x + 4/y ≥ 9/2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ 1/x + 4/y = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_y_l3194_319433


namespace NUMINAMATH_CALUDE_joel_contributed_22_toys_l3194_319485

/-- The number of toys Joel contributed to the donation -/
def joels_toys (toys_from_friends : ℕ) (total_toys : ℕ) : ℕ :=
  2 * ((total_toys - toys_from_friends) / 3)

/-- Theorem stating that Joel contributed 22 toys -/
theorem joel_contributed_22_toys : 
  joels_toys 75 108 = 22 := by
  sorry

end NUMINAMATH_CALUDE_joel_contributed_22_toys_l3194_319485


namespace NUMINAMATH_CALUDE_exercise_weights_after_training_l3194_319495

def calculate_final_weight (initial_weight : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl (λ acc change => acc * (1 + change)) initial_weight

def bench_press_changes : List ℝ := [-0.8, 0.6, -0.2, 2.0]
def squat_changes : List ℝ := [-0.5, 0.4, 1.0]
def deadlift_changes : List ℝ := [-0.3, 0.8, -0.4, 0.5]

theorem exercise_weights_after_training (initial_bench : ℝ) (initial_squat : ℝ) (initial_deadlift : ℝ) 
    (h1 : initial_bench = 500) 
    (h2 : initial_squat = 400) 
    (h3 : initial_deadlift = 600) :
  (calculate_final_weight initial_bench bench_press_changes = 384) ∧
  (calculate_final_weight initial_squat squat_changes = 560) ∧
  (calculate_final_weight initial_deadlift deadlift_changes = 680.4) := by
  sorry

#eval calculate_final_weight 500 bench_press_changes
#eval calculate_final_weight 400 squat_changes
#eval calculate_final_weight 600 deadlift_changes

end NUMINAMATH_CALUDE_exercise_weights_after_training_l3194_319495


namespace NUMINAMATH_CALUDE_unique_solution_implies_n_equals_8_l3194_319426

-- Define the quadratic equation
def quadratic_equation (n : ℝ) (x : ℝ) : ℝ := 4 * x^2 + n * x + 4

-- Define the discriminant of the quadratic equation
def discriminant (n : ℝ) : ℝ := n^2 - 4 * 4 * 4

-- Theorem statement
theorem unique_solution_implies_n_equals_8 :
  ∃! x : ℝ, quadratic_equation 8 x = 0 ∧
  ∀ n : ℝ, (∃! x : ℝ, quadratic_equation n x = 0) → n = 8 ∨ n = -8 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_n_equals_8_l3194_319426


namespace NUMINAMATH_CALUDE_A_3_1_equals_13_l3194_319422

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_1_equals_13 : A 3 1 = 13 := by
  sorry

end NUMINAMATH_CALUDE_A_3_1_equals_13_l3194_319422


namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l3194_319457

/-- Given a square divided into four congruent rectangles, 
    if the perimeter of each rectangle is 30 inches, 
    then the perimeter of the square is 48 inches. -/
theorem square_perimeter_from_rectangle_perimeter :
  ∀ s : ℝ,
  s > 0 →
  (2 * s + 2 * (s / 4) = 30) →
  (4 * s = 48) :=
by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l3194_319457


namespace NUMINAMATH_CALUDE_chopped_cube_height_l3194_319453

/-- Given a unit cube with a corner chopped off through the midpoints of the three adjacent edges,
    when the freshly-cut face is placed on a table, the height of the remaining solid is 29/32. -/
theorem chopped_cube_height : 
  let cube_edge : ℝ := 1
  let midpoint_factor : ℝ := 1/2
  let chopped_volume : ℝ := 3/32
  let remaining_volume : ℝ := 1 - chopped_volume
  let base_area : ℝ := cube_edge^2
  remaining_volume / base_area = 29/32 := by sorry

end NUMINAMATH_CALUDE_chopped_cube_height_l3194_319453


namespace NUMINAMATH_CALUDE_inequality_bound_l3194_319421

theorem inequality_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (a / x + b / y) > M) →
  M < a + b + 2 * Real.sqrt (a * b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_bound_l3194_319421


namespace NUMINAMATH_CALUDE_gate_buyers_pay_more_l3194_319497

/-- Calculates the difference in total amount paid between gate buyers and pre-buyers --/
def ticketPriceDifference (preBuyerCount : ℕ) (gateBuyerCount : ℕ) (preBuyerPrice : ℕ) (gateBuyerPrice : ℕ) : ℕ :=
  gateBuyerCount * gateBuyerPrice - preBuyerCount * preBuyerPrice

theorem gate_buyers_pay_more :
  ticketPriceDifference 20 30 155 200 = 2900 := by
  sorry

end NUMINAMATH_CALUDE_gate_buyers_pay_more_l3194_319497


namespace NUMINAMATH_CALUDE_valid_squares_count_l3194_319474

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  topLeft : Nat × Nat

/-- Checks if a square contains at least 6 black squares -/
def hasAtLeastSixBlackSquares (s : Square) : Bool :=
  sorry

/-- Counts the number of valid squares on the board -/
def countValidSquares (boardSize : Nat) : Nat :=
  sorry

theorem valid_squares_count :
  countValidSquares 10 = 140 :=
sorry

end NUMINAMATH_CALUDE_valid_squares_count_l3194_319474


namespace NUMINAMATH_CALUDE_mother_age_twice_alex_l3194_319432

/-- Alex's birth year -/
def alexBirthYear : ℕ := 2000

/-- The year when Alex's mother's age was five times his age -/
def referenceYear : ℕ := 2010

/-- Alex's mother's age is five times Alex's age in the reference year -/
axiom mother_age_five_times (y : ℕ) : y - alexBirthYear = 10 → y = referenceYear → 
  ∃ (motherAge : ℕ), motherAge = 5 * (y - alexBirthYear)

/-- The year when Alex's mother's age will be twice his age -/
def targetYear : ℕ := 2040

theorem mother_age_twice_alex :
  ∃ (motherAge alexAge : ℕ),
    motherAge = 2 * alexAge ∧
    alexAge = targetYear - alexBirthYear ∧
    motherAge = (referenceYear - alexBirthYear) * 5 + (targetYear - referenceYear) :=
sorry

end NUMINAMATH_CALUDE_mother_age_twice_alex_l3194_319432


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3194_319483

theorem interest_rate_calculation (simple_interest principal time_period : ℚ) 
  (h1 : simple_interest = 4016.25)
  (h2 : time_period = 5)
  (h3 : principal = 80325) :
  simple_interest * 100 / (principal * time_period) = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3194_319483


namespace NUMINAMATH_CALUDE_ab_less_than_a_plus_b_l3194_319407

theorem ab_less_than_a_plus_b (a b : ℝ) (ha : a < 1) (hb : b > 1) : a * b < a + b := by
  sorry

end NUMINAMATH_CALUDE_ab_less_than_a_plus_b_l3194_319407


namespace NUMINAMATH_CALUDE_unique_solution_l3194_319494

theorem unique_solution (a b c : ℝ) : 
  a > 2 ∧ b > 2 ∧ c > 2 →
  (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 48 →
  a = 7 ∧ b = 5 ∧ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l3194_319494


namespace NUMINAMATH_CALUDE_male_alligators_mating_season_l3194_319412

/-- Represents the alligator population on Lagoon Island -/
structure AlligatorPopulation where
  males : ℕ
  adultFemales : ℕ
  juvenileFemales : ℕ

/-- Calculates the total number of alligators -/
def totalAlligators (pop : AlligatorPopulation) : ℕ :=
  pop.males + pop.adultFemales + pop.juvenileFemales

/-- Represents the population ratio of males:adult females:juvenile females -/
structure PopulationRatio where
  maleRatio : ℕ
  adultFemaleRatio : ℕ
  juvenileFemaleRatio : ℕ

/-- Theorem: Given the conditions, the number of male alligators during mating season is 10 -/
theorem male_alligators_mating_season
  (ratio : PopulationRatio)
  (nonMatingAdultFemales : ℕ)
  (resourceLimit : ℕ)
  (turtleRatio : ℕ)
  (h1 : ratio.maleRatio = 2 ∧ ratio.adultFemaleRatio = 3 ∧ ratio.juvenileFemaleRatio = 5)
  (h2 : nonMatingAdultFemales = 15)
  (h3 : resourceLimit = 200)
  (h4 : turtleRatio = 3)
  : ∃ (pop : AlligatorPopulation),
    pop.males = 10 ∧
    pop.adultFemales = 2 * nonMatingAdultFemales ∧
    totalAlligators pop ≤ resourceLimit ∧
    turtleRatio * (totalAlligators pop) ≤ 3 * resourceLimit :=
by sorry


end NUMINAMATH_CALUDE_male_alligators_mating_season_l3194_319412


namespace NUMINAMATH_CALUDE_inequality_implication_l3194_319490

theorem inequality_implication (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3194_319490


namespace NUMINAMATH_CALUDE_pizza_theorem_l3194_319462

def pizza_problem (total_slices : ℕ) (slices_per_person : ℕ) : ℕ :=
  (total_slices / slices_per_person) - 1

theorem pizza_theorem :
  pizza_problem 12 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l3194_319462


namespace NUMINAMATH_CALUDE_initial_bees_in_hive_l3194_319430

theorem initial_bees_in_hive (additional_bees : ℕ) (total_bees : ℕ) (h1 : additional_bees = 9) (h2 : total_bees = 25) :
  total_bees - additional_bees = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_bees_in_hive_l3194_319430


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3194_319448

theorem simplify_polynomial (r : ℝ) : (2 * r^2 + 5 * r - 3) - (r^2 + 4 * r - 6) = r^2 + r + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3194_319448


namespace NUMINAMATH_CALUDE_inequality_holds_iff_even_l3194_319410

theorem inequality_holds_iff_even (n : ℕ+) :
  (∀ x : ℝ, 3 * x^(n : ℕ) + n * (x + 2) - 3 ≥ n * x^2) ↔ Even n := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_even_l3194_319410


namespace NUMINAMATH_CALUDE_non_monotonic_interval_l3194_319400

-- Define the function
def f (x : ℝ) : ℝ := |2*x - 1|

-- Define the property of being non-monotonic in an interval
def is_non_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧ 
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- State the theorem
theorem non_monotonic_interval (k : ℝ) :
  is_non_monotonic f (k - 1) (k + 1) ↔ -1 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_non_monotonic_interval_l3194_319400


namespace NUMINAMATH_CALUDE_decreasing_f_iff_a_in_range_l3194_319486

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem decreasing_f_iff_a_in_range (a : ℝ) :
  is_decreasing (f a) ↔ 0 < a ∧ a ≤ 1/4 :=
sorry

end NUMINAMATH_CALUDE_decreasing_f_iff_a_in_range_l3194_319486


namespace NUMINAMATH_CALUDE_three_distinct_roots_reciprocal_l3194_319493

theorem three_distinct_roots_reciprocal (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    a * x^5 + b * x^4 + c = 0 ∧
    a * y^5 + b * y^4 + c = 0 ∧
    a * z^5 + b * z^4 + c = 0) →
  (∃ u v w : ℝ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    c * u^5 + b * u + a = 0 ∧
    c * v^5 + b * v + a = 0 ∧
    c * w^5 + b * w + a = 0) :=
by sorry

end NUMINAMATH_CALUDE_three_distinct_roots_reciprocal_l3194_319493


namespace NUMINAMATH_CALUDE_correct_arrangements_l3194_319443

/-- The number of people in the row -/
def n : ℕ := 8

/-- The number of special people (A, B, C, D, E) -/
def k : ℕ := 5

/-- Function to calculate the number of arrangements -/
def count_arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating the correct number of arrangements -/
theorem correct_arrangements :
  count_arrangements n k = 11520 := by sorry

end NUMINAMATH_CALUDE_correct_arrangements_l3194_319443


namespace NUMINAMATH_CALUDE_customers_after_family_l3194_319440

/-- Represents the taco truck's sales during lunch rush -/
def taco_truck_sales (soft_taco_price hard_taco_price : ℕ) 
  (family_hard_tacos family_soft_tacos : ℕ)
  (other_customers : ℕ) (total_revenue : ℕ) : Prop :=
  let family_revenue := family_hard_tacos * hard_taco_price + family_soft_tacos * soft_taco_price
  let other_revenue := other_customers * 2 * soft_taco_price
  family_revenue + other_revenue = total_revenue

/-- Theorem stating the number of customers after the family -/
theorem customers_after_family : 
  taco_truck_sales 2 5 4 3 10 66 := by sorry

end NUMINAMATH_CALUDE_customers_after_family_l3194_319440


namespace NUMINAMATH_CALUDE_picnic_men_count_l3194_319447

/-- Represents the number of people at a picnic -/
structure PicnicAttendance where
  total : Nat
  men : Nat
  women : Nat
  adults : Nat
  children : Nat

/-- Defines the conditions for a valid picnic attendance -/
def ValidPicnicAttendance (p : PicnicAttendance) : Prop :=
  p.total = 200 ∧
  p.men = p.women + 20 ∧
  p.adults = p.children + 20 ∧
  p.total = p.men + p.women + p.children ∧
  p.adults = p.men + p.women

theorem picnic_men_count (p : PicnicAttendance) (h : ValidPicnicAttendance p) : p.men = 65 := by
  sorry

end NUMINAMATH_CALUDE_picnic_men_count_l3194_319447


namespace NUMINAMATH_CALUDE_min_a_value_l3194_319415

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x * (x^3 - 3*x + 3) - a * exp x - x

theorem min_a_value :
  ∀ a : ℝ, (∃ x : ℝ, x ≥ -2 ∧ f a x ≤ 0) → a ≥ 1 - 1/exp 1 :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l3194_319415


namespace NUMINAMATH_CALUDE_game_night_sandwiches_l3194_319402

theorem game_night_sandwiches (num_friends : ℕ) (sandwiches_per_friend : ℕ) 
  (h1 : num_friends = 7) (h2 : sandwiches_per_friend = 5) : 
  num_friends * sandwiches_per_friend = 35 := by
  sorry

end NUMINAMATH_CALUDE_game_night_sandwiches_l3194_319402


namespace NUMINAMATH_CALUDE_arccos_negative_half_l3194_319439

theorem arccos_negative_half : Real.arccos (-1/2) = 2*π/3 := by sorry

end NUMINAMATH_CALUDE_arccos_negative_half_l3194_319439


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_l3194_319420

theorem floor_plus_self_unique (r : ℝ) : ⌊r⌋ + r = 15.75 ↔ r = 7.25 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_l3194_319420


namespace NUMINAMATH_CALUDE_sibling_ages_l3194_319456

theorem sibling_ages (sister_age brother_age : ℕ) : 
  (brother_age - 2 = 2 * (sister_age - 2)) →
  (brother_age - 8 = 5 * (sister_age - 8)) →
  (sister_age = 10 ∧ brother_age = 18) :=
by sorry

end NUMINAMATH_CALUDE_sibling_ages_l3194_319456


namespace NUMINAMATH_CALUDE_chord_intersection_lengths_l3194_319401

-- Define the circle and its properties
def circle_radius : ℝ := 6

-- Define the chord EJ and its properties
def chord_length : ℝ := 10

-- Define the point M where EJ intersects GH
def point_M (x : ℝ) : Prop := 0 < x ∧ x < 2 * circle_radius

-- Define the lengths of GM and MH
def length_GM (x : ℝ) : ℝ := x
def length_MH (x : ℝ) : ℝ := 2 * circle_radius - x

-- Theorem statement
theorem chord_intersection_lengths :
  ∃ x : ℝ, point_M x ∧ 
    length_GM x = 6 + Real.sqrt 11 ∧
    length_MH x = 6 - Real.sqrt 11 :=
sorry

end NUMINAMATH_CALUDE_chord_intersection_lengths_l3194_319401


namespace NUMINAMATH_CALUDE_solution_difference_l3194_319438

-- Define the equation
def equation (x : ℝ) : Prop :=
  (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3

-- Define the theorem
theorem solution_difference (r s : ℝ) : 
  equation r ∧ equation s ∧ r ≠ s ∧ r > s → r - s = 3 := by
  sorry


end NUMINAMATH_CALUDE_solution_difference_l3194_319438


namespace NUMINAMATH_CALUDE_rational_expression_equality_algebraic_expression_equality_l3194_319413

/-- Prove the equality of the given rational expression -/
theorem rational_expression_equality (m : ℝ) (hm1 : m ≠ -4) (hm2 : m ≠ -2) : 
  (m^2 - 16) / (m^2 + 8*m + 16) / ((m - 4) / (2*m + 8)) * ((m - 2) / (m + 2)) = 2*(m - 2) / (m + 2) := by
  sorry

/-- Prove the equality of the given algebraic expression -/
theorem algebraic_expression_equality (x : ℝ) (hx1 : x ≠ -2) (hx2 : x ≠ 2) : 
  3 / (x + 2) + 1 / (2 - x) - (2*x) / (4 - x^2) = 4 / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_rational_expression_equality_algebraic_expression_equality_l3194_319413


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_l3194_319403

theorem right_triangle_leg_sum (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  b = a + 2 →        -- legs differ by 2
  c = 53 →           -- hypotenuse is 53
  a + b = 104 :=     -- sum of legs is 104
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_l3194_319403


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3194_319436

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 4*x - 22 = 0 ↔ (x - 2)^2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3194_319436


namespace NUMINAMATH_CALUDE_max_dot_product_on_circle_l3194_319496

theorem max_dot_product_on_circle :
  ∀ (P : ℝ × ℝ),
  P.1^2 + P.2^2 = 1 →
  let A : ℝ × ℝ := (-2, 0)
  let O : ℝ × ℝ := (0, 0)
  let AO : ℝ × ℝ := (O.1 - A.1, O.2 - A.2)
  let AP : ℝ × ℝ := (P.1 - A.1, P.2 - A.2)
  (AO.1 * AP.1 + AO.2 * AP.2) ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_dot_product_on_circle_l3194_319496


namespace NUMINAMATH_CALUDE_equation_solutions_l3194_319488

theorem equation_solutions (a b : ℝ) (h : a + b = 0) :
  (∃! x : ℝ, a * x + b = 0) ∨ (∀ x : ℝ, a * x + b = 0) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3194_319488


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3194_319424

theorem negative_fraction_comparison : -3/2 < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3194_319424


namespace NUMINAMATH_CALUDE_one_third_1206_percent_of_400_l3194_319480

theorem one_third_1206_percent_of_400 : 
  (1206 / 3) / 400 * 100 = 100.5 := by sorry

end NUMINAMATH_CALUDE_one_third_1206_percent_of_400_l3194_319480


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3194_319461

theorem sum_of_coefficients (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (1 - 2*x)^5 = a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5 →
  a₀ + a₁ + a₂ + a₃ + a₄ = 33 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3194_319461


namespace NUMINAMATH_CALUDE_cubic_expression_value_l3194_319499

theorem cubic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : 
  m^3 + 2*m^2 - 2001 = -2000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l3194_319499


namespace NUMINAMATH_CALUDE_exact_defective_selection_l3194_319482

def total_products : ℕ := 100
def defective_products : ℕ := 3
def products_to_select : ℕ := 4
def defective_to_select : ℕ := 2

theorem exact_defective_selection :
  (Nat.choose defective_products defective_to_select) *
  (Nat.choose (total_products - defective_products) (products_to_select - defective_to_select)) = 13968 := by
  sorry

end NUMINAMATH_CALUDE_exact_defective_selection_l3194_319482


namespace NUMINAMATH_CALUDE_system_solution_l3194_319441

theorem system_solution : ∃ (x y : ℚ), 
  (4 * x + 3 * y = 1) ∧ 
  (6 * x - 9 * y = -8) ∧ 
  (x = -5/18) ∧ 
  (y = 19/27) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3194_319441


namespace NUMINAMATH_CALUDE_article_cost_l3194_319411

/-- The cost of an article given selling conditions. -/
theorem article_cost (sell_price_1 sell_price_2 : ℚ) (gain_increase : ℚ) : 
  sell_price_1 = 700 →
  sell_price_2 = 750 →
  gain_increase = 1/10 →
  ∃ (cost gain : ℚ), 
    cost + gain = sell_price_1 ∧
    cost + gain * (1 + gain_increase) = sell_price_2 ∧
    cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_article_cost_l3194_319411


namespace NUMINAMATH_CALUDE_jokes_theorem_l3194_319408

def calculate_jokes (initial : ℕ) : ℕ :=
  initial + 2 * initial + 4 * initial + 8 * initial + 16 * initial

def total_jokes : ℕ :=
  calculate_jokes 11 + calculate_jokes 7 + calculate_jokes 5 + calculate_jokes 3

theorem jokes_theorem : total_jokes = 806 := by
  sorry

end NUMINAMATH_CALUDE_jokes_theorem_l3194_319408


namespace NUMINAMATH_CALUDE_sequence_non_positive_l3194_319435

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h_ineq : ∀ k, k < n → a k.pred - 2 * a k + a k.succ ≥ 0) :
  ∀ i, i ≤ n → a i ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l3194_319435


namespace NUMINAMATH_CALUDE_joan_can_buy_5_apples_l3194_319467

/-- Represents the grocery shopping problem --/
def grocery_problem (total_money : ℕ) (hummus_price : ℕ) (hummus_quantity : ℕ)
  (chicken_price : ℕ) (bacon_price : ℕ) (vegetable_price : ℕ) (apple_price : ℕ) : Prop :=
  let remaining_money := total_money - (hummus_price * hummus_quantity + chicken_price + bacon_price + vegetable_price)
  remaining_money / apple_price = 5

/-- Theorem stating that Joan can buy 5 apples with her remaining money --/
theorem joan_can_buy_5_apples :
  grocery_problem 60 5 2 20 10 10 2 := by
  sorry

#check joan_can_buy_5_apples

end NUMINAMATH_CALUDE_joan_can_buy_5_apples_l3194_319467


namespace NUMINAMATH_CALUDE_two_green_marbles_probability_l3194_319460

/-- The probability of drawing two green marbles consecutively without replacement -/
theorem two_green_marbles_probability 
  (red green white blue : ℕ) 
  (h_red : red = 3)
  (h_green : green = 4)
  (h_white : white = 8)
  (h_blue : blue = 5) : 
  (green : ℚ) / (red + green + white + blue) * 
  ((green - 1) : ℚ) / (red + green + white + blue - 1) = 3 / 95 := by
sorry

end NUMINAMATH_CALUDE_two_green_marbles_probability_l3194_319460


namespace NUMINAMATH_CALUDE_remainder_4015_div_32_l3194_319468

theorem remainder_4015_div_32 : 4015 % 32 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4015_div_32_l3194_319468


namespace NUMINAMATH_CALUDE_base6_to_base10_conversion_l3194_319473

-- Define the base 6 number as a list of digits
def base6_number : List Nat := [5, 4, 3, 2, 1]

-- Define the base of the number system
def base : Nat := 6

-- Function to convert a list of digits in base 6 to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

-- Theorem statement
theorem base6_to_base10_conversion :
  to_base_10 base6_number base = 7465 := by
  sorry

end NUMINAMATH_CALUDE_base6_to_base10_conversion_l3194_319473


namespace NUMINAMATH_CALUDE_jimmy_lodging_cost_l3194_319466

/-- Calculates the total lodging cost for Jimmy's vacation --/
def total_lodging_cost (hostel_nights : ℕ) (hostel_cost_per_night : ℕ) 
  (cabin_nights : ℕ) (cabin_total_cost_per_night : ℕ) (cabin_friends : ℕ) : ℕ :=
  hostel_nights * hostel_cost_per_night + 
  cabin_nights * cabin_total_cost_per_night / (cabin_friends + 1)

/-- Theorem stating that Jimmy's total lodging cost is $75 --/
theorem jimmy_lodging_cost : 
  total_lodging_cost 3 15 2 45 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_lodging_cost_l3194_319466


namespace NUMINAMATH_CALUDE_tangent_lines_not_always_same_l3194_319484

-- Define a curve as a function from ℝ to ℝ
def Curve := ℝ → ℝ

-- Define a point in ℝ²
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in ℝ² as a pair of slope and y-intercept
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a tangent line to a curve at a point
def tangentLineToCurve (f : Curve) (p : Point) : Line := sorry

-- Define a tangent line passing through a point
def tangentLineThroughPoint (p : Point) : Line := sorry

-- The theorem to prove
theorem tangent_lines_not_always_same (f : Curve) (p : Point) : 
  ¬ ∀ (f : Curve) (p : Point), tangentLineToCurve f p = tangentLineThroughPoint p :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_not_always_same_l3194_319484


namespace NUMINAMATH_CALUDE_range_of_a_l3194_319465

-- Define the universal set U
def U : Set ℝ := {x : ℝ | 0 < x ∧ x < 9}

-- Define set A parameterized by a
def A (a : ℝ) : Set ℝ := {x : ℝ | 1 < x ∧ x < a}

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (∃ x, x ∈ A a) ∧ ¬(A a ⊆ U) ↔ 1 < a ∧ a ≤ 9 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3194_319465


namespace NUMINAMATH_CALUDE_student_number_calculation_l3194_319459

theorem student_number_calculation (x : ℕ) (h : x = 129) : 2 * x - 148 = 110 := by
  sorry

end NUMINAMATH_CALUDE_student_number_calculation_l3194_319459


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3194_319451

theorem fraction_evaluation : 
  (1 - (1/4 + 1/5)) / (1 - 2/3) = 33/20 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3194_319451


namespace NUMINAMATH_CALUDE_birds_in_tree_l3194_319419

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29) 
  (h2 : final_birds = 42) : 
  final_birds - initial_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l3194_319419
