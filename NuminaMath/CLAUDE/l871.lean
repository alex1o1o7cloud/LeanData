import Mathlib

namespace NUMINAMATH_CALUDE_eggs_per_tray_l871_87163

theorem eggs_per_tray (total_trays : ℕ) (total_eggs : ℕ) (eggs_per_tray : ℕ) : 
  total_trays = 7 →
  total_eggs = 70 →
  total_eggs = total_trays * eggs_per_tray →
  eggs_per_tray = 10 := by
sorry

end NUMINAMATH_CALUDE_eggs_per_tray_l871_87163


namespace NUMINAMATH_CALUDE_fraction_multiplication_result_l871_87169

theorem fraction_multiplication_result : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5040 = 1512 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_result_l871_87169


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l871_87105

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

theorem tenth_term_of_sequence (a : ℚ) (r : ℚ) (h : a = 4 ∧ r = 1) :
  geometric_sequence a r 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l871_87105


namespace NUMINAMATH_CALUDE_expression_simplification_l871_87166

theorem expression_simplification (a b : ℚ) (ha : a = -1) (hb : b = 1/2) :
  2 * a^2 * b - (3 * a * b^2 - (4 * a * b^2 - 2 * a^2 * b)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l871_87166


namespace NUMINAMATH_CALUDE_proportion_solve_l871_87142

theorem proportion_solve (x : ℚ) : (3 : ℚ) / 12 = x / 16 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solve_l871_87142


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l871_87180

theorem distance_between_complex_points :
  let z₁ : ℂ := 2 + 3*I
  let z₂ : ℂ := -2 + 2*I
  Complex.abs (z₁ - z₂) = Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l871_87180


namespace NUMINAMATH_CALUDE_two_color_theorem_l871_87148

/-- A type representing a region in the plane --/
structure Region

/-- A type representing a color --/
inductive Color
| Red
| Blue

/-- A type representing a line or circle --/
inductive Divider
| Line
| Circle

/-- Predicate to check if two regions are adjacent --/
def adjacent (r1 r2 : Region) : Prop := sorry

/-- Function to represent a coloring of regions --/
def coloring (R : Set Region) : Region → Color := sorry

/-- The main theorem --/
theorem two_color_theorem (S : Set Divider) :
  ∃ (R : Set Region) (c : Region → Color),
    (∀ r1 r2 : Region, r1 ∈ R → r2 ∈ R → adjacent r1 r2 → c r1 ≠ c r2) :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l871_87148


namespace NUMINAMATH_CALUDE_line_parameterization_l871_87186

/-- Given a line y = 3x - 11 parameterized by (x, y) = (r, 1) + t(4, k),
    prove that r = 4 and k = 12 -/
theorem line_parameterization (r k : ℝ) : 
  (∀ t : ℝ, (r + 4*t, 1 + k*t) ∈ {p : ℝ × ℝ | p.2 = 3*p.1 - 11}) ↔ 
  (r = 4 ∧ k = 12) :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_l871_87186


namespace NUMINAMATH_CALUDE_cube_volume_problem_l871_87128

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  (a + 2) * a * (a - 2) = a^3 - 24 → 
  a^3 = 216 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l871_87128


namespace NUMINAMATH_CALUDE_aluminum_weight_l871_87113

-- Define the weights of the metal pieces
def iron_weight : ℝ := 11.17
def weight_difference : ℝ := 10.33

-- Theorem to prove
theorem aluminum_weight :
  iron_weight - weight_difference = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_aluminum_weight_l871_87113


namespace NUMINAMATH_CALUDE_not_divisible_five_power_l871_87131

theorem not_divisible_five_power (n k : ℕ) : ¬ ((5^k - 1) ∣ (5^n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_five_power_l871_87131


namespace NUMINAMATH_CALUDE_max_sum_of_digits_l871_87191

theorem max_sum_of_digits (x z : ℕ) : 
  x ≤ 9 → z ≤ 9 → x > z → 99 * (x - z) = 693 → 
  ∃ d : ℕ, d = 11 ∧ ∀ x' z' : ℕ, x' ≤ 9 → z' ≤ 9 → x' > z' → 99 * (x' - z') = 693 → x' + z' ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_l871_87191


namespace NUMINAMATH_CALUDE_probability_red_black_heart_value_l871_87164

/-- The probability of drawing a red card first, then a black card, and then a red heart
    from a deck of 104 cards with 52 red cards (of which 26 are hearts) and 52 black cards. -/
def probability_red_black_heart (total_cards : ℕ) (red_cards : ℕ) (black_cards : ℕ) (heart_cards : ℕ) : ℚ :=
  (red_cards : ℚ) / total_cards *
  (black_cards : ℚ) / (total_cards - 1) *
  (heart_cards - 1 : ℚ) / (total_cards - 2)

/-- The probability of drawing a red card first, then a black card, and then a red heart
    from a deck of 104 cards with 52 red cards (of which 26 are hearts) and 52 black cards
    is equal to 25/3978. -/
theorem probability_red_black_heart_value :
  probability_red_black_heart 104 52 52 26 = 25 / 3978 :=
by
  sorry

#eval probability_red_black_heart 104 52 52 26

end NUMINAMATH_CALUDE_probability_red_black_heart_value_l871_87164


namespace NUMINAMATH_CALUDE_expression_evaluation_l871_87159

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  (x + 2*y)^2 - (x + y)*(x - y) = -11/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l871_87159


namespace NUMINAMATH_CALUDE_round_robin_tournament_teams_l871_87181

/-- Represents the total points in a round-robin tournament -/
def totalPoints (n : ℕ) : ℕ := n * (n - 1)

/-- The set of reported total points and their averages -/
def reportedPoints : Finset ℕ := {3086, 2018, 1238, 2162, 2552, 1628, 2114}

/-- Theorem stating that if one of the reported points is correct, then there are 47 teams -/
theorem round_robin_tournament_teams :
  ∃ (p : ℕ), p ∈ reportedPoints ∧ totalPoints 47 = p :=
sorry

end NUMINAMATH_CALUDE_round_robin_tournament_teams_l871_87181


namespace NUMINAMATH_CALUDE_unique_solution_to_system_l871_87172

/-- The number of integer solutions to the system of equations:
    x^2 - 4xy + 3y^2 + z^2 = 45
    x^2 + 5yz - z^2 = -52
    -2x^2 + xy - 7z^2 = -101 -/
theorem unique_solution_to_system : 
  ∃! (x y z : ℤ), 
    x^2 - 4*x*y + 3*y^2 + z^2 = 45 ∧ 
    x^2 + 5*y*z - z^2 = -52 ∧ 
    -2*x^2 + x*y - 7*z^2 = -101 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_system_l871_87172


namespace NUMINAMATH_CALUDE_polynomial_division_l871_87153

-- Define the polynomial
def p (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the divisor polynomial
def q (x : ℝ) : ℝ := x^2 + 3*x - 4

-- State the theorem
theorem polynomial_division (a b c : ℝ) 
  (h : ∀ x, p a b c x = 0 → q x = 0) : 
  (4*a + c = 12) ∧ (2*a - 2*b - c = 14) := by
  sorry


end NUMINAMATH_CALUDE_polynomial_division_l871_87153


namespace NUMINAMATH_CALUDE_sum_of_three_polynomials_no_roots_l871_87104

/-- Given three quadratic polynomials, if the sum of any two has no roots, 
    then the sum of all three has no roots. -/
theorem sum_of_three_polynomials_no_roots 
  (a b c d e f : ℝ) 
  (h1 : ∀ x, (2*x^2 + (a + c)*x + (b + d)) ≠ 0)
  (h2 : ∀ x, (2*x^2 + (c + e)*x + (d + f)) ≠ 0)
  (h3 : ∀ x, (2*x^2 + (e + a)*x + (f + b)) ≠ 0) :
  ∀ x, (3*x^2 + (a + c + e)*x + (b + d + f)) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_polynomials_no_roots_l871_87104


namespace NUMINAMATH_CALUDE_ellen_calorie_instruction_l871_87135

/-- The total number of calories Ellen was instructed to eat in a day -/
def total_calories : ℕ := 2200

/-- The number of calories Ellen ate for breakfast -/
def breakfast_calories : ℕ := 353

/-- The number of calories Ellen had for lunch -/
def lunch_calories : ℕ := 885

/-- The number of calories Ellen had for afternoon snack -/
def snack_calories : ℕ := 130

/-- The number of calories Ellen has left for dinner -/
def dinner_calories : ℕ := 832

/-- Theorem stating that the total calories Ellen was instructed to eat
    is equal to the sum of all meals and snacks -/
theorem ellen_calorie_instruction :
  total_calories = breakfast_calories + lunch_calories + snack_calories + dinner_calories :=
by sorry

end NUMINAMATH_CALUDE_ellen_calorie_instruction_l871_87135


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l871_87167

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  Complex.im ((1 - i) / ((1 + i)^2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l871_87167


namespace NUMINAMATH_CALUDE_bead_problem_solutions_l871_87160

/-- Represents the possible total number of beads -/
def PossibleTotals : Set ℕ := {107, 109, 111, 113, 115, 117}

/-- Represents a solution to the bead problem -/
structure BeadSolution where
  x : ℕ -- number of 19-gram beads
  y : ℕ -- number of 17-gram beads

/-- Checks if a BeadSolution is valid -/
def isValidSolution (s : BeadSolution) : Prop :=
  19 * s.x + 17 * s.y = 2017 ∧ s.x + s.y ∈ PossibleTotals

/-- Theorem stating that there exist valid solutions for all possible totals -/
theorem bead_problem_solutions :
  ∀ n ∈ PossibleTotals, ∃ s : BeadSolution, isValidSolution s ∧ s.x + s.y = n :=
sorry

end NUMINAMATH_CALUDE_bead_problem_solutions_l871_87160


namespace NUMINAMATH_CALUDE_problem_solution_l871_87111

theorem problem_solution (θ : Real) (x : Real) :
  let A := (5 * Real.sin θ + 4 * Real.cos θ) / (3 * Real.sin θ + Real.cos θ)
  let B := x^3 + 1/x^3
  Real.tan θ = 2 →
  x + 1/x = 2 * A →
  A = 2 ∧ B = 52 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l871_87111


namespace NUMINAMATH_CALUDE_parabola_symmetry_l871_87197

/-- Represents a parabola in 2D space -/
structure Parabola where
  equation : ℝ → ℝ

/-- Two parabolas are symmetric about the origin -/
def symmetric_about_origin (p1 p2 : Parabola) : Prop :=
  ∀ x y : ℝ, p1.equation x = y ↔ p2.equation (-x) = -y

theorem parabola_symmetry (C1 C2 : Parabola) 
  (h1 : C1.equation = fun x ↦ (x - 2)^2 + 3)
  (h2 : symmetric_about_origin C1 C2) :
  C2.equation = fun x ↦ -(x + 2)^2 - 3 := by
  sorry


end NUMINAMATH_CALUDE_parabola_symmetry_l871_87197


namespace NUMINAMATH_CALUDE_teairra_shirt_count_l871_87108

/-- The number of shirts Teairra has in her closet -/
def num_shirts : ℕ := sorry

/-- The total number of pants Teairra has -/
def total_pants : ℕ := 24

/-- The number of plaid shirts -/
def plaid_shirts : ℕ := 3

/-- The number of purple pants -/
def purple_pants : ℕ := 5

/-- The number of items (shirts and pants) that are neither plaid nor purple -/
def neither_plaid_nor_purple : ℕ := 21

theorem teairra_shirt_count : num_shirts = 5 := by
  sorry

end NUMINAMATH_CALUDE_teairra_shirt_count_l871_87108


namespace NUMINAMATH_CALUDE_three_parallel_non_coplanar_lines_planes_l871_87124

-- Define a structure for a line in 3D space
structure Line3D where
  -- We don't need to specify the exact representation of a line
  -- for this abstract problem

-- Define a property for lines being parallel
def parallel (l1 l2 : Line3D) : Prop := sorry

-- Define a property for lines being coplanar
def coplanar (l1 l2 l3 : Line3D) : Prop := sorry

-- Define a function to count planes through two lines
def planes_through_two_lines (l1 l2 : Line3D) : ℕ := sorry

-- Theorem statement
theorem three_parallel_non_coplanar_lines_planes :
  ∀ (a b c : Line3D),
  parallel a b ∧ parallel b c ∧ parallel a c →
  ¬coplanar a b c →
  (planes_through_two_lines a b + 
   planes_through_two_lines b c + 
   planes_through_two_lines a c) = 3 := 
sorry

end NUMINAMATH_CALUDE_three_parallel_non_coplanar_lines_planes_l871_87124


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l871_87194

-- Define the universe set U
def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}

-- Define set S
def S : Set ℕ := {1, 2, 4, 5}

-- Define set T
def T : Set ℕ := {3, 5, 7}

-- Theorem statement
theorem intersection_complement_equal : S ∩ (U \ T) = {1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l871_87194


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l871_87182

theorem simplify_and_evaluate (a : ℝ) (h : a = 1 - Real.sqrt 2) :
  a * (a - 9) - (a + 3) * (a - 3) = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l871_87182


namespace NUMINAMATH_CALUDE_negation_of_positive_square_plus_x_positive_l871_87158

theorem negation_of_positive_square_plus_x_positive :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_positive_square_plus_x_positive_l871_87158


namespace NUMINAMATH_CALUDE_soda_barrel_leak_time_l871_87132

/-- The time it takes to fill one barrel with the leak -/
def leak_fill_time : ℝ := 5

/-- The normal filling time for one barrel -/
def normal_fill_time : ℝ := 3

/-- The number of barrels -/
def num_barrels : ℝ := 12

/-- The additional time it takes to fill all barrels with the leak -/
def additional_time : ℝ := 24

theorem soda_barrel_leak_time :
  leak_fill_time * num_barrels = normal_fill_time * num_barrels + additional_time :=
by sorry

end NUMINAMATH_CALUDE_soda_barrel_leak_time_l871_87132


namespace NUMINAMATH_CALUDE_transform_f_to_g_l871_87140

def f (x : ℝ) : ℝ := 4 * (x - 3)^2 + 4
def g (x : ℝ) : ℝ := 4 * (x + 3)^2 - 4

theorem transform_f_to_g : 
  ∀ x : ℝ, g x = f (x + 6) - 8 := by sorry

end NUMINAMATH_CALUDE_transform_f_to_g_l871_87140


namespace NUMINAMATH_CALUDE_incorrect_negation_l871_87155

theorem incorrect_negation : 
  ¬(¬(∀ x : ℝ, x^2 - x = 0 → x = 0 ∨ x = 1) ↔ 
    (∀ x : ℝ, x^2 - x = 0 → x ≠ 0 ∧ x ≠ 1)) := by sorry

end NUMINAMATH_CALUDE_incorrect_negation_l871_87155


namespace NUMINAMATH_CALUDE_sebastian_took_no_arabs_l871_87151

theorem sebastian_took_no_arabs (x : ℕ) (y : ℕ) (z : ℕ) : x > 0 →
  -- x is the initial number of each type of soldier
  -- y is the number of cowboys taken (equal to remaining Eskimos)
  -- z is the number of Arab soldiers taken
  y ≤ x →  -- Number of cowboys taken cannot exceed initial number
  4 * x / 3 = y + (x - y) + x / 3 + z →  -- Total soldiers taken
  z = 0 := by
sorry

end NUMINAMATH_CALUDE_sebastian_took_no_arabs_l871_87151


namespace NUMINAMATH_CALUDE_unique_modular_residue_l871_87184

theorem unique_modular_residue :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -3736 [ZMOD 6] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_residue_l871_87184


namespace NUMINAMATH_CALUDE_common_days_off_l871_87199

/-- Earl's work cycle in days -/
def earl_cycle : ℕ := 4

/-- Bob's work cycle in days -/
def bob_cycle : ℕ := 10

/-- Total number of days -/
def total_days : ℕ := 1000

/-- Number of common rest days in one LCM period -/
def common_rest_days_per_lcm : ℕ := 2

/-- Theorem stating the number of common days off for Earl and Bob -/
theorem common_days_off : ℕ := by
  sorry

end NUMINAMATH_CALUDE_common_days_off_l871_87199


namespace NUMINAMATH_CALUDE_bookstore_revenue_theorem_l871_87175

structure BookStore where
  total_books : ℕ
  novels : ℕ
  biographies : ℕ
  science_books : ℕ
  novel_price : ℚ
  biography_price : ℚ
  science_book_price : ℚ
  novel_discount : ℚ
  biography_discount : ℚ
  science_book_discount : ℚ
  remaining_novels : ℕ
  remaining_biographies : ℕ
  remaining_science_books : ℕ
  sales_tax : ℚ

def calculate_total_revenue (store : BookStore) : ℚ :=
  let sold_novels := store.novels - store.remaining_novels
  let sold_biographies := store.biographies - store.remaining_biographies
  let sold_science_books := store.science_books - store.remaining_science_books
  let novel_revenue := (sold_novels : ℚ) * store.novel_price * (1 - store.novel_discount)
  let biography_revenue := (sold_biographies : ℚ) * store.biography_price * (1 - store.biography_discount)
  let science_book_revenue := (sold_science_books : ℚ) * store.science_book_price * (1 - store.science_book_discount)
  let total_discounted_revenue := novel_revenue + biography_revenue + science_book_revenue
  total_discounted_revenue * (1 + store.sales_tax)

theorem bookstore_revenue_theorem (store : BookStore) 
  (h1 : store.total_books = 500)
  (h2 : store.novels + store.biographies + store.science_books = store.total_books)
  (h3 : store.novels - store.remaining_novels = (3 * store.novels) / 5)
  (h4 : store.biographies - store.remaining_biographies = (2 * store.biographies) / 3)
  (h5 : store.science_books - store.remaining_science_books = (7 * store.science_books) / 10)
  (h6 : store.novel_price = 8)
  (h7 : store.biography_price = 12)
  (h8 : store.science_book_price = 10)
  (h9 : store.novel_discount = 1/4)
  (h10 : store.biography_discount = 3/10)
  (h11 : store.science_book_discount = 1/5)
  (h12 : store.remaining_novels = 60)
  (h13 : store.remaining_biographies = 65)
  (h14 : store.remaining_science_books = 50)
  (h15 : store.sales_tax = 1/20)
  : calculate_total_revenue store = 2696.4 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_revenue_theorem_l871_87175


namespace NUMINAMATH_CALUDE_number_divisibility_l871_87101

theorem number_divisibility (A B C D : ℤ) :
  let N := 1000*D + 100*C + 10*B + A
  (∃ k : ℤ, A + 2*B = 4*k → ∃ m : ℤ, N = 4*m) ∧
  (∃ k : ℤ, A + 2*B + 4*C = 8*k → ∃ m : ℤ, N = 8*m) ∧
  (∃ k : ℤ, A + 2*B + 4*C + 8*D = 16*k ∧ ∃ j : ℤ, B = 2*j → ∃ m : ℤ, N = 16*m) :=
by sorry

end NUMINAMATH_CALUDE_number_divisibility_l871_87101


namespace NUMINAMATH_CALUDE_cyrus_pages_left_l871_87130

/-- Represents the number of pages Cyrus writes on each day --/
def pages_written : Fin 4 → ℕ
| 0 => 25  -- Day 1
| 1 => 2 * 25  -- Day 2
| 2 => 2 * (2 * 25)  -- Day 3
| 3 => 10  -- Day 4

/-- The total number of pages Cyrus needs to write --/
def total_pages : ℕ := 500

/-- The number of pages Cyrus still needs to write --/
def pages_left : ℕ := total_pages - (pages_written 0 + pages_written 1 + pages_written 2 + pages_written 3)

theorem cyrus_pages_left : pages_left = 315 := by
  sorry

end NUMINAMATH_CALUDE_cyrus_pages_left_l871_87130


namespace NUMINAMATH_CALUDE_smallest_norm_w_l871_87123

/-- Given a vector w such that ‖w + (4, 2)‖ = 10, 
    the smallest possible value of ‖w‖ is 10 - 2√5 -/
theorem smallest_norm_w (w : ℝ × ℝ) 
    (h : ‖w + (4, 2)‖ = 10) : 
    ∃ (w_min : ℝ × ℝ), ‖w_min‖ = 10 - 2 * Real.sqrt 5 ∧ 
    ∀ (v : ℝ × ℝ), ‖v + (4, 2)‖ = 10 → ‖w_min‖ ≤ ‖v‖ := by
  sorry

end NUMINAMATH_CALUDE_smallest_norm_w_l871_87123


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l871_87171

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l871_87171


namespace NUMINAMATH_CALUDE_exists_function_satisfying_equation_l871_87173

theorem exists_function_satisfying_equation : 
  ∃ f : ℤ → ℤ, ∀ a b : ℤ, f (a + b) - f (a * b) = f a * f b - 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_function_satisfying_equation_l871_87173


namespace NUMINAMATH_CALUDE_max_value_theorem_l871_87178

theorem max_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (heq : a * (a + b + c) = b * c) : 
  a / (b + c) ≤ (Real.sqrt 2 - 1) / 2 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    a₀ * (a₀ + b₀ + c₀) = b₀ * c₀ ∧ 
    a₀ / (b₀ + c₀) = (Real.sqrt 2 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l871_87178


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l871_87100

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l871_87100


namespace NUMINAMATH_CALUDE_initial_water_percentage_l871_87183

theorem initial_water_percentage 
  (initial_volume : ℝ) 
  (added_water : ℝ) 
  (final_water_percentage : ℝ) :
  initial_volume = 120 →
  added_water = 8 →
  final_water_percentage = 25 →
  (initial_volume * (20 / 100) + added_water) / (initial_volume + added_water) = final_water_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l871_87183


namespace NUMINAMATH_CALUDE_boot_purchase_theorem_l871_87198

def boot_purchase_problem (initial_amount hand_sanitizer_discount toilet_paper_cost : ℚ) : ℚ :=
  let hand_sanitizer_cost : ℚ := 6
  let large_ham_cost : ℚ := 2 * toilet_paper_cost
  let cheese_cost : ℚ := hand_sanitizer_cost / 2
  let total_spent : ℚ := toilet_paper_cost + hand_sanitizer_cost + large_ham_cost + cheese_cost
  let remaining : ℚ := initial_amount - total_spent
  let savings : ℚ := remaining * (1/5)
  let spendable : ℚ := remaining - savings
  let per_twin : ℚ := spendable / 2
  let boot_cost : ℚ := per_twin * 4
  let total_boot_cost : ℚ := boot_cost * 2
  (total_boot_cost - spendable) / 2

theorem boot_purchase_theorem :
  boot_purchase_problem 100 (1/4) 12 = 66 := by sorry

end NUMINAMATH_CALUDE_boot_purchase_theorem_l871_87198


namespace NUMINAMATH_CALUDE_sum_equals_zero_l871_87152

theorem sum_equals_zero (a b c : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a * b + c^2 + 4 = 0) : 
  a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_zero_l871_87152


namespace NUMINAMATH_CALUDE_geometric_sum_proof_l871_87139

theorem geometric_sum_proof : 
  let a : ℚ := 3/2
  let r : ℚ := 3/2
  let n : ℕ := 15
  let sum : ℚ := (a * (1 - r^n)) / (1 - r)
  sum = 42948417/32768 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_proof_l871_87139


namespace NUMINAMATH_CALUDE_discount_rates_sum_l871_87129

-- Define the regular prices
def fox_price : ℝ := 15
def pony_price : ℝ := 18

-- Define the number of pairs purchased
def fox_pairs : ℕ := 3
def pony_pairs : ℕ := 2

-- Define the total savings
def total_savings : ℝ := 8.55

-- Define the approximate discount rate for Pony jeans
def pony_discount_rate : ℝ := 0.15

-- Define the discount rates as variables
variable (fox_discount_rate : ℝ)

-- Theorem statement
theorem discount_rates_sum :
  fox_discount_rate + pony_discount_rate = 0.22 :=
by sorry

end NUMINAMATH_CALUDE_discount_rates_sum_l871_87129


namespace NUMINAMATH_CALUDE_photo_arrangements_l871_87189

def num_students : ℕ := 7

def arrangements (n : ℕ) (pair_together : Bool) (avoid_adjacent : Bool) : ℕ := 
  sorry

theorem photo_arrangements : 
  arrangements num_students true true = 1200 := by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l871_87189


namespace NUMINAMATH_CALUDE_division_by_power_equals_negative_exponent_l871_87195

theorem division_by_power_equals_negative_exponent 
  (a : ℝ) (n : ℤ) (h : a > 0) : 
  1 / (a ^ n) = a ^ (0 - n) := by sorry

end NUMINAMATH_CALUDE_division_by_power_equals_negative_exponent_l871_87195


namespace NUMINAMATH_CALUDE_floor_equation_solution_l871_87143

theorem floor_equation_solution (x : ℝ) : 
  (⌊(5 + 6*x) / 8⌋ : ℝ) = (15*x - 7) / 5 ↔ x = 7/15 ∨ x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l871_87143


namespace NUMINAMATH_CALUDE_smallest_m_divisible_by_seven_l871_87149

theorem smallest_m_divisible_by_seven :
  ∃ m : ℕ, m = 6 ∧
  (∀ k : ℕ, k < m → (k^3 + 3^k) % 7 ≠ 0 ∨ (k^2 + 3^k) % 7 ≠ 0) ∧
  (m^3 + 3^m) % 7 = 0 ∧ (m^2 + 3^m) % 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_divisible_by_seven_l871_87149


namespace NUMINAMATH_CALUDE_infinitely_many_prime_divisors_l871_87157

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluation of a quadratic polynomial at an integer -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℤ) : ℤ :=
  p.a * x^2 + p.b * x + p.c

/-- The set of prime divisors of a polynomial's values -/
def primeDivisors (p : QuadraticPolynomial) : Set ℕ :=
  {q : ℕ | Nat.Prime q ∧ ∃ n : ℤ, (q : ℤ) ∣ p.eval n}

/-- The main theorem: there are infinitely many prime divisors for any quadratic polynomial -/
theorem infinitely_many_prime_divisors (p : QuadraticPolynomial) :
  Set.Infinite (primeDivisors p) := by
  sorry


end NUMINAMATH_CALUDE_infinitely_many_prime_divisors_l871_87157


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l871_87193

theorem unique_solution_for_equation :
  ∃! (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (∀ x : ℝ, (a * x + b) ^ 2016 + (x ^ 2 + c * x + d) ^ 1008 = 8 * (x - 2) ^ 2016) ∧
  a = 2 ^ (1 / 672) ∧
  b = -2 * 2 ^ (1 / 672) ∧
  c = -4 ∧
  d = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l871_87193


namespace NUMINAMATH_CALUDE_complex_number_imaginary_part_l871_87168

theorem complex_number_imaginary_part (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (z.im = 2) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_imaginary_part_l871_87168


namespace NUMINAMATH_CALUDE_ratio_of_Q_at_one_and_minus_one_l871_87106

/-- The polynomial g(x) = x^2009 + 19x^2008 + 1 -/
def g (x : ℂ) : ℂ := x^2009 + 19*x^2008 + 1

/-- The set of distinct zeros of g(x) -/
def S : Finset ℂ := sorry

/-- The polynomial Q of degree 2009 -/
noncomputable def Q : Polynomial ℂ := sorry

theorem ratio_of_Q_at_one_and_minus_one 
  (h1 : ∀ s ∈ S, g s = 0)
  (h2 : Finset.card S = 2009)
  (h3 : ∀ s ∈ S, Q.eval (s + 1/s) = 0)
  (h4 : Polynomial.degree Q = 2009) :
  Q.eval 1 / Q.eval (-1) = 361 / 331 := by sorry

end NUMINAMATH_CALUDE_ratio_of_Q_at_one_and_minus_one_l871_87106


namespace NUMINAMATH_CALUDE_expression_value_l871_87165

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 3)    -- absolute value of m is 3
  : (a + b) / m + m^2 - c * d = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l871_87165


namespace NUMINAMATH_CALUDE_laura_change_l871_87174

/-- Calculates the change Laura should receive after her shopping trip -/
theorem laura_change : 
  let pants_cost : ℕ := 2 * 64
  let shirts_cost : ℕ := 4 * 42
  let shoes_cost : ℕ := 3 * 78
  let jackets_cost : ℕ := 2 * 103
  let watch_cost : ℕ := 215
  let jewelry_cost : ℕ := 2 * 120
  let total_cost : ℕ := pants_cost + shirts_cost + shoes_cost + jackets_cost + watch_cost + jewelry_cost
  let amount_given : ℕ := 800
  Int.ofNat amount_given - Int.ofNat total_cost = -391 := by
  sorry

end NUMINAMATH_CALUDE_laura_change_l871_87174


namespace NUMINAMATH_CALUDE_solutions_for_twenty_l871_87114

-- Define a function that counts the number of distinct integer solutions
def count_solutions (n : ℕ+) : ℕ := 4 * n

-- State the theorem
theorem solutions_for_twenty : count_solutions 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_solutions_for_twenty_l871_87114


namespace NUMINAMATH_CALUDE_nth_equation_holds_l871_87134

theorem nth_equation_holds (n : ℕ) :
  (n : ℚ) / (n + 2) * (1 - 1 / (n + 1)) = n^2 / ((n + 1) * (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_holds_l871_87134


namespace NUMINAMATH_CALUDE_repunit_divisibility_l871_87156

theorem repunit_divisibility (p : Nat) (h_prime : Prime p) (h_not_two : p ≠ 2) (h_not_five : p ≠ 5) :
  ∃ n : Nat, ∃ k : Nat, k > 0 ∧ p ∣ (10^n - 1) / 9 :=
sorry

end NUMINAMATH_CALUDE_repunit_divisibility_l871_87156


namespace NUMINAMATH_CALUDE_system_solution_l871_87109

theorem system_solution : 
  ∀ x y : ℚ, 
  x^2 - 9*y^2 = 0 ∧ x + y = 1 → 
  (x = 3/4 ∧ y = 1/4) ∨ (x = 3/2 ∧ y = -1/2) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l871_87109


namespace NUMINAMATH_CALUDE_simple_interest_problem_l871_87150

/-- Given a sum P at simple interest rate R for 10 years, if increasing the rate by 5%
    results in Rs. 300 more interest, then P must equal 600. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 300 → P = 600 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l871_87150


namespace NUMINAMATH_CALUDE_square_roots_problem_l871_87126

theorem square_roots_problem (a : ℝ) (x : ℝ) (h1 : a > 0) 
  (h2 : (3*x - 2)^2 = a) (h3 : (5*x + 6)^2 = a) : a = 49/4 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l871_87126


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_by_nine_l871_87125

theorem smallest_digit_for_divisibility_by_nine :
  ∃ (d : ℕ), d < 10 ∧ 
  (∀ (x : ℕ), x < d → ¬(9 ∣ (438000 + x * 100 + 4))) ∧
  (9 ∣ (438000 + d * 100 + 4)) ∧
  d = 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_by_nine_l871_87125


namespace NUMINAMATH_CALUDE_lecture_schedule_ways_l871_87121

/-- The number of lecturers --/
def n : ℕ := 8

/-- The number of constrained pairs --/
def k : ℕ := 2

/-- The number of ways to schedule n lecturers with k constrained pairs --/
def schedule_ways (n : ℕ) (k : ℕ) : ℕ := n.factorial / (2^k)

/-- Theorem stating the number of ways to schedule the lectures --/
theorem lecture_schedule_ways :
  schedule_ways n k = 10080 := by
  sorry

end NUMINAMATH_CALUDE_lecture_schedule_ways_l871_87121


namespace NUMINAMATH_CALUDE_triangle_third_vertex_l871_87185

/-- Given a triangle with vertices at (8,5), (0,0), and (x,0) where x < 0,
    if the area of the triangle is 40 square units, then x = -16. -/
theorem triangle_third_vertex (x : ℝ) (h1 : x < 0) :
  (1/2 : ℝ) * abs (8 * 0 - x * 5) = 40 → x = -16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_vertex_l871_87185


namespace NUMINAMATH_CALUDE_pool_water_volume_l871_87119

/-- Proves that the volume of water removed from a rectangular pool is 2250 gallons -/
theorem pool_water_volume (length width : ℝ) (depth_inches : ℝ) (conversion_factor : ℝ) : 
  length = 60 → 
  width = 10 → 
  depth_inches = 6 → 
  conversion_factor = 7.5 → 
  length * width * (depth_inches / 12) * conversion_factor = 2250 :=
by
  sorry

#check pool_water_volume

end NUMINAMATH_CALUDE_pool_water_volume_l871_87119


namespace NUMINAMATH_CALUDE_book_club_unique_books_book_club_unique_books_eq_61_l871_87136

theorem book_club_unique_books : ℕ :=
  let tony_books : ℕ := 23
  let dean_books : ℕ := 20
  let breanna_books : ℕ := 30
  let piper_books : ℕ := 26
  let asher_books : ℕ := 25
  let tony_dean_shared : ℕ := 5
  let breanna_piper_asher_shared : ℕ := 7
  let dean_piper_shared : ℕ := 6
  let dean_piper_tony_shared : ℕ := 3
  let asher_breanna_tony_shared : ℕ := 8
  let all_shared : ℕ := 2
  let breanna_piper_shared : ℕ := 9
  let breanna_piper_dean_shared : ℕ := 4
  let breanna_piper_asher_shared : ℕ := 2

  let total_books : ℕ := tony_books + dean_books + breanna_books + piper_books + asher_books
  let overlaps : ℕ := tony_dean_shared + 
                      2 * breanna_piper_asher_shared + 
                      2 * dean_piper_tony_shared + 
                      (dean_piper_shared - dean_piper_tony_shared) + 
                      2 * (asher_breanna_tony_shared - all_shared) + 
                      4 * all_shared + 
                      (breanna_piper_shared - breanna_piper_dean_shared - breanna_piper_asher_shared) + 
                      2 * breanna_piper_dean_shared + 
                      breanna_piper_asher_shared

  total_books - overlaps
  
theorem book_club_unique_books_eq_61 : book_club_unique_books = 61 := by
  sorry

end NUMINAMATH_CALUDE_book_club_unique_books_book_club_unique_books_eq_61_l871_87136


namespace NUMINAMATH_CALUDE_village_population_distribution_l871_87170

theorem village_population_distribution (pop_20k_to_50k : ℝ) (pop_under_20k : ℝ) (pop_50k_and_above : ℝ) :
  pop_20k_to_50k = 45 →
  pop_under_20k = 30 →
  pop_50k_and_above = 25 →
  pop_20k_to_50k + pop_under_20k = 75 :=
by sorry

end NUMINAMATH_CALUDE_village_population_distribution_l871_87170


namespace NUMINAMATH_CALUDE_tangent_slope_product_l871_87115

theorem tangent_slope_product (x₀ : ℝ) : 
  let y₁ : ℝ → ℝ := λ x => 2 - 1/x
  let y₂ : ℝ → ℝ := λ x => x^3 - x^2 + x
  let y₁' : ℝ → ℝ := λ x => 1/x^2
  let y₂' : ℝ → ℝ := λ x => 3*x^2 - 2*x + 1
  (y₁' x₀) * (y₂' x₀) = 3 → x₀ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_product_l871_87115


namespace NUMINAMATH_CALUDE_pecans_weight_l871_87190

def total_nuts : ℝ := 0.52
def almonds : ℝ := 0.14

theorem pecans_weight : total_nuts - almonds = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_pecans_weight_l871_87190


namespace NUMINAMATH_CALUDE_intersection_M_N_l871_87146

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the complement of N in U
def complement_N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := {x | x ∈ U ∧ x ∉ complement_N}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l871_87146


namespace NUMINAMATH_CALUDE_common_zero_implies_f0_or_f1_zero_l871_87162

/-- A quadratic function f(x) = x^2 + px + q -/
def f (p q x : ℝ) : ℝ := x^2 + p * x + q

/-- The composition f(f(f(x))) -/
def triple_f (p q x : ℝ) : ℝ := f p q (f p q (f p q x))

/-- Theorem: If f and triple_f have a common zero, then f(0) = 0 or f(1) = 0 -/
theorem common_zero_implies_f0_or_f1_zero (p q : ℝ) :
  (∃ m, f p q m = 0 ∧ triple_f p q m = 0) →
  f p q 0 = 0 ∨ f p q 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_common_zero_implies_f0_or_f1_zero_l871_87162


namespace NUMINAMATH_CALUDE_pizza_order_theorem_l871_87103

def pizza_order_cost (base_price : ℕ) (topping_price : ℕ) (tip : ℕ) : Prop :=
  let pepperoni_cost : ℕ := base_price + topping_price
  let sausage_cost : ℕ := base_price + topping_price
  let olive_mushroom_cost : ℕ := base_price + 2 * topping_price
  let total_before_tip : ℕ := pepperoni_cost + sausage_cost + olive_mushroom_cost
  let total_with_tip : ℕ := total_before_tip + tip
  total_with_tip = 39

theorem pizza_order_theorem :
  pizza_order_cost 10 1 5 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_order_theorem_l871_87103


namespace NUMINAMATH_CALUDE_robert_ate_more_than_nickel_l871_87118

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 7

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- The difference in chocolates eaten between Robert and Nickel -/
def chocolate_difference : ℕ := robert_chocolates - nickel_chocolates

theorem robert_ate_more_than_nickel : chocolate_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_ate_more_than_nickel_l871_87118


namespace NUMINAMATH_CALUDE_negation_of_proposition_l871_87192

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ m : ℝ, m ≥ 0 → 4^m ≥ 4*m)) ↔ (∃ m : ℝ, m ≥ 0 ∧ 4^m < 4*m) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l871_87192


namespace NUMINAMATH_CALUDE_longest_segment_squared_l871_87196

-- Define the diameter of the pizza
def diameter : ℝ := 16

-- Define the number of slices
def num_slices : ℕ := 4

-- Define the longest line segment in a slice
def longest_segment (d : ℝ) (n : ℕ) : ℝ := d

-- Theorem statement
theorem longest_segment_squared (d : ℝ) (n : ℕ) :
  d = diameter → n = num_slices → (longest_segment d n)^2 = 256 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_squared_l871_87196


namespace NUMINAMATH_CALUDE_total_yellow_marbles_l871_87187

theorem total_yellow_marbles (mary joan tim lisa : ℕ) 
  (h1 : mary = 9) 
  (h2 : joan = 3) 
  (h3 : tim = 5) 
  (h4 : lisa = 7) : 
  mary + joan + tim + lisa = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_yellow_marbles_l871_87187


namespace NUMINAMATH_CALUDE_multiples_of_15_between_10_and_150_l871_87138

theorem multiples_of_15_between_10_and_150 : 
  ∃ n : ℕ, n = (Finset.filter (λ x => 15 ∣ x ∧ x > 10 ∧ x < 150) (Finset.range 150)).card ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_10_and_150_l871_87138


namespace NUMINAMATH_CALUDE_alok_order_l871_87176

/-- The number of chapatis ordered -/
def chapatis : ℕ := 16

/-- The number of rice plates ordered -/
def rice_plates : ℕ := 5

/-- The number of ice-cream cups ordered -/
def ice_cream_cups : ℕ := 6

/-- The cost of each chapati in rupees -/
def chapati_cost : ℕ := 6

/-- The cost of each rice plate in rupees -/
def rice_cost : ℕ := 45

/-- The cost of each mixed vegetable plate in rupees -/
def veg_cost : ℕ := 70

/-- The total amount paid by Alok in rupees -/
def total_paid : ℕ := 985

/-- The number of mixed vegetable plates ordered by Alok -/
def veg_plates : ℕ := (total_paid - (chapatis * chapati_cost + rice_plates * rice_cost)) / veg_cost

theorem alok_order : veg_plates = 9 := by sorry

end NUMINAMATH_CALUDE_alok_order_l871_87176


namespace NUMINAMATH_CALUDE_quadratic_roots_l871_87117

def quadratic_function (a c : ℝ) (x : ℝ) : ℝ := a * x^2 - 2*a*x + c

theorem quadratic_roots (a c : ℝ) (h : a ≠ 0) :
  (quadratic_function a c (-1) = 0) →
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 3 ∧
    ∀ x : ℝ, quadratic_function a c x = 0 ↔ (x = x₁ ∨ x = x₂)) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l871_87117


namespace NUMINAMATH_CALUDE_cubic_equation_integer_roots_l871_87107

theorem cubic_equation_integer_roots :
  ∃! p : ℝ, 
    (∃ x y z : ℕ+, 
      (5 * (x : ℝ)^3 - 5*(p+1)*(x : ℝ)^2 + (71*p - 1)*(x : ℝ) + 1 = 66*p) ∧
      (5 * (y : ℝ)^3 - 5*(p+1)*(y : ℝ)^2 + (71*p - 1)*(y : ℝ) + 1 = 66*p) ∧
      (5 * (z : ℝ)^3 - 5*(p+1)*(z : ℝ)^2 + (71*p - 1)*(z : ℝ) + 1 = 66*p) ∧
      x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
    p = 76 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_roots_l871_87107


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l871_87145

theorem arithmetic_calculation : 8 / 4 - 3 - 9 + 3 * 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l871_87145


namespace NUMINAMATH_CALUDE_charging_piles_growth_equation_l871_87112

/-- Given the number of charging piles built in the first and third months, 
    and the monthly average growth rate, this theorem states the equation 
    that relates these quantities. -/
theorem charging_piles_growth_equation 
  (initial_piles : ℕ) 
  (final_piles : ℕ) 
  (x : ℝ) 
  (h1 : initial_piles = 301)
  (h2 : final_piles = 500)
  (h3 : x ≥ 0) -- Assuming non-negative growth rate
  (h4 : x ≤ 1) -- Assuming growth rate is at most 100%
  : initial_piles * (1 + x)^2 = final_piles := by
  sorry

end NUMINAMATH_CALUDE_charging_piles_growth_equation_l871_87112


namespace NUMINAMATH_CALUDE_cube_ending_with_ones_l871_87154

theorem cube_ending_with_ones (k : ℕ) : ∃ n : ℤ, ∃ m : ℕ, n^3 = m * 10^k + (10^k - 1) := by
  sorry

end NUMINAMATH_CALUDE_cube_ending_with_ones_l871_87154


namespace NUMINAMATH_CALUDE_fifth_term_is_648_l871_87141

/-- A geometric sequence with 7 terms, first term 8, and last term 5832 -/
def GeometricSequence : Type := 
  { a : Fin 7 → ℝ // a 0 = 8 ∧ a 6 = 5832 ∧ ∀ i j, i < j → (a j) / (a i) = (a 1) / (a 0) }

/-- The fifth term of the geometric sequence is 648 -/
theorem fifth_term_is_648 (seq : GeometricSequence) : seq.val 4 = 648 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_648_l871_87141


namespace NUMINAMATH_CALUDE_solution_of_equation_l871_87120

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -3^x else 1 - x^2

-- State the theorem
theorem solution_of_equation (x : ℝ) :
  f x = -3 ↔ x = 1 ∨ x = -2 :=
sorry

end NUMINAMATH_CALUDE_solution_of_equation_l871_87120


namespace NUMINAMATH_CALUDE_jakes_balloons_l871_87122

theorem jakes_balloons (allan_initial : ℕ) (allan_bought : ℕ) (jake_difference : ℕ) :
  allan_initial = 2 →
  allan_bought = 3 →
  jake_difference = 1 →
  allan_initial + allan_bought + jake_difference = 6 :=
by sorry

end NUMINAMATH_CALUDE_jakes_balloons_l871_87122


namespace NUMINAMATH_CALUDE_motorboat_distance_l871_87133

theorem motorboat_distance (boat_speed : ℝ) (time_with_current time_against_current : ℝ) :
  boat_speed = 10 →
  time_with_current = 2 →
  time_against_current = 3 →
  ∃ (distance current_speed : ℝ),
    distance = (boat_speed + current_speed) * time_with_current ∧
    distance = (boat_speed - current_speed) * time_against_current ∧
    distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_motorboat_distance_l871_87133


namespace NUMINAMATH_CALUDE_product_smallest_prime_composite_l871_87127

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def isComposite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, 1 < m ∧ m < n ∧ m ∣ n

def smallestPrime : ℕ := 2

def smallestComposite : ℕ := 4

theorem product_smallest_prime_composite :
  isPrime smallestPrime ∧
  isComposite smallestComposite ∧
  (∀ p : ℕ, isPrime p → p ≥ smallestPrime) ∧
  (∀ c : ℕ, isComposite c → c ≥ smallestComposite) →
  smallestPrime * smallestComposite = 8 :=
by sorry

end NUMINAMATH_CALUDE_product_smallest_prime_composite_l871_87127


namespace NUMINAMATH_CALUDE_function_point_coefficient_l871_87161

/-- Given a function f(x) = ax³ - 2x that passes through the point (-1, 4), prove that a = -2 -/
theorem function_point_coefficient (a : ℝ) : 
  (fun x : ℝ => a * x^3 - 2*x) (-1) = 4 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_point_coefficient_l871_87161


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l871_87116

-- Define the points
def M : ℝ × ℝ := (-2, 3)
def P : ℝ × ℝ := (1, 0)

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Define the Law of Reflection
def law_of_reflection (incident : ℝ × ℝ → ℝ × ℝ → Prop) (reflected : ℝ × ℝ → ℝ × ℝ → Prop) : Prop :=
  ∀ p q r, incident p q → reflected q r → (q.2 = 0) → 
    (p.2 - q.2) * (r.1 - q.1) = (r.2 - q.2) * (p.1 - q.1)

-- State the theorem
theorem reflected_ray_equation :
  ∃ (incident reflected : ℝ × ℝ → ℝ × ℝ → Prop),
    incident M P ∧ P ∈ x_axis ∧ law_of_reflection incident reflected →
    ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧
               ∀ x y : ℝ, reflected P (x, y) ↔ a * x + b * y + c = 0 ∧
               a = 1 ∧ b = 1 ∧ c = -1 := by
  sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l871_87116


namespace NUMINAMATH_CALUDE_sheets_per_ream_l871_87110

theorem sheets_per_ream (cost_per_ream : ℕ) (sheets_needed : ℕ) (total_cost : ℕ) :
  cost_per_ream = 27 →
  sheets_needed = 5000 →
  total_cost = 270 →
  (sheets_needed / (total_cost / cost_per_ream) : ℕ) = 500 := by
  sorry

end NUMINAMATH_CALUDE_sheets_per_ream_l871_87110


namespace NUMINAMATH_CALUDE_solution_value_l871_87188

theorem solution_value (a b x y : ℝ) : 
  x = 1 ∧ y = 1 ∧ 
  a * x + b * y = 2 ∧ 
  x - b * y = 3 →
  a - b = 6 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l871_87188


namespace NUMINAMATH_CALUDE_eight_fifteen_div_sixty_four_cubed_l871_87177

theorem eight_fifteen_div_sixty_four_cubed (x : ℕ) :
  (8 : ℝ) ^ 15 / (64 : ℝ) ^ 3 = (8 : ℝ) ^ 9 := by
  sorry

end NUMINAMATH_CALUDE_eight_fifteen_div_sixty_four_cubed_l871_87177


namespace NUMINAMATH_CALUDE_accounting_client_time_ratio_l871_87137

/-- Given a total work time and time spent calling clients, 
    calculate the ratio of time spent doing accounting to time spent calling clients. -/
theorem accounting_client_time_ratio 
  (total_time : ℕ) 
  (client_time : ℕ) 
  (h1 : total_time = 560) 
  (h2 : client_time = 70) : 
  (total_time - client_time) / client_time = 7 := by
  sorry

end NUMINAMATH_CALUDE_accounting_client_time_ratio_l871_87137


namespace NUMINAMATH_CALUDE_pet_store_puppies_l871_87179

/-- The number of puppies sold --/
def puppies_sold : ℕ := 39

/-- The number of cages used --/
def cages_used : ℕ := 3

/-- The number of puppies per cage --/
def puppies_per_cage : ℕ := 2

/-- The initial number of puppies in the pet store --/
def initial_puppies : ℕ := puppies_sold + cages_used * puppies_per_cage

theorem pet_store_puppies : initial_puppies = 45 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l871_87179


namespace NUMINAMATH_CALUDE_intersection_negative_y_axis_max_value_condition_l871_87144

/-- Linear function y = 3x + 4 - 2m -/
def f (x m : ℝ) : ℝ := 3 * x + 4 - 2 * m

theorem intersection_negative_y_axis (m : ℝ) :
  (f 0 m < 0) ↔ (m > 2) := by sorry

theorem max_value_condition (m : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 3 → f x m ≤ -4) ∧ (∃ x, -2 ≤ x ∧ x ≤ 3 ∧ f x m = -4) →
  m = 8.5 := by sorry

end NUMINAMATH_CALUDE_intersection_negative_y_axis_max_value_condition_l871_87144


namespace NUMINAMATH_CALUDE_parabola_vertex_coefficients_l871_87102

/-- Prove that for a parabola y = ax² + bx with vertex at (3,3), the values of a and b are a = -1/3 and b = 2. -/
theorem parabola_vertex_coefficients (a b : ℝ) : 
  (∀ x, 3 = a * x^2 + b * x ↔ x = 3) ∧ (3 = a * 3^2 + b * 3) → 
  a = -1/3 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_coefficients_l871_87102


namespace NUMINAMATH_CALUDE_road_trip_ratio_l871_87147

theorem road_trip_ratio : 
  ∀ (x : ℝ),
  x > 0 →
  x + 2*x + 40 + 2*(x + 2*x + 40) = 560 →
  40 / x = 9 / 11 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_ratio_l871_87147
