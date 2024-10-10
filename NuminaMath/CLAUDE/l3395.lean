import Mathlib

namespace quadratic_equations_solutions_l3395_339565

theorem quadratic_equations_solutions :
  (∀ x, x * (x - 1) - 3 * (x - 1) = 0 ↔ x = 1 ∨ x = 3) ∧
  (∀ x, x^2 + 2*x - 1 = 0 ↔ x = -1 + Real.sqrt 2 ∨ x = -1 - Real.sqrt 2) := by
  sorry

end quadratic_equations_solutions_l3395_339565


namespace endpoint_sum_l3395_339545

/-- Given a line segment with one endpoint at (10, -5) and its midpoint,
    when scaled by a factor of 2 along each axis, results in the point (12, -18),
    prove that the sum of the coordinates of the other endpoint is -11. -/
theorem endpoint_sum (x y : ℝ) : 
  (10 + x) / 2 = 6 ∧ (-5 + y) / 2 = -9 → x + y = -11 := by
  sorry

end endpoint_sum_l3395_339545


namespace count_valid_pairs_l3395_339569

def validPair (x y : ℕ) : Prop :=
  2 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 16 ∧ 3 * x = y

theorem count_valid_pairs :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 4 ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs ↔ validPair p.1 p.2) :=
sorry

end count_valid_pairs_l3395_339569


namespace complement_intersection_M_N_l3395_339542

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {1, 3, 4, 6}

theorem complement_intersection_M_N : (M ∩ N)ᶜ = {2, 4, 5, 6} := by sorry

end complement_intersection_M_N_l3395_339542


namespace rhombus_side_length_l3395_339539

/-- A rhombus with perimeter 60 cm has sides of length 15 cm each. -/
theorem rhombus_side_length (perimeter : ℝ) (side_length : ℝ) : 
  perimeter = 60 → side_length * 4 = perimeter → side_length = 15 := by
  sorry

#check rhombus_side_length

end rhombus_side_length_l3395_339539


namespace square_root_equation_l3395_339564

theorem square_root_equation (t s : ℝ) : t = 15 * s^2 → t = 3.75 → s = 0.5 := by
  sorry

end square_root_equation_l3395_339564


namespace sum_equals_zero_l3395_339503

theorem sum_equals_zero : 1 + 1 - 2 + 3 + 5 - 8 + 13 + 21 - 34 = 0 := by
  sorry

end sum_equals_zero_l3395_339503


namespace distance_to_other_focus_l3395_339520

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 12 = 1

/-- Distance from a point to one focus is 3 -/
def distance_to_one_focus (x y : ℝ) : Prop :=
  ∃ (fx fy : ℝ), (x - fx)^2 + (y - fy)^2 = 3^2

/-- Theorem: If a point is on the ellipse and its distance to one focus is 3,
    then its distance to the other focus is 5 -/
theorem distance_to_other_focus
  (x y : ℝ)
  (h1 : is_on_ellipse x y)
  (h2 : distance_to_one_focus x y) :
  ∃ (gx gy : ℝ), (x - gx)^2 + (y - gy)^2 = 5^2 :=
sorry

end distance_to_other_focus_l3395_339520


namespace intersection_complement_when_a_three_a_greater_than_four_when_A_subset_B_l3395_339568

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

-- Theorem for part 1
theorem intersection_complement_when_a_three :
  A ∩ (Set.univ \ B 3) = Set.Icc 3 4 := by sorry

-- Theorem for part 2
theorem a_greater_than_four_when_A_subset_B (a : ℝ) :
  A ⊆ B a → a > 4 := by sorry

end intersection_complement_when_a_three_a_greater_than_four_when_A_subset_B_l3395_339568


namespace system_solution_l3395_339578

-- Define the system of equations
def equation1 (x y : ℚ) : Prop := (2 * x - 3) / (3 * x - y) = 3 / 5
def equation2 (x y : ℚ) : Prop := x^2 + y = 7

-- Define the solution set
def solution_set : Set (ℚ × ℚ) := {(-2/3, 47/9), (3, 4)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℚ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set :=
by sorry

end system_solution_l3395_339578


namespace flour_already_added_l3395_339557

/-- Given a cake recipe and Mary's baking progress, calculate the cups of flour already added. -/
theorem flour_already_added
  (total_flour : ℕ)  -- Total cups of flour required by the recipe
  (sugar : ℕ)        -- Cups of sugar required by the recipe
  (h1 : total_flour = 14)  -- The recipe requires 14 cups of flour
  (h2 : sugar = 9)         -- The recipe requires 9 cups of sugar
  : total_flour - (sugar + 1) = 4 := by
  sorry

end flour_already_added_l3395_339557


namespace polynomial_comparison_l3395_339501

theorem polynomial_comparison : ∀ x : ℝ, (x - 3) * (x - 2) > (x + 1) * (x - 6) := by
  sorry

end polynomial_comparison_l3395_339501


namespace remainder_problem_l3395_339590

theorem remainder_problem (N : ℕ) (h : N % 899 = 63) : N % 29 = 5 := by
  sorry

end remainder_problem_l3395_339590


namespace intersection_point_on_graph_and_y_axis_l3395_339582

/-- The quadratic function f(x) = (x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- The point where f intersects the y-axis -/
def intersection_point : ℝ × ℝ := (0, 3)

/-- Theorem: The intersection_point lies on both the y-axis and the graph of f -/
theorem intersection_point_on_graph_and_y_axis :
  (intersection_point.1 = 0) ∧ 
  (intersection_point.2 = f intersection_point.1) :=
by sorry

end intersection_point_on_graph_and_y_axis_l3395_339582


namespace desired_circle_properties_l3395_339589

/-- The first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

/-- The second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

/-- The line on which the center of the desired circle lies -/
def centerLine (x y : ℝ) : Prop := x + y = 0

/-- The equation of the desired circle -/
def desiredCircle (x y : ℝ) : Prop := (x + 3)^2 + (y - 3)^2 = 10

/-- Theorem stating that the desired circle passes through the intersection points of circle1 and circle2,
    and its center lies on the centerLine -/
theorem desired_circle_properties :
  ∀ x y : ℝ, 
    (circle1 x y ∧ circle2 x y) → 
    desiredCircle x y ∧ 
    ∃ cx cy : ℝ, centerLine cx cy ∧ desiredCircle (x - cx) (y - cy) := by
  sorry


end desired_circle_properties_l3395_339589


namespace find_y_value_l3395_339550

-- Define the operation
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

-- State the theorem
theorem find_y_value : ∃ y : ℤ, customOp y 12 = 110 ∧ y = 11 := by
  sorry

end find_y_value_l3395_339550


namespace max_quadratic_solution_power_l3395_339506

/-- Given positive integers a, b, c that are powers of k, and r is the unique real solution
    to ax^2 - bx + c = 0 where r < 100, prove that the maximum possible value of r is 64 -/
theorem max_quadratic_solution_power (k a b c : ℕ+) (r : ℝ) :
  (∃ m n p : ℕ, a = k ^ m ∧ b = k ^ n ∧ c = k ^ p) →
  (∀ x : ℝ, a * x^2 - b * x + c = 0 ↔ x = r) →
  r < 100 →
  r ≤ 64 :=
sorry

end max_quadratic_solution_power_l3395_339506


namespace minimum_value_theorem_l3395_339529

theorem minimum_value_theorem (x y m : ℝ) :
  x > 0 →
  y > 0 →
  (4 / x + 9 / y = m) →
  (∀ a b : ℝ, a > 0 → b > 0 → 4 / a + 9 / b = m → x + y ≤ a + b) →
  x + y = 5 / 6 →
  m = 30 := by
sorry

end minimum_value_theorem_l3395_339529


namespace consecutive_negative_integers_sum_l3395_339513

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 3080 → n + (n + 1) = -111 := by
  sorry

end consecutive_negative_integers_sum_l3395_339513


namespace marks_remaining_money_l3395_339581

/-- Calculates the remaining money after a purchase --/
def remaining_money (initial_amount : ℕ) (num_books : ℕ) (price_per_book : ℕ) : ℕ :=
  initial_amount - (num_books * price_per_book)

/-- Proves that Mark is left with $35 after his purchase --/
theorem marks_remaining_money :
  remaining_money 85 10 5 = 35 := by
  sorry

end marks_remaining_money_l3395_339581


namespace lcm_24_30_40_l3395_339517

theorem lcm_24_30_40 : Nat.lcm (Nat.lcm 24 30) 40 = 120 := by sorry

end lcm_24_30_40_l3395_339517


namespace max_digit_occurrence_l3395_339583

/-- Represents the range of apartment numbers on each floor -/
def apartment_range : Set ℕ := {n | 0 ≤ n ∧ n ≤ 35}

/-- Counts the occurrences of a digit in a given number -/
def count_digit (d : ℕ) (n : ℕ) : ℕ := sorry

/-- Counts the occurrences of a digit in a range of numbers -/
def count_digit_in_range (d : ℕ) (range : Set ℕ) : ℕ := sorry

/-- Counts the occurrences of a digit in the hundreds place for a floor -/
def count_digit_hundreds (d : ℕ) (floor : ℕ) : ℕ := sorry

/-- The main theorem stating that the maximum occurrence of any digit is 36 -/
theorem max_digit_occurrence :
  ∃ d : ℕ, d < 10 ∧
    (count_digit_in_range d apartment_range +
     count_digit_in_range d apartment_range +
     count_digit_in_range d apartment_range +
     count_digit_hundreds 1 1 +
     count_digit_hundreds 2 2 +
     count_digit_hundreds 3 3) = 36 ∧
    ∀ d' : ℕ, d' < 10 →
      (count_digit_in_range d' apartment_range +
       count_digit_in_range d' apartment_range +
       count_digit_in_range d' apartment_range +
       count_digit_hundreds 1 1 +
       count_digit_hundreds 2 2 +
       count_digit_hundreds 3 3) ≤ 36 := by
  sorry

#check max_digit_occurrence

end max_digit_occurrence_l3395_339583


namespace climb_10_stairs_l3395_339576

/-- Function representing the number of ways to climb n stairs -/
def climb_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | n + 4 => climb_ways (n + 3) + climb_ways (n + 2) + climb_ways n

/-- Theorem stating that there are 151 ways to climb 10 stairs -/
theorem climb_10_stairs : climb_ways 10 = 151 := by
  sorry

end climb_10_stairs_l3395_339576


namespace equation_solve_for_n_l3395_339508

theorem equation_solve_for_n (s P k c n : ℝ) (h1 : c > 0) (h2 : P = s / (c * (1 + k)^n)) :
  n = Real.log (s / (P * c)) / Real.log (1 + k) := by
  sorry

end equation_solve_for_n_l3395_339508


namespace book_arrangement_count_book_arrangement_proof_l3395_339584

theorem book_arrangement_count : ℕ → ℕ → ℕ
  | 4, 6 => 17280
  | _, _ => 0

/-- The number of ways to arrange math and history books with specific constraints -/
theorem book_arrangement_proof (m h : ℕ) (hm : m = 4) (hh : h = 6) :
  book_arrangement_count m h = 4 * 3 * 2 * Nat.factorial h :=
by sorry

end book_arrangement_count_book_arrangement_proof_l3395_339584


namespace total_marks_calculation_l3395_339533

theorem total_marks_calculation (obtained_marks : ℝ) (percentage : ℝ) (total_marks : ℝ) : 
  obtained_marks = 450 → percentage = 90 → obtained_marks = (percentage / 100) * total_marks → 
  total_marks = 500 := by
  sorry

end total_marks_calculation_l3395_339533


namespace function_growth_l3395_339546

theorem function_growth (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) : 
  f a > Real.exp a * f 0 :=
sorry

end function_growth_l3395_339546


namespace P_necessary_not_sufficient_for_Q_l3395_339591

-- Define propositions P and Q
def P (x : ℝ) : Prop := |x - 1| < 4
def Q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- Theorem statement
theorem P_necessary_not_sufficient_for_Q :
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬Q x) :=
sorry

end P_necessary_not_sufficient_for_Q_l3395_339591


namespace equation_solutions_l3395_339566

theorem equation_solutions : 
  (∀ x : ℝ, x^2 - 10*x + 16 = 0 ↔ x = 8 ∨ x = 2) ∧
  (∀ x : ℝ, x*(x-3) = 6-2*x ↔ x = 3 ∨ x = -2) := by sorry

end equation_solutions_l3395_339566


namespace max_value_of_sum_and_reciprocal_l3395_339536

theorem max_value_of_sum_and_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 := by
  sorry

end max_value_of_sum_and_reciprocal_l3395_339536


namespace arithmetic_geometric_ratio_l3395_339518

-- Define the arithmetic sequence
def arithmetic_seq (a₁ a₂ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ - a₁ = d ∧ a₁ - (-2) = d ∧ (-8) - a₂ = d

-- Define the geometric sequence
def geometric_seq (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ r : ℝ, b₁ / (-2) = r ∧ b₂ / b₁ = r ∧ b₃ / b₂ = r ∧ (-8) / b₃ = r

theorem arithmetic_geometric_ratio
  (a₁ a₂ b₁ b₂ b₃ : ℝ)
  (h_arith : arithmetic_seq a₁ a₂)
  (h_geom : geometric_seq b₁ b₂ b₃) :
  (a₂ - a₁) / b₂ = 1/2 := by
  sorry

end arithmetic_geometric_ratio_l3395_339518


namespace sector_arc_length_l3395_339573

/-- Given a sector with area 9 and central angle 2 radians, its arc length is 6. -/
theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) : 
  area = 9 → angle = 2 → arc_length = 6 := by
  sorry

end sector_arc_length_l3395_339573


namespace fixed_point_on_AB_l3395_339588

-- Define the circle C
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line L
def Line (x y : ℝ) : Prop := x + y = 9

-- Define a point P on line L
def PointOnLine (P : ℝ × ℝ) : Prop := Line P.1 P.2

-- Define tangent line from P to circle C
def TangentLine (P A : ℝ × ℝ) : Prop :=
  Circle A.1 A.2 ∧ (∃ t : ℝ, A.1 = P.1 + t * (A.2 - P.2) ∧ A.2 = P.2 - t * (A.1 - P.1))

-- Theorem statement
theorem fixed_point_on_AB (P A B : ℝ × ℝ) :
  PointOnLine P →
  TangentLine P A →
  TangentLine P B →
  A ≠ B →
  ∃ t : ℝ, (4/9 : ℝ) = A.1 + t * (B.1 - A.1) ∧ (8/9 : ℝ) = A.2 + t * (B.2 - A.2) :=
sorry

end fixed_point_on_AB_l3395_339588


namespace bruno_initial_books_l3395_339579

/-- The number of books Bruno initially had -/
def initial_books : ℕ := sorry

/-- The number of books Bruno lost -/
def lost_books : ℕ := 4

/-- The number of books Bruno's dad gave him -/
def gained_books : ℕ := 10

/-- The final number of books Bruno had -/
def final_books : ℕ := 39

/-- Theorem stating that Bruno initially had 33 books -/
theorem bruno_initial_books : 
  initial_books = 33 ∧ 
  initial_books - lost_books + gained_books = final_books :=
sorry

end bruno_initial_books_l3395_339579


namespace second_caterer_cheaper_at_34_l3395_339555

/-- Represents the cost function for a caterer -/
structure CatererCost where
  basicFee : ℕ
  perPersonCost : ℕ

/-- Calculates the total cost for a given number of people -/
def totalCost (c : CatererCost) (people : ℕ) : ℕ :=
  c.basicFee + c.perPersonCost * people

/-- First caterer's cost structure -/
def caterer1 : CatererCost :=
  { basicFee := 50, perPersonCost := 18 }

/-- Second caterer's cost structure -/
def caterer2 : CatererCost :=
  { basicFee := 150, perPersonCost := 15 }

/-- Theorem stating that 34 is the least number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper_at_34 :
  (∀ n : ℕ, n < 34 → totalCost caterer1 n ≤ totalCost caterer2 n) ∧
  (totalCost caterer1 34 > totalCost caterer2 34) :=
sorry

end second_caterer_cheaper_at_34_l3395_339555


namespace line_equation_specific_l3395_339500

/-- The equation of a line with given slope and y-intercept -/
def line_equation (slope : ℝ) (y_intercept : ℝ) : ℝ → ℝ := λ x => slope * x + y_intercept

/-- Theorem: The equation of a line with slope 2 and y-intercept 1 is y = 2x + 1 -/
theorem line_equation_specific : line_equation 2 1 = λ x => 2 * x + 1 := by
  sorry

end line_equation_specific_l3395_339500


namespace largest_smallest_factor_l3395_339511

theorem largest_smallest_factor (a b c : ℕ+) : 
  a * b * c = 2160 → 
  ∃ (x : ℕ+), x ≤ a ∧ x ≤ b ∧ x ≤ c ∧ 
  (∀ (y : ℕ+), y ≤ a ∧ y ≤ b ∧ y ≤ c → y ≤ x) ∧ 
  x ≤ 10 :=
by sorry

end largest_smallest_factor_l3395_339511


namespace two_books_different_genres_count_l3395_339521

/-- Represents the number of books in each genre -/
def booksPerGenre : ℕ := 3

/-- Represents the number of genres -/
def numberOfGenres : ℕ := 4

/-- Represents the number of genres to choose -/
def genresToChoose : ℕ := 2

/-- Calculates the number of ways to choose two books of different genres -/
def chooseTwoBooksOfDifferentGenres : ℕ :=
  Nat.choose numberOfGenres genresToChoose * booksPerGenre * booksPerGenre

theorem two_books_different_genres_count :
  chooseTwoBooksOfDifferentGenres = 54 := by
  sorry

end two_books_different_genres_count_l3395_339521


namespace union_of_M_and_N_l3395_339556

-- Define the sets M and N
def M : Set ℝ := {x | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x | x < -5 ∨ x > 5}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x | x < -5 ∨ x > -3} := by
  sorry

end union_of_M_and_N_l3395_339556


namespace smallest_n_congruence_l3395_339574

theorem smallest_n_congruence (n : ℕ) : n = 11 ↔ 
  (n > 0 ∧ 19 * n ≡ 546 [ZMOD 13] ∧ 
   ∀ m : ℕ, m > 0 ∧ m < n → ¬(19 * m ≡ 546 [ZMOD 13])) := by
  sorry

end smallest_n_congruence_l3395_339574


namespace inequality_implies_a_range_l3395_339562

/-- If ln x - ax ≤ 2a² - 3 holds for all x > 0, then a ≥ 1 -/
theorem inequality_implies_a_range (a : ℝ) :
  (∀ x > 0, Real.log x - a * x ≤ 2 * a^2 - 3) → a ≥ 1 := by
  sorry

end inequality_implies_a_range_l3395_339562


namespace max_visible_cubes_9x9x9_l3395_339548

/-- Represents a cube made of unit cubes --/
structure Cube where
  size : ℕ

/-- Calculates the number of visible unit cubes from a corner of the cube --/
def visibleUnitCubes (c : Cube) : ℕ :=
  3 * c.size^2 - 3 * (c.size - 1) + 1

/-- Theorem: The maximum number of visible unit cubes from a single point in a 9x9x9 cube is 220 --/
theorem max_visible_cubes_9x9x9 :
  ∀ (c : Cube), c.size = 9 → visibleUnitCubes c = 220 := by
  sorry

#eval visibleUnitCubes { size := 9 }

end max_visible_cubes_9x9x9_l3395_339548


namespace password_decryption_probability_l3395_339540

theorem password_decryption_probability 
  (p : ℝ) 
  (hp : p = 1 / 4) 
  (n : ℕ) 
  (hn : n = 3) :
  (n.choose 2 : ℝ) * p^2 * (1 - p) = 9 / 64 := by
  sorry

end password_decryption_probability_l3395_339540


namespace crepe_myrtle_count_l3395_339596

theorem crepe_myrtle_count (total : ℕ) (pink : ℕ) (red : ℕ) (white : ℕ) : 
  total = 42 →
  pink = total / 3 →
  red = 2 →
  white > pink →
  white > red →
  total = pink + red + white →
  white = 26 := by
sorry

end crepe_myrtle_count_l3395_339596


namespace sum_of_reciprocals_shifted_roots_l3395_339570

theorem sum_of_reciprocals_shifted_roots (a b c : ℂ) : 
  (a^3 - a - 2 = 0) → (b^3 - b - 2 = 0) → (c^3 - c - 2 = 0) →
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 1 := by
  sorry

end sum_of_reciprocals_shifted_roots_l3395_339570


namespace skipping_odometer_theorem_l3395_339541

/-- Represents an odometer that skips the digit 6 -/
def SkippingOdometer : Type := ℕ

/-- Converts a regular odometer reading to a skipping odometer reading -/
def toSkippingReading (n : ℕ) : SkippingOdometer :=
  sorry

/-- Converts a skipping odometer reading back to the actual distance -/
def toActualDistance (s : SkippingOdometer) : ℕ :=
  sorry

theorem skipping_odometer_theorem :
  toActualDistance (toSkippingReading 1464) = 2005 :=
sorry

end skipping_odometer_theorem_l3395_339541


namespace vector_collinearity_l3395_339572

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- The problem statement -/
theorem vector_collinearity (m : ℝ) :
  collinear (m + 3, 2) (m, 1) → m = 3 := by
  sorry

end vector_collinearity_l3395_339572


namespace union_equality_iff_a_in_range_l3395_339526

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- State the theorem
theorem union_equality_iff_a_in_range (a : ℝ) :
  M a ∪ N = N ↔ a ∈ Set.Icc (-2) 2 := by sorry

end union_equality_iff_a_in_range_l3395_339526


namespace complex_sum_modulus_l3395_339519

theorem complex_sum_modulus : 
  Complex.abs (1/5 - (2/5)*Complex.I) + Complex.abs (3/5 + (4/5)*Complex.I) = (1 + Real.sqrt 5) / Real.sqrt 5 := by
  sorry

end complex_sum_modulus_l3395_339519


namespace egg_difference_l3395_339595

/-- Given that Megan bought 2 dozen eggs, 3 eggs broke, and twice as many cracked,
    prove that the difference between eggs in perfect condition and cracked eggs is 9. -/
theorem egg_difference (total : ℕ) (broken : ℕ) (cracked : ℕ) :
  total = 2 * 12 →
  broken = 3 →
  cracked = 2 * broken →
  total - (broken + cracked) - cracked = 9 := by
  sorry

end egg_difference_l3395_339595


namespace opposite_reciprocal_problem_l3395_339559

theorem opposite_reciprocal_problem (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 4) : 
  (a + b = 0) ∧ 
  (c * d = 1) ∧ 
  ((a + b) / 3 + m^2 - 5 * c * d = 11) := by
sorry

end opposite_reciprocal_problem_l3395_339559


namespace factorial_divisibility_l3395_339577

theorem factorial_divisibility (n : ℕ) (M : ℕ) (h : Nat.factorial 100 = 12^n * M) 
  (h_max : ∀ k : ℕ, Nat.factorial 100 = 12^k * M → k ≤ n) : 
  (2 ∣ M) ∧ ¬(3 ∣ M) := by
sorry

end factorial_divisibility_l3395_339577


namespace count_digit_six_is_280_l3395_339507

/-- Count of digit 6 in integers from 100 to 999 -/
def count_digit_six : ℕ :=
  let hundreds := 100  -- 600 to 699
  let tens := 9 * 10   -- 10 numbers per hundred, 9 hundreds
  let ones := 9 * 10   -- 10 numbers per hundred, 9 hundreds
  hundreds + tens + ones

/-- The count of digit 6 in integers from 100 to 999 is 280 -/
theorem count_digit_six_is_280 : count_digit_six = 280 := by
  sorry

end count_digit_six_is_280_l3395_339507


namespace study_time_for_average_score_l3395_339523

/-- Represents the relationship between study time and test score -/
structure StudyScoreRelation where
  studyTime : ℝ
  score : ℝ
  ratio : ℝ
  direct_relation : ratio = score / studyTime

/-- The problem setup and solution -/
theorem study_time_for_average_score
  (first_exam : StudyScoreRelation)
  (h_first_exam : first_exam.studyTime = 3 ∧ first_exam.score = 60)
  (target_average : ℝ)
  (h_target_average : target_average = 75)
  : ∃ (second_exam : StudyScoreRelation),
    second_exam.ratio = first_exam.ratio ∧
    (first_exam.score + second_exam.score) / 2 = target_average ∧
    second_exam.studyTime = 4.5 := by
  sorry

end study_time_for_average_score_l3395_339523


namespace f_lower_bound_a_range_l3395_339516

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x + 2*a + 3|

-- Theorem 1: For all real x and a, f(x) ≥ 2
theorem f_lower_bound (x a : ℝ) : f x a ≥ 2 := by
  sorry

-- Theorem 2: If f(-3/2) < 3, then -1 < a < 0
theorem a_range (a : ℝ) : f (-3/2) a < 3 → -1 < a ∧ a < 0 := by
  sorry

end f_lower_bound_a_range_l3395_339516


namespace ferry_speed_difference_l3395_339554

/-- Represents the speed and time of a ferry journey -/
structure FerryJourney where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a ferry -/
def distance (journey : FerryJourney) : ℝ :=
  journey.speed * journey.time

theorem ferry_speed_difference :
  let ferryP : FerryJourney := { speed := 8, time := 3 }
  let ferryQ : FerryJourney := { speed := (3 * distance ferryP) / (ferryP.time + 5), time := ferryP.time + 5 }
  ferryQ.speed - ferryP.speed = 1 := by
  sorry

end ferry_speed_difference_l3395_339554


namespace total_students_is_44_l3395_339563

/-- Represents the number of students who borrowed a specific number of books -/
structure BookBorrowers where
  zero : Nat
  one : Nat
  two : Nat
  threeOrMore : Nat

/-- Calculates the total number of students -/
def totalStudents (b : BookBorrowers) : Nat :=
  b.zero + b.one + b.two + b.threeOrMore

/-- Calculates the minimum number of books borrowed -/
def minBooksBorrowed (b : BookBorrowers) : Nat :=
  0 * b.zero + 1 * b.one + 2 * b.two + 3 * b.threeOrMore

/-- Theorem stating that the total number of students in the class is 44 -/
theorem total_students_is_44 (b : BookBorrowers) : 
  b.zero = 2 → 
  b.one = 12 → 
  b.two = 14 → 
  minBooksBorrowed b = 2 * totalStudents b → 
  totalStudents b = 44 := by
  sorry

end total_students_is_44_l3395_339563


namespace equation_is_linear_l3395_339528

/-- Definition of a linear equation with two variables -/
def is_linear_equation_two_vars (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation 3x = 2y -/
def equation (x y : ℝ) : Prop := 3 * x = 2 * y

/-- Theorem: The equation 3x = 2y is a linear equation with two variables -/
theorem equation_is_linear : is_linear_equation_two_vars equation := by
  sorry

end equation_is_linear_l3395_339528


namespace vector_angle_in_circle_l3395_339597

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define the theorem
theorem vector_angle_in_circle (O A B C : ℝ × ℝ) (r : ℝ) :
  A ∈ Circle O r →
  B ∈ Circle O r →
  C ∈ Circle O r →
  (A.1 - O.1, A.2 - O.2) = (1/2) * ((B.1 - A.1, B.2 - A.2) + (C.1 - A.1, C.2 - A.2)) →
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 :=
by sorry

end vector_angle_in_circle_l3395_339597


namespace total_people_in_program_l3395_339514

theorem total_people_in_program (parents : Nat) (pupils : Nat) 
  (h1 : parents = 105) (h2 : pupils = 698) : 
  parents + pupils = 803 := by
  sorry

end total_people_in_program_l3395_339514


namespace colleen_pencils_colleen_pencils_proof_l3395_339560

theorem colleen_pencils (joy_pencils : ℕ) (pencil_cost : ℕ) (colleen_extra : ℕ) : ℕ :=
  let joy_total := joy_pencils * pencil_cost
  let colleen_total := joy_total + colleen_extra
  colleen_total / pencil_cost

#check colleen_pencils 30 4 80 = 50

theorem colleen_pencils_proof :
  colleen_pencils 30 4 80 = 50 := by
  sorry

end colleen_pencils_colleen_pencils_proof_l3395_339560


namespace jessie_weight_loss_l3395_339575

/-- Jessie's weight loss calculation -/
theorem jessie_weight_loss (initial_weight current_weight : ℕ) :
  initial_weight = 69 →
  current_weight = 34 →
  initial_weight - current_weight = 35 := by
sorry

end jessie_weight_loss_l3395_339575


namespace sandy_age_l3395_339509

theorem sandy_age (S M : ℕ) (h1 : M = S + 14) (h2 : S / M = 7 / 9) : S = 49 := by
  sorry

end sandy_age_l3395_339509


namespace triangle_side_values_l3395_339543

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_values :
  ∀ y : ℕ+,
    (triangle_exists 8 11 (y.val ^ 2 - 1) ↔ y.val = 3 ∨ y.val = 4) :=
by sorry

end triangle_side_values_l3395_339543


namespace agnes_twice_jane_age_l3395_339593

/-- The number of years until Agnes is twice as old as Jane -/
def years_until_double_age (agnes_age : ℕ) (jane_age : ℕ) : ℕ :=
  (agnes_age - 2 * jane_age) / (2 - 1)

/-- Theorem stating that it will take 13 years for Agnes to be twice as old as Jane -/
theorem agnes_twice_jane_age (agnes_current_age jane_current_age : ℕ) 
  (h1 : agnes_current_age = 25) 
  (h2 : jane_current_age = 6) : 
  years_until_double_age agnes_current_age jane_current_age = 13 := by
  sorry

end agnes_twice_jane_age_l3395_339593


namespace sufficient_lunks_for_bananas_l3395_339534

/-- Represents the exchange rate between lunks and kunks -/
def lunk_to_kunk_rate : ℚ := 6 / 10

/-- Represents the exchange rate between kunks and bananas -/
def kunk_to_banana_rate : ℚ := 5 / 3

/-- The number of bananas we want to purchase -/
def target_bananas : ℕ := 24

/-- The number of lunks we claim is sufficient -/
def claimed_lunks : ℕ := 25

theorem sufficient_lunks_for_bananas :
  ∃ (kunks : ℚ),
    kunks * kunk_to_banana_rate ≥ target_bananas ∧
    kunks ≤ claimed_lunks * lunk_to_kunk_rate :=
by
  sorry

#check sufficient_lunks_for_bananas

end sufficient_lunks_for_bananas_l3395_339534


namespace money_distribution_l3395_339571

theorem money_distribution (a b c : ℕ) 
  (total : a + b + c = 400)
  (ac_sum : a + c = 300)
  (bc_sum : b + c = 150) :
  c = 50 := by
  sorry

end money_distribution_l3395_339571


namespace rational_equation_solution_l3395_339551

theorem rational_equation_solution (x : ℝ) : 
  (1 / (x^2 + 8*x - 6) + 1 / (x^2 + 5*x - 6) + 1 / (x^2 - 14*x - 6) = 0) ↔ 
  (x = 3 ∨ x = -2 ∨ x = -6 ∨ x = 1) :=
by sorry

end rational_equation_solution_l3395_339551


namespace grapefruit_orchards_l3395_339531

theorem grapefruit_orchards (total : ℕ) (lemon : ℕ) (orange : ℕ) (lime_grapefruit : ℕ) :
  total = 16 →
  lemon = 8 →
  orange = lemon / 2 →
  lime_grapefruit = total - lemon - orange →
  lime_grapefruit / 2 = 2 :=
by
  sorry

end grapefruit_orchards_l3395_339531


namespace electionWaysCount_l3395_339580

/-- Represents the Science Club with its election rules -/
structure ScienceClub where
  totalMembers : Nat
  aliceIndex : Nat
  bobIndex : Nat

/-- Represents the possible election outcomes -/
inductive ElectionOutcome
  | WithoutAliceAndBob (president secretary treasurer : Nat)
  | WithAliceAndBob (treasurer : Nat)

/-- Checks if an election outcome is valid according to the club's rules -/
def isValidOutcome (club : ScienceClub) (outcome : ElectionOutcome) : Prop :=
  match outcome with
  | ElectionOutcome.WithoutAliceAndBob p s t =>
      p ≠ club.aliceIndex ∧ p ≠ club.bobIndex ∧
      s ≠ club.aliceIndex ∧ s ≠ club.bobIndex ∧
      t ≠ club.aliceIndex ∧ t ≠ club.bobIndex ∧
      p ≠ s ∧ p ≠ t ∧ s ≠ t ∧
      p < club.totalMembers ∧ s < club.totalMembers ∧ t < club.totalMembers
  | ElectionOutcome.WithAliceAndBob t =>
      t ≠ club.aliceIndex ∧ t ≠ club.bobIndex ∧
      t < club.totalMembers

/-- Counts the number of valid election outcomes -/
def countValidOutcomes (club : ScienceClub) : Nat :=
  sorry

/-- The main theorem stating the number of ways to elect officers -/
theorem electionWaysCount (club : ScienceClub) 
    (h1 : club.totalMembers = 25)
    (h2 : club.aliceIndex < club.totalMembers)
    (h3 : club.bobIndex < club.totalMembers)
    (h4 : club.aliceIndex ≠ club.bobIndex) :
    countValidOutcomes club = 10649 :=
  sorry

end electionWaysCount_l3395_339580


namespace bus_bike_time_difference_l3395_339598

/-- Proves that the difference between bus and bike commute times is 10 minutes -/
theorem bus_bike_time_difference :
  ∀ (bus_time : ℕ),
  (30 + 3 * bus_time + 10 = 160) →
  (bus_time - 30 = 10) := by
  sorry

end bus_bike_time_difference_l3395_339598


namespace power_fraction_simplification_l3395_339586

theorem power_fraction_simplification :
  (3^2023 + 3^2021) / (3^2023 - 3^2021) = 5/4 := by
  sorry

end power_fraction_simplification_l3395_339586


namespace square_binomial_constant_l3395_339567

theorem square_binomial_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 200*x + c = (x + a)^2) → c = 10000 := by
  sorry

end square_binomial_constant_l3395_339567


namespace quadratic_function_properties_l3395_339561

/-- A quadratic function satisfying certain conditions -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h1 : ∀ x, f a b c (x + 1) - f a b c x = 2 * x)
  (h2 : f a b c 0 = 1) :
  (∀ x, f a b c x = x^2 - x + 1) ∧
  (∀ m, (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f a b c x > 2 * x + m) ↔ m ≤ -1) :=
sorry

end quadratic_function_properties_l3395_339561


namespace first_complete_column_coverage_l3395_339594

theorem first_complete_column_coverage : ∃ n : ℕ, 
  n = 32 ∧ 
  (∀ k ≤ n, ∃ m ≤ n, m * (m + 1) / 2 % 12 = k % 12) ∧
  (∀ j < n, ¬(∀ k ≤ 11, ∃ m ≤ j, m * (m + 1) / 2 % 12 = k % 12)) := by
  sorry

end first_complete_column_coverage_l3395_339594


namespace franklin_gathering_handshakes_l3395_339515

/-- Represents a gathering of married couples -/
structure Gathering where
  couples : Nat
  men : Nat
  women : Nat
  total_people : Nat

/-- Calculates the number of handshakes in the gathering -/
def handshakes (g : Gathering) : Nat :=
  let men_handshakes := g.men.choose 2
  let men_women_handshakes := g.men * (g.women - 1)
  men_handshakes + men_women_handshakes

theorem franklin_gathering_handshakes :
  ∀ g : Gathering,
    g.couples = 15 →
    g.men = g.couples →
    g.women = g.couples →
    g.total_people = g.men + g.women →
    handshakes g = 315 := by
  sorry

#eval handshakes { couples := 15, men := 15, women := 15, total_people := 30 }

end franklin_gathering_handshakes_l3395_339515


namespace park_area_l3395_339504

/-- The area of a rectangular park with perimeter 120 feet and length three times the width is 675 square feet. -/
theorem park_area (length width : ℝ) : 
  (2 * length + 2 * width = 120) →
  (length = 3 * width) →
  (length * width = 675) :=
by
  sorry

end park_area_l3395_339504


namespace consecutive_even_product_divisible_l3395_339522

theorem consecutive_even_product_divisible (n : ℕ) : 
  ∃ k : ℕ, (2*n) * (2*n + 2) * (2*n + 4) * (2*n + 6) = 240 * k := by
  sorry

end consecutive_even_product_divisible_l3395_339522


namespace zero_multiple_of_all_primes_l3395_339510

theorem zero_multiple_of_all_primes : ∃! x : ℤ, ∀ p : ℕ, Nat.Prime p → ∃ k : ℤ, x = k * p :=
sorry

end zero_multiple_of_all_primes_l3395_339510


namespace chess_team_arrangements_l3395_339592

def num_boys : ℕ := 3
def num_girls : ℕ := 2

def arrange_chess_team (boys : ℕ) (girls : ℕ) : ℕ :=
  (girls.factorial) * (boys.factorial)

theorem chess_team_arrangements :
  arrange_chess_team num_boys num_girls = 12 := by
  sorry

end chess_team_arrangements_l3395_339592


namespace multiple_of_seven_problem_l3395_339527

theorem multiple_of_seven_problem (start : Nat) (count : Nat) (result : Nat) : 
  start = 21 → count = 47 → result = 329 → 
  ∃ (n : Nat), result = start + 7 * (count - 1) ∧ result % 7 = 0 := by
  sorry

end multiple_of_seven_problem_l3395_339527


namespace trailingZerosOfSquareMinusFactorial_l3395_339537

-- Define the number we're working with
def n : ℕ := 999999

-- Define the factorial function
def factorial (m : ℕ) : ℕ :=
  match m with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

-- Define a function to count trailing zeros
def countTrailingZeros (x : ℕ) : ℕ :=
  if x = 0 then 0
  else if x % 10 = 0 then 1 + countTrailingZeros (x / 10)
  else 0

-- Theorem statement
theorem trailingZerosOfSquareMinusFactorial :
  countTrailingZeros (n^2 - factorial 6) = 0 := by
  sorry

end trailingZerosOfSquareMinusFactorial_l3395_339537


namespace sum_of_circle_areas_l3395_339587

/-- Represents a right triangle with mutually externally tangent circles at its vertices -/
structure TriangleWithCircles where
  /-- Side lengths of the right triangle -/
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  /-- Radii of the circles centered at the vertices -/
  r1 : ℝ
  r2 : ℝ
  r3 : ℝ
  /-- Conditions for the triangle and circles -/
  triangle_sides : side1^2 + side2^2 = hypotenuse^2
  circle_tangency1 : r1 + r2 = side1
  circle_tangency2 : r1 + r3 = side2
  circle_tangency3 : r2 + r3 = hypotenuse

/-- The sum of the areas of the three circles in a 6-8-10 right triangle with
    mutually externally tangent circles at its vertices is 56π -/
theorem sum_of_circle_areas (t : TriangleWithCircles)
    (h1 : t.side1 = 6)
    (h2 : t.side2 = 8)
    (h3 : t.hypotenuse = 10) :
  π * (t.r1^2 + t.r2^2 + t.r3^2) = 56 * π := by
  sorry

end sum_of_circle_areas_l3395_339587


namespace contrapositive_theorem_l3395_339505

theorem contrapositive_theorem (a b c : ℝ) :
  (abc = 0 → a = 0 ∨ b = 0 ∨ c = 0) ↔ (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 → abc ≠ 0) :=
by sorry

end contrapositive_theorem_l3395_339505


namespace min_abs_w_l3395_339544

theorem min_abs_w (w : ℂ) (h : Complex.abs (w - 2*I) + Complex.abs (w + 3) = 6) :
  ∃ (min_abs : ℝ), min_abs = 1 ∧ ∀ (z : ℂ), Complex.abs (w - 2*I) + Complex.abs (w + 3) = 6 → Complex.abs z ≥ min_abs :=
sorry

end min_abs_w_l3395_339544


namespace union_A_B_complement_B_intersect_A_C_subset_A_implies_a_leq_3_l3395_339502

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Theorem statements
theorem union_A_B : A ∪ B = {x | x > 2} := by sorry

theorem complement_B_intersect_A : (𝓤 \ B) ∩ A = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

theorem C_subset_A_implies_a_leq_3 (a : ℝ) : C a ⊆ A → a ≤ 3 := by sorry

end union_A_B_complement_B_intersect_A_C_subset_A_implies_a_leq_3_l3395_339502


namespace range_of_m_l3395_339599

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_domain : ∀ x, f x ≠ 0 → x ∈ Set.Icc (-2 : ℝ) 2
axiom f_decreasing : ∀ x y, x < y ∧ x ∈ Set.Icc (-2 : ℝ) 0 → f x > f y

-- Define the inequality condition
def inequality_condition (m : ℝ) : Prop := f (1 - m) + f (1 - m^2) < 0

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, inequality_condition m ↔ m ∈ Set.Icc (-1 : ℝ) 1 ∧ m ≠ 1 :=
sorry

end range_of_m_l3395_339599


namespace travel_time_calculation_l3395_339547

/-- Represents the travel times of a motorcyclist and cyclist meeting on a road --/
def TravelTimes (t m c : ℝ) : Prop :=
  t > 0 ∧ 
  m > t ∧ 
  c > t ∧
  m - t = 2 ∧ 
  c - t = 4.5

theorem travel_time_calculation (t m c : ℝ) (h : TravelTimes t m c) : 
  m = 5 ∧ c = 7.5 := by
  sorry

#check travel_time_calculation

end travel_time_calculation_l3395_339547


namespace archer_problem_l3395_339532

theorem archer_problem (n m : ℕ) : 
  (10 < n) → 
  (n < 20) → 
  (5 * m = 3 * (n - m)) → 
  (n = 16 ∧ m = 6) := by
sorry

end archer_problem_l3395_339532


namespace equality_check_l3395_339538

theorem equality_check : 
  (3^2 ≠ 2^3) ∧ 
  (-(3 * 2)^2 ≠ -3 * 2^2) ∧ 
  (-|2^3| ≠ |-2^3|) ∧ 
  (-2^3 = (-2)^3) := by
  sorry

end equality_check_l3395_339538


namespace horner_v4_value_l3395_339535

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_step (v : ℝ) (x : ℝ) (a : ℝ) : ℝ := v * x + a

theorem horner_v4_value :
  let x := -4
  let v0 := 3
  let v1 := horner_step v0 x 5
  let v2 := horner_step v1 x 6
  let v3 := horner_step v2 x 79
  let v4 := horner_step v3 x (-8)
  v4 = 220 :=
by sorry

end horner_v4_value_l3395_339535


namespace smallest_cube_ending_576_l3395_339524

theorem smallest_cube_ending_576 : 
  ∀ n : ℕ, n > 0 → n < 706 → n^3 % 1000 ≠ 576 ∧ 706^3 % 1000 = 576 := by
  sorry

end smallest_cube_ending_576_l3395_339524


namespace initial_solution_amount_l3395_339552

theorem initial_solution_amount 
  (x : ℝ) -- initial amount of solution in ml
  (h1 : x - 200 + 1000 = 2000) -- equation representing the process
  : x = 1200 :=
by sorry

end initial_solution_amount_l3395_339552


namespace highest_points_fewer_wins_l3395_339512

/-- Represents a football team in the tournament -/
structure Team :=
  (id : Nat)
  (wins : Nat)
  (draws : Nat)
  (losses : Nat)

/-- Calculates the points for a team based on their wins and draws -/
def points (t : Team) : Nat :=
  3 * t.wins + t.draws

/-- Represents the tournament results -/
structure TournamentResult :=
  (teams : Finset Team)
  (team_count : Nat)
  (hteam_count : teams.card = team_count)

/-- Theorem stating that it's possible for a team to have the highest points but fewer wins -/
theorem highest_points_fewer_wins (tr : TournamentResult) 
  (h_six_teams : tr.team_count = 6) : 
  ∃ (t1 t2 : Team), t1 ∈ tr.teams ∧ t2 ∈ tr.teams ∧ 
    (∀ t ∈ tr.teams, points t1 ≥ points t) ∧
    t1.wins < t2.wins :=
  sorry

end highest_points_fewer_wins_l3395_339512


namespace leo_marbles_l3395_339530

theorem leo_marbles (total_marbles : ℕ) (marbles_per_pack : ℕ) 
  (manny_fraction : ℚ) (neil_fraction : ℚ) :
  total_marbles = 400 →
  marbles_per_pack = 10 →
  manny_fraction = 1/4 →
  neil_fraction = 1/8 →
  (total_marbles / marbles_per_pack : ℚ) * (1 - manny_fraction - neil_fraction) = 25 := by
  sorry

end leo_marbles_l3395_339530


namespace election_votes_proof_l3395_339525

/-- The total number of votes in a school election where Emily received 45 votes, 
    which accounted for 25% of the total votes. -/
def total_votes : ℕ := 180

/-- Emily's votes in the election -/
def emily_votes : ℕ := 45

/-- The percentage of total votes that Emily received -/
def emily_percentage : ℚ := 25 / 100

theorem election_votes_proof : 
  total_votes = emily_votes / emily_percentage :=
by sorry

end election_votes_proof_l3395_339525


namespace power_two_equals_four_l3395_339585

theorem power_two_equals_four : 2^2 = 4 := by
  sorry

end power_two_equals_four_l3395_339585


namespace positive_real_inequality_l3395_339549

theorem positive_real_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end positive_real_inequality_l3395_339549


namespace robotics_club_age_problem_l3395_339558

theorem robotics_club_age_problem (total_members : ℕ) (girls : ℕ) (boys : ℕ) (adults : ℕ)
  (overall_avg : ℚ) (girls_avg : ℚ) (boys_avg : ℚ) :
  total_members = 30 →
  girls = 10 →
  boys = 10 →
  adults = 10 →
  overall_avg = 22 →
  girls_avg = 18 →
  boys_avg = 20 →
  (total_members * overall_avg - girls * girls_avg - boys * boys_avg) / adults = 28 := by
  sorry

end robotics_club_age_problem_l3395_339558


namespace office_age_problem_l3395_339553

theorem office_age_problem (total_persons : Nat) (group1_persons : Nat) (group2_persons : Nat)
  (total_avg_age : ℝ) (group1_avg_age : ℝ) (group2_avg_age : ℝ)
  (h1 : total_persons = 16)
  (h2 : group1_persons = 5)
  (h3 : group2_persons = 9)
  (h4 : total_avg_age = 15)
  (h5 : group1_avg_age = 14)
  (h6 : group2_avg_age = 16)
  (h7 : group1_persons + group2_persons + 2 = total_persons) :
  ∃ (person15_age : ℝ),
    person15_age = total_persons * total_avg_age -
      (group1_persons * group1_avg_age + group2_persons * group2_avg_age) ∧
    person15_age = 26 := by
  sorry

end office_age_problem_l3395_339553
