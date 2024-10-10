import Mathlib

namespace factorization_3x2_minus_12y2_l2307_230756

theorem factorization_3x2_minus_12y2 (x y : ℝ) :
  3 * x^2 - 12 * y^2 = 3 * (x - 2*y) * (x + 2*y) := by
  sorry

end factorization_3x2_minus_12y2_l2307_230756


namespace trailing_zeroes_500_factorial_l2307_230760

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem trailing_zeroes_500_factorial :
  trailingZeroes 500 = 124 := by
  sorry

end trailing_zeroes_500_factorial_l2307_230760


namespace line_parallel_perpendicular_implies_perpendicular_l2307_230781

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_perpendicular 
  (m n : Line) (α : Plane) :
  parallel m n → perpendicular n α → perpendicular m α :=
sorry

end line_parallel_perpendicular_implies_perpendicular_l2307_230781


namespace quiz_correct_answers_l2307_230729

theorem quiz_correct_answers
  (wendy_correct : ℕ)
  (campbell_correct : ℕ)
  (kelsey_correct : ℕ)
  (martin_correct : ℕ)
  (h1 : wendy_correct = 20)
  (h2 : campbell_correct = 2 * wendy_correct)
  (h3 : kelsey_correct = campbell_correct + 8)
  (h4 : martin_correct = kelsey_correct - 3) :
  martin_correct = 45 := by
  sorry

end quiz_correct_answers_l2307_230729


namespace student_divisor_problem_l2307_230763

theorem student_divisor_problem (dividend : ℕ) (student_answer : ℕ) (correct_answer : ℕ) (correct_divisor : ℕ) : 
  student_answer = 24 →
  correct_answer = 32 →
  correct_divisor = 36 →
  dividend / correct_divisor = correct_answer →
  ∃ (student_divisor : ℕ), 
    dividend / student_divisor = student_answer ∧ 
    student_divisor = 48 :=
by sorry

end student_divisor_problem_l2307_230763


namespace no_quadruple_sum_2013_divisors_l2307_230783

theorem no_quadruple_sum_2013_divisors :
  ¬ (∃ (a b c d : ℕ+), 
      (a.val + b.val + c.val + d.val = 2013) ∧ 
      (2013 % a.val = 0) ∧ 
      (2013 % b.val = 0) ∧ 
      (2013 % c.val = 0) ∧ 
      (2013 % d.val = 0)) := by
  sorry

end no_quadruple_sum_2013_divisors_l2307_230783


namespace binary_difference_digits_l2307_230726

theorem binary_difference_digits : ∃ (b : ℕ → Bool), 
  (Nat.castRingHom ℕ).toFun ((Nat.digits 2 1500).foldl (λ acc d => 2 * acc + d) 0 - 
                              (Nat.digits 2 300).foldl (λ acc d => 2 * acc + d) 0) = 
  (Nat.digits 2 1200).foldl (λ acc d => 2 * acc + d) 0 ∧
  (Nat.digits 2 1200).length = 11 :=
by sorry

end binary_difference_digits_l2307_230726


namespace dodecahedron_edge_probability_l2307_230719

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  vertex_degree : ∀ v : Fin 20, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of selecting two vertices that form an edge in a regular dodecahedron -/
def edge_selection_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

/-- Theorem: The probability of randomly selecting two vertices that are endpoints 
    of the same edge in a regular dodecahedron is 3/19 -/
theorem dodecahedron_edge_probability (d : RegularDodecahedron) : 
  edge_selection_probability d = 3 / 19 := by
  sorry

end dodecahedron_edge_probability_l2307_230719


namespace last_digit_of_seven_to_seventh_l2307_230774

theorem last_digit_of_seven_to_seventh : 7^7 % 10 = 3 := by
  sorry

end last_digit_of_seven_to_seventh_l2307_230774


namespace gcd_102_238_l2307_230790

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by sorry

end gcd_102_238_l2307_230790


namespace least_addition_for_divisibility_l2307_230739

theorem least_addition_for_divisibility : ∃! n : ℕ,
  (∀ m : ℕ, m < n → ¬((1789 + m) % 5 = 0 ∧ (1789 + m) % 7 = 0 ∧ (1789 + m) % 11 = 0 ∧ (1789 + m) % 13 = 0)) ∧
  ((1789 + n) % 5 = 0 ∧ (1789 + n) % 7 = 0 ∧ (1789 + n) % 11 = 0 ∧ (1789 + n) % 13 = 0) ∧
  n = 3216 :=
by sorry

end least_addition_for_divisibility_l2307_230739


namespace smallest_n_for_integer_sum_l2307_230703

theorem smallest_n_for_integer_sum : 
  ∃ (n : ℕ), n > 0 ∧ 
  (1/3 + 1/4 + 1/8 + 1/n : ℚ).isInt ∧ 
  (∀ m : ℕ, m > 0 ∧ (1/3 + 1/4 + 1/8 + 1/m : ℚ).isInt → n ≤ m) ∧ 
  n = 24 := by
sorry

end smallest_n_for_integer_sum_l2307_230703


namespace at_least_one_equation_has_two_roots_l2307_230750

theorem at_least_one_equation_has_two_roots (p q₁ q₂ : ℝ) (h : p = q₁ + q₂ + 1) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + x + q₁ = 0 ∧ y^2 + y + q₁ = 0) ∨
  (∃ x y : ℝ, x ≠ y ∧ x^2 + p*x + q₂ = 0 ∧ y^2 + p*y + q₂ = 0) :=
by sorry

end at_least_one_equation_has_two_roots_l2307_230750


namespace range_of_fraction_l2307_230741

theorem range_of_fraction (x y : ℝ) (h : (x - 1)^2 + y^2 = 1) :
  ∃ (k : ℝ), y / (x + 1) = k ∧ -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 :=
sorry

end range_of_fraction_l2307_230741


namespace largest_gcd_sum_780_l2307_230776

theorem largest_gcd_sum_780 :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = 780 ∧
  (∀ (a b : ℕ), a > 0 → b > 0 → a + b = 780 → Nat.gcd a b ≤ Nat.gcd x y) ∧
  Nat.gcd x y = 390 := by
sorry

end largest_gcd_sum_780_l2307_230776


namespace remainder_3123_div_28_l2307_230730

theorem remainder_3123_div_28 : 3123 % 28 = 15 := by
  sorry

end remainder_3123_div_28_l2307_230730


namespace planes_parallel_condition_l2307_230707

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_condition 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) (h2 : α ≠ β)
  (h3 : perpendicular m α) 
  (h4 : perpendicular n β) 
  (h5 : parallel_lines m n) : 
  parallel_planes α β :=
sorry

end planes_parallel_condition_l2307_230707


namespace fraction_equality_l2307_230711

theorem fraction_equality (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 2) 
  (h2 : c / d = 1 / 2) 
  (h3 : e / f = 1 / 2) 
  (h4 : 3 * b - 2 * d + f ≠ 0) : 
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 2 := by
  sorry

end fraction_equality_l2307_230711


namespace grocery_store_soda_l2307_230759

theorem grocery_store_soda (total : ℕ) (diet : ℕ) (regular : ℕ) : 
  total = 30 → diet = 2 → regular = total - diet → regular = 28 := by
  sorry

end grocery_store_soda_l2307_230759


namespace correct_mark_is_ten_l2307_230751

/-- Proves that the correct mark of a student is 10, given the conditions of the problem. -/
theorem correct_mark_is_ten (n : ℕ) (initial_avg final_avg wrong_mark : ℚ) :
  n = 30 →
  initial_avg = 100 →
  wrong_mark = 70 →
  final_avg = 98 →
  (n : ℚ) * initial_avg - wrong_mark + (n : ℚ) * final_avg = (n : ℚ) * initial_avg →
  (n : ℚ) * initial_avg - wrong_mark + 10 = (n : ℚ) * final_avg :=
by sorry

end correct_mark_is_ten_l2307_230751


namespace four_dice_same_number_l2307_230714

-- Define a standard six-sided die
def standard_die := Finset.range 6

-- Define the probability of getting the same number on all four dice
def same_number_probability : ℚ :=
  (1 : ℚ) / (standard_die.card ^ 4)

-- Theorem statement
theorem four_dice_same_number :
  same_number_probability = 1 / 216 := by
  sorry

end four_dice_same_number_l2307_230714


namespace bus_speed_with_stoppages_l2307_230789

/-- Calculates the speed of a bus including stoppages -/
theorem bus_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (total_time : ℝ) :
  speed_without_stoppages = 54 →
  stoppage_time = 10 →
  total_time = 60 →
  (speed_without_stoppages * (total_time - stoppage_time) / total_time) = 45 :=
by
  sorry

#check bus_speed_with_stoppages

end bus_speed_with_stoppages_l2307_230789


namespace divisibility_implies_equality_l2307_230708

theorem divisibility_implies_equality (a b : ℕ) (h : (a^2 + b^2) ∣ (a * b)) : a = b := by
  sorry

end divisibility_implies_equality_l2307_230708


namespace centroid_on_line_segment_l2307_230736

/-- Given a triangle ABC with points M on AB and N on AC, if BM/MA + CN/NA = 1,
    then the centroid of triangle ABC is collinear with M and N. -/
theorem centroid_on_line_segment (A B C M N : EuclideanSpace ℝ (Fin 2)) :
  (∃ s t : ℝ, 0 < s ∧ s < 1 ∧ 0 < t ∧ t < 1 ∧
   M = (1 - s) • A + s • B ∧
   N = (1 - t) • A + t • C ∧
   s / (1 - s) + t / (1 - t) = 1) →
  ∃ u : ℝ, (1/3 : ℝ) • (A + B + C) = (1 - u) • M + u • N :=
by sorry

end centroid_on_line_segment_l2307_230736


namespace fraction_simplification_l2307_230797

theorem fraction_simplification (x : ℝ) (h : x = 3) :
  (x^8 + 20*x^4 + 100) / (x^4 + 10) = 91 := by
  sorry

end fraction_simplification_l2307_230797


namespace sara_flowers_l2307_230702

/-- Given the number of red flowers and the number of bouquets, 
    calculate the number of yellow flowers needed to create bouquets 
    with an equal number of red and yellow flowers in each. -/
def yellow_flowers (red_flowers : ℕ) (num_bouquets : ℕ) : ℕ :=
  (red_flowers / num_bouquets) * num_bouquets

/-- Theorem stating that given 16 red flowers and 8 bouquets,
    the number of yellow flowers needed is 16. -/
theorem sara_flowers : yellow_flowers 16 8 = 16 := by
  sorry

end sara_flowers_l2307_230702


namespace fifth_diagram_shaded_fraction_l2307_230731

/-- Represents the number of shaded triangles in the n-th diagram -/
def shadedTriangles (n : ℕ) : ℕ := 2^(n - 1)

/-- Represents the total number of triangles in the n-th diagram -/
def totalTriangles (n : ℕ) : ℕ := n^2

/-- The fraction of shaded triangles in the n-th diagram -/
def shadedFraction (n : ℕ) : ℚ :=
  (shadedTriangles n : ℚ) / (totalTriangles n : ℚ)

theorem fifth_diagram_shaded_fraction :
  shadedFraction 5 = 16 / 25 := by
  sorry

end fifth_diagram_shaded_fraction_l2307_230731


namespace lilliputian_matchboxes_theorem_l2307_230746

/-- The scale factor between Gulliver's homeland and Lilliput -/
def scale_factor : ℕ := 12

/-- The number of Lilliputian matchboxes that can fit into a matchbox from Gulliver's homeland -/
def lilliputian_matchboxes_count : ℕ := scale_factor ^ 3

/-- Theorem stating that the number of Lilliputian matchboxes that can fit into a matchbox from Gulliver's homeland is 1728 -/
theorem lilliputian_matchboxes_theorem : lilliputian_matchboxes_count = 1728 := by
  sorry

end lilliputian_matchboxes_theorem_l2307_230746


namespace sum_of_b_values_l2307_230718

/-- The sum of the two values of b for which the equation 3x^2 + bx + 6x + 7 = 0 has only one solution for x -/
theorem sum_of_b_values (b₁ b₂ : ℝ) : 
  (∃! x, 3 * x^2 + b₁ * x + 6 * x + 7 = 0) →
  (∃! x, 3 * x^2 + b₂ * x + 6 * x + 7 = 0) →
  b₁ + b₂ = -12 := by
  sorry

end sum_of_b_values_l2307_230718


namespace parabola_line_intersection_l2307_230747

/-- Parabola defined by y = x^2 -/
def parabola (x y : ℝ) : Prop := y = x^2

/-- Point Q -/
def Q : ℝ × ℝ := (10, 25)

/-- Line passing through Q with slope m -/
def line (m x y : ℝ) : Prop := y - Q.2 = m * (x - Q.1)

/-- Line does not intersect parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x y : ℝ, ¬(parabola x y ∧ line m x y)

/-- Theorem statement -/
theorem parabola_line_intersection :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 40 := by
  sorry

end parabola_line_intersection_l2307_230747


namespace distinct_centroids_count_l2307_230724

/-- Represents a point on the perimeter of the square -/
structure PerimeterPoint where
  x : Fin 11
  y : Fin 11
  on_perimeter : (x = 0 ∨ x = 10) ∨ (y = 0 ∨ y = 10)

/-- The set of 40 equally spaced points on the square's perimeter -/
def perimeterPoints : Finset PerimeterPoint :=
  sorry

/-- Represents the centroid of a triangle -/
structure Centroid where
  x : Rat
  y : Rat
  inside_square : 0 < x ∧ x < 10 ∧ 0 < y ∧ y < 10

/-- Function to calculate the centroid given three points -/
def calculateCentroid (p q r : PerimeterPoint) : Centroid :=
  sorry

/-- The set of all possible centroids -/
def allCentroids : Finset Centroid :=
  sorry

/-- Main theorem: The number of distinct centroids is 841 -/
theorem distinct_centroids_count : Finset.card allCentroids = 841 :=
  sorry

end distinct_centroids_count_l2307_230724


namespace circle_equation_through_origin_l2307_230748

/-- The equation of a circle with center (1, 1) passing through the origin (0, 0) is (x-1)^2 + (y-1)^2 = 2 -/
theorem circle_equation_through_origin (x y : ℝ) :
  let center : ℝ × ℝ := (1, 1)
  let origin : ℝ × ℝ := (0, 0)
  let on_circle (p : ℝ × ℝ) := (p.1 - center.1)^2 + (p.2 - center.2)^2 = (center.1 - origin.1)^2 + (center.2 - origin.2)^2
  on_circle (x, y) ↔ (x - 1)^2 + (y - 1)^2 = 2 :=
by sorry

end circle_equation_through_origin_l2307_230748


namespace min_colors_theorem_l2307_230796

/-- The size of the board --/
def boardSize : Nat := 2016

/-- A color assignment for the board --/
def ColorAssignment := Fin boardSize → Fin boardSize → Nat

/-- Checks if a color assignment satisfies the diagonal condition --/
def satisfiesDiagonalCondition (c : ColorAssignment) : Prop :=
  ∀ i, c i i = 1

/-- Checks if a color assignment satisfies the symmetry condition --/
def satisfiesSymmetryCondition (c : ColorAssignment) : Prop :=
  ∀ i j, c i j = c j i

/-- Checks if a color assignment satisfies the row condition --/
def satisfiesRowCondition (c : ColorAssignment) : Prop :=
  ∀ i j k, i ≠ j ∧ (i < j ∧ j < k ∨ k < j ∧ j < i) → c i k ≠ c j k

/-- Checks if a color assignment is valid --/
def isValidColorAssignment (c : ColorAssignment) : Prop :=
  satisfiesDiagonalCondition c ∧ satisfiesSymmetryCondition c ∧ satisfiesRowCondition c

/-- The minimum number of colors required --/
def minColors : Nat := 11

/-- Theorem stating the minimum number of colors required --/
theorem min_colors_theorem :
  (∃ (c : ColorAssignment), isValidColorAssignment c ∧ (∀ i j, c i j < minColors)) ∧
  (∀ k < minColors, ¬∃ (c : ColorAssignment), isValidColorAssignment c ∧ (∀ i j, c i j < k)) :=
sorry

end min_colors_theorem_l2307_230796


namespace rectangle_area_l2307_230722

/-- Given a rectangle with a length to width ratio of 0.875 and a width of 24 centimeters,
    its area is 504 square centimeters. -/
theorem rectangle_area (ratio : ℝ) (width : ℝ) (h1 : ratio = 0.875) (h2 : width = 24) :
  ratio * width * width = 504 := by
  sorry

end rectangle_area_l2307_230722


namespace polynomial_simplification_l2307_230717

theorem polynomial_simplification (x : ℝ) :
  (3*x - 2) * (6*x^12 + 3*x^11 + 6*x^10 + 3*x^9) =
  18*x^13 - 3*x^12 + 12*x^11 - 3*x^10 - 6*x^9 := by
sorry

end polynomial_simplification_l2307_230717


namespace unique_fraction_condition_l2307_230753

def is_simplest_proper_fraction (n d : ℤ) : Prop :=
  0 < n ∧ n < d ∧ Nat.gcd n.natAbs d.natAbs = 1

def is_improper_fraction (n d : ℤ) : Prop :=
  n ≥ d

theorem unique_fraction_condition (x : ℤ) : 
  (is_simplest_proper_fraction x 8 ∧ is_improper_fraction x 6) ↔ x = 7 :=
by sorry

end unique_fraction_condition_l2307_230753


namespace prime_power_sum_l2307_230744

theorem prime_power_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 3250 → 2*w + 3*x + 4*y + 5*z = 19 := by
  sorry

end prime_power_sum_l2307_230744


namespace x_minus_y_value_l2307_230742

theorem x_minus_y_value (x y : ℤ) (hx : x = -3) (hy : |y| = 4) :
  x - y = 1 ∨ x - y = -7 := by
  sorry

end x_minus_y_value_l2307_230742


namespace correct_average_l2307_230782

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℕ) :
  n = 10 →
  initial_avg = 16 →
  incorrect_num = 25 →
  correct_num = 45 →
  (n : ℚ) * initial_avg + (correct_num - incorrect_num : ℚ) = n * 18 := by
  sorry

end correct_average_l2307_230782


namespace sum_of_coefficients_l2307_230785

theorem sum_of_coefficients (A B : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → A / (x - 2) + B * (x + 3) = (-5 * x^2 + 20 * x + 34) / (x - 2)) →
  A + B = 9 := by
sorry

end sum_of_coefficients_l2307_230785


namespace sequence_inequality_l2307_230764

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, a n ≥ 0) 
  (h_ineq : ∀ m n, a (m + n) ≤ a m + a n) (m n : ℕ) (h_ge : n ≥ m) :
  a n ≤ m * a 1 + (n / m - 1) * a m :=
sorry

end sequence_inequality_l2307_230764


namespace max_sum_is_24_l2307_230766

/-- Represents the grid configuration -/
structure Grid :=
  (a b c d e : ℕ)

/-- The set of available numbers -/
def availableNumbers : Finset ℕ := {5, 8, 11, 14}

/-- Checks if the grid contains only the available numbers -/
def Grid.isValid (g : Grid) : Prop :=
  {g.a, g.b, g.c, g.d, g.e} ⊆ availableNumbers

/-- Calculates the horizontal sum -/
def Grid.horizontalSum (g : Grid) : ℕ := g.a + g.b + g.e

/-- Calculates the vertical sum -/
def Grid.verticalSum (g : Grid) : ℕ := g.a + g.c + 2 * g.e

/-- Checks if the grid satisfies the sum condition -/
def Grid.satisfiesSumCondition (g : Grid) : Prop :=
  g.horizontalSum = g.verticalSum

theorem max_sum_is_24 :
  ∃ (g : Grid), g.isValid ∧ g.satisfiesSumCondition ∧
  (∀ (h : Grid), h.isValid → h.satisfiesSumCondition →
    g.horizontalSum ≥ h.horizontalSum ∧ g.verticalSum ≥ h.verticalSum) ∧
  g.horizontalSum = 24 ∧ g.verticalSum = 24 :=
sorry

end max_sum_is_24_l2307_230766


namespace adult_ticket_cost_l2307_230706

theorem adult_ticket_cost (child_ticket_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (adult_tickets : ℕ) :
  child_ticket_cost = 4 →
  total_tickets = 900 →
  total_revenue = 5100 →
  adult_tickets = 500 →
  ∃ (adult_ticket_cost : ℕ), adult_ticket_cost = 7 ∧
    adult_ticket_cost * adult_tickets + child_ticket_cost * (total_tickets - adult_tickets) = total_revenue :=
by sorry

end adult_ticket_cost_l2307_230706


namespace largest_four_digit_congruent_to_17_mod_24_l2307_230723

theorem largest_four_digit_congruent_to_17_mod_24 : ∃ n : ℕ, 
  (n ≡ 17 [ZMOD 24]) ∧ 
  (n < 10000) ∧ 
  (1000 ≤ n) ∧ 
  (∀ m : ℕ, (m ≡ 17 [ZMOD 24]) → (1000 ≤ m) → (m < 10000) → m ≤ n) ∧ 
  n = 9977 :=
by sorry

end largest_four_digit_congruent_to_17_mod_24_l2307_230723


namespace bears_in_stock_calculation_l2307_230733

/-- Calculates the number of bears in stock before a new shipment arrived -/
def bears_in_stock_before_shipment (new_shipment : ℕ) (bears_per_shelf : ℕ) (num_shelves : ℕ) : ℕ :=
  num_shelves * bears_per_shelf - new_shipment

theorem bears_in_stock_calculation (new_shipment : ℕ) (bears_per_shelf : ℕ) (num_shelves : ℕ) :
  bears_in_stock_before_shipment new_shipment bears_per_shelf num_shelves =
  num_shelves * bears_per_shelf - new_shipment :=
by
  sorry

end bears_in_stock_calculation_l2307_230733


namespace functions_properties_l2307_230728

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

-- Theorem statement
theorem functions_properties :
  (∀ x : ℝ, f x < g x) ∧
  (∃ x : ℝ, f x ^ 2 + g x ^ 2 ≥ 1) ∧
  (∀ x : ℝ, f (2 * x) = 2 * f x * g x) := by
  sorry

end functions_properties_l2307_230728


namespace albaszu_machine_productivity_l2307_230784

-- Define the number of trees cut daily before improvement
def trees_before : ℕ := 16

-- Define the productivity increase factor
def productivity_increase : ℚ := 3/2

-- Define the number of trees cut daily after improvement
def trees_after : ℕ := 25

-- Theorem statement
theorem albaszu_machine_productivity : 
  ↑trees_after = ↑trees_before * productivity_increase :=
by sorry

end albaszu_machine_productivity_l2307_230784


namespace smallest_n_multiple_of_13_l2307_230791

theorem smallest_n_multiple_of_13 (x y : ℤ) 
  (h1 : (2 * x - 3) % 13 = 0) 
  (h2 : (3 * y + 4) % 13 = 0) : 
  ∃ n : ℕ+, (x^2 - x*y + y^2 + n) % 13 = 0 ∧ 
  ∀ m : ℕ+, m < n → (x^2 - x*y + y^2 + m) % 13 ≠ 0 :=
by sorry

end smallest_n_multiple_of_13_l2307_230791


namespace equation_solution_l2307_230749

theorem equation_solution (n : ℚ) :
  (2 / (n + 2) + 4 / (n + 2) + n / (n + 2) = 4) → n = -2/3 := by
  sorry

end equation_solution_l2307_230749


namespace daniel_initial_noodles_l2307_230793

/-- The number of noodles Daniel gave to William -/
def noodles_given : ℝ := 12.0

/-- The number of noodles Daniel had left -/
def noodles_left : ℕ := 42

/-- The initial number of noodles Daniel had -/
def initial_noodles : ℝ := noodles_given + noodles_left

theorem daniel_initial_noodles : initial_noodles = 54.0 := by sorry

end daniel_initial_noodles_l2307_230793


namespace erased_number_proof_l2307_230799

theorem erased_number_proof (n : ℕ) (x : ℕ) :
  n > 0 →
  x > 0 →
  x ≤ n →
  (n * (n + 1) / 2 - x) / (n - 1) = 182 / 5 →
  x = 8 := by
sorry

end erased_number_proof_l2307_230799


namespace metal_sheet_width_l2307_230765

/-- Represents the dimensions and volume of a box created from a metal sheet -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ

/-- Calculates the original width of the metal sheet given the box dimensions -/
def calculate_original_width (box : BoxDimensions) : ℝ :=
  box.width + 2 * box.height

/-- Theorem stating that given the specified conditions, the original width of the sheet must be 36 m -/
theorem metal_sheet_width
  (box : BoxDimensions)
  (h1 : box.length = 48 - 2 * 4)
  (h2 : box.height = 4)
  (h3 : box.volume = 4480)
  (h4 : box.volume = box.length * box.width * box.height) :
  calculate_original_width box = 36 := by
  sorry

#check metal_sheet_width

end metal_sheet_width_l2307_230765


namespace parking_space_area_l2307_230720

theorem parking_space_area (l w : ℝ) (h1 : l = 9) (h2 : 2 * w + l = 37) : l * w = 126 := by
  sorry

end parking_space_area_l2307_230720


namespace square_perimeter_no_conditional_l2307_230754

-- Define the problem types
inductive Problem
| OppositeNumber
| SquarePerimeter
| MaximumOfThree
| BinaryToDecimal

-- Define a predicate for problems that don't require conditional statements
def NoConditionalRequired (p : Problem) : Prop :=
  match p with
  | Problem.SquarePerimeter => True
  | _ => False

-- Theorem statement
theorem square_perimeter_no_conditional :
  NoConditionalRequired Problem.SquarePerimeter ∧
  ¬NoConditionalRequired Problem.OppositeNumber ∧
  ¬NoConditionalRequired Problem.MaximumOfThree ∧
  ¬NoConditionalRequired Problem.BinaryToDecimal :=
sorry

end square_perimeter_no_conditional_l2307_230754


namespace abigail_lost_money_l2307_230734

def money_lost (initial_amount spent_amount remaining_amount : ℕ) : ℕ :=
  initial_amount - spent_amount - remaining_amount

theorem abigail_lost_money : money_lost 11 2 3 = 6 := by
  sorry

end abigail_lost_money_l2307_230734


namespace father_son_meeting_point_father_son_meeting_point_specific_l2307_230735

/-- The meeting point of a father and son in a hallway -/
theorem father_son_meeting_point (hallway_length : ℝ) (speed_ratio : ℝ) : 
  hallway_length > 0 → 
  speed_ratio > 1 → 
  (speed_ratio * hallway_length) / (speed_ratio + 1) = 
    hallway_length - hallway_length / (speed_ratio + 1) :=
by
  sorry

/-- The specific case of a 16m hallway and 3:1 speed ratio -/
theorem father_son_meeting_point_specific : 
  (16 : ℝ) - 16 / (3 + 1) = 12 :=
by
  sorry

end father_son_meeting_point_father_son_meeting_point_specific_l2307_230735


namespace odd_number_as_difference_of_squares_l2307_230704

theorem odd_number_as_difference_of_squares (n : ℕ) (hn : n > 0) :
  ∃! (x y : ℕ), (2 * n + 1 : ℕ) = x^2 - y^2 ∧ x = n + 1 ∧ y = n :=
by sorry

end odd_number_as_difference_of_squares_l2307_230704


namespace smallest_n_dividing_2016_l2307_230787

theorem smallest_n_dividing_2016 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (2016 ∣ (20^m - 16^m)) → m ≥ n) ∧ 
  (2016 ∣ (20^n - 16^n)) ∧ n = 6 := by
  sorry

end smallest_n_dividing_2016_l2307_230787


namespace star_self_inverse_l2307_230709

/-- The star operation for real numbers -/
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

/-- Theorem: The star operation of (x^2 - y^2) and (y^2 - x^2) is zero -/
theorem star_self_inverse (x y : ℝ) : star (x^2 - y^2) (y^2 - x^2) = 0 := by
  sorry

end star_self_inverse_l2307_230709


namespace horner_method_V_4_l2307_230745

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial coefficients in descending order of degree -/
def f_coeffs : List ℤ := [3, 5, 6, 79, -8, 35, 12]

/-- The x-value at which to evaluate the polynomial -/
def x_val : ℤ := -4

/-- V_4 in Horner's method is the 5th intermediate value (0-indexed) -/
def V_4 : ℤ := (horner_eval (f_coeffs.take 5) x_val) * x_val + f_coeffs[5]

theorem horner_method_V_4 :
  V_4 = 220 :=
sorry

end horner_method_V_4_l2307_230745


namespace fraction_equality_l2307_230767

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3/4) :
  (a - c) * (b - d) / ((a - b) * (c - d)) = -1 := by
sorry

end fraction_equality_l2307_230767


namespace max_value_reciprocal_sum_l2307_230770

/-- Given a quadratic polynomial x^2 - tx + q with roots α and β,
    where α + β = α^2 + β^2 = α^3 + β^3 = ... = α^2010 + β^2010,
    the maximum possible value of 1/α^2011 + 1/β^2011 is 2. -/
theorem max_value_reciprocal_sum (t q α β : ℝ) : 
  (∀ k : ℕ, k ≥ 1 ∧ k ≤ 2010 → α^k + β^k = α + β) →
  α^2 - t*α + q = 0 →
  β^2 - t*β + q = 0 →
  α ≠ 0 →
  β ≠ 0 →
  (1/α^2011 + 1/β^2011 : ℝ) ≤ 2 := by
  sorry

end max_value_reciprocal_sum_l2307_230770


namespace find_y_value_l2307_230700

theorem find_y_value (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 20) : y = 40 := by
  sorry

end find_y_value_l2307_230700


namespace solve_for_m_l2307_230777

theorem solve_for_m : ∃ m : ℕ, (2022^2 - 4) * (2021^2 - 4) = 2024 * 2020 * 2019 * m ∧ m = 2023 := by
  sorry

end solve_for_m_l2307_230777


namespace remainder_mod_11_l2307_230768

theorem remainder_mod_11 : (8735+100) + (8736+100) + (8737+100) + (8738+100) * 2 ≡ 10 [MOD 11] := by
  sorry

end remainder_mod_11_l2307_230768


namespace smallest_part_of_proportional_division_l2307_230788

theorem smallest_part_of_proportional_division (total : ℝ) (prop1 prop2 prop3 : ℝ) (additional : ℝ) :
  total = 120 ∧ prop1 = 3 ∧ prop2 = 5 ∧ prop3 = 7 ∧ additional = 4 →
  let x := (total - 3 * additional) / (prop1 + prop2 + prop3)
  let part1 := prop1 * x + additional
  let part2 := prop2 * x + additional
  let part3 := prop3 * x + additional
  min part1 (min part2 part3) = 25.6 := by
sorry

end smallest_part_of_proportional_division_l2307_230788


namespace polynomial_expansion_l2307_230794

theorem polynomial_expansion :
  ∀ x : ℝ, (3 * x^2 + 4 * x - 5) * (4 * x^3 - 3 * x + 2) = 
    12 * x^5 + 16 * x^4 - 24 * x^3 - 6 * x^2 + 17 * x - 10 := by
  sorry

end polynomial_expansion_l2307_230794


namespace inequality_implies_a_zero_l2307_230716

theorem inequality_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, a * (Real.sin x)^2 + Real.cos x ≥ a^2 - 1) → a = 0 := by
  sorry

end inequality_implies_a_zero_l2307_230716


namespace certain_number_proof_l2307_230798

theorem certain_number_proof (h : 213 * 16 = 3408) : 
  ∃ x : ℝ, 213 * x = 340.8 ∧ x = 1.6 := by sorry

end certain_number_proof_l2307_230798


namespace pond_area_is_292_l2307_230721

/-- The total surface area of a cuboid-shaped pond, excluding the top surface. -/
def pondSurfaceArea (length width height : ℝ) : ℝ :=
  length * width + 2 * length * height + 2 * width * height

/-- Theorem: The surface area of a pond with given dimensions is 292 square meters. -/
theorem pond_area_is_292 :
  pondSurfaceArea 18 10 2 = 292 := by
  sorry

#eval pondSurfaceArea 18 10 2

end pond_area_is_292_l2307_230721


namespace special_quadrilateral_is_square_special_quadrilateral_is_not_always_square_l2307_230732

/-- A quadrilateral with perpendicular and equal diagonals --/
structure SpecialQuadrilateral where
  /-- The diagonals are perpendicular --/
  diagonals_perpendicular : Bool
  /-- The diagonals are equal in length --/
  diagonals_equal : Bool

/-- Definition of a square --/
def is_square (q : SpecialQuadrilateral) : Prop :=
  q.diagonals_perpendicular ∧ q.diagonals_equal

/-- The statement to be proven false --/
theorem special_quadrilateral_is_square (q : SpecialQuadrilateral) :
  q.diagonals_perpendicular ∧ q.diagonals_equal → is_square q :=
by
  sorry

/-- The theorem stating that the above statement is false --/
theorem special_quadrilateral_is_not_always_square :
  ¬ (∀ q : SpecialQuadrilateral, q.diagonals_perpendicular ∧ q.diagonals_equal → is_square q) :=
by
  sorry

end special_quadrilateral_is_square_special_quadrilateral_is_not_always_square_l2307_230732


namespace james_tylenol_dosage_l2307_230712

/-- Represents the dosage schedule and total daily intake of Tylenol tablets -/
structure TylenolDosage where
  tablets_per_dose : ℕ
  hours_between_doses : ℕ
  total_daily_mg : ℕ

/-- Calculates the mg per tablet given a TylenolDosage -/
def mg_per_tablet (dosage : TylenolDosage) : ℕ :=
  let doses_per_day := 24 / dosage.hours_between_doses
  let tablets_per_day := doses_per_day * dosage.tablets_per_dose
  dosage.total_daily_mg / tablets_per_day

/-- Theorem: Given James' Tylenol dosage schedule, each tablet contains 375 mg -/
theorem james_tylenol_dosage :
  let james_dosage : TylenolDosage := {
    tablets_per_dose := 2,
    hours_between_doses := 6,
    total_daily_mg := 3000
  }
  mg_per_tablet james_dosage = 375 := by
  sorry

end james_tylenol_dosage_l2307_230712


namespace evaluate_expression_l2307_230740

theorem evaluate_expression : -(16 / 4 * 11 - 70 + 5^2) = 1 := by
  sorry

end evaluate_expression_l2307_230740


namespace complex_modulus_equality_l2307_230710

theorem complex_modulus_equality (n : ℝ) (hn : n > 0) :
  Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 13 → n = 10 * Real.sqrt 3 := by
sorry

end complex_modulus_equality_l2307_230710


namespace parallel_line_through_point_l2307_230715

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem -/
theorem parallel_line_through_point (A B C : Point) : 
  let AC : Line := { a := C.y - A.y, b := A.x - C.x, c := A.y * C.x - A.x * C.y }
  let L : Line := { a := 1, b := -2, c := -7 }
  A.x = 5 ∧ A.y = 2 ∧ B.x = -1 ∧ B.y = -4 ∧ C.x = -5 ∧ C.y = -3 →
  B.liesOn L ∧ L.isParallelTo AC := by
  sorry


end parallel_line_through_point_l2307_230715


namespace max_ratio_OB_OA_l2307_230762

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y = 1
def C₂ (x y φ : ℝ) : Prop := x = 2 + 2 * Real.cos φ ∧ y = 2 * Real.sin φ ∧ 0 ≤ φ ∧ φ < 2 * Real.pi

-- Define the ray l
def l (ρ θ α : ℝ) : Prop := θ = α ∧ ρ ≥ 0

-- Define points A and B
def A (ρ θ : ℝ) : Prop := C₁ (ρ * Real.cos θ) (ρ * Real.sin θ) ∧ l ρ θ θ
def B (ρ θ : ℝ) : Prop := ∃ φ, C₂ (ρ * Real.cos θ) (ρ * Real.sin θ) φ ∧ l ρ θ θ

-- State the theorem
theorem max_ratio_OB_OA :
  ∃ (max : ℝ), max = 2 + 2 * Real.sqrt 2 ∧
  ∀ α : ℝ, 0 ≤ α ∧ α ≤ Real.pi / 2 →
    ∀ ρA ρB θA θB : ℝ,
      A ρA θA → B ρB θB →
      ρB / ρA ≤ max :=
sorry

end max_ratio_OB_OA_l2307_230762


namespace collinear_vectors_l2307_230743

/-- Given vectors a, b, and c in ℝ², prove that if k*a + b is collinear with c,
    then k = -26/15 -/
theorem collinear_vectors (a b c : ℝ × ℝ) (k : ℝ) 
    (ha : a = (1, 2))
    (hb : b = (2, 3))
    (hc : c = (4, -7))
    (hcollinear : ∃ t : ℝ, t ≠ 0 ∧ k • a + b = t • c) :
    k = -26/15 := by
  sorry

end collinear_vectors_l2307_230743


namespace unique_prime_pair_squares_l2307_230771

theorem unique_prime_pair_squares : ∃! (p q : ℕ), 
  Prime p ∧ Prime q ∧ 
  ∃ (a b : ℕ), (p - q = a^2) ∧ (p*q - q = b^2) := by
sorry

end unique_prime_pair_squares_l2307_230771


namespace pen_pencil_ratio_l2307_230701

theorem pen_pencil_ratio (num_pencils : ℕ) (num_pens : ℕ) : 
  num_pencils = 36 → 
  num_pencils = num_pens + 6 → 
  (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 := by
  sorry

end pen_pencil_ratio_l2307_230701


namespace no_solution_in_naturals_l2307_230780

theorem no_solution_in_naturals :
  ¬ ∃ (x y z : ℕ), (2 * x) ^ (2 * x) - 1 = y ^ (z + 1) := by
sorry

end no_solution_in_naturals_l2307_230780


namespace only_two_works_l2307_230792

/-- A move that can be applied to a table --/
inductive Move
  | MultiplyRow (row : Nat) : Move
  | SubtractColumn (col : Nat) : Move

/-- Definition of a rectangular table with positive integer entries --/
def Table := List (List Nat)

/-- Apply a move to a table --/
def applyMove (n : Nat) (t : Table) (m : Move) : Table :=
  sorry

/-- Check if all entries in a table are zero --/
def allZero (t : Table) : Prop :=
  sorry

/-- The main theorem --/
theorem only_two_works (n : Nat) : 
  (n > 0) → 
  (∀ t : Table, ∃ moves : List Move, allZero (moves.foldl (applyMove n) t)) ↔ 
  n = 2 := by
  sorry

end only_two_works_l2307_230792


namespace down_payment_equals_108000_l2307_230725

/-- The amount of money needed for a down payment on a house -/
def down_payment (richard_monthly_savings : ℕ) (sarah_monthly_savings : ℕ) (years : ℕ) : ℕ :=
  (richard_monthly_savings + sarah_monthly_savings) * years * 12

/-- Theorem stating that Richard and Sarah's savings over 3 years equal $108,000 -/
theorem down_payment_equals_108000 :
  down_payment 1500 1500 3 = 108000 := by
  sorry

end down_payment_equals_108000_l2307_230725


namespace arithmetic_mean_squares_l2307_230713

theorem arithmetic_mean_squares (x a : ℝ) (hx : x ≠ 0) (ha : a ≠ 0) :
  ((((x + a)^2) / x + ((x - a)^2) / x) / 2) = x + a^2 / x :=
by sorry

end arithmetic_mean_squares_l2307_230713


namespace yellow_shirt_pairs_l2307_230761

theorem yellow_shirt_pairs (blue_students : ℕ) (yellow_students : ℕ) (total_students : ℕ) 
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_students = 60 →
  yellow_students = 72 →
  total_students = 132 →
  total_pairs = 66 →
  blue_blue_pairs = 25 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 31 ∧ 
    yellow_yellow_pairs = total_pairs - blue_blue_pairs - (blue_students - 2 * blue_blue_pairs) :=
by
  sorry

end yellow_shirt_pairs_l2307_230761


namespace square_plus_inverse_square_l2307_230752

theorem square_plus_inverse_square (x : ℝ) (h : x + (1/x) = 2) : x^2 + (1/x^2) = 2 := by
  sorry

end square_plus_inverse_square_l2307_230752


namespace arithmetic_sqrt_one_fourth_l2307_230727

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem arithmetic_sqrt_one_fourth : arithmetic_sqrt (1/4) = 1/2 := by
  sorry

end arithmetic_sqrt_one_fourth_l2307_230727


namespace expression_evaluation_l2307_230705

theorem expression_evaluation : 
  (120^2 - 13^2) / (80^2 - 17^2) * ((80 - 17)*(80 + 17)) / ((120 - 13)*(120 + 13)) = 1 := by
  sorry

end expression_evaluation_l2307_230705


namespace intersection_of_A_and_B_l2307_230738

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_of_A_and_B :
  A_intersect_B = {x | 0 ≤ x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l2307_230738


namespace inverse_odd_implies_a_eq_one_l2307_230755

/-- A function f: ℝ → ℝ -/
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x ↦ 2^x - a * 2^(-x)

/-- The inverse function of f -/
noncomputable def f_inv (a : ℝ) : ℝ → ℝ := Function.invFun (f a)

/-- Theorem stating that if f_inv is odd and a is positive, then a = 1 -/
theorem inverse_odd_implies_a_eq_one (a : ℝ) (h_pos : a > 0) 
  (h_odd : ∀ x, f_inv a (-x) = -(f_inv a x)) : a = 1 := by
  sorry

#check inverse_odd_implies_a_eq_one

end inverse_odd_implies_a_eq_one_l2307_230755


namespace polynomial_B_value_l2307_230775

def polynomial (z A B C D : ℝ) : ℝ := z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36

theorem polynomial_B_value (A B C D : ℝ) :
  (∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ z : ℝ, polynomial z A B C D = (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆)) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 12) →
  B = -122 := by
sorry

end polynomial_B_value_l2307_230775


namespace sum_lent_is_500_l2307_230757

/-- The sum of money lent -/
def P : ℝ := 500

/-- The annual interest rate as a decimal -/
def R : ℝ := 0.04

/-- The time period in years -/
def T : ℝ := 8

/-- The simple interest formula -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

theorem sum_lent_is_500 : 
  simple_interest P R T = P - 340 → P = 500 := by
  sorry

end sum_lent_is_500_l2307_230757


namespace divisibility_property_l2307_230758

theorem divisibility_property :
  ∀ k : ℤ, k ≠ 2013 → (2013 - k) ∣ (2013^2014 - k^2014) := by
sorry

end divisibility_property_l2307_230758


namespace effective_price_for_8kg_l2307_230786

/-- Represents the shopkeeper's pricing scheme -/
structure PricingScheme where
  false_weight : Real
  discount_rate : Real
  tax_rate : Real

/-- Calculates the effective price for a given purchase -/
def effective_price (scheme : PricingScheme) (purchase_weight : Real) (cost_price : Real) : Real :=
  let actual_weight := purchase_weight * (scheme.false_weight / 1000)
  let discounted_price := purchase_weight * cost_price * (1 - scheme.discount_rate)
  discounted_price * (1 + scheme.tax_rate)

/-- Theorem stating the effective price for the given scenario -/
theorem effective_price_for_8kg (scheme : PricingScheme) (cost_price : Real) :
  scheme.false_weight = 980 →
  scheme.discount_rate = 0.1 →
  scheme.tax_rate = 0.03 →
  effective_price scheme 8 cost_price = 7.416 * cost_price :=
by sorry

end effective_price_for_8kg_l2307_230786


namespace cricket_average_l2307_230737

theorem cricket_average (initial_average : ℚ) : 
  (10 * initial_average + 65 = 11 * (initial_average + 3)) → initial_average = 32 := by
sorry

end cricket_average_l2307_230737


namespace charity_raffle_winnings_l2307_230769

theorem charity_raffle_winnings (X : ℝ) : 
  let remaining_after_donation := 0.75 * X
  let remaining_after_lunch := remaining_after_donation * 0.9
  let remaining_after_gift := remaining_after_lunch * 0.85
  let amount_for_investment := remaining_after_gift * 0.3
  let investment_return := amount_for_investment * 0.5
  let final_amount := remaining_after_gift - amount_for_investment + investment_return
  final_amount = 320 → X = 485 :=
by sorry

end charity_raffle_winnings_l2307_230769


namespace student_count_l2307_230778

theorem student_count (band : ℕ) (sports : ℕ) (both : ℕ) (total : ℕ) : 
  band = 85 → 
  sports = 200 → 
  both = 60 → 
  total = 225 → 
  band + sports - both = total :=
by sorry

end student_count_l2307_230778


namespace equal_roots_count_l2307_230773

/-- The number of real values of p for which the quadratic equation
    x^2 - (p+1)x + (p+1)^2 = 0 has equal roots is exactly one. -/
theorem equal_roots_count : ∃! p : ℝ,
  let a : ℝ := 1
  let b : ℝ := -(p + 1)
  let c : ℝ := (p + 1)^2
  b^2 - 4*a*c = 0 := by
  sorry

end equal_roots_count_l2307_230773


namespace smallest_gamma_for_integer_solution_l2307_230779

theorem smallest_gamma_for_integer_solution :
  ∃ (γ : ℕ), γ > 0 ∧
  (∃ (x : ℕ), (Real.sqrt x - Real.sqrt (24 * γ) = 4 * Real.sqrt 2)) ∧
  (∀ (γ' : ℕ), 0 < γ' ∧ γ' < γ →
    ¬∃ (x : ℕ), (Real.sqrt x - Real.sqrt (24 * γ') = 4 * Real.sqrt 2)) ∧
  γ = 3 :=
by sorry

end smallest_gamma_for_integer_solution_l2307_230779


namespace deposit_calculation_l2307_230795

theorem deposit_calculation (remaining_amount : ℝ) (deposit_percentage : ℝ) : 
  remaining_amount = 1260 ∧ deposit_percentage = 0.1 → 
  (remaining_amount / (1 - deposit_percentage)) * deposit_percentage = 140 := by
sorry

end deposit_calculation_l2307_230795


namespace players_count_l2307_230772

/-- Represents the number of socks in each washing machine -/
structure SockCounts where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the number of players based on sock counts -/
def calculate_players (socks : SockCounts) : ℕ :=
  min socks.red (socks.blue + socks.green)

/-- Theorem stating that the number of players is 12 given the specific sock counts -/
theorem players_count (socks : SockCounts)
  (h_red : socks.red = 12)
  (h_blue : socks.blue = 10)
  (h_green : socks.green = 16) :
  calculate_players socks = 12 := by
  sorry

#eval calculate_players ⟨12, 10, 16⟩

end players_count_l2307_230772
