import Mathlib

namespace fraction_cube_three_fourths_cubed_l175_17549

theorem fraction_cube (a b : ℚ) : (a / b) ^ 3 = a ^ 3 / b ^ 3 := by sorry

theorem three_fourths_cubed : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by sorry

end fraction_cube_three_fourths_cubed_l175_17549


namespace handshake_arrangements_mod_1000_l175_17530

/-- The number of ways 10 people can shake hands, where each person shakes hands with exactly two others -/
def handshake_arrangements : ℕ := sorry

/-- Theorem stating that the number of handshake arrangements is congruent to 688 modulo 1000 -/
theorem handshake_arrangements_mod_1000 : 
  handshake_arrangements ≡ 688 [ZMOD 1000] := by sorry

end handshake_arrangements_mod_1000_l175_17530


namespace prize_distribution_methods_l175_17522

-- Define the number of prizes
def num_prizes : ℕ := 6

-- Define the number of people
def num_people : ℕ := 5

-- Define a function to calculate combinations
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define a function to calculate permutations
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Theorem statement
theorem prize_distribution_methods :
  (combination num_prizes 2) * (permutation num_people num_people) =
  (number_of_distribution_methods : ℕ) :=
sorry

end prize_distribution_methods_l175_17522


namespace gcd_204_85_l175_17550

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l175_17550


namespace log_9_729_l175_17509

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_9_729 : log 9 729 = 3 := by sorry

end log_9_729_l175_17509


namespace base_conversion_subtraction_l175_17536

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_conversion_subtraction :
  let base_9_number := [4, 2, 3]  -- 324 in base 9 (least significant digit first)
  let base_6_number := [1, 2, 2]  -- 221 in base 6 (least significant digit first)
  (to_base_10 base_9_number 9) - (to_base_10 base_6_number 6) = 180 := by
  sorry

end base_conversion_subtraction_l175_17536


namespace last_digit_of_product_l175_17572

theorem last_digit_of_product : (3^65 * 6^59 * 7^71) % 10 = 4 := by
  sorry

end last_digit_of_product_l175_17572


namespace min_value_theorem_l175_17570

theorem min_value_theorem (x y : ℝ) (h1 : x + y = 1) (h2 : x > 0) (h3 : y > 0) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 
    (1 / (2 * x) + x / (y + 1)) ≤ (1 / (2 * a) + a / (b + 1))) ∧
  (1 / (2 * x) + x / (y + 1) = 5/4) :=
sorry

end min_value_theorem_l175_17570


namespace smaller_number_proof_l175_17599

theorem smaller_number_proof (x y : ℝ) : 
  y = 2 * x - 3 → 
  x + y = 51 → 
  min x y = 18 := by
sorry

end smaller_number_proof_l175_17599


namespace avocados_bought_by_sister_georgie_guacamole_problem_l175_17597

theorem avocados_bought_by_sister (avocados_per_serving : ℕ) (initial_avocados : ℕ) (servings_made : ℕ) : ℕ :=
  let total_avocados_needed := avocados_per_serving * servings_made
  let additional_avocados := total_avocados_needed - initial_avocados
  additional_avocados

theorem georgie_guacamole_problem :
  avocados_bought_by_sister 3 5 3 = 4 := by
  sorry

end avocados_bought_by_sister_georgie_guacamole_problem_l175_17597


namespace pigeons_on_pole_l175_17551

theorem pigeons_on_pole (initial_pigeons : ℕ) (pigeons_flew_away : ℕ) (pigeons_left : ℕ) : 
  initial_pigeons = 8 → pigeons_flew_away = 3 → pigeons_left = initial_pigeons - pigeons_flew_away → pigeons_left = 5 := by
  sorry

end pigeons_on_pole_l175_17551


namespace sum_last_two_digits_lfs_l175_17537

/-- Lucas Factorial Series function -/
def lucasFactorialSeries : ℕ → ℕ
| 0 => 2
| 1 => 1
| 2 => 3
| 3 => 4
| 4 => 7
| 5 => 11
| _ => 0

/-- Calculate factorial -/
def factorial : ℕ → ℕ
| 0 => 1
| n + 1 => (n + 1) * factorial n

/-- Get last two digits of a number -/
def lastTwoDigits (n : ℕ) : ℕ :=
  n % 100

/-- Sum of last two digits of factorials in Lucas Factorial Series -/
def sumLastTwoDigitsLFS : ℕ :=
  let series := List.range 6
  series.foldl (fun acc i => acc + lastTwoDigits (factorial (lucasFactorialSeries i))) 0

/-- Main theorem -/
theorem sum_last_two_digits_lfs :
  sumLastTwoDigitsLFS = 73 := by
  sorry


end sum_last_two_digits_lfs_l175_17537


namespace min_squares_sum_l175_17576

theorem min_squares_sum (n : ℕ) (h1 : n < 8) (h2 : ∃ a : ℕ, 3 * n + 1 = a ^ 2) :
  (∃ k : ℕ, (∃ (x y z : ℕ), n + 1 = x^2 + y^2 + z^2) ∧
            (∀ m : ℕ, m < k → ¬∃ (a b c : ℕ), n + 1 = a^2 + b^2 + c^2 ∧ 
              (∀ i : ℕ, i > m → c^2 = 0))) ∧
  (∀ k : ℕ, (∃ (x y z : ℕ), n + 1 = x^2 + y^2 + z^2) ∧
            (∀ m : ℕ, m < k → ¬∃ (a b c : ℕ), n + 1 = a^2 + b^2 + c^2 ∧ 
              (∀ i : ℕ, i > m → c^2 = 0)) → k ≥ 3) :=
by sorry

end min_squares_sum_l175_17576


namespace rectangle_area_double_triangle_area_double_circle_area_quadruple_fraction_unchanged_triple_negative_more_negative_all_statements_correct_l175_17523

-- Statement A
theorem rectangle_area_double (b h : ℝ) (h_pos : 0 < h) :
  2 * (b * h) = b * (2 * h) := by sorry

-- Statement B
theorem triangle_area_double (b h : ℝ) (h_pos : 0 < h) :
  2 * ((1/2) * b * h) = (1/2) * (2 * b) * h := by sorry

-- Statement C
theorem circle_area_quadruple (r : ℝ) (r_pos : 0 < r) :
  (π * (2 * r)^2) = 4 * (π * r^2) := by sorry

-- Statement D
theorem fraction_unchanged (a b : ℝ) (b_nonzero : b ≠ 0) :
  (2 * a) / (2 * b) = a / b := by sorry

-- Statement E
theorem triple_negative_more_negative (x : ℝ) (x_neg : x < 0) :
  3 * x < x := by sorry

-- All statements are correct
theorem all_statements_correct :
  (∀ b h, 0 < h → 2 * (b * h) = b * (2 * h)) ∧
  (∀ b h, 0 < h → 2 * ((1/2) * b * h) = (1/2) * (2 * b) * h) ∧
  (∀ r, 0 < r → (π * (2 * r)^2) = 4 * (π * r^2)) ∧
  (∀ a b, b ≠ 0 → (2 * a) / (2 * b) = a / b) ∧
  (∀ x, x < 0 → 3 * x < x) := by sorry

end rectangle_area_double_triangle_area_double_circle_area_quadruple_fraction_unchanged_triple_negative_more_negative_all_statements_correct_l175_17523


namespace ann_age_l175_17598

/-- The complex age relationship between Ann and Barbara --/
def age_relationship (a b : ℕ) : Prop :=
  ∃ y : ℕ, b = a / 2 + 2 * y ∧ y = a - b

/-- The theorem stating Ann's age given the conditions --/
theorem ann_age :
  ∀ a b : ℕ,
  age_relationship a b →
  a + b = 54 →
  a = 29 :=
by
  sorry

end ann_age_l175_17598


namespace expression_change_l175_17518

theorem expression_change (x : ℝ) (b : ℝ) (h : b > 0) :
  (b*x)^2 - 5 - (x^2 - 5) = (b^2 - 1) * x^2 := by
  sorry

end expression_change_l175_17518


namespace remainder_problem_l175_17595

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k < 41) 
  (h3 : k % 5 = 2) 
  (h4 : k % 6 = 5) : 
  k % 7 = 3 := by
sorry

end remainder_problem_l175_17595


namespace f_10_eq_3_div_5_l175_17548

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (x : ℝ) (h : x > 0) : f x = 2 * f (1/x) * Real.log x + 1

theorem f_10_eq_3_div_5 : f 10 = 3/5 := by sorry

end f_10_eq_3_div_5_l175_17548


namespace expression_equality_l175_17557

theorem expression_equality : 
  -21 * (2/3) + 3 * (1/4) - (-2/3) - (1/4) = -18 := by
  sorry

end expression_equality_l175_17557


namespace inequality_solution_set_l175_17519

theorem inequality_solution_set (x : ℝ) : 2 * x - 3 > 7 - x ↔ x > 10 / 3 := by sorry

end inequality_solution_set_l175_17519


namespace absolute_value_equation_unique_solution_l175_17543

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x + 3| := by sorry

end absolute_value_equation_unique_solution_l175_17543


namespace gumdrop_cost_l175_17539

/-- Given a total amount of 224 cents and the ability to buy 28 gumdrops,
    prove that the cost of each gumdrop is 8 cents. -/
theorem gumdrop_cost (total : ℕ) (quantity : ℕ) (h1 : total = 224) (h2 : quantity = 28) :
  total / quantity = 8 := by
  sorry

end gumdrop_cost_l175_17539


namespace school_bus_capacity_l175_17565

/-- Calculates the total number of students that can be seated on a bus --/
def bus_capacity (rows : ℕ) (sections_per_row : ℕ) (students_per_section : ℕ) : ℕ :=
  rows * sections_per_row * students_per_section

/-- Theorem: A bus with 13 rows, 2 sections per row, and 2 students per section can seat 52 students --/
theorem school_bus_capacity : bus_capacity 13 2 2 = 52 := by
  sorry

end school_bus_capacity_l175_17565


namespace equidistant_points_characterization_l175_17506

/-- A ray in a plane --/
structure Ray where
  start : ℝ × ℝ
  direction : ℝ × ℝ

/-- The set of points equidistant from two rays --/
def EquidistantPoints (ray1 ray2 : Ray) : Set (ℝ × ℝ) :=
  sorry

/-- Angle bisector of two lines --/
def AngleBisector (line1 line2 : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

/-- Perpendicular bisector of a segment --/
def PerpendicularBisector (a b : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Parabola with focus and directrix --/
def Parabola (focus : ℝ × ℝ) (directrix : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

/-- The line containing a ray --/
def LineContainingRay (ray : Ray) : Set (ℝ × ℝ) :=
  sorry

theorem equidistant_points_characterization (ray1 ray2 : Ray) :
  EquidistantPoints ray1 ray2 =
    (AngleBisector (LineContainingRay ray1) (LineContainingRay ray2)) ∪
    (if ray1.start ≠ ray2.start then PerpendicularBisector ray1.start ray2.start else ∅) ∪
    (Parabola ray1.start (LineContainingRay ray2)) ∪
    (Parabola ray2.start (LineContainingRay ray1)) :=
  sorry

end equidistant_points_characterization_l175_17506


namespace rhombus_perimeter_l175_17511

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end rhombus_perimeter_l175_17511


namespace count_numbers_with_property_l175_17580

/-- Represents a four-digit number -/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n ≤ 9999 }

/-- Extracts the leftmost digit of a four-digit number -/
def leftmostDigit (n : FourDigitNumber) : ℕ := n.val / 1000

/-- Extracts the three-digit number obtained by removing the leftmost digit -/
def rightThreeDigits (n : FourDigitNumber) : ℕ := n.val % 1000

/-- Checks if a four-digit number satisfies the given property -/
def satisfiesProperty (n : FourDigitNumber) : Prop :=
  7 * (rightThreeDigits n) = n.val

theorem count_numbers_with_property :
  ∃ (S : Finset FourDigitNumber),
    (∀ n ∈ S, satisfiesProperty n) ∧
    (∀ n : FourDigitNumber, satisfiesProperty n → n ∈ S) ∧
    Finset.card S = 5 := by
  sorry

end count_numbers_with_property_l175_17580


namespace smallest_fraction_between_l175_17591

theorem smallest_fraction_between (p q : ℕ+) : 
  (6 : ℚ) / 11 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (5 : ℚ) / 9 ∧ 
  (∀ (p' q' : ℕ+), (6 : ℚ) / 11 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (5 : ℚ) / 9 → q ≤ q') →
  q.val - p.val = 9 := by
  sorry

end smallest_fraction_between_l175_17591


namespace grandfather_grandmother_age_difference_is_two_l175_17542

/-- The age difference between Milena's grandfather and grandmother -/
def grandfather_grandmother_age_difference (milena_age : ℕ) (grandmother_age_factor : ℕ) (milena_grandfather_age_difference : ℕ) : ℕ :=
  (milena_age + milena_grandfather_age_difference) - (milena_age * grandmother_age_factor)

theorem grandfather_grandmother_age_difference_is_two :
  grandfather_grandmother_age_difference 7 9 58 = 2 := by
  sorry

end grandfather_grandmother_age_difference_is_two_l175_17542


namespace existence_of_b₁_b₂_l175_17502

theorem existence_of_b₁_b₂ (a₁ a₂ : ℝ) 
  (h₁ : a₁ ≥ 0) (h₂ : a₂ ≥ 0) (h₃ : a₁ + a₂ = 1) : 
  ∃ b₁ b₂ : ℝ, b₁ ≥ 0 ∧ b₂ ≥ 0 ∧ b₁ + b₂ = 1 ∧ 
  (5/4 - a₁) * b₁ + 3 * (5/4 - a₂) * b₂ > 1 := by
sorry

end existence_of_b₁_b₂_l175_17502


namespace seven_balls_three_boxes_l175_17507

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 365 ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 365 := by sorry

end seven_balls_three_boxes_l175_17507


namespace store_credit_card_discount_l175_17512

theorem store_credit_card_discount (original_price sale_discount_percent coupon_discount total_savings : ℝ) :
  original_price = 125 ∧
  sale_discount_percent = 20 ∧
  coupon_discount = 10 ∧
  total_savings = 44 →
  let sale_discount := original_price * (sale_discount_percent / 100)
  let price_after_sale := original_price - sale_discount
  let price_after_coupon := price_after_sale - coupon_discount
  let store_credit_discount := total_savings - sale_discount - coupon_discount
  (store_credit_discount / price_after_coupon) * 100 = 10 := by sorry

end store_credit_card_discount_l175_17512


namespace complete_square_l175_17508

theorem complete_square (x : ℝ) : x^2 - 8*x + 15 = 0 ↔ (x - 4)^2 = 1 := by
  sorry

end complete_square_l175_17508


namespace min_distance_ellipse_to_N_l175_17515

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x + 3)^2 + y^2) + Real.sqrt ((x - 3)^2 + y^2) = 10

/-- The fixed point N -/
def N : ℝ × ℝ := (-6, 0)

/-- The minimum distance from a point on the ellipse to N -/
def min_distance_to_N : ℝ := 1

/-- Theorem stating the minimum distance from any point on the ellipse to N is 1 -/
theorem min_distance_ellipse_to_N :
  ∀ x y : ℝ, ellipse_equation x y →
  ∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧
  ∀ (q : ℝ × ℝ), ellipse_equation q.1 q.2 →
  dist p N ≤ dist q N ∧ dist p N = min_distance_to_N :=
sorry

end min_distance_ellipse_to_N_l175_17515


namespace parabola_intersection_l175_17513

/-- Given a parabola y = ax^2 + x + c that intersects the x-axis at x = 1, prove that a + c = -1 -/
theorem parabola_intersection (a c : ℝ) : 
  (∀ x, a*x^2 + x + c = 0 → x = 1) → a + c = -1 := by
  sorry

end parabola_intersection_l175_17513


namespace circle_through_origin_equation_l175_17535

/-- Defines a circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point lies on a circle -/
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem circle_through_origin_equation 
  (c : Circle) 
  (h1 : c.center = (3, 4)) 
  (h2 : onCircle c (0, 0)) : 
  ∀ (x y : ℝ), onCircle c (x, y) ↔ (x - 3)^2 + (y - 4)^2 = 25 := by
sorry

end circle_through_origin_equation_l175_17535


namespace smallest_prime_factor_of_2310_l175_17538

theorem smallest_prime_factor_of_2310 : Nat.minFac 2310 = 2 := by
  sorry

end smallest_prime_factor_of_2310_l175_17538


namespace library_books_problem_l175_17573

theorem library_books_problem (initial_books : ℕ) : 
  initial_books - 120 + 35 - 15 = 150 → initial_books = 250 := by
  sorry

end library_books_problem_l175_17573


namespace binomial_expansion_arithmetic_progression_l175_17504

theorem binomial_expansion_arithmetic_progression (n : ℕ) : 
  (∃ (a d : ℚ), 
    (1 : ℚ) = a ∧ 
    (n : ℚ) / 2 = a + d ∧ 
    (n * (n - 1) : ℚ) / 8 = a + 2 * d) ↔ 
  n = 8 := by sorry

end binomial_expansion_arithmetic_progression_l175_17504


namespace steve_book_earnings_l175_17563

/-- Calculates an author's net earnings from book sales -/
def authorNetEarnings (copies : ℕ) (earningsPerCopy : ℚ) (agentPercentage : ℚ) : ℚ :=
  let totalEarnings := copies * earningsPerCopy
  let agentCommission := totalEarnings * agentPercentage
  totalEarnings - agentCommission

/-- Proves that given the specified conditions, the author's net earnings are $1,800,000 -/
theorem steve_book_earnings :
  authorNetEarnings 1000000 2 (1/10) = 1800000 := by
  sorry

#eval authorNetEarnings 1000000 2 (1/10)

end steve_book_earnings_l175_17563


namespace exponent_comparison_l175_17559

theorem exponent_comparison : 65^1000 - 8^2001 > 0 := by
  sorry

end exponent_comparison_l175_17559


namespace functional_equation_solution_l175_17584

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

/-- The main theorem stating that any function satisfying the functional equation
    must be of the form f(x) = kx for some constant k -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x := by
  sorry

end functional_equation_solution_l175_17584


namespace distance_to_AB_l175_17590

/-- Triangle ABC with point M inside -/
structure TriangleWithPoint where
  -- Define the triangle
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Define the distances from M to sides
  distMAC : ℝ
  distMBC : ℝ
  -- Conditions
  AB_positive : AB > 0
  BC_positive : BC > 0
  AC_positive : AC > 0
  distMAC_positive : distMAC > 0
  distMBC_positive : distMBC > 0
  -- Triangle inequality
  triangle_inequality : AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB
  -- M is inside the triangle
  M_inside : distMAC < AC ∧ distMBC < BC

/-- The theorem to be proved -/
theorem distance_to_AB (t : TriangleWithPoint) 
  (h1 : t.AB = 10) 
  (h2 : t.BC = 17) 
  (h3 : t.AC = 21) 
  (h4 : t.distMAC = 2) 
  (h5 : t.distMBC = 4) : 
  ∃ (distMAB : ℝ), distMAB = 29 / 5 := by
  sorry

end distance_to_AB_l175_17590


namespace intersection_points_with_constraints_l175_17528

/-- The number of intersection points of n lines -/
def intersectionPoints (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of lines -/
def numLines : ℕ := 10

/-- The number of parallel line pairs -/
def numParallelPairs : ℕ := 1

/-- The number of lines intersecting at a single point -/
def numConcurrentLines : ℕ := 3

theorem intersection_points_with_constraints :
  intersectionPoints numLines - numParallelPairs - (numConcurrentLines.choose 2 - 1) = 42 := by
  sorry

end intersection_points_with_constraints_l175_17528


namespace football_team_right_handed_count_l175_17556

theorem football_team_right_handed_count 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (left_handed_fraction : ℚ) :
  total_players = 120 →
  throwers = 67 →
  left_handed_fraction = 2 / 5 →
  ∃ (right_handed : ℕ), 
    right_handed = throwers + (total_players - throwers - Int.floor ((total_players - throwers : ℚ) * left_handed_fraction)) ∧
    right_handed = 99 :=
by sorry

end football_team_right_handed_count_l175_17556


namespace root_difference_quadratic_equation_l175_17588

theorem root_difference_quadratic_equation :
  let a : ℝ := 2
  let b : ℝ := 5
  let c : ℝ := -12
  let larger_root : ℝ := (-b + (b^2 - 4*a*c).sqrt) / (2*a)
  let smaller_root : ℝ := (-b - (b^2 - 4*a*c).sqrt) / (2*a)
  larger_root - smaller_root = 5.5 :=
by sorry

end root_difference_quadratic_equation_l175_17588


namespace derivative_of_f_l175_17593

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp (2*x)

theorem derivative_of_f :
  deriv f = λ x => Real.exp (2*x) * (2*x + 2*x^2) := by sorry

end derivative_of_f_l175_17593


namespace roselyn_initial_books_l175_17526

def books_problem (books_to_rebecca : ℕ) (books_remaining : ℕ) : Prop :=
  let books_to_mara := 3 * books_to_rebecca
  let total_given := books_to_rebecca + books_to_mara
  let initial_books := total_given + books_remaining
  initial_books = 220

theorem roselyn_initial_books :
  books_problem 40 60 := by
  sorry

end roselyn_initial_books_l175_17526


namespace rose_count_prediction_l175_17503

/-- Given a sequence of rose counts for four consecutive months, where the differences
    between consecutive counts form an arithmetic sequence with a common difference of 12,
    prove that the next term in the sequence will be 224. -/
theorem rose_count_prediction (a b c d : ℕ) (hab : b - a = 18) (hbc : c - b = 30) (hcd : d - c = 42) :
  d + (d - c + 12) = 224 :=
sorry

end rose_count_prediction_l175_17503


namespace sin_330_degrees_l175_17516

theorem sin_330_degrees : Real.sin (330 * π / 180) = -(1 / 2) := by
  sorry

end sin_330_degrees_l175_17516


namespace base_number_proof_l175_17578

theorem base_number_proof (x : ℝ) (h : Real.sqrt (x^12) = 64) : x = 2 := by
  sorry

end base_number_proof_l175_17578


namespace triangle_angle_B_l175_17541

/-- In a triangle ABC, given side lengths and an angle, prove that angle B has two possible values. -/
theorem triangle_angle_B (a b : ℝ) (A B : ℝ) : 
  a = (5 * Real.sqrt 3) / 3 → 
  b = 5 → 
  A = π / 6 → 
  (B = π / 3 ∨ B = 2 * π / 3) := by
  sorry

end triangle_angle_B_l175_17541


namespace special_circle_equation_l175_17552

/-- A circle passing through the origin and point (1, 1) with its center on the line 2x + 3y + 1 = 0 -/
def special_circle (x y : ℝ) : Prop :=
  (x - 4)^2 + (y + 3)^2 = 25

/-- The line on which the center of the circle lies -/
def center_line (x y : ℝ) : Prop :=
  2*x + 3*y + 1 = 0

theorem special_circle_equation :
  ∀ x y : ℝ,
  (special_circle x y ↔
    (x^2 + y^2 = 0 ∨ (x - 1)^2 + (y - 1)^2 = 0) ∧
    ∃ c_x c_y : ℝ, center_line c_x c_y ∧ (x - c_x)^2 + (y - c_y)^2 = (c_x^2 + c_y^2)) :=
by sorry

end special_circle_equation_l175_17552


namespace dereks_initial_lunch_spending_l175_17527

/-- Represents the problem of determining Derek's initial lunch spending --/
theorem dereks_initial_lunch_spending 
  (derek_initial : ℕ) 
  (derek_dad_lunch : ℕ) 
  (derek_extra_lunch : ℕ) 
  (dave_initial : ℕ) 
  (dave_mom_lunch : ℕ) 
  (dave_extra : ℕ) 
  (h1 : derek_initial = 40)
  (h2 : derek_dad_lunch = 11)
  (h3 : derek_extra_lunch = 5)
  (h4 : dave_initial = 50)
  (h5 : dave_mom_lunch = 7)
  (h6 : dave_extra = 33)
  : ∃ (derek_self_lunch : ℕ), 
    derek_self_lunch = 14 ∧ 
    dave_initial - dave_mom_lunch = 
    (derek_initial - (derek_self_lunch + derek_dad_lunch + derek_extra_lunch)) + dave_extra :=
by sorry

end dereks_initial_lunch_spending_l175_17527


namespace intersection_segment_length_l175_17514

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := x^2 = -4*y
def line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ line A.1 A.2 ∧
  parabola B.1 B.2 ∧ line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_segment_length :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by sorry

end intersection_segment_length_l175_17514


namespace sqrt_sum_reciprocals_l175_17579

theorem sqrt_sum_reciprocals : Real.sqrt (1 / 4 + 1 / 25) = Real.sqrt 29 / 10 := by
  sorry

end sqrt_sum_reciprocals_l175_17579


namespace f_monotonicity_and_intersection_l175_17596

/-- The function f(x) = x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem f_monotonicity_and_intersection (a : ℝ) :
  (∀ x : ℝ, a ≥ 1/3 → Monotone (f a)) ∧
  (∃ x y : ℝ, x = 1 ∧ y = a + 1 ∧ f a x = y ∧ f' a x * x = y) ∧
  (∃ x y : ℝ, x = -1 ∧ y = -a - 1 ∧ f a x = y ∧ f' a x * x = y) := by
  sorry

end f_monotonicity_and_intersection_l175_17596


namespace remainder_1999_11_mod_8_l175_17525

theorem remainder_1999_11_mod_8 : 1999^11 % 8 = 7 := by
  sorry

end remainder_1999_11_mod_8_l175_17525


namespace team_average_score_l175_17589

theorem team_average_score (lefty_score : ℕ) (righty_score : ℕ) (other_score : ℕ) :
  lefty_score = 20 →
  righty_score = lefty_score / 2 →
  other_score = righty_score * 6 →
  (lefty_score + righty_score + other_score) / 3 = 30 := by
  sorry

end team_average_score_l175_17589


namespace smallest_composite_no_small_factors_l175_17567

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p, p < k → Nat.Prime p → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 529) ∧
  (has_no_prime_factors_less_than 529 20) ∧
  (∀ m : ℕ, m < 529 → ¬(is_composite m ∧ has_no_prime_factors_less_than m 20)) :=
sorry

end smallest_composite_no_small_factors_l175_17567


namespace total_apples_eaten_l175_17569

def simone_daily_consumption : ℚ := 1/2
def simone_days : ℕ := 16
def lauri_daily_consumption : ℚ := 1/3
def lauri_days : ℕ := 15

theorem total_apples_eaten :
  (simone_daily_consumption * simone_days + lauri_daily_consumption * lauri_days : ℚ) = 13 := by
  sorry

end total_apples_eaten_l175_17569


namespace smallest_number_divisible_by_1_to_18_not_19_20_l175_17587

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def divisible_up_to (n m : ℕ) : Prop := ∀ i : ℕ, 1 ≤ i → i ≤ m → is_divisible n i

theorem smallest_number_divisible_by_1_to_18_not_19_20 :
  ∃ n : ℕ, 
    n > 0 ∧
    divisible_up_to n 18 ∧
    ¬(is_divisible n 19) ∧
    ¬(is_divisible n 20) ∧
    ∀ m : ℕ, m > 0 → divisible_up_to m 18 → ¬(is_divisible m 19) → ¬(is_divisible m 20) → n ≤ m :=
by
  sorry

#eval 12252240

end smallest_number_divisible_by_1_to_18_not_19_20_l175_17587


namespace tetrahedron_properties_l175_17582

-- Define the vertices of the tetrahedron
def A₁ : ℝ × ℝ × ℝ := (4, -1, 3)
def A₂ : ℝ × ℝ × ℝ := (-2, 1, 0)
def A₃ : ℝ × ℝ × ℝ := (0, -5, 1)
def A₄ : ℝ × ℝ × ℝ := (3, 2, -6)

-- Function to calculate the volume of a tetrahedron
def tetrahedron_volume (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Function to calculate the height from a point to a plane defined by three points
def height_to_plane (point plane1 plane2 plane3 : ℝ × ℝ × ℝ) : ℝ := sorry

theorem tetrahedron_properties :
  let volume := tetrahedron_volume A₁ A₂ A₃ A₄
  let height := height_to_plane A₄ A₁ A₂ A₃
  volume = 136 / 3 ∧ height = 17 / Real.sqrt 5 := by sorry

end tetrahedron_properties_l175_17582


namespace circular_fields_area_difference_l175_17540

theorem circular_fields_area_difference (r₁ r₂ : ℝ) (h : r₁ / r₂ = 3 / 10) :
  1 - (π * r₁^2) / (π * r₂^2) = 91 / 100 := by
  sorry

end circular_fields_area_difference_l175_17540


namespace trigonometric_inequality_l175_17592

theorem trigonometric_inequality (a b A B : ℝ) :
  (∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ) ≥ 0) →
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
sorry

end trigonometric_inequality_l175_17592


namespace flash_catch_up_distance_l175_17555

theorem flash_catch_up_distance 
  (v : ℝ) -- Ace's speed
  (z : ℝ) -- Flash's speed multiplier
  (k : ℝ) -- Ace's head start distance
  (t₀ : ℝ) -- Time Ace runs before Flash starts
  (h₁ : v > 0) -- Ace's speed is positive
  (h₂ : z > 1) -- Flash is faster than Ace
  (h₃ : k ≥ 0) -- Head start is non-negative
  (h₄ : t₀ ≥ 0) -- Time before Flash starts is non-negative
  : 
  ∃ (t : ℝ), t > 0 ∧ z * v * t = v * (t + t₀) + k ∧
  z * v * t = z * (t₀ * v + k) / (z - 1) :=
sorry

end flash_catch_up_distance_l175_17555


namespace probability_shirt_shorts_hat_l175_17585

/-- The number of shirts in the drawer -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the drawer -/
def num_shorts : ℕ := 7

/-- The number of pairs of socks in the drawer -/
def num_socks : ℕ := 6

/-- The number of hats in the drawer -/
def num_hats : ℕ := 3

/-- The total number of articles of clothing in the drawer -/
def total_articles : ℕ := num_shirts + num_shorts + num_socks + num_hats

/-- The number of articles to be chosen -/
def num_chosen : ℕ := 3

theorem probability_shirt_shorts_hat : 
  (num_shirts.choose 1 * num_shorts.choose 1 * num_hats.choose 1 : ℚ) / 
  (total_articles.choose num_chosen) = 63 / 770 :=
sorry

end probability_shirt_shorts_hat_l175_17585


namespace game_cost_calculation_l175_17577

theorem game_cost_calculation (total_earned : ℕ) (blade_cost : ℕ) (num_games : ℕ) 
  (h1 : total_earned = 69)
  (h2 : blade_cost = 24)
  (h3 : num_games = 9)
  (h4 : total_earned ≥ blade_cost) :
  (total_earned - blade_cost) / num_games = 5 := by
  sorry

end game_cost_calculation_l175_17577


namespace trigonometric_inequalities_l175_17517

theorem trigonometric_inequalities (α β γ : ℝ) (h : α + β + γ = 0) :
  (|Real.cos (α + β)| ≤ |Real.cos α| + |Real.sin β|) ∧
  (|Real.sin (α + β)| ≤ |Real.cos α| + |Real.cos β|) ∧
  (|Real.cos α| + |Real.cos β| + |Real.cos γ| ≥ 1) :=
by sorry

end trigonometric_inequalities_l175_17517


namespace intersection_complement_equals_l175_17583

def U : Finset Int := {-1, 0, 1, 2, 3, 4}
def A : Finset Int := {2, 3}
def B : Finset Int := {1, 2, 3, 4} \ A

theorem intersection_complement_equals : B ∩ (U \ A) = {1, 4} := by
  sorry

end intersection_complement_equals_l175_17583


namespace candy_cost_in_dollars_l175_17558

/-- The cost of a single piece of candy in cents -/
def candy_cost : ℕ := 2

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of candy pieces we're calculating the cost for -/
def candy_pieces : ℕ := 500

theorem candy_cost_in_dollars : 
  (candy_pieces * candy_cost) / cents_per_dollar = 10 := by
  sorry

end candy_cost_in_dollars_l175_17558


namespace short_stack_pancakes_l175_17554

/-- The number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- The number of customers who ordered short stack -/
def short_stack_orders : ℕ := 9

/-- The number of customers who ordered big stack -/
def big_stack_orders : ℕ := 6

/-- The total number of pancakes needed -/
def total_pancakes : ℕ := 57

/-- The number of pancakes in a short stack -/
def short_stack : ℕ := 3

theorem short_stack_pancakes :
  short_stack * short_stack_orders + big_stack * big_stack_orders = total_pancakes :=
by sorry

end short_stack_pancakes_l175_17554


namespace duck_race_charity_amount_l175_17566

/-- The amount of money raised for charity in the annual rubber duck race -/
def charity_money_raised (regular_price : ℝ) (large_price : ℝ) (regular_sold : ℕ) (large_sold : ℕ) : ℝ :=
  regular_price * (regular_sold : ℝ) + large_price * (large_sold : ℝ)

/-- Theorem stating the amount of money raised for charity in the given scenario -/
theorem duck_race_charity_amount :
  charity_money_raised 3 5 221 185 = 1588 :=
by
  sorry

end duck_race_charity_amount_l175_17566


namespace inequalities_proof_l175_17544

theorem inequalities_proof (a b c : ℝ) (h1 : a > 0) (h2 : a > b) (h3 : b > c) : 
  (a * b > b * c) ∧ (a * c > b * c) ∧ (a * b > a * c) ∧ (a + b > b + c) := by
  sorry

end inequalities_proof_l175_17544


namespace game_result_l175_17532

def g (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 5 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2]
def betty_rolls : List ℕ := [10, 3, 3, 2]

theorem game_result : 
  (List.sum (List.map g allie_rolls)) * (List.sum (List.map g betty_rolls)) = 66 := by
  sorry

end game_result_l175_17532


namespace grade_change_impossible_l175_17547

theorem grade_change_impossible : ∀ (n1 n2 n3 n4 : ℤ),
  2 * n1 + n2 - 2 * n3 - n4 = 27 ∧
  -n1 + 2 * n2 + n3 - 2 * n4 = -27 →
  False :=
by
  sorry

end grade_change_impossible_l175_17547


namespace compound_weight_l175_17564

/-- Given a compound with a molecular weight of 1188, prove that the total weight of 4 moles is 4752,
    while the molecular weight remains constant. -/
theorem compound_weight (molecular_weight : ℕ) (num_moles : ℕ) :
  molecular_weight = 1188 → num_moles = 4 →
  (num_moles * molecular_weight = 4752) ∧ (molecular_weight = 1188) := by
  sorry

#check compound_weight

end compound_weight_l175_17564


namespace total_games_won_l175_17524

/-- The number of games won by the Chicago Bulls -/
def bulls_wins : ℕ := 70

/-- The number of games won by the Miami Heat -/
def heat_wins : ℕ := bulls_wins + 5

/-- The number of games won by the New York Knicks -/
def knicks_wins : ℕ := 2 * heat_wins

/-- The number of games won by the Los Angeles Lakers -/
def lakers_wins : ℕ := (3 * (bulls_wins + knicks_wins)) / 2

/-- The total number of games won by all four teams -/
def total_wins : ℕ := bulls_wins + heat_wins + knicks_wins + lakers_wins

theorem total_games_won : total_wins = 625 := by
  sorry

end total_games_won_l175_17524


namespace alyssa_pears_l175_17500

theorem alyssa_pears (total_pears nancy_pears : ℕ) 
  (h1 : total_pears = 59) 
  (h2 : nancy_pears = 17) : 
  total_pears - nancy_pears = 42 := by
sorry

end alyssa_pears_l175_17500


namespace brother_age_problem_l175_17501

theorem brother_age_problem (younger_age older_age : ℕ) : 
  younger_age + older_age = 26 → 
  older_age = younger_age + 2 → 
  older_age = 14 := by
sorry

end brother_age_problem_l175_17501


namespace clothing_factory_payment_theorem_l175_17533

/-- Represents the payment calculation for two discount plans in a clothing factory. -/
def ClothingFactoryPayment (x : ℕ) : Prop :=
  let suitPrice : ℕ := 400
  let tiePrice : ℕ := 80
  let numSuits : ℕ := 20
  let y₁ : ℕ := suitPrice * numSuits + (x - numSuits) * tiePrice
  let y₂ : ℕ := (suitPrice * numSuits + tiePrice * x) * 9 / 10
  (x > 20) →
  (y₁ = 80 * x + 6400) ∧
  (y₂ = 72 * x + 7200) ∧
  (x = 30 → y₁ < y₂)

theorem clothing_factory_payment_theorem :
  ∀ x : ℕ, ClothingFactoryPayment x :=
sorry

end clothing_factory_payment_theorem_l175_17533


namespace remaining_slices_l175_17520

def total_slices : ℕ := 2 * 8

def slices_after_friends : ℕ := total_slices - (total_slices / 4)

def slices_after_family : ℕ := slices_after_friends - (slices_after_friends / 3)

def slices_after_alex : ℕ := slices_after_family - 3

theorem remaining_slices : slices_after_alex = 5 := by
  sorry

end remaining_slices_l175_17520


namespace abc_mod_seven_l175_17553

theorem abc_mod_seven (a b c : ℕ) (ha : a < 7) (hb : b < 7) (hc : c < 7)
  (h1 : (a + 3*b + 2*c) % 7 = 2)
  (h2 : (2*a + b + 3*c) % 7 = 3)
  (h3 : (3*a + 2*b + c) % 7 = 5) :
  (a * b * c) % 7 = 1 := by
sorry

end abc_mod_seven_l175_17553


namespace pink_shells_count_l175_17534

theorem pink_shells_count (total : ℕ) (purple yellow blue orange : ℕ) 
  (h1 : total = 65)
  (h2 : purple = 13)
  (h3 : yellow = 18)
  (h4 : blue = 12)
  (h5 : orange = 14) :
  total - (purple + yellow + blue + orange) = 8 := by
  sorry

end pink_shells_count_l175_17534


namespace log_base_2_derivative_l175_17575

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end log_base_2_derivative_l175_17575


namespace integer_solutions_quadratic_equation_l175_17581

theorem integer_solutions_quadratic_equation :
  ∀ x y : ℤ, x^2 + 2*x*y + 3*y^2 - 2*x + y + 1 = 0 ↔ 
  (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -1) ∨ (x = 3 ∧ y = -1) :=
by sorry

end integer_solutions_quadratic_equation_l175_17581


namespace modular_arithmetic_problem_l175_17510

theorem modular_arithmetic_problem :
  ∃ (a b c : ℤ),
    (7 * a) % 60 = 1 ∧
    (13 * b) % 60 = 1 ∧
    (17 * c) % 60 = 1 ∧
    (4 * a + 12 * b - 6 * c) % 60 = 58 := by
  sorry

end modular_arithmetic_problem_l175_17510


namespace monotonicity_and_range_of_a_l175_17545

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := (x + a) / (a * Real.exp x)

theorem monotonicity_and_range_of_a :
  (a ≠ 0) →
  ((a > 0 → 
    (∀ x₁ x₂, x₁ < 1 - a ∧ x₂ < 1 - a → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂, x₁ > 1 - a ∧ x₂ > 1 - a → f a x₁ > f a x₂)) ∧
   (a < 0 → 
    (∀ x₁ x₂, x₁ < 1 - a ∧ x₂ < 1 - a → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, x₁ > 1 - a ∧ x₂ > 1 - a → f a x₁ < f a x₂))) ∧
  ((∀ x > 0, (3 + 2 * Real.log x) / Real.exp x ≤ f a x + 2 * x) →
   (a ∈ Set.Iic (-1/2) ∪ Set.Ioi 0)) := by
  sorry

end monotonicity_and_range_of_a_l175_17545


namespace gcd_of_90_and_405_l175_17571

theorem gcd_of_90_and_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_of_90_and_405_l175_17571


namespace max_gcd_consecutive_terms_l175_17562

def b (n : ℕ) : ℕ := (n^2).factorial + n

theorem max_gcd_consecutive_terms : 
  ∃ (k : ℕ), k ≥ 1 ∧ Nat.gcd (b k) (b (k+1)) = 2 ∧ 
  ∀ (n : ℕ), n ≥ 1 → Nat.gcd (b n) (b (n+1)) ≤ 2 :=
sorry

end max_gcd_consecutive_terms_l175_17562


namespace polynomial_division_remainder_l175_17594

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^6 + X^5 + 2*X^3 - X^2 + 3 = (X + 2) * (X - 1) * q + (-X + 5) := by
  sorry

end polynomial_division_remainder_l175_17594


namespace cube_volume_ratio_l175_17521

theorem cube_volume_ratio (edge_ratio : ℝ) (small_volume : ℝ) :
  edge_ratio = 4.999999999999999 →
  small_volume = 1 →
  (edge_ratio ^ 3) * small_volume = 125 :=
by
  sorry

end cube_volume_ratio_l175_17521


namespace geometric_sequence_sum_l175_17505

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ+ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ+, a n > 0) →
  a 1 * a 3 + 2 * a 2 * a 5 + a 4 * a 6 = 36 →
  a 2 + a 5 = 6 := by
  sorry

end geometric_sequence_sum_l175_17505


namespace solution_difference_l175_17574

theorem solution_difference : ∃ p q : ℝ, 
  (p - 4) * (p + 4) = 24 * p - 96 ∧ 
  (q - 4) * (q + 4) = 24 * q - 96 ∧ 
  p ≠ q ∧ 
  p > q ∧ 
  p - q = 16 := by sorry

end solution_difference_l175_17574


namespace baker_leftover_cupcakes_l175_17529

/-- Represents the cupcake distribution problem --/
def cupcake_distribution (total_cupcakes nutty_cupcakes gluten_free_cupcakes num_children : ℕ)
  (num_nut_allergic num_gluten_only : ℕ) : ℕ :=
  let regular_cupcakes := total_cupcakes - nutty_cupcakes - gluten_free_cupcakes
  let nutty_per_child := nutty_cupcakes / (num_children - num_nut_allergic)
  let nutty_distributed := nutty_per_child * (num_children - num_nut_allergic)
  let regular_per_child := regular_cupcakes / num_children
  let regular_distributed := regular_per_child * num_children
  let leftover_nutty := nutty_cupcakes - nutty_distributed
  let leftover_regular := regular_cupcakes - regular_distributed
  leftover_nutty + leftover_regular

/-- Theorem stating that given the specific conditions, Ms. Baker will have 5 cupcakes left over --/
theorem baker_leftover_cupcakes :
  cupcake_distribution 84 18 25 7 2 1 = 5 := by
  sorry

end baker_leftover_cupcakes_l175_17529


namespace exists_irrational_less_than_four_l175_17586

theorem exists_irrational_less_than_four : ∃ x : ℝ, Irrational x ∧ x < 4 := by
  sorry

end exists_irrational_less_than_four_l175_17586


namespace product_of_polynomials_l175_17531

/-- Given two polynomials A(d) and B(d) whose product is C(d), prove that k + m = -4 --/
theorem product_of_polynomials (k m : ℚ) : 
  (∀ d : ℚ, (5*d^2 - 2*d + k) * (4*d^2 + m*d - 9) = 20*d^4 - 28*d^3 + 13*d^2 - m*d - 18) → 
  k + m = -4 := by
  sorry

end product_of_polynomials_l175_17531


namespace gemma_pizza_order_l175_17546

/-- The number of pizzas Gemma ordered -/
def number_of_pizzas : ℕ := 4

/-- The cost of each pizza in dollars -/
def pizza_cost : ℕ := 10

/-- The tip amount in dollars -/
def tip_amount : ℕ := 5

/-- The amount Gemma paid with in dollars -/
def payment_amount : ℕ := 50

/-- The change Gemma received in dollars -/
def change_amount : ℕ := 5

theorem gemma_pizza_order :
  number_of_pizzas * pizza_cost + tip_amount = payment_amount - change_amount :=
sorry

end gemma_pizza_order_l175_17546


namespace election_winner_percentage_l175_17568

theorem election_winner_percentage (winner_votes loser_votes : ℕ) 
  (h1 : winner_votes = 1344)
  (h2 : winner_votes - loser_votes = 288) :
  (winner_votes : ℚ) / ((winner_votes : ℚ) + (loser_votes : ℚ)) * 100 = 56 := by
  sorry

end election_winner_percentage_l175_17568


namespace mean_equality_implies_x_value_l175_17560

theorem mean_equality_implies_x_value : 
  ∃ x : ℝ, (6 + 9 + 18) / 3 = (x + 15) / 2 → x = 7 := by
  sorry

end mean_equality_implies_x_value_l175_17560


namespace june_rainfall_l175_17561

def rainfall_march : ℝ := 3.79
def rainfall_april : ℝ := 4.5
def rainfall_may : ℝ := 3.95
def rainfall_july : ℝ := 4.67
def average_rainfall : ℝ := 4
def num_months : ℕ := 5

theorem june_rainfall :
  let total_rainfall := average_rainfall * num_months
  let known_rainfall := rainfall_march + rainfall_april + rainfall_may + rainfall_july
  let june_rainfall := total_rainfall - known_rainfall
  june_rainfall = 3.09 := by sorry

end june_rainfall_l175_17561
