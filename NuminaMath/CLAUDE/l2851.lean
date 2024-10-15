import Mathlib

namespace NUMINAMATH_CALUDE_point_position_on_line_l2851_285139

/-- Given five points on a line and a point P satisfying certain conditions, prove the position of P -/
theorem point_position_on_line (a b c d : ℝ) :
  let O := (0 : ℝ)
  let A := a
  let B := b
  let C := c
  let D := d
  ∀ P, b ≤ P ∧ P ≤ c →
  (A - P) / (P - D) = (B - P) / (P - C) →
  P = (a * c - b * d) / (a - b + c - d) :=
by sorry

end NUMINAMATH_CALUDE_point_position_on_line_l2851_285139


namespace NUMINAMATH_CALUDE_max_value_theorem_l2851_285182

theorem max_value_theorem (x y z : ℝ) (h : x + 2 * y + z = 4) :
  ∃ (max : ℝ), max = 4 ∧ ∀ (a b c : ℝ), a + 2 * b + c = 4 → a * b + a * c + b * c ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2851_285182


namespace NUMINAMATH_CALUDE_travel_time_calculation_l2851_285102

theorem travel_time_calculation (total_time subway_time : ℕ) 
  (h1 : total_time = 38)
  (h2 : subway_time = 10)
  (h3 : total_time = subway_time + 2 * subway_time + (total_time - subway_time - 2 * subway_time)) :
  total_time - subway_time - 2 * subway_time = 8 :=
by sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l2851_285102


namespace NUMINAMATH_CALUDE_lightest_box_weight_l2851_285156

/-- Given three boxes with pairwise sums of weights 83 kg, 85 kg, and 86 kg,
    the weight of the lightest box is 41 kg. -/
theorem lightest_box_weight (s m l : ℝ) : 
  s ≤ m ∧ m ≤ l ∧ 
  m + s = 83 ∧ 
  l + s = 85 ∧ 
  l + m = 86 → 
  s = 41 := by
sorry

end NUMINAMATH_CALUDE_lightest_box_weight_l2851_285156


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l2851_285100

theorem green_shirt_pairs (total_students : ℕ) (red_shirts : ℕ) (green_shirts : ℕ) 
  (total_pairs : ℕ) (red_red_pairs : ℕ) :
  total_students = 180 →
  red_shirts = 83 →
  green_shirts = 97 →
  total_pairs = 90 →
  red_red_pairs = 35 →
  red_shirts + green_shirts = total_students →
  2 * total_pairs = total_students →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 42 ∧ 
    green_green_pairs + red_red_pairs + (green_shirts - 2 * green_green_pairs) = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l2851_285100


namespace NUMINAMATH_CALUDE_post_office_mail_handling_l2851_285117

/-- Represents the number of months required for a post office to handle a given amount of mail --/
def months_to_handle_mail (letters_per_day : ℕ) (packages_per_day : ℕ) (days_per_month : ℕ) (total_mail : ℕ) : ℕ :=
  total_mail / ((letters_per_day + packages_per_day) * days_per_month)

/-- Theorem stating that it takes 6 months to handle 14400 pieces of mail given the specified conditions --/
theorem post_office_mail_handling :
  months_to_handle_mail 60 20 30 14400 = 6 := by
  sorry

end NUMINAMATH_CALUDE_post_office_mail_handling_l2851_285117


namespace NUMINAMATH_CALUDE_alex_chocolates_l2851_285188

theorem alex_chocolates : 
  ∀ n : ℕ, n ≥ 150 ∧ n % 19 = 17 → n ≥ 150 := by
  sorry

end NUMINAMATH_CALUDE_alex_chocolates_l2851_285188


namespace NUMINAMATH_CALUDE_fractional_equation_elimination_l2851_285171

theorem fractional_equation_elimination (x : ℝ) : 
  (1 - (5*x + 2) / (x * (x + 1)) = 3 / (x + 1)) → 
  (x^2 - 7*x - 2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_elimination_l2851_285171


namespace NUMINAMATH_CALUDE_parallelogram_area_l2851_285198

/-- The area of a parallelogram with base 3.6 and height 2.5 times the base is 32.4 -/
theorem parallelogram_area : 
  let base : ℝ := 3.6
  let height : ℝ := 2.5 * base
  let area : ℝ := base * height
  area = 32.4 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2851_285198


namespace NUMINAMATH_CALUDE_unique_prime_solution_l2851_285125

theorem unique_prime_solution :
  ∀ p q : ℕ,
    Prime p → Prime q →
    p^3 - q^5 = (p + q)^2 →
    p = 7 ∧ q = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l2851_285125


namespace NUMINAMATH_CALUDE_first_18_even_numbers_average_l2851_285104

/-- The sequence of even numbers -/
def evenSequence : ℕ → ℕ
  | 0 => 2
  | n + 1 => evenSequence n + 2

/-- The sum of the first n terms in the even number sequence -/
def evenSum (n : ℕ) : ℕ :=
  (List.range n).map evenSequence |>.sum

/-- The average of the first n terms in the even number sequence -/
def evenAverage (n : ℕ) : ℚ :=
  evenSum n / n

theorem first_18_even_numbers_average :
  evenAverage 18 = 19 := by
  sorry

end NUMINAMATH_CALUDE_first_18_even_numbers_average_l2851_285104


namespace NUMINAMATH_CALUDE_not_all_zero_equiv_one_nonzero_l2851_285138

theorem not_all_zero_equiv_one_nonzero (a b c : ℝ) :
  (¬(a = 0 ∧ b = 0 ∧ c = 0)) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_not_all_zero_equiv_one_nonzero_l2851_285138


namespace NUMINAMATH_CALUDE_solve_equation_l2851_285148

theorem solve_equation : 
  ∃ x : ℝ, 3 + 2 * (8 - x) = 24.16 ∧ x = -2.58 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2851_285148


namespace NUMINAMATH_CALUDE_triangle_max_tan_diff_l2851_285170

open Real

theorem triangle_max_tan_diff (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a * cos B - b * cos A = c / 2 →
  (∀ θ, 0 < θ ∧ θ < π → tan (A - B) ≤ tan (A - θ)) →
  B = π / 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_tan_diff_l2851_285170


namespace NUMINAMATH_CALUDE_buying_goods_equations_l2851_285195

/-- Represents the problem of buying goods collectively --/
def BuyingGoods (x y : ℤ) : Prop :=
  (∃ (leftover : ℤ), 8 * x - y = leftover ∧ leftover = 3) ∧
  (∃ (shortage : ℤ), y - 7 * x = shortage ∧ shortage = 4)

/-- The correct system of equations for the buying goods problem --/
theorem buying_goods_equations (x y : ℤ) :
  BuyingGoods x y ↔ (8 * x - 3 = y ∧ 7 * x + 4 = y) :=
sorry

end NUMINAMATH_CALUDE_buying_goods_equations_l2851_285195


namespace NUMINAMATH_CALUDE_product_three_consecutive_odds_divisible_by_three_l2851_285152

theorem product_three_consecutive_odds_divisible_by_three (n : ℤ) (h : n > 0) :
  ∃ k : ℤ, (2*n + 1) * (2*n + 3) * (2*n + 5) = 3 * k :=
by
  sorry

end NUMINAMATH_CALUDE_product_three_consecutive_odds_divisible_by_three_l2851_285152


namespace NUMINAMATH_CALUDE_train_platform_length_l2851_285189

/-- The length of the platform given the conditions of the train problem -/
theorem train_platform_length 
  (train_speed : ℝ) 
  (opposite_train_speed : ℝ) 
  (crossing_time : ℝ) 
  (platform_passing_time : ℝ) 
  (h1 : train_speed = 48) 
  (h2 : opposite_train_speed = 42) 
  (h3 : crossing_time = 12) 
  (h4 : platform_passing_time = 45) : 
  ∃ (platform_length : ℝ), 
    (abs (platform_length - 600) < 1) ∧ 
    (platform_length = train_speed * (5/18) * platform_passing_time) :=
sorry


end NUMINAMATH_CALUDE_train_platform_length_l2851_285189


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2851_285113

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) → x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2851_285113


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2851_285172

/-- Given a complex number z and a real number a satisfying the equation (2+i)z = a+2i,
    where the real part of z is twice its imaginary part, prove that a = 3/2. -/
theorem complex_equation_solution (z : ℂ) (a : ℝ) 
    (h1 : (2 + Complex.I) * z = a + 2 * Complex.I)
    (h2 : z.re = 2 * z.im) : 
  a = 3/2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2851_285172


namespace NUMINAMATH_CALUDE_mixture_composition_l2851_285185

/-- Represents a solution with a given percentage of carbonated water -/
structure Solution :=
  (carbonated_water_percent : ℝ)
  (h_percent : 0 ≤ carbonated_water_percent ∧ carbonated_water_percent ≤ 1)

/-- Represents a mixture of two solutions -/
structure Mixture (P Q : Solution) :=
  (p_volume : ℝ)
  (q_volume : ℝ)
  (h_positive : 0 < p_volume ∧ 0 < q_volume)
  (carbonated_water_percent : ℝ)
  (h_mixture_percent : 0 ≤ carbonated_water_percent ∧ carbonated_water_percent ≤ 1)
  (h_balance : p_volume * P.carbonated_water_percent + q_volume * Q.carbonated_water_percent = 
               (p_volume + q_volume) * carbonated_water_percent)

/-- The main theorem to prove -/
theorem mixture_composition 
  (P : Solution) 
  (Q : Solution) 
  (mix : Mixture P Q) 
  (h_P : P.carbonated_water_percent = 0.8) 
  (h_Q : Q.carbonated_water_percent = 0.55) 
  (h_mix : mix.carbonated_water_percent = 0.6) : 
  mix.p_volume / (mix.p_volume + mix.q_volume) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_mixture_composition_l2851_285185


namespace NUMINAMATH_CALUDE_power_of_three_plus_five_mod_eight_l2851_285194

theorem power_of_three_plus_five_mod_eight : (3^101 + 5) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_plus_five_mod_eight_l2851_285194


namespace NUMINAMATH_CALUDE_assignment_count_correct_l2851_285120

/-- The number of ways to assign 5 students to 5 universities with exactly 2 visiting Peking University -/
def assignment_count : ℕ := 640

/-- The number of students -/
def num_students : ℕ := 5

/-- The number of universities -/
def num_universities : ℕ := 5

/-- The number of students who should visit Peking University -/
def peking_visitors : ℕ := 2

theorem assignment_count_correct : 
  assignment_count = (num_students.choose peking_visitors) * 
    (num_universities - 1) ^ (num_students - peking_visitors) := by
  sorry

end NUMINAMATH_CALUDE_assignment_count_correct_l2851_285120


namespace NUMINAMATH_CALUDE_triangle_heights_sum_ge_nine_times_inradius_l2851_285166

/-- Given a triangle with heights h₁, h₂, h₃ and an inscribed circle of radius r,
    the sum of the heights is greater than or equal to 9 times the radius. -/
theorem triangle_heights_sum_ge_nine_times_inradius 
  (h₁ h₂ h₃ r : ℝ) 
  (height_positive : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0)
  (inradius_positive : r > 0)
  (triangle_heights : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    h₁ = 2 * (a * b * c).sqrt / (a * (a + b + c)) ∧
    h₂ = 2 * (a * b * c).sqrt / (b * (a + b + c)) ∧
    h₃ = 2 * (a * b * c).sqrt / (c * (a + b + c)))
  (inradius_def : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    r = (a * b * c).sqrt / (a + b + c)) :
  h₁ + h₂ + h₃ ≥ 9 * r := by
  sorry

end NUMINAMATH_CALUDE_triangle_heights_sum_ge_nine_times_inradius_l2851_285166


namespace NUMINAMATH_CALUDE_problem_solution_l2851_285160

theorem problem_solution (x y z : ℤ) : 
  x = 12 → y = 18 → z = x - y → z * (x + y) = -180 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2851_285160


namespace NUMINAMATH_CALUDE_ben_peas_count_l2851_285159

/-- The number of sugar snap peas Ben wants to pick initially -/
def total_peas : ℕ := 56

/-- The time it takes Ben to pick all the peas (in minutes) -/
def total_time : ℕ := 7

/-- The number of peas Ben can pick in 9 minutes -/
def peas_in_9_min : ℕ := 72

/-- The time it takes Ben to pick 72 peas (in minutes) -/
def time_for_72_peas : ℕ := 9

/-- Theorem stating that the number of sugar snap peas Ben wants to pick initially is 56 -/
theorem ben_peas_count : 
  (total_peas : ℚ) / total_time = (peas_in_9_min : ℚ) / time_for_72_peas ∧
  total_peas = 56 := by
  sorry


end NUMINAMATH_CALUDE_ben_peas_count_l2851_285159


namespace NUMINAMATH_CALUDE_unique_natural_solution_l2851_285121

theorem unique_natural_solution : 
  ∃! (n : ℕ), n^5 - 2*n^4 - 7*n^2 - 7*n + 3 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_natural_solution_l2851_285121


namespace NUMINAMATH_CALUDE_power_of_power_three_l2851_285134

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l2851_285134


namespace NUMINAMATH_CALUDE_school_classes_count_l2851_285129

/-- Proves that the number of classes in a school is 1, given the conditions of the reading program -/
theorem school_classes_count (s : ℕ) (h1 : s > 0) : ∃ c : ℕ,
  (c * s = 1) ∧
  (6 * 12 * (c * s) = 72) :=
by
  sorry

#check school_classes_count

end NUMINAMATH_CALUDE_school_classes_count_l2851_285129


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l2851_285192

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≥ 1} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x ≥ |a - 4|) ↔ a ∈ Set.Icc (-1 : ℝ) 9 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l2851_285192


namespace NUMINAMATH_CALUDE_mryak_bryak_price_difference_l2851_285179

/-- The price of one "mryak" in rubles -/
def mryak_price : ℝ := sorry

/-- The price of one "bryak" in rubles -/
def bryak_price : ℝ := sorry

/-- Three "mryak" are 10 rubles more expensive than five "bryak" -/
axiom price_relation1 : 3 * mryak_price = 5 * bryak_price + 10

/-- Six "mryak" are 31 rubles more expensive than eight "bryak" -/
axiom price_relation2 : 6 * mryak_price = 8 * bryak_price + 31

/-- The price difference between seven "mryak" and nine "bryak" is 38 rubles -/
theorem mryak_bryak_price_difference : 7 * mryak_price - 9 * bryak_price = 38 := by
  sorry

end NUMINAMATH_CALUDE_mryak_bryak_price_difference_l2851_285179


namespace NUMINAMATH_CALUDE_trapezoid_bases_l2851_285101

/-- An isosceles trapezoid with the given properties -/
structure IsoscelesTrapezoid where
  -- The lengths of the two bases
  base1 : ℝ
  base2 : ℝ
  -- The length of the side
  side : ℝ
  -- The ratio of areas divided by the midline
  areaRatio : ℝ
  -- Conditions
  side_length : side = 3 ∨ side = 5
  inscribable : base1 + base2 = 2 * side
  area_ratio : areaRatio = 5 / 11

/-- The theorem stating the lengths of the bases -/
theorem trapezoid_bases (t : IsoscelesTrapezoid) : 
  (t.base1 = 1 ∧ t.base2 = 7) ∨ (t.base1 = 7 ∧ t.base2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_bases_l2851_285101


namespace NUMINAMATH_CALUDE_no_real_sqrt_negative_quadratic_l2851_285108

theorem no_real_sqrt_negative_quadratic :
  ¬ ∃ x : ℝ, ∃ y : ℝ, y^2 = -(x^2 + 2*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_no_real_sqrt_negative_quadratic_l2851_285108


namespace NUMINAMATH_CALUDE_counterexample_exists_l2851_285190

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 2)) ∧ n = 27 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2851_285190


namespace NUMINAMATH_CALUDE_extended_twelve_basketball_conference_games_l2851_285173

/-- Calculates the number of games in a basketball conference with specific rules --/
def conference_games (teams_per_division : ℕ) (divisions : ℕ) (intra_division_games : ℕ) : ℕ :=
  let total_teams := teams_per_division * divisions
  let games_per_team := (teams_per_division - 1) * intra_division_games + teams_per_division * (divisions - 1)
  total_teams * games_per_team / 2

/-- Theorem stating the number of games in the Extended Twelve Basketball Conference --/
theorem extended_twelve_basketball_conference_games :
  conference_games 8 2 3 = 232 := by
  sorry

end NUMINAMATH_CALUDE_extended_twelve_basketball_conference_games_l2851_285173


namespace NUMINAMATH_CALUDE_equal_intercept_line_perpendicular_line_l2851_285130

-- Define the point (2, 3)
def point : ℝ × ℝ := (2, 3)

-- Define the lines given in the problem
def line1 (x y : ℝ) : Prop := x - 2*y - 3 = 0
def line2 (x y : ℝ) : Prop := 2*x - 3*y - 2 = 0
def line3 (x y : ℝ) : Prop := 7*x + 5*y + 1 = 0

-- Define the concept of a line having equal intercepts
def has_equal_intercepts (a b c : ℝ) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c/a = c/b

-- Define perpendicularity of lines
def perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

-- Statement for the first part of the problem
theorem equal_intercept_line :
  ∃ (a b c : ℝ), (a * point.1 + b * point.2 + c = 0) ∧
  has_equal_intercepts a b c ∧
  ((a = 3 ∧ b = -2 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -5)) := by sorry

-- Statement for the second part of the problem
theorem perpendicular_line :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧
  ∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧
  perpendicular a b 7 5 ∧
  a = 5 ∧ b = -7 ∧ c = -3 := by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_perpendicular_line_l2851_285130


namespace NUMINAMATH_CALUDE_fib_100_mod_9_l2851_285168

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The 100th Fibonacci number is congruent to 3 modulo 9 -/
theorem fib_100_mod_9 : fib 100 % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fib_100_mod_9_l2851_285168


namespace NUMINAMATH_CALUDE_colored_triangle_existence_l2851_285118

-- Define the number of colors
def num_colors : ℕ := 1992

-- Define a type for colors
def Color := Fin num_colors

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for triangles
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define a function to check if a point is on a side of a triangle (excluding vertices)
def on_side (p : Point) (t : Triangle) : Prop := sorry

-- Define a function to check if a side of a triangle contains a point of a given color
def side_has_color (t : Triangle) (c : Color) : Prop := sorry

-- State the theorem
theorem colored_triangle_existence :
  (∀ c : Color, ∃ p : Point, coloring p = c) →
  ∀ T : Triangle, ∃ T' : Triangle,
    congruent T T' ∧
    ∃ c1 c2 c3 : Color,
      side_has_color T' c1 ∧
      side_has_color T' c2 ∧
      side_has_color T' c3 :=
by sorry

end NUMINAMATH_CALUDE_colored_triangle_existence_l2851_285118


namespace NUMINAMATH_CALUDE_binomial_18_4_l2851_285151

theorem binomial_18_4 : Nat.choose 18 4 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_4_l2851_285151


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l2851_285128

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℚ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l2851_285128


namespace NUMINAMATH_CALUDE_expression_evaluation_l2851_285164

theorem expression_evaluation : -20 + 12 * ((5 + 15) / 4) = 40 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2851_285164


namespace NUMINAMATH_CALUDE_book_sale_problem_l2851_285142

theorem book_sale_problem (cost_loss book_loss_price book_gain_price : ℝ) :
  cost_loss = 175 →
  book_loss_price = book_gain_price →
  book_loss_price = 0.85 * cost_loss →
  ∃ cost_gain : ℝ,
    book_gain_price = 1.19 * cost_gain ∧
    cost_loss + cost_gain = 300 :=
by sorry

end NUMINAMATH_CALUDE_book_sale_problem_l2851_285142


namespace NUMINAMATH_CALUDE_triangle_formation_check_l2851_285158

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of line segments -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(3, 4, 9), (50, 60, 12), (11, 11, 31), (20, 30, 50)]

theorem triangle_formation_check :
  ∃! set : ℝ × ℝ × ℝ, set ∈ segment_sets ∧ 
    let (a, b, c) := set
    can_form_triangle a b c :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_check_l2851_285158


namespace NUMINAMATH_CALUDE_inequalities_hold_l2851_285143

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ((a + b) * (1/a + 1/b) ≥ 4) ∧ 
  (a^2 + b^2 + 2 ≥ 2*a + 2*b) ∧ 
  (Real.sqrt (abs (a - b)) ≥ Real.sqrt a - Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_hold_l2851_285143


namespace NUMINAMATH_CALUDE_ellipse_and_line_problem_l2851_285180

-- Define the ellipse C
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define the line l
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the problem statement
theorem ellipse_and_line_problem 
  (C : Ellipse)
  (l₁ : Line)
  (l₂ : Line)
  (h₁ : l₁.slope = Real.sqrt 3)
  (h₂ : l₁.intercept = -2 * Real.sqrt 3)
  (h₃ : C.a^2 - C.b^2 = 4)
  (h₄ : (C.a^2 - C.b^2) / C.a^2 = 6 / 9)
  (h₅ : l₂.intercept = -3) :
  (C.a^2 = 6 ∧ C.b^2 = 2) ∧ 
  ((l₂.slope = Real.sqrt 3 ∧ l₂.intercept = -3) ∨ 
   (l₂.slope = -Real.sqrt 3 ∧ l₂.intercept = -3)) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_and_line_problem_l2851_285180


namespace NUMINAMATH_CALUDE_min_distance_between_sets_l2851_285122

/-- The minimum distance between a point on the set defined by y² - 3x² - 2xy - 9 - 12x = 0
    and a point on the set defined by x² - 8y + 23 + 6x + y² = 0 -/
theorem min_distance_between_sets :
  let set1 := {(x, y) : ℝ × ℝ | y^2 - 3*x^2 - 2*x*y - 9 - 12*x = 0}
  let set2 := {(x, y) : ℝ × ℝ | x^2 - 8*y + 23 + 6*x + y^2 = 0}
  ∃ (min_dist : ℝ), min_dist = (7 * Real.sqrt 10) / 10 - Real.sqrt 2 ∧
    ∀ (a : ℝ × ℝ) (b : ℝ × ℝ), a ∈ set1 → b ∈ set2 →
      Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_sets_l2851_285122


namespace NUMINAMATH_CALUDE_min_phi_value_l2851_285157

/-- Given a function f and a constant φ, this theorem proves that under certain conditions,
    the minimum value of φ is 5π/12. -/
theorem min_phi_value (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin (2 * x) * Real.cos (2 * φ) + Real.cos (2 * x) * Real.sin (2 * φ)) →
  φ > 0 →
  (∀ x, f x = f (2 * π / 3 - x)) →
  ∃ k : ℤ, φ = k * π / 2 - π / 12 ∧ 
  (∀ m : ℤ, m * π / 2 - π / 12 > 0 → φ ≤ m * π / 2 - π / 12) :=
sorry

end NUMINAMATH_CALUDE_min_phi_value_l2851_285157


namespace NUMINAMATH_CALUDE_max_min_quadratic_function_l2851_285119

theorem max_min_quadratic_function :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x - 2
  let interval : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
  (∀ x ∈ interval, f x ≤ -2) ∧
  (∃ x ∈ interval, f x = -2) ∧
  (∀ x ∈ interval, f x ≥ -6) ∧
  (∃ x ∈ interval, f x = -6) :=
by sorry

end NUMINAMATH_CALUDE_max_min_quadratic_function_l2851_285119


namespace NUMINAMATH_CALUDE_four_points_planes_l2851_285199

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A set of four points in space -/
def FourPoints : Type := Fin 4 → Point3D

/-- Three points are not collinear if they don't lie on the same line -/
def NotCollinear (p q r : Point3D) : Prop :=
  ∀ t : ℝ, (q.x - p.x, q.y - p.y, q.z - p.z) ≠ t • (r.x - p.x, r.y - p.y, r.z - p.z)

/-- The number of planes determined by any three points from a set of four points -/
def NumberOfPlanes (points : FourPoints) : ℕ :=
  sorry

/-- Theorem: Given four points in space where any three are not collinear,
    the number of planes determined by any three of these points is either 1 or 4 -/
theorem four_points_planes (points : FourPoints)
    (h : ∀ i j k : Fin 4, i ≠ j → j ≠ k → i ≠ k → NotCollinear (points i) (points j) (points k)) :
    NumberOfPlanes points = 1 ∨ NumberOfPlanes points = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_points_planes_l2851_285199


namespace NUMINAMATH_CALUDE_james_daily_trips_l2851_285105

/-- The number of bags James can carry per trip -/
def bags_per_trip : ℕ := 10

/-- The total number of bags James delivers in 5 days -/
def total_bags : ℕ := 1000

/-- The number of days James works -/
def total_days : ℕ := 5

/-- The number of trips James takes each day -/
def trips_per_day : ℕ := total_bags / (bags_per_trip * total_days)

theorem james_daily_trips : trips_per_day = 20 := by
  sorry

end NUMINAMATH_CALUDE_james_daily_trips_l2851_285105


namespace NUMINAMATH_CALUDE_factory_working_days_l2851_285197

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 4560

/-- The number of toys produced per day -/
def toys_per_day : ℕ := 1140

/-- The number of working days per week -/
def working_days : ℕ := toys_per_week / toys_per_day

theorem factory_working_days :
  working_days = 4 :=
sorry

end NUMINAMATH_CALUDE_factory_working_days_l2851_285197


namespace NUMINAMATH_CALUDE_marilyn_initial_caps_l2851_285177

/-- The number of bottle caps Marilyn has initially -/
def initial_caps : ℕ := sorry

/-- The number of bottle caps Nancy gives to Marilyn -/
def nancy_caps : ℕ := 36

/-- The total number of bottle caps Marilyn has after receiving Nancy's caps -/
def total_caps : ℕ := 87

/-- Theorem stating that Marilyn's initial number of bottle caps is 51 -/
theorem marilyn_initial_caps : 
  initial_caps + nancy_caps = total_caps → initial_caps = 51 := by sorry

end NUMINAMATH_CALUDE_marilyn_initial_caps_l2851_285177


namespace NUMINAMATH_CALUDE_correct_num_footballs_l2851_285140

/-- The number of footballs bought by the school gym -/
def num_footballs : ℕ := 22

/-- The number of basketballs bought by the school gym -/
def num_basketballs : ℕ := 6

/-- Theorem stating that the number of footballs is correct given the conditions -/
theorem correct_num_footballs : 
  (num_footballs = 3 * num_basketballs + 4) ∧ 
  (num_footballs = 4 * num_basketballs - 2) := by
  sorry

#check correct_num_footballs

end NUMINAMATH_CALUDE_correct_num_footballs_l2851_285140


namespace NUMINAMATH_CALUDE_twelve_people_circular_arrangements_l2851_285112

/-- The number of distinct circular arrangements of n people, considering rotational symmetry -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem: The number of distinct circular arrangements of 12 people, considering rotational symmetry, is equal to 11! -/
theorem twelve_people_circular_arrangements : 
  circularArrangements 12 = Nat.factorial 11 := by
  sorry

end NUMINAMATH_CALUDE_twelve_people_circular_arrangements_l2851_285112


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2851_285175

/-- Given two lines in the plane, this theorem states that if one line passes through 
    a specific point and is perpendicular to the other line, then it has a specific equation. -/
theorem perpendicular_line_equation 
  (l₁ : Real → Real → Prop) 
  (l₂ : Real → Real → Prop) 
  (h₁ : l₁ = fun x y ↦ 2 * x - 3 * y + 4 = 0) 
  (h₂ : l₂ = fun x y ↦ 3 * x + 2 * y - 1 = 0) : 
  (∀ x y, l₂ x y ↔ (x = -1 ∧ y = 2 ∨ 
    ∃ m : Real, m * (2 : Real) / 3 = -1 ∧ 
    y - 2 = m * (x + 1))) := by 
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2851_285175


namespace NUMINAMATH_CALUDE_red_in_B_equals_black_in_C_l2851_285161

-- Define the types for balls and boxes
inductive Color : Type
| Red : Color
| Black : Color

structure Box :=
  (red : Nat)
  (black : Nat)

-- Define the initial state
def initial_state (n : Nat) : Box × Box × Box :=
  ⟨⟨0, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩⟩

-- Define the process of distributing balls
def distribute_balls (n : Nat) : Box × Box × Box :=
  sorry

-- Theorem statement
theorem red_in_B_equals_black_in_C (n : Nat) (h : Even n) :
  let ⟨boxA, boxB, boxC⟩ := distribute_balls n
  boxB.red = boxC.black := by sorry

end NUMINAMATH_CALUDE_red_in_B_equals_black_in_C_l2851_285161


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2851_285136

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  8 * a^3 + 27 * b^3 + 125 * c^3 + 1 / (a * b * c) ≥ 10 * Real.sqrt 6 :=
by sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  8 * a^3 + 27 * b^3 + 125 * c^3 + 1 / (a * b * c) = 10 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2851_285136


namespace NUMINAMATH_CALUDE_max_plus_min_equals_13_l2851_285181

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem max_plus_min_equals_13 :
  ∃ (a b : ℝ), (∀ x ∈ domain, f x ≤ a) ∧
               (∀ x ∈ domain, b ≤ f x) ∧
               (∃ x₁ ∈ domain, f x₁ = a) ∧
               (∃ x₂ ∈ domain, f x₂ = b) ∧
               a + b = 13 :=
by sorry

end NUMINAMATH_CALUDE_max_plus_min_equals_13_l2851_285181


namespace NUMINAMATH_CALUDE_min_value_inequality_l2851_285103

theorem min_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b + c) (h2 : b + c ≤ 3 * a) (h3 : 3 * b^2 ≤ a * (a + c)) (h4 : a * (a + c) ≤ 5 * b^2) :
  ∃ (x : ℝ), ∀ (y : ℝ), (b - 2*c) / a ≥ x ∧ (b - 2*c) / a = x ↔ b / a = 4/5 ∧ c / a = 11/5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2851_285103


namespace NUMINAMATH_CALUDE_percentage_problem_l2851_285155

theorem percentage_problem (P : ℝ) (x : ℝ) : 
  x = 840 → P * x = 0.15 * 1500 - 15 → P = 0.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l2851_285155


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l2851_285153

theorem cow_chicken_problem (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 2 * (cows + chickens) + 16) → cows = 8 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l2851_285153


namespace NUMINAMATH_CALUDE_fraction_addition_l2851_285145

theorem fraction_addition (d : ℝ) : (5 + 4 * d) / 8 + 3 = (29 + 4 * d) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2851_285145


namespace NUMINAMATH_CALUDE_investment_problem_l2851_285186

/-- Proves that given the conditions of the investment problem, the invested sum is 15000 --/
theorem investment_problem (P : ℝ) 
  (h1 : P * (15 / 100) * 2 - P * (12 / 100) * 2 = 900) : 
  P = 15000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l2851_285186


namespace NUMINAMATH_CALUDE_increasing_sequence_condition_l2851_285163

theorem increasing_sequence_condition (a : ℝ) :
  (∀ n : ℕ+, (n : ℝ) - a < ((n + 1) : ℝ) - a) ↔ a < (3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_increasing_sequence_condition_l2851_285163


namespace NUMINAMATH_CALUDE_crayon_ratio_l2851_285107

def initial_crayons : ℕ := 18
def new_crayons : ℕ := 20
def total_crayons : ℕ := 29

theorem crayon_ratio :
  (initial_crayons - (total_crayons - new_crayons)) * 2 = initial_crayons :=
sorry

end NUMINAMATH_CALUDE_crayon_ratio_l2851_285107


namespace NUMINAMATH_CALUDE_playground_area_l2851_285176

/-- Proves that a rectangular playground with given conditions has an area of 29343.75 square feet -/
theorem playground_area : 
  ∀ (width length : ℝ),
  length = 3 * width + 40 →
  2 * (width + length) = 820 →
  width * length = 29343.75 := by
sorry

end NUMINAMATH_CALUDE_playground_area_l2851_285176


namespace NUMINAMATH_CALUDE_line_properties_l2851_285165

-- Define the line l
def line_l (x y : ℝ) : Prop := 2 * x + y + 1 = 0

-- Define points A and P
def point_A : ℝ × ℝ := (3, 2)
def point_P : ℝ × ℝ := (3, 0)

-- Define the perpendicular line l₁
def line_l1 (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Define the parallel lines l₂
def line_l2_1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def line_l2_2 (x y : ℝ) : Prop := 2 * x + y - 11 = 0

-- Theorem statement
theorem line_properties :
  (∀ x y : ℝ, line_l1 x y ↔ (x = point_A.1 ∧ y = point_A.2) ∨ 
    (∃ k : ℝ, x = point_A.1 + k ∧ y = point_A.2 - k/2)) ∧
  (∀ x y : ℝ, (line_l2_1 x y ∨ line_l2_2 x y) ↔
    (∃ k : ℝ, x = k ∧ y = -2*k + 1) ∧
    (|2 * point_P.1 + point_P.2 + 1| / Real.sqrt 5 = Real.sqrt 5 ∨
     |2 * point_P.1 + point_P.2 + 11| / Real.sqrt 5 = Real.sqrt 5)) :=
by sorry


end NUMINAMATH_CALUDE_line_properties_l2851_285165


namespace NUMINAMATH_CALUDE_exists_non_isosceles_with_isosceles_bisector_base_l2851_285133

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define an angle bisector
def AngleBisector (T : Triangle) (vertex : ℕ) : ℝ × ℝ → ℝ × ℝ := sorry

-- Define the base of an angle bisector
def BaseBisector (T : Triangle) (vertex : ℕ) : ℝ × ℝ := sorry

-- Define isosceles property for a triangle
def IsIsosceles (T : Triangle) : Prop := sorry

-- Define the triangle formed by the bases of angle bisectors
def BisectorBaseTriangle (T : Triangle) : Triangle :=
  { A := BaseBisector T 0,
    B := BaseBisector T 1,
    C := BaseBisector T 2 }

theorem exists_non_isosceles_with_isosceles_bisector_base :
  ∃ T : Triangle,
    IsIsosceles (BisectorBaseTriangle T) ∧
    ¬IsIsosceles T :=
  sorry

end NUMINAMATH_CALUDE_exists_non_isosceles_with_isosceles_bisector_base_l2851_285133


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2851_285124

theorem complex_fraction_simplification : 
  (((12^4 + 484) * (24^4 + 484) * (36^4 + 484) * (48^4 + 484) * (60^4 + 484)) : ℚ) /
  ((6^4 + 484) * (18^4 + 484) * (30^4 + 484) * (42^4 + 484) * (54^4 + 484)) = 181 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2851_285124


namespace NUMINAMATH_CALUDE_carrot_consumption_theorem_l2851_285162

theorem carrot_consumption_theorem :
  ∃ (x y z : ℕ), x + y + z = 15 ∧ z % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_carrot_consumption_theorem_l2851_285162


namespace NUMINAMATH_CALUDE_price_reduction_equation_l2851_285126

/-- Represents the price reduction scenario for a medicine -/
structure PriceReduction where
  original_price : ℝ
  final_price : ℝ
  reduction_percentage : ℝ

/-- Theorem stating the relationship between original price, final price, and reduction percentage -/
theorem price_reduction_equation (pr : PriceReduction) 
  (h1 : pr.original_price = 25)
  (h2 : pr.final_price = 16)
  : pr.original_price * (1 - pr.reduction_percentage)^2 = pr.final_price := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l2851_285126


namespace NUMINAMATH_CALUDE_driver_net_pay_driver_net_pay_result_l2851_285141

/-- Calculate the net rate of pay for a driver given specific conditions --/
theorem driver_net_pay (travel_time : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) 
  (earnings_per_mile : ℝ) (gas_price : ℝ) : ℝ :=
  let total_distance := travel_time * speed
  let gas_used := total_distance / fuel_efficiency
  let total_earnings := earnings_per_mile * total_distance
  let gas_cost := gas_price * gas_used
  let net_earnings := total_earnings - gas_cost
  let net_rate := net_earnings / travel_time
  net_rate

/-- The driver's net rate of pay is $39.75 per hour --/
theorem driver_net_pay_result : 
  driver_net_pay 3 75 25 0.65 3 = 39.75 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_driver_net_pay_result_l2851_285141


namespace NUMINAMATH_CALUDE_triangle_altitude_equals_twice_base_l2851_285167

/-- Given a square with side length x and a triangle with base x, 
    if their areas are equal, then the altitude of the triangle is 2x. -/
theorem triangle_altitude_equals_twice_base (x : ℝ) (h : x > 0) : 
  x^2 = (1/2) * x * (2*x) := by sorry

end NUMINAMATH_CALUDE_triangle_altitude_equals_twice_base_l2851_285167


namespace NUMINAMATH_CALUDE_inequality_of_reciprocals_l2851_285137

theorem inequality_of_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y * z) + y / (z * x) + z / (x * y) ≥ 1 / x + 1 / y + 1 / z := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_reciprocals_l2851_285137


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l2851_285106

theorem sqrt_expression_equality : 
  Real.sqrt 12 - Real.sqrt 2 * (Real.sqrt 8 - 3 * Real.sqrt (1/2)) = 2 * Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l2851_285106


namespace NUMINAMATH_CALUDE_lcm_1364_884_minus_100_l2851_285183

def lcm_minus_100 (a b : Nat) : Nat :=
  Nat.lcm a b - 100

theorem lcm_1364_884_minus_100 :
  lcm_minus_100 1364 884 = 1509692 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1364_884_minus_100_l2851_285183


namespace NUMINAMATH_CALUDE_fraction_sum_equals_61_30_l2851_285132

theorem fraction_sum_equals_61_30 :
  (3 + 6 + 9) / (2 + 5 + 8) + (2 + 5 + 8) / (3 + 6 + 9) = 61 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_61_30_l2851_285132


namespace NUMINAMATH_CALUDE_book_loss_percentage_l2851_285154

/-- If the cost price of 8 books equals the selling price of 16 books, then the loss percentage is 50% -/
theorem book_loss_percentage (C S : ℝ) (h : 8 * C = 16 * S) : (C - S) / C * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_loss_percentage_l2851_285154


namespace NUMINAMATH_CALUDE_foot_to_total_distance_ratio_l2851_285150

/-- Proves that the ratio of distance traveled by foot to total distance is 1:4 -/
theorem foot_to_total_distance_ratio :
  let total_distance : ℝ := 40
  let bus_distance : ℝ := total_distance / 2
  let car_distance : ℝ := 10
  let foot_distance : ℝ := total_distance - bus_distance - car_distance
  foot_distance / total_distance = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_foot_to_total_distance_ratio_l2851_285150


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l2851_285111

theorem cubic_polynomials_common_roots (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
    r^3 + a*r^2 + 10*r + 3 = 0 ∧
    r^3 + b*r^2 + 21*r + 12 = 0 ∧
    s^3 + a*s^2 + 10*s + 3 = 0 ∧
    s^3 + b*s^2 + 21*s + 12 = 0) →
  a = 9 ∧ b = 10 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l2851_285111


namespace NUMINAMATH_CALUDE_least_cars_serviced_per_day_l2851_285184

/-- The number of cars that can be serviced in a workday by two mechanics -/
def cars_serviced_per_day (hours_per_day : ℕ) (rate1 : ℕ) (rate2 : ℕ) : ℕ :=
  (rate1 + rate2) * hours_per_day

/-- Theorem stating the least number of cars that can be serviced by Paul and Jack in a workday -/
theorem least_cars_serviced_per_day :
  cars_serviced_per_day 8 2 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_least_cars_serviced_per_day_l2851_285184


namespace NUMINAMATH_CALUDE_nines_in_sixty_houses_l2851_285109

def count_nines (n : ℕ) : ℕ :=
  (n + 10) / 10

theorem nines_in_sixty_houses :
  count_nines 60 = 6 := by
  sorry

end NUMINAMATH_CALUDE_nines_in_sixty_houses_l2851_285109


namespace NUMINAMATH_CALUDE_modulus_of_2_minus_i_l2851_285131

theorem modulus_of_2_minus_i :
  let z : ℂ := 2 - I
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_2_minus_i_l2851_285131


namespace NUMINAMATH_CALUDE_gcd_204_85_l2851_285178

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l2851_285178


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l2851_285116

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

-- State the theorem
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a : ℝ, a ∈ N → a ∈ M) ∧ (∃ a : ℝ, a ∈ M ∧ a ∉ N) :=
by sorry

end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l2851_285116


namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l2851_285127

theorem square_difference_divided_by_nine : (104^2 - 95^2) / 9 = 199 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l2851_285127


namespace NUMINAMATH_CALUDE_pencils_per_row_indeterminate_l2851_285146

theorem pencils_per_row_indeterminate (rows : ℕ) (crayons_per_row : ℕ) (total_crayons : ℕ) :
  rows = 7 →
  crayons_per_row = 30 →
  total_crayons = 210 →
  ∀ (pencils_per_row : ℕ), ∃ (total_pencils : ℕ),
    total_pencils = rows * pencils_per_row :=
by sorry

end NUMINAMATH_CALUDE_pencils_per_row_indeterminate_l2851_285146


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_composable_l2851_285114

/-- A fourth-degree polynomial -/
structure FourthDegreePolynomial where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  A_nonzero : A ≠ 0

/-- Condition for a fourth-degree polynomial to be expressible as a composition of two quadratic polynomials -/
def is_composable (f : FourthDegreePolynomial) : Prop :=
  f.D = (f.B * f.C) / (2 * f.A) - (f.B^3) / (8 * f.A^2)

/-- Theorem stating the necessary and sufficient condition for a fourth-degree polynomial 
    to be expressible as a composition of two quadratic polynomials -/
theorem fourth_degree_polynomial_composable (f : FourthDegreePolynomial) :
  (∃ (p q : ℝ → ℝ), (∀ x, f.A * x^4 + f.B * x^3 + f.C * x^2 + f.D * x + f.E = p (q x)) ∧
                     (∃ a b c r s t, p x = a * x^2 + b * x + c ∧
                                     q x = r * x^2 + s * x + t)) ↔
  is_composable f :=
sorry

end NUMINAMATH_CALUDE_fourth_degree_polynomial_composable_l2851_285114


namespace NUMINAMATH_CALUDE_percentage_of_sheet_used_for_typing_l2851_285191

/-- Calculates the percentage of a rectangular sheet used for typing, given its dimensions and margins. -/
theorem percentage_of_sheet_used_for_typing 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (side_margin : ℝ) 
  (top_bottom_margin : ℝ) 
  (h1 : sheet_length = 30)
  (h2 : sheet_width = 20)
  (h3 : side_margin = 2)
  (h4 : top_bottom_margin = 3)
  : (((sheet_width - 2 * side_margin) * (sheet_length - 2 * top_bottom_margin)) / (sheet_width * sheet_length)) * 100 = 64 := by
  sorry

#check percentage_of_sheet_used_for_typing

end NUMINAMATH_CALUDE_percentage_of_sheet_used_for_typing_l2851_285191


namespace NUMINAMATH_CALUDE_town_population_l2851_285174

theorem town_population (increase_rate : ℝ) (future_population : ℕ) :
  increase_rate = 0.1 →
  future_population = 242 →
  ∃ present_population : ℕ,
    present_population * (1 + increase_rate) = future_population ∧
    present_population = 220 := by
  sorry

end NUMINAMATH_CALUDE_town_population_l2851_285174


namespace NUMINAMATH_CALUDE_sum_of_ages_is_fifty_l2851_285193

/-- The sum of ages of 5 children born at intervals of 3 years -/
def sum_of_ages (youngest_age : ℕ) (interval : ℕ) (num_children : ℕ) : ℕ :=
  let ages := List.range num_children
  ages.map (fun i => youngest_age + i * interval) |> List.sum

/-- Theorem stating the sum of ages for the given conditions -/
theorem sum_of_ages_is_fifty :
  sum_of_ages 4 3 5 = 50 := by
  sorry

#eval sum_of_ages 4 3 5

end NUMINAMATH_CALUDE_sum_of_ages_is_fifty_l2851_285193


namespace NUMINAMATH_CALUDE_ellipse_equation_and_chord_length_l2851_285187

noncomputable section

-- Define the ellipse C₁
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola C₂
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Define the left endpoint of the ellipse
def left_endpoint : ℝ × ℝ := (-Real.sqrt 6, 0)

-- Define the line l₂
def line_l2 (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - 2)

theorem ellipse_equation_and_chord_length 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ellipse a b (Prod.fst parabola_focus) (Prod.snd parabola_focus))
  (h4 : ellipse a b (Prod.fst left_endpoint) (Prod.snd left_endpoint)) :
  (∀ x y, ellipse a b x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
  (∃ A B : ℝ × ℝ, 
    ellipse a b A.1 A.2 ∧ 
    ellipse a b B.1 B.2 ∧
    line_l2 A.1 A.2 ∧
    line_l2 B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 6 / 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_chord_length_l2851_285187


namespace NUMINAMATH_CALUDE_distance_calculation_l2851_285110

theorem distance_calculation (train_speed ship_speed : ℝ) (time_difference : ℝ) (distance : ℝ) : 
  train_speed = 48 →
  ship_speed = 60 →
  time_difference = 2 →
  distance / train_speed = distance / ship_speed + time_difference →
  distance = 480 := by
sorry

end NUMINAMATH_CALUDE_distance_calculation_l2851_285110


namespace NUMINAMATH_CALUDE_trigonometric_roots_problem_l2851_285144

open Real

theorem trigonometric_roots_problem (α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : (tan α)^2 - 5*(tan α) + 6 = 0) (h4 : (tan β)^2 - 5*(tan β) + 6 = 0) :
  (α + β = 3*π/4) ∧ (¬ ∃ (x : ℝ), tan (2*(α + β)) = x) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_roots_problem_l2851_285144


namespace NUMINAMATH_CALUDE_first_class_product_rate_l2851_285135

/-- Given a product with a pass rate and a rate of first-class products among qualified products,
    calculate the overall rate of first-class products. -/
theorem first_class_product_rate
  (pass_rate : ℝ)
  (first_class_rate_qualified : ℝ)
  (h1 : pass_rate = 0.95)
  (h2 : first_class_rate_qualified = 0.2) :
  pass_rate * first_class_rate_qualified = 0.19 :=
by sorry

end NUMINAMATH_CALUDE_first_class_product_rate_l2851_285135


namespace NUMINAMATH_CALUDE_minimum_fourth_quarter_score_l2851_285147

def required_average : ℝ := 85
def num_quarters : ℕ := 4
def first_quarter_score : ℝ := 84
def second_quarter_score : ℝ := 80
def third_quarter_score : ℝ := 78

theorem minimum_fourth_quarter_score :
  let total_required := required_average * num_quarters
  let current_total := first_quarter_score + second_quarter_score + third_quarter_score
  let minimum_score := total_required - current_total
  minimum_score = 98 := by sorry

end NUMINAMATH_CALUDE_minimum_fourth_quarter_score_l2851_285147


namespace NUMINAMATH_CALUDE_initial_money_calculation_l2851_285149

theorem initial_money_calculation (initial_amount : ℚ) : 
  (2/5 : ℚ) * initial_amount = 600 → initial_amount = 1500 := by
  sorry

#check initial_money_calculation

end NUMINAMATH_CALUDE_initial_money_calculation_l2851_285149


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2851_285123

/-- The number of games in a chess tournament where each player plays twice against every other player. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 19 players, where each player plays twice against every other player, the total number of games played is 684. -/
theorem chess_tournament_games :
  tournament_games 19 = 342 ∧ 2 * tournament_games 19 = 684 := by
  sorry

#eval 2 * tournament_games 19

end NUMINAMATH_CALUDE_chess_tournament_games_l2851_285123


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2851_285115

theorem negation_of_proposition (p : Prop) :
  (¬ (∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2851_285115


namespace NUMINAMATH_CALUDE_circle_passes_through_intersection_point_l2851_285169

-- Define the two lines
def line1 (x y : ℝ) : Prop := x + 2*y + 1 = 0
def line2 (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (4, 3)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 25

-- Theorem statement
theorem circle_passes_through_intersection_point :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ circle_equation x y :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_intersection_point_l2851_285169


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2851_285196

theorem quadratic_function_property (a m : ℝ) (h_a : a > 0) : 
  let f := fun (x : ℝ) ↦ x^2 + x + a
  f m < 0 → f (m + 1) > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2851_285196
