import Mathlib

namespace NUMINAMATH_CALUDE_journey_fraction_l2922_292216

theorem journey_fraction (total_distance : ℝ) (bus_fraction : ℚ) (foot_distance : ℝ) :
  total_distance = 130 →
  bus_fraction = 17 / 20 →
  foot_distance = 6.5 →
  ∃ rail_fraction : ℚ,
    rail_fraction + bus_fraction + (foot_distance / total_distance) = 1 ∧
    rail_fraction = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_journey_fraction_l2922_292216


namespace NUMINAMATH_CALUDE_order_of_6_wrt_f_l2922_292275

def f (x : ℕ) : ℕ := x^2 % 13

def iterateF (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => f (iterateF n x)

theorem order_of_6_wrt_f :
  ∀ k : ℕ, k > 0 → k < 36 → iterateF k 6 ≠ 6 ∧ iterateF 36 6 = 6 := by sorry

end NUMINAMATH_CALUDE_order_of_6_wrt_f_l2922_292275


namespace NUMINAMATH_CALUDE_xy_range_l2922_292279

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 2/x + 3*y + 4/y = 10) : 
  1 ≤ x*y ∧ x*y ≤ 8/3 := by
sorry

end NUMINAMATH_CALUDE_xy_range_l2922_292279


namespace NUMINAMATH_CALUDE_final_result_l2922_292203

def program_result : ℕ → ℕ → ℕ
| 0, s => s
| (n+1), s => program_result n (s * (11 - n))

theorem final_result : program_result 3 1 = 990 := by
  sorry

#eval program_result 3 1

end NUMINAMATH_CALUDE_final_result_l2922_292203


namespace NUMINAMATH_CALUDE_salt_solution_percentage_l2922_292239

def is_valid_salt_solution (initial_salt_percent : ℝ) : Prop :=
  let replaced_volume : ℝ := 1/4
  let final_salt_percent : ℝ := 16
  let replacing_salt_percent : ℝ := 31
  (1 - replaced_volume) * initial_salt_percent + replaced_volume * replacing_salt_percent = final_salt_percent

theorem salt_solution_percentage :
  ∃ (x : ℝ), is_valid_salt_solution x ∧ x = 11 :=
sorry

end NUMINAMATH_CALUDE_salt_solution_percentage_l2922_292239


namespace NUMINAMATH_CALUDE_frustum_radius_l2922_292257

theorem frustum_radius (r : ℝ) :
  (r > 0) →
  (3 * (2 * π * r) = 2 * π * (3 * r)) →
  (π * (r + 3 * r) * 3 = 84 * π) →
  r = 7 := by sorry

end NUMINAMATH_CALUDE_frustum_radius_l2922_292257


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_4_area_is_8_l2922_292236

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (Real.sqrt 2 * t.b - t.c) / t.a = Real.cos t.C / Real.cos t.A

-- Theorem 1: Angle A is π/4
theorem angle_A_is_pi_over_4 (t : Triangle) (h : satisfiesCondition t) : t.A = π / 4 := by
  sorry

-- Theorem 2: Area of the triangle is 8 under specific conditions
theorem area_is_8 (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.a = 10) (h3 : t.b = 8 * Real.sqrt 2) 
  (h4 : t.C < t.A ∧ t.C < t.B) : 
  (1 / 2) * t.a * t.b * Real.sin t.C = 8 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_4_area_is_8_l2922_292236


namespace NUMINAMATH_CALUDE_star_polygon_n_value_l2922_292296

/-- Represents an n-pointed regular star polygon -/
structure RegularStarPolygon where
  n : ℕ
  angle_A : ℝ
  angle_B : ℝ

/-- Properties of the regular star polygon -/
def is_valid_star_polygon (star : RegularStarPolygon) : Prop :=
  star.n > 0 ∧
  star.angle_A > 0 ∧
  star.angle_B > 0 ∧
  star.angle_A = star.angle_B - 15 ∧
  star.n * (star.angle_A + star.angle_B) = 360

theorem star_polygon_n_value (star : RegularStarPolygon) 
  (h : is_valid_star_polygon star) : star.n = 24 :=
by sorry

end NUMINAMATH_CALUDE_star_polygon_n_value_l2922_292296


namespace NUMINAMATH_CALUDE_factorization_problems_l2922_292235

theorem factorization_problems (a x y : ℝ) : 
  (a * (a - 2) + 2 * (a - 2) = (a - 2) * (a + 2)) ∧ 
  (3 * x^2 - 6 * x * y + 3 * y^2 = 3 * (x - y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l2922_292235


namespace NUMINAMATH_CALUDE_vanessa_score_l2922_292234

/-- Vanessa's basketball game score calculation -/
theorem vanessa_score (total_score : ℕ) (other_players : ℕ) (other_avg : ℕ) : 
  total_score = 65 → other_players = 7 → other_avg = 5 → 
  total_score - (other_players * other_avg) = 30 := by
sorry

end NUMINAMATH_CALUDE_vanessa_score_l2922_292234


namespace NUMINAMATH_CALUDE_five_boys_three_girls_arrangements_l2922_292204

/-- The number of arrangements of boys and girls in a row -/
def arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  (Nat.factorial (num_boys + 1)) * (Nat.factorial num_girls)

/-- Theorem stating the number of arrangements for 5 boys and 3 girls -/
theorem five_boys_three_girls_arrangements :
  arrangements 5 3 = 4320 := by
  sorry

end NUMINAMATH_CALUDE_five_boys_three_girls_arrangements_l2922_292204


namespace NUMINAMATH_CALUDE_fifth_power_last_digit_l2922_292224

theorem fifth_power_last_digit (n : ℕ) : 10 ∣ (n^5 - n) := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_last_digit_l2922_292224


namespace NUMINAMATH_CALUDE_right_triangle_circles_coincide_l2922_292260

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle_at_B : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the circles
def circle_BC (B C D : ℝ × ℝ) : Prop :=
  (D.1 - B.1) * (C.1 - D.1) + (D.2 - B.2) * (C.2 - D.2) = 0

def circle_AB (A B E : ℝ × ℝ) : Prop :=
  (E.1 - A.1) * (B.1 - E.1) + (E.2 - A.2) * (B.2 - E.2) = 0

-- Define the theorem
theorem right_triangle_circles_coincide 
  (A B C D E : ℝ × ℝ) 
  (h_triangle : Triangle A B C) 
  (h_circle_BC : circle_BC B C D) 
  (h_circle_AB : circle_AB A B E) 
  (h_area : abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 = 200)
  (h_AC : ((C.1 - A.1)^2 + (C.2 - A.2)^2).sqrt = 40) :
  D = E := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_circles_coincide_l2922_292260


namespace NUMINAMATH_CALUDE_solution_value_l2922_292252

-- Define the solution sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the parameters a and b
variable (a b : ℝ)

-- State the theorem
theorem solution_value (h : {x | a*x^2 + b*x + 2 > 0} = A ∩ B) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2922_292252


namespace NUMINAMATH_CALUDE_positive_root_m_value_l2922_292289

theorem positive_root_m_value (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1 - x) / (x - 2) = m / (2 - x) - 2) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_positive_root_m_value_l2922_292289


namespace NUMINAMATH_CALUDE_crackers_per_friend_l2922_292254

theorem crackers_per_friend (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) : 
  total_crackers = 36 →
  num_friends = 18 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 2 := by
sorry

end NUMINAMATH_CALUDE_crackers_per_friend_l2922_292254


namespace NUMINAMATH_CALUDE_banana_bread_flour_calculation_hannahs_banana_bread_flour_l2922_292297

/-- Given the ratio of flour to banana mush, bananas per cup of mush, and total bananas used,
    calculate the number of cups of flour needed. -/
theorem banana_bread_flour_calculation 
  (flour_to_mush_ratio : ℚ) 
  (bananas_per_mush : ℕ) 
  (total_bananas : ℕ) : ℚ :=
  by
  sorry

/-- Prove that for Hannah's banana bread recipe, she needs 15 cups of flour. -/
theorem hannahs_banana_bread_flour : 
  banana_bread_flour_calculation 3 4 20 = 15 :=
  by
  sorry

end NUMINAMATH_CALUDE_banana_bread_flour_calculation_hannahs_banana_bread_flour_l2922_292297


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2922_292277

theorem polynomial_divisibility (a b c d m : ℤ) 
  (h1 : (5 : ℤ) ∣ (a * m^3 + b * m^2 + c * m + d))
  (h2 : ¬((5 : ℤ) ∣ d)) :
  ∃ n : ℤ, (5 : ℤ) ∣ (d * n^3 + c * n^2 + b * n + a) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2922_292277


namespace NUMINAMATH_CALUDE_article_pricing_gain_percent_l2922_292229

/-- Proves that if the cost price of 50 articles equals the selling price of 35 articles,
    then the gain percent is 300/7. -/
theorem article_pricing_gain_percent
  (C : ℝ) -- Cost price of one article
  (S : ℝ) -- Selling price of one article
  (h : 50 * C = 35 * S) -- Given condition
  : (S - C) / C * 100 = 300 / 7 := by
  sorry

end NUMINAMATH_CALUDE_article_pricing_gain_percent_l2922_292229


namespace NUMINAMATH_CALUDE_coprime_linear_combination_l2922_292227

theorem coprime_linear_combination (m n : ℕ+) (h : Nat.Coprime m n) :
  ∃ N : ℕ, ∀ k : ℕ, k ≥ N → ∃ a b : ℕ, k = a * m + b * n ∧
  (∀ N' : ℕ, (∀ k : ℕ, k ≥ N' → ∃ a b : ℕ, k = a * m + b * n) → N' ≥ N) ∧
  N = m * n - m - n + 1 :=
sorry

end NUMINAMATH_CALUDE_coprime_linear_combination_l2922_292227


namespace NUMINAMATH_CALUDE_not_divisible_by_59_l2922_292226

theorem not_divisible_by_59 (x y : ℕ) 
  (h1 : ¬ 59 ∣ x) 
  (h2 : ¬ 59 ∣ y) 
  (h3 : 59 ∣ (3 * x + 28 * y)) : 
  ¬ 59 ∣ (5 * x + 16 * y) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_59_l2922_292226


namespace NUMINAMATH_CALUDE_relationship_abc_l2922_292220

theorem relationship_abc : ∀ a b c : ℝ,
  a = -1/3 * 9 →
  b = 2 - 4 →
  c = 2 / (-1/2) →
  c < a ∧ a < b :=
by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l2922_292220


namespace NUMINAMATH_CALUDE_work_completion_time_l2922_292230

theorem work_completion_time 
  (john_time : ℝ) 
  (rose_time : ℝ) 
  (dave_time : ℝ) 
  (h1 : john_time = 8) 
  (h2 : rose_time = 16) 
  (h3 : dave_time = 12) : 
  (1 / (1 / john_time + 1 / rose_time + 1 / dave_time)) = 48 / 13 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l2922_292230


namespace NUMINAMATH_CALUDE_race_first_part_length_l2922_292270

theorem race_first_part_length 
  (total_length : ℝ)
  (second_part : ℝ)
  (third_part : ℝ)
  (last_part : ℝ)
  (h1 : total_length = 74.5)
  (h2 : second_part = 21.5)
  (h3 : third_part = 21.5)
  (h4 : last_part = 16) :
  total_length - (second_part + third_part + last_part) = 15.5 := by
sorry

end NUMINAMATH_CALUDE_race_first_part_length_l2922_292270


namespace NUMINAMATH_CALUDE_cake_muffin_probability_l2922_292223

theorem cake_muffin_probability (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) 
  (h_total : total = 100)
  (h_cake : cake = 50)
  (h_muffin : muffin = 40)
  (h_both : both = 16) :
  (total - (cake + muffin - both)) / total = 26 / 100 := by
sorry

end NUMINAMATH_CALUDE_cake_muffin_probability_l2922_292223


namespace NUMINAMATH_CALUDE_probability_three_different_suits_l2922_292215

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := StandardDeck / NumberOfSuits

/-- The probability of selecting three cards of different suits from a standard deck without replacement -/
theorem probability_three_different_suits : 
  (39 : ℚ) / 51 * 24 / 50 = 156 / 425 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_different_suits_l2922_292215


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l2922_292251

/-- If 9x^2 - 24x + c is a perfect square of a binomial, then c = 16 -/
theorem quadratic_perfect_square (c : ℝ) : 
  (∃ a b : ℝ, ∀ x, 9*x^2 - 24*x + c = (a*x + b)^2) → c = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l2922_292251


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l2922_292248

def U : Set ℕ := {x | 0 < x ∧ x ≤ 6}
def M : Set ℕ := {1, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement_equals : M ∩ (U \ N) = {1, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l2922_292248


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2922_292233

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := 4 / (1 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2922_292233


namespace NUMINAMATH_CALUDE_light_bulbs_not_broken_l2922_292262

/-- The number of light bulbs not broken in both the foyer and kitchen -/
def num_not_broken (kitchen_total : ℕ) (foyer_broken : ℕ) : ℕ :=
  let kitchen_broken := (3 * kitchen_total) / 5
  let kitchen_not_broken := kitchen_total - kitchen_broken
  let foyer_total := foyer_broken * 3
  let foyer_not_broken := foyer_total - foyer_broken
  kitchen_not_broken + foyer_not_broken

/-- Theorem stating that the number of light bulbs not broken in both the foyer and kitchen is 34 -/
theorem light_bulbs_not_broken :
  num_not_broken 35 10 = 34 := by
  sorry

end NUMINAMATH_CALUDE_light_bulbs_not_broken_l2922_292262


namespace NUMINAMATH_CALUDE_cube_root_of_three_times_two_to_fifth_l2922_292214

theorem cube_root_of_three_times_two_to_fifth (x : ℝ) : 
  x^3 = 2^5 + 2^5 + 2^5 → x = 6 * 6^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_three_times_two_to_fifth_l2922_292214


namespace NUMINAMATH_CALUDE_smallest_area_three_interior_points_l2922_292273

/-- A square with diagonals aligned with coordinate axes -/
structure AlignedSquare where
  side : ℝ
  center : ℝ × ℝ

/-- Count of interior lattice points in a square -/
def interiorLatticePoints (s : AlignedSquare) : ℕ := sorry

/-- The area of an AlignedSquare -/
def area (s : AlignedSquare) : ℝ := s.side * s.side

/-- Theorem: Smallest area of an AlignedSquare with exactly three interior lattice points is 8 -/
theorem smallest_area_three_interior_points :
  ∃ (s : AlignedSquare), 
    interiorLatticePoints s = 3 ∧ 
    area s = 8 ∧
    ∀ (t : AlignedSquare), interiorLatticePoints t = 3 → area t ≥ 8 := by sorry

end NUMINAMATH_CALUDE_smallest_area_three_interior_points_l2922_292273


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2922_292265

theorem solve_linear_equation (x y : ℝ) :
  2 * x + y = 5 → x = (5 - y) / 2 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2922_292265


namespace NUMINAMATH_CALUDE_rectangle_diagonal_squares_l2922_292259

/-- The number of unit squares that the diagonals of a rectangle pass through -/
def diagonalSquares (width : ℕ) (height : ℕ) : ℕ :=
  2 * (width - 1 + height - 1 + 1) - 2

/-- Theorem: For a 20 × 19 rectangle with one corner at the origin and sides parallel to the coordinate axes,
    the number of unit squares that the two diagonals pass through is 74. -/
theorem rectangle_diagonal_squares :
  diagonalSquares 20 19 = 74 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_squares_l2922_292259


namespace NUMINAMATH_CALUDE_vanilla_cookie_price_l2922_292249

theorem vanilla_cookie_price 
  (chocolate_count : ℕ) 
  (vanilla_count : ℕ) 
  (chocolate_price : ℚ) 
  (total_revenue : ℚ) 
  (h1 : chocolate_count = 220)
  (h2 : vanilla_count = 70)
  (h3 : chocolate_price = 1)
  (h4 : total_revenue = 360) :
  ∃ (vanilla_price : ℚ), 
    vanilla_price = 2 ∧ 
    chocolate_count * chocolate_price + vanilla_count * vanilla_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_vanilla_cookie_price_l2922_292249


namespace NUMINAMATH_CALUDE_max_value_of_f_l2922_292276

-- Define the function
def f (x : ℝ) : ℝ := x * (3 - 2 * x)

-- Define the domain
def domain (x : ℝ) : Prop := 0 < x ∧ x ≤ 1

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), max = 9/8 ∧ ∀ (x : ℝ), domain x → f x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2922_292276


namespace NUMINAMATH_CALUDE_problem_statement_l2922_292291

theorem problem_statement (a b c A B C : ℝ) 
  (eq1 : a + b + c = 0)
  (eq2 : A + B + C = 0)
  (eq3 : a / A + b / B + c / C = 0)
  (hA : A ≠ 0)
  (hB : B ≠ 0)
  (hC : C ≠ 0) :
  a * A^2 + b * B^2 + c * C^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2922_292291


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_9_divisible_by_7_l2922_292221

def ends_in_9 (n : ℕ) : Prop := n % 10 = 9

theorem smallest_positive_integer_ending_in_9_divisible_by_7 :
  ∃ (n : ℕ), n > 0 ∧ ends_in_9 n ∧ n % 7 = 0 ∧
  ∀ (m : ℕ), m > 0 → ends_in_9 m → m % 7 = 0 → m ≥ n :=
by
  use 49
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_9_divisible_by_7_l2922_292221


namespace NUMINAMATH_CALUDE_complex_cube_root_unity_l2922_292207

/-- Given that i is the imaginary unit and z = -1/2 + (√3/2)i, prove that z^2 + z + 1 = 0 -/
theorem complex_cube_root_unity (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = -1/2 + (Real.sqrt 3 / 2) * i →
  z^2 + z + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_unity_l2922_292207


namespace NUMINAMATH_CALUDE_remaining_ripe_mangoes_l2922_292293

theorem remaining_ripe_mangoes (total_mangoes : ℕ) (ripe_fraction : ℚ) (eaten_fraction : ℚ) : 
  total_mangoes = 400 →
  ripe_fraction = 3/5 →
  eaten_fraction = 3/5 →
  (total_mangoes : ℚ) * ripe_fraction * (1 - eaten_fraction) = 96 := by
  sorry

end NUMINAMATH_CALUDE_remaining_ripe_mangoes_l2922_292293


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2922_292218

theorem partial_fraction_decomposition (a b c d : ℤ) (h : a * d ≠ b * c) :
  ∃ (r s : ℚ), ∀ (x : ℝ), 
    1 / ((a * x + b) * (c * x + d)) = r / (a * x + b) + s / (c * x + d) ∧
    r = a / (a * d - b * c) ∧
    s = -c / (a * d - b * c) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2922_292218


namespace NUMINAMATH_CALUDE_production_rate_equation_l2922_292232

/-- Represents the production rates of a master and apprentice -/
theorem production_rate_equation (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x < 40) 
  (h3 : x + (40 - x) = 40) 
  (h4 : ∃ t : ℝ, t > 0 ∧ x * t = 300 ∧ (40 - x) * t = 100) : 
  300 / x = 100 / (40 - x) := by
  sorry

end NUMINAMATH_CALUDE_production_rate_equation_l2922_292232


namespace NUMINAMATH_CALUDE_profit_percentage_l2922_292271

/-- If selling an article at 2/3 of price P results in a 10% loss,
    then selling it at price P results in a 35% profit. -/
theorem profit_percentage (P : ℝ) (P_pos : P > 0) : 
  (∃ C : ℝ, C > 0 ∧ (2/3 * P) = (0.9 * C)) →
  ((P - ((2/3 * P) / 0.9)) / ((2/3 * P) / 0.9)) * 100 = 35 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_l2922_292271


namespace NUMINAMATH_CALUDE_sarah_apples_to_teachers_l2922_292294

def apples_given_to_teachers (initial_apples : ℕ) (locker_apples : ℕ) (friend_apples : ℕ) 
  (classmate_apples : ℕ) (traded_apples : ℕ) (close_friends : ℕ) (eaten_apples : ℕ) 
  (final_apples : ℕ) : ℕ :=
  initial_apples - locker_apples - friend_apples - classmate_apples - traded_apples - 
  close_friends - eaten_apples - final_apples

theorem sarah_apples_to_teachers :
  apples_given_to_teachers 50 10 3 8 4 5 1 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sarah_apples_to_teachers_l2922_292294


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2922_292211

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 3 → x^3 - 27 > 0) ↔ (∃ x : ℝ, x > 3 ∧ x^3 - 27 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2922_292211


namespace NUMINAMATH_CALUDE_toothpick_grid_problem_l2922_292284

/-- Calculates the total number of toothpicks in a grid -/
def total_toothpicks (length width : ℕ) (has_divider : Bool) : ℕ :=
  let vertical_lines := length + 1 + (if has_divider then 1 else 0)
  let vertical_toothpicks := vertical_lines * width
  let horizontal_lines := width + 1
  let horizontal_toothpicks := horizontal_lines * length
  vertical_toothpicks + horizontal_toothpicks

/-- The problem statement -/
theorem toothpick_grid_problem :
  total_toothpicks 40 25 true = 2090 := by
  sorry


end NUMINAMATH_CALUDE_toothpick_grid_problem_l2922_292284


namespace NUMINAMATH_CALUDE_girls_from_valley_l2922_292299

theorem girls_from_valley (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (highland_students : ℕ) (valley_students : ℕ) (highland_boys : ℕ)
  (h1 : total_students = 120)
  (h2 : total_boys = 70)
  (h3 : total_girls = 50)
  (h4 : highland_students = 45)
  (h5 : valley_students = 75)
  (h6 : highland_boys = 30)
  (h7 : total_students = total_boys + total_girls)
  (h8 : total_students = highland_students + valley_students)
  (h9 : total_boys ≥ highland_boys) :
  valley_students - (total_boys - highland_boys) = 35 := by
sorry

end NUMINAMATH_CALUDE_girls_from_valley_l2922_292299


namespace NUMINAMATH_CALUDE_max_ab_value_l2922_292298

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b + a + b = 1) :
  a * b ≤ 3 - 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_ab_value_l2922_292298


namespace NUMINAMATH_CALUDE_snail_reaches_tree_in_26_days_l2922_292267

/-- The number of days it takes for a snail to reach a tree given its daily movement pattern -/
def snail_journey_days (s l₁ l₂ : ℕ) : ℕ :=
  let daily_progress := l₁ - l₂
  let days_to_reach_near := (s - l₁) / daily_progress
  days_to_reach_near + 1

/-- Theorem stating that the snail reaches the tree in 26 days under the given conditions -/
theorem snail_reaches_tree_in_26_days :
  snail_journey_days 30 5 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_snail_reaches_tree_in_26_days_l2922_292267


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2922_292212

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (3 - 2 * i) / (1 + 4 * i) = -5/17 - 14/17 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2922_292212


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l2922_292282

theorem min_value_sum_squares (x y : ℝ) (h : x^2 + 2*x*y - 3*y^2 = 1) :
  ∃ (min : ℝ), (∀ (a b : ℝ), a^2 + 2*a*b - 3*b^2 = 1 → a^2 + b^2 ≥ min) ∧
  min = (Real.sqrt 5 + 1) / 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l2922_292282


namespace NUMINAMATH_CALUDE_dividend_calculation_l2922_292240

theorem dividend_calculation (divisor remainder quotient : ℕ) : 
  divisor = 17 → remainder = 8 → quotient = 4 → 
  divisor * quotient + remainder = 76 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2922_292240


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2922_292209

theorem trigonometric_inequality (x : ℝ) :
  9.286 * (Real.sin x)^3 * Real.sin (π/2 - 3*x) + (Real.cos x)^3 * Real.cos (π/2 - 3*x) > 3*Real.sqrt 3/8 →
  ∃ n : ℤ, π/12 + n*π/2 < x ∧ x < π/6 + n*π/2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l2922_292209


namespace NUMINAMATH_CALUDE_singing_competition_winner_l2922_292266

/-- Represents the contestants in the singing competition -/
inductive Contestant : Type
  | one | two | three | four | five | six

/-- Represents the students making guesses -/
inductive Student : Type
  | A | B | C | D

def guess (s : Student) (c : Contestant) : Prop :=
  match s with
  | Student.A => c = Contestant.four ∨ c = Contestant.five
  | Student.B => c ≠ Contestant.three
  | Student.C => c = Contestant.one ∨ c = Contestant.two ∨ c = Contestant.six
  | Student.D => c ≠ Contestant.four ∧ c ≠ Contestant.five ∧ c ≠ Contestant.six

theorem singing_competition_winner :
  ∃! (winner : Contestant),
    (∃! (correct_guesser : Student), guess correct_guesser winner) ∧
    (∀ (c : Contestant), c ≠ winner → ¬ guess Student.A c ∧ ¬ guess Student.B c ∧ ¬ guess Student.C c ∧ ¬ guess Student.D c) ∧
    winner = Contestant.three :=
by sorry

end NUMINAMATH_CALUDE_singing_competition_winner_l2922_292266


namespace NUMINAMATH_CALUDE_abs_value_of_z_l2922_292295

/-- The absolute value of the complex number z = (2i)/(1+i) - 2i is √2 -/
theorem abs_value_of_z : Complex.abs ((2 * Complex.I) / (1 + Complex.I) - 2 * Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_of_z_l2922_292295


namespace NUMINAMATH_CALUDE_num_solutions_is_four_l2922_292250

/-- The number of distinct solutions to the system of equations:
    (x - 2y + 3)(4x + y - 5) = 0 and (x + 2y - 5)(3x - 4y + 6) = 0 -/
def num_solutions : ℕ :=
  let eq1 (x y : ℝ) := (x - 2*y + 3)*(4*x + y - 5) = 0
  let eq2 (x y : ℝ) := (x + 2*y - 5)*(3*x - 4*y + 6) = 0
  4  -- The actual number of solutions

theorem num_solutions_is_four :
  num_solutions = 4 := by sorry

end NUMINAMATH_CALUDE_num_solutions_is_four_l2922_292250


namespace NUMINAMATH_CALUDE_student_earnings_theorem_l2922_292285

/-- Calculates the monthly earnings of a student working as a courier after tax deduction -/
def monthly_earnings_after_tax (daily_rate : ℝ) (days_per_week : ℕ) (weeks_per_month : ℕ) (tax_rate : ℝ) : ℝ :=
  let gross_monthly_earnings := daily_rate * (days_per_week : ℝ) * (weeks_per_month : ℝ)
  let tax_amount := gross_monthly_earnings * tax_rate
  gross_monthly_earnings - tax_amount

/-- Theorem stating that the monthly earnings of the student after tax is 17400 rubles -/
theorem student_earnings_theorem :
  monthly_earnings_after_tax 1250 4 4 0.13 = 17400 := by
  sorry

end NUMINAMATH_CALUDE_student_earnings_theorem_l2922_292285


namespace NUMINAMATH_CALUDE_ferry_hat_count_l2922_292281

theorem ferry_hat_count :
  ∀ (total_adults : ℕ) (children : ℕ) 
    (women_hat_percent : ℚ) (men_hat_percent : ℚ) (children_hat_percent : ℚ),
  total_adults = 3000 →
  children = 500 →
  women_hat_percent = 25 / 100 →
  men_hat_percent = 15 / 100 →
  children_hat_percent = 30 / 100 →
  ∃ (women : ℕ) (men : ℕ),
    women = men ∧
    women + men = total_adults ∧
    (↑women * women_hat_percent + ↑men * men_hat_percent + ↑children * children_hat_percent : ℚ) = 750 :=
by sorry

end NUMINAMATH_CALUDE_ferry_hat_count_l2922_292281


namespace NUMINAMATH_CALUDE_leo_current_weight_l2922_292243

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 98

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 170 - leo_weight

/-- Theorem stating that Leo's current weight is 98 pounds -/
theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = 170) →
  leo_weight = 98 := by
sorry

end NUMINAMATH_CALUDE_leo_current_weight_l2922_292243


namespace NUMINAMATH_CALUDE_class_composition_l2922_292261

theorem class_composition (initial_girls : ℕ) (initial_boys : ℕ) (girls_left : ℕ) :
  initial_girls * 6 = initial_boys * 5 →
  (initial_girls - girls_left) * 3 = initial_boys * 2 →
  girls_left = 20 →
  initial_boys = 120 := by
  sorry

end NUMINAMATH_CALUDE_class_composition_l2922_292261


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2922_292264

theorem complex_equation_solution (a : ℝ) :
  (2 + a * Complex.I) / (1 + Complex.I) = (3 : ℂ) + Complex.I → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2922_292264


namespace NUMINAMATH_CALUDE_sum_of_distances_is_12_sqrt_2_l2922_292217

-- Define the curve C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define point P
def P : ℝ × ℝ := (-2, -4)

-- Define the intersection points M and N (existence assumed)
axiom M_exists : ∃ M : ℝ × ℝ, C M.1 M.2 ∧ l M.1 M.2
axiom N_exists : ∃ N : ℝ × ℝ, C N.1 N.2 ∧ l N.1 N.2
axiom M_ne_N : ∀ M N : ℝ × ℝ, C M.1 M.2 ∧ l M.1 M.2 ∧ C N.1 N.2 ∧ l N.1 N.2 → M ≠ N

-- Theorem statement
theorem sum_of_distances_is_12_sqrt_2 :
  ∃ M N : ℝ × ℝ, C M.1 M.2 ∧ l M.1 M.2 ∧ C N.1 N.2 ∧ l N.1 N.2 ∧ M ≠ N ∧
  Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) + Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2) = 12 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_is_12_sqrt_2_l2922_292217


namespace NUMINAMATH_CALUDE_even_function_sufficient_not_necessary_l2922_292242

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def exists_symmetric_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = f (-x₀)

theorem even_function_sufficient_not_necessary :
  (∀ f : ℝ → ℝ, is_even_function f → exists_symmetric_point f) ∧
  ¬(∀ f : ℝ → ℝ, exists_symmetric_point f → is_even_function f) := by
  sorry

end NUMINAMATH_CALUDE_even_function_sufficient_not_necessary_l2922_292242


namespace NUMINAMATH_CALUDE_M_N_intersection_empty_l2922_292258

def M : Set ℂ :=
  {z | ∃ t : ℝ, t ≠ -1 ∧ t ≠ 0 ∧ z = t / (1 + t) + Complex.I * ((1 + t) / t)}

def N : Set ℂ :=
  {z | ∃ t : ℝ, |t| ≤ 1 ∧ z = Real.sqrt 2 * (Complex.cos (Real.arcsin t) + Complex.I * Complex.cos (Real.arccos t))}

theorem M_N_intersection_empty : M ∩ N = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_N_intersection_empty_l2922_292258


namespace NUMINAMATH_CALUDE_annulus_area_l2922_292213

/-- An annulus is formed by two concentric circles with radii R and r, where R > r.
    x is the length of a tangent line from a point on the outer circle to the inner circle. -/
theorem annulus_area (R r x : ℝ) (h1 : R > r) (h2 : R^2 - r^2 = x^2) : 
  π * R^2 - π * r^2 = π * x^2 := by sorry

end NUMINAMATH_CALUDE_annulus_area_l2922_292213


namespace NUMINAMATH_CALUDE_wood_per_sack_l2922_292231

theorem wood_per_sack (total_wood : ℕ) (num_sacks : ℕ) (wood_per_sack : ℕ) 
  (h1 : total_wood = 80) 
  (h2 : num_sacks = 4) 
  (h3 : wood_per_sack = total_wood / num_sacks) :
  wood_per_sack = 20 := by
  sorry

end NUMINAMATH_CALUDE_wood_per_sack_l2922_292231


namespace NUMINAMATH_CALUDE_bob_oyster_shucking_l2922_292269

/-- Given that Bob can shuck 10 oysters in 5 minutes, this theorem proves
    that he can shuck 240 oysters in 2 hours. -/
theorem bob_oyster_shucking (bob_rate : ℕ) (bob_time : ℕ) (total_time : ℕ) :
  bob_rate = 10 →
  bob_time = 5 →
  total_time = 120 →
  (total_time / bob_time) * bob_rate = 240 :=
by
  sorry

#check bob_oyster_shucking

end NUMINAMATH_CALUDE_bob_oyster_shucking_l2922_292269


namespace NUMINAMATH_CALUDE_movie_production_people_l2922_292228

/-- The number of people at the movie production --/
def num_people : ℕ := 50

/-- The cost of hiring actors --/
def actor_cost : ℕ := 1200

/-- The cost of food per person --/
def food_cost_per_person : ℕ := 3

/-- The total cost of the movie production --/
def total_cost : ℕ := 10000 - 5950

/-- The equipment rental cost is twice the combined cost of food and actors --/
def equipment_cost (p : ℕ) : ℕ := 2 * (food_cost_per_person * p + actor_cost)

/-- The total cost calculation based on the number of people --/
def calculated_cost (p : ℕ) : ℕ :=
  actor_cost + food_cost_per_person * p + equipment_cost p

theorem movie_production_people :
  calculated_cost num_people = total_cost :=
by sorry

end NUMINAMATH_CALUDE_movie_production_people_l2922_292228


namespace NUMINAMATH_CALUDE_sin_x_bounds_l2922_292205

theorem sin_x_bounds (x : ℝ) (h : 0 < x) (h' : x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := by
  sorry

end NUMINAMATH_CALUDE_sin_x_bounds_l2922_292205


namespace NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l2922_292241

theorem smallest_third_term_geometric_progression 
  (a b c : ℝ) 
  (arithmetic_prog : a = 7 ∧ c - b = b - a) 
  (geometric_prog : ∃ r : ℝ, r > 0 ∧ (b + 3) = a * r ∧ (c + 22) = (b + 3) * r) :
  c + 22 ≥ 23 + 16 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l2922_292241


namespace NUMINAMATH_CALUDE_min_cost_14000_l2922_292222

/-- Represents the number of soup + main course combinations -/
def x : ℕ := 15

/-- Represents the number of salad + main course combinations -/
def y : ℕ := 0

/-- Represents the number of all three dish combinations -/
def z : ℕ := 0

/-- Represents the number of standalone main courses -/
def q : ℕ := 35

/-- The cost of a salad -/
def salad_cost : ℕ := 200

/-- The cost of soup + main course -/
def soup_main_cost : ℕ := 350

/-- The cost of salad + main course -/
def salad_main_cost : ℕ := 350

/-- The cost of soup + salad + main course -/
def all_three_cost : ℕ := 500

/-- The total number of main courses required -/
def total_main : ℕ := 50

/-- The total number of salads required -/
def total_salad : ℕ := 30

/-- The total number of soups required -/
def total_soup : ℕ := 15

theorem min_cost_14000 :
  (x + y + z + q = total_main) ∧
  (y + z = total_salad) ∧
  (x + z = total_soup) ∧
  (∀ x' y' z' q' : ℕ,
    (x' + y' + z' + q' = total_main) →
    (y' + z' = total_salad) →
    (x' + z' = total_soup) →
    soup_main_cost * x + salad_main_cost * y + all_three_cost * z + salad_cost * q ≤
    soup_main_cost * x' + salad_main_cost * y' + all_three_cost * z' + salad_cost * q') →
  soup_main_cost * x + salad_main_cost * y + all_three_cost * z + salad_cost * q = 14000 :=
sorry

end NUMINAMATH_CALUDE_min_cost_14000_l2922_292222


namespace NUMINAMATH_CALUDE_bens_age_l2922_292219

theorem bens_age (b j : ℕ) : 
  b = 3 * j + 10 →  -- Ben's age is 10 years more than thrice Jane's age
  b + j = 70 →      -- The sum of their ages is 70
  b = 55 :=         -- Ben's age is 55
by sorry

end NUMINAMATH_CALUDE_bens_age_l2922_292219


namespace NUMINAMATH_CALUDE_balance_theorem_l2922_292280

-- Define the weights of balls as real numbers
variable (R G O B : ℝ)

-- Define the balance relationships
axiom red_green : 4 * R = 8 * G
axiom orange_green : 3 * O = 6 * G
axiom green_blue : 8 * G = 6 * B

-- Theorem to prove
theorem balance_theorem : 3 * R + 2 * O + 4 * B = (46/3) * G := by
  sorry

end NUMINAMATH_CALUDE_balance_theorem_l2922_292280


namespace NUMINAMATH_CALUDE_percentage_relationship_l2922_292286

theorem percentage_relationship (A B n c : ℝ) : 
  A > 0 → B > 0 → B > A → 
  A * (1 + n / 100) = B → B * (1 - c / 100) = A →
  A * Real.sqrt (100 + n) = B * Real.sqrt (100 - c) := by
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l2922_292286


namespace NUMINAMATH_CALUDE_draw_balls_theorem_l2922_292268

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def red_score : ℕ := 2
def white_score : ℕ := 1
def balls_to_draw : ℕ := 4
def min_score : ℕ := 5

/-- The number of ways to draw 4 balls from a bag containing 4 red balls and 6 white balls,
    where red balls score 2 points and white balls score 1 point,
    such that the total score is not less than 5 points. -/
def ways_to_draw : ℕ := 195

theorem draw_balls_theorem :
  ways_to_draw = 195 :=
sorry

end NUMINAMATH_CALUDE_draw_balls_theorem_l2922_292268


namespace NUMINAMATH_CALUDE_smoothie_cost_l2922_292200

/-- The cost of Morgan's smoothie given the prices of other items and the transaction details. -/
theorem smoothie_cost (hamburger_cost onion_rings_cost amount_paid change_received : ℕ) : 
  hamburger_cost = 4 →
  onion_rings_cost = 2 →
  amount_paid = 20 →
  change_received = 11 →
  amount_paid - change_received - (hamburger_cost + onion_rings_cost) = 3 := by
  sorry

#check smoothie_cost

end NUMINAMATH_CALUDE_smoothie_cost_l2922_292200


namespace NUMINAMATH_CALUDE_system_solution_l2922_292256

theorem system_solution (u v w : ℝ) : 
  (u + v * w = 20 ∧ v + w * u = 20 ∧ w + u * v = 20) ↔ 
  ((u, v, w) = (4, 4, 4) ∨ 
   (u, v, w) = (-5, -5, -5) ∨ 
   (u, v, w) = (1, 1, 19) ∨ 
   (u, v, w) = (19, 1, 1) ∨ 
   (u, v, w) = (1, 19, 1)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2922_292256


namespace NUMINAMATH_CALUDE_win_bonus_area_l2922_292288

/-- The combined area of WIN and BONUS sectors in a circular spinner -/
theorem win_bonus_area (r : ℝ) (p_win : ℝ) (p_bonus : ℝ) : 
  r = 8 → p_win = 1/4 → p_bonus = 1/8 → 
  (p_win + p_bonus) * (π * r^2) = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_win_bonus_area_l2922_292288


namespace NUMINAMATH_CALUDE_find_a_l2922_292290

theorem find_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = a * x + 6) →
  f (-1) = 8 →
  a = -2 := by sorry

end NUMINAMATH_CALUDE_find_a_l2922_292290


namespace NUMINAMATH_CALUDE_remainder_problem_l2922_292246

theorem remainder_problem (k : ℕ) (h1 : k > 0) (h2 : k < 42) 
  (h3 : k % 5 = 2) (h4 : k % 6 = 5) : k % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2922_292246


namespace NUMINAMATH_CALUDE_plane_equation_l2922_292255

/-- The equation of a plane passing through a point and parallel to another plane -/
theorem plane_equation (x y z : ℝ) : ∃ (A B C D : ℤ),
  -- The plane passes through the point (2,3,-1)
  A * 2 + B * 3 + C * (-1) + D = 0 ∧
  -- The plane is parallel to 3x - 4y + 2z = 5
  ∃ (k : ℝ), k ≠ 0 ∧ A = k * 3 ∧ B = k * (-4) ∧ C = k * 2 ∧
  -- The equation is in the form Ax + By + Cz + D = 0
  A * x + B * y + C * z + D = 0 ∧
  -- A is positive
  A > 0 ∧
  -- The greatest common divisor of |A|, |B|, |C|, and |D| is 1
  Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1 ∧
  -- The specific solution
  A = 3 ∧ B = -4 ∧ C = 2 ∧ D = 8 := by
sorry

end NUMINAMATH_CALUDE_plane_equation_l2922_292255


namespace NUMINAMATH_CALUDE_triangle_inequality_l2922_292206

theorem triangle_inequality (a b c : ℝ) (h : |((a^2 + b^2 - c^2) / (a*b))| < 2) :
  |((b^2 + c^2 - a^2) / (b*c))| < 2 ∧ |((c^2 + a^2 - b^2) / (c*a))| < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2922_292206


namespace NUMINAMATH_CALUDE_evaluate_expression_l2922_292263

theorem evaluate_expression : (16^24) / (32^12) = 8^12 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2922_292263


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2922_292292

/-- A quadratic function with roots at 2 and -4, and a minimum value of 32 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  root1 : a * 2^2 + b * 2 + c = 0
  root2 : a * (-4)^2 + b * (-4) + c = 0
  min_value : ∀ x, a * x^2 + b * x + c ≥ 32

theorem sum_of_coefficients (f : QuadraticFunction) : f.a + f.b + f.c = 160 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2922_292292


namespace NUMINAMATH_CALUDE_trig_identity_proof_l2922_292208

theorem trig_identity_proof : 
  (2 * Real.sin (80 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2922_292208


namespace NUMINAMATH_CALUDE_pria_distance_driven_l2922_292225

/-- Calculates the distance driven with a full tank of gas given the advertised mileage,
    tank capacity, and difference between advertised and actual mileage. -/
def distance_driven (advertised_mileage : ℝ) (tank_capacity : ℝ) (mileage_difference : ℝ) : ℝ :=
  (advertised_mileage - mileage_difference) * tank_capacity

/-- Proves that given the specified conditions, the distance driven is 372 miles. -/
theorem pria_distance_driven :
  distance_driven 35 12 4 = 372 := by
  sorry

end NUMINAMATH_CALUDE_pria_distance_driven_l2922_292225


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2922_292244

theorem polynomial_division_remainder (x : ℝ) : 
  x^1004 % ((x^2 + 1) * (x - 1)) = x^2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2922_292244


namespace NUMINAMATH_CALUDE_range_of_k_for_fractional_equation_l2922_292210

theorem range_of_k_for_fractional_equation :
  ∀ k x : ℝ,
  (x > 0) →
  (x ≠ 2) →
  (1 / (x - 2) + 3 = (3 - k) / (2 - x)) →
  (k > -2 ∧ k ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_for_fractional_equation_l2922_292210


namespace NUMINAMATH_CALUDE_log_equality_l2922_292237

theorem log_equality (y : ℝ) (h : y = (Real.log 16 / Real.log 4) ^ (Real.log 4 / Real.log 16)) :
  Real.log y / Real.log 5 = (1/2) * (Real.log 2 / Real.log 5) := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l2922_292237


namespace NUMINAMATH_CALUDE_line_equation_proof_l2922_292278

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (given_line : Line) (p : Point) (result_line : Line) : 
  given_line.a = 2 ∧ given_line.b = -3 ∧ given_line.c = 5 ∧
  p.x = -2 ∧ p.y = 1 ∧
  result_line.a = 2 ∧ result_line.b = -3 ∧ result_line.c = 7 →
  pointOnLine p result_line ∧ parallel given_line result_line :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2922_292278


namespace NUMINAMATH_CALUDE_hotel_stay_cost_l2922_292245

/-- Calculates the total cost for a group staying at a hotel. -/
def total_hotel_cost (cost_per_night : ℕ) (num_nights : ℕ) (num_people : ℕ) : ℕ :=
  cost_per_night * num_nights * num_people

/-- Proves that the total cost for 3 people staying 3 nights at $40 per night is $360. -/
theorem hotel_stay_cost :
  total_hotel_cost 40 3 3 = 360 := by
sorry

end NUMINAMATH_CALUDE_hotel_stay_cost_l2922_292245


namespace NUMINAMATH_CALUDE_lost_revenue_calculation_l2922_292238

/-- Represents the revenue calculation for a movie theater --/
def theater_revenue (capacity : ℕ) (general_price : ℚ) (child_price : ℚ) (senior_price : ℚ) 
  (veteran_discount : ℚ) (general_sold : ℕ) (child_sold : ℕ) (senior_sold : ℕ) (veteran_sold : ℕ) : ℚ :=
  let actual_revenue := general_sold * general_price + child_sold * child_price + 
                        senior_sold * senior_price + veteran_sold * (general_price - veteran_discount)
  let max_potential_revenue := capacity * general_price
  max_potential_revenue - actual_revenue

/-- Theorem stating the lost revenue for the given scenario --/
theorem lost_revenue_calculation : 
  theater_revenue 50 10 6 8 2 20 3 4 2 = 234 := by sorry

end NUMINAMATH_CALUDE_lost_revenue_calculation_l2922_292238


namespace NUMINAMATH_CALUDE_benzene_required_for_reaction_l2922_292287

-- Define the molecules and their molar ratios in the reaction
structure Reaction :=
  (benzene : ℚ)
  (methane : ℚ)
  (toluene : ℚ)
  (hydrogen : ℚ)

-- Define the balanced equation
def balanced_equation : Reaction := ⟨1, 1, 1, 1⟩

-- Theorem statement
theorem benzene_required_for_reaction 
  (methane_input : ℚ) 
  (hydrogen_output : ℚ) :
  methane_input = 2 →
  hydrogen_output = 2 →
  methane_input * balanced_equation.benzene / balanced_equation.methane = 2 :=
by sorry

end NUMINAMATH_CALUDE_benzene_required_for_reaction_l2922_292287


namespace NUMINAMATH_CALUDE_special_integers_proof_l2922_292253

theorem special_integers_proof (k : ℕ) (h : k ≥ 2) :
  (∀ m n : ℕ, 1 ≤ m ∧ m < n ∧ n ≤ k → ¬(k ∣ (n^(n-1) - m^(m-1)))) ↔ (k = 2 ∨ k = 3) :=
sorry

end NUMINAMATH_CALUDE_special_integers_proof_l2922_292253


namespace NUMINAMATH_CALUDE_triangle_area_is_five_l2922_292202

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 2 * x - 5 * y - 10 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 5

/-- The y-intercept of the line -/
def y_intercept : ℝ := -2

/-- The area of the triangle -/
def triangle_area : ℝ := 5

/-- Theorem: The area of the triangle formed by the line 2x - 5y - 10 = 0 and the coordinate axes is 5 -/
theorem triangle_area_is_five : 
  triangle_area = (1/2) * x_intercept * (-y_intercept) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_five_l2922_292202


namespace NUMINAMATH_CALUDE_real_roots_imply_real_roots_l2922_292272

/-- Given a quadratic equation x^2 + px + q = 0 with real roots, 
    prove that related equations also have real roots. -/
theorem real_roots_imply_real_roots 
  (p q k x₁ x₂ : ℝ) 
  (hk : k ≠ 0) 
  (hx : x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0) :
  ∃ (y₁ y₂ z₁ z₂ z₁' z₂' : ℝ), 
    (y₁^2 + (k + 1/k)*p*y₁ + p^2 + q*(k - 1/k)^2 = 0 ∧ 
     y₂^2 + (k + 1/k)*p*y₂ + p^2 + q*(k - 1/k)^2 = 0) ∧
    (z₁^2 - y₁*z₁ + q = 0 ∧ z₂^2 - y₁*z₂ + q = 0) ∧
    (z₁'^2 - y₂*z₁' + q = 0 ∧ z₂'^2 - y₂*z₂' + q = 0) ∧
    y₁ = k*x₁ + (1/k)*x₂ ∧ 
    y₂ = k*x₂ + (1/k)*x₁ ∧
    z₁ = k*x₁ ∧ 
    z₂ = (1/k)*x₂ ∧ 
    z₁' = k*x₂ ∧ 
    z₂' = (1/k)*x₁ := by
  sorry

end NUMINAMATH_CALUDE_real_roots_imply_real_roots_l2922_292272


namespace NUMINAMATH_CALUDE_count_fourth_powers_between_10_and_10000_l2922_292283

theorem count_fourth_powers_between_10_and_10000 : 
  (Finset.filter (fun n : ℕ => 10 ≤ n^4 ∧ n^4 ≤ 10000) (Finset.range (10000 + 1))).card = 19 :=
by sorry

end NUMINAMATH_CALUDE_count_fourth_powers_between_10_and_10000_l2922_292283


namespace NUMINAMATH_CALUDE_principal_amount_calculation_l2922_292247

theorem principal_amount_calculation (rate : ℝ) (interest : ℝ) (time : ℝ) :
  rate = 0.08333333333333334 →
  interest = 400 →
  time = 4 →
  interest = (interest / (rate * time)) * rate * time :=
by
  sorry

end NUMINAMATH_CALUDE_principal_amount_calculation_l2922_292247


namespace NUMINAMATH_CALUDE_snail_climb_theorem_l2922_292274

/-- The number of days it takes for a snail to climb out of a well -/
def snail_climb_days (well_depth : ℝ) (day_climb : ℝ) (night_slide : ℝ) : ℕ :=
  sorry

/-- Theorem: A snail starting 1 meter below the top of a well, 
    climbing 30 cm during the day and sliding down 20 cm each night, 
    will take 8 days to reach the top of the well -/
theorem snail_climb_theorem : 
  snail_climb_days 1 0.3 0.2 = 8 := by sorry

end NUMINAMATH_CALUDE_snail_climb_theorem_l2922_292274


namespace NUMINAMATH_CALUDE_orange_buckets_total_l2922_292201

theorem orange_buckets_total (x y : ℝ) : 
  x = 2 * 22.5 + 3 →
  y = x - 11.5 →
  22.5 + x + y = 107 := by
sorry

end NUMINAMATH_CALUDE_orange_buckets_total_l2922_292201
