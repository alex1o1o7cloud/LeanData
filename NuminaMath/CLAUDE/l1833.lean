import Mathlib

namespace right_triangle_segment_ratio_l1833_183341

theorem right_triangle_segment_ratio (x y z u v : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
    (h4 : x^2 + y^2 = z^2) (h5 : x * z = u * (u + v)) (h6 : y * z = v * (u + v)) (h7 : 3 * y = 4 * x) :
    9 * v = 16 * u := by
  sorry

end right_triangle_segment_ratio_l1833_183341


namespace parabola_equation_l1833_183395

/-- A parabola with vertex at the origin and axis along the y-axis passing through (30, -40) with focus at (0, -45/4) has the equation x^2 = -45/2 * y -/
theorem parabola_equation (p : ℝ × ℝ) (f : ℝ × ℝ) :
  p.1 = 30 ∧ p.2 = -40 ∧ f.1 = 0 ∧ f.2 = -45/4 →
  ∀ x y : ℝ, (x^2 = -45/2 * y ↔ (x - f.1)^2 + (y - f.2)^2 = (y - p.2)^2) :=
by sorry

end parabola_equation_l1833_183395


namespace cos_alpha_for_point_on_terminal_side_l1833_183323

/-- Given a point P(-3, -4) on the terminal side of angle α, prove that cos α = -3/5 -/
theorem cos_alpha_for_point_on_terminal_side (α : Real) :
  let P : Prod Real Real := (-3, -4)
  ∃ (r : Real), r > 0 ∧ P = (r * Real.cos α, r * Real.sin α) →
  Real.cos α = -3/5 := by
sorry

end cos_alpha_for_point_on_terminal_side_l1833_183323


namespace quadratic_max_value_l1833_183348

/-- Given a quadratic function y = -3x^2 + 6x + 4, prove that its maximum value is 7 -/
theorem quadratic_max_value :
  let f : ℝ → ℝ := λ x ↦ -3 * x^2 + 6 * x + 4
  ∃ x_max : ℝ, ∀ x : ℝ, f x ≤ f x_max ∧ f x_max = 7 := by
  sorry

end quadratic_max_value_l1833_183348


namespace constant_distance_l1833_183358

/-- Ellipse E with eccentricity 1/2 and area of triangle F₁PF₂ equal to 3 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : a^2/4 + b^2/3 = 1

/-- Point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2/E.a^2 + y^2/E.b^2 = 1

/-- Point on the line y = 2√3 -/
structure PointOnLine where
  x : ℝ
  y : ℝ
  h : y = 2 * Real.sqrt 3

/-- The theorem to be proved -/
theorem constant_distance (E : Ellipse) (M : PointOnEllipse E) (N : PointOnLine) 
  (h : (M.x * N.x + M.y * N.y) / (M.x^2 + M.y^2).sqrt / (N.x^2 + N.y^2).sqrt = 0) :
  ((M.y * N.x - M.x * N.y)^2 / ((M.x - N.x)^2 + (M.y - N.y)^2)).sqrt = Real.sqrt 3 := by
  sorry

end constant_distance_l1833_183358


namespace average_marks_math_chem_l1833_183320

/-- Given the total marks in mathematics and physics is 70, and chemistry score is 20 marks more than physics score, prove that the average marks in mathematics and chemistry is 45. -/
theorem average_marks_math_chem (math physics chem : ℕ) : 
  math + physics = 70 → 
  chem = physics + 20 → 
  (math + chem) / 2 = 45 := by
sorry

end average_marks_math_chem_l1833_183320


namespace infinitely_many_primes_of_the_year_l1833_183365

/-- A prime p is a Prime of the Year if there exists a positive integer n such that n^2 + 1 ≡ 0 (mod p^2007) -/
def PrimeOfTheYear (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ n : ℕ, n > 0 ∧ (n^2 + 1) % p^2007 = 0

/-- There are infinitely many Primes of the Year -/
theorem infinitely_many_primes_of_the_year :
  ∀ N : ℕ, ∃ p : ℕ, p > N ∧ PrimeOfTheYear p :=
sorry

end infinitely_many_primes_of_the_year_l1833_183365


namespace total_students_l1833_183386

theorem total_students (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 21) 
  (h2 : rank_from_left = 11) : 
  rank_from_right + rank_from_left - 1 = 31 := by
  sorry

end total_students_l1833_183386


namespace odd_function_2019_l1833_183356

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_2019 (f : ℝ → ℝ) 
  (h_odd : IsOdd f)
  (h_sym : ∀ x, f (1 + x) = f (1 - x))
  (h_f1 : f 1 = 9) :
  f 2019 = -9 := by
  sorry

end odd_function_2019_l1833_183356


namespace line_translation_theorem_l1833_183344

/-- Represents a line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Translates a line vertically and horizontally -/
def translateLine (l : Line) (vertical : ℝ) (horizontal : ℝ) : Line :=
  { slope := l.slope,
    yIntercept := l.yIntercept - vertical - l.slope * horizontal }

theorem line_translation_theorem :
  let initialLine : Line := { slope := 2, yIntercept := 1 }
  let translatedLine := translateLine initialLine 3 2
  translatedLine = { slope := 2, yIntercept := -6 } := by sorry

end line_translation_theorem_l1833_183344


namespace polynomial_factorization_l1833_183370

theorem polynomial_factorization :
  ∀ x : ℂ, x^15 + x^10 + 1 = (x^3 - 1) * (x^12 + x^9 + x^6 + x^3 + 1) := by
  sorry

end polynomial_factorization_l1833_183370


namespace a_1_greater_than_one_l1833_183376

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

theorem a_1_greater_than_one (seq : ArithmeticSequence)
  (sum_condition : seq.a 1 + seq.a 2 + seq.a 5 > 13)
  (geometric_condition : seq.a 2 ^ 2 = seq.a 1 * seq.a 5) :
  seq.a 1 > 1 := by
  sorry

end a_1_greater_than_one_l1833_183376


namespace least_marbles_theorem_l1833_183377

/-- The least number of marbles that can be divided equally among 4, 5, 7, and 8 children
    and is a perfect square. -/
def least_marbles : ℕ := 19600

/-- Predicate to check if a number is divisible by 4, 5, 7, and 8. -/
def divisible_by_4_5_7_8 (n : ℕ) : Prop :=
  n % 4 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0

/-- Predicate to check if a number is a perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem least_marbles_theorem :
  divisible_by_4_5_7_8 least_marbles ∧
  is_perfect_square least_marbles ∧
  ∀ n : ℕ, n < least_marbles →
    ¬(divisible_by_4_5_7_8 n ∧ is_perfect_square n) :=
by sorry

end least_marbles_theorem_l1833_183377


namespace sin_690_degrees_l1833_183389

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end sin_690_degrees_l1833_183389


namespace floor_ceil_calculation_l1833_183357

theorem floor_ceil_calculation : 
  ⌊(18 : ℝ) / 5 * (-33 : ℝ) / 4⌋ - ⌈(18 : ℝ) / 5 * ⌈(-33 : ℝ) / 4⌉⌉ = -2 := by
  sorry

end floor_ceil_calculation_l1833_183357


namespace profit_maximum_l1833_183345

/-- The profit function for a product with selling price m -/
def profit (m : ℝ) : ℝ := (m - 8) * (900 - 15 * m)

/-- The expression claimed to represent the maximum profit -/
def maxProfitExpr (m : ℝ) : ℝ := -15 * (m - 34)^2 + 10140

theorem profit_maximum :
  ∃ m₀ : ℝ, 
    (∀ m : ℝ, profit m ≤ profit m₀) ∧ 
    (∀ m : ℝ, maxProfitExpr m = profit m) ∧
    (maxProfitExpr m₀ = profit m₀) :=
sorry

end profit_maximum_l1833_183345


namespace root_sum_sixth_power_l1833_183374

theorem root_sum_sixth_power (r s : ℝ) 
  (h1 : r + s = Real.sqrt 7)
  (h2 : r * s = 1) : 
  r^6 + s^6 = 527 := by
  sorry

end root_sum_sixth_power_l1833_183374


namespace range_of_m_l1833_183326

-- Define the propositions p and q
def p (x : ℝ) : Prop := -2 ≤ (4 - x) / 3 ∧ (4 - x) / 3 ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, ¬(p x) → ¬(q x m)) ∧  -- ¬p is necessary for ¬q
  (∃ x, ¬(p x) ∧ q x m) ∧     -- ¬p is not sufficient for ¬q
  (m > 0) →                   -- m is positive
  m ≥ 9 :=                    -- The range of m
by sorry

end range_of_m_l1833_183326


namespace line_slopes_product_l1833_183305

theorem line_slopes_product (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (∃ θ : ℝ, m = Real.tan (3 * θ) ∧ n = Real.tan θ) →
  m = 9 * n →
  m * n = 81 / 13 := by
  sorry

end line_slopes_product_l1833_183305


namespace sum_c_n_d_n_over_8_n_l1833_183367

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the sequences c_n and d_n
def c_n_d_n (n : ℕ) : ℂ := (3 + 2 * i) ^ n

-- Define c_n as the real part of c_n_d_n
def c_n (n : ℕ) : ℝ := (c_n_d_n n).re

-- Define d_n as the imaginary part of c_n_d_n
def d_n (n : ℕ) : ℝ := (c_n_d_n n).im

-- State the theorem
theorem sum_c_n_d_n_over_8_n :
  ∑' n, (c_n n * d_n n) / (8 : ℝ) ^ n = 6 / 17 := by sorry

end sum_c_n_d_n_over_8_n_l1833_183367


namespace right_triangle_side_length_l1833_183361

theorem right_triangle_side_length (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) (h5 : c = 17) (h6 : a = 15) : b = 8 := by
  sorry

end right_triangle_side_length_l1833_183361


namespace dans_age_l1833_183332

theorem dans_age (x : ℕ) : (x + 16 = 4 * (x - 8)) → x = 16 := by
  sorry

end dans_age_l1833_183332


namespace negation_of_sum_equals_one_l1833_183347

theorem negation_of_sum_equals_one (a b : ℝ) :
  ¬(a + b = 1) ↔ (a + b > 1 ∨ a + b < 1) :=
by sorry

end negation_of_sum_equals_one_l1833_183347


namespace parabola_shift_theorem_l1833_183346

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c - k }

theorem parabola_shift_theorem (x y : ℝ) :
  let original := Parabola.mk 3 0 0
  let shifted := shift_parabola original 1 2
  y = 3 * x^2 → y = 3 * (x - 1)^2 - 2 := by sorry

end parabola_shift_theorem_l1833_183346


namespace sum_simplification_l1833_183333

theorem sum_simplification :
  (-1)^2002 + (-1)^2003 + 2^2004 - 2^2003 = 2^2003 := by
  sorry

end sum_simplification_l1833_183333


namespace local_minimum_implies_a_equals_one_max_a_for_positive_f_main_result_l1833_183343

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - a * x / (x + 1)

-- Theorem for part (I)
theorem local_minimum_implies_a_equals_one (a : ℝ) :
  (∀ x, x > -1 → f a x ≥ f a 0) → a = 1 := by sorry

-- Theorem for part (II)
theorem max_a_for_positive_f (a : ℝ) :
  (∀ x, x > 0 → f a x > 0) → a ≤ 1 := by sorry

-- Theorem combining both parts
theorem main_result :
  (∃ a : ℝ, (∀ x, x > -1 → f a x ≥ f a 0) ∧ 
   (∀ a', (∀ x, x > 0 → f a' x > 0) → a' ≤ a)) ∧
  (∃ a : ℝ, a = 1 ∧ (∀ x, x > -1 → f a x ≥ f a 0) ∧ 
   (∀ a', (∀ x, x > 0 → f a' x > 0) → a' ≤ a)) := by sorry

end local_minimum_implies_a_equals_one_max_a_for_positive_f_main_result_l1833_183343


namespace team_combinations_theorem_l1833_183321

/-- The number of ways to select k elements from n elements --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid team combinations --/
def validCombinations (totalMale totalFemale teamSize : ℕ) : ℕ :=
  (choose totalMale 1 * choose totalFemale 2) +
  (choose totalMale 2 * choose totalFemale 1)

theorem team_combinations_theorem :
  validCombinations 5 4 3 = 70 := by sorry

end team_combinations_theorem_l1833_183321


namespace local_max_value_is_four_l1833_183382

/-- The function f(x) = x^3 - 3ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + 2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem local_max_value_is_four (a : ℝ) :
  (∃ x : ℝ, IsLocalMin (f a) 1) →
  (∃ x : ℝ, IsLocalMax (f a) x ∧ f a x = 4) :=
by sorry

end local_max_value_is_four_l1833_183382


namespace circle_radius_l1833_183316

/-- The circle C is defined by the equation x^2 + y^2 - 4x - 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The radius of a circle is the distance from its center to any point on the circle -/
def is_radius (r : ℝ) (center : ℝ × ℝ) (equation : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, equation x y → (x - center.1)^2 + (y - center.2)^2 = r^2

/-- The radius of the circle C defined by x^2 + y^2 - 4x - 2y + 1 = 0 is equal to 2 -/
theorem circle_radius : ∃ center, is_radius 2 center circle_equation := by
  sorry

end circle_radius_l1833_183316


namespace ny_mets_fans_count_l1833_183378

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The total number of baseball fans in the town -/
def total_fans : ℕ := 390

/-- Checks if the given fan counts satisfy the ratio conditions -/
def satisfies_ratios (fans : FanCounts) : Prop :=
  3 * fans.mets = 2 * fans.yankees ∧
  4 * fans.red_sox = 5 * fans.mets

/-- Checks if the given fan counts sum up to the total number of fans -/
def satisfies_total (fans : FanCounts) : Prop :=
  fans.yankees + fans.mets + fans.red_sox = total_fans

/-- The main theorem stating that there are 104 NY Mets fans -/
theorem ny_mets_fans_count :
  ∃ (fans : FanCounts),
    satisfies_ratios fans ∧
    satisfies_total fans ∧
    fans.mets = 104 :=
  sorry

end ny_mets_fans_count_l1833_183378


namespace gcd_condition_iff_special_form_l1833_183310

theorem gcd_condition_iff_special_form (n m : ℕ) (hn : n > 0) (hm : m > 0) :
  Nat.gcd ((n + 1)^m - n) ((n + 1)^(m+3) - n) > 1 ↔
  ∃ (k l : ℕ), k > 0 ∧ l > 0 ∧ n = 7*k - 6 ∧ m = 3*l :=
sorry

end gcd_condition_iff_special_form_l1833_183310


namespace trains_crossing_time_l1833_183387

/-- Proves that two trains of equal length traveling in opposite directions will cross each other in 12 seconds -/
theorem trains_crossing_time (length : ℝ) (time1 time2 : ℝ) 
  (h1 : length = 120)
  (h2 : time1 = 10)
  (h3 : time2 = 15) : 
  (2 * length) / (length / time1 + length / time2) = 12 := by
  sorry


end trains_crossing_time_l1833_183387


namespace min_n_constant_term_is_correct_l1833_183319

/-- The minimum positive integer n such that the expansion of (x^2 + 1/(2x^3))^n contains a constant term, where x is a positive integer. -/
def min_n_constant_term : ℕ := 5

/-- Predicate to check if the expansion of (x^2 + 1/(2x^3))^n contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ (k : ℕ), 2 * n = 5 * k

theorem min_n_constant_term_is_correct :
  (∀ m : ℕ, m < min_n_constant_term → ¬has_constant_term m) ∧
  has_constant_term min_n_constant_term :=
sorry

end min_n_constant_term_is_correct_l1833_183319


namespace min_trig_fraction_l1833_183328

theorem min_trig_fraction :
  (∀ x : ℝ, (Real.sin x)^6 + (Real.cos x)^6 + 1 ≥ 5/6 * ((Real.sin x)^4 + (Real.cos x)^4 + 1)) ∧
  (∃ x : ℝ, (Real.sin x)^6 + (Real.cos x)^6 + 1 = 5/6 * ((Real.sin x)^4 + (Real.cos x)^4 + 1)) :=
by sorry

end min_trig_fraction_l1833_183328


namespace heracles_age_l1833_183311

/-- Proves that Heracles' age is 10 years old given the conditions of the problem -/
theorem heracles_age : 
  ∀ (heracles_age : ℕ) (audrey_age : ℕ),
  audrey_age = heracles_age + 7 →
  audrey_age + 3 = 2 * heracles_age →
  heracles_age = 10 := by
sorry

end heracles_age_l1833_183311


namespace middle_term_is_plus_minus_six_l1833_183342

/-- The coefficient of the middle term in the expansion of (a ± 3b)² -/
def middle_term_coefficient (a b : ℝ) : Set ℝ :=
  {x : ℝ | ∃ (sign : ℝ) (h : sign = 1 ∨ sign = -1), 
    (a + sign * 3 * b)^2 = a^2 + x * a * b + 9 * b^2}

/-- Theorem stating that the coefficient of the middle term is either 6 or -6 -/
theorem middle_term_is_plus_minus_six (a b : ℝ) : 
  middle_term_coefficient a b = {6, -6} := by
sorry

end middle_term_is_plus_minus_six_l1833_183342


namespace regular_polygon_sides_l1833_183335

theorem regular_polygon_sides (n₁ n₂ : ℕ) : 
  n₁ % 2 = 0 → 
  n₂ % 2 = 0 → 
  (n₁ - 2) * 180 + (n₂ - 2) * 180 = 1800 → 
  ((n₁ = 4 ∧ n₂ = 10) ∨ (n₁ = 10 ∧ n₂ = 4) ∨ (n₁ = 6 ∧ n₂ = 8) ∨ (n₁ = 8 ∧ n₂ = 6)) :=
by sorry

end regular_polygon_sides_l1833_183335


namespace solution_set_part1_range_of_a_part2_l1833_183334

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |2*x - 5|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 2 x ≥ 5} = {x : ℝ | x ≤ 2 ∨ x ≥ 8/3} := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | a > 0 ∧ ∀ x ∈ Set.Icc a (2*a - 2), f a x ≤ |x + 4|} = Set.Ioo 2 (13/5) := by sorry

end solution_set_part1_range_of_a_part2_l1833_183334


namespace insufficient_pharmacies_l1833_183392

/-- Represents a grid of streets -/
structure StreetGrid where
  north_south : Nat
  west_east : Nat

/-- Represents a pharmacy's coverage area -/
structure PharmacyCoverage where
  width : Nat
  height : Nat

/-- Calculates the number of street segments in a grid -/
def streetSegments (grid : StreetGrid) : Nat :=
  2 * (grid.north_south - 1) * grid.west_east

/-- Calculates the number of intersections covered by a single pharmacy -/
def intersectionsCovered (coverage : PharmacyCoverage) : Nat :=
  (coverage.width - 1) * (coverage.height - 1)

/-- Theorem stating that 12 pharmacies are not enough to cover all street segments -/
theorem insufficient_pharmacies
  (grid : StreetGrid)
  (coverage : PharmacyCoverage)
  (h_grid : grid = { north_south := 10, west_east := 10 })
  (h_coverage : coverage = { width := 7, height := 7 })
  (h_pharmacies : Nat := 12) :
  h_pharmacies * intersectionsCovered coverage < streetSegments grid := by
  sorry

end insufficient_pharmacies_l1833_183392


namespace jogger_count_difference_l1833_183394

/-- Proves the difference in jogger counts between Christopher and Alexander --/
theorem jogger_count_difference :
  ∀ (christopher_count tyson_count alexander_count : ℕ),
  christopher_count = 80 →
  christopher_count = 20 * tyson_count →
  alexander_count = tyson_count + 22 →
  christopher_count - alexander_count = 54 := by
sorry

end jogger_count_difference_l1833_183394


namespace cube_surface_area_ratio_l1833_183354

theorem cube_surface_area_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (ratio : a = 7 * b) :
  (6 * a^2) / (6 * b^2) = 49 := by
  sorry

end cube_surface_area_ratio_l1833_183354


namespace binary_sum_equals_decimal_l1833_183337

/-- Converts a binary number represented as a sum of powers of 2 to its decimal equivalent -/
def binary_to_decimal (powers : List Nat) : Nat :=
  powers.foldl (fun acc p => acc + 2^p) 0

theorem binary_sum_equals_decimal : 
  let a := binary_to_decimal [0, 1, 2, 3, 4, 5, 6, 7, 8]  -- 111111111₂
  let b := binary_to_decimal [2, 3, 4, 5]                 -- 110110₂
  a + b = 571 := by sorry

end binary_sum_equals_decimal_l1833_183337


namespace polygon_sides_count_l1833_183331

theorem polygon_sides_count (n : ℕ) : 
  (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end polygon_sides_count_l1833_183331


namespace only_D_is_simple_random_sample_l1833_183300

/-- Represents a sampling method --/
inductive SamplingMethod
| A  -- Every 1 million postcards form a lottery group
| B  -- Sample a package every 30 minutes
| C  -- Draw from different staff categories
| D  -- Select 3 out of 10 products randomly

/-- Defines what constitutes a simple random sample --/
def isSimpleRandomSample (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.A => false  -- Fixed interval
  | SamplingMethod.B => false  -- Fixed interval
  | SamplingMethod.C => false  -- Stratified sampling
  | SamplingMethod.D => true   -- Equal probability for each item

/-- Theorem stating that only method D is a simple random sample --/
theorem only_D_is_simple_random_sample :
  ∀ (method : SamplingMethod), isSimpleRandomSample method ↔ method = SamplingMethod.D :=
by sorry

end only_D_is_simple_random_sample_l1833_183300


namespace vector_BC_l1833_183393

/-- Given vectors BA and CA in 2D space, prove that vector BC is their difference. -/
theorem vector_BC (BA CA : Fin 2 → ℝ) (h1 : BA = ![1, 2]) (h2 : CA = ![4, 5]) :
  BA - CA = ![-3, -3] := by
  sorry

end vector_BC_l1833_183393


namespace smallest_sum_of_reciprocals_l1833_183380

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 → 
  (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ) → 
  (x : ℕ) + (y : ℕ) = 45 :=
by sorry

end smallest_sum_of_reciprocals_l1833_183380


namespace circle_diameter_from_area_l1833_183339

theorem circle_diameter_from_area (A : Real) (π : Real) (h : π > 0) :
  A = 225 * π → ∃ d : Real, d > 0 ∧ A = π * (d / 2)^2 ∧ d = 30 := by
  sorry

end circle_diameter_from_area_l1833_183339


namespace loan_payoff_period_l1833_183390

-- Define the costs and monthly payment difference
def house_cost : ℕ := 480000
def trailer_cost : ℕ := 120000
def monthly_payment_diff : ℕ := 1500

-- Define the theorem
theorem loan_payoff_period :
  ∃ (n : ℕ), 
    n * trailer_cost = (n * monthly_payment_diff + trailer_cost) * house_cost ∧
    n = 20 * 12 :=
by sorry

end loan_payoff_period_l1833_183390


namespace min_hypotenuse_max_inscribed_circle_radius_l1833_183391

/-- A right-angled triangle with perimeter 1 meter -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  hypotenuse : ℝ
  perimeter_eq_one : a + b + hypotenuse = 1
  right_angle : a^2 + b^2 = hypotenuse^2
  positive : 0 < a ∧ 0 < b ∧ 0 < hypotenuse

/-- The minimum length of the hypotenuse in a right-angled triangle with perimeter 1 meter -/
theorem min_hypotenuse (t : RightTriangle) : t.hypotenuse ≥ Real.sqrt 2 - 1 := by sorry

/-- The maximum radius of the inscribed circle in a right-angled triangle with perimeter 1 meter -/
theorem max_inscribed_circle_radius (t : RightTriangle) : 
  t.a * t.b / (t.a + t.b + t.hypotenuse) ≤ 3/2 - Real.sqrt 2 := by sorry

end min_hypotenuse_max_inscribed_circle_radius_l1833_183391


namespace problem_solution_l1833_183383

theorem problem_solution (p_xavier p_yvonne p_zelda : ℚ) 
  (h_xavier : p_xavier = 1/5)
  (h_yvonne : p_yvonne = 1/2)
  (h_zelda : p_zelda = 5/8)
  (h_independent : True) -- Assumption of independence
  : p_xavier * p_yvonne * (1 - p_zelda) = 3/80 := by
  sorry

end problem_solution_l1833_183383


namespace fifteen_students_prefer_dogs_l1833_183312

/-- Represents the preferences of students in a class survey -/
structure ClassPreferences where
  total_students : ℕ
  dogs_videogames_chocolate : Rat
  dogs_videogames_vanilla : Rat
  dogs_movies_chocolate : Rat
  dogs_movies_vanilla : Rat
  cats_movies_chocolate : Rat
  cats_movies_vanilla : Rat
  cats_videogames_chocolate : Rat
  cats_videogames_vanilla : Rat

/-- Theorem stating that 15 students prefer dogs given the survey results -/
theorem fifteen_students_prefer_dogs (prefs : ClassPreferences) : 
  prefs.total_students = 30 ∧
  prefs.dogs_videogames_chocolate = 25/100 ∧
  prefs.dogs_videogames_vanilla = 5/100 ∧
  prefs.dogs_movies_chocolate = 10/100 ∧
  prefs.dogs_movies_vanilla = 10/100 ∧
  prefs.cats_movies_chocolate = 15/100 ∧
  prefs.cats_movies_vanilla = 10/100 ∧
  prefs.cats_videogames_chocolate = 5/100 ∧
  prefs.cats_videogames_vanilla = 10/100 →
  (prefs.dogs_videogames_chocolate + prefs.dogs_videogames_vanilla + 
   prefs.dogs_movies_chocolate + prefs.dogs_movies_vanilla) * prefs.total_students = 15 := by
  sorry


end fifteen_students_prefer_dogs_l1833_183312


namespace lower_right_is_two_l1833_183379

/-- Represents a 5x5 grid of integers -/
def Grid := Fin 5 → Fin 5 → Nat

/-- Check if all numbers in a list are distinct -/
def allDistinct (l : List Nat) : Prop := l.Nodup

/-- Check if a grid satisfies the row constraint -/
def validRows (g : Grid) : Prop :=
  ∀ i, allDistinct [g i 0, g i 1, g i 2, g i 3, g i 4]

/-- Check if a grid satisfies the column constraint -/
def validColumns (g : Grid) : Prop :=
  ∀ j, allDistinct [g 0 j, g 1 j, g 2 j, g 3 j, g 4 j]

/-- Check if all numbers in the grid are between 1 and 5 -/
def validNumbers (g : Grid) : Prop :=
  ∀ i j, 1 ≤ g i j ∧ g i j ≤ 5

/-- Check if the sum of the first row is 15 -/
def firstRowSum15 (g : Grid) : Prop :=
  g 0 0 + g 0 1 + g 0 2 + g 0 3 + g 0 4 = 15

/-- Check if the given numbers in the grid match the problem description -/
def matchesGivenNumbers (g : Grid) : Prop :=
  g 0 0 = 1 ∧ g 0 2 = 3 ∧ g 0 3 = 4 ∧
  g 1 0 = 5 ∧ g 1 2 = 1 ∧ g 1 4 = 3 ∧
  g 2 1 = 4 ∧ g 2 3 = 5 ∧
  g 3 0 = 4

theorem lower_right_is_two (g : Grid) 
  (hrows : validRows g)
  (hcols : validColumns g)
  (hnums : validNumbers g)
  (hsum : firstRowSum15 g)
  (hgiven : matchesGivenNumbers g) :
  g 4 4 = 2 := by
  sorry

end lower_right_is_two_l1833_183379


namespace bridge_concrete_total_l1833_183318

/-- The amount of concrete needed for a bridge -/
structure BridgeConcrete where
  roadway_deck : ℕ
  single_anchor : ℕ
  num_anchors : ℕ
  supporting_pillars : ℕ

/-- The total amount of concrete needed for the bridge -/
def total_concrete (b : BridgeConcrete) : ℕ :=
  b.roadway_deck + b.single_anchor * b.num_anchors + b.supporting_pillars

/-- Theorem: The total amount of concrete needed for the bridge is 4800 tons -/
theorem bridge_concrete_total :
  let b : BridgeConcrete := {
    roadway_deck := 1600,
    single_anchor := 700,
    num_anchors := 2,
    supporting_pillars := 1800
  }
  total_concrete b = 4800 := by sorry

end bridge_concrete_total_l1833_183318


namespace pears_picked_total_l1833_183359

/-- The number of pears Mike picked -/
def mike_pears : ℕ := 8

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 7

/-- The total number of pears picked -/
def total_pears : ℕ := mike_pears + jason_pears

theorem pears_picked_total : total_pears = 15 := by
  sorry

end pears_picked_total_l1833_183359


namespace third_pair_weight_l1833_183371

def dumbbell_system (weight1 weight2 weight3 : ℕ) : Prop :=
  weight1 * 2 + weight2 * 2 + weight3 * 2 = 32

theorem third_pair_weight :
  ∃ (weight3 : ℕ), dumbbell_system 3 5 weight3 ∧ weight3 = 16 :=
by
  sorry

end third_pair_weight_l1833_183371


namespace art_gallery_pieces_l1833_183350

theorem art_gallery_pieces (total : ℕ) 
  (h1 : total / 3 = total / 3)  -- 1/3 of pieces are on display
  (h2 : (total / 3) / 6 = (total / 3) / 6)  -- 1/6 of displayed pieces are sculptures
  (h3 : (total * 2 / 3) / 3 = (total * 2 / 3) / 3)  -- 1/3 of non-displayed pieces are paintings
  (h4 : total * 2 / 3 * 2 / 3 = 800)  -- 800 sculptures are not on display
  : total = 1800 := by
  sorry

end art_gallery_pieces_l1833_183350


namespace smallest_max_sum_l1833_183324

theorem smallest_max_sum (a b c d e : ℕ+) 
  (sum_eq : a + b + c + d + e = 3060)
  (ae_lower_bound : a + e ≥ 1300) :
  let M := max (a + b) (max (b + c) (max (c + d) (d + e)))
  ∀ (a' b' c' d' e' : ℕ+),
    a' + b' + c' + d' + e' = 3060 →
    a' + e' ≥ 1300 →
    max (a' + b') (max (b' + c') (max (c' + d') (d' + e'))) ≥ 1174 :=
by sorry

end smallest_max_sum_l1833_183324


namespace line_plane_perpendicularity_parallelism_l1833_183325

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity_parallelism
  (m n : Line) (α β : Plane)
  (h_different_lines : m ≠ n)
  (h_different_planes : α ≠ β)
  (h_m_perp_α : perpendicular m α)
  (h_n_parallel_β : parallel_line_plane n β)
  (h_α_parallel_β : parallel_plane α β) :
  perpendicular_lines m n :=
sorry

end line_plane_perpendicularity_parallelism_l1833_183325


namespace partnership_profit_theorem_l1833_183384

/-- Calculates the total profit for a partnership given investments and one partner's profit share -/
def calculate_total_profit (tom_investment : ℕ) (jose_investment : ℕ) (tom_months : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : ℕ :=
  let tom_total := tom_investment * tom_months
  let jose_total := jose_investment * jose_months
  let ratio_sum := (tom_total / (tom_total.gcd jose_total)) + (jose_total / (tom_total.gcd jose_total))
  (ratio_sum * jose_profit) / (jose_total / (tom_total.gcd jose_total))

theorem partnership_profit_theorem (tom_investment jose_investment tom_months jose_months jose_profit : ℕ) 
  (h1 : tom_investment = 30000)
  (h2 : jose_investment = 45000)
  (h3 : tom_months = 12)
  (h4 : jose_months = 10)
  (h5 : jose_profit = 40000) :
  calculate_total_profit tom_investment jose_investment tom_months jose_months jose_profit = 72000 := by
  sorry

#eval calculate_total_profit 30000 45000 12 10 40000

end partnership_profit_theorem_l1833_183384


namespace unique_function_property_l1833_183317

theorem unique_function_property (f : ℕ → ℕ) :
  (f 1 > 0) ∧
  (∀ m n : ℕ, f (m^2 + n^2) = (f m)^2 + (f n)^2) →
  (∀ n : ℕ, f n = n) :=
by sorry

end unique_function_property_l1833_183317


namespace mexican_olympiad_1988_l1833_183304

theorem mexican_olympiad_1988 (f : ℕ+ → ℕ+) 
  (h : ∀ m n : ℕ+, f (f m + f n) = m + n) : 
  f 1988 = 1988 := by sorry

end mexican_olympiad_1988_l1833_183304


namespace necessary_but_not_sufficient_l1833_183327

/-- The function f(x) = ax + 3 --/
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

/-- The zero point of f(x) is in the interval (-1, 2) --/
def has_zero_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 < x ∧ x < 2 ∧ f a x = 0

/-- The statement is a necessary but not sufficient condition --/
theorem necessary_but_not_sufficient :
  (∀ a : ℝ, 3 < a ∧ a < 4 → has_zero_in_interval a) ∧
  (∃ a : ℝ, has_zero_in_interval a ∧ (a ≤ 3 ∨ 4 ≤ a)) :=
sorry

end necessary_but_not_sufficient_l1833_183327


namespace justice_plants_l1833_183307

theorem justice_plants (ferns palms succulents total_wanted : ℕ) : 
  ferns = 3 → palms = 5 → succulents = 7 → total_wanted = 24 →
  total_wanted - (ferns + palms + succulents) = 9 := by
sorry

end justice_plants_l1833_183307


namespace cos_double_angle_for_tan_two_l1833_183375

theorem cos_double_angle_for_tan_two (θ : Real) (h : Real.tan θ = 2) : 
  Real.cos (2 * θ) = -3/5 := by
  sorry

end cos_double_angle_for_tan_two_l1833_183375


namespace increasing_continuous_function_intermediate_values_l1833_183369

theorem increasing_continuous_function_intermediate_values 
  (f : ℝ → ℝ) (M N : ℝ) :
  (∀ x y, x ∈ Set.Icc 0 2 → y ∈ Set.Icc 0 2 → x < y → f x < f y) →
  ContinuousOn f (Set.Icc 0 2) →
  f 0 = M →
  f 2 = N →
  M > 0 →
  N > 0 →
  (∃ x₁ ∈ Set.Icc 0 2, f x₁ = (M + N) / 2) ∧
  (∃ x₂ ∈ Set.Icc 0 2, f x₂ = Real.sqrt (M * N)) :=
by sorry

end increasing_continuous_function_intermediate_values_l1833_183369


namespace smartphone_cost_smartphone_cost_proof_l1833_183360

theorem smartphone_cost (initial_savings : ℕ) (saving_months : ℕ) (weeks_per_month : ℕ) (weekly_savings : ℕ) : ℕ :=
  let total_weeks := saving_months * weeks_per_month
  let total_savings := weekly_savings * total_weeks
  initial_savings + total_savings

#check smartphone_cost 40 2 4 15 = 160

theorem smartphone_cost_proof :
  smartphone_cost 40 2 4 15 = 160 := by
  sorry

end smartphone_cost_smartphone_cost_proof_l1833_183360


namespace prime_sum_fraction_l1833_183336

theorem prime_sum_fraction (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r)
  (hdistinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (ha : ∃ (a : ℕ), a = (p + q) / r + (q + r) / p + (r + p) / q) :
  ∃ (a : ℕ), a = 7 := by
sorry

end prime_sum_fraction_l1833_183336


namespace fixed_point_on_line_l1833_183363

theorem fixed_point_on_line (m : ℝ) : 
  (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end fixed_point_on_line_l1833_183363


namespace animal_sanctuary_l1833_183340

theorem animal_sanctuary (total : ℕ) (difference : ℕ) : total = 450 ∧ difference = 75 → ∃ (dogs cats : ℕ), cats = dogs + difference ∧ dogs + cats = total ∧ cats = 262 := by
  sorry

end animal_sanctuary_l1833_183340


namespace remainder_theorem_l1833_183399

theorem remainder_theorem (n : ℤ) (h : n % 7 = 2) : (3 * n - 7) % 7 = 6 := by
  sorry

end remainder_theorem_l1833_183399


namespace angle_with_special_supplement_complement_l1833_183373

theorem angle_with_special_supplement_complement : ∃ (x : ℝ), 
  0 < x ∧ x < 180 ∧ (180 - x) = 5 * (90 - x) ∧ x = 67.5 := by
  sorry

end angle_with_special_supplement_complement_l1833_183373


namespace interest_equality_second_sum_l1833_183385

/-- Given a total sum divided into two parts, where the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years
    at 5% per annum, prove that the second part is equal to 1680 rupees. -/
theorem interest_equality_second_sum (total : ℚ) (first_part : ℚ) (second_part : ℚ) :
  total = 2730 →
  total = first_part + second_part →
  (first_part * 3 * 8) / 100 = (second_part * 5 * 3) / 100 →
  second_part = 1680 := by
  sorry

end interest_equality_second_sum_l1833_183385


namespace all_but_one_are_sum_of_two_primes_l1833_183353

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p + q = n

theorem all_but_one_are_sum_of_two_primes :
  ∀ k : ℕ, k > 0 → is_sum_of_two_primes (1 + 10 * k) :=
by sorry

end all_but_one_are_sum_of_two_primes_l1833_183353


namespace negation_of_proposition_l1833_183351

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, |x| < 0) ↔ (∃ x : ℝ, |x| ≥ 0) :=
by sorry

end negation_of_proposition_l1833_183351


namespace projection_vector_l1833_183398

def a : Fin 3 → ℝ := ![0, 1, 1]
def b : Fin 3 → ℝ := ![1, 1, 0]

theorem projection_vector :
  let proj_a_b := (a • b) / (a • a) • a
  proj_a_b = ![0, 1/2, 1/2] := by
sorry

end projection_vector_l1833_183398


namespace sin_780_degrees_l1833_183329

theorem sin_780_degrees : Real.sin (780 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_780_degrees_l1833_183329


namespace reciprocal_sum_problem_l1833_183372

theorem reciprocal_sum_problem (x y z : ℝ) 
  (h1 : 1/x + 1/y + 1/z = 2) 
  (h2 : 1/x^2 + 1/y^2 + 1/z^2 = 1) : 
  1/(x*y) + 1/(y*z) + 1/(z*x) = 3/2 := by
  sorry

end reciprocal_sum_problem_l1833_183372


namespace farm_chickens_l1833_183309

theorem farm_chickens (total : ℕ) 
  (h1 : (total : ℚ) * (1/5) = (total : ℚ) * (20/100))  -- 20% of chickens are BCM
  (h2 : ((total : ℚ) * (1/5)) * (4/5) = ((total : ℚ) * (20/100)) * (80/100))  -- 80% of BCM are hens
  (h3 : ((total : ℚ) * (1/5)) * (4/5) = 16)  -- There are 16 BCM hens
  : total = 100 := by
sorry

end farm_chickens_l1833_183309


namespace min_gold_chips_l1833_183396

/-- Represents a box of chips with gold, silver, and bronze chips. -/
structure ChipBox where
  gold : ℕ
  silver : ℕ
  bronze : ℕ

/-- Checks if a ChipBox satisfies the given conditions. -/
def isValidChipBox (box : ChipBox) : Prop :=
  box.bronze ≥ 2 * box.silver ∧
  box.bronze ≤ box.gold / 4 ∧
  box.silver + box.bronze ≥ 75

/-- Theorem stating the minimum number of gold chips in a valid ChipBox. -/
theorem min_gold_chips (box : ChipBox) :
  isValidChipBox box → box.gold ≥ 200 := by
  sorry

#check min_gold_chips

end min_gold_chips_l1833_183396


namespace strawberry_pancakes_l1833_183306

theorem strawberry_pancakes (total : ℕ) (blueberry : ℕ) (banana : ℕ) (chocolate : ℕ) 
  (h1 : total = 150)
  (h2 : blueberry = 45)
  (h3 : banana = 60)
  (h4 : chocolate = 25) :
  total - (blueberry + banana + chocolate) = 20 := by
  sorry

end strawberry_pancakes_l1833_183306


namespace calculation_proof_inequality_system_solution_l1833_183314

-- Problem 1
theorem calculation_proof : 
  Real.sqrt 4 + 2 * Real.sin (45 * π / 180) - (π - 3)^0 + |Real.sqrt 2 - 2| = 3 := by sorry

-- Problem 2
theorem inequality_system_solution (x : ℝ) : 
  (2 * (x + 2) - x ≤ 5 ∧ (4 * x + 1) / 3 > x - 1) ↔ (-4 < x ∧ x ≤ 1) := by sorry

end calculation_proof_inequality_system_solution_l1833_183314


namespace sum_of_roots_cubic_l1833_183397

theorem sum_of_roots_cubic (x₁ x₂ x₃ k m : ℝ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)
  (h_root₁ : 2 * x₁^3 - k * x₁ = m)
  (h_root₂ : 2 * x₂^3 - k * x₂ = m)
  (h_root₃ : 2 * x₃^3 - k * x₃ = m) :
  x₁ + x₂ + x₃ = 0 := by
sorry

end sum_of_roots_cubic_l1833_183397


namespace democrats_ratio_l1833_183366

/-- Proves that the ratio of democrats to total participants is 1:3 -/
theorem democrats_ratio (total : ℕ) (female_democrats : ℕ) :
  total = 810 →
  female_democrats = 135 →
  let female := 2 * female_democrats
  let male := total - female
  let male_democrats := male / 4
  let total_democrats := female_democrats + male_democrats
  (total_democrats : ℚ) / total = 1 / 3 := by
  sorry

end democrats_ratio_l1833_183366


namespace isosceles_trapezoid_side_length_l1833_183322

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ
  side : ℝ

/-- The theorem stating the relationship between the trapezoid's properties -/
theorem isosceles_trapezoid_side_length 
  (t : IsoscelesTrapezoid) 
  (h1 : t.base1 = 9) 
  (h2 : t.base2 = 15) 
  (h3 : t.area = 48) : 
  t.side = 5 := by sorry

end isosceles_trapezoid_side_length_l1833_183322


namespace least_addition_for_divisibility_l1833_183381

def divisors : List Nat := [5, 7, 11, 13, 17, 19]

theorem least_addition_for_divisibility (x : Nat) : 
  (∀ d ∈ divisors, (5432 + x) % d = 0) ∧
  (∀ y < x, ∃ d ∈ divisors, (5432 + y) % d ≠ 0) →
  x = 1611183 := by sorry

end least_addition_for_divisibility_l1833_183381


namespace inequality_solution_set_l1833_183330

theorem inequality_solution_set (x : ℝ) : 
  (x - 5) / (x + 1) ≤ 0 ∧ x + 1 ≠ 0 ↔ x ∈ Set.Ioc (-1) 5 :=
by sorry

end inequality_solution_set_l1833_183330


namespace limit_of_exponential_l1833_183364

theorem limit_of_exponential (a : ℝ) :
  (a > 1 → ∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → a^x > M) ∧
  (0 < a ∧ a < 1 → ∀ ε : ℝ, ε > 0 → ∃ N : ℝ, ∀ x : ℝ, x > N → a^x < ε) :=
by sorry

end limit_of_exponential_l1833_183364


namespace shaded_triangles_area_sum_l1833_183313

/-- The sum of areas of shaded triangles in an infinite geometric series --/
theorem shaded_triangles_area_sum (x y z : ℝ) (h1 : x = 8) (h2 : y = 8) (h3 : z = 8) 
  (h4 : x^2 = y^2 + z^2) : 
  let initial_area := (1/2) * y * z
  let first_shaded_area := (1/4) * initial_area
  let ratio := (1/4 : ℝ)
  (initial_area * ratio) / (1 - ratio) = 32/3 := by
  sorry

end shaded_triangles_area_sum_l1833_183313


namespace shooting_outcomes_l1833_183303

/-- Represents the number of shots -/
def num_shots : ℕ := 6

/-- Represents the number of hits we're interested in -/
def num_hits : ℕ := 3

/-- Calculates the total number of possible outcomes for n shots -/
def total_outcomes (n : ℕ) : ℕ := 2^n

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of outcomes with exactly k hits out of n shots -/
def outcomes_with_k_hits (n k : ℕ) : ℕ := choose n k

/-- Calculates the number of outcomes with exactly k hits and exactly 2 consecutive hits out of n shots -/
def outcomes_with_k_hits_and_2_consecutive (n k : ℕ) : ℕ := choose (n - k + 1) 2

theorem shooting_outcomes :
  (total_outcomes num_shots = 64) ∧
  (outcomes_with_k_hits num_shots num_hits = 20) ∧
  (outcomes_with_k_hits_and_2_consecutive num_shots num_hits = 6) := by
  sorry

end shooting_outcomes_l1833_183303


namespace monotonic_increasing_implies_a_eq_neg_six_l1833_183352

/-- The function f(x) defined as the absolute value of 2x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + a|

/-- The property of f being monotonically increasing on [3, +∞) -/
def monotonic_increasing_from_three (a : ℝ) : Prop :=
  ∀ x y, 3 ≤ x → x ≤ y → f a x ≤ f a y

/-- Theorem stating that a must be -6 for f to be monotonically increasing on [3, +∞) -/
theorem monotonic_increasing_implies_a_eq_neg_six :
  ∃ a, monotonic_increasing_from_three a ↔ a = -6 :=
sorry

end monotonic_increasing_implies_a_eq_neg_six_l1833_183352


namespace steven_amanda_hike_difference_l1833_183349

/-- The number of hikes Camila has gone on -/
def camila_hikes : ℕ := 7

/-- The number of times Amanda has gone hiking compared to Camila -/
def amanda_multiplier : ℕ := 8

/-- The number of hikes Amanda has gone on -/
def amanda_hikes : ℕ := camila_hikes * amanda_multiplier

/-- The number of hikes Camila plans to go on per week -/
def camila_weekly_plan : ℕ := 4

/-- The number of weeks Camila plans to hike to match Steven -/
def camila_weeks_plan : ℕ := 16

/-- The total number of hikes Camila aims for to match Steven -/
def steven_hikes : ℕ := camila_hikes + camila_weekly_plan * camila_weeks_plan

theorem steven_amanda_hike_difference :
  steven_hikes - amanda_hikes = 15 := by
  sorry

end steven_amanda_hike_difference_l1833_183349


namespace functional_equation_zero_value_l1833_183315

theorem functional_equation_zero_value 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * y) = f x + f y) : 
  f 0 = 0 := by
sorry

end functional_equation_zero_value_l1833_183315


namespace wage_increase_hours_decrease_l1833_183355

theorem wage_increase_hours_decrease (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  let new_wage := 1.5 * w
  let new_hours := h / 1.5
  let percent_decrease := 100 * (1 - 1 / 1.5)
  new_wage * new_hours = w * h ∧ 
  100 * (h - new_hours) / h = percent_decrease := by
  sorry

end wage_increase_hours_decrease_l1833_183355


namespace rotation_of_point_transformed_curve_equation_l1833_183362

def rotation_pi_over_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), p.1)

def transformation_T2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.2)

def compose_transformations (f g : ℝ × ℝ → ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  f (g p)

def parabola (x : ℝ) : ℝ := x^2

theorem rotation_of_point :
  rotation_pi_over_2 (2, 1) = (-1, 2) := by sorry

theorem transformed_curve_equation (x y : ℝ) :
  (∃ t : ℝ, compose_transformations transformation_T2 rotation_pi_over_2 (t, parabola t) = (x, y)) ↔
  y - x = y^2 := by sorry

end rotation_of_point_transformed_curve_equation_l1833_183362


namespace stratified_sample_size_is_15_l1833_183368

/-- Represents the number of workers in each age group -/
structure WorkerGroups where
  young : Nat
  middle_aged : Nat
  older : Nat

/-- Calculates the total sample size for a stratified sample -/
def stratified_sample_size (workers : WorkerGroups) (young_sample : Nat) : Nat :=
  let total_workers := workers.young + workers.middle_aged + workers.older
  let sampling_ratio := workers.young / young_sample
  total_workers / sampling_ratio

/-- Theorem: The stratified sample size for the given worker distribution is 15 -/
theorem stratified_sample_size_is_15 :
  let workers : WorkerGroups := ⟨35, 25, 15⟩
  stratified_sample_size workers 7 = 15 := by
  sorry

#eval stratified_sample_size ⟨35, 25, 15⟩ 7

end stratified_sample_size_is_15_l1833_183368


namespace tan_3_negative_l1833_183338

theorem tan_3_negative : Real.tan 3 < 0 := by
  sorry

end tan_3_negative_l1833_183338


namespace fruit_shop_problem_l1833_183388

/-- Fruit shop problem -/
theorem fruit_shop_problem 
  (may_total : ℝ) 
  (may_cost_A may_cost_B : ℝ)
  (june_cost_A june_cost_B : ℝ)
  (june_increase : ℝ)
  (june_total_quantity : ℝ)
  (h_may_total : may_total = 1700)
  (h_may_cost_A : may_cost_A = 8)
  (h_may_cost_B : may_cost_B = 18)
  (h_june_cost_A : june_cost_A = 10)
  (h_june_cost_B : june_cost_B = 20)
  (h_june_increase : june_increase = 300)
  (h_june_total_quantity : june_total_quantity = 120) :
  ∃ (may_quantity_A may_quantity_B : ℝ),
    may_quantity_A * may_cost_A + may_quantity_B * may_cost_B = may_total ∧
    may_quantity_A * june_cost_A + may_quantity_B * june_cost_B = may_total + june_increase ∧
    may_quantity_A = 100 ∧
    may_quantity_B = 50 ∧
    (∃ (june_quantity_A : ℝ),
      june_quantity_A ≤ 3 * (june_total_quantity - june_quantity_A) ∧
      june_quantity_A * june_cost_A + (june_total_quantity - june_quantity_A) * june_cost_B = 1500 ∧
      ∀ (other_june_quantity_A : ℝ),
        other_june_quantity_A ≤ 3 * (june_total_quantity - other_june_quantity_A) →
        other_june_quantity_A * june_cost_A + (june_total_quantity - other_june_quantity_A) * june_cost_B ≥ 1500) :=
by sorry

end fruit_shop_problem_l1833_183388


namespace average_string_length_l1833_183308

theorem average_string_length : 
  let string1 : ℚ := 2
  let string2 : ℚ := 5
  let string3 : ℚ := 7
  let total_length : ℚ := string1 + string2 + string3
  let num_strings : ℕ := 3
  (total_length / num_strings) = 14 / 3 := by
sorry

end average_string_length_l1833_183308


namespace union_of_M_and_N_l1833_183301

def M : Set ℝ := {x | |x| = 1}
def N : Set ℝ := {x | x^2 ≠ x}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end union_of_M_and_N_l1833_183301


namespace quadratic_transformation_l1833_183302

-- Define the quadratic function
def quadratic (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

-- Define the transformed quadratic function
def transformed_quadratic (m h k : ℝ) (x : ℝ) : ℝ := m * (x - h)^2 + k

-- State the theorem
theorem quadratic_transformation (p q r : ℝ) :
  (∃ m k : ℝ, ∀ x : ℝ, quadratic p q r x = transformed_quadratic 5 3 15 x) →
  (∃ m k : ℝ, ∀ x : ℝ, quadratic (4*p) (4*q) (4*r) x = transformed_quadratic m 3 k x) :=
by sorry

end quadratic_transformation_l1833_183302
