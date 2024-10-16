import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l303_30303

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a_1 * a_3 = 4 and a_4 = 8, then a_1 + q = 3 or a_1 + q = -3 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 * a 3 = 4 →               -- given condition
  a 4 = 8 →                     -- given condition
  (a 1 + q = 3 ∨ a 1 + q = -3)  -- conclusion
  := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l303_30303


namespace NUMINAMATH_CALUDE_triangle_perimeter_l303_30327

/-- Given a large square of side length y and a smaller square of side length x inside it,
    with four congruent right-angled triangles filling the corners,
    the perimeter of one such triangle is (y-x)(1+√2)/√2 -/
theorem triangle_perimeter (x y : ℝ) (h : 0 < x ∧ x < y) :
  let leg := (y - x) / 2
  let hypotenuse := (y - x) / Real.sqrt 2
  leg + leg + hypotenuse = (y - x) * (1 + Real.sqrt 2) / Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_perimeter_l303_30327


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_800_l303_30347

-- Define the sum of positive divisors function
def sum_of_divisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem distinct_prime_factors_of_divisor_sum_800 :
  (Nat.factors (sum_of_divisors 800)).length = 4 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_800_l303_30347


namespace NUMINAMATH_CALUDE_no_solution_exists_l303_30309

/-- Set A defined as {(x, y) | x = n, y = na + b, n ∈ ℤ} -/
def A (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ n : ℤ, p.1 = n ∧ p.2 = n * a + b}

/-- Set B defined as {(x, y) | x = m, y = 3m^2 + 15, m ∈ ℤ} -/
def B : Set (ℝ × ℝ) :=
  {p | ∃ m : ℤ, p.1 = m ∧ p.2 = 3 * m^2 + 15}

/-- Set C defined as {(x, y) | x^2 + y^2 ≤ 144, x, y ∈ ℝ} -/
def C : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 ≤ 144}

/-- Theorem stating that there do not exist real numbers a and b satisfying both conditions -/
theorem no_solution_exists : ¬∃ a b : ℝ, (A a b ∩ B).Nonempty ∧ (a, b) ∈ C := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l303_30309


namespace NUMINAMATH_CALUDE_trig_expression_value_l303_30339

theorem trig_expression_value (α : Real) (h : Real.tan (α / 2) = 4) :
  (6 * Real.sin α - 7 * Real.cos α + 1) / (8 * Real.sin α + 9 * Real.cos α - 1) = -85/44 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l303_30339


namespace NUMINAMATH_CALUDE_total_lives_calculation_l303_30340

theorem total_lives_calculation (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) : 
  initial_players = 8 → additional_players = 2 → lives_per_player = 6 →
  (initial_players + additional_players) * lives_per_player = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_calculation_l303_30340


namespace NUMINAMATH_CALUDE_min_pumps_for_given_reservoir_l303_30389

/-- Represents the characteristics of a reservoir with a leakage problem -/
structure Reservoir where
  single_pump_time : ℝ
  double_pump_time : ℝ
  target_time : ℝ

/-- Calculates the minimum number of pumps needed to fill the reservoir within the target time -/
def min_pumps_needed (r : Reservoir) : ℕ :=
  sorry

/-- Theorem stating that for the given reservoir conditions, at least 3 pumps are needed -/
theorem min_pumps_for_given_reservoir :
  let r : Reservoir := {
    single_pump_time := 8,
    double_pump_time := 3.2,
    target_time := 2
  }
  min_pumps_needed r = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_pumps_for_given_reservoir_l303_30389


namespace NUMINAMATH_CALUDE_pancakes_theorem_l303_30375

/-- The number of pancakes left after Bobby and his dog eat some. -/
def pancakes_left (total : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) : ℕ :=
  total - (bobby_ate + dog_ate)

/-- Theorem: Given 21 pancakes, if Bobby eats 5 and his dog eats 7, there are 9 pancakes left. -/
theorem pancakes_theorem : pancakes_left 21 5 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pancakes_theorem_l303_30375


namespace NUMINAMATH_CALUDE_smallest_ends_in_9_divisible_by_13_l303_30354

/-- A positive integer that ends in 9 -/
def EndsIn9 (n : ℕ) : Prop := n % 10 = 9 ∧ n > 0

/-- The smallest positive integer that ends in 9 and is divisible by 13 -/
def SmallestEndsIn9DivisibleBy13 : ℕ := 99

theorem smallest_ends_in_9_divisible_by_13 :
  EndsIn9 SmallestEndsIn9DivisibleBy13 ∧
  SmallestEndsIn9DivisibleBy13 % 13 = 0 ∧
  ∀ n : ℕ, EndsIn9 n ∧ n % 13 = 0 → n ≥ SmallestEndsIn9DivisibleBy13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_ends_in_9_divisible_by_13_l303_30354


namespace NUMINAMATH_CALUDE_complex_power_sum_l303_30371

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^24 + 1/z^24 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l303_30371


namespace NUMINAMATH_CALUDE_tetrahedron_circumsphere_radius_l303_30318

/-- Given a tetrahedron P-ABC with edge lengths PA = BC = √6, PB = AC = √8, and PC = AB = √10,
    the radius of its circumsphere is √3. -/
theorem tetrahedron_circumsphere_radius 
  (P A B C : EuclideanSpace ℝ (Fin 3))
  (h_PA : ‖P - A‖ = Real.sqrt 6)
  (h_BC : ‖B - C‖ = Real.sqrt 6)
  (h_PB : ‖P - B‖ = Real.sqrt 8)
  (h_AC : ‖A - C‖ = Real.sqrt 8)
  (h_PC : ‖P - C‖ = Real.sqrt 10)
  (h_AB : ‖A - B‖ = Real.sqrt 10) :
  ∃ (center : EuclideanSpace ℝ (Fin 3)), 
    (‖center - P‖ = Real.sqrt 3 ∧ 
     ‖center - A‖ = Real.sqrt 3 ∧ 
     ‖center - B‖ = Real.sqrt 3 ∧ 
     ‖center - C‖ = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_circumsphere_radius_l303_30318


namespace NUMINAMATH_CALUDE_area_FYG_value_l303_30386

/-- Represents a trapezoid EFGH with point Y at the intersection of diagonals -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  area : ℝ

/-- The area of triangle FYG in the given trapezoid -/
def area_FYG (t : Trapezoid) : ℝ := sorry

theorem area_FYG_value (t : Trapezoid) (h1 : t.EF = 15) (h2 : t.GH = 25) (h3 : t.area = 200) :
  area_FYG t = 46.875 := by sorry

end NUMINAMATH_CALUDE_area_FYG_value_l303_30386


namespace NUMINAMATH_CALUDE_merchant_scale_problem_merchant_loss_l303_30335

theorem merchant_scale_problem (m n : ℝ) (hm : m > 0) (hn : n > 0) (hne : m ≠ n) :
  m / n + n / m > 2 :=
sorry

theorem merchant_loss (m n : ℝ) (hm : m > 0) (hn : n > 0) (hne : m ≠ n) :
  let x := n / m
  let y := m / n
  x + y > 2 :=
sorry

end NUMINAMATH_CALUDE_merchant_scale_problem_merchant_loss_l303_30335


namespace NUMINAMATH_CALUDE_product_of_large_integers_l303_30373

theorem product_of_large_integers : ∃ (A B : ℕ), 
  (A > 2009^182) ∧ 
  (B > 2009^182) ∧ 
  (3^2008 + 4^2009 = A * B) := by
sorry

end NUMINAMATH_CALUDE_product_of_large_integers_l303_30373


namespace NUMINAMATH_CALUDE_average_equality_implies_z_l303_30300

theorem average_equality_implies_z (z : ℝ) : 
  (8 + 11 + 20) / 3 = (14 + z) / 2 → z = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_equality_implies_z_l303_30300


namespace NUMINAMATH_CALUDE_irrational_floor_congruence_l303_30395

theorem irrational_floor_congruence (k : ℕ) (h : k ≥ 2) :
  ∃ r : ℝ, Irrational r ∧ ∀ m : ℕ, (⌊r^m⌋ : ℤ) ≡ -1 [ZMOD k] :=
sorry

end NUMINAMATH_CALUDE_irrational_floor_congruence_l303_30395


namespace NUMINAMATH_CALUDE_tree_growth_problem_l303_30374

/-- A tree growth problem -/
theorem tree_growth_problem (initial_height : ℝ) (growth_rate : ℝ) (initial_age : ℝ) (target_height : ℝ) :
  initial_height = 5 →
  growth_rate = 3 →
  initial_age = 1 →
  target_height = 23 →
  ∃ (years : ℝ), 
    initial_height + growth_rate * years = target_height ∧
    years + initial_age = 7 :=
by sorry

end NUMINAMATH_CALUDE_tree_growth_problem_l303_30374


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l303_30384

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
  (∀ x, 3 * x^2 + 1 = 6 * x) →
  (∀ x, a * x^2 + b * x + c = 0) →
  b = 6 →
  a = -3 ∧ c = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l303_30384


namespace NUMINAMATH_CALUDE_birth_year_problem_l303_30381

theorem birth_year_problem (x : ℕ) (h1 : x^2 - 2*x ≥ 1900) (h2 : x^2 - 2*x < 1950) : 
  (x^2 - 2*x + x = 1936) := by
  sorry

end NUMINAMATH_CALUDE_birth_year_problem_l303_30381


namespace NUMINAMATH_CALUDE_two_digit_number_division_l303_30377

theorem two_digit_number_division (x y : ℕ) : 
  (1 ≤ x) ∧ (x ≤ 9) ∧ (0 ≤ y) ∧ (y ≤ 9) →
  (10 * x + y) / (x + y) = 7 ∧ (10 * x + y) % (x + y) = 6 →
  (10 * x + y = 62) ∨ (10 * x + y = 83) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_division_l303_30377


namespace NUMINAMATH_CALUDE_max_value_sum_l303_30307

theorem max_value_sum (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (h_sum : a^2 + b^2 + c^2 + d^2 + e^2 = 504) :
  ∃ (N a_N b_N c_N d_N e_N : ℝ),
    (∀ x y z w v : ℝ, x > 0 → y > 0 → z > 0 → w > 0 → v > 0 → 
      x^2 + y^2 + z^2 + w^2 + v^2 = 504 → 
      x*z + 3*y*z + 4*z*w + 8*z*v ≤ N) ∧
    (a_N*c_N + 3*b_N*c_N + 4*c_N*d_N + 8*c_N*e_N = N) ∧
    (a_N^2 + b_N^2 + c_N^2 + d_N^2 + e_N^2 = 504) ∧
    (N + a_N + b_N + c_N + d_N + e_N = 32 + 756 * Real.sqrt 10 + 6 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_l303_30307


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l303_30306

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  (n ≥ 3) → 
  (interior_angle = 144) → 
  (interior_angle = (n - 2) * 180 / n) →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l303_30306


namespace NUMINAMATH_CALUDE_spade_then_ace_probability_l303_30328

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Theorem: The probability of drawing a spade first and an Ace second from a standard 52-card deck is 1/52 -/
theorem spade_then_ace_probability :
  (NumSpades / StandardDeck) * (NumAces / (StandardDeck - 1)) = 1 / StandardDeck :=
sorry

end NUMINAMATH_CALUDE_spade_then_ace_probability_l303_30328


namespace NUMINAMATH_CALUDE_solution_exists_l303_30308

theorem solution_exists (R₀ : ℝ) : ∃ x₁ x₂ x₃ : ℤ,
  x₁ > ⌊R₀⌋ ∧ x₂ > ⌊R₀⌋ ∧ x₃ > ⌊R₀⌋ ∧ x₁^2 + x₂^2 + x₃^2 = x₁ * x₂ * x₃ := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l303_30308


namespace NUMINAMATH_CALUDE_square_side_length_l303_30388

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 1/4 → side^2 = area → side = 1/2 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l303_30388


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l303_30362

def geometric_sequence (a : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℤ) (r : ℤ) :
  geometric_sequence a r → a 1 = 1 → r = -2 →
  a 1 + |a 2| + |a 3| + a 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l303_30362


namespace NUMINAMATH_CALUDE_at_least_one_seedling_exactly_one_success_l303_30344

-- Define the probabilities
def prob_A_seedling : ℝ := 0.6
def prob_B_seedling : ℝ := 0.5
def prob_A_survive : ℝ := 0.7
def prob_B_survive : ℝ := 0.9

-- Theorem 1: Probability that at least one type of fruit tree becomes a seedling
theorem at_least_one_seedling :
  1 - (1 - prob_A_seedling) * (1 - prob_B_seedling) = 0.8 := by sorry

-- Theorem 2: Probability that exactly one type of fruit tree is successfully cultivated and survives
theorem exactly_one_success :
  let prob_A_success := prob_A_seedling * prob_A_survive
  let prob_B_success := prob_B_seedling * prob_B_survive
  prob_A_success * (1 - prob_B_success) + (1 - prob_A_success) * prob_B_success = 0.492 := by sorry

end NUMINAMATH_CALUDE_at_least_one_seedling_exactly_one_success_l303_30344


namespace NUMINAMATH_CALUDE_intersection_S_T_l303_30387

-- Define the sets S and T
def S : Set ℝ := {x | x < -5 ∨ x > 5}
def T : Set ℝ := {x | -7 < x ∧ x < 3}

-- State the theorem
theorem intersection_S_T : S ∩ T = {x | -7 < x ∧ x < -5} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l303_30387


namespace NUMINAMATH_CALUDE_circle_radius_from_intersecting_chords_l303_30324

theorem circle_radius_from_intersecting_chords (a b d : ℝ) (ha : a > 0) (hb : b > 0) (hd : d > 0) :
  ∃ (r : ℝ),
    (r = (a/d) * Real.sqrt (a^2 + b^2 - 2*b * Real.sqrt (a^2 - d^2))) ∨
    (r = (a/d) * Real.sqrt (a^2 + b^2 + 2*b * Real.sqrt (a^2 - d^2))) ∨
    (a = d ∧ r = Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_intersecting_chords_l303_30324


namespace NUMINAMATH_CALUDE_sequence_decreasing_l303_30382

theorem sequence_decreasing (a : ℕ → ℝ) (h1 : a 1 > 0) (h2 : ∀ n : ℕ, a (n + 1) / a n = 1 / 2) :
  ∀ n m : ℕ, n < m → a m < a n :=
sorry

end NUMINAMATH_CALUDE_sequence_decreasing_l303_30382


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_c_value_l303_30345

theorem quadratic_roots_imply_c_value (c : ℝ) :
  (∀ x : ℝ, x^2 + 5*x + c = 0 ↔ x = (-5 + Real.sqrt c) / 2 ∨ x = (-5 - Real.sqrt c) / 2) →
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_c_value_l303_30345


namespace NUMINAMATH_CALUDE_expected_winnings_is_four_thirds_l303_30331

/-- Represents the faces of the coin -/
inductive Face
  | one
  | two
  | three
  | four

/-- The probability of each face appearing -/
def probability (f : Face) : ℚ :=
  match f with
  | Face.one => 5/12
  | Face.two => 1/3
  | Face.three => 1/6
  | Face.four => 1/12

/-- The winnings associated with each face -/
def winnings (f : Face) : ℤ :=
  match f with
  | Face.one => 2
  | Face.two => 0
  | Face.three => -2
  | Face.four => 10

/-- The expected winnings when tossing the coin -/
def expectedWinnings : ℚ :=
  (probability Face.one * winnings Face.one) +
  (probability Face.two * winnings Face.two) +
  (probability Face.three * winnings Face.three) +
  (probability Face.four * winnings Face.four)

theorem expected_winnings_is_four_thirds :
  expectedWinnings = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_winnings_is_four_thirds_l303_30331


namespace NUMINAMATH_CALUDE_problem_solution_l303_30365

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a - 1}

theorem problem_solution :
  (∀ a : ℝ, a = 3 → M ∪ (Nᶜ a) = Set.univ) ∧
  (∀ a : ℝ, N a ⊆ M ↔ a ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l303_30365


namespace NUMINAMATH_CALUDE_typing_orders_count_l303_30349

/-- The number of letters to be typed -/
def n : ℕ := 9

/-- The index of the letter that has already been typed -/
def typed_letter : ℕ := 8

/-- The set of possible remaining letters after the typed_letter has been removed -/
def remaining_letters : Finset ℕ := Finset.filter (λ x => x ≠ typed_letter ∧ x ≤ n) (Finset.range (n + 1))

/-- The number of possible typing orders for the remaining letters -/
def num_typing_orders : ℕ :=
  (Finset.range 8).sum (λ k => Nat.choose 7 k * (k + 2))

theorem typing_orders_count :
  num_typing_orders = 704 :=
sorry

end NUMINAMATH_CALUDE_typing_orders_count_l303_30349


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l303_30391

theorem scientific_notation_equality : 935000000 = 9.35 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l303_30391


namespace NUMINAMATH_CALUDE_zongzi_sales_l303_30313

/-- The cost and profit calculation for zongzi sales during the Dragon Boat Festival --/
theorem zongzi_sales (x : ℝ) (m : ℝ) : 
  /- Cost price after festival -/
  (∀ y, y > 0 → 240 / y - 4 = 240 / (y + 2) → y = x) →
  /- Total cost constraint -/
  ((12 : ℝ) * m + 10 * (400 - m) ≤ 4600) →
  /- Profit calculation -/
  (∀ w, w = 2 * m + 2400) →
  /- Conclusions -/
  (x = 10 ∧ m = 300 ∧ (2 * 300 + 2400 = 3000)) := by
  sorry


end NUMINAMATH_CALUDE_zongzi_sales_l303_30313


namespace NUMINAMATH_CALUDE_number_difference_l303_30368

theorem number_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) :
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_number_difference_l303_30368


namespace NUMINAMATH_CALUDE_roadster_paving_cement_usage_l303_30359

/-- The amount of cement used for Lexi's street in tons -/
def lexi_cement : ℝ := 10

/-- The amount of cement used for Tess's street in tons -/
def tess_cement : ℝ := 5.1

/-- The total amount of cement used by Roadster's Paving Company in tons -/
def total_cement : ℝ := lexi_cement + tess_cement

theorem roadster_paving_cement_usage :
  total_cement = 15.1 := by sorry

end NUMINAMATH_CALUDE_roadster_paving_cement_usage_l303_30359


namespace NUMINAMATH_CALUDE_equation_solution_l303_30383

theorem equation_solution : ∃ x : ℚ, x - 1/2 = 2/5 - 1/4 ∧ x = 13/20 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l303_30383


namespace NUMINAMATH_CALUDE_min_even_integers_l303_30320

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 30 →
  a + b + c + d = 50 →
  a + b + c + d + e + f = 70 →
  Even e →
  Even f →
  (∃ (count : ℕ), count ≥ 2 ∧ 
    count = (if Even a then 1 else 0) +
            (if Even b then 1 else 0) +
            (if Even c then 1 else 0) +
            (if Even d then 1 else 0) + 2) :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l303_30320


namespace NUMINAMATH_CALUDE_average_study_time_difference_l303_30336

def daily_differences : List Int := [15, -5, 25, -10, 5, 20, -15]

def days_in_week : Nat := 7

theorem average_study_time_difference :
  (daily_differences.sum : ℚ) / days_in_week = 5 := by sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l303_30336


namespace NUMINAMATH_CALUDE_S_5_equals_31_l303_30343

-- Define the sequence and its partial sum
def S (n : ℕ) : ℕ := 2^n - 1

-- State the theorem
theorem S_5_equals_31 : S 5 = 31 := by
  sorry

end NUMINAMATH_CALUDE_S_5_equals_31_l303_30343


namespace NUMINAMATH_CALUDE_min_value_expression_l303_30319

theorem min_value_expression (c : ℝ) (a b : ℝ) (hc : c > 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (h_eq : 4 * a^2 - 2 * a * b + b^2 - c = 0)
  (h_max : ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → 4 * x^2 - 2 * x * y + y^2 - c = 0 → |2 * x + y| ≤ |2 * a + b|) :
  ∃ (k : ℝ), k = 1/a + 2/b + 4/c ∧ k ≥ -1 ∧ (∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z > 0 →
    4 * x^2 - 2 * x * y + y^2 - z = 0 → 1/x + 2/y + 4/z ≥ k) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l303_30319


namespace NUMINAMATH_CALUDE_prop1_prop2_prop3_l303_30358

-- Define the function f(x)
def f (p q x : ℝ) : ℝ := x * abs x + p * x + q

-- Proposition 1: When q = 0, f(x) is an odd function
theorem prop1 (p : ℝ) : 
  ∀ x : ℝ, f p 0 (-x) = -(f p 0 x) := by sorry

-- Proposition 2: The graph of y = f(x) is symmetric with respect to the point (0,q)
theorem prop2 (p q : ℝ) :
  ∀ x : ℝ, f p q x - q = -(f p q (-x) - q) := by sorry

-- Proposition 3: When p = 0 and q > 0, the equation f(x) = 0 has exactly one real root
theorem prop3 (q : ℝ) (hq : q > 0) :
  ∃! x : ℝ, f 0 q x = 0 := by sorry

end NUMINAMATH_CALUDE_prop1_prop2_prop3_l303_30358


namespace NUMINAMATH_CALUDE_expression_evaluation_l303_30357

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 2) :
  3 * x^(y + 1) + 4 * y^(x + 1) = 145 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l303_30357


namespace NUMINAMATH_CALUDE_expression_evaluation_l303_30348

theorem expression_evaluation : -1^5 + (-3)^0 - (Real.sqrt 2)^2 + 4 * |-(1/4)| = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l303_30348


namespace NUMINAMATH_CALUDE_geometric_series_sum_is_15_11_l303_30338

/-- The sum of an infinite geometric series with first term a and common ratio r -/
def geometricSeriesSum (a : ℚ) (r : ℚ) : ℚ := a / (1 - r)

/-- The first term of the given geometric series -/
def a : ℚ := 5 / 3

/-- The common ratio of the given geometric series -/
def r : ℚ := -2 / 9

/-- The theorem stating that the sum of the given infinite geometric series is 15/11 -/
theorem geometric_series_sum_is_15_11 : geometricSeriesSum a r = 15 / 11 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_is_15_11_l303_30338


namespace NUMINAMATH_CALUDE_divisibility_statements_l303_30380

theorem divisibility_statements :
  (∃ n : ℤ, 24 = 4 * n) ∧
  (∃ m : ℤ, 152 = 19 * m) ∧ ¬(∃ k : ℤ, 96 = 19 * k) ∧
  ((∃ p : ℤ, 75 = 15 * p) ∨ (∃ q : ℤ, 90 = 15 * q)) ∧
  ((∃ r : ℤ, 28 = 14 * r) ∧ (∃ s : ℤ, 56 = 14 * s)) ∧
  (∃ t : ℤ, 180 = 6 * t) :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_statements_l303_30380


namespace NUMINAMATH_CALUDE_victors_journey_l303_30363

/-- The distance from Victor's home to the airport --/
def s : ℝ := 240

/-- Victor's initial speed --/
def initial_speed : ℝ := 60

/-- Victor's increased speed --/
def increased_speed : ℝ := 80

/-- Time spent at initial speed --/
def initial_time : ℝ := 0.5

/-- Time difference if Victor continued at initial speed --/
def late_time : ℝ := 0.25

/-- Time difference after increasing speed --/
def early_time : ℝ := 0.25

theorem victors_journey :
  ∃ (t : ℝ),
    s = initial_speed * initial_time + initial_speed * (t + late_time) ∧
    s = initial_speed * initial_time + increased_speed * (t - early_time) :=
by sorry

end NUMINAMATH_CALUDE_victors_journey_l303_30363


namespace NUMINAMATH_CALUDE_unpainted_cubes_not_multiple_of_painted_cubes_l303_30369

theorem unpainted_cubes_not_multiple_of_painted_cubes (n : ℕ) (h : n ≥ 1) :
  ¬(6 * n^2 + 12 * n + 8 ∣ n^3) := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_not_multiple_of_painted_cubes_l303_30369


namespace NUMINAMATH_CALUDE_michael_sarah_games_l303_30390

def total_players : ℕ := 12
def players_per_game : ℕ := 6

theorem michael_sarah_games (michael sarah : Fin total_players) 
  (h_distinct : michael ≠ sarah) :
  (Finset.univ.filter (λ game : Finset (Fin total_players) => 
    game.card = players_per_game ∧ 
    michael ∈ game ∧ 
    sarah ∈ game)).card = Nat.choose (total_players - 2) (players_per_game - 2) := by
  sorry

end NUMINAMATH_CALUDE_michael_sarah_games_l303_30390


namespace NUMINAMATH_CALUDE_bird_nest_twigs_l303_30311

theorem bird_nest_twigs (twigs_in_circle : ℕ) (additional_twigs_per_weave : ℕ) (twigs_still_needed : ℕ) :
  twigs_in_circle = 12 →
  additional_twigs_per_weave = 6 →
  twigs_still_needed = 48 →
  (twigs_in_circle * additional_twigs_per_weave - twigs_still_needed : ℚ) / (twigs_in_circle * additional_twigs_per_weave) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_bird_nest_twigs_l303_30311


namespace NUMINAMATH_CALUDE_d_necessary_not_sufficient_for_a_l303_30356

-- Define propositions
variable (A B C D : Prop)

-- Define the relationships between propositions
variable (h1 : A → B ∧ ¬(B → A))
variable (h2 : B → C ∧ ¬(C → B))
variable (h3 : D ↔ C)

-- Theorem to prove
theorem d_necessary_not_sufficient_for_a :
  (D → A) ∧ ¬(A → D) :=
sorry

end NUMINAMATH_CALUDE_d_necessary_not_sufficient_for_a_l303_30356


namespace NUMINAMATH_CALUDE_total_training_hours_endurance_training_hours_l303_30325

/-- Represents the training schedule for a goalkeeper --/
structure GoalkeeperSchedule where
  diving_catching : ℝ
  strength_conditioning : ℝ
  goalkeeper_specific : ℝ
  footwork : ℝ
  reaction_time : ℝ
  aerial_ball : ℝ
  shot_stopping : ℝ
  defensive_communication : ℝ
  game_simulation : ℝ
  endurance : ℝ

/-- Calculates the total training hours per week --/
def weekly_hours (s : GoalkeeperSchedule) : ℝ :=
  s.diving_catching + s.strength_conditioning + s.goalkeeper_specific +
  s.footwork + s.reaction_time + s.aerial_ball + s.shot_stopping +
  s.defensive_communication + s.game_simulation + s.endurance

/-- Mike's weekly training schedule --/
def mike_schedule : GoalkeeperSchedule :=
  { diving_catching := 2
  , strength_conditioning := 4
  , goalkeeper_specific := 2
  , footwork := 2
  , reaction_time := 1
  , aerial_ball := 3.5
  , shot_stopping := 1.5
  , defensive_communication := 1.5
  , game_simulation := 3
  , endurance := 3
  }

/-- The number of weeks Mike will train --/
def training_weeks : ℕ := 3

/-- Theorem: Mike's total training hours over 3 weeks is 70.5 --/
theorem total_training_hours :
  (weekly_hours mike_schedule) * training_weeks = 70.5 := by sorry

/-- Theorem: Mike's endurance training hours over 3 weeks is 9 --/
theorem endurance_training_hours :
  mike_schedule.endurance * training_weeks = 9 := by sorry

end NUMINAMATH_CALUDE_total_training_hours_endurance_training_hours_l303_30325


namespace NUMINAMATH_CALUDE_max_grain_mass_on_platform_l303_30330

/-- Represents a rectangular platform with grain piled on it. -/
structure GrainPlatform where
  length : ℝ
  width : ℝ
  grainDensity : ℝ
  maxAngle : ℝ

/-- Calculates the maximum mass of grain on the platform. -/
def maxGrainMass (platform : GrainPlatform) : ℝ :=
  sorry

/-- Theorem stating the maximum mass of grain on the given platform. -/
theorem max_grain_mass_on_platform :
  let platform : GrainPlatform := {
    length := 8,
    width := 5,
    grainDensity := 1200,
    maxAngle := π / 4  -- 45 degrees in radians
  }
  maxGrainMass platform = 47500  -- 47.5 tons in kg
  := by sorry

end NUMINAMATH_CALUDE_max_grain_mass_on_platform_l303_30330


namespace NUMINAMATH_CALUDE_student_meeting_probability_l303_30310

def library_open_time : ℝ := 120

theorem student_meeting_probability (n : ℝ) : 
  (0 < n) → 
  (n < library_open_time) → 
  ((library_open_time - n)^2 / library_open_time^2 = 1/2) → 
  (n = 120 - 60 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_student_meeting_probability_l303_30310


namespace NUMINAMATH_CALUDE_min_distance_sum_five_digit_numbers_l303_30341

theorem min_distance_sum_five_digit_numbers (x₁ x₂ x₃ x₄ x₅ : ℕ) :
  -- Define the constraints
  x₅ ≥ 9 →
  x₄ + x₅ ≥ 99 →
  x₃ + x₄ + x₅ ≥ 999 →
  x₂ + x₃ + x₄ + x₅ ≥ 9999 →
  x₁ + x₂ + x₃ + x₄ + x₅ = 99999 →
  -- The theorem to prove
  x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ ≥ 101105 :=
by sorry

#check min_distance_sum_five_digit_numbers

end NUMINAMATH_CALUDE_min_distance_sum_five_digit_numbers_l303_30341


namespace NUMINAMATH_CALUDE_optimal_fraction_sum_l303_30302

theorem optimal_fraction_sum (A B C D : ℕ) : 
  (A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9) →  -- A, B, C, D are digits
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →  -- A, B, C, D are different
  (C + D ≥ 5) →  -- C + D is at least 5
  (∃ k : ℕ, k * (C + D) = A + B) →  -- (A+B)/(C+D) is an integer
  (A + B ≤ 14) :=  -- The maximum possible value of A+B is 14
by sorry

end NUMINAMATH_CALUDE_optimal_fraction_sum_l303_30302


namespace NUMINAMATH_CALUDE_janet_total_distance_l303_30378

/-- Represents Janet's training schedule for a week --/
structure WeekSchedule where
  running_days : Nat
  running_miles : Nat
  cycling_days : Nat
  cycling_miles : Nat
  swimming_days : Nat
  swimming_miles : Nat
  hiking_days : Nat
  hiking_miles : Nat

/-- Calculates the total distance for a given week schedule --/
def weekTotalDistance (schedule : WeekSchedule) : Nat :=
  schedule.running_days * schedule.running_miles +
  schedule.cycling_days * schedule.cycling_miles +
  schedule.swimming_days * schedule.swimming_miles +
  schedule.hiking_days * schedule.hiking_miles

/-- Janet's training schedule for three weeks --/
def janetSchedule : List WeekSchedule := [
  { running_days := 5, running_miles := 8, cycling_days := 3, cycling_miles := 7, swimming_days := 0, swimming_miles := 0, hiking_days := 0, hiking_miles := 0 },
  { running_days := 4, running_miles := 10, cycling_days := 0, cycling_miles := 0, swimming_days := 2, swimming_miles := 2, hiking_days := 0, hiking_miles := 0 },
  { running_days := 5, running_miles := 6, cycling_days := 0, cycling_miles := 0, swimming_days := 0, swimming_miles := 0, hiking_days := 2, hiking_miles := 3 }
]

/-- Theorem: Janet's total training distance is 141 miles --/
theorem janet_total_distance :
  (janetSchedule.map weekTotalDistance).sum = 141 := by
  sorry

end NUMINAMATH_CALUDE_janet_total_distance_l303_30378


namespace NUMINAMATH_CALUDE_hexagon_area_error_l303_30351

/-- If there's an 8% error in excess while measuring the sides of a hexagon, 
    the percentage of error in the estimated area is 16.64%. -/
theorem hexagon_area_error (s : ℝ) (h : s > 0) : 
  let true_area := (3 * Real.sqrt 3 / 2) * s^2
  let measured_side := 1.08 * s
  let estimated_area := (3 * Real.sqrt 3 / 2) * measured_side^2
  (estimated_area - true_area) / true_area * 100 = 16.64 := by
sorry

end NUMINAMATH_CALUDE_hexagon_area_error_l303_30351


namespace NUMINAMATH_CALUDE_exam_pass_count_l303_30366

theorem exam_pass_count (total_candidates : ℕ) (avg_all : ℚ) (avg_pass : ℚ) (avg_fail : ℚ) : 
  total_candidates = 120 →
  avg_all = 35 →
  avg_pass = 39 →
  avg_fail = 15 →
  ∃ (pass_count : ℕ), pass_count = 100 ∧ pass_count ≤ total_candidates :=
by sorry

end NUMINAMATH_CALUDE_exam_pass_count_l303_30366


namespace NUMINAMATH_CALUDE_factor_count_l303_30360

/-- The number of positive factors of 180 that are also multiples of 15 -/
def count_factors : ℕ :=
  (Finset.filter (λ x => x ∣ 180 ∧ 15 ∣ x) (Finset.range 181)).card

theorem factor_count : count_factors = 6 := by
  sorry

end NUMINAMATH_CALUDE_factor_count_l303_30360


namespace NUMINAMATH_CALUDE_riley_made_three_mistakes_l303_30326

/-- Represents the number of mistakes made by Riley in the math contest. -/
def riley_mistakes : ℕ := sorry

/-- Represents the number of mistakes made by Ofelia in the math contest. -/
def ofelia_mistakes : ℕ := sorry

/-- The total number of questions in the math contest. -/
def total_questions : ℕ := 35

/-- Theorem stating that Riley made exactly 3 mistakes in the math contest. -/
theorem riley_made_three_mistakes :
  (riley_mistakes + ofelia_mistakes = 17) ∧
  (ofelia_mistakes = total_questions - ((total_questions - riley_mistakes) / 2 + 5)) →
  riley_mistakes = 3 :=
by sorry

end NUMINAMATH_CALUDE_riley_made_three_mistakes_l303_30326


namespace NUMINAMATH_CALUDE_max_value_of_f_l303_30352

/-- The function f(x) = x^3 - 3ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + 2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem max_value_of_f (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f a x ≥ f a 2) →
  (∃ x : ℝ, f a x = 18 ∧ ∀ y : ℝ, f a y ≤ f a x) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l303_30352


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l303_30393

/-- The quadratic function y = x^2 + ax + a - 2 -/
def f (a x : ℝ) : ℝ := x^2 + a*x + a - 2

theorem quadratic_function_properties (a : ℝ) :
  -- The function always has two distinct real roots
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ∧
  -- The distance between the roots is minimized when a = 2
  (∀ b : ℝ, ∃ x₁ x₂ : ℝ, f b x₁ = 0 ∧ f b x₂ = 0 → 
    |x₁ - x₂| ≥ |(-2 : ℝ) - 2|) ∧
  -- When both roots are in the interval (-2, 2), a is in the interval (-2/3, 2)
  (∀ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ -2 < x₁ ∧ x₁ < 2 ∧ -2 < x₂ ∧ x₂ < 2 → 
    -2/3 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l303_30393


namespace NUMINAMATH_CALUDE_village_news_spread_l303_30333

/-- Represents the village and its news spreading dynamics -/
structure Village where
  inhabitants : Finset Nat
  acquaintances : Nat → Finset Nat
  news_spreads : ∀ (n : Nat), n ∈ inhabitants → ∀ (m : Nat), m ∈ acquaintances n → m ∈ inhabitants

/-- A village satisfies the problem conditions -/
def ValidVillage (v : Village) : Prop :=
  v.inhabitants.card = 1000 ∧
  (∀ (news : Nat → Prop) (start : Nat → Prop),
    (∃ (d : Nat), ∀ (n : Nat), n ∈ v.inhabitants → news n))

/-- Represents the spread of news over time -/
def NewsSpread (v : Village) (informed : Finset Nat) (days : Nat) : Finset Nat :=
  sorry

/-- The main theorem to be proved -/
theorem village_news_spread (v : Village) (h : ValidVillage v) :
  ∃ (informed : Finset Nat),
    informed.card = 90 ∧
    ∀ (n : Nat), n ∈ v.inhabitants →
      n ∈ NewsSpread v informed 10 :=
sorry

end NUMINAMATH_CALUDE_village_news_spread_l303_30333


namespace NUMINAMATH_CALUDE_base_k_representation_of_fraction_l303_30337

/-- The base of the number system -/
def k : ℕ+ := sorry

/-- The fraction we're representing -/
def fraction : ℚ := 11 / 85

/-- The repeating part of the base-k representation -/
def repeating_part : ℕ × ℕ := (3, 5)

/-- The value of the repeating base-k representation -/
def repeating_value (k : ℕ+) (rep : ℕ × ℕ) : ℚ :=
  (rep.1 : ℚ) / (k : ℚ) + (rep.2 : ℚ) / ((k : ℚ) ^ 2) 
  / (1 - 1 / ((k : ℚ) ^ 2))

/-- The main theorem -/
theorem base_k_representation_of_fraction :
  repeating_value k repeating_part = fraction ∧ k = 25 := by sorry

end NUMINAMATH_CALUDE_base_k_representation_of_fraction_l303_30337


namespace NUMINAMATH_CALUDE_total_cost_calculation_l303_30332

/-- The total cost of sandwiches and sodas -/
def total_cost (sandwich_price : ℚ) (soda_price : ℚ) (sandwich_quantity : ℕ) (soda_quantity : ℕ) : ℚ :=
  sandwich_price * sandwich_quantity + soda_price * soda_quantity

/-- Theorem: The total cost of 2 sandwiches at $2.49 each and 4 sodas at $1.87 each is $12.46 -/
theorem total_cost_calculation :
  total_cost (249/100) (187/100) 2 4 = 1246/100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l303_30332


namespace NUMINAMATH_CALUDE_vertical_shift_equivalence_l303_30370

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the vertical shift transformation
def verticalShift (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f x + k

-- Theorem statement
theorem vertical_shift_equivalence :
  ∀ (x : ℝ), verticalShift f 2 x = f x + 2 := by sorry

end NUMINAMATH_CALUDE_vertical_shift_equivalence_l303_30370


namespace NUMINAMATH_CALUDE_problem_statement_l303_30316

theorem problem_statement (x y : ℝ) 
  (h1 : 1/x + 1/y = 4)
  (h2 : x*y + x + y = 5) : 
  x^2*y + x*y^2 = 4 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l303_30316


namespace NUMINAMATH_CALUDE_andy_problem_count_l303_30372

/-- The number of problems Andy solves when he completes problems from 80 to 125 inclusive -/
def problems_solved : ℕ := 125 - 80 + 1

theorem andy_problem_count : problems_solved = 46 := by
  sorry

end NUMINAMATH_CALUDE_andy_problem_count_l303_30372


namespace NUMINAMATH_CALUDE_fraction_evaluation_l303_30364

theorem fraction_evaluation : (5 : ℝ) / (1 - 1/2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l303_30364


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l303_30379

theorem polar_to_cartesian :
  let ρ : ℝ := 4
  let θ : ℝ := π / 3
  let x : ℝ := ρ * Real.cos θ
  let y : ℝ := ρ * Real.sin θ
  (x = 2 ∧ y = 2 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l303_30379


namespace NUMINAMATH_CALUDE_garden_problem_l303_30398

theorem garden_problem (a : ℝ) : 
  a > 0 → 
  (a + 3)^2 = 2 * a^2 + 9 → 
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_garden_problem_l303_30398


namespace NUMINAMATH_CALUDE_system_solution_inequality_solution_l303_30314

-- Define the system of equations
def system_eq (x y : ℝ) : Prop :=
  5 * x + 2 * y = 25 ∧ 3 * x + 4 * y = 15

-- Define the linear inequality
def linear_ineq (x : ℝ) : Prop :=
  2 * x - 6 < 3 * x

-- Theorem for the system of equations
theorem system_solution :
  ∃! (x y : ℝ), system_eq x y ∧ x = 5 ∧ y = 0 :=
sorry

-- Theorem for the linear inequality
theorem inequality_solution :
  ∀ x : ℝ, linear_ineq x ↔ x > -6 :=
sorry

end NUMINAMATH_CALUDE_system_solution_inequality_solution_l303_30314


namespace NUMINAMATH_CALUDE_max_value_on_interval_l303_30305

/-- The function we're analyzing -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The closed interval [0, 3] -/
def I : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

theorem max_value_on_interval :
  ∃ (c : ℝ), c ∈ I ∧ f c = 6 ∧ ∀ x ∈ I, f x ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l303_30305


namespace NUMINAMATH_CALUDE_percentage_of_students_owning_only_cats_l303_30399

theorem percentage_of_students_owning_only_cats
  (total_students : ℕ)
  (cat_owners : ℕ)
  (dog_owners : ℕ)
  (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 75)
  (h3 : dog_owners = 150)
  (h4 : both_owners = 25) :
  (cat_owners - both_owners) * 100 / total_students = 10 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_students_owning_only_cats_l303_30399


namespace NUMINAMATH_CALUDE_bicycle_car_arrival_l303_30385

theorem bicycle_car_arrival (x : ℝ) (h : x > 0) : 
  (10 / x - 10 / (2 * x) = 1 / 3) ↔ 
  (10 / x = 10 / (2 * x) + 1 / 3) :=
sorry

end NUMINAMATH_CALUDE_bicycle_car_arrival_l303_30385


namespace NUMINAMATH_CALUDE_smallest_common_factor_l303_30312

theorem smallest_common_factor (n : ℕ) : 
  (∃ (k : ℕ), k > 1 ∧ k ∣ (11*n - 4) ∧ k ∣ (8*n - 5)) ∧ 
  (∀ (m : ℕ), m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (11*m - 4) ∧ k ∣ (8*m - 5))) → 
  n = 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l303_30312


namespace NUMINAMATH_CALUDE_area_between_circles_and_xaxis_l303_30397

/-- The area of the region bound by two circles and the x-axis -/
theorem area_between_circles_and_xaxis :
  let c1_center : ℝ × ℝ := (3, 5)
  let c2_center : ℝ × ℝ := (9, 5)
  let radius : ℝ := 3
  let rectangle_area : ℝ := (c2_center.1 - c1_center.1) * radius
  let sector_area : ℝ := (1/4) * π * radius^2
  rectangle_area - 2 * sector_area = 18 - (9/2) * π := by sorry

end NUMINAMATH_CALUDE_area_between_circles_and_xaxis_l303_30397


namespace NUMINAMATH_CALUDE_line_intersection_l303_30367

theorem line_intersection (a b c : ℝ) : 
  (3 = a * 1 + b) ∧ 
  (3 = b * 1 + c) ∧ 
  (3 = c * 1 + a) → 
  a = (3/2 : ℝ) ∧ b = (3/2 : ℝ) ∧ c = (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_l303_30367


namespace NUMINAMATH_CALUDE_product_sum_inequality_l303_30394

theorem product_sum_inequality (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = p ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a * b = p → x + y ≤ a + b :=
sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l303_30394


namespace NUMINAMATH_CALUDE_compute_expression_l303_30304

theorem compute_expression : 3 * 3^4 - 9^19 / 9^17 = 162 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l303_30304


namespace NUMINAMATH_CALUDE_reciprocal_sum_equals_five_l303_30321

theorem reciprocal_sum_equals_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x + y = 5 * x * y) (h2 : x = 2 * y) : 
  1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equals_five_l303_30321


namespace NUMINAMATH_CALUDE_combination_properties_l303_30396

theorem combination_properties (n m : ℕ+) (h : n > m) :
  (Nat.choose n m = Nat.choose n (n - m)) ∧
  (Nat.choose n m + Nat.choose n (m - 1) = Nat.choose (n + 1) m) := by
  sorry

end NUMINAMATH_CALUDE_combination_properties_l303_30396


namespace NUMINAMATH_CALUDE_eleven_divides_difference_l303_30301

/-- Represents a three-digit number ABC where A, B, and C are distinct digits and A ≠ 0 -/
structure ThreeDigitNumber where
  A : Nat
  B : Nat
  C :Nat
  h1 : A ≠ 0
  h2 : A < 10
  h3 : B < 10
  h4 : C < 10
  h5 : A ≠ B
  h6 : B ≠ C
  h7 : A ≠ C

/-- Converts a ThreeDigitNumber to its numerical value -/
def toNumber (n : ThreeDigitNumber) : Nat :=
  100 * n.A + 10 * n.B + n.C

/-- Reverses a ThreeDigitNumber -/
def reverse (n : ThreeDigitNumber) : Nat :=
  100 * n.C + 10 * n.B + n.A

theorem eleven_divides_difference (n : ThreeDigitNumber) :
  11 ∣ (toNumber n - reverse n) := by
  sorry

#check eleven_divides_difference

end NUMINAMATH_CALUDE_eleven_divides_difference_l303_30301


namespace NUMINAMATH_CALUDE_house_rent_fraction_l303_30322

def salary : ℚ := 140000

def food_fraction : ℚ := 1/5
def clothes_fraction : ℚ := 3/5
def remaining_amount : ℚ := 14000

theorem house_rent_fraction :
  ∃ (house_rent_fraction : ℚ),
    house_rent_fraction * salary + food_fraction * salary + clothes_fraction * salary + remaining_amount = salary ∧
    house_rent_fraction = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_house_rent_fraction_l303_30322


namespace NUMINAMATH_CALUDE_martians_cannot_hold_hands_l303_30350

/-- Represents the number of hands a Martian has -/
def martian_hands : ℕ := 3

/-- Represents the number of Martians -/
def num_martians : ℕ := 7

/-- Calculates the total number of hands for all Martians -/
def total_hands : ℕ := martian_hands * num_martians

/-- Theorem stating that seven Martians cannot hold hands with each other -/
theorem martians_cannot_hold_hands : ¬ (total_hands % 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_martians_cannot_hold_hands_l303_30350


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l303_30355

/-- The number of positive integers satisfying the inequality -/
def count_satisfying_integers : ℕ := 8

/-- The inequality function -/
def inequality (n : ℤ) : Prop :=
  (n + 7) * (n - 4) * (n - 10) < 0

theorem count_integers_satisfying_inequality :
  (∃ (S : Finset ℤ), S.card = count_satisfying_integers ∧
    (∀ n ∈ S, n > 0 ∧ inequality n) ∧
    (∀ n : ℤ, n > 0 → inequality n → n ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l303_30355


namespace NUMINAMATH_CALUDE_kids_meals_sold_l303_30342

theorem kids_meals_sold (kids_meals : ℕ) (adult_meals : ℕ) : 
  (kids_meals : ℚ) / (adult_meals : ℚ) = 2 / 1 →
  kids_meals + adult_meals = 12 →
  kids_meals = 8 := by
sorry

end NUMINAMATH_CALUDE_kids_meals_sold_l303_30342


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l303_30353

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 5) % 12 = 0 ∧
  (n - 5) % 16 = 0 ∧
  (n - 5) % 18 = 0 ∧
  (n - 5) % 21 = 0 ∧
  (n - 5) % 28 = 0

theorem smallest_number_divisible_by_all :
  ∀ m : ℕ, m < 1013 → ¬(is_divisible_by_all m) ∧ is_divisible_by_all 1013 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l303_30353


namespace NUMINAMATH_CALUDE_total_cookies_sum_l303_30392

/-- The number of cookies Kristy baked -/
def total_cookies : ℕ := sorry

/-- The number of cookies Kristy ate -/
def kristy_ate : ℕ := 2

/-- The number of cookies Kristy gave to her brother -/
def brother_got : ℕ := 1

/-- The number of cookies taken by the first friend -/
def first_friend_took : ℕ := 3

/-- The number of cookies taken by the second friend -/
def second_friend_took : ℕ := 5

/-- The number of cookies taken by the third friend -/
def third_friend_took : ℕ := 5

/-- The number of cookies left -/
def cookies_left : ℕ := 6

/-- Theorem stating that the total number of cookies is the sum of all distributed and remaining cookies -/
theorem total_cookies_sum : 
  total_cookies = kristy_ate + brother_got + first_friend_took + 
                  second_friend_took + third_friend_took + cookies_left :=
by sorry

end NUMINAMATH_CALUDE_total_cookies_sum_l303_30392


namespace NUMINAMATH_CALUDE_odd_function_through_points_l303_30376

/-- An odd function passing through two specific points -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem odd_function_through_points :
  (∀ x, f a b c (-x) = -(f a b c x)) →
  f a b c (-Real.sqrt 2) = Real.sqrt 2 →
  f a b c (2 * Real.sqrt 2) = 10 * Real.sqrt 2 →
  ∃ g : ℝ → ℝ, (∀ x, g x = x^3 - 3*x) ∧
              (∀ x, f a b c x = g x) ∧
              (∀ m, (∃! x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ g x₁ + m = 0 ∧ g x₂ + m = 0 ∧ g x₃ + m = 0) ↔
                    -2 < m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_odd_function_through_points_l303_30376


namespace NUMINAMATH_CALUDE_inequality_proof_l303_30323

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l303_30323


namespace NUMINAMATH_CALUDE_tan_alpha_equals_two_implies_expression_equals_three_l303_30329

theorem tan_alpha_equals_two_implies_expression_equals_three (α : Real) 
  (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (2 * Real.sin α - 3 * Real.cos α) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_two_implies_expression_equals_three_l303_30329


namespace NUMINAMATH_CALUDE_vector_operation_l303_30334

theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (3, 1)) :
  2 • a - b = (-1, 3) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l303_30334


namespace NUMINAMATH_CALUDE_like_terms_xy_value_l303_30317

theorem like_terms_xy_value (a b : ℝ) (x y : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ 2 * a^x * b^3 = k * (-a^2 * b^(1-y))) →
  x * y = -4 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_xy_value_l303_30317


namespace NUMINAMATH_CALUDE_combined_salaries_of_four_l303_30361

/-- Given 5 individuals with an average monthly salary and one known salary, 
    prove the sum of the other four salaries. -/
theorem combined_salaries_of_four (average_salary : ℕ) (known_salary : ℕ) 
  (h1 : average_salary = 9000)
  (h2 : known_salary = 5000) :
  4 * average_salary - known_salary = 40000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_of_four_l303_30361


namespace NUMINAMATH_CALUDE_mk97_equality_check_l303_30315

/-- The MK-97 microcalculator operations -/
class Calculator where
  /-- Check if two numbers are equal -/
  equal : ℝ → ℝ → Prop
  /-- Add two numbers -/
  add : ℝ → ℝ → ℝ
  /-- Find roots of a quadratic equation -/
  quadratic_roots : ℝ → ℝ → Option (ℝ × ℝ)

/-- The theorem to be proved -/
theorem mk97_equality_check (x : ℝ) :
  x = 1 ↔ x ≠ 0 ∧ (4 * (x^2 - x) = 0) :=
sorry

end NUMINAMATH_CALUDE_mk97_equality_check_l303_30315


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l303_30346

theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b*x + 1764 = (x + p) * (x + q)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬(∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b'*x + 1764 = (x + p) * (x + q))) ∧
  b = 84 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l303_30346
