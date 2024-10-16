import Mathlib

namespace NUMINAMATH_CALUDE_log2_derivative_l677_67718

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log2_derivative_l677_67718


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l677_67780

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence definition
  q > 0 →  -- positive common ratio
  a 3 * a 9 = 2 * (a 5)^2 →  -- given condition
  a 2 = 2 →  -- given condition
  a 1 = Real.sqrt 2 := by  -- conclusion to prove
sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l677_67780


namespace NUMINAMATH_CALUDE_merry_go_round_area_l677_67745

theorem merry_go_round_area (diameter : Real) (h : diameter = 2) :
  let radius : Real := diameter / 2
  let area : Real := π * radius ^ 2
  area = π := by
  sorry

end NUMINAMATH_CALUDE_merry_go_round_area_l677_67745


namespace NUMINAMATH_CALUDE_course_selection_schemes_l677_67775

def number_of_courses : ℕ := 7
def courses_to_choose : ℕ := 4

def total_combinations : ℕ := Nat.choose number_of_courses courses_to_choose

def forbidden_combinations : ℕ := Nat.choose (number_of_courses - 2) (courses_to_choose - 2)

theorem course_selection_schemes :
  total_combinations - forbidden_combinations = 25 := by sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l677_67775


namespace NUMINAMATH_CALUDE_sum_area_ABC_DEF_l677_67702

-- Define the points and lengths
variable (A B C D E F G : ℝ × ℝ)
variable (AB BG GE DE : ℝ)

-- Define the areas of triangles
def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define the conditions
axiom AB_length : AB = 2
axiom BG_length : BG = 3
axiom GE_length : GE = 4
axiom DE_length : DE = 5

axiom sum_area_BCG_EFG : area_triangle B C G + area_triangle E F G = 24
axiom sum_area_AGF_CDG : area_triangle A G F + area_triangle C D G = 51

-- State the theorem
theorem sum_area_ABC_DEF :
  area_triangle A B C + area_triangle D E F = 23 :=
sorry

end NUMINAMATH_CALUDE_sum_area_ABC_DEF_l677_67702


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_coordinates_l677_67738

/-- A point in a 2D coordinate system. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the y-axis. -/
def symmetricPointYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem symmetric_point_y_axis_coordinates :
  let B : Point := { x := -3, y := 4 }
  let A : Point := symmetricPointYAxis B
  A.x = 3 ∧ A.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_coordinates_l677_67738


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l677_67723

theorem diophantine_equation_solutions : 
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 4 * p.2 = 806 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 807) (Finset.range 807))).card = 67 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l677_67723


namespace NUMINAMATH_CALUDE_words_per_page_l677_67716

theorem words_per_page (total_pages : Nat) (max_words_per_page : Nat) (total_words_mod : Nat) :
  total_pages = 154 →
  max_words_per_page = 120 →
  total_words_mod = 221 →
  ∃ words_per_page : Nat,
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % 217 = total_words_mod ∧
    words_per_page = 106 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_l677_67716


namespace NUMINAMATH_CALUDE_book_loss_percentage_l677_67714

/-- Proves the loss percentage on a book given specific conditions --/
theorem book_loss_percentage
  (total_cost : ℝ)
  (cost_book1 : ℝ)
  (gain_percentage : ℝ)
  (h1 : total_cost = 540)
  (h2 : cost_book1 = 315)
  (h3 : gain_percentage = 19)
  (h4 : ∃ (selling_price : ℝ),
    selling_price = cost_book1 * (1 - loss_percentage / 100) ∧
    selling_price = (total_cost - cost_book1) * (1 + gain_percentage / 100)) :
  ∃ (loss_percentage : ℝ), loss_percentage = 15 := by
sorry


end NUMINAMATH_CALUDE_book_loss_percentage_l677_67714


namespace NUMINAMATH_CALUDE_range_of_g_l677_67744

theorem range_of_g (x : ℝ) : 3/4 ≤ Real.cos x ^ 4 + Real.sin x ^ 2 ∧ Real.cos x ^ 4 + Real.sin x ^ 2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l677_67744


namespace NUMINAMATH_CALUDE_parallel_line_to_hyperbola_asymptote_l677_67765

/-- Given a hyperbola (x²/16) - (y²/9) = 1 and a line y = kx - 1 parallel to one of its asymptotes,
    where k > 0, prove that k = 3/4 -/
theorem parallel_line_to_hyperbola_asymptote
  (k : ℝ)
  (h1 : k > 0)
  (h2 : ∃ (x y : ℝ), y = k * x - 1 ∧ (x^2 / 16) - (y^2 / 9) = 1)
  (h3 : ∃ (m : ℝ), (∀ (x y : ℝ), y = m * x → (x^2 / 16) - (y^2 / 9) = 1) ∧
                   (∃ (b : ℝ), ∀ (x : ℝ), k * x - 1 = m * x + b)) :
  k = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_to_hyperbola_asymptote_l677_67765


namespace NUMINAMATH_CALUDE_sum_of_roots_l677_67752

theorem sum_of_roots (p : ℝ) : 
  let q : ℝ := p^2 - 1
  let f : ℝ → ℝ := λ x ↦ x^2 - p*x + q
  ∃ r s : ℝ, f r = 0 ∧ f s = 0 ∧ r + s = p :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l677_67752


namespace NUMINAMATH_CALUDE_slag_transport_theorem_l677_67776

/-- Represents the daily transport capacity of a team in tons -/
structure TransportCapacity where
  daily : ℝ

/-- Represents a construction team -/
structure Team where
  capacity : TransportCapacity

/-- Represents the project parameters -/
structure Project where
  totalSlag : ℝ
  teamA : Team
  teamB : Team
  transportCost : ℝ

/-- The main theorem to prove -/
theorem slag_transport_theorem (p : Project) 
  (h1 : p.teamA.capacity.daily = p.teamB.capacity.daily * (5/3))
  (h2 : 4000 / p.teamA.capacity.daily + 2 = 3000 / p.teamB.capacity.daily)
  (h3 : p.totalSlag = 7000)
  (h4 : ∃ m : ℝ, 
    (p.teamA.capacity.daily + m) * 7 + 
    (p.teamB.capacity.daily + m/300) * 9 = p.totalSlag) :
  p.teamA.capacity.daily = 500 ∧ 
  (p.teamB.capacity.daily + (50/300)) * 9 * p.transportCost = 157500 := by
  sorry

#check slag_transport_theorem

end NUMINAMATH_CALUDE_slag_transport_theorem_l677_67776


namespace NUMINAMATH_CALUDE_triangle_problem_l677_67701

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

theorem triangle_problem (t : Triangle) 
  (h1 : 2 * Real.sin t.A * Real.cos t.B + Real.sin t.B = 2 * Real.sin t.C)
  (h2 : t.a = 4 * Real.sqrt 3)
  (h3 : t.b + t.c = 8) : 
  t.A = Real.pi / 3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A : Real) = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l677_67701


namespace NUMINAMATH_CALUDE_quarter_count_in_collection_l677_67795

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin type in cents --/
def coinValue : CoinType → Nat
  | CoinType.Penny => 1
  | CoinType.Nickel => 5
  | CoinType.Dime => 10
  | CoinType.Quarter => 25
  | CoinType.HalfDollar => 50

/-- A collection of coins --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  quarters : Nat
  halfDollars : Nat

/-- The total value of a coin collection in cents --/
def totalValue (c : CoinCollection) : Nat :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime +
  c.quarters * coinValue CoinType.Quarter +
  c.halfDollars * coinValue CoinType.HalfDollar

/-- The total number of coins in a collection --/
def totalCoins (c : CoinCollection) : Nat :=
  c.pennies + c.nickels + c.dimes + c.quarters + c.halfDollars

theorem quarter_count_in_collection :
  ∀ c : CoinCollection,
    totalCoins c = 11 ∧
    totalValue c = 163 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    c.quarters ≥ 1 ∧
    c.halfDollars ≥ 1
    →
    c.quarters = 2 :=
by sorry

end NUMINAMATH_CALUDE_quarter_count_in_collection_l677_67795


namespace NUMINAMATH_CALUDE_area_swept_by_line_segment_l677_67794

/-- The area swept by a line segment connecting a fixed point to a moving point on a unit circle --/
theorem area_swept_by_line_segment (A : ℝ × ℝ) (t₁ t₂ : ℝ) : 
  A = (2, 0) →
  t₁ = 15 * π / 180 →
  t₂ = 45 * π / 180 →
  let P (t : ℝ) := (Real.sin (2 * t - π / 3), Real.cos (2 * t - π / 3))
  let area := (t₂ - t₁) / 2
  area = π / 6 := by sorry

end NUMINAMATH_CALUDE_area_swept_by_line_segment_l677_67794


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_7999999999_l677_67791

theorem largest_prime_factor_of_7999999999 :
  let n : ℕ := 7999999999
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p * q) →
  (∃ p : ℕ, Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Prime q → q ∣ n → q ≤ p) ∧
  (∃ p : ℕ, Prime p ∧ p ∣ n ∧ p = 4002001) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_7999999999_l677_67791


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_3_to_m_l677_67705

def m : ℕ := 2011^2 + 3^2011

theorem units_digit_of_m_squared_plus_3_to_m (m : ℕ := 2011^2 + 3^2011) : 
  (m^2 + 3^m) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_3_to_m_l677_67705


namespace NUMINAMATH_CALUDE_power_two_congruence_l677_67725

theorem power_two_congruence (n : ℕ) (a : ℤ) (hn : n ≥ 1) (ha : Odd a) :
  a ^ (2 ^ n) ≡ 1 [ZMOD (2 ^ (n + 2))] := by
  sorry

end NUMINAMATH_CALUDE_power_two_congruence_l677_67725


namespace NUMINAMATH_CALUDE_function_value_theorem_l677_67768

/-- A function f(x) = a(x+2)^2 + 3 passing through points (-2, 3) and (0, 7) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 2)^2 + 3

/-- The theorem stating that given the conditions, a+3a+2 equals 6 -/
theorem function_value_theorem (a : ℝ) :
  f a (-2) = 3 ∧ f a 0 = 7 → a + 3*a + 2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_function_value_theorem_l677_67768


namespace NUMINAMATH_CALUDE_investment_growth_l677_67797

/-- Computes the final amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- The problem statement -/
theorem investment_growth : 
  let principal : ℝ := 3000
  let rate : ℝ := 0.07
  let years : ℕ := 25
  ⌊compound_interest principal rate years⌋ = 16281 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l677_67797


namespace NUMINAMATH_CALUDE_problem_2004_l677_67707

theorem problem_2004 (a : ℝ) : 
  (|2004 - a| + Real.sqrt (a - 2005) = a) → (a - 2004^2 = 2005) := by
  sorry

end NUMINAMATH_CALUDE_problem_2004_l677_67707


namespace NUMINAMATH_CALUDE_closest_point_on_line_l677_67722

/-- The point on the line y = 3x - 1 that is closest to (1,4) is (-3/5, -4/5) -/
theorem closest_point_on_line (x y : ℝ) : 
  y = 3 * x - 1 → 
  (x - (-3/5))^2 + (y - (-4/5))^2 ≤ (x - 1)^2 + (y - 4)^2 :=
by sorry

end NUMINAMATH_CALUDE_closest_point_on_line_l677_67722


namespace NUMINAMATH_CALUDE_max_square_in_unit_triangle_l677_67730

/-- A triangle with base and height both equal to √2 maximizes the area of the inscribed square among all unit-area triangles. -/
theorem max_square_in_unit_triangle :
  ∀ (base height : ℝ) (square_side : ℝ),
    base > 0 → height > 0 → square_side > 0 →
    (1/2) * base * height = 1 →
    square_side^2 ≤ 1/2 →
    square_side^2 ≤ (base * height) / (base + height)^2 →
    square_side^2 ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_max_square_in_unit_triangle_l677_67730


namespace NUMINAMATH_CALUDE_outer_circle_radius_l677_67793

theorem outer_circle_radius (r : ℝ) : 
  r > 0 ∧ 
  (π * (1.2 * r)^2 - π * 3^2) = (π * r^2 - π * 6^2) * 2.109375 → 
  r = 10 := by
  sorry

end NUMINAMATH_CALUDE_outer_circle_radius_l677_67793


namespace NUMINAMATH_CALUDE_square_sum_from_linear_system_l677_67731

theorem square_sum_from_linear_system (x y : ℝ) :
  x - y = 18 → x + y = 22 → x^2 + y^2 = 404 := by sorry

end NUMINAMATH_CALUDE_square_sum_from_linear_system_l677_67731


namespace NUMINAMATH_CALUDE_shaded_area_proof_l677_67729

theorem shaded_area_proof (carpet_side : ℝ) (S T : ℝ) : 
  carpet_side = 9 →
  carpet_side / S = 3 →
  S / T = 3 →
  S * S + 8 * T * T = 17 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l677_67729


namespace NUMINAMATH_CALUDE_largest_digit_for_two_digit_quotient_l677_67782

theorem largest_digit_for_two_digit_quotient :
  ∀ n : ℕ, n ≤ 4 ∧ (n * 100 + 5) / 5 < 100 ∧
  ∀ m : ℕ, m > n → (m * 100 + 5) / 5 ≥ 100 →
  4 = n :=
sorry

end NUMINAMATH_CALUDE_largest_digit_for_two_digit_quotient_l677_67782


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l677_67769

/-- A quadratic equation with parameter m -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*(m+1)*x + m^2 + 5

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := 8*m - 16

/-- The condition for real roots -/
def has_real_roots (m : ℝ) : Prop := discriminant m ≥ 0

/-- The relation between roots and m -/
def roots_relation (m : ℝ) (x₁ x₂ : ℝ) : Prop :=
  quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ (x₁ - 1)*(x₂ - 1) = 28

theorem quadratic_roots_theorem (m : ℝ) :
  has_real_roots m →
  (∃ x₁ x₂, roots_relation m x₁ x₂) →
  m ≥ 2 ∧ m = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l677_67769


namespace NUMINAMATH_CALUDE_simultaneous_work_time_l677_67790

/-- The time taken for two workers to fill a truck when working simultaneously -/
theorem simultaneous_work_time (rate1 rate2 : ℚ) (h1 : rate1 = 1 / 6) (h2 : rate2 = 1 / 8) :
  1 / (rate1 + rate2) = 24 / 7 := by sorry

end NUMINAMATH_CALUDE_simultaneous_work_time_l677_67790


namespace NUMINAMATH_CALUDE_min_m_value_l677_67720

theorem min_m_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a > b) (hbc : b > c) (hcd : c > d) :
  ∃ (m : ℝ), m = 9 ∧ 
  (∀ (k : ℝ), k < 9 → 
    ∃ (x y z w : ℝ), x > y ∧ y > z ∧ z > w ∧ w > 0 ∧
      Real.log (2004 : ℝ) / Real.log (y / x) + 
      Real.log (2004 : ℝ) / Real.log (z / y) + 
      Real.log (2004 : ℝ) / Real.log (w / z) < 
      k * (Real.log (2004 : ℝ) / Real.log (w / x))) ∧
  (∀ (a' b' c' d' : ℝ), a' > b' ∧ b' > c' ∧ c' > d' ∧ d' > 0 →
    Real.log (2004 : ℝ) / Real.log (b' / a') + 
    Real.log (2004 : ℝ) / Real.log (c' / b') + 
    Real.log (2004 : ℝ) / Real.log (d' / c') ≥ 
    9 * (Real.log (2004 : ℝ) / Real.log (d' / a'))) := by
  sorry

end NUMINAMATH_CALUDE_min_m_value_l677_67720


namespace NUMINAMATH_CALUDE_solve_linear_equation_l677_67704

theorem solve_linear_equation (x : ℝ) : x + 1 = 4 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l677_67704


namespace NUMINAMATH_CALUDE_vishal_investment_percentage_l677_67773

theorem vishal_investment_percentage (raghu_investment trishul_investment vishal_investment : ℝ) : 
  raghu_investment = 2400 →
  trishul_investment = 0.9 * raghu_investment →
  vishal_investment + trishul_investment + raghu_investment = 6936 →
  (vishal_investment - trishul_investment) / trishul_investment = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_vishal_investment_percentage_l677_67773


namespace NUMINAMATH_CALUDE_min_cut_off_length_is_82_l677_67783

/-- Represents the rope cutting problem with given constraints -/
def RopeCuttingProblem (total_length : ℕ) (piece_lengths : List ℕ) (max_pieces : ℕ) : Prop :=
  total_length = 89 ∧
  piece_lengths = [7, 3, 1] ∧
  max_pieces = 25

/-- The minimum length of rope that must be cut off -/
def MinCutOffLength (total_length : ℕ) (piece_lengths : List ℕ) (max_pieces : ℕ) : ℕ := 82

/-- Theorem stating the minimum cut-off length for the rope cutting problem -/
theorem min_cut_off_length_is_82
  (total_length : ℕ) (piece_lengths : List ℕ) (max_pieces : ℕ)
  (h : RopeCuttingProblem total_length piece_lengths max_pieces) :
  MinCutOffLength total_length piece_lengths max_pieces = 82 := by
  sorry

end NUMINAMATH_CALUDE_min_cut_off_length_is_82_l677_67783


namespace NUMINAMATH_CALUDE_rock_collection_inconsistency_l677_67713

theorem rock_collection_inconsistency (J : ℤ) : ¬ (∃ (jose albert : ℤ),
  jose = J - 14 ∧
  albert = jose + 20 ∧
  albert = J + 6) := by
  sorry

end NUMINAMATH_CALUDE_rock_collection_inconsistency_l677_67713


namespace NUMINAMATH_CALUDE_number_satisfying_equation_l677_67709

theorem number_satisfying_equation : ∃! x : ℝ, 3 * (x + 2) = 24 + x := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_equation_l677_67709


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l677_67763

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a ≥ 5) →
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l677_67763


namespace NUMINAMATH_CALUDE_transistors_2010_l677_67711

/-- Moore's law: number of transistors doubles every two years -/
def moores_law (years : ℕ) : ℕ → ℕ := fun n => n * 2^(years / 2)

/-- Number of transistors in 1995 -/
def transistors_1995 : ℕ := 2000000

/-- Years between 1995 and 2010 -/
def years_passed : ℕ := 15

theorem transistors_2010 :
  moores_law years_passed transistors_1995 = 256000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_2010_l677_67711


namespace NUMINAMATH_CALUDE_james_coins_value_l677_67785

theorem james_coins_value (p n : ℕ) : 
  p + n = 15 →
  p - 1 = n →
  p * 1 + n * 5 = 43 :=
by sorry

end NUMINAMATH_CALUDE_james_coins_value_l677_67785


namespace NUMINAMATH_CALUDE_club_co_presidents_selection_l677_67733

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of members in the club --/
def club_members : ℕ := 18

/-- The number of co-presidents to be chosen --/
def co_presidents : ℕ := 3

theorem club_co_presidents_selection :
  choose club_members co_presidents = 816 := by
  sorry

end NUMINAMATH_CALUDE_club_co_presidents_selection_l677_67733


namespace NUMINAMATH_CALUDE_salmon_trip_count_l677_67746

theorem salmon_trip_count (male_salmon female_salmon : ℕ) 
  (h1 : male_salmon = 712261)
  (h2 : female_salmon = 259378) : 
  male_salmon + female_salmon = 971639 := by
  sorry

end NUMINAMATH_CALUDE_salmon_trip_count_l677_67746


namespace NUMINAMATH_CALUDE_line_translation_proof_l677_67781

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given distance -/
def translateLine (l : Line) (distance : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + distance }

theorem line_translation_proof :
  let originalLine : Line := { slope := 2, intercept := -3 }
  let translatedLine := translateLine originalLine 6
  translatedLine = { slope := 2, intercept := 3 } := by
sorry

end NUMINAMATH_CALUDE_line_translation_proof_l677_67781


namespace NUMINAMATH_CALUDE_chord_ratio_l677_67741

-- Define the circle and points
variable (circle : Type) (A B C D E P : circle)

-- Define the distance function
variable (dist : circle → circle → ℝ)

-- State the theorem
theorem chord_ratio (h1 : dist A P = 5)
                    (h2 : dist C P = 9)
                    (h3 : dist D E = 4) :
  dist B P / dist E P = 81 / 805 := by sorry

end NUMINAMATH_CALUDE_chord_ratio_l677_67741


namespace NUMINAMATH_CALUDE_inequality_solution_l677_67734

theorem inequality_solution (x : ℝ) : 
  (9*x^2 + 27*x - 40) / ((3*x - 4)*(x + 5)) < 5 ↔ 
  (x > -6 ∧ x < -5) ∨ (x > 4/3 ∧ x < 5/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l677_67734


namespace NUMINAMATH_CALUDE_problem_solutions_l677_67736

theorem problem_solutions : 
  (1) * (1/3 - 1/4 - 1/2) / (-1/24) = 10 ∧ 
  -(3^2) - (-2/3) * 6 + (-2)^3 = -13 := by sorry

end NUMINAMATH_CALUDE_problem_solutions_l677_67736


namespace NUMINAMATH_CALUDE_train_crossing_cars_l677_67779

/-- Represents the properties of a train passing through a crossing -/
structure TrainCrossing where
  cars_in_sample : ℕ
  sample_time : ℕ
  total_time : ℕ

/-- Calculates the number of cars in the train, rounded to the nearest multiple of 10 -/
def cars_in_train (tc : TrainCrossing) : ℕ :=
  let rate := tc.cars_in_sample / tc.sample_time
  let total_cars := rate * tc.total_time
  ((total_cars + 5) / 10) * 10

/-- Theorem stating that for the given train crossing scenario, the number of cars is 120 -/
theorem train_crossing_cars :
  let tc : TrainCrossing := { cars_in_sample := 9, sample_time := 15, total_time := 210 }
  cars_in_train tc = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_cars_l677_67779


namespace NUMINAMATH_CALUDE_correct_calculation_l677_67748

theorem correct_calculation (n m : ℝ) : n * m^2 - 2 * m^2 * n = -m^2 * n := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l677_67748


namespace NUMINAMATH_CALUDE_smallest_number_l677_67753

theorem smallest_number (a b c : ℝ) (ha : a = -0.5) (hb : b = 3) (hc : c = -2) :
  min a (min b c) = c := by sorry

end NUMINAMATH_CALUDE_smallest_number_l677_67753


namespace NUMINAMATH_CALUDE_unique_solution_l677_67786

theorem unique_solution : ∃! x : ℝ, ((52 + x) * 3 - 60) / 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l677_67786


namespace NUMINAMATH_CALUDE_lottery_win_probability_l677_67747

def megaBallCount : ℕ := 27
def winnerBallCount : ℕ := 44
def winnerBallPick : ℕ := 5

theorem lottery_win_probability :
  (1 : ℚ) / megaBallCount * (1 : ℚ) / Nat.choose winnerBallCount winnerBallPick = 1 / 29322216 :=
by sorry

end NUMINAMATH_CALUDE_lottery_win_probability_l677_67747


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l677_67703

/-- Proves that given two people moving in opposite directions for 4 hours,
    with one person moving at 3 km/hr and the distance between them after 4 hours being 20 km,
    the speed of the other person is 2 km/hr. -/
theorem opposite_direction_speed
  (time : ℝ)
  (pooja_speed : ℝ)
  (distance : ℝ)
  (h1 : time = 4)
  (h2 : pooja_speed = 3)
  (h3 : distance = 20) :
  ∃ (other_speed : ℝ), other_speed = 2 ∧ distance = (other_speed + pooja_speed) * time :=
sorry

end NUMINAMATH_CALUDE_opposite_direction_speed_l677_67703


namespace NUMINAMATH_CALUDE_aprons_to_sew_tomorrow_l677_67740

def total_aprons : ℕ := 150
def aprons_before_today : ℕ := 13
def aprons_today : ℕ := 3 * aprons_before_today

def aprons_sewn_so_far : ℕ := aprons_before_today + aprons_today
def remaining_aprons : ℕ := total_aprons - aprons_sewn_so_far
def aprons_tomorrow : ℕ := remaining_aprons / 2

theorem aprons_to_sew_tomorrow : aprons_tomorrow = 49 := by
  sorry

end NUMINAMATH_CALUDE_aprons_to_sew_tomorrow_l677_67740


namespace NUMINAMATH_CALUDE_pedestrian_speed_theorem_l677_67766

/-- Given two pedestrians moving in the same direction, this theorem proves
    that the speed of the second pedestrian is either 6 m/s or 20/3 m/s,
    given the initial conditions. -/
theorem pedestrian_speed_theorem 
  (S₀ : ℝ) (v₁ : ℝ) (t : ℝ) (S : ℝ)
  (h₁ : S₀ = 200) 
  (h₂ : v₁ = 7)
  (h₃ : t = 5 * 60) -- 5 minutes in seconds
  (h₄ : S = 100) :
  ∃ v₂ : ℝ, (v₂ = 6 ∨ v₂ = 20/3) ∧ 
  (S₀ - S = (v₁ - v₂) * t) :=
by sorry

end NUMINAMATH_CALUDE_pedestrian_speed_theorem_l677_67766


namespace NUMINAMATH_CALUDE_sams_cycling_speed_l677_67742

/-- Given the cycling speeds of three friends, prove Sam's speed -/
theorem sams_cycling_speed 
  (lucas_speed : ℚ) 
  (maya_speed_ratio : ℚ) 
  (lucas_sam_ratio : ℚ)
  (h1 : lucas_speed = 5)
  (h2 : maya_speed_ratio = 4 / 5)
  (h3 : lucas_sam_ratio = 9 / 8) :
  lucas_speed * (8 / 9) = 40 / 9 := by
  sorry

#check sams_cycling_speed

end NUMINAMATH_CALUDE_sams_cycling_speed_l677_67742


namespace NUMINAMATH_CALUDE_denominator_of_0_27_repeating_l677_67706

def repeating_decimal_to_fraction (a b : ℕ) : ℚ :=
  (a : ℚ) / (99 : ℚ)

theorem denominator_of_0_27_repeating :
  (repeating_decimal_to_fraction 27 2).den = 11 := by
  sorry

end NUMINAMATH_CALUDE_denominator_of_0_27_repeating_l677_67706


namespace NUMINAMATH_CALUDE_abs_x_minus_one_necessary_not_sufficient_l677_67717

theorem abs_x_minus_one_necessary_not_sufficient :
  (∀ x : ℝ, x * (x + 1) < 0 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (x + 1) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_necessary_not_sufficient_l677_67717


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l677_67749

-- Define the quadratic equation
def quadratic_equation (x m : ℚ) : Prop :=
  3 * x^2 - 7 * x + m = 0

-- Define the condition for exactly one solution
def has_exactly_one_solution (m : ℚ) : Prop :=
  ∃! x, quadratic_equation x m

-- Theorem statement
theorem unique_solution_quadratic :
  ∀ m : ℚ, has_exactly_one_solution m → m = 49 / 12 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l677_67749


namespace NUMINAMATH_CALUDE_sum_cube_minus_twice_sum_square_is_zero_l677_67767

theorem sum_cube_minus_twice_sum_square_is_zero
  (p q r s : ℝ)
  (sum_condition : p + q + r + s = 8)
  (sum_square_condition : p^2 + q^2 + r^2 + s^2 = 16) :
  p^3 + q^3 + r^3 + s^3 - 2*(p^2 + q^2 + r^2 + s^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_cube_minus_twice_sum_square_is_zero_l677_67767


namespace NUMINAMATH_CALUDE_simplify_product_of_sqrt_l677_67737

theorem simplify_product_of_sqrt (y : ℝ) (hy : y > 0) :
  Real.sqrt (45 * y) * Real.sqrt (20 * y) * Real.sqrt (30 * y) = 30 * y * Real.sqrt (30 * y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_sqrt_l677_67737


namespace NUMINAMATH_CALUDE_parakeets_per_cage_l677_67759

theorem parakeets_per_cage 
  (num_cages : ℝ)
  (parrots_per_cage : ℝ)
  (total_birds : ℕ)
  (h1 : num_cages = 6.0)
  (h2 : parrots_per_cage = 6.0)
  (h3 : total_birds = 48) :
  (total_birds : ℝ) - num_cages * parrots_per_cage = num_cages * 2 :=
by sorry

end NUMINAMATH_CALUDE_parakeets_per_cage_l677_67759


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_special_properties_l677_67710

/-- A perfect power is a number of the form n^k where n and k are both natural numbers ≥ 2 -/
def is_perfect_power (x : ℕ) : Prop :=
  ∃ (n k : ℕ), n ≥ 2 ∧ k ≥ 2 ∧ x = n^k

/-- An arithmetic progression is a sequence where the difference between successive terms is constant -/
def is_arithmetic_progression (s : ℕ → ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ i, s i = a + i * d

theorem arithmetic_progression_with_special_properties :
  ∃ (s : ℕ → ℕ),
    is_arithmetic_progression s ∧
    (∀ i ∈ Finset.range 2016, ¬is_perfect_power (s i)) ∧
    is_perfect_power (Finset.prod (Finset.range 2016) s) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_special_properties_l677_67710


namespace NUMINAMATH_CALUDE_meals_per_day_l677_67762

theorem meals_per_day (people : ℕ) (total_plates : ℕ) (days : ℕ) (plates_per_meal : ℕ)
  (h1 : people = 6)
  (h2 : total_plates = 144)
  (h3 : days = 4)
  (h4 : plates_per_meal = 2)
  : (total_plates / (people * days * plates_per_meal) : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_meals_per_day_l677_67762


namespace NUMINAMATH_CALUDE_thirteen_binary_l677_67712

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Checks if a list of booleans represents a given natural number in binary -/
def is_binary_rep (n : ℕ) (bits : List Bool) : Prop :=
  to_binary n = bits

theorem thirteen_binary :
  is_binary_rep 13 [true, false, true, true] := by sorry

end NUMINAMATH_CALUDE_thirteen_binary_l677_67712


namespace NUMINAMATH_CALUDE_square_sum_difference_equals_243_l677_67728

theorem square_sum_difference_equals_243 : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 243 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_difference_equals_243_l677_67728


namespace NUMINAMATH_CALUDE_monotone_increasing_inequalities_l677_67778

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Theorem statement
theorem monotone_increasing_inequalities 
  (h1 : ∀ x, f' x > 0) 
  (h2 : ∀ x, HasDerivAt f (f' x) x) 
  (x₁ x₂ : ℝ) 
  (h3 : x₁ ≠ x₂) : 
  (f x₁ - f x₂) * (x₁ - x₂) > 0 ∧ 
  (f x₁ - f x₂) * (x₂ - x₁) < 0 ∧ 
  (f x₂ - f x₁) * (x₂ - x₁) > 0 :=
by sorry

end NUMINAMATH_CALUDE_monotone_increasing_inequalities_l677_67778


namespace NUMINAMATH_CALUDE_proportion_problem_l677_67755

theorem proportion_problem (y : ℝ) : (0.75 / 2 = 3 / y) → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l677_67755


namespace NUMINAMATH_CALUDE_coin_value_difference_l677_67760

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value in cents for a given coin count -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Represents the problem constraints -/
def validCoinCount (coins : CoinCount) : Prop :=
  coins.pennies ≥ 1 ∧ coins.nickels ≥ 1 ∧ coins.dimes ≥ 1 ∧
  coins.pennies + coins.nickels + coins.dimes = 3030

theorem coin_value_difference :
  ∃ (maxCoins minCoins : CoinCount),
    validCoinCount maxCoins ∧
    validCoinCount minCoins ∧
    (∀ c, validCoinCount c → totalValue c ≤ totalValue maxCoins) ∧
    (∀ c, validCoinCount c → totalValue c ≥ totalValue minCoins) ∧
    totalValue maxCoins - totalValue minCoins = 21182 :=
sorry

end NUMINAMATH_CALUDE_coin_value_difference_l677_67760


namespace NUMINAMATH_CALUDE_product_real_implies_b_value_l677_67770

theorem product_real_implies_b_value (z₁ z₂ : ℂ) (b : ℝ) :
  z₁ = 1 + I →
  z₂ = 2 + b * I →
  (z₁ * z₂).im = 0 →
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_product_real_implies_b_value_l677_67770


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l677_67719

/-- The probability of selecting 2 red balls from a bag containing 6 red, 4 blue, and 2 green balls -/
theorem probability_two_red_balls (red : ℕ) (blue : ℕ) (green : ℕ) 
  (h_red : red = 6) (h_blue : blue = 4) (h_green : green = 2) : 
  (Nat.choose red 2 : ℚ) / (Nat.choose (red + blue + green) 2) = 5 / 22 :=
sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l677_67719


namespace NUMINAMATH_CALUDE_smallest_set_size_l677_67771

theorem smallest_set_size (n : ℕ) (hn : n > 0) :
  let S := {S : Finset ℕ | S ⊆ Finset.range n ∧
    ∀ β : ℝ, β > 0 → (∀ s ∈ S, ∃ m : ℕ, s = ⌊β * m⌋) →
      ∀ k ∈ Finset.range n, ∃ m : ℕ, k = ⌊β * m⌋}
  ∃ S₀ ∈ S, S₀.card = n / 2 + 1 ∧ ∀ S' ∈ S, S'.card ≥ S₀.card :=
sorry

end NUMINAMATH_CALUDE_smallest_set_size_l677_67771


namespace NUMINAMATH_CALUDE_dads_nickels_l677_67756

/-- The number of nickels Tim had initially -/
def initial_nickels : ℕ := 9

/-- The number of nickels Tim has now -/
def current_nickels : ℕ := 12

/-- The number of nickels Tim's dad gave him -/
def nickels_from_dad : ℕ := current_nickels - initial_nickels

theorem dads_nickels : nickels_from_dad = 3 := by
  sorry

end NUMINAMATH_CALUDE_dads_nickels_l677_67756


namespace NUMINAMATH_CALUDE_oliver_bath_frequency_l677_67799

def bucket_capacity : ℕ := 120
def buckets_to_fill : ℕ := 14
def buckets_removed : ℕ := 3
def weekly_water_usage : ℕ := 9240

theorem oliver_bath_frequency :
  let full_tub := bucket_capacity * buckets_to_fill
  let water_removed := bucket_capacity * buckets_removed
  let water_per_bath := full_tub - water_removed
  weekly_water_usage / water_per_bath = 7 := by sorry

end NUMINAMATH_CALUDE_oliver_bath_frequency_l677_67799


namespace NUMINAMATH_CALUDE_fuel_cost_savings_l677_67727

theorem fuel_cost_savings (old_efficiency : ℝ) (old_fuel_cost : ℝ) 
  (efficiency_improvement : ℝ) (fuel_cost_increase : ℝ) (journey_distance : ℝ)
  (h1 : efficiency_improvement = 0.6)
  (h2 : fuel_cost_increase = 0.25)
  (h3 : journey_distance = 1000) : 
  let new_efficiency := old_efficiency * (1 + efficiency_improvement)
  let new_fuel_cost := old_fuel_cost * (1 + fuel_cost_increase)
  let old_journey_cost := journey_distance / old_efficiency * old_fuel_cost
  let new_journey_cost := journey_distance / new_efficiency * new_fuel_cost
  let percent_savings := (1 - new_journey_cost / old_journey_cost) * 100
  percent_savings = 21.875 := by
sorry

#eval (1 - (1000 / (1.6 * 1) * 1.25) / (1000 / 1 * 1)) * 100

end NUMINAMATH_CALUDE_fuel_cost_savings_l677_67727


namespace NUMINAMATH_CALUDE_min_distance_to_line_l677_67754

-- Define the vectors
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

-- Define the theorem
theorem min_distance_to_line 
  (m n : ℝ) 
  (h : (a.1 - m) * (-m) + (a.2 - n) * (b.2 - n) = 0) : 
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
  ∀ (x y : ℝ), x + y + 1 = 0 → 
  Real.sqrt ((x - m)^2 + (y - n)^2) ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l677_67754


namespace NUMINAMATH_CALUDE_section_B_avg_weight_l677_67787

def section_A_students : ℕ := 50
def section_B_students : ℕ := 40
def total_students : ℕ := section_A_students + section_B_students
def section_A_avg_weight : ℝ := 50
def total_avg_weight : ℝ := 58.89

theorem section_B_avg_weight :
  let section_B_weight := total_students * total_avg_weight - section_A_students * section_A_avg_weight
  section_B_weight / section_B_students = 70.0025 := by
sorry

end NUMINAMATH_CALUDE_section_B_avg_weight_l677_67787


namespace NUMINAMATH_CALUDE_uncovered_side_length_l677_67757

/-- A rectangular field with three sides fenced -/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing : ℝ

/-- The uncovered side of the field is 20 feet long -/
theorem uncovered_side_length (field : FencedField) 
  (h_area : field.area = 120)
  (h_fencing : field.fencing = 32)
  (h_rectangle : field.area = field.length * field.width)
  (h_fence_sides : field.fencing = field.length + 2 * field.width) :
  field.length = 20 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_side_length_l677_67757


namespace NUMINAMATH_CALUDE_textbook_cost_decrease_l677_67700

theorem textbook_cost_decrease (original_cost new_cost : ℝ) 
  (h1 : original_cost = 75)
  (h2 : new_cost = 60) :
  (1 - new_cost / original_cost) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_textbook_cost_decrease_l677_67700


namespace NUMINAMATH_CALUDE_tan_equality_periodic_l677_67726

theorem tan_equality_periodic (n : ℤ) : 
  -180 < n ∧ n < 180 → 
  Real.tan (n * π / 180) = Real.tan (1540 * π / 180) → 
  n = 40 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_periodic_l677_67726


namespace NUMINAMATH_CALUDE_best_card_to_disprove_l677_67758

-- Define the set of visible card sides
inductive CardSide
| Letter (c : Char)
| Number (n : Nat)

-- Define a card as a pair of sides
def Card := (CardSide × CardSide)

-- Define the property of being a consonant
def isConsonant (c : Char) : Prop := c ∈ ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

-- Define the property of being an odd number
def isOdd (n : Nat) : Prop := n % 2 = 1

-- John's statement as a function
def johnsStatement (card : Card) : Prop :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) => isConsonant c → isOdd n
  | (CardSide.Number n, CardSide.Letter c) => isConsonant c → isOdd n
  | _ => True

-- The set of visible card sides
def visibleSides : List CardSide := [CardSide.Letter 'A', CardSide.Letter 'B', CardSide.Number 7, CardSide.Number 8, CardSide.Number 9]

-- The theorem to prove
theorem best_card_to_disprove (cards : List Card) :
  (∀ card ∈ cards, (CardSide.Number 8 ∈ card.1 :: card.2 :: []) →
    ¬(∀ c ∈ cards, johnsStatement c)) →
  (∀ side ∈ visibleSides, side ≠ CardSide.Number 8 →
    ∃ c ∈ cards, (side ∈ c.1 :: c.2 :: []) ∧
      (∀ card ∈ cards, (side ∈ card.1 :: card.2 :: []) →
        (∃ c' ∈ cards, ¬johnsStatement c'))) :=
by sorry

end NUMINAMATH_CALUDE_best_card_to_disprove_l677_67758


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l677_67798

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {y | ∃ x ∈ M, y = 2 * x}

theorem intersection_of_M_and_N : M ∩ N = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l677_67798


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l677_67735

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 7 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ 3 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l677_67735


namespace NUMINAMATH_CALUDE_problem_1_l677_67777

theorem problem_1 : 3^2 * (-1 + 3) - (-16) / 8 = 20 := by sorry

end NUMINAMATH_CALUDE_problem_1_l677_67777


namespace NUMINAMATH_CALUDE_power_calculation_l677_67751

theorem power_calculation : 16^16 * 8^8 / 4^40 = 256 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l677_67751


namespace NUMINAMATH_CALUDE_largest_a_value_l677_67721

theorem largest_a_value (a : ℝ) :
  (5 * Real.sqrt ((3 * a)^2 + 2^2) - 5 * a^2 - 2) / (Real.sqrt (2 + 3 * a^2) + 2) = 1 →
  ∃ y : ℝ, y^2 - (5 * Real.sqrt 3 - 1) * y + 5 = 0 ∧
           y ≥ (5 * Real.sqrt 3 - 1 + Real.sqrt ((5 * Real.sqrt 3 - 1)^2 - 20)) / 2 ∧
           a = Real.sqrt ((y^2 - 2) / 3) ∧
           ∀ a' : ℝ, (5 * Real.sqrt ((3 * a')^2 + 2^2) - 5 * a'^2 - 2) / (Real.sqrt (2 + 3 * a'^2) + 2) = 1 →
                     a' ≤ a :=
by sorry

end NUMINAMATH_CALUDE_largest_a_value_l677_67721


namespace NUMINAMATH_CALUDE_min_length_PQ_l677_67788

-- Define the arithmetic sequence property
def is_arithmetic_sequence (a b c : ℝ) : Prop := b = (a + c) / 2

-- Define the moving line l
def line_l (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

-- Define point A
def point_A : ℝ × ℝ := (1, 2)

-- Define the line that Q lies on
def line_Q (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0

-- Define the minimum distance function
def min_distance (A P Q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem min_length_PQ (a b c : ℝ) :
  a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 →
  is_arithmetic_sequence a b c →
  ∃ (P Q : ℝ × ℝ),
    line_l a b c P.1 P.2 ∧
    line_Q Q.1 Q.2 ∧
    (∀ (P' Q' : ℝ × ℝ),
      line_l a b c P'.1 P'.2 →
      line_Q Q'.1 Q'.2 →
      min_distance point_A P Q ≤ min_distance point_A P' Q') →
    min_distance point_A P Q = 1 :=
sorry

end NUMINAMATH_CALUDE_min_length_PQ_l677_67788


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_including_13_l677_67724

theorem unique_number_with_three_prime_divisors_including_13 :
  ∀ x n : ℕ,
  x = 9^n - 1 →
  (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
  13 ∣ x →
  x = 728 := by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_including_13_l677_67724


namespace NUMINAMATH_CALUDE_two_is_sup_of_satisfying_set_l677_67739

/-- A sequence of positive integers satisfying the given inequality -/
def SatisfyingSequence (r : ℝ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, (a n ≤ a (n + 2)) ∧ ((a (n + 2))^2 ≤ (a n)^2 + r * (a (n + 1)))

/-- The property that a sequence eventually becomes constant -/
def EventuallyConstant (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n ≥ M, a (n + 2) = a n

/-- The set of real numbers r satisfying the given condition -/
def SatisfyingSet : Set ℝ :=
  {r : ℝ | ∀ a : ℕ → ℕ, SatisfyingSequence r a → EventuallyConstant a}

/-- The main theorem: 2 is the supremum of the satisfying set -/
theorem two_is_sup_of_satisfying_set : 
  IsLUB SatisfyingSet 2 := by sorry

end NUMINAMATH_CALUDE_two_is_sup_of_satisfying_set_l677_67739


namespace NUMINAMATH_CALUDE_wickets_in_last_match_is_five_l677_67796

/-- Represents the bowling statistics of a cricket player -/
structure BowlingStats where
  initialAverage : ℝ
  runsLastMatch : ℕ
  averageDecrease : ℝ
  wicketsBeforeLastMatch : ℕ

/-- Calculates the number of wickets taken in the last match -/
def wicketsInLastMatch (stats : BowlingStats) : ℕ :=
  sorry

/-- Theorem stating that given the specific bowling statistics, the number of wickets in the last match is 5 -/
theorem wickets_in_last_match_is_five (stats : BowlingStats) 
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.runsLastMatch = 26)
  (h3 : stats.averageDecrease = 0.4)
  (h4 : stats.wicketsBeforeLastMatch = 85) :
  wicketsInLastMatch stats = 5 := by
  sorry

end NUMINAMATH_CALUDE_wickets_in_last_match_is_five_l677_67796


namespace NUMINAMATH_CALUDE_ab_plus_cd_equals_zero_l677_67789

theorem ab_plus_cd_equals_zero 
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : a*d - b*c = -1) : 
  a*b + c*d = 0 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_equals_zero_l677_67789


namespace NUMINAMATH_CALUDE_supermarket_spending_l677_67732

theorem supermarket_spending (F : ℚ) : 
  (∃ (M : ℚ), 
    M = 150 ∧ 
    F * M + (1/3) * M + (1/10) * M + 10 = M) →
  F = 1/2 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l677_67732


namespace NUMINAMATH_CALUDE_cistern_fill_time_l677_67764

def fill_time (rate_a rate_b rate_c : ℚ) : ℚ :=
  1 / (rate_a + rate_b + rate_c)

theorem cistern_fill_time :
  let rate_a : ℚ := 1 / 10
  let rate_b : ℚ := 1 / 12
  let rate_c : ℚ := -1 / 15
  fill_time rate_a rate_b rate_c = 60 / 7 :=
by sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l677_67764


namespace NUMINAMATH_CALUDE_cosine_amplitude_l677_67784

/-- Given a cosine function y = a * cos(bx) where a > 0 and b > 0,
    if the maximum value is 3 and the minimum value is -3, then a = 3 -/
theorem cosine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmax : ∀ x, a * Real.cos (b * x) ≤ 3) 
  (hmin : ∀ x, a * Real.cos (b * x) ≥ -3)
  (hreach_max : ∃ x, a * Real.cos (b * x) = 3)
  (hreach_min : ∃ x, a * Real.cos (b * x) = -3) : 
  a = 3 := by sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l677_67784


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l677_67715

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the foci of a hyperbola -/
structure Foci where
  F₁ : Point
  F₂ : Point

/-- Checks if a point is on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1

/-- Calculates the angle between three points -/
noncomputable def angle (p₁ p₂ p₃ : Point) : ℝ := sorry

/-- Calculates the eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) (f : Foci) (p : Point) :
  on_hyperbola h p →
  angle f.F₁ p f.F₂ = Real.pi / 2 →
  2 * angle p f.F₁ f.F₂ = angle p f.F₂ f.F₁ →
  eccentricity h = Real.sqrt 3 + 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l677_67715


namespace NUMINAMATH_CALUDE_orthogonal_projection_magnitude_l677_67761

/-- Given two vectors a and b in ℝ², prove that the magnitude of the orthogonal projection of a onto b is √5 -/
theorem orthogonal_projection_magnitude (a b : ℝ × ℝ) (h1 : a = (3, -1)) (h2 : b = (1, -2)) :
  ‖(((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b)‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_projection_magnitude_l677_67761


namespace NUMINAMATH_CALUDE_flour_amount_proof_l677_67708

/-- The amount of flour in the first combination -/
def flour_amount : ℝ := 17.78

/-- The cost per pound of sugar and flour -/
def cost_per_pound : ℝ := 0.45

/-- The total cost of both combinations -/
def total_cost : ℝ := 26

theorem flour_amount_proof :
  (40 * cost_per_pound + flour_amount * cost_per_pound = total_cost) ∧
  (30 * cost_per_pound + 25 * cost_per_pound = total_cost) →
  flour_amount = 17.78 := by sorry

end NUMINAMATH_CALUDE_flour_amount_proof_l677_67708


namespace NUMINAMATH_CALUDE_noah_total_earnings_l677_67772

/-- Noah's work schedule and earnings --/
structure WorkSchedule where
  hours_week1 : ℕ
  hours_week2 : ℕ
  earnings_difference : ℚ

/-- Calculate total earnings for two weeks --/
def total_earnings (w : WorkSchedule) (hourly_wage : ℚ) : ℚ :=
  hourly_wage * (w.hours_week1 + w.hours_week2)

/-- Noah's work data --/
def noah_work : WorkSchedule :=
  { hours_week1 := 18
  , hours_week2 := 25
  , earnings_difference := 54 }

/-- Theorem: Noah's total earnings for both weeks --/
theorem noah_total_earnings :
  ∃ (hourly_wage : ℚ),
    hourly_wage * (noah_work.hours_week2 - noah_work.hours_week1) = noah_work.earnings_difference ∧
    total_earnings noah_work hourly_wage = 331.71 := by
  sorry

end NUMINAMATH_CALUDE_noah_total_earnings_l677_67772


namespace NUMINAMATH_CALUDE_exactly_one_true_l677_67750

def X : Set Int := {x | -2 < x ∧ x ≤ 3}

def p (a : ℝ) : Prop := ∀ x ∈ X, (1/3 : ℝ) * x^2 < 2*a - 3

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + a = 0

theorem exactly_one_true (a : ℝ) : 
  (p a ∧ ¬(q a)) ∨ (¬(p a) ∧ q a) ↔ a ≤ 1 ∨ a > 3 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_true_l677_67750


namespace NUMINAMATH_CALUDE_louise_pictures_l677_67774

def total_pictures (vertical horizontal haphazard : ℕ) : ℕ :=
  vertical + horizontal + haphazard

theorem louise_pictures : 
  ∀ (vertical horizontal haphazard : ℕ),
    vertical = 10 →
    horizontal = vertical / 2 →
    haphazard = 5 →
    total_pictures vertical horizontal haphazard = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_louise_pictures_l677_67774


namespace NUMINAMATH_CALUDE_fifteen_is_counterexample_l677_67792

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_counterexample (n : ℕ) : Prop :=
  ¬(is_prime n) ∧ ¬(is_prime (n - 3))

theorem fifteen_is_counterexample :
  is_counterexample 15 :=
sorry

end NUMINAMATH_CALUDE_fifteen_is_counterexample_l677_67792


namespace NUMINAMATH_CALUDE_cartesian_product_subset_cartesian_product_intersection_y_axis_representation_l677_67743

-- Define the Cartesian product
def cartesianProduct (A B : Set α) : Set (α × α) :=
  {p | p.1 ∈ A ∧ p.2 ∈ B}

-- Statement 1
theorem cartesian_product_subset {A B C : Set α} (h : A ⊆ C) :
  cartesianProduct A B ⊆ cartesianProduct C B :=
sorry

-- Statement 2
theorem cartesian_product_intersection {A B C : Set α} :
  cartesianProduct A (B ∩ C) = cartesianProduct A B ∩ cartesianProduct A C :=
sorry

-- Statement 3
theorem y_axis_representation {R : Type} [LinearOrderedField R] :
  cartesianProduct {(0 : R)} (Set.univ : Set R) = {p : R × R | p.1 = 0} :=
sorry

end NUMINAMATH_CALUDE_cartesian_product_subset_cartesian_product_intersection_y_axis_representation_l677_67743
