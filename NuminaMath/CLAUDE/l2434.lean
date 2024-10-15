import Mathlib

namespace NUMINAMATH_CALUDE_similar_triangles_hypotenuse_l2434_243481

-- Define the properties of the triangles
def smallTriangleArea : ℝ := 8
def largeTriangleArea : ℝ := 200
def smallTriangleHypotenuse : ℝ := 10

-- Define the theorem
theorem similar_triangles_hypotenuse :
  ∃ (smallLeg1 smallLeg2 largeLeg1 largeLeg2 largeHypotenuse : ℝ),
    -- Conditions for the smaller triangle
    smallLeg1 > 0 ∧ smallLeg2 > 0 ∧
    smallLeg1 * smallLeg2 / 2 = smallTriangleArea ∧
    smallLeg1^2 + smallLeg2^2 = smallTriangleHypotenuse^2 ∧
    -- Conditions for the larger triangle
    largeLeg1 > 0 ∧ largeLeg2 > 0 ∧
    largeLeg1 * largeLeg2 / 2 = largeTriangleArea ∧
    largeLeg1^2 + largeLeg2^2 = largeHypotenuse^2 ∧
    -- Similarity condition
    largeLeg1 / smallLeg1 = largeLeg2 / smallLeg2 ∧
    -- Conclusion
    largeHypotenuse = 50 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_hypotenuse_l2434_243481


namespace NUMINAMATH_CALUDE_angle_function_equality_l2434_243464

/-- Given an angle α in the third quadrant, if cos(α - 3π/2) = 1/5, then
    (sin(α - π/2) * cos(3π/2 + α) * tan(π - α)) / (tan(-α - π) * sin(-α - π)) = 2√6/5 -/
theorem angle_function_equality (α : Real) 
    (h1 : π < α ∧ α < 3*π/2)  -- α is in the third quadrant
    (h2 : Real.cos (α - 3*π/2) = 1/5) :
    (Real.sin (α - π/2) * Real.cos (3*π/2 + α) * Real.tan (π - α)) / 
    (Real.tan (-α - π) * Real.sin (-α - π)) = 2 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_function_equality_l2434_243464


namespace NUMINAMATH_CALUDE_survey_properties_l2434_243424

/-- Represents a student in the survey -/
structure Student where
  physicalCondition : String

/-- Represents the survey conducted by the school -/
structure Survey where
  students : List Student
  classes : Nat

/-- Defines the sample of the survey -/
def sample (s : Survey) : String :=
  s.students.map (λ student => student.physicalCondition) |> toString

/-- Defines the sample size of the survey -/
def sampleSize (s : Survey) : Nat :=
  s.students.length

/-- Theorem stating the properties of the survey -/
theorem survey_properties (s : Survey) 
  (h1 : s.students.length = 190)
  (h2 : s.classes = 19) :
  sample s = "physical condition of 190 students" ∧ 
  sampleSize s = 190 := by
  sorry

#check survey_properties

end NUMINAMATH_CALUDE_survey_properties_l2434_243424


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2434_243448

/-- Represents the ratio of two numbers as a pair of integers -/
structure Ratio where
  num : Int
  den : Int
  pos : 0 < den

def Ratio.of (a b : Int) (h : 0 < b) : Ratio :=
  ⟨a, b, h⟩

theorem age_ratio_problem (rahul_future_age deepak_current_age : ℕ) 
  (h1 : rahul_future_age = 50)
  (h2 : deepak_current_age = 33) :
  ∃ (r : Ratio), r = Ratio.of 4 3 (by norm_num) ∧ 
    (rahul_future_age - 6 : ℚ) / deepak_current_age = r.num / r.den := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2434_243448


namespace NUMINAMATH_CALUDE_simplify_and_sum_exponents_l2434_243440

theorem simplify_and_sum_exponents 
  (a b c : ℝ) 
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) : 
  ∃ (k : ℝ), 
    (k > 0) ∧ 
    (k^3 = a^2 * b^4 * c^2) ∧ 
    ((72 * a^5 * b^7 * c^14)^(1/3) = 2 * 3^(2/3) * a * b * c^4 * k) ∧
    (1 + 1 + 4 = 6) := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_sum_exponents_l2434_243440


namespace NUMINAMATH_CALUDE_product_of_max_min_a_l2434_243442

theorem product_of_max_min_a (a b c : ℝ) 
  (sum_eq : a + b + c = 15) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 100) : 
  let f := fun x : ℝ => (5 + (5 * Real.sqrt 6) / 3) * (5 - (5 * Real.sqrt 6) / 3)
  f a = 25 / 3 := by
sorry

end NUMINAMATH_CALUDE_product_of_max_min_a_l2434_243442


namespace NUMINAMATH_CALUDE_triangle_formation_l2434_243401

theorem triangle_formation (a : ℝ) : 
  (0 < a ∧ a + 3 > 5 ∧ a + 5 > 3 ∧ 3 + 5 > a) ↔ a = 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l2434_243401


namespace NUMINAMATH_CALUDE_critical_point_iff_a_in_range_l2434_243441

/-- The function f(x) = x³ - ax² + ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + a*x + 3

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + a

/-- A critical point of f exists if and only if f'(x) = 0 has real solutions -/
def has_critical_point (a : ℝ) : Prop := ∃ x : ℝ, f' a x = 0

/-- The main theorem: f(x) has a critical point iff a ∈ (-∞, 0) ∪ (3, +∞) -/
theorem critical_point_iff_a_in_range (a : ℝ) :
  has_critical_point a ↔ a < 0 ∨ a > 3 := by sorry

end NUMINAMATH_CALUDE_critical_point_iff_a_in_range_l2434_243441


namespace NUMINAMATH_CALUDE_cheerleader_count_l2434_243480

theorem cheerleader_count (size2 : ℕ) (size6 : ℕ) : 
  size2 = 4 → size6 = 10 → size2 + size6 + (size6 / 2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_cheerleader_count_l2434_243480


namespace NUMINAMATH_CALUDE_enchanted_creatures_gala_handshakes_l2434_243413

/-- The number of handshakes at the Enchanted Creatures Gala -/
theorem enchanted_creatures_gala_handshakes : 
  let num_goblins : ℕ := 30
  let num_trolls : ℕ := 20
  let goblin_handshakes := num_goblins * (num_goblins - 1) / 2
  let goblin_troll_handshakes := num_goblins * num_trolls
  goblin_handshakes + goblin_troll_handshakes = 1035 := by
  sorry

#check enchanted_creatures_gala_handshakes

end NUMINAMATH_CALUDE_enchanted_creatures_gala_handshakes_l2434_243413


namespace NUMINAMATH_CALUDE_cubic_monotonicity_implies_one_intersection_one_intersection_not_implies_monotonicity_l2434_243444

-- Define the cubic function
def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Define strict monotonicity
def strictly_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y ∨ (∀ x y, x < y → f x > f y)

-- Define the property of intersecting x-axis exactly once
def intersects_x_axis_once (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

-- Main theorem
theorem cubic_monotonicity_implies_one_intersection
  (a b c d : ℝ) (h : a ≠ 0) :
  strictly_monotonic (f a b c d) →
  intersects_x_axis_once (f a b c d) :=
sorry

-- Counterexample theorem
theorem one_intersection_not_implies_monotonicity :
  ∃ a b c d : ℝ,
    intersects_x_axis_once (f a b c d) ∧
    ¬strictly_monotonic (f a b c d) :=
sorry

end NUMINAMATH_CALUDE_cubic_monotonicity_implies_one_intersection_one_intersection_not_implies_monotonicity_l2434_243444


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2434_243438

/-- The solution set of the inequality x + 2/(x+1) > 2 -/
theorem solution_set_inequality (x : ℝ) : x + 2 / (x + 1) > 2 ↔ x ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2434_243438


namespace NUMINAMATH_CALUDE_lacustrine_glacial_monoliths_l2434_243482

-- Define the total number of monoliths
def total_monoliths : ℕ := 98

-- Define the probability of a monolith being sand
def prob_sand : ℚ := 1/7

-- Define the probability of a monolith being marine loam
def prob_marine_loam : ℚ := 9/14

-- Theorem statement
theorem lacustrine_glacial_monoliths :
  let sand_monoliths := (prob_sand * total_monoliths : ℚ).num
  let loam_monoliths := total_monoliths - sand_monoliths
  let marine_loam_monoliths := (prob_marine_loam * loam_monoliths : ℚ).num
  let lacustrine_glacial_loam_monoliths := loam_monoliths - marine_loam_monoliths
  sand_monoliths + lacustrine_glacial_loam_monoliths = 44 := by
  sorry

end NUMINAMATH_CALUDE_lacustrine_glacial_monoliths_l2434_243482


namespace NUMINAMATH_CALUDE_investment_bankers_count_l2434_243471

/-- Proves that the number of investment bankers is 4 given the problem conditions -/
theorem investment_bankers_count : 
  ∀ (total_bill : ℝ) (avg_cost : ℝ) (num_clients : ℕ),
  total_bill = 756 →
  avg_cost = 70 →
  num_clients = 5 →
  ∃ (num_bankers : ℕ),
    num_bankers = 4 ∧
    total_bill = (avg_cost * (num_bankers + num_clients : ℝ)) * 1.2 :=
by sorry

end NUMINAMATH_CALUDE_investment_bankers_count_l2434_243471


namespace NUMINAMATH_CALUDE_fraction_sum_lower_bound_l2434_243429

theorem fraction_sum_lower_bound (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y + z) + y / (z + x) + z / (x + y) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_lower_bound_l2434_243429


namespace NUMINAMATH_CALUDE_coin_toss_probability_l2434_243493

-- Define the probability of landing heads
def p : ℚ := 3/5

-- Define the number of tosses
def n : ℕ := 4

-- Define the number of desired heads
def k : ℕ := 2

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

-- Define the probability of getting exactly k heads in n tosses
def probability (p : ℚ) (n k : ℕ) : ℚ :=
  binomial_coeff n k * p^k * (1 - p)^(n - k)

-- State the theorem
theorem coin_toss_probability : probability p n k = 216/625 := by sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l2434_243493


namespace NUMINAMATH_CALUDE_optimal_gcd_l2434_243432

/-- The number of integers to choose from (0 to 81 inclusive) -/
def n : ℕ := 82

/-- The set of numbers to choose from -/
def S : Finset ℕ := Finset.range n

/-- Amy's strategy: A function that takes the current state and returns Amy's choice -/
def amy_strategy : S → S → ℕ → ℕ := sorry

/-- Bob's strategy: A function that takes the current state and returns Bob's choice -/
def bob_strategy : S → S → ℕ → ℕ := sorry

/-- The sum of Amy's chosen numbers -/
def A (amy_nums : Finset ℕ) : ℕ := amy_nums.sum id

/-- The sum of Bob's chosen numbers -/
def B (bob_nums : Finset ℕ) : ℕ := bob_nums.sum id

/-- The game result when Amy and Bob play optimally -/
def optimal_play : Finset ℕ × Finset ℕ := sorry

/-- The theorem stating the optimal gcd when Amy and Bob play optimally -/
theorem optimal_gcd :
  let (amy_nums, bob_nums) := optimal_play
  Nat.gcd (A amy_nums) (B bob_nums) = 41 := by sorry

end NUMINAMATH_CALUDE_optimal_gcd_l2434_243432


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_equation_l2434_243421

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if the distance from the right focus to the left vertex is equal to twice
    the distance from it to the asymptote, then the equation of its asymptote
    is 4x ± 3y = 0. -/
theorem hyperbola_asymptote_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_distance : ∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 →
    (∃ (c : ℝ), a + c = 2 * (b * c / Real.sqrt (a^2 + b^2)))) :
  ∃ (k : ℝ), k > 0 ∧ (∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 → (4*x = 3*y ∨ 4*x = -3*y)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_equation_l2434_243421


namespace NUMINAMATH_CALUDE_translation_theorem_l2434_243430

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point in 2D space -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_theorem :
  let A : Point := { x := -3, y := 2 }
  let A' : Point := translate (translate A 4 0) 0 (-3)
  A'.x = 1 ∧ A'.y = -1 := by sorry

end NUMINAMATH_CALUDE_translation_theorem_l2434_243430


namespace NUMINAMATH_CALUDE_job_size_ratio_l2434_243400

/-- Given two jobs with different numbers of workers and days, 
    prove that the ratio of work done in the new job to the original job is 3. -/
theorem job_size_ratio (original_workers original_days new_workers new_days : ℕ) 
    (h1 : original_workers = 250)
    (h2 : original_days = 16)
    (h3 : new_workers = 600)
    (h4 : new_days = 20) : 
    (new_workers * new_days) / (original_workers * original_days) = 3 := by
  sorry


end NUMINAMATH_CALUDE_job_size_ratio_l2434_243400


namespace NUMINAMATH_CALUDE_game_properties_l2434_243455

/-- Represents the "What? Where? When?" game --/
structure Game where
  num_envelopes : Nat
  points_to_win : Nat
  num_games : Nat

/-- Calculates the expected number of points for one team in multiple games --/
def expectedPoints (g : Game) : ℝ :=
  sorry

/-- Calculates the probability of a specific envelope being chosen --/
def envelopeProbability (g : Game) : ℝ :=
  sorry

/-- Theorem stating the expected points and envelope probability for the given game --/
theorem game_properties :
  let g : Game := { num_envelopes := 13, points_to_win := 6, num_games := 100 }
  (expectedPoints g = 465) ∧ (envelopeProbability g = 12 / 13) := by
  sorry

end NUMINAMATH_CALUDE_game_properties_l2434_243455


namespace NUMINAMATH_CALUDE_martins_bells_l2434_243443

theorem martins_bells (S B : ℤ) : 
  S = B / 3 + B^2 / 4 →
  S + B = 52 →
  B > 0 →
  B = 12 := by
sorry

end NUMINAMATH_CALUDE_martins_bells_l2434_243443


namespace NUMINAMATH_CALUDE_student_arrangements_l2434_243436

/-- The number of students in the row -/
def n : ℕ := 7

/-- Calculate the number of arrangements where two students are adjacent -/
def arrangements_two_adjacent (n : ℕ) : ℕ := sorry

/-- Calculate the number of arrangements where three students are adjacent -/
def arrangements_three_adjacent (n : ℕ) : ℕ := sorry

/-- Calculate the number of arrangements where two students are adjacent and one student is not at either end -/
def arrangements_two_adjacent_one_not_end (n : ℕ) : ℕ := sorry

/-- Calculate the number of arrangements where three students are together and the other four are together -/
def arrangements_two_groups (n : ℕ) : ℕ := sorry

theorem student_arrangements :
  arrangements_two_adjacent n = 1440 ∧
  arrangements_three_adjacent n = 720 ∧
  arrangements_two_adjacent_one_not_end n = 960 ∧
  arrangements_two_groups n = 288 := by sorry

end NUMINAMATH_CALUDE_student_arrangements_l2434_243436


namespace NUMINAMATH_CALUDE_exponential_inequality_l2434_243496

theorem exponential_inequality (a b : ℝ) (h : a > b) : (2 : ℝ) ^ a > (2 : ℝ) ^ b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2434_243496


namespace NUMINAMATH_CALUDE_sum_of_squares_verify_sum_of_squares_l2434_243452

theorem sum_of_squares : ℕ → Prop
  | 1009 => 1009 = 15^2 + 28^2
  | 2018 => 2018 = 43^2 + 13^2
  | _ => True

theorem verify_sum_of_squares :
  sum_of_squares 1009 ∧ sum_of_squares 2018 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_verify_sum_of_squares_l2434_243452


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2434_243460

def expression (n : ℕ) : ℤ :=
  10 * (n - 3)^5 - 2 * n^2 + 20 * n - 36

theorem largest_n_divisible_by_seven :
  ∃ (n : ℕ), n < 50000 ∧
    7 ∣ expression n ∧
    ∀ (m : ℕ), m < 50000 → 7 ∣ expression m → m ≤ n :=
by
  use 49999
  sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2434_243460


namespace NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l2434_243428

theorem equation_represents_pair_of_lines : 
  ∃ (m₁ m₂ : ℝ), ∀ (x y : ℝ), 4 * x^2 - 9 * y^2 = 0 ↔ (y = m₁ * x ∨ y = m₂ * x) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l2434_243428


namespace NUMINAMATH_CALUDE_smallest_a_value_l2434_243474

/-- Represents a parabola with equation y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem stating the smallest possible value of a for the given parabola -/
theorem smallest_a_value (p : Parabola) 
  (vertex_x : p.a * (1/3)^2 + p.b * (1/3) + p.c = -5/9) 
  (a_positive : p.a > 0)
  (sum_integer : ∃ n : ℤ, p.a + p.b + p.c = n) :
  p.a ≥ 5/4 ∧ ∃ (q : Parabola), q.a = 5/4 ∧ 
    q.a * (1/3)^2 + q.b * (1/3) + q.c = -5/9 ∧ 
    q.a > 0 ∧ 
    (∃ n : ℤ, q.a + q.b + q.c = n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2434_243474


namespace NUMINAMATH_CALUDE_sum_of_like_monomials_l2434_243491

/-- The sum of like monomials -/
theorem sum_of_like_monomials (m n : ℕ) :
  (2 * n : ℤ) * X^(m + 2) * Y^7 + (-4 * m : ℤ) * X^4 * Y^(3 * n - 2) = -2 * X^4 * Y^7 :=
by
  sorry

#check sum_of_like_monomials

end NUMINAMATH_CALUDE_sum_of_like_monomials_l2434_243491


namespace NUMINAMATH_CALUDE_quarter_circle_square_perimeter_l2434_243484

/-- The perimeter of a region bounded by quarter-circular arcs constructed at each corner of a square with sides measuring 4/π is equal to 8. -/
theorem quarter_circle_square_perimeter :
  let square_side : ℝ := 4 / Real.pi
  let quarter_circle_radius : ℝ := square_side
  let quarter_circle_count : ℕ := 4
  let region_perimeter : ℝ := quarter_circle_count * (Real.pi * quarter_circle_radius / 2)
  region_perimeter = 8 := by
  sorry

end NUMINAMATH_CALUDE_quarter_circle_square_perimeter_l2434_243484


namespace NUMINAMATH_CALUDE_division_result_l2434_243427

theorem division_result (x : ℚ) : x / 5000 = 0.0114 → x = 57 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l2434_243427


namespace NUMINAMATH_CALUDE_mrs_brown_shoe_price_l2434_243497

/-- Calculates the final price for a Mother's Day purchase with additional discount for multiple children -/
def mothersDayPrice (originalPrice : ℝ) (numChildren : ℕ) : ℝ :=
  let mothersDayDiscount := 0.1
  let additionalDiscount := 0.04
  let discountedPrice := originalPrice * (1 - mothersDayDiscount)
  if numChildren ≥ 3 then
    discountedPrice * (1 - additionalDiscount)
  else
    discountedPrice

theorem mrs_brown_shoe_price :
  mothersDayPrice 125 4 = 108 := by
  sorry

end NUMINAMATH_CALUDE_mrs_brown_shoe_price_l2434_243497


namespace NUMINAMATH_CALUDE_simplest_form_sum_l2434_243447

theorem simplest_form_sum (a b : ℕ) (h : a = 63 ∧ b = 117) :
  let (n, d) := (a / gcd a b, b / gcd a b)
  n + d = 20 := by
sorry

end NUMINAMATH_CALUDE_simplest_form_sum_l2434_243447


namespace NUMINAMATH_CALUDE_bobs_total_bushels_l2434_243494

/-- Calculates the number of bushels from a row of corn, rounding down to the nearest whole bushel -/
def bushelsFromRow (stalks : ℕ) (stalksPerBushel : ℕ) : ℕ :=
  stalks / stalksPerBushel

/-- Represents Bob's corn harvest -/
structure CornHarvest where
  row1 : (ℕ × ℕ)
  row2 : (ℕ × ℕ)
  row3 : (ℕ × ℕ)
  row4 : (ℕ × ℕ)
  row5 : (ℕ × ℕ)
  row6 : (ℕ × ℕ)
  row7 : (ℕ × ℕ)

/-- Calculates the total bushels of corn from Bob's harvest -/
def totalBushels (harvest : CornHarvest) : ℕ :=
  bushelsFromRow harvest.row1.1 harvest.row1.2 +
  bushelsFromRow harvest.row2.1 harvest.row2.2 +
  bushelsFromRow harvest.row3.1 harvest.row3.2 +
  bushelsFromRow harvest.row4.1 harvest.row4.2 +
  bushelsFromRow harvest.row5.1 harvest.row5.2 +
  bushelsFromRow harvest.row6.1 harvest.row6.2 +
  bushelsFromRow harvest.row7.1 harvest.row7.2

/-- Bob's actual corn harvest -/
def bobsHarvest : CornHarvest :=
  { row1 := (82, 8)
    row2 := (94, 9)
    row3 := (78, 7)
    row4 := (96, 12)
    row5 := (85, 10)
    row6 := (91, 13)
    row7 := (88, 11) }

theorem bobs_total_bushels :
  totalBushels bobsHarvest = 62 := by
  sorry

end NUMINAMATH_CALUDE_bobs_total_bushels_l2434_243494


namespace NUMINAMATH_CALUDE_julia_tag_game_l2434_243434

theorem julia_tag_game (total : ℕ) (monday : ℕ) (tuesday : ℕ) 
  (h1 : total = 18) 
  (h2 : monday = 4) 
  (h3 : total = monday + tuesday) : 
  tuesday = 14 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_game_l2434_243434


namespace NUMINAMATH_CALUDE_percentage_loss_is_twenty_percent_l2434_243473

/-- Calculates the percentage loss given the selling conditions --/
def calculate_percentage_loss (initial_articles : ℕ) (initial_price : ℚ) (initial_gain_percent : ℚ) 
  (final_articles : ℚ) (final_price : ℚ) : ℚ :=
  let initial_cost := initial_price / (1 + initial_gain_percent / 100)
  let cost_per_article := initial_cost / initial_articles
  let final_cost := cost_per_article * final_articles
  let loss := final_cost - final_price
  (loss / final_cost) * 100

/-- The percentage loss is 20% given the specified conditions --/
theorem percentage_loss_is_twenty_percent :
  calculate_percentage_loss 20 60 20 20 40 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_loss_is_twenty_percent_l2434_243473


namespace NUMINAMATH_CALUDE_panda_increase_l2434_243420

/-- Represents the number of animals in the zoo -/
structure ZooPopulation where
  cheetahs : ℕ
  pandas : ℕ

/-- The ratio of cheetahs to pandas is 1:3 -/
def valid_ratio (pop : ZooPopulation) : Prop :=
  3 * pop.cheetahs = pop.pandas

theorem panda_increase (old_pop new_pop : ZooPopulation) :
  valid_ratio old_pop →
  valid_ratio new_pop →
  new_pop.cheetahs = old_pop.cheetahs + 2 →
  new_pop.pandas = old_pop.pandas + 6 := by
  sorry

end NUMINAMATH_CALUDE_panda_increase_l2434_243420


namespace NUMINAMATH_CALUDE_pizza_distribution_l2434_243489

/-- Given the number of brothers, slices in small and large pizzas, and the number of each type of pizza ordered, 
    calculate the number of slices each brother can eat. -/
def slices_per_brother (num_brothers : ℕ) (slices_small : ℕ) (slices_large : ℕ) 
                       (num_small : ℕ) (num_large : ℕ) : ℕ :=
  (num_small * slices_small + num_large * slices_large) / num_brothers

/-- Theorem stating that under the given conditions, each brother can eat 12 slices of pizza. -/
theorem pizza_distribution :
  slices_per_brother 3 8 14 1 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_distribution_l2434_243489


namespace NUMINAMATH_CALUDE_max_salary_is_220000_l2434_243472

/-- Represents a basketball team with salary constraints -/
structure BasketballTeam where
  num_players : ℕ
  min_salary : ℕ
  salary_cap : ℕ

/-- Calculates the maximum possible salary for the highest-paid player -/
def max_highest_salary (team : BasketballTeam) : ℕ :=
  team.salary_cap - (team.num_players - 1) * team.min_salary

/-- Theorem stating the maximum possible salary for the highest-paid player -/
theorem max_salary_is_220000 (team : BasketballTeam) 
  (h1 : team.num_players = 15)
  (h2 : team.min_salary = 20000)
  (h3 : team.salary_cap = 500000) :
  max_highest_salary team = 220000 := by
  sorry

#eval max_highest_salary { num_players := 15, min_salary := 20000, salary_cap := 500000 }

end NUMINAMATH_CALUDE_max_salary_is_220000_l2434_243472


namespace NUMINAMATH_CALUDE_h_closed_form_l2434_243488

def h : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * h n + 2 * n

theorem h_closed_form (n : ℕ) : h n = 2^n + n^2 - n := by
  sorry

end NUMINAMATH_CALUDE_h_closed_form_l2434_243488


namespace NUMINAMATH_CALUDE_max_savings_bread_l2434_243416

/-- Represents the pricing structure for raisin bread -/
structure BreadPricing where
  single : Rat
  seven : Rat
  dozen : Rat

/-- Calculates the cost of buying bread given a pricing structure and quantities -/
def calculateCost (pricing : BreadPricing) (singles sevens dozens : Nat) : Rat :=
  pricing.single * singles + pricing.seven * sevens + pricing.dozen * dozens

/-- Theorem stating the maximum amount that can be saved -/
theorem max_savings_bread (pricing : BreadPricing) 
  (h1 : pricing.single = 3/10)
  (h2 : pricing.seven = 1)
  (h3 : pricing.dozen = 9/5)
  (budget : Rat)
  (h4 : budget = 10) :
  ∃ (singles sevens dozens : Nat),
    let total_pieces := singles + 7 * sevens + 12 * dozens
    let cost := calculateCost pricing singles sevens dozens
    total_pieces ≥ 60 ∧ cost ≤ budget ∧ budget - cost = 6/5 ∧
    ∀ (s s' d' : Nat), 
      let total_pieces' := s + 7 * s' + 12 * d'
      let cost' := calculateCost pricing s s' d'
      total_pieces' ≥ 60 ∧ cost' ≤ budget → budget - cost' ≤ 6/5 :=
by sorry

end NUMINAMATH_CALUDE_max_savings_bread_l2434_243416


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2434_243418

theorem polynomial_evaluation :
  let y : ℤ := -2
  (y^3 - y^2 + 2*y + 2 : ℤ) = -14 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2434_243418


namespace NUMINAMATH_CALUDE_sum_of_y_coordinates_l2434_243457

theorem sum_of_y_coordinates : ∀ y₁ y₂ : ℝ,
  (4 - (-1))^2 + (y₁ - 3)^2 = 8^2 →
  (4 - (-1))^2 + (y₂ - 3)^2 = 8^2 →
  y₁ + y₂ = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_y_coordinates_l2434_243457


namespace NUMINAMATH_CALUDE_triangle_angle_determinant_l2434_243423

theorem triangle_angle_determinant (θ φ ψ : Real) 
  (h : θ + φ + ψ = Real.pi) : 
  let M : Matrix (Fin 3) (Fin 3) Real := ![
    ![Real.cos θ, Real.sin θ, 1],
    ![Real.cos φ, Real.sin φ, 1],
    ![Real.cos ψ, Real.sin ψ, 1]
  ]
  Matrix.det M = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_determinant_l2434_243423


namespace NUMINAMATH_CALUDE_complex_equation_implies_sum_l2434_243409

def complex (a b : ℝ) : ℂ := Complex.mk a b

theorem complex_equation_implies_sum (a b : ℝ) :
  complex 9 3 * complex a b = complex 10 4 →
  a + b = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_sum_l2434_243409


namespace NUMINAMATH_CALUDE_tower_height_differences_l2434_243402

/-- Heights of towers in meters -/
def CN_Tower_height : ℝ := 553
def CN_Tower_Space_Needle_diff : ℝ := 369
def Eiffel_Tower_height : ℝ := 330
def Jeddah_Tower_predicted_height : ℝ := 1000

/-- Calculate the Space Needle height -/
def Space_Needle_height : ℝ := CN_Tower_height - CN_Tower_Space_Needle_diff

/-- Theorem stating the height differences -/
theorem tower_height_differences :
  (Eiffel_Tower_height - Space_Needle_height = 146) ∧
  (Jeddah_Tower_predicted_height - Eiffel_Tower_height = 670) :=
by sorry

end NUMINAMATH_CALUDE_tower_height_differences_l2434_243402


namespace NUMINAMATH_CALUDE_subtraction_proof_l2434_243446

theorem subtraction_proof :
  900000009000 - 123456789123 = 776543220777 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_proof_l2434_243446


namespace NUMINAMATH_CALUDE_hannah_mug_collection_l2434_243433

theorem hannah_mug_collection (total_mugs : ℕ) (num_colors : ℕ) (yellow_mugs : ℕ) :
  total_mugs = 40 →
  num_colors = 4 →
  yellow_mugs = 12 →
  let red_mugs := yellow_mugs / 2
  let blue_mugs := 3 * red_mugs
  total_mugs = blue_mugs + red_mugs + yellow_mugs + (total_mugs - (blue_mugs + red_mugs + yellow_mugs)) →
  (total_mugs - (blue_mugs + red_mugs + yellow_mugs)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_hannah_mug_collection_l2434_243433


namespace NUMINAMATH_CALUDE_bug_position_after_2010_jumps_l2434_243463

/-- Represents the points on the circle -/
inductive Point
| one
| two
| three
| four
| five

/-- Determines if a point is odd-numbered -/
def is_odd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true

/-- Calculates the next point based on the current point -/
def next_point (p : Point) : Point :=
  match p with
  | Point.one => Point.three
  | Point.two => Point.five
  | Point.three => Point.five
  | Point.four => Point.two
  | Point.five => Point.two

/-- Calculates the point after n jumps -/
def point_after_jumps (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | Nat.succ m => next_point (point_after_jumps start m)

theorem bug_position_after_2010_jumps :
  point_after_jumps Point.five 2010 = Point.two :=
sorry

end NUMINAMATH_CALUDE_bug_position_after_2010_jumps_l2434_243463


namespace NUMINAMATH_CALUDE_candy_count_is_twelve_l2434_243466

/-- The total number of candy pieces Wendy and her brother have -/
def total_candy (brother_candy : ℕ) (wendy_boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  brother_candy + wendy_boxes * pieces_per_box

/-- Theorem: The total number of candy pieces Wendy and her brother have is 12 -/
theorem candy_count_is_twelve :
  total_candy 6 2 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_is_twelve_l2434_243466


namespace NUMINAMATH_CALUDE_power_of_three_product_fourth_root_l2434_243458

theorem power_of_three_product_fourth_root (x : ℝ) : 
  (3^12 * 3^8)^(1/4) = 81 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_product_fourth_root_l2434_243458


namespace NUMINAMATH_CALUDE_smallest_double_when_two_moved_l2434_243468

def ends_with_two (n : ℕ) : Prop := n % 10 = 2

def move_two_to_front (n : ℕ) : ℕ :=
  let s := toString n
  let len := s.length
  if len > 1 then
    let front := s.dropRight 1
    let last := s.takeRight 1
    (last ++ front).toNat!
  else n

theorem smallest_double_when_two_moved : ∃ (n : ℕ),
  ends_with_two n ∧
  move_two_to_front n = 2 * n ∧
  ∀ (m : ℕ), m < n → ¬(ends_with_two m ∧ move_two_to_front m = 2 * m) ∧
  n = 105263157894736842 :=
sorry

end NUMINAMATH_CALUDE_smallest_double_when_two_moved_l2434_243468


namespace NUMINAMATH_CALUDE_max_ab_value_l2434_243450

/-- Two circles C₁ and C₂ -/
structure Circles where
  a : ℝ
  b : ℝ

/-- C₁: (x-a)² + (y+2)² = 4 -/
def C₁ (c : Circles) (x y : ℝ) : Prop :=
  (x - c.a)^2 + (y + 2)^2 = 4

/-- C₂: (x+b)² + (y+2)² = 1 -/
def C₂ (c : Circles) (x y : ℝ) : Prop :=
  (x + c.b)^2 + (y + 2)^2 = 1

/-- The circles are externally tangent -/
def externally_tangent (c : Circles) : Prop :=
  c.a + c.b = 3

/-- The maximum value of ab is 9/4 -/
theorem max_ab_value (c : Circles) (h : externally_tangent c) :
  c.a * c.b ≤ 9/4 ∧ ∃ (c' : Circles), externally_tangent c' ∧ c'.a * c'.b = 9/4 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l2434_243450


namespace NUMINAMATH_CALUDE_total_green_peaches_l2434_243414

/-- Represents a basket of peaches -/
structure Basket :=
  (red : ℕ)
  (green : ℕ)

/-- Proves that the total number of green peaches is 9 given the conditions -/
theorem total_green_peaches
  (b1 b2 b3 : Basket)
  (h1 : b1.red = 4)
  (h2 : b2.red = 4)
  (h3 : b3.red = 3)
  (h_total : b1.red + b1.green + b2.red + b2.green + b3.red + b3.green = 20) :
  b1.green + b2.green + b3.green = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_green_peaches_l2434_243414


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l2434_243412

-- Define a triangle type
structure Triangle where
  base : ℝ
  median1 : ℝ
  median2 : ℝ

-- Define the area function
def triangleArea (t : Triangle) : ℝ :=
  sorry  -- The actual calculation would go here

-- Theorem statement
theorem triangle_area_theorem (t : Triangle) 
  (h1 : t.base = 20)
  (h2 : t.median1 = 18)
  (h3 : t.median2 = 24) : 
  triangleArea t = 288 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_theorem_l2434_243412


namespace NUMINAMATH_CALUDE_sphere_radius_from_shadow_and_pole_l2434_243483

/-- The radius of a sphere given its shadow and a reference pole -/
theorem sphere_radius_from_shadow_and_pole 
  (sphere_shadow : ℝ) 
  (pole_height : ℝ) 
  (pole_shadow : ℝ) 
  (h_sphere_shadow : sphere_shadow = 15)
  (h_pole_height : pole_height = 1.5)
  (h_pole_shadow : pole_shadow = 3) :
  let tan_theta := pole_height / pole_shadow
  let radius := sphere_shadow * tan_theta
  radius = 7.5 := by sorry

end NUMINAMATH_CALUDE_sphere_radius_from_shadow_and_pole_l2434_243483


namespace NUMINAMATH_CALUDE_power_difference_lower_bound_l2434_243453

theorem power_difference_lower_bound 
  (m n : ℕ) 
  (h1 : m > 1) 
  (h2 : 2^(2*m + 1) - n^2 ≥ 0) : 
  2^(2*m + 1) - n^2 ≥ 7 := by
sorry

end NUMINAMATH_CALUDE_power_difference_lower_bound_l2434_243453


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2434_243486

/-- 
Given a quadratic equation x^2 - 4x - a = 0, prove that it has two distinct real roots
if and only if a > -4.
-/
theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x - a = 0 ∧ y^2 - 4*y - a = 0) ↔ a > -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2434_243486


namespace NUMINAMATH_CALUDE_root_range_l2434_243490

theorem root_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
   2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
   |x₁ - 2*n| = k * Real.sqrt x₁ ∧
   |x₂ - 2*n| = k * Real.sqrt x₂) →
  0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1) := by
sorry

end NUMINAMATH_CALUDE_root_range_l2434_243490


namespace NUMINAMATH_CALUDE_five_student_committees_from_eight_l2434_243470

theorem five_student_committees_from_eight (n : ℕ) (k : ℕ) : n = 8 → k = 5 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_student_committees_from_eight_l2434_243470


namespace NUMINAMATH_CALUDE_jerry_birthday_mean_l2434_243467

def aunt_gift : ℝ := 9
def uncle_gift : ℝ := 9
def friend_gift1 : ℝ := 22
def friend_gift2 : ℝ := 23
def friend_gift3 : ℝ := 22
def friend_gift4 : ℝ := 22
def sister_gift : ℝ := 7

def total_amount : ℝ := aunt_gift + uncle_gift + friend_gift1 + friend_gift2 + friend_gift3 + friend_gift4 + sister_gift
def number_of_gifts : ℕ := 7

theorem jerry_birthday_mean :
  total_amount / number_of_gifts = 16.29 := by sorry

end NUMINAMATH_CALUDE_jerry_birthday_mean_l2434_243467


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2434_243411

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 : ℂ) / (2 + Complex.I) ∧ Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2434_243411


namespace NUMINAMATH_CALUDE_sum_m_twice_n_l2434_243451

/-- The sum of m and twice n is equal to m + 2n -/
theorem sum_m_twice_n (m n : ℤ) : m + 2*n = m + 2*n := by sorry

end NUMINAMATH_CALUDE_sum_m_twice_n_l2434_243451


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2434_243431

def polynomial (x : ℝ) : ℝ :=
  3 * (2 * x^6 - x^5 + 4 * x^3 - 7) - 5 * (x^4 - 2 * x^3 + 3 * x^2 + 1) + 6 * (x^7 - 5)

theorem sum_of_coefficients :
  polynomial 1 = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2434_243431


namespace NUMINAMATH_CALUDE_random_selection_probability_l2434_243477

theorem random_selection_probability (m : ℝ) : 
  (m > 0) → 
  (2 * m) / (4 - (-2)) = 1 / 3 → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_random_selection_probability_l2434_243477


namespace NUMINAMATH_CALUDE_min_m_and_range_l2434_243439

noncomputable section

def f (x : ℝ) := (1 + x)^2 - 2 * Real.log (1 + x)

theorem min_m_and_range (x₀ : ℝ) (h : x₀ ∈ Set.Icc 0 1) :
  (∀ x ∈ Set.Icc 0 1, f x - (4 - 2 * Real.log 2) ≤ 0) ∧
  (f x₀ - 1 ≤ 0 → ∀ m : ℝ, f x₀ - m ≤ 0 → m ≥ 1) :=
by sorry

end

end NUMINAMATH_CALUDE_min_m_and_range_l2434_243439


namespace NUMINAMATH_CALUDE_rectangle_length_equals_nine_l2434_243407

-- Define the side length of the square
def square_side : ℝ := 6

-- Define the width of the rectangle
def rectangle_width : ℝ := 4

-- Define the area of the square
def square_area : ℝ := square_side * square_side

-- Define the area of the rectangle
def rectangle_area (length : ℝ) : ℝ := length * rectangle_width

-- Theorem statement
theorem rectangle_length_equals_nine :
  ∃ (length : ℝ), rectangle_area length = square_area ∧ length = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_equals_nine_l2434_243407


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l2434_243495

theorem matrix_inverse_proof : 
  let M : Matrix (Fin 3) (Fin 3) ℚ := ![![7/29, 5/29, 0], ![3/29, 2/29, 0], ![0, 0, 1]]
  let A : Matrix (Fin 3) (Fin 3) ℚ := ![![2, -5, 0], ![-3, 7, 0], ![0, 0, 1]]
  M * A = (1 : Matrix (Fin 3) (Fin 3) ℚ) := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l2434_243495


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2434_243408

theorem cyclic_sum_inequality (a b c : ℝ) (k : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 2) (habc : a * b * c = 1) :
  (a^k / (a + b)) + (b^k / (b + c)) + (c^k / (c + a)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2434_243408


namespace NUMINAMATH_CALUDE_ramanujan_identities_l2434_243406

/-- The function f₂ₙ as defined in Ramanujan's identities -/
def f (n : ℕ) (a b c d : ℝ) : ℝ :=
  (b + c + d)^(2*n) + (a + b + c)^(2*n) + (a - d)^(2*n) - 
  (a + c + d)^(2*n) - (a + b + d)^(2*n) - (b - c)^(2*n)

/-- Ramanujan's identities -/
theorem ramanujan_identities (a b c d : ℝ) (h : a * d = b * c) : 
  f 1 a b c d = 0 ∧ f 2 a b c d = 0 ∧ 64 * (f 3 a b c d) * (f 5 a b c d) = 45 * (f 4 a b c d)^2 := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_identities_l2434_243406


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l2434_243410

/-- The speed of the man in still water -/
def man_speed : ℝ := 10

/-- The speed of the stream -/
noncomputable def stream_speed : ℝ := sorry

/-- The downstream distance -/
def downstream_distance : ℝ := 28

/-- The upstream distance -/
def upstream_distance : ℝ := 12

/-- The time taken for both upstream and downstream journeys -/
def journey_time : ℝ := 2

theorem man_speed_in_still_water :
  (man_speed + stream_speed) * journey_time = downstream_distance ∧
  (man_speed - stream_speed) * journey_time = upstream_distance →
  man_speed = 10 := by
sorry

end NUMINAMATH_CALUDE_man_speed_in_still_water_l2434_243410


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2434_243475

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → a 1 = 1 → q = -2 →
  a 1 + |a 2| + a 3 + |a 4| = 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2434_243475


namespace NUMINAMATH_CALUDE_squares_property_l2434_243499

theorem squares_property (a b c : ℕ) 
  (h : a^2 + b^2 + c^2 = (a - b)^2 + (b - c)^2 + (c - a)^2) :
  ∃ (w x y z : ℕ), 
    a * b = w^2 ∧ 
    b * c = x^2 ∧ 
    c * a = y^2 ∧ 
    a * b + b * c + c * a = z^2 := by
  sorry

end NUMINAMATH_CALUDE_squares_property_l2434_243499


namespace NUMINAMATH_CALUDE_cubic_decreasing_implies_m_negative_l2434_243415

def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 - x

theorem cubic_decreasing_implies_m_negative :
  ∀ m : ℝ, (∀ x y : ℝ, x < y → f m x > f m y) → m < 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_decreasing_implies_m_negative_l2434_243415


namespace NUMINAMATH_CALUDE_remainder_theorem_l2434_243476

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 7 * k - 1) :
  (n^2 + 3*n + 4) % 7 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2434_243476


namespace NUMINAMATH_CALUDE_cylinder_height_l2434_243469

/-- For a cylinder with base radius 3, if its lateral surface area is 1/2 of its total surface area, then its height is 3 -/
theorem cylinder_height (h : ℝ) (h_pos : h > 0) : 
  (2 * π * 3 * h) = (1/2) * (2 * π * 3 * h + 2 * π * 3^2) → h = 3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_height_l2434_243469


namespace NUMINAMATH_CALUDE_smallest_n_for_perfect_square_sum_l2434_243461

theorem smallest_n_for_perfect_square_sum (n : ℕ) : n = 7 ↔ 
  (∀ k ≥ n, ∀ x ∈ Finset.range k, ∃ y ∈ Finset.range k, y ≠ x ∧ ∃ m : ℕ, x + y = m^2) ∧
  (∀ n' < n, ∃ k ≥ n', ∃ x ∈ Finset.range k, ∀ y ∈ Finset.range k, y = x ∨ ∀ m : ℕ, x + y ≠ m^2) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_perfect_square_sum_l2434_243461


namespace NUMINAMATH_CALUDE_matrix_power_difference_l2434_243404

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem matrix_power_difference : 
  B^10 - 3 * B^9 = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_difference_l2434_243404


namespace NUMINAMATH_CALUDE_power_calculation_l2434_243454

theorem power_calculation : 16^16 * 2^10 / 4^22 = 2^30 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2434_243454


namespace NUMINAMATH_CALUDE_card_73_is_8_l2434_243405

def card_sequence : List String := [
  "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K",
  "A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"
]

def cycle_length : Nat := card_sequence.length

theorem card_73_is_8 : 
  card_sequence[(73 - 1) % cycle_length] = "8" := by
  sorry

end NUMINAMATH_CALUDE_card_73_is_8_l2434_243405


namespace NUMINAMATH_CALUDE_relay_race_total_time_l2434_243485

/-- Represents the data for each athlete in the relay race -/
structure AthleteData where
  distance : ℕ
  time : ℕ

/-- Calculates the total time of the relay race given the data for each athlete -/
def relay_race_time (athletes : Vector AthleteData 8) : ℕ :=
  athletes.toList.map (·.time) |>.sum

theorem relay_race_total_time : ∃ (athletes : Vector AthleteData 8),
  (athletes.get 0).distance = 200 ∧ (athletes.get 0).time = 55 ∧
  (athletes.get 1).distance = 300 ∧ (athletes.get 1).time = (athletes.get 0).time + 10 ∧
  (athletes.get 2).distance = 250 ∧ (athletes.get 2).time = (athletes.get 1).time - 15 ∧
  (athletes.get 3).distance = 150 ∧ (athletes.get 3).time = (athletes.get 0).time - 25 ∧
  (athletes.get 4).distance = 400 ∧ (athletes.get 4).time = 80 ∧
  (athletes.get 5).distance = 350 ∧ (athletes.get 5).time = (athletes.get 4).time - 20 ∧
  (athletes.get 6).distance = 275 ∧ (athletes.get 6).time = 70 ∧
  (athletes.get 7).distance = 225 ∧ (athletes.get 7).time = (athletes.get 6).time - 5 ∧
  relay_race_time athletes = 475 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_total_time_l2434_243485


namespace NUMINAMATH_CALUDE_min_sum_inequality_l2434_243462

theorem min_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 3 / Real.rpow 162 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_inequality_l2434_243462


namespace NUMINAMATH_CALUDE_a_gt_b_gt_c_l2434_243445

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log Real.pi
noncomputable def c : ℝ := Real.log 0.9 / Real.log 2

theorem a_gt_b_gt_c : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_a_gt_b_gt_c_l2434_243445


namespace NUMINAMATH_CALUDE_smallest_c_value_l2434_243417

theorem smallest_c_value (c d : ℝ) (h_nonneg_c : c ≥ 0) (h_nonneg_d : d ≥ 0)
  (h_cos_eq : ∀ x : ℤ, Real.cos (c * x + d) = Real.cos (17 * x)) :
  c ≥ 17 ∧ ∃ (c' : ℝ), c' ≥ 0 ∧ c' < 17 → ¬(∀ x : ℤ, Real.cos (c' * x + d) = Real.cos (17 * x)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l2434_243417


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2434_243403

theorem arithmetic_mean_problem (a : ℝ) : 
  (1 + a) / 2 = 2 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2434_243403


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l2434_243465

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d < 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d
  h3 : a 2 * a 4 = 12
  h4 : a 2 + a 4 = 8

/-- The theorem stating the existence of a unique solution and sum of first 10 terms -/
theorem arithmetic_sequence_solution (seq : ArithmeticSequence) :
  ∃! (a₁ : ℝ), 
    (seq.a 1 = a₁) ∧
    (∃! (d : ℝ), d = seq.d) ∧
    (∃ (S₁₀ : ℝ), S₁₀ = (10 * seq.a 1) + (10 * 9 / 2 * seq.d)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l2434_243465


namespace NUMINAMATH_CALUDE_max_value_theorem_l2434_243422

theorem max_value_theorem (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 2) :
  (x^2 * y) / (x + y) + (y^2 * z) / (y + z) + (z^2 * x) / (z + x) ≤ 1 ∧
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧
    (x^2 * y) / (x + y) + (y^2 * z) / (y + z) + (z^2 * x) / (z + x) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2434_243422


namespace NUMINAMATH_CALUDE_optimal_point_distribution_l2434_243492

/-- A configuration of points in a space -/
structure PointConfiguration where
  total_points : ℕ
  num_groups : ℕ
  group_sizes : List ℕ
  no_collinear_triple : Prop
  distinct_group_sizes : Prop
  sum_of_sizes_equals_total : group_sizes.sum = total_points

/-- The number of triangles formed by choosing one point from each of any three different groups -/
def num_triangles (config : PointConfiguration) : ℕ :=
  sorry

/-- The optimal configuration maximizes the number of triangles -/
def is_optimal (config : PointConfiguration) : Prop :=
  ∀ other : PointConfiguration, num_triangles config ≥ num_triangles other

/-- The theorem stating the optimal configuration -/
theorem optimal_point_distribution :
  ∃ (optimal_config : PointConfiguration),
    optimal_config.total_points = 1989 ∧
    optimal_config.num_groups = 30 ∧
    optimal_config.group_sizes = [51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81] ∧
    is_optimal optimal_config :=
  sorry

end NUMINAMATH_CALUDE_optimal_point_distribution_l2434_243492


namespace NUMINAMATH_CALUDE_blue_candy_count_l2434_243419

theorem blue_candy_count (total red : ℕ) (h1 : total = 3409) (h2 : red = 145) :
  total - red = 3264 := by
  sorry

end NUMINAMATH_CALUDE_blue_candy_count_l2434_243419


namespace NUMINAMATH_CALUDE_cubic_inequality_l2434_243449

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 + 30*x > 0 ↔ (0 < x ∧ x < 5) ∨ (x > 6) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2434_243449


namespace NUMINAMATH_CALUDE_subset_arithmetic_result_l2434_243456

theorem subset_arithmetic_result (M : Finset ℕ) 
  (h_card : M.card = 13)
  (h_bounds : ∀ m ∈ M, 100 ≤ m ∧ m ≤ 999) :
  ∃ S : Finset ℕ, S ⊆ M ∧ 
  ∃ a b c d e f : ℕ, 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧
    3 < (b : ℚ) / a + (d : ℚ) / c + (f : ℚ) / e ∧
    (b : ℚ) / a + (d : ℚ) / c + (f : ℚ) / e < 4 :=
sorry

end NUMINAMATH_CALUDE_subset_arithmetic_result_l2434_243456


namespace NUMINAMATH_CALUDE_point_M_coordinates_l2434_243426

-- Define point M
def M (a : ℝ) : ℝ × ℝ := (a + 3, a + 1)

-- Define the condition for a point to be on the x-axis
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

-- Theorem statement
theorem point_M_coordinates :
  ∀ a : ℝ, on_x_axis (M a) → M a = (2, 0) :=
by sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l2434_243426


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2434_243487

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_a5 (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 3 * a 11 = 16) :
  a 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2434_243487


namespace NUMINAMATH_CALUDE_number_equation_l2434_243459

theorem number_equation (x : ℤ) : 45 - (x - (37 - (15 - 20))) = 59 ↔ x = 28 := by sorry

end NUMINAMATH_CALUDE_number_equation_l2434_243459


namespace NUMINAMATH_CALUDE_spinner_probability_l2434_243425

-- Define the spinner sections
def spinner_sections : List ℕ := [3, 6, 1, 4, 8, 10, 2, 7]

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := sorry

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Prop := sorry

-- Define a function to count elements satisfying a condition
def count_if (l : List ℕ) (f : ℕ → Prop) : ℕ := sorry

-- Theorem statement
theorem spinner_probability :
  let favorable_outcomes := count_if spinner_sections (λ n => is_prime n ∨ is_odd n)
  let total_outcomes := spinner_sections.length
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_spinner_probability_l2434_243425


namespace NUMINAMATH_CALUDE_ryan_lost_leaves_l2434_243478

theorem ryan_lost_leaves (initial_leaves : ℕ) (broken_leaves : ℕ) (remaining_leaves : ℕ) : 
  initial_leaves = 89 → broken_leaves = 43 → remaining_leaves = 22 → 
  initial_leaves - (initial_leaves - remaining_leaves - broken_leaves) - broken_leaves = remaining_leaves :=
by
  sorry

end NUMINAMATH_CALUDE_ryan_lost_leaves_l2434_243478


namespace NUMINAMATH_CALUDE_sqrt_three_multiplication_l2434_243498

theorem sqrt_three_multiplication : Real.sqrt 3 * (2 * Real.sqrt 3 - 2) = 6 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_multiplication_l2434_243498


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_for_given_problem_l2434_243437

/-- Calculates the interval for systematic sampling -/
def systematicSamplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

theorem systematic_sampling_interval_for_given_problem :
  let populationSize : ℕ := 1000
  let sampleSize : ℕ := 40
  systematicSamplingInterval populationSize sampleSize = 25 := by
  sorry

#eval systematicSamplingInterval 1000 40

end NUMINAMATH_CALUDE_systematic_sampling_interval_for_given_problem_l2434_243437


namespace NUMINAMATH_CALUDE_pastor_prayer_difference_l2434_243435

/-- Pastor Paul's daily prayer count on weekdays -/
def paul_weekday : ℕ := 20

/-- Pastor Paul's Sunday prayer count -/
def paul_sunday : ℕ := 2 * paul_weekday

/-- Pastor Bruce's weekday prayer count -/
def bruce_weekday : ℕ := paul_weekday / 2

/-- Pastor Bruce's Sunday prayer count -/
def bruce_sunday : ℕ := 2 * paul_sunday

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Number of weekdays in a week -/
def weekdays : ℕ := 6

theorem pastor_prayer_difference :
  paul_weekday * weekdays + paul_sunday - (bruce_weekday * weekdays + bruce_sunday) = 20 := by
  sorry

end NUMINAMATH_CALUDE_pastor_prayer_difference_l2434_243435


namespace NUMINAMATH_CALUDE_a_in_M_sufficient_not_necessary_for_a_in_N_l2434_243479

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {-1, 0, 1, 2}

theorem a_in_M_sufficient_not_necessary_for_a_in_N :
  (∀ a, a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by sorry

end NUMINAMATH_CALUDE_a_in_M_sufficient_not_necessary_for_a_in_N_l2434_243479
