import Mathlib

namespace NUMINAMATH_CALUDE_N_q_odd_iff_prime_power_l2095_209504

/-- The number of integers a such that 0 < a < q/4 and gcd(a,q) = 1 -/
def N_q (q : ℕ) : ℕ :=
  (Finset.filter (fun a => a > 0 ∧ a < q / 4 ∧ Nat.gcd a q = 1) (Finset.range q)).card

/-- A prime p is congruent to 5 or 7 modulo 8 -/
def is_prime_5_or_7_mod_8 (p : ℕ) : Prop :=
  Nat.Prime p ∧ (p % 8 = 5 ∨ p % 8 = 7)

theorem N_q_odd_iff_prime_power (q : ℕ) (h_odd : Odd q) :
  Odd (N_q q) ↔ ∃ (p k : ℕ), q = p^k ∧ k > 0 ∧ is_prime_5_or_7_mod_8 p :=
sorry

end NUMINAMATH_CALUDE_N_q_odd_iff_prime_power_l2095_209504


namespace NUMINAMATH_CALUDE_line_equations_l2095_209522

-- Define a line passing through (-1, 3) with equal absolute intercepts
def line_through_point_with_equal_intercepts (a b c : ℝ) : Prop :=
  -- The line passes through (-1, 3)
  a * (-1) + b * 3 + c = 0 ∧
  -- The line has intercepts of equal absolute values on x and y axes
  ∃ k : ℝ, k ≠ 0 ∧ (a * k + c = 0 ∨ b * k + c = 0) ∧ (a * (-k) + c = 0 ∨ b * (-k) + c = 0)

-- Theorem stating the possible equations of the line
theorem line_equations :
  ∃ (a b c : ℝ),
    line_through_point_with_equal_intercepts a b c ∧
    ((a = 3 ∧ b = 1 ∧ c = 0) ∨
     (a = 1 ∧ b = -1 ∧ c = -4) ∨
     (a = 1 ∧ b = 1 ∧ c = -2)) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_l2095_209522


namespace NUMINAMATH_CALUDE_total_faces_painted_is_48_l2095_209508

/-- The number of outer faces of a cuboid -/
def cuboid_faces : ℕ := 6

/-- The number of cuboids -/
def num_cuboids : ℕ := 8

/-- The total number of faces painted -/
def total_faces_painted : ℕ := cuboid_faces * num_cuboids

/-- Theorem: The total number of faces painted on 8 identical cuboids is 48 -/
theorem total_faces_painted_is_48 : total_faces_painted = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_faces_painted_is_48_l2095_209508


namespace NUMINAMATH_CALUDE_collinear_probability_value_l2095_209556

/-- A 5x5 grid of dots -/
def Grid := Fin 5 × Fin 5

/-- The total number of dots in the grid -/
def total_dots : ℕ := 25

/-- The number of sets of 5 collinear dots in the grid -/
def collinear_sets : ℕ := 12

/-- The number of ways to choose 5 dots from the grid -/
def total_choices : ℕ := Nat.choose total_dots 5

/-- The probability of selecting 5 collinear dots from the grid -/
def collinear_probability : ℚ := collinear_sets / total_choices

theorem collinear_probability_value :
  collinear_probability = 12 / 53130 :=
sorry

end NUMINAMATH_CALUDE_collinear_probability_value_l2095_209556


namespace NUMINAMATH_CALUDE_total_spent_is_124_l2095_209576

/-- The total amount spent on entertainment and additional expenses -/
def total_spent (computer_game_cost movie_ticket_cost num_tickets snack_cost transportation_cost num_trips : ℕ) : ℕ :=
  computer_game_cost + 
  movie_ticket_cost * num_tickets + 
  snack_cost + 
  transportation_cost * num_trips

/-- Theorem stating that the total amount spent is $124 given the specific costs -/
theorem total_spent_is_124 :
  total_spent 66 12 3 7 5 3 = 124 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_124_l2095_209576


namespace NUMINAMATH_CALUDE_rational_absolute_difference_sum_l2095_209515

theorem rational_absolute_difference_sum (a b : ℚ) : 
  |a - b| = a + b → a ≥ 0 ∧ b ≥ 0 := by sorry

end NUMINAMATH_CALUDE_rational_absolute_difference_sum_l2095_209515


namespace NUMINAMATH_CALUDE_missing_angle_measure_l2095_209599

/-- A convex polygon with n sides --/
structure ConvexPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- The sum of interior angles of a convex polygon --/
def interior_angle_sum (p : ConvexPolygon) : ℝ :=
  (p.n - 2) * 180

/-- The theorem to prove --/
theorem missing_angle_measure (p : ConvexPolygon) 
  (sum_without_one : ℝ) 
  (h_sum : sum_without_one = 3025) :
  interior_angle_sum p - sum_without_one = 35 := by
  sorry

end NUMINAMATH_CALUDE_missing_angle_measure_l2095_209599


namespace NUMINAMATH_CALUDE_valid_solutions_are_only_solutions_l2095_209525

/-- A structure representing a solution to the system of equations -/
structure Solution :=
  (x y z t : ℕ)

/-- The set of all valid solutions -/
def valid_solutions : Set Solution :=
  { ⟨1,1,2,3⟩, ⟨3,2,1,1⟩, ⟨4,1,3,1⟩, ⟨1,3,4,1⟩ }

/-- Predicate to check if a solution satisfies the equations -/
def satisfies_equations (s : Solution) : Prop :=
  ∃ a b : ℕ,
    s.x^2 + s.y^2 = a ∧
    s.z^2 + s.t^2 = b ∧
    (s.x^2 + s.t^2) * (s.z^2 + s.y^2) = 50

/-- Theorem stating that the valid solutions are the only ones satisfying the equations -/
theorem valid_solutions_are_only_solutions :
  ∀ s : Solution, satisfies_equations s ↔ s ∈ valid_solutions :=
sorry

end NUMINAMATH_CALUDE_valid_solutions_are_only_solutions_l2095_209525


namespace NUMINAMATH_CALUDE_joe_savings_l2095_209514

def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000
def money_left : ℕ := 1000

theorem joe_savings : 
  flight_cost + hotel_cost + food_cost + money_left = 6000 := by
  sorry

end NUMINAMATH_CALUDE_joe_savings_l2095_209514


namespace NUMINAMATH_CALUDE_inequality_proof_l2095_209590

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) + Real.sqrt (c^2 - c*a + a^2) + 9 * (a*b*c)^(1/3) ≤ 4*(a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2095_209590


namespace NUMINAMATH_CALUDE_min_fence_length_l2095_209526

theorem min_fence_length (area : ℝ) (h : area = 64) :
  ∃ (length width : ℝ), length > 0 ∧ width > 0 ∧
  length * width = area ∧
  ∀ (l w : ℝ), l > 0 → w > 0 → l * w = area →
  2 * (length + width) ≤ 2 * (l + w) ∧
  2 * (length + width) = 32 :=
sorry

end NUMINAMATH_CALUDE_min_fence_length_l2095_209526


namespace NUMINAMATH_CALUDE_jeans_pricing_l2095_209562

theorem jeans_pricing (C : ℝ) (R : ℝ) :
  C > 0 →
  R > 0 →
  1.96 * C = 1.4 * R →
  (R - C) / C * 100 = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_jeans_pricing_l2095_209562


namespace NUMINAMATH_CALUDE_bennys_savings_l2095_209505

/-- Proves that Benny's savings in January (and February) must be $19 given the conditions -/
theorem bennys_savings (x : ℕ) : 2 * x + 8 = 46 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_bennys_savings_l2095_209505


namespace NUMINAMATH_CALUDE_ellipse_max_min_sum_absolute_values_l2095_209573

theorem ellipse_max_min_sum_absolute_values :
  ∀ x y : ℝ, x^2/4 + y^2/9 = 1 →
  (∃ a b : ℝ, a^2/4 + b^2/9 = 1 ∧ |a| + |b| = 3) ∧
  (∃ c d : ℝ, c^2/4 + d^2/9 = 1 ∧ |c| + |d| = 2) ∧
  (∀ z w : ℝ, z^2/4 + w^2/9 = 1 → |z| + |w| ≤ 3 ∧ |z| + |w| ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_max_min_sum_absolute_values_l2095_209573


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l2095_209594

theorem quadratic_roots_ratio (q : ℝ) : 
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ 
   r + s = -8 ∧ r * s = q ∧ 
   ∀ x : ℝ, x^2 + 8*x + q = 0 ↔ (x = r ∨ x = s)) → 
  q = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l2095_209594


namespace NUMINAMATH_CALUDE_min_rectangles_cover_square_l2095_209533

/-- The smallest number of 2-by-3 non-overlapping rectangles needed to cover a 12-by-12 square exactly -/
def min_rectangles : ℕ := 24

/-- The side length of the square -/
def square_side : ℕ := 12

/-- The width of the rectangle -/
def rect_width : ℕ := 2

/-- The height of the rectangle -/
def rect_height : ℕ := 3

/-- The area of the square -/
def square_area : ℕ := square_side ^ 2

/-- The area of a single rectangle -/
def rect_area : ℕ := rect_width * rect_height

theorem min_rectangles_cover_square :
  min_rectangles * rect_area = square_area ∧
  ∃ (rows columns : ℕ),
    rows * columns = min_rectangles ∧
    rows * rect_height = square_side ∧
    columns * rect_width = square_side :=
by sorry

end NUMINAMATH_CALUDE_min_rectangles_cover_square_l2095_209533


namespace NUMINAMATH_CALUDE_women_average_age_l2095_209501

/-- The average age of two women given specific conditions about a group of men --/
theorem women_average_age (n : ℕ) (A : ℝ) (age1 age2 : ℕ) (increase : ℝ) : 
  n = 10 ∧ age1 = 18 ∧ age2 = 22 ∧ increase = 6 →
  (n : ℝ) * (A + increase) - (n : ℝ) * A = 
    (((n : ℝ) * (A + increase) - (n : ℝ) * A + age1 + age2) / 2) * 2 - (age1 + age2) →
  ((n : ℝ) * (A + increase) - (n : ℝ) * A + age1 + age2) / 2 = 50 :=
by sorry

end NUMINAMATH_CALUDE_women_average_age_l2095_209501


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_range_l2095_209586

-- Define the line equation
def line (k x y : ℝ) : Prop := 2 * k * x - y + 1 = 0

-- Define the ellipse equation
def ellipse (x y m : ℝ) : Prop := (x^2 / 9) + (y^2 / m) = 1

-- State the theorem
theorem ellipse_line_intersection_range (k : ℝ) :
  (∀ m : ℝ, (∀ x y : ℝ, line k x y → ellipse x y m → (∃ x' y' : ℝ, line k x' y' ∧ ellipse x' y' m))) →
  (∃ S : Set ℝ, S = {m : ℝ | m ∈ Set.Icc 1 9 ∪ Set.Ioi 9}) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_range_l2095_209586


namespace NUMINAMATH_CALUDE_total_allowance_is_8330_l2095_209561

/-- Calculates the total weekly allowance for a group of students --/
def total_weekly_allowance (total_students : ℕ) 
  (percent1 percent2 percent3 : ℚ)
  (allowance1 allowance2 allowance3 allowance4 : ℚ) : ℚ :=
  let remaining_percent := 1 - (percent1 + percent2 + percent3)
  let daily_total := 
    (total_students : ℚ) * (
      percent1 * allowance1 +
      percent2 * allowance2 +
      percent3 * allowance3 +
      remaining_percent * allowance4
    )
  7 * daily_total

/-- Theorem stating that the total weekly allowance for 200 students
    with the given percentages and daily allowances is $8330 --/
theorem total_allowance_is_8330 :
  total_weekly_allowance 200 (45/100) (30/100) (15/100) 6 4 7 10 = 8330 := by
  sorry

end NUMINAMATH_CALUDE_total_allowance_is_8330_l2095_209561


namespace NUMINAMATH_CALUDE_power_of_three_equals_square_minus_sixteen_l2095_209538

theorem power_of_three_equals_square_minus_sixteen (a n : ℕ+) :
  (3 : ℕ) ^ (n : ℕ) = (a : ℕ) ^ 2 - 16 ↔ a = 5 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equals_square_minus_sixteen_l2095_209538


namespace NUMINAMATH_CALUDE_alpha_beta_range_l2095_209595

theorem alpha_beta_range (α β : ℝ) (h1 : 1 < α) (h2 : α < 3) (h3 : -4 < β) (h4 : β < 2) :
  -12 < α * (-abs β) ∧ α * (-abs β) < -2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_range_l2095_209595


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2095_209530

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 →
  (-2 * x^2 + 5 * x - 6) / (x^3 + x) = -6 / x + (4 * x + 5) / (x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2095_209530


namespace NUMINAMATH_CALUDE_extra_hours_worked_l2095_209550

def hours_week1 : ℕ := 35
def hours_week2 : ℕ := 35
def hours_week3 : ℕ := 48
def hours_week4 : ℕ := 48

theorem extra_hours_worked : 
  (hours_week3 + hours_week4) - (hours_week1 + hours_week2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_extra_hours_worked_l2095_209550


namespace NUMINAMATH_CALUDE_total_digits_of_powers_l2095_209587

theorem total_digits_of_powers : ∃ m n : ℕ,
  (10^(m-1) < 2^2019 ∧ 2^2019 < 10^m) ∧
  (10^(n-1) < 5^2019 ∧ 5^2019 < 10^n) ∧
  m + n = 2020 :=
by sorry

end NUMINAMATH_CALUDE_total_digits_of_powers_l2095_209587


namespace NUMINAMATH_CALUDE_area_ratio_second_third_neighbor_octagons_l2095_209502

/-- A regular octagon -/
structure RegularOctagon where
  -- Add necessary fields

/-- The octagon formed by connecting second neighboring vertices -/
def secondNeighborOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The octagon formed by connecting third neighboring vertices -/
def thirdNeighborOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- The theorem stating the ratio of areas -/
theorem area_ratio_second_third_neighbor_octagons (o : RegularOctagon) :
  area (secondNeighborOctagon o) / area (thirdNeighborOctagon o) = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_second_third_neighbor_octagons_l2095_209502


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2095_209511

theorem subtraction_preserves_inequality (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2095_209511


namespace NUMINAMATH_CALUDE_area_triangle_PQR_l2095_209598

/-- Square pyramid with given dimensions and points --/
structure SquarePyramid where
  baseSide : ℝ
  altitude : ℝ
  P : ℝ  -- Distance from W to P along WO
  Q : ℝ  -- Distance from Y to Q along YO
  R : ℝ  -- Distance from X to R along XO

/-- Theorem stating the area of triangle PQR in the given square pyramid --/
theorem area_triangle_PQR (pyramid : SquarePyramid)
  (h1 : pyramid.baseSide = 4)
  (h2 : pyramid.altitude = 8)
  (h3 : pyramid.P = 1/4 * (pyramid.baseSide * Real.sqrt 2 / 2))
  (h4 : pyramid.Q = 1/2 * (pyramid.baseSide * Real.sqrt 2 / 2))
  (h5 : pyramid.R = 3/4 * (pyramid.baseSide * Real.sqrt 2 / 2)) :
  let WO := Real.sqrt ((pyramid.baseSide * Real.sqrt 2 / 2)^2 + pyramid.altitude^2)
  let PQ := pyramid.Q - pyramid.P
  let RQ := pyramid.R - pyramid.Q
  1/2 * PQ * RQ = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_PQR_l2095_209598


namespace NUMINAMATH_CALUDE_parabola_vertex_l2095_209593

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := y^2 - 8*x + 6*y + 17 = 0

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, -3)

/-- Theorem: The vertex of the parabola y^2 - 8x + 6y + 17 = 0 is at the point (1, -3) -/
theorem parabola_vertex : 
  ∀ x y : ℝ, parabola_eq x y → (x, y) = vertex ∨ ∃ t : ℝ, parabola_eq (x + t) y :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2095_209593


namespace NUMINAMATH_CALUDE_min_avg_of_two_l2095_209585

theorem min_avg_of_two (a b c d : ℕ+) : 
  (a + b + c + d : ℝ) / 4 = 50 → 
  max c d ≤ 130 →
  (a + b : ℝ) / 2 ≥ 35 :=
by sorry

end NUMINAMATH_CALUDE_min_avg_of_two_l2095_209585


namespace NUMINAMATH_CALUDE_incorrect_propositions_are_one_and_three_l2095_209554

-- Define a proposition as a structure with an id and a correctness value
structure Proposition :=
  (id : Nat)
  (isCorrect : Bool)

-- Define our set of propositions
def propositions : List Proposition := [
  ⟨1, false⟩,  -- Three points determine a plane
  ⟨2, true⟩,   -- A rectangle is a plane figure
  ⟨3, false⟩,  -- Three lines intersecting in pairs determine a plane
  ⟨4, true⟩    -- Two intersecting planes divide the space into four regions
]

-- Define a function to get incorrect propositions
def getIncorrectPropositions (props : List Proposition) : List Nat :=
  (props.filter (λ p => !p.isCorrect)).map Proposition.id

-- Theorem statement
theorem incorrect_propositions_are_one_and_three :
  getIncorrectPropositions propositions = [1, 3] := by
  sorry

end NUMINAMATH_CALUDE_incorrect_propositions_are_one_and_three_l2095_209554


namespace NUMINAMATH_CALUDE_min_moves_to_single_color_l2095_209589

/-- Represents a move on the chessboard -/
structure Move where
  m : Nat
  n : Nat

/-- Represents the chessboard -/
def Chessboard := Fin 7 → Fin 7 → Bool

/-- Applies a move to the chessboard -/
def applyMove (board : Chessboard) (move : Move) : Chessboard :=
  sorry

/-- Checks if the board is of a single color -/
def isSingleColor (board : Chessboard) : Bool :=
  sorry

/-- Initial chessboard with alternating colors -/
def initialBoard : Chessboard :=
  sorry

/-- Theorem stating the minimum number of moves required -/
theorem min_moves_to_single_color :
  ∃ (moves : List Move),
    moves.length = 6 ∧
    isSingleColor (moves.foldl applyMove initialBoard) ∧
    ∀ (otherMoves : List Move),
      isSingleColor (otherMoves.foldl applyMove initialBoard) →
      otherMoves.length ≥ 6 :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_single_color_l2095_209589


namespace NUMINAMATH_CALUDE_danny_apples_danny_bought_73_apples_l2095_209557

def pinky_apples : ℕ := 36
def total_apples : ℕ := 109

theorem danny_apples : ℕ → Prop :=
  fun x => x = total_apples - pinky_apples

theorem danny_bought_73_apples : danny_apples 73 := by
  sorry

end NUMINAMATH_CALUDE_danny_apples_danny_bought_73_apples_l2095_209557


namespace NUMINAMATH_CALUDE_coefficient_of_x_in_expansion_l2095_209539

-- Define the binomial expansion function
def binomialCoefficient (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the coefficient of x
def coefficientOfX (a b : ℤ) (n : ℕ) : ℤ :=
  binomialCoefficient n 2 * (b ^ 2)

-- Theorem statement
theorem coefficient_of_x_in_expansion :
  coefficientOfX 1 (-2) 5 = 40 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_in_expansion_l2095_209539


namespace NUMINAMATH_CALUDE_profit_calculation_l2095_209584

/-- Profit calculation for a company --/
theorem profit_calculation (total_profit second_half_profit first_half_profit : ℚ) :
  total_profit = 3635000 →
  first_half_profit = second_half_profit + 2750000 →
  total_profit = first_half_profit + second_half_profit →
  second_half_profit = 442500 := by
sorry

end NUMINAMATH_CALUDE_profit_calculation_l2095_209584


namespace NUMINAMATH_CALUDE_train_length_l2095_209577

/-- Calculate the length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (time : ℝ) : 
  speed = 72 → platform_length = 230 → time = 26 → 
  (speed * 1000 / 3600) * time - platform_length = 290 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2095_209577


namespace NUMINAMATH_CALUDE_triangle_max_area_l2095_209529

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C,
    if (a-b+c)/c = b/(a+b-c) and a = 2, then the maximum area of triangle ABC is √3. -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  (a - b + c) / c = b / (a + b - c) →
  a = 2 →
  ∃ (S : ℝ), S ≤ Real.sqrt 3 ∧ 
    (∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ S) ∧
    (∃ (b' c' : ℝ), (1/2) * b' * c' * Real.sin A = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2095_209529


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2095_209537

/-- The line equation passes through a fixed point for all values of k -/
theorem fixed_point_on_line (k : ℝ) : 
  (2 * k - 1) * 2 - (k - 2) * 3 - (k + 4) = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2095_209537


namespace NUMINAMATH_CALUDE_citrus_yield_probability_l2095_209536

/-- Represents the yield recovery rates in the first year -/
def first_year_rates : List ℝ := [1.0, 0.9, 0.8]

/-- Represents the probabilities of yield recovery rates in the first year -/
def first_year_probs : List ℝ := [0.2, 0.4, 0.4]

/-- Represents the growth rates in the second year -/
def second_year_rates : List ℝ := [1.5, 1.25, 1.0]

/-- Represents the probabilities of growth rates in the second year -/
def second_year_probs : List ℝ := [0.3, 0.3, 0.4]

/-- Calculates the probability of reaching exactly the pre-disaster yield after two years -/
def probability_pre_disaster_yield (f_rates : List ℝ) (f_probs : List ℝ) (s_rates : List ℝ) (s_probs : List ℝ) : ℝ :=
  sorry

theorem citrus_yield_probability :
  probability_pre_disaster_yield first_year_rates first_year_probs second_year_rates second_year_probs = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_citrus_yield_probability_l2095_209536


namespace NUMINAMATH_CALUDE_factors_of_2_pow_96_minus_1_l2095_209583

theorem factors_of_2_pow_96_minus_1 :
  ∃ (a b : ℕ), 60 < a ∧ a < 70 ∧ 60 < b ∧ b < 70 ∧
  a ≠ b ∧
  (2^96 - 1) % a = 0 ∧ (2^96 - 1) % b = 0 ∧
  (∀ c : ℕ, 60 < c → c < 70 → c ≠ a → c ≠ b → (2^96 - 1) % c ≠ 0) ∧
  a = 63 ∧ b = 65 :=
by sorry

end NUMINAMATH_CALUDE_factors_of_2_pow_96_minus_1_l2095_209583


namespace NUMINAMATH_CALUDE_cube_surface_area_l2095_209560

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1000 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 600 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2095_209560


namespace NUMINAMATH_CALUDE_amy_avocado_business_l2095_209592

/-- Proves that n = 50 given the conditions of Amy's avocado business --/
theorem amy_avocado_business (n : ℕ) (hn : n > 0) : 
  (15 * n : ℕ) % 3 = 0 ∧ 
  (15 * n : ℕ) % 5 = 0 ∧ 
  4 * ((15 * n : ℕ) / 5) - 2 * ((15 * n : ℕ) / 3) = 100 → 
  n = 50 := by
  sorry

end NUMINAMATH_CALUDE_amy_avocado_business_l2095_209592


namespace NUMINAMATH_CALUDE_annas_age_at_marriage_l2095_209596

/-- Proves Anna's age at marriage given the conditions of the problem -/
theorem annas_age_at_marriage
  (josh_age_at_marriage : ℕ)
  (years_of_marriage : ℕ)
  (combined_age_factor : ℕ)
  (h1 : josh_age_at_marriage = 22)
  (h2 : years_of_marriage = 30)
  (h3 : combined_age_factor = 5)
  (h4 : josh_age_at_marriage + years_of_marriage + (josh_age_at_marriage + years_of_marriage + anna_age_at_marriage) = combined_age_factor * josh_age_at_marriage) :
  anna_age_at_marriage = 28 :=
by
  sorry

#check annas_age_at_marriage

end NUMINAMATH_CALUDE_annas_age_at_marriage_l2095_209596


namespace NUMINAMATH_CALUDE_max_at_two_l2095_209531

/-- The function f(x) defined as x(x-c)² --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) with respect to x --/
def f_derivative (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

theorem max_at_two (c : ℝ) :
  (∀ x : ℝ, f c x ≤ f c 2) ↔ c = 6 := by sorry

end NUMINAMATH_CALUDE_max_at_two_l2095_209531


namespace NUMINAMATH_CALUDE_ned_short_sleeve_shirts_l2095_209549

/-- The number of short sleeve shirts Ned had to wash -/
def short_sleeve_shirts : ℕ := sorry

/-- The number of long sleeve shirts Ned had to wash -/
def long_sleeve_shirts : ℕ := 21

/-- The number of shirts Ned washed before school started -/
def washed_shirts : ℕ := 29

/-- The number of shirts Ned did not wash -/
def unwashed_shirts : ℕ := 1

/-- The total number of shirts Ned had to wash -/
def total_shirts : ℕ := washed_shirts + unwashed_shirts

theorem ned_short_sleeve_shirts :
  short_sleeve_shirts = total_shirts - long_sleeve_shirts :=
by sorry

end NUMINAMATH_CALUDE_ned_short_sleeve_shirts_l2095_209549


namespace NUMINAMATH_CALUDE_pizzas_served_today_l2095_209500

theorem pizzas_served_today (lunch_pizzas dinner_pizzas : ℕ) 
  (h1 : lunch_pizzas = 9) 
  (h2 : dinner_pizzas = 6) : 
  lunch_pizzas + dinner_pizzas = 15 := by
  sorry

end NUMINAMATH_CALUDE_pizzas_served_today_l2095_209500


namespace NUMINAMATH_CALUDE_seminar_handshakes_l2095_209570

/-- The number of people attending the seminar -/
def n : ℕ := 12

/-- The number of pairs of people who don't shake hands -/
def excluded_pairs : ℕ := 1

/-- The total number of handshakes in the seminar -/
def total_handshakes : ℕ := n.choose 2 - excluded_pairs

/-- Theorem stating the total number of handshakes in the seminar -/
theorem seminar_handshakes : total_handshakes = 65 := by
  sorry

end NUMINAMATH_CALUDE_seminar_handshakes_l2095_209570


namespace NUMINAMATH_CALUDE_rick_ironing_rate_l2095_209540

/-- The number of dress shirts Rick can iron in an hour -/
def shirts_per_hour : ℕ := sorry

/-- The number of dress pants Rick can iron in an hour -/
def pants_per_hour : ℕ := 3

/-- The number of hours Rick spent ironing dress shirts -/
def hours_ironing_shirts : ℕ := 3

/-- The number of hours Rick spent ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- The total number of pieces of clothing Rick ironed -/
def total_pieces : ℕ := 27

theorem rick_ironing_rate : shirts_per_hour = 4 := by
  sorry

end NUMINAMATH_CALUDE_rick_ironing_rate_l2095_209540


namespace NUMINAMATH_CALUDE_expression_evaluation_l2095_209546

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := Real.sqrt 2
  (x + 2 * y)^2 - x * (x + 4 * y) + (1 - y) * (1 + y) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2095_209546


namespace NUMINAMATH_CALUDE_point_coordinates_l2095_209520

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates :
  ∀ (P : Point),
    SecondQuadrant P →
    DistToXAxis P = 3 →
    DistToYAxis P = 4 →
    P.x = -4 ∧ P.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2095_209520


namespace NUMINAMATH_CALUDE_max_value_expression_l2095_209553

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 4 - Real.sqrt (x^4 + 16)) / x ≤ 2 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2095_209553


namespace NUMINAMATH_CALUDE_proportion_theorem_l2095_209545

theorem proportion_theorem (y : ℝ) : 
  (0.75 : ℝ) / 0.9 = y / 6 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_theorem_l2095_209545


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_nonnegative_solutions_l2095_209510

theorem quadratic_two_distinct_nonnegative_solutions (a : ℝ) :
  (6 - 3 * a > 0) →
  (a > 0) →
  (3 * a^2 + a - 2 ≥ 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ≥ 0 ∧ x₂ ≥ 0 ∧
    3 * x₁^2 - 3 * a * x₁ + a = 0 ∧
    3 * x₂^2 - 3 * a * x₂ + a = 0) ↔
  (2/3 ≤ a ∧ a < 5/3) ∨ (5/3 < a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_nonnegative_solutions_l2095_209510


namespace NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l2095_209512

/-- Given a real number a, we define a function f on (-∞, a] such that f(x) = x + 1.
    We assume that for all x and y in (-∞, a], f(x+y) ≤ 2f(x) - 3f(y).
    This theorem states that under these conditions, a must be less than or equal to -2. -/
theorem function_inequality_implies_upper_bound (a : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, x ≤ a → f x = x + 1)
  (h2 : ∀ x y, x ≤ a → y ≤ a → f (x + y) ≤ 2 * f x - 3 * f y) :
  a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l2095_209512


namespace NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l2095_209572

theorem modulo_equivalence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 10403 [ZMOD 15] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l2095_209572


namespace NUMINAMATH_CALUDE_equation_solutions_l2095_209521

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => 1/((x - 2)*(x - 3)) + 1/((x - 3)*(x - 4)) + 1/((x - 4)*(x - 5))
  ∀ x : ℝ, f x = 1/8 ↔ x = 13 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2095_209521


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2095_209532

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x > 2}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2095_209532


namespace NUMINAMATH_CALUDE_mike_fish_per_hour_l2095_209559

/-- Represents the number of fish Mike can catch in one hour -/
def M : ℕ := sorry

/-- The number of fish Jim can catch in one hour -/
def jim_catch : ℕ := 2 * M

/-- The number of fish Bob can catch in one hour -/
def bob_catch : ℕ := 3 * M

/-- The total number of fish caught by all three in 40 minutes -/
def total_40min : ℕ := (2 * M / 3) + (4 * M / 3) + (2 * M)

/-- The number of fish Jim catches in the remaining 20 minutes -/
def jim_20min : ℕ := 2 * M / 3

/-- The total number of fish caught in one hour -/
def total_catch : ℕ := total_40min + jim_20min

theorem mike_fish_per_hour : 
  (total_catch = 140) → (M = 30) := by sorry

end NUMINAMATH_CALUDE_mike_fish_per_hour_l2095_209559


namespace NUMINAMATH_CALUDE_percentage_women_after_hiring_l2095_209552

/-- Percentage of women in a multinational company after new hires --/
theorem percentage_women_after_hiring (country_a_initial : ℕ) (country_b_initial : ℕ)
  (country_a_men_ratio : ℚ) (country_b_women_ratio : ℚ)
  (country_a_new_hires : ℕ) (country_b_new_hires : ℕ)
  (country_a_new_men_ratio : ℚ) (country_b_new_women_ratio : ℚ)
  (h1 : country_a_initial = 90)
  (h2 : country_b_initial = 150)
  (h3 : country_a_men_ratio = 2/3)
  (h4 : country_b_women_ratio = 3/5)
  (h5 : country_a_new_hires = 5)
  (h6 : country_b_new_hires = 8)
  (h7 : country_a_new_men_ratio = 3/5)
  (h8 : country_b_new_women_ratio = 1/2) :
  ∃ (percentage : ℚ), abs (percentage - 4980/10000) < 1/1000 ∧
  percentage = (country_a_initial * (1 - country_a_men_ratio) + country_b_initial * country_b_women_ratio +
    country_a_new_hires * (1 - country_a_new_men_ratio) + country_b_new_hires * country_b_new_women_ratio) /
    (country_a_initial + country_b_initial + country_a_new_hires + country_b_new_hires) * 100 :=
by
  sorry


end NUMINAMATH_CALUDE_percentage_women_after_hiring_l2095_209552


namespace NUMINAMATH_CALUDE_village_population_theorem_l2095_209517

theorem village_population_theorem (total_population : ℕ) 
  (h1 : total_population = 800) 
  (h2 : ∃ (part : ℕ), 4 * part = total_population) 
  (h3 : ∃ (male_population : ℕ), male_population = 2 * (total_population / 4)) :
  ∃ (male_population : ℕ), male_population = 400 := by
sorry

end NUMINAMATH_CALUDE_village_population_theorem_l2095_209517


namespace NUMINAMATH_CALUDE_age_difference_l2095_209574

/-- Given three people A, B, and C, where the total age of A and B is 18 years more than
    the total age of B and C, prove that C is 18 years younger than A. -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 18) : A = C + 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2095_209574


namespace NUMINAMATH_CALUDE_alloy_mixture_chromium_balance_l2095_209582

/-- Represents the composition of an alloy mixture -/
structure AlloyMixture where
  first_alloy_amount : ℝ
  first_alloy_chromium_percent : ℝ
  second_alloy_amount : ℝ
  second_alloy_chromium_percent : ℝ
  new_alloy_chromium_percent : ℝ

/-- The alloy mixture satisfies the chromium balance equation -/
def satisfies_chromium_balance (mixture : AlloyMixture) : Prop :=
  mixture.first_alloy_chromium_percent * mixture.first_alloy_amount +
  mixture.second_alloy_chromium_percent * mixture.second_alloy_amount =
  mixture.new_alloy_chromium_percent * (mixture.first_alloy_amount + mixture.second_alloy_amount)

/-- Theorem: The alloy mixture satisfies the chromium balance equation -/
theorem alloy_mixture_chromium_balance 
  (mixture : AlloyMixture)
  (h1 : mixture.second_alloy_amount = 35)
  (h2 : mixture.second_alloy_chromium_percent = 0.08)
  (h3 : mixture.new_alloy_chromium_percent = 0.101) :
  satisfies_chromium_balance mixture :=
sorry

end NUMINAMATH_CALUDE_alloy_mixture_chromium_balance_l2095_209582


namespace NUMINAMATH_CALUDE_arithmetic_sequence_lower_bound_l2095_209578

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property a₁² + a₂ₙ₊₁² = 1 -/
def SequenceProperty (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a 1 ^ 2 + a (2 * n + 1) ^ 2 = 1

theorem arithmetic_sequence_lower_bound
  (a : ℕ → ℝ) (h1 : ArithmeticSequence a) (h2 : SequenceProperty a) :
  ∀ n : ℕ, a (n + 1) ^ 2 + a (3 * n + 1) ^ 2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_lower_bound_l2095_209578


namespace NUMINAMATH_CALUDE_bearing_and_ring_problem_l2095_209543

theorem bearing_and_ring_problem :
  ∃ (x y : ℕ),
    (x = 25 ∧ y = 16) ∨ (x = 16 ∧ y = 25) ∧
    (x : ℤ) + 2 = y ∧
    (y : ℤ) = x + 2 ∧
    x * ((y : ℤ) - 2) + y * (x + 2) - 800 = 2 * (y - x) ∧
    x * x + y * y = 881 :=
by sorry

end NUMINAMATH_CALUDE_bearing_and_ring_problem_l2095_209543


namespace NUMINAMATH_CALUDE_coach_number_divisibility_l2095_209548

/-- A function that checks if a number is of the form aabb, abba, or abab -/
def isValidFormat (n : ℕ) : Prop :=
  ∃ a b : ℕ, 
    (n = a * 1000 + a * 100 + b * 10 + b) ∨ 
    (n = a * 1000 + b * 100 + b * 10 + a) ∨ 
    (n = a * 1000 + b * 100 + a * 10 + b)

/-- The set of possible ages of the children -/
def childrenAges : Set ℕ := {3, 4, 5, 6, 7, 8, 9, 10, 11}

/-- The theorem to be proved -/
theorem coach_number_divisibility 
  (N : ℕ) 
  (h1 : isValidFormat N) 
  (h2 : ∀ (x : ℕ), x ∈ childrenAges → x ≠ 10 → N % x = 0) 
  (h3 : N % 10 ≠ 0) 
  (h4 : 1000 ≤ N ∧ N < 10000) : 
  ∃ (a b : ℕ), N = 7000 + 700 + 40 + 4 := by
  sorry

end NUMINAMATH_CALUDE_coach_number_divisibility_l2095_209548


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l2095_209579

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def axis_of_symmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) (h : is_even_function (fun x => f (x + 3))) :
  axis_of_symmetry f 3 :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l2095_209579


namespace NUMINAMATH_CALUDE_acid_mixture_percentage_l2095_209588

theorem acid_mixture_percentage (a w : ℚ) :
  a > 0 ∧ w > 0 →
  (a + 1) / (a + w + 1) = 1/4 →
  (a + 1) / (a + w + 2) = 1/5 →
  a / (a + w) = 2/11 :=
by sorry

end NUMINAMATH_CALUDE_acid_mixture_percentage_l2095_209588


namespace NUMINAMATH_CALUDE_p_value_l2095_209581

/-- The maximum value of x satisfying the inequality |x^2-4x+p|+|x-3|≤5 is 3 -/
def max_x_condition (p : ℝ) : Prop :=
  ∀ x : ℝ, |x^2 - 4*x + p| + |x - 3| ≤ 5 → x ≤ 3

/-- Theorem stating that p = 8 given the condition -/
theorem p_value : ∃ p : ℝ, max_x_condition p ∧ p = 8 :=
sorry

end NUMINAMATH_CALUDE_p_value_l2095_209581


namespace NUMINAMATH_CALUDE_counterexample_exists_l2095_209513

theorem counterexample_exists : ∃ n : ℕ, 
  (∀ m : ℕ, m * m ≠ n) ∧ ¬(Nat.Prime (n + 4)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2095_209513


namespace NUMINAMATH_CALUDE_triangle_area_l2095_209506

theorem triangle_area (a b c : ℝ) (h_a : a = 39) (h_b : b = 36) (h_c : c = 15) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 270 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2095_209506


namespace NUMINAMATH_CALUDE_negation_equivalence_l2095_209569

theorem negation_equivalence : 
  (¬∃ x₀ : ℝ, x₀^2 - 2*x₀ + 4 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2095_209569


namespace NUMINAMATH_CALUDE_cyclist_final_speed_l2095_209566

/-- Calculates the final speed of a cyclist given initial speed, acceleration, and time. -/
def final_speed (initial_speed : ℝ) (acceleration : ℝ) (time : ℝ) : ℝ :=
  initial_speed + acceleration * time

/-- Converts speed from m/s to km/h. -/
def ms_to_kmh (speed_ms : ℝ) : ℝ :=
  speed_ms * 3.6

theorem cyclist_final_speed :
  let initial_speed := 16 -- m/s
  let acceleration := 0.5 -- m/s²
  let time := 2 * 3600 -- 2 hours in seconds
  let final_speed_ms := final_speed initial_speed acceleration time
  let final_speed_kmh := ms_to_kmh final_speed_ms
  final_speed_kmh = 13017.6 := by sorry

end NUMINAMATH_CALUDE_cyclist_final_speed_l2095_209566


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l2095_209597

/-- Calculates the number of medium stores to be drawn in stratified sampling -/
def medium_stores_drawn (total_stores : ℕ) (medium_stores : ℕ) (sample_size : ℕ) : ℕ :=
  (medium_stores * sample_size) / total_stores

theorem stratified_sampling_medium_stores 
  (total_stores : ℕ) (medium_stores : ℕ) (sample_size : ℕ)
  (h1 : total_stores = 300)
  (h2 : medium_stores = 75)
  (h3 : sample_size = 20) :
  medium_stores_drawn total_stores medium_stores sample_size = 5 := by
sorry

#eval medium_stores_drawn 300 75 20

end NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l2095_209597


namespace NUMINAMATH_CALUDE_triangle_side_ratio_sum_bounds_l2095_209535

theorem triangle_side_ratio_sum_bounds (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  1 < a / (b + c) + b / (c + a) + c / (a + b) ∧ a / (b + c) + b / (c + a) + c / (a + b) < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_sum_bounds_l2095_209535


namespace NUMINAMATH_CALUDE_function_property_l2095_209564

theorem function_property (f : ℕ+ → ℕ) 
  (h1 : ∀ (x y : ℕ+), f (x * y) = f x + f y)
  (h2 : f 10 = 16)
  (h3 : f 40 = 26)
  (h4 : f 8 = 12) :
  f 1000 = 48 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2095_209564


namespace NUMINAMATH_CALUDE_circle_bisection_l2095_209507

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def Circle.center (c : Circle) : ℝ × ℝ := sorry

/-- A line bisects a circle if it passes through the circle's center -/
def bisects (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  l.equation x₀ y₀

/-- The main theorem -/
theorem circle_bisection (c : Circle) (l : Line) (a : ℝ) :
  c.equation = (fun x y ↦ x^2 + y^2 + 2*x - 4*y = 0) →
  l.equation = (fun x y ↦ 3*x + y + a = 0) →
  bisects l c →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_circle_bisection_l2095_209507


namespace NUMINAMATH_CALUDE_cos_two_alpha_zero_l2095_209555

theorem cos_two_alpha_zero (α : Real) 
  (h : Real.sin (π/6 - α) = Real.cos (π/6 + α)) : 
  Real.cos (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_zero_l2095_209555


namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l2095_209568

/-- A triangle with sides a, b, and c is either isosceles or right-angled if (a-b)(a^2+b^2-c^2) = 0 --/
theorem triangle_isosceles_or_right_angled 
  {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_equation : (a - b) * (a^2 + b^2 - c^2) = 0) : 
  (a = b ∨ a = c ∨ b = c) ∨ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l2095_209568


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l2095_209518

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 5 ways to distribute 5 indistinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 5 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l2095_209518


namespace NUMINAMATH_CALUDE_power_87_plus_3_mod_7_l2095_209509

theorem power_87_plus_3_mod_7 : (2^87 + 3) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_87_plus_3_mod_7_l2095_209509


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2095_209580

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo (-1/2 : ℝ) (1/3 : ℝ) = {x : ℝ | a * x^2 + b * x + 2 > 0}) :
  {x : ℝ | 2 * x^2 + b * x + a < 0} = Set.Ioo (-2 : ℝ) (3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2095_209580


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l2095_209558

theorem stratified_sampling_male_count :
  ∀ (total_employees : ℕ) 
    (female_employees : ℕ) 
    (sample_size : ℕ),
  total_employees = 120 →
  female_employees = 72 →
  sample_size = 15 →
  (total_employees - female_employees) * sample_size / total_employees = 6 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l2095_209558


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2095_209544

def A : Set ℤ := {0, 1, 2}

def U : Set ℤ := {z | ∃ x y, x ∈ A ∧ y ∈ A ∧ z = x - y}

theorem complement_of_A_in_U :
  (A : Set ℤ)ᶜ ∩ U = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2095_209544


namespace NUMINAMATH_CALUDE_speed_ratio_l2095_209591

/-- Represents the position and speed of an object moving in a straight line. -/
structure Mover where
  speed : ℝ
  initialPosition : ℝ

/-- The problem setup -/
def problem (a b : Mover) : Prop :=
  -- A and B move uniformly along two straight paths intersecting at right angles at point O
  -- When A is at O, B is 400 yards short of O
  a.initialPosition = 0 ∧ b.initialPosition = -400 ∧
  -- In 3 minutes, they are equidistant from O
  (3 * a.speed)^2 = (-400 + 3 * b.speed)^2 ∧
  -- In 10 minutes (3 + 7 minutes), they are again equidistant from O
  (10 * a.speed)^2 = (-400 + 10 * b.speed)^2

/-- The theorem to be proved -/
theorem speed_ratio (a b : Mover) :
  problem a b → a.speed / b.speed = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_l2095_209591


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2095_209527

/-- Given a rectangular prism with side face areas of √2, √3, and √6, its volume is √6 -/
theorem rectangular_prism_volume (a b c : ℝ) 
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : a * c = Real.sqrt 6) : 
  a * b * c = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2095_209527


namespace NUMINAMATH_CALUDE_cylinder_volume_theorem_l2095_209571

/-- Represents the dimensions of a rectangle formed by unrolling a cylinder's lateral surface -/
structure UnrolledCylinder where
  side1 : ℝ
  side2 : ℝ

/-- Calculates the possible volumes of a cylinder given its unrolled lateral surface dimensions -/
def possible_cylinder_volumes (uc : UnrolledCylinder) : Set ℝ :=
  let v1 := (uc.side1 / (2 * Real.pi)) ^ 2 * Real.pi * uc.side2
  let v2 := (uc.side2 / (2 * Real.pi)) ^ 2 * Real.pi * uc.side1
  {v1, v2}

/-- Theorem stating that a cylinder with unrolled lateral surface of 8π and 4π has volume 32π² or 64π² -/
theorem cylinder_volume_theorem (uc : UnrolledCylinder) 
    (h1 : uc.side1 = 8 * Real.pi) (h2 : uc.side2 = 4 * Real.pi) : 
    possible_cylinder_volumes uc = {32 * Real.pi ^ 2, 64 * Real.pi ^ 2} := by
  sorry

#check cylinder_volume_theorem

end NUMINAMATH_CALUDE_cylinder_volume_theorem_l2095_209571


namespace NUMINAMATH_CALUDE_expand_product_l2095_209541

theorem expand_product (x : ℝ) : (x + 4) * (x - 5 + 2) = x^2 + x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2095_209541


namespace NUMINAMATH_CALUDE_circles_common_chord_l2095_209516

/-- Two circles with equations x² + y² - 2x + 2y - 2 = 0 and x² + y² - 2mx = 0 (m > 0) 
    have a common chord of length 2 if and only if m = √6/2 -/
theorem circles_common_chord (m : ℝ) (hm : m > 0) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + 2*y - 2 = 0 ∧ x^2 + y^2 - 2*m*x = 0) ∧ 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 - 2*x₁ + 2*y₁ - 2 = 0 ∧
    x₁^2 + y₁^2 - 2*m*x₁ = 0 ∧
    x₂^2 + y₂^2 - 2*x₂ + 2*y₂ - 2 = 0 ∧
    x₂^2 + y₂^2 - 2*m*x₂ = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4) ↔ 
  m = Real.sqrt 6 / 2 := by
sorry

end NUMINAMATH_CALUDE_circles_common_chord_l2095_209516


namespace NUMINAMATH_CALUDE_m_range_l2095_209575

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - m)^x < (3 - m)^y

theorem m_range : 
  (∃ m : ℝ, (¬(p m ∧ q m)) ∧ (p m ∨ q m)) ↔ 
  (∃ m : ℝ, m ≥ 1 ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_m_range_l2095_209575


namespace NUMINAMATH_CALUDE_projection_matrix_values_l2095_209534

def isProjectionMatrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, 20/36; c, 16/36]

theorem projection_matrix_values :
  ∀ a c : ℚ, isProjectionMatrix (P a c) → a = 1/27 ∧ c = 5/27 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l2095_209534


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2095_209565

theorem sum_of_roots_quadratic (x : ℝ) : 
  (4 * x + 3) * (3 * x - 8) = 0 → 
  ∃ x₁ x₂ : ℝ, x₁ + x₂ = 23 / 12 ∧ 
    ((4 * x₁ + 3) * (3 * x₁ - 8) = 0) ∧ 
    ((4 * x₂ + 3) * (3 * x₂ - 8) = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2095_209565


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2095_209523

theorem rectangle_perimeter (a b : ℕ) : 
  a ≠ b →  -- non-square condition
  a * b - 3 * (a + b) = 3 * a * b - 9 →  -- given equation
  2 * (a + b) = 14 :=  -- perimeter = 14
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2095_209523


namespace NUMINAMATH_CALUDE_digital_earth_storage_technologies_l2095_209547

-- Define the set of all possible technologies
inductive Technology
| Nano
| LaserHolographic
| Protein
| Distributed
| Virtual
| Spatial
| Visualization

-- Define the property of contributing to digital Earth data storage
def contributesToDigitalEarthStorage (tech : Technology) : Prop :=
  match tech with
  | Technology.Nano => true
  | Technology.LaserHolographic => true
  | Technology.Protein => true
  | Technology.Distributed => true
  | _ => false

-- Define the set of technologies that contribute to digital Earth storage
def contributingTechnologies : Set Technology :=
  {tech | contributesToDigitalEarthStorage tech}

-- Theorem statement
theorem digital_earth_storage_technologies :
  contributingTechnologies = {Technology.Nano, Technology.LaserHolographic, Technology.Protein, Technology.Distributed} :=
by sorry

end NUMINAMATH_CALUDE_digital_earth_storage_technologies_l2095_209547


namespace NUMINAMATH_CALUDE_card_relationship_l2095_209542

theorem card_relationship (c : ℝ) (h1 : c > 0) : 
  let b := 1.2 * c
  let d := 1.4 * b
  d = 1.68 * c := by sorry

end NUMINAMATH_CALUDE_card_relationship_l2095_209542


namespace NUMINAMATH_CALUDE_day_197_is_wednesday_l2095_209519

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a year -/
structure DayInYear where
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- Returns true if two days fall on the same day of the week -/
def sameDayOfWeek (day1 day2 : DayInYear) : Prop :=
  (day2.dayNumber - day1.dayNumber) % 7 = 0

theorem day_197_is_wednesday (day15 day197 : DayInYear) :
  day15.dayNumber = 15 →
  day197.dayNumber = 197 →
  day15.dayOfWeek = DayOfWeek.Wednesday →
  sameDayOfWeek day15 day197 →
  day197.dayOfWeek = DayOfWeek.Wednesday :=
by
  sorry

end NUMINAMATH_CALUDE_day_197_is_wednesday_l2095_209519


namespace NUMINAMATH_CALUDE_min_total_cost_both_measures_l2095_209567

/-- Represents the cost and effectiveness of a preventive measure -/
structure PreventiveMeasure where
  cost : ℝ
  effectiveness : ℝ

/-- Calculates the total cost given the initial probability, potential loss,
    and a list of implemented preventive measures -/
def totalCost (initialProb : ℝ) (potentialLoss : ℝ) (measures : List PreventiveMeasure) : ℝ :=
  let measuresCost := measures.foldl (fun acc m => acc + m.cost) 0
  let finalProb := measures.foldl (fun acc m => acc * (1 - m.effectiveness)) initialProb
  measuresCost + finalProb * potentialLoss

/-- Theorem stating that the minimum total cost is achieved by implementing both measures -/
theorem min_total_cost_both_measures
  (initialProb : ℝ)
  (potentialLoss : ℝ)
  (measureA : PreventiveMeasure)
  (measureB : PreventiveMeasure)
  (h_initialProb : initialProb = 0.3)
  (h_potentialLoss : potentialLoss = 400)
  (h_measureA : measureA = { cost := 0.45, effectiveness := 0.9 })
  (h_measureB : measureB = { cost := 0.3, effectiveness := 0.85 }) :
  (totalCost initialProb potentialLoss [measureA, measureB] ≤ 
   min (totalCost initialProb potentialLoss [])
      (min (totalCost initialProb potentialLoss [measureA])
           (totalCost initialProb potentialLoss [measureB]))) ∧
  (totalCost initialProb potentialLoss [measureA, measureB] = 81) := by
  sorry

#check min_total_cost_both_measures

end NUMINAMATH_CALUDE_min_total_cost_both_measures_l2095_209567


namespace NUMINAMATH_CALUDE_trig_identity_l2095_209551

theorem trig_identity (θ : Real) (h : θ ≠ 0) (h2 : θ ≠ π/2) : 
  (Real.tan θ)^2 - (Real.sin θ)^2 = (Real.tan θ)^2 * (Real.sin θ)^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2095_209551


namespace NUMINAMATH_CALUDE_lake_crossing_time_difference_l2095_209503

theorem lake_crossing_time_difference 
  (lake_width : ℝ) 
  (janet_speed : ℝ) 
  (sister_speed : ℝ) 
  (h1 : lake_width = 60) 
  (h2 : janet_speed = 30) 
  (h3 : sister_speed = 12) : 
  (lake_width / sister_speed) - (lake_width / janet_speed) = 3 := by
sorry

end NUMINAMATH_CALUDE_lake_crossing_time_difference_l2095_209503


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l2095_209528

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (k : ℕ), k < 8 ∧ (964807 - k) % 8 = 0 ∧ ∀ (m : ℕ), m < k → (964807 - m) % 8 ≠ 0 ∧ k = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l2095_209528


namespace NUMINAMATH_CALUDE_regular_pentagon_side_length_l2095_209524

/-- The length of a side of a regular pentagon with perimeter 125 is 25 -/
theorem regular_pentagon_side_length :
  ∀ (side_length : ℝ),
    side_length > 0 →
    side_length * 5 = 125 →
    side_length = 25 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_side_length_l2095_209524


namespace NUMINAMATH_CALUDE_sum_of_divisors_of_twelve_l2095_209563

theorem sum_of_divisors_of_twelve : (Finset.filter (λ x => 12 % x = 0) (Finset.range 13)).sum id = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_of_twelve_l2095_209563
