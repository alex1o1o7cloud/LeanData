import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_l791_79173

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- The property that f satisfies for all positive integers m and n -/
def SatisfiesEquation (f : PositiveIntFunction) : Prop :=
  ∀ m n : ℕ+, f (f (f m)^2 + 2 * (f n)^2) = m^2 + 2 * n^2

/-- The identity function on positive integers -/
def identityFunction : PositiveIntFunction := λ n => n

/-- The theorem stating that the identity function is the only one satisfying the equation -/
theorem unique_solution :
  ∀ f : PositiveIntFunction, SatisfiesEquation f ↔ f = identityFunction :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l791_79173


namespace NUMINAMATH_CALUDE_karlsson_candies_l791_79122

/-- The number of ones initially written on the board -/
def initial_ones : ℕ := 28

/-- The number of minutes the process continues -/
def total_minutes : ℕ := 28

/-- The number of edges in a complete graph with n vertices -/
def complete_graph_edges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The maximum number of candies Karlsson could eat -/
def max_candies : ℕ := complete_graph_edges initial_ones

theorem karlsson_candies :
  max_candies = 378 :=
sorry

end NUMINAMATH_CALUDE_karlsson_candies_l791_79122


namespace NUMINAMATH_CALUDE_base_7_representation_of_500_l791_79189

/-- Converts a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 7 to a natural number -/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_7_representation_of_500 :
  toBase7 500 = [1, 3, 1, 3] ∧ fromBase7 [1, 3, 1, 3] = 500 := by
  sorry

end NUMINAMATH_CALUDE_base_7_representation_of_500_l791_79189


namespace NUMINAMATH_CALUDE_line_point_distance_l791_79125

/-- Given five points O, A, B, C, D on a line, with Q and P also on the line,
    prove that OP = 2q under the given conditions. -/
theorem line_point_distance (a b c d q : ℝ) : 
  ∀ (x : ℝ), 
  (0 < a) → (a < b) → (b < c) → (c < d) →  -- Points are in order
  (0 < q) → (q < d) →  -- Q is on the line
  (b ≤ x) → (x ≤ c) →  -- P is between B and C
  ((a - x) / (x - d) = (b - x) / (x - c)) →  -- AP : PD = BP : PC
  (x = 2 * q) →  -- P is twice as far from O as Q is
  x = 2 * q := by
  sorry

end NUMINAMATH_CALUDE_line_point_distance_l791_79125


namespace NUMINAMATH_CALUDE_complex_square_roots_l791_79182

theorem complex_square_roots (z : ℂ) : 
  z ^ 2 = -115 + 66 * I ↔ z = 3 + 11 * I ∨ z = -3 - 11 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_roots_l791_79182


namespace NUMINAMATH_CALUDE_team_pays_seventy_percent_l791_79151

/-- Represents the archer's arrow usage and costs --/
structure ArcherData where
  shots_per_day : ℕ
  days_per_week : ℕ
  recovery_rate : ℚ
  arrow_cost : ℚ
  weekly_spending : ℚ

/-- Calculates the percentage of arrow costs paid by the team --/
def team_payment_percentage (data : ArcherData) : ℚ :=
  let total_shots := data.shots_per_day * data.days_per_week
  let unrecovered_arrows := total_shots * (1 - data.recovery_rate)
  let total_cost := unrecovered_arrows * data.arrow_cost
  let team_contribution := total_cost - data.weekly_spending
  (team_contribution / total_cost) * 100

/-- Theorem stating that the team pays 70% of the archer's arrow costs --/
theorem team_pays_seventy_percent (data : ArcherData)
  (h1 : data.shots_per_day = 200)
  (h2 : data.days_per_week = 4)
  (h3 : data.recovery_rate = 1/5)
  (h4 : data.arrow_cost = 11/2)
  (h5 : data.weekly_spending = 1056) :
  team_payment_percentage data = 70 := by
  sorry

end NUMINAMATH_CALUDE_team_pays_seventy_percent_l791_79151


namespace NUMINAMATH_CALUDE_pet_ownership_problem_l791_79153

/-- Represents the number of students in each section of the Venn diagram -/
structure PetOwnership where
  dogs_only : ℕ
  cats_only : ℕ
  other_only : ℕ
  dogs_cats : ℕ
  cats_other : ℕ
  dogs_other : ℕ
  all_three : ℕ

/-- The main theorem to prove -/
theorem pet_ownership_problem (po : PetOwnership) : po.all_three = 4 :=
  by
  have total_students : ℕ := 40
  have dog_fraction : Rat := 5 / 8
  have cat_fraction : Rat := 1 / 4
  have other_pet_count : ℕ := 8
  have no_pet_count : ℕ := 4

  have dogs_only : po.dogs_only = 15 := by sorry
  have cats_only : po.cats_only = 3 := by sorry
  have other_only : po.other_only = 2 := by sorry

  have dog_eq : po.dogs_only + po.dogs_cats + po.dogs_other + po.all_three = (total_students : ℚ) * dog_fraction := by sorry
  have cat_eq : po.cats_only + po.dogs_cats + po.cats_other + po.all_three = (total_students : ℚ) * cat_fraction := by sorry
  have other_eq : po.other_only + po.cats_other + po.dogs_other + po.all_three = other_pet_count := by sorry
  have total_eq : po.dogs_only + po.cats_only + po.other_only + po.dogs_cats + po.cats_other + po.dogs_other + po.all_three = total_students - no_pet_count := by sorry

  sorry

end NUMINAMATH_CALUDE_pet_ownership_problem_l791_79153


namespace NUMINAMATH_CALUDE_remainder_problem_l791_79155

theorem remainder_problem (x y : ℤ) 
  (hx : x % 126 = 11) 
  (hy : y % 126 = 25) : 
  (x + y + 23) % 63 = 59 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l791_79155


namespace NUMINAMATH_CALUDE_smallest_a_l791_79164

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the vertex condition
def vertex_condition (a b c : ℝ) : Prop :=
  parabola a b c (1/3) = -4/3

-- Define the integer condition
def integer_condition (a b c : ℝ) : Prop :=
  ∃ n : ℤ, 3*a + 2*b + c = n

-- State the theorem
theorem smallest_a (a b c : ℝ) :
  vertex_condition a b c →
  integer_condition a b c →
  a > 0 →
  (∀ a' b' c' : ℝ, vertex_condition a' b' c' → integer_condition a' b' c' → a' > 0 → a' ≥ a) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_l791_79164


namespace NUMINAMATH_CALUDE_expression_factorization_l791_79128

theorem expression_factorization (x : ℝ) :
  (16 * x^7 + 32 * x^5 - 9) - (4 * x^7 - 8 * x^5 + 9) = 2 * (6 * x^7 + 20 * x^5 - 9) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l791_79128


namespace NUMINAMATH_CALUDE_prime_sequence_l791_79145

theorem prime_sequence (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
by sorry

end NUMINAMATH_CALUDE_prime_sequence_l791_79145


namespace NUMINAMATH_CALUDE_abs_sum_lt_sum_abs_iff_product_neg_l791_79115

theorem abs_sum_lt_sum_abs_iff_product_neg (a b : ℝ) :
  |a + b| < |a| + |b| ↔ a * b < 0 :=
sorry

end NUMINAMATH_CALUDE_abs_sum_lt_sum_abs_iff_product_neg_l791_79115


namespace NUMINAMATH_CALUDE_existence_of_integers_l791_79118

theorem existence_of_integers (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ (x y k : ℤ), 0 < 2 * k ∧ 2 * k < p ∧ k * p + 3 = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_l791_79118


namespace NUMINAMATH_CALUDE_prob_at_least_one_value_l791_79133

/-- The probability of picking a road from A to B that is at least 5 miles long -/
def prob_AB : ℚ := 2/3

/-- The probability of picking a road from B to C that is at least 5 miles long -/
def prob_BC : ℚ := 3/4

/-- The probability that at least one of the randomly picked roads (one from A to B, one from B to C) is at least 5 miles long -/
def prob_at_least_one : ℚ := 1 - (1 - prob_AB) * (1 - prob_BC)

theorem prob_at_least_one_value : prob_at_least_one = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_value_l791_79133


namespace NUMINAMATH_CALUDE_initial_money_calculation_l791_79191

/-- Calculates the initial amount of money given the cost of bread, peanut butter, and the amount left over --/
theorem initial_money_calculation (bread_cost : ℝ) (bread_quantity : ℕ) (peanut_butter_cost : ℝ) (money_left : ℝ) : 
  bread_cost = 2.25 →
  bread_quantity = 3 →
  peanut_butter_cost = 2 →
  money_left = 5.25 →
  bread_cost * (bread_quantity : ℝ) + peanut_butter_cost + money_left = 14 :=
by sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l791_79191


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_value_l791_79160

def M (a : ℝ) : Set ℝ := {x | x - a = 0}
def N (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem intersection_equality_implies_a_value (a : ℝ) :
  M a ∩ N a = N a → a = 0 ∨ a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_value_l791_79160


namespace NUMINAMATH_CALUDE_specific_rectangle_triangles_l791_79174

/-- Represents a rectangle with a grid and diagonals -/
structure GridRectangle where
  width : ℕ
  height : ℕ
  vertical_spacing : ℕ
  horizontal_spacing : ℕ

/-- Counts the number of triangles in a GridRectangle -/
def count_triangles (rect : GridRectangle) : ℕ :=
  sorry

/-- The main theorem stating the number of triangles in the specific configuration -/
theorem specific_rectangle_triangles :
  let rect : GridRectangle := {
    width := 40,
    height := 10,
    vertical_spacing := 10,
    horizontal_spacing := 5
  }
  count_triangles rect = 74 := by
  sorry

end NUMINAMATH_CALUDE_specific_rectangle_triangles_l791_79174


namespace NUMINAMATH_CALUDE_square_area_after_cut_l791_79167

theorem square_area_after_cut (x : ℝ) : 
  x > 0 → x * (x - 3) = 40 → x^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_after_cut_l791_79167


namespace NUMINAMATH_CALUDE_abs_square_eq_neg_cube_l791_79195

theorem abs_square_eq_neg_cube (a b : ℤ) (h1 : a = -8) (h2 : b = -4) : 
  |a|^2 = -(b^3) := by
  sorry

end NUMINAMATH_CALUDE_abs_square_eq_neg_cube_l791_79195


namespace NUMINAMATH_CALUDE_point_on_transformed_plane_l791_79188

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Apply a similarity transformation to a plane -/
def similarityTransform (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Check if a point lies on a plane -/
def pointOnPlane (point : Point) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

theorem point_on_transformed_plane :
  let originalPlane : Plane := { a := 3, b := -1, c := 2, d := 4 }
  let k : ℝ := 1/2
  let transformedPlane := similarityTransform originalPlane k
  let pointA : Point := { x := -1, y := 1, z := 1 }
  pointOnPlane pointA transformedPlane := by sorry

end NUMINAMATH_CALUDE_point_on_transformed_plane_l791_79188


namespace NUMINAMATH_CALUDE_first_day_price_is_four_l791_79138

/-- Represents the sales data for a pen store over three days -/
structure PenSales where
  price1 : ℝ  -- Price per pen on the first day
  quantity1 : ℝ  -- Number of pens sold on the first day

/-- The revenue is the same for all three days given the pricing and quantity changes -/
def sameRevenue (s : PenSales) : Prop :=
  s.price1 * s.quantity1 = (s.price1 - 1) * (s.quantity1 + 100) ∧
  s.price1 * s.quantity1 = (s.price1 + 2) * (s.quantity1 - 100)

/-- The price on the first day is 4 yuan -/
theorem first_day_price_is_four (s : PenSales) (h : sameRevenue s) : s.price1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_day_price_is_four_l791_79138


namespace NUMINAMATH_CALUDE_probability_first_greater_than_second_l791_79126

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 3), (5, 4)}

theorem probability_first_greater_than_second :
  (Finset.card favorable_outcomes : ℚ) / (Finset.card card_set ^ 2 : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_first_greater_than_second_l791_79126


namespace NUMINAMATH_CALUDE_consecutive_terms_iff_equation_l791_79150

/-- Sequence a_k defined by a_0 = 0, a_1 = n, and a_{k+1} = n^2 * a_k - a_{k-1} -/
def sequence_a (n : ℕ) : ℕ → ℤ
  | 0 => 0
  | 1 => n
  | (k + 2) => n^2 * sequence_a n (k + 1) - sequence_a n k

/-- Predicate to check if two integers are consecutive terms in the sequence -/
def are_consecutive_terms (n a b : ℕ) : Prop :=
  ∃ k : ℕ, sequence_a n k = a ∧ sequence_a n (k + 1) = b

theorem consecutive_terms_iff_equation (n : ℕ) (hn : n > 0) (a b : ℕ) (hab : a ≤ b) :
  are_consecutive_terms n a b ↔ a^2 + b^2 = n^2 * (a * b + 1) :=
sorry

end NUMINAMATH_CALUDE_consecutive_terms_iff_equation_l791_79150


namespace NUMINAMATH_CALUDE_line_x_intercept_l791_79136

/-- Given a straight line passing through points (2, -2) and (-3, 7), 
    its x-intercept is 8/9 -/
theorem line_x_intercept : 
  ∀ (f : ℝ → ℝ), 
  (∃ m b : ℝ, ∀ x, f x = m * x + b) → -- f is a linear function
  f 2 = -2 →                         -- f passes through (2, -2)
  f (-3) = 7 →                       -- f passes through (-3, 7)
  ∃ x : ℝ, x = 8/9 ∧ f x = 0 :=      -- x-intercept is 8/9
by
  sorry


end NUMINAMATH_CALUDE_line_x_intercept_l791_79136


namespace NUMINAMATH_CALUDE_probability_face_card_is_three_thirteenths_l791_79198

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of jacks, queens, and kings in a standard deck -/
def face_cards : ℕ := 12

/-- The probability of drawing a jack, queen, or king from a standard deck -/
def probability_face_card : ℚ := face_cards / deck_size

theorem probability_face_card_is_three_thirteenths :
  probability_face_card = 3 / 13 := by sorry

end NUMINAMATH_CALUDE_probability_face_card_is_three_thirteenths_l791_79198


namespace NUMINAMATH_CALUDE_first_player_wins_l791_79161

/-- Represents a position on the 8x8 grid --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Defines the possible moves --/
inductive Move
  | Right
  | Up
  | UpRight

/-- Applies a move to a position --/
def applyMove (p : Position) (m : Move) : Position :=
  match m with
  | Move.Right => ⟨p.x + 1, p.y⟩
  | Move.Up => ⟨p.x, p.y + 1⟩
  | Move.UpRight => ⟨p.x + 1, p.y + 1⟩

/-- Checks if a position is within the 8x8 grid --/
def isValidPosition (p : Position) : Prop :=
  1 ≤ p.x ∧ p.x ≤ 8 ∧ 1 ≤ p.y ∧ p.y ≤ 8

/-- Defines a winning position --/
def isWinningPosition (p : Position) : Prop :=
  p.x = 8 ∧ p.y = 8

/-- Theorem: The first player has a winning strategy --/
theorem first_player_wins :
  ∃ (m : Move), isValidPosition (applyMove ⟨1, 1⟩ m) ∧
  ∀ (p : Position),
    isValidPosition p →
    ¬isWinningPosition p →
    (p.x % 2 = 0 ∧ p.y % 2 = 0) →
    ∃ (m : Move),
      isValidPosition (applyMove p m) ∧
      ¬(applyMove p m).x % 2 = 0 ∧
      ¬(applyMove p m).y % 2 = 0 :=
by sorry

#check first_player_wins

end NUMINAMATH_CALUDE_first_player_wins_l791_79161


namespace NUMINAMATH_CALUDE_ellipse_vertex_focus_distance_l791_79196

/-- Given an ellipse with equation x²/16 + y²/12 = 1, 
    prove that the distance between the left vertex and the right focus is 6. -/
theorem ellipse_vertex_focus_distance : 
  let a : ℝ := 4  -- semi-major axis
  let b : ℝ := 2 * Real.sqrt 3  -- semi-minor axis
  let c : ℝ := Real.sqrt (a^2 - b^2)  -- focal distance
  a + c = 6 := by sorry

end NUMINAMATH_CALUDE_ellipse_vertex_focus_distance_l791_79196


namespace NUMINAMATH_CALUDE_mms_given_to_sister_correct_l791_79147

/-- The number of m&m's Cheryl gave to her sister -/
def mms_given_to_sister (initial : ℕ) (eaten_lunch : ℕ) (eaten_dinner : ℕ) : ℕ :=
  initial - (eaten_lunch + eaten_dinner)

/-- Theorem stating that the number of m&m's given to sister is correct -/
theorem mms_given_to_sister_correct (initial : ℕ) (eaten_lunch : ℕ) (eaten_dinner : ℕ) 
  (h1 : initial ≥ eaten_lunch + eaten_dinner) :
  mms_given_to_sister initial eaten_lunch eaten_dinner = initial - (initial - (eaten_lunch + eaten_dinner)) :=
by
  sorry

#eval mms_given_to_sister 25 7 5

end NUMINAMATH_CALUDE_mms_given_to_sister_correct_l791_79147


namespace NUMINAMATH_CALUDE_circle_equation_l791_79171

/-- A circle with center (a, 1) that is tangent to both lines x-y+1=0 and x-y-3=0 -/
structure TangentCircle where
  a : ℝ
  center : ℝ × ℝ
  tangent_line1 : ℝ → ℝ → ℝ
  tangent_line2 : ℝ → ℝ → ℝ
  center_def : center = (a, 1)
  tangent_line1_def : tangent_line1 = fun x y => x - y + 1
  tangent_line2_def : tangent_line2 = fun x y => x - y - 3
  is_tangent1 : ∃ (x y : ℝ), tangent_line1 x y = 0 ∧ (x - a)^2 + (y - 1)^2 = (x - center.1)^2 + (y - center.2)^2
  is_tangent2 : ∃ (x y : ℝ), tangent_line2 x y = 0 ∧ (x - a)^2 + (y - 1)^2 = (x - center.1)^2 + (y - center.2)^2

/-- The standard equation of the circle is (x-2)^2+(y-1)^2=2 -/
theorem circle_equation (c : TangentCircle) : 
  ∃ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 2 ∧ 
  (x - c.center.1)^2 + (y - c.center.2)^2 = (x - 2)^2 + (y - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l791_79171


namespace NUMINAMATH_CALUDE_no_valid_grid_l791_79176

/-- Represents a grid of stars -/
def StarGrid := Fin 10 → Fin 10 → Bool

/-- Counts the number of stars in a 2x2 square starting at (i, j) -/
def countStars2x2 (grid : StarGrid) (i j : Fin 10) : Nat :=
  (grid i j).toNat + (grid i (j+1)).toNat + (grid (i+1) j).toNat + (grid (i+1) (j+1)).toNat

/-- Counts the number of stars in a 3x1 rectangle starting at (i, j) -/
def countStars3x1 (grid : StarGrid) (i j : Fin 10) : Nat :=
  (grid i j).toNat + (grid i (j+1)).toNat + (grid i (j+2)).toNat

/-- Checks if the grid satisfies the conditions -/
def isValidGrid (grid : StarGrid) : Prop :=
  (∀ i j, i < 9 ∧ j < 9 → countStars2x2 grid i j = 2) ∧
  (∀ i j, j < 8 → countStars3x1 grid i j = 1)

theorem no_valid_grid : ¬∃ (grid : StarGrid), isValidGrid grid := by
  sorry

end NUMINAMATH_CALUDE_no_valid_grid_l791_79176


namespace NUMINAMATH_CALUDE_workshop_workers_l791_79103

/-- The total number of workers in a workshop given specific salary conditions -/
theorem workshop_workers (average_salary : ℚ) (technician_count : ℕ) (technician_salary : ℚ) (rest_salary : ℚ) : 
  average_salary = 850 ∧ 
  technician_count = 7 ∧ 
  technician_salary = 1000 ∧ 
  rest_salary = 780 →
  ∃ (total_workers : ℕ), total_workers = 22 ∧
    (technician_count : ℚ) * technician_salary + 
    (total_workers - technician_count : ℚ) * rest_salary = 
    (total_workers : ℚ) * average_salary :=
by
  sorry


end NUMINAMATH_CALUDE_workshop_workers_l791_79103


namespace NUMINAMATH_CALUDE_equation_has_real_root_l791_79170

theorem equation_has_real_root (a b c : ℝ) : 
  ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l791_79170


namespace NUMINAMATH_CALUDE_factorization_equality_l791_79199

theorem factorization_equality (a b : ℝ) : 12 * a^3 * b - 12 * a^2 * b + 3 * a * b = 3 * a * b * (2*a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l791_79199


namespace NUMINAMATH_CALUDE_average_excluding_extremes_for_given_batsman_l791_79142

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : ℕ
  average : ℚ
  highestScore : ℕ
  scoreDifference : ℕ

/-- Calculates the average excluding highest and lowest scores -/
def averageExcludingExtremes (stats : BatsmanStats) : ℚ :=
  let totalRuns := stats.average * stats.innings
  let lowestScore := stats.highestScore - stats.scoreDifference
  let remainingRuns := totalRuns - stats.highestScore - lowestScore
  remainingRuns / (stats.innings - 2)

/-- Theorem stating the average excluding extremes for given conditions -/
theorem average_excluding_extremes_for_given_batsman :
  let stats : BatsmanStats := {
    innings := 46,
    average := 60,
    highestScore := 174,
    scoreDifference := 140
  }
  averageExcludingExtremes stats = 58 := by
  sorry

end NUMINAMATH_CALUDE_average_excluding_extremes_for_given_batsman_l791_79142


namespace NUMINAMATH_CALUDE_sequence_sum_l791_79166

theorem sequence_sum (a b : ℕ) : 
  let seq : List ℕ := [a, b, a + b, a + 2*b, 2*a + 3*b, a + 2*b + 7, a + 2*b + 14, 2*a + 4*b + 21, 3*a + 6*b + 35]
  2*a + 3*b = 7 → 
  3*a + 6*b + 35 = 47 → 
  seq.sum = 122 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l791_79166


namespace NUMINAMATH_CALUDE_simplify_exponent_division_l791_79194

theorem simplify_exponent_division (x : ℝ) (h : x ≠ 0) : x^6 / x^2 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponent_division_l791_79194


namespace NUMINAMATH_CALUDE_largest_reciprocal_l791_79104

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/3 → b = 2/5 → c = 1 → d = 5 → e = 1986 →
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l791_79104


namespace NUMINAMATH_CALUDE_unique_square_number_l791_79185

/-- A function to convert a two-digit number to its decimal representation -/
def twoDigitToNumber (a b : ℕ) : ℕ := 10 * a + b

/-- A function to convert a three-digit number to its decimal representation -/
def threeDigitToNumber (c₁ c₂ b : ℕ) : ℕ := 100 * c₁ + 10 * c₂ + b

/-- Theorem stating that under given conditions, ccb must be 441 -/
theorem unique_square_number (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c →
  b = 1 →
  0 < a → a < 10 →
  0 ≤ c → c < 10 →
  (twoDigitToNumber a b)^2 = threeDigitToNumber c c b →
  threeDigitToNumber c c b > 300 →
  threeDigitToNumber c c b = 441 := by
sorry

end NUMINAMATH_CALUDE_unique_square_number_l791_79185


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l791_79102

theorem triangle_side_calculation (AB : ℝ) : 
  AB = 10 →
  ∃ (AC AD CD : ℝ),
    -- ABD is a 45-45-90 triangle
    AD = AB ∧
    -- ACD is a 30-60-90 triangle
    CD = 2 * AC ∧
    AD^2 = AC^2 + CD^2 ∧
    -- The result we want to prove
    CD = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l791_79102


namespace NUMINAMATH_CALUDE_combinations_with_repetition_l791_79121

/-- F_n^r represents the number of r-combinatorial selections from [1, n] with repetition allowed -/
def F (n : ℕ) (r : ℕ) : ℕ := sorry

/-- C_n^r represents the binomial coefficient (n choose r) -/
def C (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The theorem states that F_n^r equals C_(n+r-1)^r -/
theorem combinations_with_repetition (n : ℕ) (r : ℕ) : F n r = C (n + r - 1) r := by
  sorry

end NUMINAMATH_CALUDE_combinations_with_repetition_l791_79121


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l791_79120

/-- Represents a conic section --/
inductive ConicSection
| Parabola
| Circle
| Ellipse
| Hyperbola
| Point
| Line
| TwoLines
| Empty

/-- Determines if the given equation represents an ellipse --/
def is_ellipse (a b h k : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≠ b

/-- The equation of the conic section --/
def conic_equation (x y : ℝ) : Prop :=
  x^2 + 6*x + 9*y^2 - 36 = 0

/-- Theorem stating that the given equation represents an ellipse --/
theorem conic_is_ellipse : 
  ∃ (a b h k : ℝ), 
    (∀ (x y : ℝ), conic_equation x y ↔ ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)) ∧
    is_ellipse a b h k :=
sorry


end NUMINAMATH_CALUDE_conic_is_ellipse_l791_79120


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l791_79146

/-- A quadratic function with a real parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The condition that f has only one zero -/
def has_one_zero (a : ℝ) : Prop := ∃! x, f a x = 0

/-- The statement to be proved -/
theorem not_sufficient_nor_necessary :
  (∃ a, a ≤ -2 ∧ ¬(has_one_zero a)) ∧ 
  (∃ a, a > -2 ∧ has_one_zero a) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l791_79146


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l791_79165

theorem fly_distance_from_ceiling :
  ∀ (x y z : ℝ),
  x = 2 ∧ y = 6 ∧ x^2 + y^2 + z^2 = 10^2 →
  z = 2 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l791_79165


namespace NUMINAMATH_CALUDE_rectangle_roots_l791_79119

/-- The polynomial whose roots we are considering -/
def f (a : ℝ) (z : ℂ) : ℂ := z^4 - 8*z^3 + 13*a*z^2 - 3*(3*a^2 + 2*a - 4)*z + 1

/-- Predicate to check if four complex numbers form vertices of a rectangle -/
def isRectangle (z₁ z₂ z₃ z₄ : ℂ) : Prop := sorry

/-- The theorem stating that a = 3 is the only real value satisfying the condition -/
theorem rectangle_roots (a : ℝ) : 
  (∃ z₁ z₂ z₃ z₄ : ℂ, f a z₁ = 0 ∧ f a z₂ = 0 ∧ f a z₃ = 0 ∧ f a z₄ = 0 ∧ 
    isRectangle z₁ z₂ z₃ z₄) ↔ a = 3 := by sorry

end NUMINAMATH_CALUDE_rectangle_roots_l791_79119


namespace NUMINAMATH_CALUDE_sandals_sold_example_l791_79123

/-- Given a ratio of shoes to sandals and the number of shoes sold, 
    calculate the number of sandals sold. -/
def sandals_sold (shoe_ratio : ℕ) (sandal_ratio : ℕ) (shoes : ℕ) : ℕ :=
  (shoes / shoe_ratio) * sandal_ratio

/-- Theorem stating that given the specific ratio and number of shoes sold,
    the number of sandals sold is 40. -/
theorem sandals_sold_example : sandals_sold 9 5 72 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sandals_sold_example_l791_79123


namespace NUMINAMATH_CALUDE_bus_problem_l791_79113

theorem bus_problem (initial : ℕ) (got_off : ℕ) (final : ℕ) :
  initial = 36 →
  got_off = 68 →
  final = 12 →
  got_off - (initial - got_off + final) = 24 :=
by sorry

end NUMINAMATH_CALUDE_bus_problem_l791_79113


namespace NUMINAMATH_CALUDE_monomial_count_l791_79168

def is_monomial (expr : String) : Bool :=
  match expr with
  | "a" => true
  | "-2ab" => true
  | "x+y" => false
  | "x^2+y^2" => false
  | "-1" => true
  | "1/2ab^2c^3" => true
  | _ => false

def expressions : List String := ["a", "-2ab", "x+y", "x^2+y^2", "-1", "1/2ab^2c^3"]

theorem monomial_count :
  (expressions.filter is_monomial).length = 4 := by sorry

end NUMINAMATH_CALUDE_monomial_count_l791_79168


namespace NUMINAMATH_CALUDE_initial_adults_on_train_l791_79116

theorem initial_adults_on_train (adults_children_diff : ℕ)
  (adults_boarding : ℕ) (children_boarding : ℕ) (people_leaving : ℕ) (final_count : ℕ)
  (h1 : adults_children_diff = 17)
  (h2 : adults_boarding = 57)
  (h3 : children_boarding = 18)
  (h4 : people_leaving = 44)
  (h5 : final_count = 502) :
  ∃ (initial_adults initial_children : ℕ),
    initial_adults = initial_children + adults_children_diff ∧
    initial_adults + initial_children + adults_boarding + children_boarding - people_leaving = final_count ∧
    initial_adults = 244 := by
  sorry

end NUMINAMATH_CALUDE_initial_adults_on_train_l791_79116


namespace NUMINAMATH_CALUDE_markese_earnings_l791_79197

/-- Given Evan's earnings E, Markese's earnings (E - 5), and their total earnings of 37,
    prove that Markese earned 16 dollars. -/
theorem markese_earnings (E : ℕ) : E + (E - 5) = 37 → E - 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_markese_earnings_l791_79197


namespace NUMINAMATH_CALUDE_four_numbers_with_equal_sums_l791_79117

theorem four_numbers_with_equal_sums (A : Finset ℕ) 
  (h1 : A.card = 12)
  (h2 : ∀ x ∈ A, 1 ≤ x ∧ x ≤ 30)
  (h3 : A.card = Finset.card (Finset.image id A)) :
  ∃ a b c d : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b = c + d := by
  sorry


end NUMINAMATH_CALUDE_four_numbers_with_equal_sums_l791_79117


namespace NUMINAMATH_CALUDE_store_profit_l791_79192

/-- Prove that a store makes a profit when selling pens purchased from two markets -/
theorem store_profit (m n : ℝ) (h : m > n) : 
  let selling_price := (m + n) / 2
  let profit_A := 40 * (selling_price - m)
  let profit_B := 60 * (selling_price - n)
  profit_A + profit_B > 0 := by
  sorry


end NUMINAMATH_CALUDE_store_profit_l791_79192


namespace NUMINAMATH_CALUDE_line_equidistant_points_l791_79107

/-- Given a line passing through (4, 4) with slope 0.5, equidistant from (0, 2) and (A, 8), prove A = -3 -/
theorem line_equidistant_points (A : ℝ) : 
  let line_point : ℝ × ℝ := (4, 4)
  let line_slope : ℝ := 0.5
  let P : ℝ × ℝ := (0, 2)
  let Q : ℝ × ℝ := (A, 8)
  let midpoint : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let perpendicular_slope : ℝ := -1 / ((Q.2 - P.2) / (Q.1 - P.1))
  (line_slope = perpendicular_slope) ∧ 
  (midpoint.2 - line_point.2 = line_slope * (midpoint.1 - line_point.1)) →
  A = -3 := by
sorry

end NUMINAMATH_CALUDE_line_equidistant_points_l791_79107


namespace NUMINAMATH_CALUDE_largest_fraction_l791_79112

theorem largest_fraction : 
  let fractions : List ℚ := [2/3, 3/4, 2/5, 11/15]
  (3/4 : ℚ) = fractions.maximum := by sorry

end NUMINAMATH_CALUDE_largest_fraction_l791_79112


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l791_79172

/-- Given a coat with an original price and a price reduction, 
    calculate the percentage reduction in price. -/
theorem price_reduction_percentage 
  (original_price : ℝ) 
  (price_reduction : ℝ) 
  (h1 : original_price = 500)
  (h2 : price_reduction = 350) : 
  (price_reduction / original_price) * 100 = 70 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l791_79172


namespace NUMINAMATH_CALUDE_larger_cube_volume_l791_79111

-- Define the volume of a smaller cube
def small_cube_volume : ℝ := 8

-- Define the number of smaller cubes
def num_small_cubes : ℕ := 2

-- Theorem statement
theorem larger_cube_volume :
  ∀ (small_edge : ℝ) (large_edge : ℝ),
  small_edge > 0 →
  large_edge > 0 →
  small_edge^3 = small_cube_volume →
  num_small_cubes * small_edge = large_edge →
  large_edge^3 = 64 := by
sorry

end NUMINAMATH_CALUDE_larger_cube_volume_l791_79111


namespace NUMINAMATH_CALUDE_probability_of_drawing_red_ball_l791_79148

theorem probability_of_drawing_red_ball (white_balls red_balls : ℕ) 
  (h1 : white_balls = 5) (h2 : red_balls = 2) : 
  (red_balls : ℚ) / (white_balls + red_balls) = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_drawing_red_ball_l791_79148


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l791_79184

theorem binomial_expansion_example : 104^3 + 3*(104^2)*2 + 3*104*(2^2) + 2^3 = 106^3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l791_79184


namespace NUMINAMATH_CALUDE_tan_two_implies_expression_one_l791_79131

theorem tan_two_implies_expression_one (x : ℝ) (h : Real.tan x = 2) :
  4 * (Real.sin x)^2 - 3 * (Real.sin x) * (Real.cos x) - 5 * (Real.cos x)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_expression_one_l791_79131


namespace NUMINAMATH_CALUDE_rational_fraction_value_l791_79129

theorem rational_fraction_value (x y : ℝ) : 
  (x - y) / (x + y) = 4 → 
  ∃ (q : ℚ), x / y = ↑q →
  x / y = -5/3 := by
sorry

end NUMINAMATH_CALUDE_rational_fraction_value_l791_79129


namespace NUMINAMATH_CALUDE_angle_C_measure_l791_79179

structure Quadrilateral where
  A : Real
  B : Real
  C : Real
  D : Real
  sum_angles : A + B + C + D = 360

def adjacent_angle_ratio (q : Quadrilateral) : Prop :=
  q.A / q.B = 2 / 7

theorem angle_C_measure (q : Quadrilateral) (h : adjacent_angle_ratio q) : q.C = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l791_79179


namespace NUMINAMATH_CALUDE_train_crossing_time_l791_79162

/-- Proves that a train of given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 450 ∧ train_speed_kmh = 180 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l791_79162


namespace NUMINAMATH_CALUDE_exam_score_proof_l791_79193

/-- Proves that the average score of students who took the exam on the assigned day is 60% -/
theorem exam_score_proof (total_students : ℕ) (assigned_day_percentage : ℝ) 
  (makeup_score : ℝ) (class_average : ℝ) : 
  total_students = 100 →
  assigned_day_percentage = 0.7 →
  makeup_score = 90 →
  class_average = 69 →
  let assigned_students := total_students * assigned_day_percentage
  let makeup_students := total_students - assigned_students
  let assigned_score := (class_average * total_students - makeup_score * makeup_students) / assigned_students
  assigned_score = 60 := by
sorry


end NUMINAMATH_CALUDE_exam_score_proof_l791_79193


namespace NUMINAMATH_CALUDE_hexagon_area_sum_l791_79177

-- Define the hexagon structure
structure Hexagon :=
  (sideLength : ℝ)
  (numSegments : ℕ)

-- Define the theorem
theorem hexagon_area_sum (h : Hexagon) (a b : ℕ) : 
  h.sideLength = 3 ∧ h.numSegments = 12 →
  ∃ (area : ℝ), area = a * Real.sqrt b ∧ a + b = 30 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_area_sum_l791_79177


namespace NUMINAMATH_CALUDE_characterization_of_special_numbers_l791_79134

/-- A function that checks if a real number has only two distinct non-zero digits, one of which is 3 -/
def hasTwoDistinctNonZeroDigitsWithThree (N : ℝ) : Prop := sorry

/-- A function that checks if a real number is a perfect square -/
def isPerfectSquare (N : ℝ) : Prop := sorry

/-- Theorem stating the characterization of numbers satisfying the given conditions -/
theorem characterization_of_special_numbers (N : ℝ) : 
  (hasTwoDistinctNonZeroDigitsWithThree N ∧ isPerfectSquare N) ↔ 
  ∃ n : ℕ, N = 36 * (100 : ℝ) ^ n :=
sorry

end NUMINAMATH_CALUDE_characterization_of_special_numbers_l791_79134


namespace NUMINAMATH_CALUDE_box_volume_problem_l791_79141

theorem box_volume_problem :
  ∃! (x : ℕ), 
    x > 3 ∧ 
    (x + 3) * (x - 3) * (x^2 + 9) < 500 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_problem_l791_79141


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l791_79158

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l791_79158


namespace NUMINAMATH_CALUDE_max_profit_at_nine_profit_function_correct_max_profit_at_nine_explicit_l791_79163

-- Define the profit function
def profit (x : ℝ) : ℝ := x^3 - 30*x^2 + 288*x - 864

-- Define the theorem
theorem max_profit_at_nine :
  ∀ x ∈ Set.Icc 9 11,
    profit x ≤ profit 9 ∧
    profit 9 = 27 := by
  sorry

-- Define the selling price range
def selling_price_range : Set ℝ := Set.Icc 9 11

-- Define the annual sales volume function
def annual_sales (x : ℝ) : ℝ := (12 - x)^2

-- State that the profit function is correct
theorem profit_function_correct :
  ∀ x ∈ selling_price_range,
    profit x = (x - 6) * annual_sales x := by
  sorry

-- State that the maximum profit occurs at x = 9
theorem max_profit_at_nine_explicit :
  ∃ x ∈ selling_price_range,
    ∀ y ∈ selling_price_range,
      profit y ≤ profit x ∧
      x = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_nine_profit_function_correct_max_profit_at_nine_explicit_l791_79163


namespace NUMINAMATH_CALUDE_eighth_roll_last_probability_l791_79139

/-- The probability of the 8th roll being the last roll when rolling a standard six-sided die 
    until getting the same number on consecutive rolls -/
def prob_eighth_roll_last : ℚ := (5^6 : ℚ) / (6^7 : ℚ)

/-- The number of sides on a standard die -/
def standard_die_sides : ℕ := 6

/-- Theorem stating that the probability of the 8th roll being the last roll is correct -/
theorem eighth_roll_last_probability : 
  prob_eighth_roll_last = (5^6 : ℚ) / (6^7 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_eighth_roll_last_probability_l791_79139


namespace NUMINAMATH_CALUDE_parabola_transformation_l791_79190

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
    b := p.b - 2 * p.a * h
    c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a
    b := p.b
    c := p.c + v }

/-- The original parabola y = x^2 + 2 -/
def original_parabola : Parabola :=
  { a := 1
    b := 0
    c := 2 }

theorem parabola_transformation :
  let p1 := shift_horizontal original_parabola (-1)
  let p2 := shift_vertical p1 (-1)
  p2 = { a := 1, b := 2, c := 1 } :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l791_79190


namespace NUMINAMATH_CALUDE_max_take_home_pay_l791_79110

/-- The income that maximizes take-home pay given a specific tax rate and fee structure -/
theorem max_take_home_pay :
  let tax_rate (x : ℝ) := 2 * x / 100
  let admin_fee := 500
  let take_home_pay (x : ℝ) := 1000 * x - (tax_rate x * 1000 * x) - admin_fee
  ∃ (x : ℝ), ∀ (y : ℝ), take_home_pay x ≥ take_home_pay y ∧ x = 25 := by
sorry

end NUMINAMATH_CALUDE_max_take_home_pay_l791_79110


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l791_79183

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  1/a + 1/b + 1/c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l791_79183


namespace NUMINAMATH_CALUDE_existence_of_solutions_l791_79137

theorem existence_of_solutions (k : ℕ) (a : ℕ) (n : Fin k → ℕ) 
  (h1 : ∀ i, a > 0 ∧ n i > 0)
  (h2 : ∀ i j, i ≠ j → Nat.gcd (n i) (n j) = 1)
  (h3 : ∀ i, a ^ (n i) % (n i) = 1)
  (h4 : ∀ i, ¬(n i ∣ a - 1)) :
  ∃ (S : Finset ℕ), S.card ≥ 2^(k+1) - 2 ∧ 
    (∀ x ∈ S, x > 1 ∧ a^x % x = 1) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_solutions_l791_79137


namespace NUMINAMATH_CALUDE_graph_transformation_l791_79169

-- Define the original function f
variable (f : ℝ → ℝ)

-- Define the symmetry operation with respect to x = 1
def symmetry_x1 (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (2 - x)

-- Define the left shift operation
def shift_left (g : ℝ → ℝ) (units : ℝ) : ℝ → ℝ := λ x => g (x + units)

-- Theorem statement
theorem graph_transformation (f : ℝ → ℝ) :
  shift_left (symmetry_x1 f) 2 = λ x => f (1 - x) := by sorry

end NUMINAMATH_CALUDE_graph_transformation_l791_79169


namespace NUMINAMATH_CALUDE_sum_of_xy_l791_79124

theorem sum_of_xy (x y : ℕ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x < 30) 
  (h4 : y < 30) 
  (h5 : x + y + x * y = 119) : 
  x + y = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xy_l791_79124


namespace NUMINAMATH_CALUDE_min_value_of_function_l791_79105

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x + 1 / (x + 1) ≥ 1 ∧ ∃ y > -1, y + 1 / (y + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l791_79105


namespace NUMINAMATH_CALUDE_inequality_properties_l791_79101

theorem inequality_properties (a b c : ℝ) (h : a < b) :
  (a + c < b + c) ∧
  (a - 2 < b - 2) ∧
  (2 * a < 2 * b) ∧
  (-3 * a > -3 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l791_79101


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l791_79132

-- Define a pentagon as a polygon with 5 sides
def Pentagon : Nat := 5

-- Theorem stating that the sum of interior angles of a pentagon is 540 degrees
theorem sum_interior_angles_pentagon :
  (Pentagon - 2) * 180 = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l791_79132


namespace NUMINAMATH_CALUDE_min_triangle_area_l791_79114

theorem min_triangle_area (p q : ℤ) : ∃ (min_area : ℚ), 
  min_area = 1 ∧ 
  ∀ (area : ℚ), area = (1 : ℚ) / 2 * |10 * p - 24 * q| → area ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l791_79114


namespace NUMINAMATH_CALUDE_construction_team_problem_l791_79159

/-- Represents the possible solutions for the original number of people in the second group -/
inductive Solution : Type
  | fiftySeven : Solution
  | twentyOne : Solution

/-- Checks if a given number satisfies the conditions of the problem -/
def satisfiesConditions (x : ℕ) : Prop :=
  ∃ (k : ℕ+), 96 - 16 = k * (x + 16) + 6

/-- The theorem stating that the only solutions are 58 and 21 -/
theorem construction_team_problem :
  ∀ x : ℕ, satisfiesConditions x ↔ (x = 58 ∨ x = 21) :=
sorry

end NUMINAMATH_CALUDE_construction_team_problem_l791_79159


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l791_79135

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def units_digit (n : ℕ) : ℕ := n % 10

def first_two_digits (n : ℕ) : ℕ := n / 100

def first_last_digits (n : ℕ) : ℕ := (n / 1000) * 10 + (n % 10)

theorem unique_four_digit_number :
  ∃! n : ℕ, is_four_digit n ∧
            ¬(n % 7 = 0) ∧
            tens_digit n = hundreds_digit n + units_digit n ∧
            first_two_digits n = 15 * units_digit n ∧
            Nat.Prime (first_last_digits n) := by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l791_79135


namespace NUMINAMATH_CALUDE_sophias_book_length_l791_79186

theorem sophias_book_length (P : ℕ) : 
  (2 : ℚ) / 3 * P = (1 : ℚ) / 3 * P + 90 → P = 270 := by
  sorry

end NUMINAMATH_CALUDE_sophias_book_length_l791_79186


namespace NUMINAMATH_CALUDE_rectangle_in_circle_l791_79180

theorem rectangle_in_circle (d p : ℝ) (h_d_pos : d > 0) (h_p_pos : p > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≥ y ∧
  (2 * x + 2 * y = p) ∧  -- perimeter condition
  (x^2 + y^2 = d^2) ∧    -- inscribed in circle condition
  (x - y = d) :=
sorry

end NUMINAMATH_CALUDE_rectangle_in_circle_l791_79180


namespace NUMINAMATH_CALUDE_hyperbola_n_range_l791_79100

def is_hyperbola (m n : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

def foci_distance (m : ℝ) : ℝ := 4

theorem hyperbola_n_range (m n : ℝ) 
  (h1 : is_hyperbola m n) 
  (h2 : foci_distance m = 4) : 
  -1 < n ∧ n < 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_n_range_l791_79100


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l791_79108

theorem power_fraction_simplification :
  (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l791_79108


namespace NUMINAMATH_CALUDE_circle_not_in_second_quadrant_l791_79181

/-- A circle in the xy-plane with center (a, 0) and radius 2 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = 4}

/-- The second quadrant of the xy-plane -/
def SecondQuadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}

/-- The circle does not pass through the second quadrant -/
def NotInSecondQuadrant (a : ℝ) : Prop :=
  Circle a ∩ SecondQuadrant = ∅

theorem circle_not_in_second_quadrant (a : ℝ) :
  NotInSecondQuadrant a → a ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_circle_not_in_second_quadrant_l791_79181


namespace NUMINAMATH_CALUDE_zoo_visit_l791_79130

/-- The number of children who saw giraffes but not pandas -/
def giraffes_not_pandas (total children_pandas children_giraffes pandas_not_giraffes : ℕ) : ℕ :=
  children_giraffes - (children_pandas - pandas_not_giraffes)

/-- Theorem stating the number of children who saw giraffes but not pandas -/
theorem zoo_visit (total children_pandas children_giraffes pandas_not_giraffes : ℕ) 
  (h1 : total = 50)
  (h2 : children_pandas = 36)
  (h3 : children_giraffes = 28)
  (h4 : pandas_not_giraffes = 15) :
  giraffes_not_pandas total children_pandas children_giraffes pandas_not_giraffes = 7 := by
  sorry


end NUMINAMATH_CALUDE_zoo_visit_l791_79130


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l791_79178

theorem quadratic_complete_square :
  ∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l791_79178


namespace NUMINAMATH_CALUDE_geometric_sequence_terms_l791_79175

theorem geometric_sequence_terms (a : ℕ → ℝ) (n : ℕ) (S_n : ℝ) : 
  (∀ k, a (k + 1) / a k = a 2 / a 1) →  -- geometric sequence condition
  a 1 + a n = 82 →
  a 3 * a (n - 2) = 81 →
  S_n = 121 →
  (∀ k, S_k = (a 1 * (1 - (a 2 / a 1)^k)) / (1 - (a 2 / a 1))) →  -- sum formula for geometric sequence
  n = 5 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_terms_l791_79175


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l791_79127

/-- Given that the solution set of (ax+1)/(x+b) > 1 is (-∞, -1) ∪ (3, +∞),
    prove that the solution set of x^2 + ax - 2b < 0 is (-3, -2) -/
theorem solution_set_equivalence (a b : ℝ) :
  ({x : ℝ | (a * x + 1) / (x + b) > 1} = {x : ℝ | x < -1 ∨ x > 3}) →
  {x : ℝ | x^2 + a*x - 2*b < 0} = {x : ℝ | -3 < x ∧ x < -2} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l791_79127


namespace NUMINAMATH_CALUDE_sample_size_definition_l791_79187

/-- Represents a population of students' exam scores -/
structure Population where
  scores : Set ℝ

/-- Represents a sample drawn from a population -/
structure Sample where
  elements : Finset ℝ

/-- Simple random sampling function -/
def simpleRandomSampling (pop : Population) (n : ℕ) : Sample :=
  sorry

theorem sample_size_definition 
  (pop : Population) 
  (sample : Sample) 
  (n : ℕ) 
  (h1 : sample = simpleRandomSampling pop n) 
  (h2 : n = 100) : 
  n = Finset.card sample.elements :=
sorry

end NUMINAMATH_CALUDE_sample_size_definition_l791_79187


namespace NUMINAMATH_CALUDE_xyz_inequality_l791_79143

theorem xyz_inequality (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ 
  x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by sorry

end NUMINAMATH_CALUDE_xyz_inequality_l791_79143


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l791_79109

theorem complex_number_quadrant (z : ℂ) (h : (2 - I) * z = 5) :
  0 < z.re ∧ 0 < z.im := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l791_79109


namespace NUMINAMATH_CALUDE_quadratic_solution_implication_l791_79140

theorem quadratic_solution_implication (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0 → 4 * a + 8 * b = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implication_l791_79140


namespace NUMINAMATH_CALUDE_sector_area_from_arc_length_l791_79157

/-- Given a circle where the arc length corresponding to a central angle of 2 radians is 4 cm,
    prove that the area of the sector formed by this central angle is 4 cm². -/
theorem sector_area_from_arc_length (r : ℝ) : 
  r * 2 = 4 → (1 / 2) * r^2 * 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_from_arc_length_l791_79157


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l791_79149

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l791_79149


namespace NUMINAMATH_CALUDE_square_area_proof_l791_79152

theorem square_area_proof (x : ℝ) : 
  (3 * x - 12 = 15 - 2 * x) → 
  ((3 * x - 12)^2 : ℝ) = 441 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l791_79152


namespace NUMINAMATH_CALUDE_product_of_one_plus_tans_l791_79156

theorem product_of_one_plus_tans (α β : Real) (h : α + β = π / 4) :
  (1 + Real.tan α) * (1 + Real.tan β) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_tans_l791_79156


namespace NUMINAMATH_CALUDE_exam_scores_difference_l791_79106

theorem exam_scores_difference (score1 score2 : ℕ) : 
  score1 = 42 →
  score2 = 33 →
  score1 = (56 * (score1 + score2)) / 100 →
  score1 - score2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_scores_difference_l791_79106


namespace NUMINAMATH_CALUDE_solution_set_implies_k_value_l791_79154

theorem solution_set_implies_k_value (k : ℝ) : 
  (∀ x : ℝ, |k * x - 4| ≤ 2 ↔ 1 ≤ x ∧ x ≤ 3) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_k_value_l791_79154


namespace NUMINAMATH_CALUDE_solution_set_part_i_min_pq_part_ii_l791_79144

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x - 3|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x ≥ 4} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := by sorry

-- Part II
theorem min_pq_part_ii (m p q : ℝ) (hm : m > 0) (hp : p > 0) (hq : q > 0) :
  (∀ x, f m x ≥ 3) ∧ (∃ x, f m x = 3) ∧ (1/p + 1/(2*q) = m) →
  ∀ r s, r > 0 ∧ s > 0 ∧ 1/r + 1/(2*s) = m → p*q ≤ r*s ∧ p*q = 1/18 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_min_pq_part_ii_l791_79144
