import Mathlib

namespace probability_three_green_marbles_l3507_350708

/-- The probability of choosing exactly k successes in n trials with probability p of success in each trial. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of green marbles -/
def green_marbles : ℕ := 8

/-- The number of purple marbles -/
def purple_marbles : ℕ := 7

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of trials -/
def num_trials : ℕ := 7

/-- The number of desired green marbles -/
def desired_green : ℕ := 3

/-- The probability of choosing a green marble in one trial -/
def prob_green : ℚ := green_marbles / total_marbles

theorem probability_three_green_marbles :
  binomial_probability num_trials desired_green prob_green = 8604112 / 15946875 := by
  sorry

end probability_three_green_marbles_l3507_350708


namespace equal_intercept_line_proof_l3507_350737

/-- A line with equal intercepts on both axes passing through point (3,2) -/
def equal_intercept_line (x y : ℝ) : Prop :=
  x + y = 5

theorem equal_intercept_line_proof :
  -- The line passes through point (3,2)
  equal_intercept_line 3 2 ∧
  -- The line has equal intercepts on both axes
  ∃ a : ℝ, a ≠ 0 ∧ equal_intercept_line a 0 ∧ equal_intercept_line 0 a :=
by
  sorry

#check equal_intercept_line_proof

end equal_intercept_line_proof_l3507_350737


namespace expression_simplification_l3507_350759

theorem expression_simplification (x : ℤ) (h1 : -2 < x) (h2 : x ≤ 2) (h3 : x = 2) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((2 / (x - 1)) - (1 / x)) = 4 := by
  sorry

end expression_simplification_l3507_350759


namespace email_count_proof_l3507_350764

/-- Calculates the total number of emails received in a month with changing email rates -/
def total_emails (days_in_month : ℕ) (initial_rate : ℕ) (new_rate : ℕ) : ℕ :=
  let half_month := days_in_month / 2
  let first_half := initial_rate * half_month
  let second_half := new_rate * half_month
  first_half + second_half

/-- Proves that given the specified conditions, the total number of emails is 675 -/
theorem email_count_proof :
  let days_in_month : ℕ := 30
  let initial_rate : ℕ := 20
  let new_rate : ℕ := 25
  total_emails days_in_month initial_rate new_rate = 675 := by
  sorry


end email_count_proof_l3507_350764


namespace election_votes_l3507_350732

theorem election_votes (candidate1_percentage : ℚ) (candidate2_votes : ℕ) :
  candidate1_percentage = 60 / 100 →
  candidate2_votes = 240 →
  ∃ total_votes : ℕ,
    candidate1_percentage * total_votes = total_votes - candidate2_votes ∧
    total_votes = 600 :=
by sorry

end election_votes_l3507_350732


namespace single_elimination_tournament_games_tournament_with_23_teams_l3507_350730

/-- In a single-elimination tournament, the number of games played is one less than the number of teams. -/
theorem single_elimination_tournament_games (n : ℕ) (n_pos : n > 0) :
  let teams := n
  let games := n - 1
  games = teams - 1 := by sorry

/-- For a tournament with 23 teams, 22 games are played. -/
theorem tournament_with_23_teams :
  let teams := 23
  let games := teams - 1
  games = 22 := by sorry

end single_elimination_tournament_games_tournament_with_23_teams_l3507_350730


namespace fraction_evaluation_l3507_350731

theorem fraction_evaluation : (15 - 3^2) / 3 = 2 := by
  sorry

end fraction_evaluation_l3507_350731


namespace bookshelf_length_is_24_l3507_350722

/-- The length of one span in centimeters -/
def span_length : ℝ := 12

/-- The number of spans in the shorter side of the bookshelf -/
def bookshelf_spans : ℝ := 2

/-- The length of the shorter side of the bookshelf in centimeters -/
def bookshelf_length : ℝ := span_length * bookshelf_spans

theorem bookshelf_length_is_24 : bookshelf_length = 24 := by
  sorry

end bookshelf_length_is_24_l3507_350722


namespace smaller_root_of_equation_l3507_350758

theorem smaller_root_of_equation : 
  ∃ (x : ℚ), (x - 5/6)^2 + (x - 5/6)*(x - 2/3) = 0 ∧ 
  x = 5/6 ∧ 
  ∀ y, ((y - 5/6)^2 + (y - 5/6)*(y - 2/3) = 0 → y ≥ 5/6) :=
by sorry

end smaller_root_of_equation_l3507_350758


namespace inscribed_rectangle_length_l3507_350721

/-- Right triangle PQR with inscribed rectangle ABCD -/
structure InscribedRectangle where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side QR -/
  qr : ℝ
  /-- Length of side PR -/
  pr : ℝ
  /-- Length of rectangle ABCD (parallel to PR) -/
  length : ℝ
  /-- Height of rectangle ABCD (parallel to PQ) -/
  height : ℝ
  /-- PQR is a right triangle -/
  is_right_triangle : pq ^ 2 + qr ^ 2 = pr ^ 2
  /-- Height is half the length -/
  height_half_length : height = length / 2
  /-- Rectangle fits in triangle -/
  fits_in_triangle : height ≤ pq ∧ length ≤ pr ∧ (pr - length) / (qr - height) = height / pq

/-- The length of the inscribed rectangle is 7.5 -/
theorem inscribed_rectangle_length (rect : InscribedRectangle) 
  (h_pq : rect.pq = 5) (h_qr : rect.qr = 12) (h_pr : rect.pr = 13) : 
  rect.length = 7.5 := by
  sorry

end inscribed_rectangle_length_l3507_350721


namespace representations_non_negative_representations_natural_l3507_350775

/-- The number of ways to represent a natural number as a sum of non-negative integers -/
def representationsNonNegative (n m : ℕ) : ℕ := Nat.choose (n + m - 1) n

/-- The number of ways to represent a natural number as a sum of natural numbers -/
def representationsNatural (n m : ℕ) : ℕ := Nat.choose (n - 1) (n - m)

/-- Theorem stating the number of ways to represent n as a sum of m non-negative integers -/
theorem representations_non_negative (n m : ℕ) :
  representationsNonNegative n m = Nat.choose (n + m - 1) n := by sorry

/-- Theorem stating the number of ways to represent n as a sum of m natural numbers -/
theorem representations_natural (n m : ℕ) (h : m ≤ n) :
  representationsNatural n m = Nat.choose (n - 1) (n - m) := by sorry

end representations_non_negative_representations_natural_l3507_350775


namespace bacteria_in_seventh_generation_l3507_350773

/-- The number of bacteria in a given generation -/
def bacteria_count (generation : ℕ) : ℕ :=
  match generation with
  | 0 => 1  -- First generation
  | n + 1 => 4 * bacteria_count n  -- Subsequent generations

/-- Theorem stating the number of bacteria in the seventh generation -/
theorem bacteria_in_seventh_generation :
  bacteria_count 6 = 4096 := by
  sorry

end bacteria_in_seventh_generation_l3507_350773


namespace inequality_solution_l3507_350725

theorem inequality_solution (a : ℝ) :
  4 ≤ a / (3 * a - 6) ∧ a / (3 * a - 6) > 12 → a < 72 / 35 :=
by sorry

end inequality_solution_l3507_350725


namespace common_roots_product_sum_of_a_b_c_l3507_350707

/-- Given two cubic equations with two common roots, prove that the product of these common roots is 10∛4 -/
theorem common_roots_product (C : ℝ) : 
  ∃ (u v w t : ℝ),
    (u^3 - 5*u + 20 = 0) ∧ 
    (v^3 - 5*v + 20 = 0) ∧ 
    (w^3 - 5*w + 20 = 0) ∧
    (u^3 + C*u^2 + 80 = 0) ∧ 
    (v^3 + C*v^2 + 80 = 0) ∧ 
    (t^3 + C*t^2 + 80 = 0) ∧
    (u ≠ v) ∧ (u ≠ w) ∧ (v ≠ w) ∧
    (u ≠ t) ∧ (v ≠ t) →
    u * v = 10 * Real.rpow 4 (1/3) :=
by sorry

/-- The sum of a, b, and c in the form a∛b where a=10, b=3, and c=4 is 17 -/
theorem sum_of_a_b_c : 10 + 3 + 4 = 17 :=
by sorry

end common_roots_product_sum_of_a_b_c_l3507_350707


namespace intersection_point_implies_sum_of_intercepts_l3507_350706

/-- Given two lines that intersect at (2,3), prove their y-intercepts sum to 10/3 -/
theorem intersection_point_implies_sum_of_intercepts :
  ∀ (a b : ℚ),
  (2 : ℚ) = (1/3 : ℚ) * 3 + a →
  (3 : ℚ) = (1/3 : ℚ) * 2 + b →
  a + b = 10/3 := by
  sorry

end intersection_point_implies_sum_of_intercepts_l3507_350706


namespace certain_number_is_100_l3507_350717

theorem certain_number_is_100 : ∃! x : ℝ, ((x / 4) + 25) * 3 = 150 := by
  sorry

end certain_number_is_100_l3507_350717


namespace joshua_toy_cars_l3507_350798

theorem joshua_toy_cars : 
  ∀ (box1 box2 box3 box4 box5 : ℕ),
    box1 = 21 →
    box2 = 31 →
    box3 = 19 →
    box4 = 45 →
    box5 = 27 →
    box1 + box2 + box3 + box4 + box5 = 143 :=
by
  sorry

end joshua_toy_cars_l3507_350798


namespace partner_calculation_l3507_350787

theorem partner_calculation (x : ℝ) : 3 * (3 * (x + 2) - 2) = 3 * (3 * x + 4) := by
  sorry

#check partner_calculation

end partner_calculation_l3507_350787


namespace goldfish_equality_month_l3507_350712

theorem goldfish_equality_month : ∃ n : ℕ, n > 0 ∧ 3^(n+1) = 96 * 2^n ∧ ∀ m : ℕ, m > 0 ∧ m < n → 3^(m+1) ≠ 96 * 2^m :=
by
  -- The proof goes here
  sorry

end goldfish_equality_month_l3507_350712


namespace chessboard_clearable_l3507_350789

/-- Represents the number of chips on a chessboard -/
def Chessboard := Fin 8 → Fin 8 → ℕ

/-- Represents an operation on the chessboard -/
inductive Operation
  | remove_column : Fin 8 → Operation
  | double_row : Fin 8 → Operation

/-- Applies an operation to a chessboard -/
def apply_operation (board : Chessboard) (op : Operation) : Chessboard :=
  match op with
  | Operation.remove_column j => fun i k => if k = j then (board i k).pred else board i k
  | Operation.double_row i => fun k j => if k = i then 2 * (board k j) else board k j

/-- Checks if the board is cleared (all cells are zero) -/
def is_cleared (board : Chessboard) : Prop :=
  ∀ i j, board i j = 0

theorem chessboard_clearable (initial_board : Chessboard) :
  ∃ (ops : List Operation), is_cleared (ops.foldl apply_operation initial_board) :=
sorry

end chessboard_clearable_l3507_350789


namespace monotonic_increasing_sine_cosine_function_l3507_350783

theorem monotonic_increasing_sine_cosine_function (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (π / 4), Monotone (fun x => a * Real.sin x + Real.cos x)) ↔ a ≥ 1 := by
  sorry

end monotonic_increasing_sine_cosine_function_l3507_350783


namespace pages_copied_for_30_dollars_l3507_350710

/-- The number of pages that can be copied for a given amount of money -/
def pages_copied (cost_per_2_pages : ℚ) (amount : ℚ) : ℚ :=
  (amount / cost_per_2_pages) * 2

/-- Theorem: Given that it costs 4 cents to copy 2 pages, 
    the number of pages that can be copied for $30 is 1500 -/
theorem pages_copied_for_30_dollars : 
  pages_copied (4/100) 30 = 1500 := by
  sorry

#eval pages_copied (4/100) 30

end pages_copied_for_30_dollars_l3507_350710


namespace ellipse_hyperbola_tangent_l3507_350784

/-- An ellipse and a hyperbola are tangent if and only if m = 8/9 -/
theorem ellipse_hyperbola_tangent (m : ℝ) : 
  (∃ x y : ℝ, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 1 ∧ 
   ∀ x' y' : ℝ, x'^2 + 9*y'^2 = 9 ∧ x'^2 - m*(y'+3)^2 = 1 → (x', y') = (x, y)) ↔ 
  m = 8/9 := by
sorry

end ellipse_hyperbola_tangent_l3507_350784


namespace valid_coloring_iff_odd_l3507_350796

/-- A coloring of edges and diagonals of an n-gon -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin n

/-- Predicate for a valid coloring -/
def is_valid_coloring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ i →
    ∃ (x y z : Fin n), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
      c x y = i ∧ c y z = j ∧ c z x = k

/-- Theorem: A valid coloring exists if and only if n is odd -/
theorem valid_coloring_iff_odd (n : ℕ) :
  (∃ c : Coloring n, is_valid_coloring n c) ↔ Odd n :=
sorry

end valid_coloring_iff_odd_l3507_350796


namespace remainder_double_n_l3507_350715

theorem remainder_double_n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end remainder_double_n_l3507_350715


namespace quadratic_inequality_condition_l3507_350772

theorem quadratic_inequality_condition (x : ℝ) : x^2 - 2*x - 3 < 0 ↔ -1 < x ∧ x < 3 := by
  sorry

end quadratic_inequality_condition_l3507_350772


namespace room_population_problem_l3507_350727

theorem room_population_problem (initial_men initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →  -- Initial ratio of men to women is 4:5
  initial_men + 2 = 14 →  -- Final number of men is 14
  (2 * (initial_women - 3) = 24) →  -- Final number of women is 24
  True :=
by sorry

end room_population_problem_l3507_350727


namespace triangular_bipyramid_existence_condition_l3507_350782

/-- A triangular bipyramid with four edges of length 1 and two edges of length x -/
structure TriangularBipyramid (x : ℝ) :=
  (edge_length_1 : ℝ := 1)
  (edge_length_x : ℝ := x)
  (num_edges_1 : ℕ := 4)
  (num_edges_x : ℕ := 2)

/-- The existence condition for a triangular bipyramid -/
def exists_triangular_bipyramid (x : ℝ) : Prop :=
  0 < x ∧ x < (Real.sqrt 6 + Real.sqrt 2) / 2

/-- Theorem stating the range of x for which a triangular bipyramid can exist -/
theorem triangular_bipyramid_existence_condition (x : ℝ) :
  (∃ t : TriangularBipyramid x, True) ↔ exists_triangular_bipyramid x :=
sorry

end triangular_bipyramid_existence_condition_l3507_350782


namespace light_reflection_theorem_l3507_350714

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Reflects a point across a line -/
def reflectPoint (p : Point) (l : Line) : Point :=
  sorry

/-- Constructs a line passing through two points -/
def lineThrough (p1 p2 : Point) : Line :=
  sorry

theorem light_reflection_theorem (P A : Point) (mirror : Line) :
  P = Point.mk 2 3 →
  A = Point.mk 1 1 →
  mirror = Line.mk 1 1 1 →
  let Q := reflectPoint P mirror
  let incidentRay := lineThrough P (Point.mk mirror.a mirror.b)
  let reflectedRay := lineThrough Q A
  incidentRay = Line.mk 2 (-1) (-1) ∧
  reflectedRay = Line.mk 4 (-5) 1 :=
sorry

end light_reflection_theorem_l3507_350714


namespace chocolate_bars_count_l3507_350720

theorem chocolate_bars_count (large_box small_boxes chocolate_bars_per_small_box : ℕ) 
  (h1 : small_boxes = 21)
  (h2 : chocolate_bars_per_small_box = 25)
  (h3 : large_box = small_boxes * chocolate_bars_per_small_box) :
  large_box = 525 := by
  sorry

end chocolate_bars_count_l3507_350720


namespace arithmetic_sequence_seventh_term_l3507_350760

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℚ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_first : a 1 = 7/9)
  (h_thirteenth : a 13 = 4/5) :
  a 7 = 71/90 := by
sorry

end arithmetic_sequence_seventh_term_l3507_350760


namespace sum_of_x_values_l3507_350746

theorem sum_of_x_values (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, Real.sqrt ((x - 2)^2) = 8 ↔ x = x₁ ∨ x = x₂) ∧ x₁ + x₂ = 4) :=
by sorry

end sum_of_x_values_l3507_350746


namespace perpendicular_tangents_imply_a_value_l3507_350766

/-- Given two curves C₁ and C₂, prove that if their tangent lines are perpendicular at x = 1, 
    then the parameter a of C₁ must equal -1 / (3e) -/
theorem perpendicular_tangents_imply_a_value (a : ℝ) :
  let C₁ : ℝ → ℝ := λ x => a * x^3 - x^2 + 2 * x
  let C₂ : ℝ → ℝ := λ x => Real.exp x
  let C₁' : ℝ → ℝ := λ x => 3 * a * x^2 - 2 * x + 2
  let C₂' : ℝ → ℝ := λ x => Real.exp x
  (C₁' 1 * C₂' 1 = -1) → a = -1 / (3 * Real.exp 1) := by
sorry

end perpendicular_tangents_imply_a_value_l3507_350766


namespace angle_bisector_inequality_l3507_350763

/-- A triangle with sides a, b, c and angle bisectors fa, fb, fc -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  fa : ℝ
  fb : ℝ
  fc : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  pos_fa : fa > 0
  pos_fb : fb > 0
  pos_fc : fc > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The inequality holds for any triangle -/
theorem angle_bisector_inequality (t : Triangle) :
  1 / t.fa + 1 / t.fb + 1 / t.fc > 1 / t.a + 1 / t.b + 1 / t.c := by
  sorry

end angle_bisector_inequality_l3507_350763


namespace charlotte_boots_cost_l3507_350788

/-- Calculates the amount Charlotte needs to bring to buy discounted boots -/
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price - (discount_rate * original_price)

/-- Proves that Charlotte needs to bring $72 for the boots -/
theorem charlotte_boots_cost : discounted_price 90 0.2 = 72 := by
  sorry

end charlotte_boots_cost_l3507_350788


namespace sum_always_positive_l3507_350754

/-- A monotonically increasing odd function -/
def MonoIncreasingOdd (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x)

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h1 : MonoIncreasingOdd f)
  (h2 : ArithmeticSequence a)
  (h3 : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 := by
  sorry

end sum_always_positive_l3507_350754


namespace combined_height_theorem_l3507_350761

/-- The conversion factor from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- Maria's height in inches -/
def maria_height_inches : ℝ := 54

/-- Ben's height in inches -/
def ben_height_inches : ℝ := 72

/-- Combined height in centimeters -/
def combined_height_cm : ℝ := (maria_height_inches + ben_height_inches) * inch_to_cm

theorem combined_height_theorem :
  combined_height_cm = 320.04 := by sorry

end combined_height_theorem_l3507_350761


namespace impossible_d_greater_than_c_l3507_350794

/-- A decreasing function on positive reals -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f y < f x

theorem impossible_d_greater_than_c
  (f : ℝ → ℝ) (a b c d : ℝ)
  (h_dec : DecreasingFunction f)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a < b ∧ b < c)
  (h_prod : f a * f b * f c < 0)
  (h_d : f d = 0) :
  ¬(d > c) := by
sorry

end impossible_d_greater_than_c_l3507_350794


namespace eighteen_men_handshakes_l3507_350729

/-- The maximum number of handshakes without cyclic handshakes for n men -/
def maxHandshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 18 men, the maximum number of handshakes without cyclic handshakes is 153 -/
theorem eighteen_men_handshakes :
  maxHandshakes 18 = 153 := by
  sorry

end eighteen_men_handshakes_l3507_350729


namespace parabola_focus_distance_l3507_350771

/-- For a parabola y² = 2px, if the distance from (4, 0) to the focus (p/2, 0) is 5, then p = 8 -/
theorem parabola_focus_distance (p : ℝ) : 
  (∀ y : ℝ, y^2 = 2*p*4) → -- point (4, y) is on the parabola
  ((4 - p/2)^2 + 0^2)^(1/2) = 5 → -- distance from (4, 0) to focus (p/2, 0) is 5
  p = 8 := by
  sorry

end parabola_focus_distance_l3507_350771


namespace square_difference_l3507_350768

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 10) : 
  (x - y)^2 = 41 := by
  sorry

end square_difference_l3507_350768


namespace percentage_of_red_shirts_l3507_350755

theorem percentage_of_red_shirts 
  (total_students : ℕ) 
  (blue_percentage : ℚ) 
  (green_percentage : ℚ) 
  (other_colors : ℕ) 
  (h1 : total_students = 600) 
  (h2 : blue_percentage = 45/100) 
  (h3 : green_percentage = 15/100) 
  (h4 : other_colors = 102) :
  (total_students - (blue_percentage * total_students + green_percentage * total_students + other_colors)) / total_students = 23/100 := by
sorry

end percentage_of_red_shirts_l3507_350755


namespace right_triangle_perimeter_equals_area_l3507_350735

theorem right_triangle_perimeter_equals_area (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a + b + c = (1/2) * a * b →
  a + b - c = 4 := by
sorry

end right_triangle_perimeter_equals_area_l3507_350735


namespace min_values_xy_and_x_plus_y_l3507_350792

theorem min_values_xy_and_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 9/y = 1) : xy ≥ 36 ∧ x + y ≥ 16 := by
  sorry

end min_values_xy_and_x_plus_y_l3507_350792


namespace computer_price_increase_l3507_350736

theorem computer_price_increase (d : ℝ) (h : 2 * d = 585) : 
  (351 - d) / d * 100 = 20 := by
  sorry

end computer_price_increase_l3507_350736


namespace adrian_days_off_l3507_350756

/-- The number of days Adrian took off in a year -/
def total_holidays : ℕ := 48

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of days Adrian took off each month -/
def days_off_per_month : ℕ := total_holidays / months_in_year

theorem adrian_days_off :
  days_off_per_month = 4 :=
by sorry

end adrian_days_off_l3507_350756


namespace solution_to_logarithmic_equation_l3507_350785

theorem solution_to_logarithmic_equation :
  ∃ x : ℝ, (3 * Real.log x - 4 * Real.log 5 = -1) ∧ (x = (62.5 : ℝ) ^ (1/3)) := by
  sorry

end solution_to_logarithmic_equation_l3507_350785


namespace probability_of_letter_in_mathematics_l3507_350769

def alphabet_size : ℕ := 26
def unique_letters_in_mathematics : ℕ := 8

theorem probability_of_letter_in_mathematics :
  (unique_letters_in_mathematics : ℚ) / (alphabet_size : ℚ) = 4 / 13 := by
sorry

end probability_of_letter_in_mathematics_l3507_350769


namespace fourth_term_of_sequence_l3507_350716

theorem fourth_term_of_sequence (x : ℤ) : 
  x^2 - 2*x - 3 < 0 → 
  ∃ (a : ℕ → ℤ), (∀ n, a (n+1) - a n = a 1 - a 0) ∧ 
                 (∀ n, a n = x → x^2 - 2*x - 3 < 0) ∧
                 (a 3 = 3 ∨ a 3 = -1) :=
sorry

end fourth_term_of_sequence_l3507_350716


namespace five_digit_divisible_by_nine_l3507_350701

theorem five_digit_divisible_by_nine :
  ∃ (n : ℕ), 
    n < 10 ∧ 
    (35000 + n * 100 + 72) % 9 = 0 ∧
    (3 + 5 + n + 7 + 2) % 9 = 0 :=
by
  sorry

end five_digit_divisible_by_nine_l3507_350701


namespace square_difference_81_49_l3507_350718

theorem square_difference_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end square_difference_81_49_l3507_350718


namespace students_liking_both_desserts_l3507_350734

/-- Given a class of students and their dessert preferences, calculate the number of students who like both desserts. -/
theorem students_liking_both_desserts
  (total : ℕ)
  (like_apple : ℕ)
  (like_chocolate : ℕ)
  (like_neither : ℕ)
  (h1 : total = 35)
  (h2 : like_apple = 20)
  (h3 : like_chocolate = 17)
  (h4 : like_neither = 8) :
  like_apple + like_chocolate - (total - like_neither) = 10 := by
  sorry

end students_liking_both_desserts_l3507_350734


namespace company_female_employees_l3507_350723

theorem company_female_employees 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (total_managers : ℕ) 
  (male_managers : ℕ) 
  (h1 : total_managers = (2 : ℕ) * total_employees / (5 : ℕ)) 
  (h2 : male_managers = (2 : ℕ) * male_employees / (5 : ℕ)) 
  (h3 : total_managers = male_managers + 200) :
  total_employees - male_employees = 500 := by
sorry

end company_female_employees_l3507_350723


namespace gray_trees_sum_l3507_350702

/-- Represents the number of trees in a photograph -/
structure PhotoTrees where
  total : ℕ
  white : ℕ
  gray : ℕ

/-- The problem statement -/
theorem gray_trees_sum (photo1 photo2 photo3 : PhotoTrees) :
  photo1.total = 100 →
  photo2.total = 90 →
  photo3.total = photo3.white →
  photo1.white = photo2.white →
  photo2.white = photo3.white →
  photo3.white = 82 →
  photo1.gray + photo2.gray = 26 :=
by sorry

end gray_trees_sum_l3507_350702


namespace paradise_park_ferris_wheel_capacity_l3507_350711

/-- The total capacity of a Ferris wheel -/
def ferris_wheel_capacity (num_seats : ℕ) (people_per_seat : ℕ) : ℕ :=
  num_seats * people_per_seat

/-- Theorem: The capacity of a Ferris wheel with 14 seats and 6 people per seat is 84 -/
theorem paradise_park_ferris_wheel_capacity :
  ferris_wheel_capacity 14 6 = 84 := by
  sorry

end paradise_park_ferris_wheel_capacity_l3507_350711


namespace ball_probability_pairs_l3507_350777

theorem ball_probability_pairs : 
  ∃! k : ℕ, ∃ S : Finset (ℕ × ℕ),
    (∀ (m n : ℕ), (m, n) ∈ S ↔ 
      (m > n ∧ n ≥ 4 ∧ m + n ≤ 40 ∧ (m - n)^2 = m + n)) ∧
    S.card = k ∧ k = 3 := by sorry

end ball_probability_pairs_l3507_350777


namespace jasmine_purchase_cost_l3507_350740

/-- Calculate the total cost for Jasmine's purchase of coffee beans and milk --/
theorem jasmine_purchase_cost :
  let coffee_pounds : ℕ := 4
  let milk_gallons : ℕ := 2
  let coffee_price_per_pound : ℚ := 5/2
  let milk_price_per_gallon : ℚ := 7/2
  let discount_rate : ℚ := 1/10
  let tax_rate : ℚ := 2/25

  let total_before_discount : ℚ := coffee_pounds * coffee_price_per_pound + milk_gallons * milk_price_per_gallon
  let discount : ℚ := discount_rate * total_before_discount
  let discounted_price : ℚ := total_before_discount - discount
  let taxes : ℚ := tax_rate * discounted_price
  let final_amount : ℚ := discounted_price + taxes

  final_amount = 1652/100 := by sorry

end jasmine_purchase_cost_l3507_350740


namespace empty_vessel_possible_l3507_350747

/-- Represents a state of water distribution among three vessels --/
structure WaterState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a pouring operation from one vessel to another --/
inductive PouringOperation
  | FromAToB
  | FromAToC
  | FromBToA
  | FromBToC
  | FromCToA
  | FromCToB

/-- Applies a pouring operation to a water state --/
def applyPouring (state : WaterState) (op : PouringOperation) : WaterState :=
  match op with
  | PouringOperation.FromAToB => 
      if state.a ≤ state.b then {a := 0, b := state.b + state.a, c := state.c}
      else {a := state.a - state.b, b := 2 * state.b, c := state.c}
  | PouringOperation.FromAToC => 
      if state.a ≤ state.c then {a := 0, b := state.b, c := state.c + state.a}
      else {a := state.a - state.c, b := state.b, c := 2 * state.c}
  | PouringOperation.FromBToA => 
      if state.b ≤ state.a then {a := state.a + state.b, b := 0, c := state.c}
      else {a := 2 * state.a, b := state.b - state.a, c := state.c}
  | PouringOperation.FromBToC => 
      if state.b ≤ state.c then {a := state.a, b := 0, c := state.c + state.b}
      else {a := state.a, b := state.b - state.c, c := 2 * state.c}
  | PouringOperation.FromCToA => 
      if state.c ≤ state.a then {a := state.a + state.c, b := state.b, c := 0}
      else {a := 2 * state.a, b := state.b, c := state.c - state.a}
  | PouringOperation.FromCToB => 
      if state.c ≤ state.b then {a := state.a, b := state.b + state.c, c := 0}
      else {a := state.a, b := 2 * state.b, c := state.c - state.b}

/-- Predicate to check if a water state has an empty vessel --/
def hasEmptyVessel (state : WaterState) : Prop :=
  state.a = 0 ∨ state.b = 0 ∨ state.c = 0

/-- The main theorem stating that it's always possible to empty a vessel --/
theorem empty_vessel_possible (initialState : WaterState) : 
  ∃ (operations : List PouringOperation), 
    hasEmptyVessel (operations.foldl applyPouring initialState) :=
  sorry

end empty_vessel_possible_l3507_350747


namespace max_min_values_l3507_350770

theorem max_min_values (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 = 2) :
  (a + b + c ≤ Real.sqrt 6) ∧
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 * Real.sqrt 6 / 4) := by
  sorry

end max_min_values_l3507_350770


namespace problem_statement_l3507_350726

theorem problem_statement :
  ∀ (x y z : ℝ), x ≥ 0 → y ≥ 0 → z ≥ 0 →
    (2 * x^3 - 3 * x^2 + 1 ≥ 0) ∧
    ((2 / (1 + x^3) + 2 / (1 + y^3) + 2 / (1 + z^3) = 3) →
      ((1 - x) / (1 - x + x^2) + (1 - y) / (1 - y + y^2) + (1 - z) / (1 - z + z^2) ≥ 0)) := by
  sorry

end problem_statement_l3507_350726


namespace max_value_3xy_plus_yz_l3507_350779

theorem max_value_3xy_plus_yz (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  3*x*y + y*z ≤ Real.sqrt 10 / 2 :=
sorry

end max_value_3xy_plus_yz_l3507_350779


namespace shaded_area_l3507_350781

/-- The area of the shaded region in a grid with given properties -/
theorem shaded_area (total_area : ℝ) (triangle_base : ℝ) (triangle_height : ℝ)
  (h1 : total_area = 38)
  (h2 : triangle_base = 12)
  (h3 : triangle_height = 4) :
  total_area - (1/2 * triangle_base * triangle_height) = 14 := by
  sorry

end shaded_area_l3507_350781


namespace expression_value_l3507_350745

theorem expression_value : (19 + 43 / 151) * 151 = 2912 := by
  sorry

end expression_value_l3507_350745


namespace square_root_equation_solutions_cube_root_equation_solution_l3507_350748

theorem square_root_equation_solutions (x : ℝ) :
  (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1 := by sorry

theorem cube_root_equation_solution (x : ℝ) :
  (x - 2)^3 = -125 ↔ x = -3 := by sorry

end square_root_equation_solutions_cube_root_equation_solution_l3507_350748


namespace ten_special_divisors_l3507_350704

theorem ten_special_divisors : ∃ (n : ℕ), 
  n > 1 ∧ 
  (∀ d : ℕ, d > 1 → d ∣ n → ∃ (a r : ℕ), r > 1 ∧ d = a^r + 1) ∧
  n = 10 := by
sorry

end ten_special_divisors_l3507_350704


namespace divisibility_of_powers_l3507_350753

-- Define the polynomial and its greatest positive root
def f (x : ℝ) := x^3 - 3*x^2 + 1

def a : ℝ := sorry

axiom a_is_root : f a = 0

axiom a_is_greatest_positive_root : 
  ∀ x > 0, f x = 0 → x ≤ a

-- Define the floor function
def floor (x : ℝ) : ℤ := sorry

-- State the theorem
theorem divisibility_of_powers : 
  (17 ∣ floor (a^1788)) ∧ (17 ∣ floor (a^1988)) := by sorry

end divisibility_of_powers_l3507_350753


namespace largest_negative_congruent_to_one_mod_23_l3507_350795

theorem largest_negative_congruent_to_one_mod_23 : 
  ∀ n : ℤ, -99999 ≤ n ∧ n < -9999 ∧ n ≡ 1 [ZMOD 23] → n ≤ -9994 :=
by sorry

end largest_negative_congruent_to_one_mod_23_l3507_350795


namespace perp_planes_necessary_not_sufficient_l3507_350797

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the relation of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define two different planes
variable (α β : Plane)
variable (h_diff : α ≠ β)

-- Define a line m in plane α
variable (m : Line)
variable (h_m_in_α : line_in_plane m α)

-- Theorem statement
theorem perp_planes_necessary_not_sufficient :
  (∀ m, line_in_plane m α → perp_line_plane m β → perp_planes α β) ∧
  (∃ m, line_in_plane m α ∧ perp_planes α β ∧ ¬perp_line_plane m β) :=
sorry

end perp_planes_necessary_not_sufficient_l3507_350797


namespace geometric_sequence_ratio_l3507_350700

/-- Given a geometric sequence {a_n} with common ratio q = 1/2 and sum of first n terms S_n, 
    prove that S_3 / a_3 = 7 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = (1 / 2) * a n) →  -- Geometric sequence with common ratio 1/2
  (∀ n, S n = a 1 * (1 - (1 / 2)^n) / (1 - (1 / 2))) →  -- Sum formula
  S 3 / a 3 = 7 := by
sorry

end geometric_sequence_ratio_l3507_350700


namespace percentage_equality_l3507_350786

theorem percentage_equality (x : ℝ) : (15 / 100 * 75 = 2.5 / 100 * x) → x = 450 := by
  sorry

end percentage_equality_l3507_350786


namespace nina_weekend_earnings_l3507_350743

/-- Calculates the total money made from jewelry sales --/
def total_money_made (necklace_price bracelet_price earring_pair_price ensemble_price : ℚ)
                     (necklaces_sold bracelets_sold earrings_sold ensembles_sold : ℕ) : ℚ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_pair_price * (earrings_sold / 2) +
  ensemble_price * ensembles_sold

/-- Theorem: Nina's weekend earnings --/
theorem nina_weekend_earnings :
  let necklace_price : ℚ := 25
  let bracelet_price : ℚ := 15
  let earring_pair_price : ℚ := 10
  let ensemble_price : ℚ := 45
  let necklaces_sold : ℕ := 5
  let bracelets_sold : ℕ := 10
  let earrings_sold : ℕ := 20
  let ensembles_sold : ℕ := 2
  total_money_made necklace_price bracelet_price earring_pair_price ensemble_price
                    necklaces_sold bracelets_sold earrings_sold ensembles_sold = 465 :=
by
  sorry

end nina_weekend_earnings_l3507_350743


namespace mary_saturday_wage_l3507_350742

/-- Represents Mary's work schedule and earnings --/
structure WorkSchedule where
  weekday_hours : Nat
  saturday_hours : Nat
  regular_weekly_earnings : Nat
  saturday_weekly_earnings : Nat

/-- Calculates Mary's Saturday hourly wage --/
def saturday_hourly_wage (schedule : WorkSchedule) : Rat :=
  let regular_hourly_wage := schedule.regular_weekly_earnings / schedule.weekday_hours
  let saturday_earnings := schedule.saturday_weekly_earnings - schedule.regular_weekly_earnings
  saturday_earnings / schedule.saturday_hours

/-- Mary's actual work schedule --/
def mary_schedule : WorkSchedule :=
  { weekday_hours := 37
  , saturday_hours := 4
  , regular_weekly_earnings := 407
  , saturday_weekly_earnings := 483 }

/-- Theorem stating that Mary's Saturday hourly wage is $19 --/
theorem mary_saturday_wage :
  saturday_hourly_wage mary_schedule = 19 := by
  sorry

end mary_saturday_wage_l3507_350742


namespace mod_twelve_difference_l3507_350799

theorem mod_twelve_difference (n : ℕ) : (51 ^ n - 27 ^ n) % 12 = 0 := by
  sorry

end mod_twelve_difference_l3507_350799


namespace subtracted_value_l3507_350705

theorem subtracted_value (chosen_number : ℕ) (subtracted_value : ℕ) : 
  chosen_number = 990 →
  (chosen_number / 9 : ℚ) - subtracted_value = 10 →
  subtracted_value = 100 := by
sorry

end subtracted_value_l3507_350705


namespace right_triangle_area_l3507_350780

theorem right_triangle_area (a c : ℝ) (h1 : a = 40) (h2 : c = 41) :
  let b := Real.sqrt (c^2 - a^2)
  (1/2) * a * b = 180 := by
  sorry

end right_triangle_area_l3507_350780


namespace haleys_concert_tickets_l3507_350749

theorem haleys_concert_tickets (ticket_price : ℕ) (extra_tickets : ℕ) (total_spent : ℕ) : 
  ticket_price = 4 → extra_tickets = 5 → total_spent = 32 → 
  ∃ (tickets_for_friends : ℕ), 
    ticket_price * (tickets_for_friends + extra_tickets) = total_spent ∧ 
    tickets_for_friends = 3 :=
by sorry

end haleys_concert_tickets_l3507_350749


namespace smallest_transformed_sum_l3507_350724

/-- The number of faces on a standard die -/
def standardDieFaces : ℕ := 6

/-- The sum we want to compare with -/
def targetSum : ℕ := 980

/-- A function to calculate the transformed sum given the number of dice -/
def transformedSum (n : ℕ) : ℤ := 5 * n - targetSum

/-- The proposition that proves the smallest possible value of S -/
theorem smallest_transformed_sum :
  ∃ (n : ℕ), 
    (n * standardDieFaces ≥ targetSum) ∧ 
    (∀ m : ℕ, m < n → m * standardDieFaces < targetSum) ∧
    (transformedSum n = 5) ∧
    (∀ k : ℕ, k < n → transformedSum k < 5) := by
  sorry

end smallest_transformed_sum_l3507_350724


namespace fraction_equality_l3507_350791

theorem fraction_equality (a b c : ℝ) (h : a^2 = b*c) :
  (a + b) / (a - b) = (c + a) / (c - a) :=
by sorry

end fraction_equality_l3507_350791


namespace sum_in_second_quadrant_l3507_350750

/-- Given two complex numbers z₁ and z₂, prove that their sum is in the second quadrant -/
theorem sum_in_second_quadrant (z₁ z₂ : ℂ) 
  (h₁ : z₁ = -3 + 4*I) (h₂ : z₂ = 2 - 3*I) : 
  let z := z₁ + z₂
  z.re < 0 ∧ z.im > 0 := by
  sorry

#check sum_in_second_quadrant

end sum_in_second_quadrant_l3507_350750


namespace keno_probability_value_l3507_350733

/-- The set of integers from 1 to 80 -/
def keno_numbers : Finset Nat := Finset.range 80

/-- The set of numbers from 1 to 80 that contain the digit 8 -/
def numbers_with_eight : Finset Nat := {8, 18, 28, 38, 48, 58, 68, 78}

/-- The set of numbers from 1 to 80 that do not contain the digit 8 -/
def numbers_without_eight : Finset Nat := keno_numbers \ numbers_with_eight

/-- The number of numbers to be drawn in a KENO game -/
def draw_count : Nat := 20

/-- The probability of drawing 20 numbers from 1 to 80 such that none contain the digit 8 -/
def keno_probability : ℚ := (Nat.choose numbers_without_eight.card draw_count : ℚ) / (Nat.choose keno_numbers.card draw_count)

theorem keno_probability_value : keno_probability = 27249 / 4267580 := by
  sorry

end keno_probability_value_l3507_350733


namespace change_in_expression_l3507_350778

theorem change_in_expression (x b : ℝ) (h : b > 0) :
  let f := fun t : ℝ => t^2 - 5*t + 2
  f (x + b) - f x = 2*b*x + b^2 - 5*b :=
by sorry

end change_in_expression_l3507_350778


namespace runners_meet_again_l3507_350751

-- Define the track circumference
def track_circumference : ℝ := 400

-- Define the runners' speeds
def runner1_speed : ℝ := 5.0
def runner2_speed : ℝ := 5.5
def runner3_speed : ℝ := 6.0

-- Define the time when runners meet again
def meeting_time : ℝ := 800

-- Theorem statement
theorem runners_meet_again :
  ∀ (t : ℝ), t = meeting_time →
  (runner1_speed * t) % track_circumference = 
  (runner2_speed * t) % track_circumference ∧
  (runner2_speed * t) % track_circumference = 
  (runner3_speed * t) % track_circumference :=
by
  sorry

#check runners_meet_again

end runners_meet_again_l3507_350751


namespace unfair_coin_expected_worth_l3507_350767

/-- An unfair coin with given probabilities for heads and tails, and corresponding gains and losses -/
structure UnfairCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  gain_heads : ℝ
  loss_tails : ℝ
  prob_sum_one : prob_heads + prob_tails = 1
  prob_nonneg : prob_heads ≥ 0 ∧ prob_tails ≥ 0

/-- The expected worth of a coin flip -/
def expected_worth (c : UnfairCoin) : ℝ :=
  c.prob_heads * c.gain_heads + c.prob_tails * (-c.loss_tails)

/-- Theorem stating the expected worth of the specific unfair coin -/
theorem unfair_coin_expected_worth :
  ∃ (c : UnfairCoin),
    c.prob_heads = 2/3 ∧
    c.prob_tails = 1/3 ∧
    c.gain_heads = 5 ∧
    c.loss_tails = 6 ∧
    expected_worth c = 4/3 := by
  sorry

end unfair_coin_expected_worth_l3507_350767


namespace linear_regression_change_l3507_350752

/-- Given a linear regression equation y = 2 - 1.5x, prove that when x increases by 1, y decreases by 1.5. -/
theorem linear_regression_change (x y : ℝ) : 
  y = 2 - 1.5 * x → (2 - 1.5 * (x + 1)) = y - 1.5 := by
  sorry

end linear_regression_change_l3507_350752


namespace cos_sum_min_value_l3507_350774

theorem cos_sum_min_value (x : ℝ) : |Real.cos x| + |Real.cos (2 * x)| ≥ Real.sqrt 2 / 2 := by
  sorry

end cos_sum_min_value_l3507_350774


namespace simplify_fraction_product_l3507_350765

theorem simplify_fraction_product : (144 : ℚ) / 18 * 9 / 108 * 6 / 4 = 2 / 3 := by
  sorry

end simplify_fraction_product_l3507_350765


namespace simplify_expression_l3507_350793

theorem simplify_expression : 1 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 1 := by
  sorry

end simplify_expression_l3507_350793


namespace train_speed_calculation_l3507_350738

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 165 →
  bridge_length = 660 →
  crossing_time = 54.995600351971845 →
  ∃ (speed : ℝ), abs (speed - 54.0036) < 0.0001 ∧ 
  speed = (train_length + bridge_length) / crossing_time * 3.6 := by
  sorry

end train_speed_calculation_l3507_350738


namespace box_volume_cubic_feet_l3507_350703

/-- Conversion factor from cubic inches to cubic feet -/
def cubic_inches_per_cubic_foot : ℕ := 1728

/-- Volume of the box in cubic inches -/
def box_volume_cubic_inches : ℕ := 1728

/-- Theorem stating that the volume of the box in cubic feet is 1 -/
theorem box_volume_cubic_feet : 
  (box_volume_cubic_inches : ℚ) / cubic_inches_per_cubic_foot = 1 := by
  sorry

end box_volume_cubic_feet_l3507_350703


namespace intersection_M_N_l3507_350739

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l3507_350739


namespace sum_of_x_y_z_l3507_350762

theorem sum_of_x_y_z (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) : x + y + z = 16 * x := by
  sorry

end sum_of_x_y_z_l3507_350762


namespace inequality_equivalence_l3507_350741

theorem inequality_equivalence (x : ℝ) :
  (x + 1) * (1 / x - 1) > 0 ↔ x ∈ Set.Ioi (-1) ∪ Set.Ioo 0 1 :=
by sorry

end inequality_equivalence_l3507_350741


namespace trigonometric_system_solution_l3507_350709

theorem trigonometric_system_solution (x y z : ℝ) 
  (eq1 : Real.sin x + 2 * Real.sin (x + y + z) = 0)
  (eq2 : Real.sin y + 3 * Real.sin (x + y + z) = 0)
  (eq3 : Real.sin z + 4 * Real.sin (x + y + z) = 0) :
  ∃ (k1 k2 k3 : ℤ), x = k1 * Real.pi ∧ y = k2 * Real.pi ∧ z = k3 * Real.pi := by
  sorry

end trigonometric_system_solution_l3507_350709


namespace average_of_w_and_x_l3507_350728

theorem average_of_w_and_x (w x y : ℝ) 
  (h1 : 3 / w + 3 / x = 3 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = 1 / 2 := by
sorry

end average_of_w_and_x_l3507_350728


namespace one_eighth_of_two_to_36_l3507_350790

theorem one_eighth_of_two_to_36 (y : ℤ) :
  (1 / 8 : ℚ) * (2 : ℚ)^36 = (2 : ℚ)^y → y = 33 := by
  sorry

end one_eighth_of_two_to_36_l3507_350790


namespace smallest_four_digit_divisible_by_25_l3507_350713

theorem smallest_four_digit_divisible_by_25 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 25 = 0 → n ≥ 1000 :=
by
  sorry

end smallest_four_digit_divisible_by_25_l3507_350713


namespace student_count_l3507_350744

theorem student_count (initial_avg : ℝ) (incorrect_height : ℝ) (actual_height : ℝ) (actual_avg : ℝ) :
  initial_avg = 175 →
  incorrect_height = 151 →
  actual_height = 136 →
  actual_avg = 174.5 →
  ∃ n : ℕ, n = 30 ∧ n * actual_avg = n * initial_avg - (incorrect_height - actual_height) :=
by sorry

end student_count_l3507_350744


namespace inequality_solution_set_l3507_350776

theorem inequality_solution_set (x : ℝ) : 1 - 3 * (x - 1) < x ↔ x > 1 := by sorry

end inequality_solution_set_l3507_350776


namespace spherical_to_cartesian_coordinates_l3507_350757

/-- Given a point M with spherical coordinates (1, π/3, π/6), 
    prove that its Cartesian coordinates are (3/4, √3/4, 1/2). -/
theorem spherical_to_cartesian_coordinates :
  let r : ℝ := 1
  let θ : ℝ := π / 3
  let φ : ℝ := π / 6
  let x : ℝ := r * Real.sin θ * Real.cos φ
  let y : ℝ := r * Real.sin θ * Real.sin φ
  let z : ℝ := r * Real.cos θ
  (x = 3/4) ∧ (y = Real.sqrt 3 / 4) ∧ (z = 1/2) := by
  sorry


end spherical_to_cartesian_coordinates_l3507_350757


namespace trigonometric_identity_l3507_350719

theorem trigonometric_identity : 
  (Real.cos (68 * π / 180) * Real.cos (8 * π / 180) - Real.cos (82 * π / 180) * Real.cos (22 * π / 180)) /
  (Real.cos (53 * π / 180) * Real.cos (23 * π / 180) - Real.cos (67 * π / 180) * Real.cos (37 * π / 180)) = 1 := by
  sorry

end trigonometric_identity_l3507_350719
