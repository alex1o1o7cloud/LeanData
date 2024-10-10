import Mathlib

namespace preimage_of_one_two_l2876_287649

/-- The mapping f from R² to R² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that (3/2, -1/2) is the preimage of (1, 2) under f -/
theorem preimage_of_one_two :
  f (3/2, -1/2) = (1, 2) := by sorry

end preimage_of_one_two_l2876_287649


namespace power_of_128_l2876_287643

theorem power_of_128 : (128 : ℝ) ^ (4/7 : ℝ) = 16 := by sorry

end power_of_128_l2876_287643


namespace unique_prime_divides_sigma_pred_l2876_287696

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Theorem: The only prime p that divides σ(p-1) is 3 -/
theorem unique_prime_divides_sigma_pred :
  ∀ p : ℕ, Nat.Prime p → (p ∣ sigma (p - 1) ↔ p = 3) := by sorry

end unique_prime_divides_sigma_pred_l2876_287696


namespace ages_relationship_l2876_287633

/-- Given the ages of Katherine (K), Mel (M), and Lexi (L), with the relationships
    M = K - 3 and L = M + 2, prove that when K = 60, M = 57 and L = 59. -/
theorem ages_relationship (K M L : ℕ) 
    (h1 : M = K - 3) 
    (h2 : L = M + 2) 
    (h3 : K = 60) : 
  M = 57 ∧ L = 59 := by
sorry

end ages_relationship_l2876_287633


namespace divisible_polynomial_sum_l2876_287666

-- Define the polynomial
def p (A B : ℝ) (x : ℂ) := x^101 + A*x + B

-- Define the condition of divisibility
def is_divisible (A B : ℝ) : Prop :=
  ∀ x : ℂ, x^2 + x + 1 = 0 → p A B x = 0

-- Theorem statement
theorem divisible_polynomial_sum (A B : ℝ) (h : is_divisible A B) : A + B = 2 := by
  sorry

end divisible_polynomial_sum_l2876_287666


namespace max_profit_at_12_ships_l2876_287640

-- Define the output function
def R (x : ℕ) : ℤ := 3700 * x + 45 * x^2 - 10 * x^3

-- Define the cost function
def C (x : ℕ) : ℤ := 460 * x + 5000

-- Define the profit function
def P (x : ℕ) : ℤ := R x - C x

-- Define the marginal profit function
def MP (x : ℕ) : ℤ := P (x + 1) - P x

-- Theorem statement
theorem max_profit_at_12_ships :
  ∀ x : ℕ, 1 ≤ x → x ≤ 20 → P x ≤ P 12 :=
sorry

end max_profit_at_12_ships_l2876_287640


namespace different_gender_selection_l2876_287630

theorem different_gender_selection (total_members : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_members = 24)
  (h2 : boys = 12)
  (h3 : girls = 12)
  (h4 : total_members = boys + girls) :
  (boys * girls) + (girls * boys) = 288 := by
sorry

end different_gender_selection_l2876_287630


namespace unique_intersection_l2876_287683

/-- The value of m for which the vertical line x = m intersects the parabola x = -4y^2 + 2y + 3 at exactly one point -/
def m : ℚ := 13/4

/-- The equation of the parabola -/
def parabola (y : ℝ) : ℝ := -4 * y^2 + 2 * y + 3

/-- Theorem stating that the vertical line x = m intersects the parabola at exactly one point -/
theorem unique_intersection :
  ∃! y : ℝ, parabola y = m :=
sorry

end unique_intersection_l2876_287683


namespace king_high_school_teachers_l2876_287676

/-- The number of students at King High School -/
def num_students : ℕ := 1500

/-- The number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- The number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 3

/-- The number of students in each class -/
def students_per_class : ℕ := 35

/-- The number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- The number of teachers at King High School -/
def num_teachers : ℕ := 86

theorem king_high_school_teachers : 
  (num_students * classes_per_student) / students_per_class / classes_per_teacher = num_teachers := by
  sorry

end king_high_school_teachers_l2876_287676


namespace second_player_wins_l2876_287675

/-- Represents a grid in the domino gluing game -/
structure Grid :=
  (size : Nat)
  (is_cut_into_dominoes : Bool)

/-- Represents a move in the domino gluing game -/
structure Move :=
  (x1 y1 x2 y2 : Nat)

/-- Represents the state of the game -/
structure GameState :=
  (grid : Grid)
  (current_player : Nat)
  (moves : List Move)

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (move : Move) : Bool :=
  sorry

/-- Checks if the game is over (i.e., the figure is connected) -/
def is_game_over (state : GameState) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Checks if a strategy is winning for a player -/
def is_winning_strategy (player : Nat) (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem: the second player has a winning strategy -/
theorem second_player_wins (grid : Grid) 
    (h1 : grid.size = 100) 
    (h2 : grid.is_cut_into_dominoes = true) : 
  ∃ (strategy : Strategy), is_winning_strategy 2 strategy :=
sorry

end second_player_wins_l2876_287675


namespace magnitude_a_minus_2b_l2876_287609

def a : ℝ × ℝ × ℝ := (3, 5, -4)
def b : ℝ × ℝ × ℝ := (2, -1, -2)

theorem magnitude_a_minus_2b : 
  ‖(a.1 - 2 * b.1, a.2 - 2 * b.2, a.2.2 - 2 * b.2.2)‖ = 5 * Real.sqrt 2 := by
  sorry

end magnitude_a_minus_2b_l2876_287609


namespace tea_store_profit_l2876_287603

theorem tea_store_profit (m n : ℝ) (h : m > n) : 
  let cost := 40 * m + 60 * n
  let revenue := 50 * (m + n)
  revenue - cost > 0 := by
sorry

end tea_store_profit_l2876_287603


namespace milton_pies_sold_l2876_287674

/-- Calculates the total number of pies sold given the number of slices ordered and slices per pie -/
def total_pies_sold (apple_slices_ordered : ℕ) (peach_slices_ordered : ℕ) 
                    (slices_per_apple_pie : ℕ) (slices_per_peach_pie : ℕ) : ℕ :=
  (apple_slices_ordered / slices_per_apple_pie) + (peach_slices_ordered / slices_per_peach_pie)

/-- Theorem stating that given the specific conditions, Milton sold 15 pies -/
theorem milton_pies_sold : 
  total_pies_sold 56 48 8 6 = 15 := by
  sorry

#eval total_pies_sold 56 48 8 6

end milton_pies_sold_l2876_287674


namespace marked_squares_theorem_l2876_287663

/-- A type representing a table with marked squares -/
def MarkedTable (n : ℕ) := Fin n → Fin n → Bool

/-- A function that checks if a square is on or above the main diagonal -/
def isAboveDiagonal {n : ℕ} (i j : Fin n) : Bool :=
  i.val ≤ j.val

/-- A function that counts the number of marked squares in a table -/
def countMarkedSquares {n : ℕ} (table : MarkedTable n) : ℕ :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => if table i j then 1 else 0))

/-- A predicate that checks if a table can be rearranged to satisfy the condition -/
def canRearrange {n : ℕ} (table : MarkedTable n) : Prop :=
  ∃ (rowPerm colPerm : Equiv.Perm (Fin n)),
    ∀ i j, table i j → isAboveDiagonal (rowPerm i) (colPerm j)

theorem marked_squares_theorem (n : ℕ) (h : n > 1) :
  ∀ (table : MarkedTable n),
    canRearrange table ↔ countMarkedSquares table ≤ n + 1 :=
by sorry

end marked_squares_theorem_l2876_287663


namespace triangle_ratio_theorem_l2876_287627

theorem triangle_ratio_theorem (a b c : ℝ) (A B C : ℝ) :
  C = π / 3 →
  c = Real.sqrt 3 →
  (3 * a + b) / (3 * Real.sin A + Real.sin B) = 2 :=
by sorry

end triangle_ratio_theorem_l2876_287627


namespace no_real_solution_for_sqrt_equation_l2876_287646

theorem no_real_solution_for_sqrt_equation :
  ¬∃ (x : ℝ), Real.sqrt (3 - Real.sqrt x) = 2 := by
sorry

end no_real_solution_for_sqrt_equation_l2876_287646


namespace expand_product_l2876_287602

theorem expand_product (x : ℝ) : (x + 4) * (x - 5) = x^2 - x - 20 := by
  sorry

end expand_product_l2876_287602


namespace ara_current_height_l2876_287655

/-- Represents a person's height and growth --/
structure Person where
  originalHeight : ℝ
  growthFactor : ℝ

/-- Calculates the current height of a person given their original height and growth factor --/
def currentHeight (p : Person) : ℝ := p.originalHeight * (1 + p.growthFactor)

/-- Theorem stating Ara's current height given the conditions --/
theorem ara_current_height (shea ara : Person) 
  (h1 : shea.growthFactor = 0.25)
  (h2 : currentHeight shea = 75)
  (h3 : ara.originalHeight = shea.originalHeight)
  (h4 : ara.growthFactor = shea.growthFactor / 3) :
  currentHeight ara = 65 := by
  sorry


end ara_current_height_l2876_287655


namespace model_a_sample_size_l2876_287678

/-- Calculates the number of items to select in stratified sampling -/
def stratified_sample_size (total_production : ℕ) (model_production : ℕ) (sample_size : ℕ) : ℕ :=
  (model_production * sample_size) / total_production

/-- Proves that the stratified sample size for Model A is 6 -/
theorem model_a_sample_size :
  stratified_sample_size 9200 1200 46 = 6 := by
sorry

end model_a_sample_size_l2876_287678


namespace min_value_of_f_l2876_287600

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt (x^2 + 2)) + Real.sqrt (x^2 + 2)

theorem min_value_of_f :
  ∃ (min_val : ℝ), (∀ x, f x ≥ min_val) ∧ (min_val = (3 * Real.sqrt 2) / 2) := by
  sorry

end min_value_of_f_l2876_287600


namespace factorization_theorem_l2876_287645

theorem factorization_theorem (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2*y) * (x + 2*y) := by
  sorry

end factorization_theorem_l2876_287645


namespace real_part_of_one_plus_i_over_i_l2876_287625

/-- The real part of (1+i)/i is 1 -/
theorem real_part_of_one_plus_i_over_i : 
  Complex.re ((1 + Complex.I) / Complex.I) = 1 := by
  sorry

end real_part_of_one_plus_i_over_i_l2876_287625


namespace inscribed_square_area_l2876_287686

/-- The area of a square inscribed in an ellipse -/
theorem inscribed_square_area (x y : ℝ) :
  (x^2 / 4 + y^2 / 8 = 1) →  -- Ellipse equation
  (∃ t : ℝ, t > 0 ∧ x = t ∧ y = t) →  -- Square vertex condition
  (4 * t^2 = 32 / 3) :=  -- Area of the square
by
  sorry


end inscribed_square_area_l2876_287686


namespace percentage_relation_l2876_287644

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.05 * x) (h2 : b = 0.25 * x) :
  a = 0.2 * b := by
  sorry

end percentage_relation_l2876_287644


namespace no_arithmetic_progression_l2876_287673

theorem no_arithmetic_progression : 
  ¬∃ (y : ℝ), (∃ (d : ℝ), (3*y + 1) - (y - 3) = d ∧ (5*y - 7) - (3*y + 1) = d) := by
  sorry

end no_arithmetic_progression_l2876_287673


namespace min_value_x_plus_2y_l2876_287664

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  x + 2*y ≥ 9 ∧ ∀ M : ℝ, ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ 1/x' + 2/y' = 1 ∧ x' + 2*y' > M :=
sorry

end min_value_x_plus_2y_l2876_287664


namespace quadratic_real_roots_condition_l2876_287651

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, 4 * x^2 - (4*k - 2) * x + k^2 = 0) → k ≤ 1/4 := by
  sorry

end quadratic_real_roots_condition_l2876_287651


namespace sum_of_digits_base_8_of_888_l2876_287652

/-- The sum of the digits of the base 8 representation of 888₁₀ is 13. -/
theorem sum_of_digits_base_8_of_888 : 
  (Nat.digits 8 888).sum = 13 := by sorry

end sum_of_digits_base_8_of_888_l2876_287652


namespace line_equations_l2876_287616

/-- Given point M -/
def M : ℝ × ℝ := (-1, 2)

/-- Given line equation -/
def L : ℝ → ℝ → ℝ := λ x y ↦ 2*x + y + 5

/-- Parallel line -/
def L_parallel : ℝ → ℝ → ℝ := λ x y ↦ 2*x + y

/-- Perpendicular line -/
def L_perpendicular : ℝ → ℝ → ℝ := λ x y ↦ x - 2*y + 5

theorem line_equations :
  (L_parallel M.1 M.2 = 0 ∧ 
   ∀ (x y : ℝ), L_parallel x y = 0 → L x y = L_parallel x y + 5) ∧
  (L_perpendicular M.1 M.2 = 0 ∧ 
   ∀ (x y : ℝ), L x y = 0 → L_perpendicular x y = 0 → x = y) := by
  sorry

end line_equations_l2876_287616


namespace range_of_a_l2876_287629

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
by sorry

end range_of_a_l2876_287629


namespace inequality_proof_l2876_287691

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := by
  sorry

end inequality_proof_l2876_287691


namespace checkerboard_theorem_l2876_287679

def board_size : Nat := 9
def num_lines : Nat := 10

/-- The number of rectangles on the checkerboard -/
def num_rectangles : Nat := (num_lines.choose 2) * (num_lines.choose 2)

/-- The number of squares on the checkerboard -/
def num_squares : Nat := (board_size * (board_size + 1) * (2 * board_size + 1)) / 6

/-- The ratio of squares to rectangles -/
def ratio : Rat := num_squares / num_rectangles

theorem checkerboard_theorem :
  num_rectangles = 2025 ∧
  num_squares = 285 ∧
  ratio = 19 / 135 ∧
  19 + 135 = 154 := by sorry

end checkerboard_theorem_l2876_287679


namespace two_digit_perfect_square_divisible_by_five_l2876_287638

theorem two_digit_perfect_square_divisible_by_five :
  ∃! n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ ∃ m : ℕ, n = m^2 ∧ n % 5 = 0 :=
by sorry

end two_digit_perfect_square_divisible_by_five_l2876_287638


namespace tan_theta_minus_pi_over_four_l2876_287689

theorem tan_theta_minus_pi_over_four (θ : Real) :
  let z : ℂ := Complex.mk (Real.cos θ - 4/5) (Real.sin θ - 3/5)
  z.re = 0 → Real.tan (θ - Real.pi/4) = -7 :=
by sorry

end tan_theta_minus_pi_over_four_l2876_287689


namespace quadratic_vertex_in_first_quadrant_l2876_287624

/-- Given a quadratic function y = ax² + bx + c where a, b, and c satisfy certain conditions,
    prove that its vertex lies in the first quadrant. -/
theorem quadratic_vertex_in_first_quadrant
  (a b c : ℝ)
  (eq1 : a - b + c = 0)
  (eq2 : 9*a + 3*b + c = 0)
  (b_pos : b > 0) :
  let x := -b / (2*a)
  let y := a * x^2 + b * x + c
  x > 0 ∧ y > 0 := by
  sorry

end quadratic_vertex_in_first_quadrant_l2876_287624


namespace rounding_317500_equals_31_8_ten_thousand_l2876_287682

-- Define rounding to the nearest thousand
def roundToThousand (n : ℕ) : ℕ :=
  (n + 500) / 1000 * 1000

-- Define representation in ten thousands
def toTenThousand (n : ℕ) : ℚ :=
  n / 10000

-- Theorem statement
theorem rounding_317500_equals_31_8_ten_thousand :
  toTenThousand (roundToThousand 317500) = 31.8 := by
  sorry

end rounding_317500_equals_31_8_ten_thousand_l2876_287682


namespace painted_cube_theorem_l2876_287622

/-- Represents the number of painted faces a small cube can have -/
inductive PaintedFaces
  | one
  | two
  | three

/-- Represents a large cube that is painted on the outside and cut into smaller cubes -/
structure PaintedCube where
  edge_length : ℕ
  small_cube_length : ℕ

/-- Counts the number of small cubes with a specific number of painted faces -/
def count_painted_faces (cube : PaintedCube) (faces : PaintedFaces) : ℕ :=
  match faces with
  | PaintedFaces.one => 0   -- Placeholder, actual calculation needed
  | PaintedFaces.two => 0   -- Placeholder, actual calculation needed
  | PaintedFaces.three => 0 -- Placeholder, actual calculation needed

/-- Theorem stating the correct count of small cubes with different numbers of painted faces -/
theorem painted_cube_theorem (cube : PaintedCube) 
    (h1 : cube.edge_length = 10)
    (h2 : cube.small_cube_length = 1) :
    count_painted_faces cube PaintedFaces.three = 8 ∧
    count_painted_faces cube PaintedFaces.two = 96 ∧
    count_painted_faces cube PaintedFaces.one = 384 := by
  sorry

end painted_cube_theorem_l2876_287622


namespace unique_solution_to_x_equals_negative_x_l2876_287684

theorem unique_solution_to_x_equals_negative_x : 
  ∀ x : ℝ, x = -x ↔ x = 0 := by sorry

end unique_solution_to_x_equals_negative_x_l2876_287684


namespace infinite_nonprime_powers_l2876_287641

theorem infinite_nonprime_powers (k : ℕ) : ∃ n : ℕ, n ≥ k ∧
  (¬ Nat.Prime (2^(2^n) + 1) ∨ ¬ Nat.Prime (2018^(2^n) + 1)) := by
  sorry

end infinite_nonprime_powers_l2876_287641


namespace regular_10gon_triangle_probability_l2876_287695

/-- Regular 10-gon -/
def regular_10gon : Set (ℝ × ℝ) := sorry

/-- Set of all segments in the 10-gon -/
def segments (polygon : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) := sorry

/-- Predicate to check if three segments form a triangle with positive area -/
def forms_triangle (s1 s2 s3 : Set (ℝ × ℝ)) : Prop := sorry

/-- The probability of forming a triangle with positive area from three randomly chosen segments -/
def triangle_probability (polygon : Set (ℝ × ℝ)) : ℚ := sorry

/-- Main theorem: The probability of forming a triangle with positive area 
    from three distinct segments chosen randomly from a regular 10-gon is 343/715 -/
theorem regular_10gon_triangle_probability : 
  triangle_probability regular_10gon = 343 / 715 := by sorry

end regular_10gon_triangle_probability_l2876_287695


namespace pursuer_catches_pursued_l2876_287667

/-- Represents a point on an infinite straight line -/
structure Point where
  position : ℝ

/-- Represents a moving object on the line -/
structure MovingObject where
  initialPosition : Point
  speed : ℝ
  direction : Bool  -- True for positive direction, False for negative

/-- The pursuer (new police car) -/
def pursuer : MovingObject := {
  initialPosition := { position := 0 },
  speed := 1,  -- Normalized to 1
  direction := true  -- Arbitrary initial direction
}

/-- The pursued (stolen police car) -/
def pursued : MovingObject := {
  initialPosition := { position := 0 },  -- Arbitrary initial position
  speed := 0.9,  -- 90% of pursuer's speed
  direction := true  -- Arbitrary initial direction
}

/-- Theorem stating that the pursuer can always catch the pursued -/
theorem pursuer_catches_pursued :
  ∃ (t : ℝ), t ≥ 0 ∧ 
  pursuer.initialPosition.position + t * pursuer.speed = 
  pursued.initialPosition.position + t * pursued.speed :=
sorry

end pursuer_catches_pursued_l2876_287667


namespace magic_box_pennies_l2876_287614

def double_daily (initial : ℕ) (days : ℕ) : ℕ :=
  initial * (2 ^ days)

theorem magic_box_pennies :
  ∃ (initial : ℕ), double_daily initial 4 = 48 ∧ initial = 3 :=
sorry

end magic_box_pennies_l2876_287614


namespace system_solution_l2876_287654

theorem system_solution (a b x y : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  x^5 * y^17 = a ∧ x^2 * y^7 = b → x = a^7 / b^17 ∧ y = b^5 / a^2 := by
  sorry

end system_solution_l2876_287654


namespace optimal_scheme_is_best_l2876_287671

/-- Represents a horticultural design scheme -/
structure Scheme where
  a : ℕ  -- number of A type designs
  b : ℕ  -- number of B type designs

/-- Checks if a scheme is feasible given the constraints -/
def is_feasible (s : Scheme) : Prop :=
  s.a + s.b = 50 ∧
  80 * s.a + 50 * s.b ≤ 3490 ∧
  40 * s.a + 90 * s.b ≤ 2950

/-- Calculates the cost of a scheme -/
def cost (s : Scheme) : ℕ :=
  800 * s.a + 960 * s.b

/-- The optimal scheme -/
def optimal_scheme : Scheme :=
  ⟨33, 17⟩

theorem optimal_scheme_is_best :
  is_feasible optimal_scheme ∧
  ∀ s : Scheme, is_feasible s → cost s ≥ cost optimal_scheme :=
sorry

end optimal_scheme_is_best_l2876_287671


namespace trig_identity_l2876_287611

theorem trig_identity (α β : ℝ) : 
  Real.sin α ^ 2 + Real.sin β ^ 2 - Real.sin α ^ 2 * Real.sin β ^ 2 + Real.cos α ^ 2 * Real.cos β ^ 2 = 1 := by
  sorry

end trig_identity_l2876_287611


namespace sufficient_condition_absolute_value_l2876_287660

theorem sufficient_condition_absolute_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 := by sorry

end sufficient_condition_absolute_value_l2876_287660


namespace translation_teams_count_l2876_287669

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of translators -/
def total_translators : ℕ := 11

/-- The number of English-only translators -/
def english_only : ℕ := 5

/-- The number of Japanese-only translators -/
def japanese_only : ℕ := 4

/-- The number of bilingual translators -/
def bilingual : ℕ := 2

/-- The size of each translation team -/
def team_size : ℕ := 4

/-- The total number of ways to form two translation teams -/
def total_ways : ℕ :=
  choose english_only team_size * choose japanese_only team_size +
  choose bilingual 1 * (choose english_only (team_size - 1) * choose japanese_only team_size +
                        choose english_only team_size * choose japanese_only (team_size - 1)) +
  choose bilingual 2 * (choose english_only (team_size - 2) * choose japanese_only team_size +
                        choose english_only team_size * choose japanese_only (team_size - 2) +
                        choose english_only (team_size - 1) * choose japanese_only (team_size - 1))

theorem translation_teams_count : total_ways = 185 := by sorry

end translation_teams_count_l2876_287669


namespace cubic_polynomial_sum_l2876_287687

/-- Given a cubic polynomial Q with specific values at 1, -1, and 0,
    prove that Q(3) + Q(-3) = 47m -/
theorem cubic_polynomial_sum (m : ℝ) (Q : ℝ → ℝ) 
  (h_cubic : ∃ (a b c : ℝ), ∀ x, Q x = a * x^3 + b * x^2 + c * x + m)
  (h_1 : Q 1 = 3 * m)
  (h_neg1 : Q (-1) = 4 * m)
  (h_0 : Q 0 = m) :
  Q 3 + Q (-3) = 47 * m := by
  sorry

end cubic_polynomial_sum_l2876_287687


namespace reversed_segment_appears_in_powers_of_two_l2876_287639

/-- The sequence of first digits of powers of 5 -/
def firstDigitsPowersOf5 : ℕ → ℕ :=
  λ n => (5^n : ℕ) % 10

/-- The sequence of first digits of powers of 2 -/
def firstDigitsPowersOf2 : ℕ → ℕ :=
  λ n => (2^n : ℕ) % 10

/-- Check if a list is a subsequence of another list -/
def isSubsequence {α : Type} [DecidableEq α] : List α → List α → Bool :=
  λ subseq seq => sorry

/-- Theorem: Any reversed segment of firstDigitsPowersOf5 appears in firstDigitsPowersOf2 -/
theorem reversed_segment_appears_in_powers_of_two :
  ∀ (start finish : ℕ),
    start ≤ finish →
    ∃ (n m : ℕ),
      isSubsequence
        ((List.range (finish - start + 1)).map (λ i => firstDigitsPowersOf5 (start + i))).reverse
        ((List.range (m - n + 1)).map (λ i => firstDigitsPowersOf2 (n + i))) = true :=
by
  sorry

end reversed_segment_appears_in_powers_of_two_l2876_287639


namespace point_in_third_quadrant_l2876_287681

theorem point_in_third_quadrant (a b : ℝ) (h : a < b ∧ b < 0) :
  (a - b < 0) ∧ (b < 0) :=
by sorry

end point_in_third_quadrant_l2876_287681


namespace no_valid_a_for_quadratic_l2876_287650

theorem no_valid_a_for_quadratic : ¬∃ (a : ℝ), 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + 2*(a+1)*x₁ - (a-1) = 0) ∧
  (x₂^2 + 2*(a+1)*x₂ - (a-1) = 0) ∧
  ((x₁ > 1 ∧ x₂ < 1) ∨ (x₁ < 1 ∧ x₂ > 1)) :=
by sorry

end no_valid_a_for_quadratic_l2876_287650


namespace no_valid_numbers_l2876_287693

def digits : List Nat := [2, 3, 5, 6, 9]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n % 15 = 0) ∧
  (∀ d : Nat, d ∈ digits → (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d)) ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10)

theorem no_valid_numbers : ¬∃ n : Nat, is_valid_number n :=
sorry

end no_valid_numbers_l2876_287693


namespace square_area_difference_l2876_287662

/-- Given two line segments where one is 2 cm longer than the other, and the difference 
    of the areas of squares drawn on these line segments is 32 sq. cm, 
    prove that the length of the longer line segment is 9 cm. -/
theorem square_area_difference (x : ℝ) 
  (h1 : (x + 2)^2 - x^2 = 32) : 
  x + 2 = 9 := by sorry

end square_area_difference_l2876_287662


namespace inverse_proportion_point_l2876_287668

/-- Given an inverse proportion function y = 14/x passing through the point (a, 7), prove that a = 2 -/
theorem inverse_proportion_point (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = 14 / x) ∧ f a = 7) → a = 2 := by
sorry

end inverse_proportion_point_l2876_287668


namespace macy_running_goal_l2876_287636

/-- Calculates the remaining miles to reach a weekly running goal -/
def remaining_miles (weekly_goal : ℕ) (daily_miles : ℕ) (days_run : ℕ) : ℕ :=
  weekly_goal - (daily_miles * days_run)

/-- Proves that given a weekly goal of 24 miles, running 3 miles per day for 6 days,
    the remaining miles to reach the goal is 6 miles -/
theorem macy_running_goal :
  remaining_miles 24 3 6 = 6 := by
sorry

end macy_running_goal_l2876_287636


namespace complex_modulus_problem_l2876_287694

theorem complex_modulus_problem (z : ℂ) (h : z⁻¹ = 1 + I) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_problem_l2876_287694


namespace least_addition_for_divisibility_l2876_287618

theorem least_addition_for_divisibility (n : ℕ) : 
  (∃ (k : ℕ), k > 0 ∧ (821562 + k) % 5 = 0) → 
  (∃ (m : ℕ), m ≥ 3 ∧ (821562 + m) % 5 = 0) ∧ 
  (821562 + 3) % 5 = 0 := by
sorry

end least_addition_for_divisibility_l2876_287618


namespace origin_outside_circle_l2876_287688

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) :
  let circle := fun (x y : ℝ) => x^2 + y^2 + 2*a*x + 2*y + (a-1)^2 = 0
  ¬ circle 0 0 := by
  sorry

end origin_outside_circle_l2876_287688


namespace cubic_extremum_l2876_287698

/-- Given a cubic function f(x) = x³ + 3ax² + bx + a² with an extremum of 0 at x = -1,
    prove that a - b = -7 -/
theorem cubic_extremum (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 + 3*a*x^2 + b*x + a^2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1-ε) (-1+ε), f x ≥ f (-1)) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1-ε) (-1+ε), f x ≤ f (-1)) ∧
  f (-1) = 0 →
  a - b = -7 :=
by sorry


end cubic_extremum_l2876_287698


namespace book_length_l2876_287606

theorem book_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 3 → area = 6 → area = length * width → length = 2 := by
sorry

end book_length_l2876_287606


namespace m_range_l2876_287620

def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem m_range (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Set.Ioo 1 2 ∪ Set.Ici 3 :=
sorry

end m_range_l2876_287620


namespace proposition_3_proposition_4_l2876_287665

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (belongs_to : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Proposition 3
theorem proposition_3 
  (α β : Plane) (a b : Line) :
  plane_perpendicular α β →
  intersect α β a →
  belongs_to b β →
  perpendicular a b →
  line_perpendicular_plane b α :=
sorry

-- Proposition 4
theorem proposition_4
  (α : Plane) (a b l : Line) :
  belongs_to a α →
  belongs_to b α →
  perpendicular l a →
  perpendicular l b →
  line_perpendicular_plane l α :=
sorry

end proposition_3_proposition_4_l2876_287665


namespace mass_of_impurities_l2876_287615

/-- Given a sample of natural sulfur, prove that the mass of impurities
    is equal to the difference between the total mass and the mass of pure sulfur. -/
theorem mass_of_impurities (total_mass pure_sulfur_mass : ℝ) :
  total_mass ≥ pure_sulfur_mass →
  total_mass - pure_sulfur_mass = total_mass - pure_sulfur_mass :=
by sorry

end mass_of_impurities_l2876_287615


namespace isosceles_triangle_perimeter_l2876_287648

theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 6 → b = 3 → c = 3 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  (b = c) →  -- Isosceles condition
  a + b + c = 15 := by
sorry

end isosceles_triangle_perimeter_l2876_287648


namespace paint_calculation_l2876_287610

theorem paint_calculation (total_paint : ℚ) : 
  (1/4 : ℚ) * total_paint + (1/2 : ℚ) * ((3/4 : ℚ) * total_paint) = 225 → 
  total_paint = 360 := by
sorry

end paint_calculation_l2876_287610


namespace distance_between_complex_points_l2876_287661

def complex_to_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

theorem distance_between_complex_points :
  let z1 : ℂ := 7 - 4*I
  let z2 : ℂ := 2 + 8*I
  let A : ℝ × ℝ := complex_to_point z1
  let B : ℝ × ℝ := complex_to_point z2
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13 := by
  sorry

end distance_between_complex_points_l2876_287661


namespace time_after_2021_hours_l2876_287612

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a time of day -/
structure TimeOfDay where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- Represents a moment in time -/
structure Moment where
  day : DayOfWeek
  time : TimeOfDay

/-- Adds hours to a given moment and returns the new moment -/
def addHours (start : Moment) (hours : Nat) : Moment :=
  sorry

theorem time_after_2021_hours :
  let start : Moment := ⟨DayOfWeek.Monday, ⟨20, 21, sorry, sorry⟩⟩
  let end_moment : Moment := addHours start 2021
  end_moment = ⟨DayOfWeek.Tuesday, ⟨1, 21, sorry, sorry⟩⟩ := by
  sorry

end time_after_2021_hours_l2876_287612


namespace sum_of_reciprocals_greater_than_one_l2876_287635

theorem sum_of_reciprocals_greater_than_one 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ > 1) 
  (h₂ : a₂ > 1) 
  (h₃ : a₃ > 1) 
  (hS : a₁ + a₂ + a₃ = a₁ + a₂ + a₃) 
  (hcond₁ : a₁^2 / (a₁ - 1) > a₁ + a₂ + a₃) 
  (hcond₂ : a₂^2 / (a₂ - 1) > a₁ + a₂ + a₃) 
  (hcond₃ : a₃^2 / (a₃ - 1) > a₁ + a₂ + a₃) : 
  1 / (a₁ + a₂) + 1 / (a₂ + a₃) + 1 / (a₃ + a₁) > 1 := by
  sorry

end sum_of_reciprocals_greater_than_one_l2876_287635


namespace sum_always_positive_l2876_287621

theorem sum_always_positive (b : ℝ) (h : b = 2) : 
  (∀ x : ℝ, (3*x^2 - 2*x + b) + (x^2 + b*x - 1) = 4*x^2 + 1) ∧
  (∀ x : ℝ, 4*x^2 + 1 > 0) := by
sorry

end sum_always_positive_l2876_287621


namespace completing_square_l2876_287647

theorem completing_square (x : ℝ) : 
  (x^2 - 4*x - 3 = 0) ↔ ((x - 2)^2 = 7) := by
  sorry

end completing_square_l2876_287647


namespace alexey_dowel_cost_l2876_287699

theorem alexey_dowel_cost (screw_cost dowel_cost : ℚ) : 
  screw_cost = 7 →
  (0.85 * (screw_cost + dowel_cost) = screw_cost + 0.5 * dowel_cost) →
  dowel_cost = 3 := by
sorry

end alexey_dowel_cost_l2876_287699


namespace balls_in_boxes_l2876_287690

def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  if k = 2 then
    (n - 1).choose (k - 1) + (n - 2).choose (k - 1)
  else
    0

theorem balls_in_boxes :
  distribute_balls 6 2 = 21 :=
sorry

end balls_in_boxes_l2876_287690


namespace total_unique_plants_l2876_287653

-- Define the flower beds as finite sets
variable (A B C : Finset ℕ)

-- Define the cardinalities of the sets
variable (card_A : Finset.card A = 600)
variable (card_B : Finset.card B = 500)
variable (card_C : Finset.card C = 400)

-- Define the intersections
variable (card_AB : Finset.card (A ∩ B) = 60)
variable (card_AC : Finset.card (A ∩ C) = 80)
variable (card_BC : Finset.card (B ∩ C) = 40)
variable (card_ABC : Finset.card (A ∩ B ∩ C) = 20)

-- Theorem statement
theorem total_unique_plants :
  Finset.card (A ∪ B ∪ C) = 1340 := by
  sorry

end total_unique_plants_l2876_287653


namespace integer_chord_lines_count_l2876_287613

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  point : Point
  direction : Point  -- Direction vector

/-- Define the circle from the problem -/
def problemCircle : Circle :=
  { center := { x := 2, y := -2 },
    radius := 5 }

/-- Define the point M -/
def pointM : Point :=
  { x := 2, y := 2 }

/-- Function to check if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

/-- Function to count lines passing through M that cut off integer-length chords -/
def countIntegerChordLines (c : Circle) (m : Point) : ℕ :=
  sorry  -- Implementation details omitted

/-- The main theorem -/
theorem integer_chord_lines_count :
  isInside pointM problemCircle →
  countIntegerChordLines problemCircle pointM = 8 := by
  sorry

end integer_chord_lines_count_l2876_287613


namespace photography_preference_l2876_287601

-- Define the number of students who dislike photography
variable (x : ℕ)

-- Define the total number of students in the class
def total : ℕ := 9 * x

-- Define the number of students who like photography
def like : ℕ := 5 * x

-- Define the number of students who are neutral towards photography
def neutral : ℕ := x + 12

-- Theorem statement
theorem photography_preference (x : ℕ) :
  like x = (total x / 2) + 3 := by
  sorry

end photography_preference_l2876_287601


namespace ritas_money_theorem_l2876_287692

/-- Calculates the remaining money after Rita's purchases --/
def ritas_remaining_money (initial_amount dresses_cost pants_cost jackets_cost transportation : ℕ) : ℕ :=
  initial_amount - (5 * dresses_cost + 3 * pants_cost + 4 * jackets_cost + transportation)

/-- Theorem stating that Rita's remaining money is 139 --/
theorem ritas_money_theorem :
  ritas_remaining_money 400 20 12 30 5 = 139 := by
  sorry

end ritas_money_theorem_l2876_287692


namespace existence_of_polynomial_and_c1_value_l2876_287685

/-- D(m) counts the number of quadruples (a₁, a₂, a₃, a₄) of distinct integers 
    with 1 ≤ aᵢ ≤ m for all i such that m divides a₁+a₂+a₃+a₄ -/
def D (m : ℕ) : ℕ := sorry

/-- The polynomial q(x) = c₃x³ + c₂x² + c₁x + c₀ -/
def q (x : ℕ) : ℕ := sorry

theorem existence_of_polynomial_and_c1_value :
  ∃ (c₃ c₂ c₁ c₀ : ℤ), 
    (∀ m : ℕ, m ≥ 5 → Odd m → D m = c₃ * m^3 + c₂ * m^2 + c₁ * m + c₀) ∧ 
    c₁ = 11 := by
  sorry

end existence_of_polynomial_and_c1_value_l2876_287685


namespace max_abs_z_quadratic_equation_l2876_287617

theorem max_abs_z_quadratic_equation (a b c z : ℂ) 
  (h1 : Complex.abs a = 1)
  (h2 : Complex.abs b = 1)
  (h3 : Complex.abs c = 1)
  (h4 : a * z^2 + 2 * b * z + c = 0) :
  Complex.abs z ≤ 1 + Real.sqrt 2 :=
sorry

end max_abs_z_quadratic_equation_l2876_287617


namespace negation_equivalence_l2876_287607

theorem negation_equivalence :
  (¬ ∀ x : ℝ, ∃ n : ℕ+, (n : ℝ) > x^2) ↔ 
  (∃ x : ℝ, ∀ n : ℕ+, (n : ℝ) < x^2) := by sorry

end negation_equivalence_l2876_287607


namespace polynomial_division_remainder_l2876_287670

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^2 - 11 * X + 18 = (X - 3) * q + 12 := by
  sorry

end polynomial_division_remainder_l2876_287670


namespace parabola_equation_points_with_y_neg_three_l2876_287623

/-- A parabola passing through (1,0) and (0,-3) with axis of symmetry x=2 -/
structure Parabola where
  -- Define the parabola using a function
  f : ℝ → ℝ
  -- The parabola passes through (1,0)
  passes_through_A : f 1 = 0
  -- The parabola passes through (0,-3)
  passes_through_B : f 0 = -3
  -- The axis of symmetry is x=2
  symmetry_axis : ∀ x, f (2 + x) = f (2 - x)

/-- The equation of the parabola is y = -(x-2)^2 + 1 -/
theorem parabola_equation (p : Parabola) : 
  ∀ x, p.f x = -(x - 2)^2 + 1 := by sorry

/-- The points (0,-3) and (4,-3) are the only points on the parabola with y-coordinate -3 -/
theorem points_with_y_neg_three (p : Parabola) :
  ∀ x, p.f x = -3 ↔ x = 0 ∨ x = 4 := by sorry

end parabola_equation_points_with_y_neg_three_l2876_287623


namespace shopkeeper_cloth_sale_l2876_287659

/-- Represents the sale of cloth by a shopkeeper -/
structure ClothSale where
  totalSellingPrice : ℕ
  lossPerMetre : ℕ
  costPricePerMetre : ℕ

/-- Calculates the number of metres of cloth sold given the sale details -/
def metresSold (sale : ClothSale) : ℕ :=
  sale.totalSellingPrice / (sale.costPricePerMetre - sale.lossPerMetre)

/-- Theorem stating that for the given conditions, the shopkeeper sold 200 metres of cloth -/
theorem shopkeeper_cloth_sale :
  let sale : ClothSale := {
    totalSellingPrice := 12000,
    lossPerMetre := 6,
    costPricePerMetre := 66
  }
  metresSold sale = 200 := by
  sorry

end shopkeeper_cloth_sale_l2876_287659


namespace polar_to_cartesian_l2876_287628

/-- Given a point M with polar coordinates (2, 2π/3), its Cartesian coordinates are (-1, √3) -/
theorem polar_to_cartesian :
  let ρ : ℝ := 2
  let θ : ℝ := 2 * π / 3
  let x : ℝ := ρ * Real.cos θ
  let y : ℝ := ρ * Real.sin θ
  (x = -1) ∧ (y = Real.sqrt 3) := by sorry

end polar_to_cartesian_l2876_287628


namespace necessary_not_sufficient_condition_l2876_287657

theorem necessary_not_sufficient_condition (a : ℝ) : 
  (∀ x, ax + 1 = 0 → x^2 + x - 6 = 0) ∧ 
  (∃ x, x^2 + x - 6 = 0 ∧ ax + 1 ≠ 0) →
  a = -1/2 ∨ a = -1/3 :=
by sorry

end necessary_not_sufficient_condition_l2876_287657


namespace second_group_cost_l2876_287672

/-- The cost of a hotdog in dollars -/
def hotdog_cost : ℚ := 1/2

/-- The cost of a soft drink in dollars -/
def soft_drink_cost : ℚ := 1/2

/-- The number of hotdogs purchased by the first group -/
def first_group_hotdogs : ℕ := 10

/-- The number of soft drinks purchased by the first group -/
def first_group_drinks : ℕ := 5

/-- The total cost of the first group's purchase in dollars -/
def first_group_total : ℚ := 25/2

/-- The number of hotdogs purchased by the second group -/
def second_group_hotdogs : ℕ := 7

/-- The number of soft drinks purchased by the second group -/
def second_group_drinks : ℕ := 4

/-- Theorem stating that the cost of the second group's purchase is $5.50 -/
theorem second_group_cost : 
  (second_group_hotdogs : ℚ) * hotdog_cost + (second_group_drinks : ℚ) * soft_drink_cost = 11/2 := by
  sorry

end second_group_cost_l2876_287672


namespace decompose_4_705_l2876_287658

theorem decompose_4_705 : 
  ∃ (units hundredths thousandths : ℕ),
    4.705 = (units : ℝ) + (7 : ℝ) / 10 + (thousandths : ℝ) / 1000 ∧
    units = 4 ∧
    thousandths = 5 := by
  sorry

end decompose_4_705_l2876_287658


namespace roots_and_d_values_l2876_287604

-- Define the polynomial p(x)
def p (c d x : ℝ) : ℝ := x^3 + c*x + d

-- Define the polynomial q(x)
def q (c d x : ℝ) : ℝ := x^3 + c*x + d - 270

-- Theorem statement
theorem roots_and_d_values (u v c d : ℝ) : 
  (p c d u = 0 ∧ p c d v = 0) ∧ 
  (q c d (u+3) = 0 ∧ q c d (v-2) = 0) →
  d = -6 ∨ d = -120 := by
sorry

end roots_and_d_values_l2876_287604


namespace factorial_sum_remainder_l2876_287642

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem factorial_sum_remainder (n : ℕ) : 
  n ≥ 50 → sum_factorials n % 25 = (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 25 := by
  sorry

end factorial_sum_remainder_l2876_287642


namespace stewart_farm_horse_food_l2876_287619

/-- Represents a farm with sheep and horses -/
structure Farm where
  sheep : ℕ
  horses : ℕ
  total_horse_food : ℕ

/-- Calculates the amount of food each horse needs per day -/
def horse_food_per_day (f : Farm) : ℚ :=
  f.total_horse_food / f.horses

/-- The Stewart farm satisfies the given conditions -/
def stewart_farm : Farm :=
  { sheep := 24,
    horses := 56,
    total_horse_food := 12880 }

theorem stewart_farm_horse_food :
  horse_food_per_day stewart_farm = 230 :=
sorry

end stewart_farm_horse_food_l2876_287619


namespace no_solution_iff_m_eq_neg_two_l2876_287677

theorem no_solution_iff_m_eq_neg_two (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (x - 5) / (x - 3) ≠ m / (x - 3) + 2) ↔ m = -2 :=
by sorry

end no_solution_iff_m_eq_neg_two_l2876_287677


namespace acoustic_guitar_price_l2876_287626

theorem acoustic_guitar_price (total_guitars : ℕ) (total_revenue : ℕ) 
  (electric_price : ℕ) (electric_count : ℕ) :
  total_guitars = 9 →
  total_revenue = 3611 →
  electric_price = 479 →
  electric_count = 4 →
  (total_revenue - electric_price * electric_count) / (total_guitars - electric_count) = 339 := by
sorry

end acoustic_guitar_price_l2876_287626


namespace inverse_function_theorem_l2876_287608

noncomputable def f (x : ℝ) : ℝ := Real.log (3 * x + 1)

def f_domain (x : ℝ) : Prop := x > -1

noncomputable def g (x : ℝ) : ℝ := (Real.exp x - 1) ^ 3

theorem inverse_function_theorem (x : ℝ) (hx : f_domain x) :
  g (f x) = x ∧ f (g x) = x :=
sorry

end inverse_function_theorem_l2876_287608


namespace lot_worth_l2876_287605

/-- Given a lot where a man owns half and sells a tenth of his share for $460, 
    prove that the worth of the entire lot is $9200. -/
theorem lot_worth (man_share : ℚ) (sold_fraction : ℚ) (sold_amount : ℕ) :
  man_share = 1/2 →
  sold_fraction = 1/10 →
  sold_amount = 460 →
  (sold_amount / sold_fraction) / man_share = 9200 := by
  sorry

end lot_worth_l2876_287605


namespace interest_rate_equation_l2876_287680

/-- Given a principal that doubles in 10 years with semiannual compounding,
    this theorem states the equation that the annual interest rate must satisfy. -/
theorem interest_rate_equation (r : ℝ) : 
  (∀ P : ℝ, P > 0 → 2 * P = P * (1 + r / 2) ^ 20) ↔ 2 = (1 + r / 2) ^ 20 :=
sorry

end interest_rate_equation_l2876_287680


namespace valid_arrangements_count_l2876_287634

/-- The number of ways to arrange 15 letters (4 D's, 6 E's, and 5 F's) with specific constraints -/
def letterArrangements : ℕ :=
  Finset.sum (Finset.range 5) (fun j =>
    Nat.choose 4 j * Nat.choose 6 (4 - j) * Nat.choose 5 j)

/-- Theorem stating that the number of valid arrangements is equal to the sum formula -/
theorem valid_arrangements_count :
  letterArrangements =
    Finset.sum (Finset.range 5) (fun j =>
      Nat.choose 4 j * Nat.choose 6 (4 - j) * Nat.choose 5 j) :=
by
  sorry

end valid_arrangements_count_l2876_287634


namespace alex_bike_trip_l2876_287631

/-- Alex's cross-country bike trip problem -/
theorem alex_bike_trip (total_distance : ℝ) (flat_speed : ℝ) (uphill_speed : ℝ) (uphill_time : ℝ)
                       (downhill_speed : ℝ) (downhill_time : ℝ) (walking_distance : ℝ) :
  total_distance = 164 →
  flat_speed = 20 →
  uphill_speed = 12 →
  uphill_time = 2.5 →
  downhill_speed = 24 →
  downhill_time = 1.5 →
  walking_distance = 8 →
  ∃ (flat_time : ℝ), 
    flat_time = 4.5 ∧ 
    total_distance = flat_speed * flat_time + uphill_speed * uphill_time + 
                     downhill_speed * downhill_time + walking_distance :=
by sorry


end alex_bike_trip_l2876_287631


namespace ball_selection_count_l2876_287632

/-- Represents the number of balls of each color -/
def ballsPerColor : ℕ := 7

/-- Represents the number of colors -/
def numberOfColors : ℕ := 3

/-- Represents the total number of balls -/
def totalBalls : ℕ := ballsPerColor * numberOfColors

/-- Checks if three numbers are non-consecutive -/
def areNonConsecutive (a b c : ℕ) : Prop :=
  (a + 1 ≠ b ∧ b + 1 ≠ c) ∧ (b + 1 ≠ a ∧ c + 1 ≠ b) ∧ (c + 1 ≠ a ∧ a + 1 ≠ c)

/-- Counts the number of ways to select 3 non-consecutive numbers from 1 to 7 -/
def nonConsecutiveSelections : ℕ := 35

/-- The main theorem to be proved -/
theorem ball_selection_count :
  (∃ (f : Fin totalBalls → ℕ × Fin numberOfColors),
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ i, (f i).1 ∈ Finset.range ballsPerColor) ∧
    (∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      areNonConsecutive (f a).1 (f b).1 (f c).1 ∧
      (f a).2 ≠ (f b).2 ∧ (f b).2 ≠ (f c).2 ∧ (f a).2 ≠ (f c).2)) →
  nonConsecutiveSelections * numberOfColors = 60 :=
sorry

end ball_selection_count_l2876_287632


namespace congruent_side_length_for_specific_triangle_l2876_287697

/-- Represents an isosceles triangle with base length and area -/
structure IsoscelesTriangle where
  base : ℝ
  area : ℝ

/-- Calculates the length of a congruent side in an isosceles triangle -/
def congruentSideLength (triangle : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating that for an isosceles triangle with base 30 and area 72, 
    the length of a congruent side is 15.75 -/
theorem congruent_side_length_for_specific_triangle :
  let triangle : IsoscelesTriangle := { base := 30, area := 72 }
  congruentSideLength triangle = 15.75 := by sorry

end congruent_side_length_for_specific_triangle_l2876_287697


namespace A_expression_l2876_287637

theorem A_expression (a : ℝ) (A : ℝ) 
  (h : 2.353 * A = (3 * a + Real.sqrt (6 * a - 1))^(1/2) + (3 * a - Real.sqrt (6 * a - 1))^(1/2)) :
  ((1/6 ≤ a ∧ a < 1/3) → A = Real.sqrt 2 / (1 - 3 * a)) ∧
  (a > 1/3 → A = Real.sqrt (12 * a - 2) / (3 * a - 1)) := by
  sorry

end A_expression_l2876_287637


namespace inequality_equivalence_l2876_287656

theorem inequality_equivalence (x : ℝ) : 
  ‖‖x - 2‖ - 1‖ ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 := by
  sorry

end inequality_equivalence_l2876_287656
