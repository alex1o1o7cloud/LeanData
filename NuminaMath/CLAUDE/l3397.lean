import Mathlib

namespace integer_solutions_system_l3397_339792

theorem integer_solutions_system : 
  {(x, y, z) : ℤ × ℤ × ℤ | x + y - z = 6 ∧ x^3 + y^3 - z^3 = 414} = 
  {(3, 8, 5), (8, 3, 5), (3, -5, -8), (-5, 8, -3), (-5, 3, -8), (8, -5, -3)} :=
by sorry

end integer_solutions_system_l3397_339792


namespace intersection_complement_equality_l3397_339757

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {1, 5} := by sorry

end intersection_complement_equality_l3397_339757


namespace absolute_sum_zero_implies_value_l3397_339747

theorem absolute_sum_zero_implies_value (a b : ℝ) :
  |3*a + b + 5| + |2*a - 2*b - 2| = 0 →
  2*a^2 - 3*a*b = -4 := by
sorry

end absolute_sum_zero_implies_value_l3397_339747


namespace complex_fraction_simplification_l3397_339750

theorem complex_fraction_simplification :
  (1 * 3 * 5 * 7 * 9) * (10 * 12 * 14 * 16 * 18) / (5 * 6 * 7 * 8 * 9)^2 = 2 := by
  sorry

end complex_fraction_simplification_l3397_339750


namespace average_songs_in_remaining_sets_l3397_339754

def bandRepertoire : ℕ := 30
def firstSetSongs : ℕ := 5
def secondSetSongs : ℕ := 7
def encoreSongs : ℕ := 2
def remainingSets : ℕ := 2

theorem average_songs_in_remaining_sets :
  (bandRepertoire - (firstSetSongs + secondSetSongs + encoreSongs)) / remainingSets = 8 := by
  sorry

end average_songs_in_remaining_sets_l3397_339754


namespace solve_equation_l3397_339721

/-- Proves that for x = 3.3333333333333335, the equation √((x * y) / 3) = x is satisfied when y = 10 -/
theorem solve_equation (x : ℝ) (y : ℝ) (h1 : x = 3.3333333333333335) (h2 : y = 10) :
  Real.sqrt ((x * y) / 3) = x := by
  sorry

end solve_equation_l3397_339721


namespace chips_purchased_l3397_339765

/-- Given that P packets of chips can be purchased for R dimes,
    and 1 dollar is worth 10 dimes, the number of packets that
    can be purchased for M dollars is 10MP/R. -/
theorem chips_purchased (P R M : ℚ) (h1 : P > 0) (h2 : R > 0) (h3 : M > 0) :
  (P / R) * (M * 10) = 10 * M * P / R :=
by sorry

end chips_purchased_l3397_339765


namespace sum_of_fractions_l3397_339780

theorem sum_of_fractions (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : ∃ n : ℤ, (a / b + b / c + c / a : ℚ) = n)
  (h2 : ∃ m : ℤ, (b / a + c / b + a / c : ℚ) = m) :
  (a / b + b / c + c / a : ℚ) = 3 ∨ (a / b + b / c + c / a : ℚ) = -3 :=
by sorry

end sum_of_fractions_l3397_339780


namespace inequality_range_l3397_339751

theorem inequality_range (a b x : ℝ) (ha : a ≠ 0) :
  (|2*a + b| + |2*a - b| ≥ |a| * (|2 + x| + |2 - x|)) → x ∈ Set.Icc (-2) 2 :=
sorry

end inequality_range_l3397_339751


namespace triangle_circle_square_sum_l3397_339793

/-- Given a system of equations representing triangles, circles, and squares,
    prove that the sum of one triangle, two circles, and one square equals 35. -/
theorem triangle_circle_square_sum : 
  ∀ (x y z : ℝ),
  (2 * x + 3 * y + z = 45) →
  (x + 5 * y + 2 * z = 58) →
  (3 * x + y + 3 * z = 62) →
  (x + 2 * y + z = 35) := by
  sorry

end triangle_circle_square_sum_l3397_339793


namespace hole_pattern_symmetry_l3397_339714

/-- Represents a rectangular piece of paper --/
structure Paper where
  length : ℝ
  width : ℝ

/-- Represents a point on the paper --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a fold operation --/
inductive Fold
  | LeftToRight
  | TopToBottom
  | Diagonal

/-- Represents the hole pattern after unfolding --/
inductive HolePattern
  | SymmetricAll
  | SingleCenter
  | VerticalOnly
  | HorizontalOnly

/-- Performs a series of folds on the paper --/
def foldPaper (p : Paper) (folds : List Fold) : Paper :=
  sorry

/-- Punches a hole at a specific location on the folded paper --/
def punchHole (p : Paper) (loc : Point) : Paper :=
  sorry

/-- Unfolds the paper and determines the resulting hole pattern --/
def unfoldAndAnalyze (p : Paper) : HolePattern :=
  sorry

/-- Main theorem: The hole pattern is symmetrical across all axes --/
theorem hole_pattern_symmetry 
  (initialPaper : Paper)
  (folds : List Fold)
  (holeLocation : Point) :
  initialPaper.length = 8 ∧ 
  initialPaper.width = 4 ∧
  folds = [Fold.LeftToRight, Fold.TopToBottom, Fold.Diagonal] ∧
  holeLocation.x = 1/4 ∧ 
  holeLocation.y = 3/4 →
  unfoldAndAnalyze (punchHole (foldPaper initialPaper folds) holeLocation) = HolePattern.SymmetricAll :=
by
  sorry

end hole_pattern_symmetry_l3397_339714


namespace volume_of_large_cube_l3397_339716

/-- Given a cube with surface area 96 cm², prove that 8 such cubes form a larger cube with volume 512 cm³ -/
theorem volume_of_large_cube (small_cube : Real → Real → Real → Real) 
  (h1 : ∀ x, small_cube x x x = 96) -- surface area of small cube is 96
  (h2 : ∀ x y z, small_cube x y z = 6 * x * y) -- definition of surface area for a cube
  (large_cube : Real → Real → Real → Real)
  (h3 : ∀ x, large_cube x x x = 8 * small_cube (x/2) (x/2) (x/2)) -- large cube is made of 8 small cubes
  : ∃ x, large_cube x x x = 512 :=
sorry

end volume_of_large_cube_l3397_339716


namespace sum_can_equal_fifty_l3397_339727

theorem sum_can_equal_fifty : ∃ (scenario : Type) (sum : scenario → ℝ), ∀ (s : scenario), sum s = 50 := by
  sorry

end sum_can_equal_fifty_l3397_339727


namespace contradictory_statement_l3397_339773

theorem contradictory_statement (x : ℝ) :
  (∀ x, x + 3 ≥ 0 → x ≥ -3) ↔ (∀ x, x + 3 < 0 → x < -3) :=
by sorry

end contradictory_statement_l3397_339773


namespace box_volume_formula_l3397_339729

/-- The volume of an open box formed from a rectangular cardboard sheet. -/
def box_volume (y : ℝ) : ℝ :=
  (20 - 2*y) * (12 - 2*y) * y

theorem box_volume_formula (y : ℝ) 
  (h : 0 < y ∧ y < 6) : -- y is positive and less than half the smaller dimension
  box_volume y = 4*y^3 - 64*y^2 + 240*y := by
  sorry

end box_volume_formula_l3397_339729


namespace rain_probability_l3397_339737

theorem rain_probability (p_day : ℝ) (p_consecutive : ℝ) 
  (h1 : p_day = 1/3)
  (h2 : p_consecutive = 1/5) :
  p_consecutive / p_day = 3/5 := by
  sorry

end rain_probability_l3397_339737


namespace composition_result_l3397_339728

-- Define the functions f and g
def f (b : ℝ) (x : ℝ) : ℝ := 5 * x + b
def g (b : ℝ) (x : ℝ) : ℝ := b * x + 3

-- State the theorem
theorem composition_result (b e : ℝ) :
  (∀ x, f b (g b x) = 15 * x + e) → e = 18 := by
  sorry

end composition_result_l3397_339728


namespace triangle_inequality_iff_squared_sum_l3397_339779

theorem triangle_inequality_iff_squared_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (2 * (a^4 + b^4 + c^4) < (a^2 + b^2 + c^2)^2) ↔ 
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
by sorry

end triangle_inequality_iff_squared_sum_l3397_339779


namespace cosine_equation_solutions_l3397_339771

theorem cosine_equation_solutions (x : Real) :
  (∃ (s : Finset Real), s.card = 14 ∧ 
    (∀ y ∈ s, -π ≤ y ∧ y ≤ π ∧ 
      Real.cos (6 * y) + (Real.cos (3 * y))^4 + (Real.sin (2 * y))^2 + (Real.cos y)^2 = 0) ∧
    (∀ z, -π ≤ z ∧ z ≤ π ∧ 
      Real.cos (6 * z) + (Real.cos (3 * z))^4 + (Real.sin (2 * z))^2 + (Real.cos z)^2 = 0 → 
      z ∈ s)) :=
by sorry

end cosine_equation_solutions_l3397_339771


namespace toms_rate_difference_l3397_339734

/-- Proves that Tom's rate is 5 steps per minute faster than Matt's, given their relative progress --/
theorem toms_rate_difference (matt_rate : ℕ) (matt_steps : ℕ) (tom_steps : ℕ) :
  matt_rate = 20 →
  matt_steps = 220 →
  tom_steps = 275 →
  ∃ (tom_rate : ℕ), tom_rate = matt_rate + 5 ∧ tom_steps * matt_rate = matt_steps * tom_rate :=
by sorry

end toms_rate_difference_l3397_339734


namespace main_age_is_46_l3397_339712

/-- Represents the ages of four siblings -/
structure Ages where
  main : ℕ
  brother : ℕ
  sister : ℕ
  youngest : ℕ

/-- Checks if the given ages satisfy all the conditions -/
def satisfiesConditions (ages : Ages) : Prop :=
  let futureAges := Ages.mk (ages.main + 10) (ages.brother + 10) (ages.sister + 10) (ages.youngest + 10)
  futureAges.main + futureAges.brother + futureAges.sister + futureAges.youngest = 88 ∧
  futureAges.main = 2 * futureAges.brother ∧
  futureAges.main = 3 * futureAges.sister ∧
  futureAges.main = 4 * futureAges.youngest ∧
  ages.brother = ages.sister + 3 ∧
  ages.sister = 2 * ages.youngest ∧
  ages.youngest = 4

theorem main_age_is_46 :
  ∃ (ages : Ages), satisfiesConditions ages ∧ ages.main = 46 := by
  sorry

end main_age_is_46_l3397_339712


namespace xyz_sum_sqrt_l3397_339736

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 15) 
  (h2 : z + x = 18) 
  (h3 : x + y = 17) : 
  Real.sqrt (x * y * z * (x + y + z)) = 10 * Real.sqrt 70 := by
  sorry

end xyz_sum_sqrt_l3397_339736


namespace largest_inscribed_circle_radius_for_specific_quad_l3397_339701

/-- A quadrilateral with given side lengths -/
structure Quadrilateral where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ

/-- The radius of the largest inscribed circle in a quadrilateral -/
def largest_inscribed_circle_radius (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating the largest inscribed circle radius for a specific quadrilateral -/
theorem largest_inscribed_circle_radius_for_specific_quad :
  let q : Quadrilateral := ⟨15, 10, 8, 13⟩
  largest_inscribed_circle_radius q = 5.7 := by
  sorry

end largest_inscribed_circle_radius_for_specific_quad_l3397_339701


namespace train_platform_passing_time_l3397_339702

/-- The time taken for a train to pass a platform -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (time_to_cross_point : ℝ) 
  (platform_length : ℝ) 
  (train_length_positive : 0 < train_length)
  (time_to_cross_point_positive : 0 < time_to_cross_point)
  (platform_length_positive : 0 < platform_length)
  (h1 : train_length = 1200)
  (h2 : time_to_cross_point = 120)
  (h3 : platform_length = 1200) : 
  (train_length + platform_length) / (train_length / time_to_cross_point) = 240 := by
sorry


end train_platform_passing_time_l3397_339702


namespace baker_pies_sold_l3397_339787

theorem baker_pies_sold (cakes : ℕ) (cake_price pie_price total_earnings : ℚ) 
  (h1 : cakes = 453)
  (h2 : cake_price = 12)
  (h3 : pie_price = 7)
  (h4 : total_earnings = 6318) :
  (total_earnings - cakes * cake_price) / pie_price = 126 :=
by sorry

end baker_pies_sold_l3397_339787


namespace wall_painting_fraction_l3397_339777

theorem wall_painting_fraction (paint_rate : ℝ) (total_time minutes : ℝ) 
  (h1 : paint_rate * total_time = 1)  -- Can paint whole wall in total_time
  (h2 : minutes / total_time = 1 / 5) -- Minutes is 1/5 of total time
  : paint_rate * minutes = 1 / 5 := by
  sorry

end wall_painting_fraction_l3397_339777


namespace hunter_always_catches_grasshopper_l3397_339724

/-- A point in the 2D integer plane -/
structure Point where
  x : Int
  y : Int

/-- The grasshopper's trajectory -/
structure Trajectory where
  start : Point
  jump : Point

/-- Spiral search function that returns the nth point in the spiral -/
def spiralSearch (n : Nat) : Point :=
  sorry

/-- Predicate to check if a point is on a trajectory at a given time -/
def onTrajectory (p : Point) (t : Trajectory) (time : Nat) : Prop :=
  p.x = t.start.x + t.jump.x * time ∧ p.y = t.start.y + t.jump.y * time

theorem hunter_always_catches_grasshopper :
  ∀ (t : Trajectory), ∃ (time : Nat), onTrajectory (spiralSearch time) t time :=
sorry

end hunter_always_catches_grasshopper_l3397_339724


namespace factorization_equality_l3397_339738

theorem factorization_equality (x y : ℝ) : x^2 * y - 2 * x * y^2 + y^3 = y * (x - y)^2 := by
  sorry

end factorization_equality_l3397_339738


namespace car_speed_problem_l3397_339746

theorem car_speed_problem (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) (distance : ℝ) :
  speed1 = 45 →
  time = 14/3 →
  distance = 490 →
  (speed1 + speed2) * time = distance →
  speed2 = 60 :=
by sorry

end car_speed_problem_l3397_339746


namespace min_value_quadratic_l3397_339774

theorem min_value_quadratic (x : ℝ) : 
  (∀ x, x^2 - 6*x + 10 ≥ 1) ∧ (∃ x, x^2 - 6*x + 10 = 1) :=
by sorry

end min_value_quadratic_l3397_339774


namespace problem_solution_l3397_339726

theorem problem_solution (p q : ℝ) 
  (h1 : 3 / p = 6)
  (h2 : 3 / q = 18)
  (h3 : abs (p - q - 0.33333333333333337) < 1e-14) :
  p = 0.5 := by
  sorry

end problem_solution_l3397_339726


namespace sin_15_cos_15_eq_quarter_l3397_339762

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end sin_15_cos_15_eq_quarter_l3397_339762


namespace fabric_cost_per_yard_l3397_339725

theorem fabric_cost_per_yard 
  (total_spent : ℝ) 
  (total_yards : ℝ) 
  (h1 : total_spent = 120) 
  (h2 : total_yards = 16) : 
  total_spent / total_yards = 7.50 := by
sorry

end fabric_cost_per_yard_l3397_339725


namespace problem_solution_l3397_339740

theorem problem_solution (a b : ℝ) 
  (sum_eq : a + b = 12) 
  (diff_sq_eq : a^2 - b^2 = 48) : 
  a - b = 4 := by
sorry

end problem_solution_l3397_339740


namespace product_increase_l3397_339776

theorem product_increase (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (a + 1) * (b + 1) = 2 * a * b) : 
  (a^2 - 1) * (b^2 - 1) = 4 * a * b := by
sorry

end product_increase_l3397_339776


namespace hyperbola_eccentricity_l3397_339790

/-- The eccentricity of a hyperbola with equation x²/4 - y²/5 = 1 is 3/2 -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 3/2 ∧ ∀ (x y : ℝ), x^2/4 - y^2/5 = 1 → 
  ∃ (a b c : ℝ), a^2 = 4 ∧ b^2 = 5 ∧ c^2 = a^2 + b^2 ∧ e = c/a := by
  sorry

end hyperbola_eccentricity_l3397_339790


namespace sum_greater_than_twice_a_l3397_339703

noncomputable section

variables (a : ℝ) (x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := x^2 + 2 * Real.cos x

def g (x : ℝ) : ℝ := (deriv f) x - 5 * x + 5 * a * Real.log x

theorem sum_greater_than_twice_a (h₁ : x₁ ≠ x₂) (h₂ : g a x₁ = g a x₂) : 
  x₁ + x₂ > 2 * a := by
  sorry

end

end sum_greater_than_twice_a_l3397_339703


namespace linear_system_k_values_l3397_339775

/-- Given a system of linear equations in two variables x and y,
    prove the value of k under certain conditions. -/
theorem linear_system_k_values (x y k : ℝ) : 
  (3 * x + y = k + 1) →
  (x + 3 * y = 3) →
  (
    ((x * y < 0) → (k = -4)) ∧
    ((x + y < 3 ∧ x - y > 1) → (4 < k ∧ k < 8))
  ) := by sorry

end linear_system_k_values_l3397_339775


namespace cat_walking_rate_l3397_339755

/-- Given a cat's walking scenario with total time, resistance time, and distance walked,
    calculate the cat's walking rate in feet per minute. -/
theorem cat_walking_rate 
  (total_time : ℝ) 
  (resistance_time : ℝ) 
  (distance_walked : ℝ) 
  (h1 : total_time = 28) 
  (h2 : resistance_time = 20) 
  (h3 : distance_walked = 64) : 
  (distance_walked / (total_time - resistance_time)) = 8 := by
  sorry

#check cat_walking_rate

end cat_walking_rate_l3397_339755


namespace number_of_divisors_of_36_l3397_339717

theorem number_of_divisors_of_36 : Finset.card (Nat.divisors 36) = 9 := by sorry

end number_of_divisors_of_36_l3397_339717


namespace complement_of_M_l3397_339719

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

-- State the theorem
theorem complement_of_M : 
  Set.compl M = {y : ℝ | y < -1} := by sorry

end complement_of_M_l3397_339719


namespace add_neg_two_three_l3397_339711

theorem add_neg_two_three : -2 + 3 = 1 := by sorry

end add_neg_two_three_l3397_339711


namespace three_card_sequence_count_l3397_339783

/-- The number of cards in the deck -/
def deck_size : ℕ := 60

/-- The number of suits in the deck -/
def num_suits : ℕ := 5

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 12

/-- The number of cards to pick -/
def cards_to_pick : ℕ := 3

/-- The number of ways to pick three different cards in sequence from the deck -/
def ways_to_pick : ℕ := deck_size * (deck_size - 1) * (deck_size - 2)

theorem three_card_sequence_count :
  deck_size = num_suits * cards_per_suit →
  ways_to_pick = 205320 := by
  sorry

end three_card_sequence_count_l3397_339783


namespace six_hardcover_books_l3397_339741

/-- Represents the purchase of a set of books with two price options --/
structure BookPurchase where
  totalVolumes : ℕ
  paperbackPrice : ℕ
  hardcoverPrice : ℕ
  totalCost : ℕ

/-- Calculates the number of hardcover books purchased --/
def hardcoverCount (purchase : BookPurchase) : ℕ :=
  sorry

/-- Theorem stating that for the given purchase scenario, 6 hardcover books were bought --/
theorem six_hardcover_books (purchase : BookPurchase) 
  (h1 : purchase.totalVolumes = 12)
  (h2 : purchase.paperbackPrice = 18)
  (h3 : purchase.hardcoverPrice = 28)
  (h4 : purchase.totalCost = 276) : 
  hardcoverCount purchase = 6 := by
  sorry

end six_hardcover_books_l3397_339741


namespace min_value_reciprocal_sum_l3397_339732

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → 4/x + 1/y ≥ 9/2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ 4/x + 1/y = 9/2) :=
by sorry

end min_value_reciprocal_sum_l3397_339732


namespace total_spent_equals_20_l3397_339782

def bracelet_price : ℕ := 4
def keychain_price : ℕ := 5
def coloring_book_price : ℕ := 3

def paula_bracelets : ℕ := 2
def paula_keychains : ℕ := 1
def olive_coloring_books : ℕ := 1
def olive_bracelets : ℕ := 1

def paula_total : ℕ := paula_bracelets * bracelet_price + paula_keychains * keychain_price
def olive_total : ℕ := olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price

theorem total_spent_equals_20 : paula_total + olive_total = 20 := by
  sorry

end total_spent_equals_20_l3397_339782


namespace final_state_theorem_l3397_339772

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Black

/-- Represents the state of the box -/
structure BoxState where
  white : Nat
  black : Nat

/-- The initial state of the box -/
def initialState : BoxState :=
  { white := 2015, black := 2015 }

/-- The final state of the box -/
def finalState : BoxState :=
  { white := 2, black := 1 }

/-- Represents one step of the ball selection process -/
def selectBalls (state : BoxState) : BoxState :=
  sorry

/-- Predicate to check if the process should stop -/
def stopCondition (state : BoxState) : Prop :=
  state.white + state.black = 3

/-- Theorem stating that the process will end with 2 white balls and 1 black ball -/
theorem final_state_theorem (state : BoxState) :
  state = initialState →
  (∃ n : Nat, (selectBalls^[n] state) = finalState ∧ stopCondition (selectBalls^[n] state)) :=
sorry

end final_state_theorem_l3397_339772


namespace grid_arithmetic_sequences_l3397_339713

/-- Given a 7x1 grid of numbers with two additional columns of length 3 and 5,
    prove that the value M satisfies the arithmetic sequence properties. -/
theorem grid_arithmetic_sequences (a : ℤ) (b c : ℚ) (M : ℚ) : 
  a = 25 ∧ 
  b = 16 ∧ 
  c = 20 ∧ 
  (∀ i : Fin 7, ∃ d : ℚ, a + i.val * d = a + 6 * d) ∧  -- row is arithmetic
  (∀ j : Fin 3, ∃ e : ℚ, a + j.val * e = b) ∧  -- first column is arithmetic
  (∀ k : Fin 5, ∃ f : ℚ, M + k.val * f = -20) ∧  -- second column is arithmetic
  (a + 3 * (b - a) / 3 = b) ∧  -- 4th element in row equals top of middle column
  (a + 6 * (M - a) / 6 = M) →  -- last element in row equals top of right column
  M = -6.25 := by
sorry

end grid_arithmetic_sequences_l3397_339713


namespace f_properties_l3397_339756

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2) * Real.log x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := Real.log x - 2 / x + 1

-- Theorem statement
theorem f_properties :
  (∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f' x = 0) ∧
  (∀ x : ℝ, x > 0 → f x > 0) := by
  sorry

end

end f_properties_l3397_339756


namespace provolone_needed_l3397_339759

def cheese_blend (m r p : ℝ) : Prop :=
  m / r = 2 ∧ p / r = 2

theorem provolone_needed (m r : ℝ) (hm : m = 20) (hr : r = 10) :
  ∃ p : ℝ, cheese_blend m r p ∧ p = 20 :=
by
  sorry

end provolone_needed_l3397_339759


namespace negative_square_inequality_l3397_339708

theorem negative_square_inequality (a b : ℝ) : a < b → b < 0 → a^2 > b^2 := by
  sorry

end negative_square_inequality_l3397_339708


namespace min_value_of_f_l3397_339794

/-- The quadratic function f(x) = x^2 + 10x + 21 -/
def f (x : ℝ) : ℝ := x^2 + 10*x + 21

/-- Theorem: The minimum value of f(x) = x^2 + 10x + 21 is -4 -/
theorem min_value_of_f : 
  ∀ x : ℝ, f x ≥ -4 ∧ ∃ x₀ : ℝ, f x₀ = -4 := by
  sorry

end min_value_of_f_l3397_339794


namespace complex_equality_l3397_339706

-- Define the complex numbers
def z1 (x y : ℝ) : ℂ := x - 1 + y * Complex.I
def z2 (x : ℝ) : ℂ := Complex.I - 3 * x

-- Theorem statement
theorem complex_equality (x y : ℝ) :
  z1 x y = z2 x → x = 1/4 ∧ y = 1 := by
  sorry

end complex_equality_l3397_339706


namespace series_sum_l3397_339778

theorem series_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hc_def : c = a - b) :
  (∑' n : ℕ, 1 / (n * c * ((n + 1) * c))) = 1 / (b * c) :=
sorry

end series_sum_l3397_339778


namespace coplanar_iff_k_eq_neg_two_l3397_339705

/-- Two lines in 3D space -/
structure Line3D where
  parameterization : ℝ → ℝ × ℝ × ℝ

/-- Checks if two lines are coplanar -/
def are_coplanar (l1 l2 : Line3D) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧
    ∀ (t s : ℝ),
      let (x1, y1, z1) := l1.parameterization s
      let (x2, y2, z2) := l2.parameterization t
      a * x1 + b * y1 + c * z1 + d =
      a * x2 + b * y2 + c * z2 + d

theorem coplanar_iff_k_eq_neg_two :
  ∀ (k : ℝ),
    let l1 : Line3D := ⟨λ s => (-1 + s, 3 - k*s, 1 + k*s)⟩
    let l2 : Line3D := ⟨λ t => (t/2, 1 + t, 2 - t)⟩
    are_coplanar l1 l2 ↔ k = -2 := by
  sorry

end coplanar_iff_k_eq_neg_two_l3397_339705


namespace horner_method_proof_l3397_339748

/-- Horner's method for evaluating a polynomial -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 + 3x^3 + 5x + 4 -/
def f : ℝ → ℝ := fun x => 2 * x^4 + 3 * x^3 + 5 * x + 4

theorem horner_method_proof :
  f 2 = horner_eval [2, 3, 0, 5, 4] 2 ∧ horner_eval [2, 3, 0, 5, 4] 2 = 70 := by
  sorry

end horner_method_proof_l3397_339748


namespace kerosene_cost_in_cents_l3397_339766

-- Define the cost of a pound of rice in dollars
def rice_cost : ℚ := 33/100

-- Define the relationship between eggs and rice
def dozen_eggs_cost (rc : ℚ) : ℚ := rc

-- Define the relationship between kerosene and eggs
def half_liter_kerosene_cost (ec : ℚ) : ℚ := (8/12) * ec

-- Define the conversion from dollars to cents
def dollars_to_cents (d : ℚ) : ℚ := 100 * d

-- State the theorem
theorem kerosene_cost_in_cents : 
  dollars_to_cents (2 * half_liter_kerosene_cost (dozen_eggs_cost rice_cost)) = 44 := by
  sorry

end kerosene_cost_in_cents_l3397_339766


namespace bedroom_difference_is_sixty_l3397_339797

/-- The difference in square footage between two bedrooms -/
def bedroom_size_difference (total_size martha_size : ℝ) : ℝ :=
  (total_size - martha_size) - martha_size

/-- Theorem: Given the total size of two bedrooms and the size of one bedroom,
    prove that the difference between the two bedroom sizes is 60 sq ft -/
theorem bedroom_difference_is_sixty
  (total_size : ℝ)
  (martha_size : ℝ)
  (h1 : total_size = 300)
  (h2 : martha_size = 120) :
  bedroom_size_difference total_size martha_size = 60 := by
  sorry

end bedroom_difference_is_sixty_l3397_339797


namespace max_sum_distances_l3397_339753

/-- Given a real number k, two lines l₁ and l₂, and points P, Q, and M,
    prove that the maximum value of |MP| + |MQ| is 4. -/
theorem max_sum_distances (k : ℝ) :
  let P : ℝ × ℝ := (0, 0)
  let Q : ℝ × ℝ := (2, 2)
  let l₁ := {(x, y) : ℝ × ℝ | k * x + y = 0}
  let l₂ := {(x, y) : ℝ × ℝ | k * x - y - 2 * k + 2 = 0}
  let circle := {M : ℝ × ℝ | (M.1 - 1)^2 + (M.2 - 1)^2 = 2}
  ∀ M ∈ circle, (‖M - P‖ + ‖M - Q‖) ≤ 4 :=
by sorry

end max_sum_distances_l3397_339753


namespace square_rectangle_area_multiplier_l3397_339749

theorem square_rectangle_area_multiplier :
  let square_perimeter : ℝ := 800
  let rectangle_length : ℝ := 125
  let rectangle_width : ℝ := 64
  let square_side : ℝ := square_perimeter / 4
  let square_area : ℝ := square_side * square_side
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  square_area / rectangle_area = 5 := by
  sorry

end square_rectangle_area_multiplier_l3397_339749


namespace odd_number_probability_l3397_339730

/-- The set of digits used to form the number -/
def digits : Finset Nat := {1, 4, 6, 9}

/-- The set of odd digits from the given set -/
def oddDigits : Finset Nat := {1, 9}

/-- The probability of forming an odd four-digit number -/
def probabilityOdd : ℚ := (oddDigits.card : ℚ) / (digits.card : ℚ)

/-- Theorem stating that the probability of forming an odd four-digit number is 1/2 -/
theorem odd_number_probability : probabilityOdd = 1/2 := by
  sorry

end odd_number_probability_l3397_339730


namespace sum_remainder_mod_9_l3397_339768

theorem sum_remainder_mod_9 : (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 9 = 6 := by
  sorry

end sum_remainder_mod_9_l3397_339768


namespace total_street_lights_l3397_339739

theorem total_street_lights (neighborhoods : ℕ) (roads_per_neighborhood : ℕ) (lights_per_side : ℕ) : 
  neighborhoods = 10 → roads_per_neighborhood = 4 → lights_per_side = 250 →
  neighborhoods * roads_per_neighborhood * lights_per_side * 2 = 20000 := by
sorry

end total_street_lights_l3397_339739


namespace perfect_square_base_9_l3397_339769

def is_base_9_digit (d : ℕ) : Prop := d < 9

def base_9_to_decimal (a b d : ℕ) : ℕ := 729 * a + 81 * b + 36 + d

theorem perfect_square_base_9 (a b d : ℕ) (ha : a ≠ 0) (hd : is_base_9_digit d) :
  ∃ (k : ℕ), (base_9_to_decimal a b d) = k^2 → d ∈ ({0, 1, 4} : Set ℕ) := by
  sorry

end perfect_square_base_9_l3397_339769


namespace function_values_l3397_339720

/-- The linear function f(x) = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x - 3

theorem function_values :
  (f 4 = 5) ∧ (f (3/2) = 0) := by sorry

end function_values_l3397_339720


namespace min_value_2x_plus_y_l3397_339767

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y + 6 = x * y) :
  2 * x + y ≥ 12 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ + 6 = x₀ * y₀ ∧ 2 * x₀ + y₀ = 12 :=
by sorry

end min_value_2x_plus_y_l3397_339767


namespace sqrt_difference_equality_l3397_339752

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (64 + 16) = Real.sqrt 170 - 4 * Real.sqrt 5 := by
  sorry

end sqrt_difference_equality_l3397_339752


namespace volume_of_rotated_figure_l3397_339788

-- Define the figure
def figure (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define the volume of the solid formed by rotation
def volume_of_rotation (f : ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem volume_of_rotated_figure :
  volume_of_rotation figure = 4 * Real.pi^2 := by sorry

end volume_of_rotated_figure_l3397_339788


namespace count_divisible_by_seven_l3397_339735

theorem count_divisible_by_seven : 
  (Finset.filter (fun n => n % 7 = 0) (Finset.Icc 200 400)).card = 29 := by
  sorry

end count_divisible_by_seven_l3397_339735


namespace x_zero_value_l3397_339795

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem x_zero_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 1) → x₀ = 1 := by
  sorry

end x_zero_value_l3397_339795


namespace polynomial_evaluation_gcd_of_three_numbers_l3397_339781

-- Problem 1: Polynomial evaluation
def f (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x - 6

theorem polynomial_evaluation : f 1 = 9 := by sorry

-- Problem 2: GCD of three numbers
theorem gcd_of_three_numbers : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by sorry

end polynomial_evaluation_gcd_of_three_numbers_l3397_339781


namespace heine_biscuits_l3397_339761

/-- The number of biscuits Mrs. Heine needs to buy for her dogs -/
def total_biscuits (num_dogs : ℕ) (biscuits_per_dog : ℕ) : ℕ :=
  num_dogs * biscuits_per_dog

/-- Theorem stating that Mrs. Heine needs to buy 6 biscuits -/
theorem heine_biscuits : total_biscuits 2 3 = 6 := by
  sorry

end heine_biscuits_l3397_339761


namespace cricket_bat_selling_price_l3397_339764

/-- The selling price of a cricket bat given profit and profit percentage -/
theorem cricket_bat_selling_price 
  (profit : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : profit = 205)
  (h2 : profit_percentage = 31.782945736434108) :
  let cost_price := profit / (profit_percentage / 100)
  let selling_price := cost_price + profit
  selling_price = 850 := by
sorry

end cricket_bat_selling_price_l3397_339764


namespace sum_of_fractions_l3397_339744

theorem sum_of_fractions : (2 : ℚ) / 20 + (4 : ℚ) / 40 + (5 : ℚ) / 50 = (3 : ℚ) / 10 := by
  sorry

end sum_of_fractions_l3397_339744


namespace new_jasmine_percentage_l3397_339722

/-- Calculates the new jasmine percentage in a solution after adding jasmine and water -/
theorem new_jasmine_percentage
  (initial_volume : ℝ)
  (initial_jasmine_percentage : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 90)
  (h2 : initial_jasmine_percentage = 5)
  (h3 : added_jasmine = 8)
  (h4 : added_water = 2) :
  let initial_jasmine := initial_volume * (initial_jasmine_percentage / 100)
  let new_jasmine := initial_jasmine + added_jasmine
  let new_volume := initial_volume + added_jasmine + added_water
  let new_percentage := (new_jasmine / new_volume) * 100
  new_percentage = 12.5 := by
sorry

end new_jasmine_percentage_l3397_339722


namespace average_monthly_sales_l3397_339710

def january_sales : ℝ := 150
def february_sales : ℝ := 90
def march_sales : ℝ := 60
def april_sales : ℝ := 140
def may_sales_before_discount : ℝ := 100
def discount_rate : ℝ := 0.2

def may_sales : ℝ := may_sales_before_discount * (1 - discount_rate)

def total_sales : ℝ := january_sales + february_sales + march_sales + april_sales + may_sales

def number_of_months : ℕ := 5

theorem average_monthly_sales :
  total_sales / number_of_months = 104 := by sorry

end average_monthly_sales_l3397_339710


namespace cubic_function_property_l3397_339707

/-- Given a cubic function f(x) = ax³ + bx - 2 where f(2015) = 7, prove that f(-2015) = -11 -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x - 2
  f 2015 = 7 → f (-2015) = -11 := by
  sorry

end cubic_function_property_l3397_339707


namespace grants_test_score_l3397_339704

theorem grants_test_score (hunter_score john_score grant_score : ℕ) :
  hunter_score = 45 →
  john_score = 2 * hunter_score →
  grant_score = john_score + 10 →
  grant_score = 100 := by
sorry

end grants_test_score_l3397_339704


namespace inequality_proof_l3397_339798

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (1 - 2*x) / Real.sqrt (x * (1 - x)) + 
  (1 - 2*y) / Real.sqrt (y * (1 - y)) + 
  (1 - 2*z) / Real.sqrt (z * (1 - z)) ≥ 
  Real.sqrt (x / (1 - x)) + 
  Real.sqrt (y / (1 - y)) + 
  Real.sqrt (z / (1 - z)) := by
sorry

end inequality_proof_l3397_339798


namespace abs_two_x_minus_one_lt_one_l3397_339700

theorem abs_two_x_minus_one_lt_one (x y : ℝ) 
  (h1 : |x - y - 1| ≤ 1/3) 
  (h2 : |2*y + 1| ≤ 1/6) : 
  |2*x - 1| < 1 := by
sorry

end abs_two_x_minus_one_lt_one_l3397_339700


namespace expected_value_of_segments_expected_value_is_1037_l3397_339760

/-- The number of points in the plane -/
def n : ℕ := 100

/-- The number of pairs connected by line segments -/
def connected_pairs : ℕ := 4026

/-- A function to calculate binomial coefficient -/
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/-- The theorem to prove -/
theorem expected_value_of_segments (no_three_collinear : True) 
  (all_points_unique : True) : ℝ :=
  let total_pairs := choose n 2
  let diff_50_pairs := choose 51 2
  let prob_segment := connected_pairs / total_pairs
  prob_segment * diff_50_pairs

/-- The main theorem stating the expected value is 1037 -/
theorem expected_value_is_1037 (no_three_collinear : True) 
  (all_points_unique : True) :
  expected_value_of_segments no_three_collinear all_points_unique = 1037 := by
  sorry

end expected_value_of_segments_expected_value_is_1037_l3397_339760


namespace chords_intersection_theorem_l3397_339743

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a chord
structure Chord (c : Circle) where
  p1 : Point
  p2 : Point

-- Define the length of a segment
def segmentLength (p1 p2 : Point) : ℝ := sorry

-- Define a right angle
def isRightAngle (p1 p2 p3 : Point) : Prop := sorry

-- Theorem statement
theorem chords_intersection_theorem (c : Circle) (ab cd : Chord c) (e : Point) :
  isRightAngle ab.p1 e cd.p1 →
  (segmentLength ab.p1 e)^2 + (segmentLength ab.p2 e)^2 + 
  (segmentLength cd.p1 e)^2 + (segmentLength cd.p2 e)^2 = 
  (2 * c.radius)^2 := by
  sorry

end chords_intersection_theorem_l3397_339743


namespace race_results_l3397_339758

-- Define the race parameters
def race_distance : ℝ := 200
def time_A : ℝ := 40
def time_B : ℝ := 50
def time_C : ℝ := 45

-- Define the time differences
def time_diff_AB : ℝ := time_B - time_A
def time_diff_AC : ℝ := time_C - time_A
def time_diff_BC : ℝ := time_C - time_B

-- Theorem statement
theorem race_results :
  (time_diff_AB = 10) ∧
  (time_diff_AC = 5) ∧
  (time_diff_BC = -5) := by
  sorry

end race_results_l3397_339758


namespace function_properties_l3397_339799

-- Define the function f(x) = -2x + 1
def f (x : ℝ) : ℝ := -2 * x + 1

-- State the theorem
theorem function_properties :
  (f 1 = -1) ∧ 
  (∃ (x y z : ℝ), x > 0 ∧ y < 0 ∧ z > 0 ∧ f x > 0 ∧ f y < 0 ∧ f z < 0) ∧
  (∀ (x y : ℝ), x < y → f x > f y) ∧
  (∃ (x : ℝ), x > 0 ∧ f x ≤ 1) :=
by sorry

end function_properties_l3397_339799


namespace parallelogram_area_14_24_l3397_339770

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 14 cm and height 24 cm is 336 cm² -/
theorem parallelogram_area_14_24 : parallelogramArea 14 24 = 336 := by
  sorry

end parallelogram_area_14_24_l3397_339770


namespace ship_arrangement_count_l3397_339709

/-- The number of ways to select and arrange ships for tasks -/
def arrange_ships (destroyers frigates selected : ℕ) (tasks : ℕ) : ℕ :=
  (Nat.choose (destroyers + frigates) selected - Nat.choose frigates selected) * Nat.factorial tasks

/-- Theorem stating the correct number of arrangements -/
theorem ship_arrangement_count :
  arrange_ships 2 6 3 3 = 216 := by
  sorry

end ship_arrangement_count_l3397_339709


namespace complex_number_coordinates_l3397_339718

theorem complex_number_coordinates (a : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a - 2 : ℂ) + (a + 1) * i = b * i) (h3 : b ≠ 0) :
  (a - 3 * i) / (2 - i) = 7/5 - 4/5 * i :=
sorry

end complex_number_coordinates_l3397_339718


namespace inscribed_cube_volume_l3397_339745

theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  inner_cube_edge ^ 3 = 192 * Real.sqrt 3 := by
  sorry

end inscribed_cube_volume_l3397_339745


namespace polynomial_simplification_l3397_339723

theorem polynomial_simplification (s r : ℝ) :
  (2 * s^2 + 5 * r - 4) - (3 * s^2 + 9 * r - 7) = -s^2 - 4 * r + 3 := by
  sorry

end polynomial_simplification_l3397_339723


namespace cable_theorem_l3397_339791

def cable_problem (basic_cost movie_cost sports_cost_diff : ℕ) : Prop :=
  let sports_cost := movie_cost - sports_cost_diff
  let total_cost := basic_cost + movie_cost + sports_cost
  total_cost = 36

theorem cable_theorem : cable_problem 15 12 3 :=
sorry

end cable_theorem_l3397_339791


namespace x_in_terms_of_abc_l3397_339784

theorem x_in_terms_of_abc (x y z a b c : ℝ) 
  (h1 : x * y / (x + y + 1) = a)
  (h2 : x * z / (x + z + 1) = b)
  (h3 : y * z / (y + z + 1) = c)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hab : a * b + a * c - b * c ≠ 0) :
  x = 2 * a * b * c / (a * b + a * c - b * c) :=
sorry

end x_in_terms_of_abc_l3397_339784


namespace flute_players_count_l3397_339763

/-- The number of people in an orchestra with specified instrument counts -/
structure Orchestra :=
  (total : ℕ)
  (drums : ℕ)
  (trombone : ℕ)
  (trumpet : ℕ)
  (french_horn : ℕ)
  (violinist : ℕ)
  (cellist : ℕ)
  (contrabassist : ℕ)
  (clarinet : ℕ)
  (conductor : ℕ)

/-- Theorem stating that the number of flute players is 4 -/
theorem flute_players_count (o : Orchestra) 
  (h1 : o.total = 21)
  (h2 : o.drums = 1)
  (h3 : o.trombone = 4)
  (h4 : o.trumpet = 2)
  (h5 : o.french_horn = 1)
  (h6 : o.violinist = 3)
  (h7 : o.cellist = 1)
  (h8 : o.contrabassist = 1)
  (h9 : o.clarinet = 3)
  (h10 : o.conductor = 1) :
  o.total - (o.drums + o.trombone + o.trumpet + o.french_horn + 
             o.violinist + o.cellist + o.contrabassist + 
             o.clarinet + o.conductor) = 4 := by
  sorry


end flute_players_count_l3397_339763


namespace man_upstream_speed_l3397_339796

/-- Given a man's speed in still water and his speed downstream, 
    calculates his speed upstream -/
def speed_upstream (speed_still : ℝ) (speed_downstream : ℝ) : ℝ :=
  2 * speed_still - speed_downstream

/-- Theorem stating that for a man with speed 60 kmph in still water 
    and 65 kmph downstream, his upstream speed is 55 kmph -/
theorem man_upstream_speed :
  speed_upstream 60 65 = 55 := by
  sorry


end man_upstream_speed_l3397_339796


namespace garrison_size_l3397_339786

/-- The initial number of days the provisions would last -/
def initial_days : ℕ := 28

/-- The number of days that passed before reinforcements arrived -/
def days_before_reinforcement : ℕ := 12

/-- The number of men that arrived as reinforcement -/
def reinforcement : ℕ := 1110

/-- The number of days the provisions would last after reinforcement arrived -/
def remaining_days : ℕ := 10

/-- The initial number of men in the garrison -/
def initial_men : ℕ := 1850

theorem garrison_size :
  ∃ (M : ℕ),
    M * initial_days = 
    (M + reinforcement) * remaining_days + 
    M * days_before_reinforcement ∧
    M = initial_men :=
by sorry

end garrison_size_l3397_339786


namespace common_tangents_count_l3397_339715

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

-- Define the number of common tangents
def num_common_tangents : ℕ := 2

-- Theorem statement
theorem common_tangents_count : 
  ∃ (n : ℕ), n = num_common_tangents ∧ 
  (∀ x y : ℝ, C1 x y ∨ C2 x y → n = 2) :=
sorry

end common_tangents_count_l3397_339715


namespace unique_solution_quadratic_l3397_339733

/-- For a quadratic equation 9x^2 + kx + 49 = 0 to have exactly one real solution,
    the positive value of k must be 42. -/
theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + k * x + 49 = 0) ↔ k = 42 := by
  sorry

end unique_solution_quadratic_l3397_339733


namespace multiple_properties_l3397_339785

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 5 * k) 
  (hb : ∃ m : ℤ, b = 10 * m) : 
  (∃ n : ℤ, b = 5 * n) ∧ 
  (∃ p : ℤ, a + b = 5 * p) ∧ 
  (∃ q : ℤ, a + b = 2 * q) := by
  sorry

end multiple_properties_l3397_339785


namespace complex_expression_equals_25_1_l3397_339789

theorem complex_expression_equals_25_1 :
  (50 + 5 * (12 / (180 / 3))^2) * Real.sin (30 * π / 180) = 25.1 := by
  sorry

end complex_expression_equals_25_1_l3397_339789


namespace equation_solution_l3397_339742

theorem equation_solution :
  ∃ x : ℚ, (3 * x - 15) / 4 = (x + 7) / 3 ∧ x = 73 / 5 := by
  sorry

end equation_solution_l3397_339742


namespace kaleb_final_amount_l3397_339731

def kaleb_business (spring_earnings summer_earnings supplies_cost : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supplies_cost

theorem kaleb_final_amount :
  kaleb_business 4 50 4 = 50 := by
  sorry

end kaleb_final_amount_l3397_339731
