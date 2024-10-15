import Mathlib

namespace NUMINAMATH_CALUDE_power_multiplication_l726_72694

theorem power_multiplication (m : ℝ) : (m^2)^3 * m^4 = m^10 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l726_72694


namespace NUMINAMATH_CALUDE_average_rope_length_l726_72651

theorem average_rope_length (rope1 rope2 : ℝ) (h1 : rope1 = 2) (h2 : rope2 = 6) :
  (rope1 + rope2) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_rope_length_l726_72651


namespace NUMINAMATH_CALUDE_binomial_18_6_l726_72675

theorem binomial_18_6 : Nat.choose 18 6 = 13260 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_6_l726_72675


namespace NUMINAMATH_CALUDE_xy_equation_l726_72687

theorem xy_equation (x y : ℝ) 
  (h1 : (x + y) / 3 = 1.888888888888889)
  (h2 : x + 2 * y = 10) :
  x + y = 5.666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_xy_equation_l726_72687


namespace NUMINAMATH_CALUDE_cubic_monomial_exists_l726_72657

/-- A cubic monomial with variables x and y and a negative coefficient exists. -/
theorem cubic_monomial_exists : ∃ (a : ℝ) (i j : ℕ), 
  a < 0 ∧ i + j = 3 ∧ (λ (x y : ℝ) => a * x^i * y^j) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_monomial_exists_l726_72657


namespace NUMINAMATH_CALUDE_stating_no_equal_area_division_for_n_gt_2_l726_72608

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents an angle bisector in a triangle -/
structure AngleBisector where
  origin : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- 
  Given a triangle and a set of angle bisectors from one vertex, 
  checks if they divide the triangle into n equal-area parts
-/
def divideIntoEqualAreas (t : Triangle) (bisectors : List AngleBisector) (n : ℕ) : Prop :=
  sorry

/-- 
  Theorem stating that for all triangles and integers n > 2, 
  it is impossible for the angle bisectors of one of the triangle's vertices 
  to divide the triangle into n equal-area parts
-/
theorem no_equal_area_division_for_n_gt_2 :
  ∀ (t : Triangle) (n : ℕ), n > 2 → ¬∃ (bisectors : List AngleBisector), 
  divideIntoEqualAreas t bisectors n :=
sorry

end NUMINAMATH_CALUDE_stating_no_equal_area_division_for_n_gt_2_l726_72608


namespace NUMINAMATH_CALUDE_ratio_and_mean_problem_l726_72610

theorem ratio_and_mean_problem (a b c : ℕ+) (h_ratio : (a : ℚ) / b = 2 / 3 ∧ (b : ℚ) / c = 3 / 4)
  (h_mean : (a + b + c : ℚ) / 3 = 42) : a = 28 := by
  sorry

end NUMINAMATH_CALUDE_ratio_and_mean_problem_l726_72610


namespace NUMINAMATH_CALUDE_michael_fish_count_l726_72619

theorem michael_fish_count (initial_fish : ℕ) (given_fish : ℕ) : 
  initial_fish = 31 → given_fish = 18 → initial_fish + given_fish = 49 := by
sorry

end NUMINAMATH_CALUDE_michael_fish_count_l726_72619


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l726_72677

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → Real.log x - x + 1 ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ Real.log x - x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l726_72677


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l726_72684

theorem imaginary_part_of_complex_fraction : Complex.im (4 * I / (1 - I)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l726_72684


namespace NUMINAMATH_CALUDE_words_per_page_l726_72603

/-- Calculates the number of words per page in books Sarah is reading --/
theorem words_per_page
  (reading_speed : ℕ)  -- Sarah's reading speed in words per minute
  (reading_time : ℕ)   -- Total reading time in hours
  (num_books : ℕ)      -- Number of books Sarah plans to read
  (pages_per_book : ℕ) -- Number of pages in each book
  (h1 : reading_speed = 40)
  (h2 : reading_time = 20)
  (h3 : num_books = 6)
  (h4 : pages_per_book = 80)
  : (reading_speed * 60 * reading_time) / (num_books * pages_per_book) = 100 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_l726_72603


namespace NUMINAMATH_CALUDE_sum_of_inscribed_circle_areas_l726_72645

/-- Given a triangle ABC with sides a, b, c, and an inscribed circle of radius r,
    prove that the sum of the areas of four inscribed circles
    (one in the original triangle and three in the smaller triangles formed by
    tangents parallel to the sides) is equal to π r² · (a² + b² + c²) / s²,
    where s is the semi-perimeter of the triangle. -/
theorem sum_of_inscribed_circle_areas
  (a b c r : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (s : ℝ) (h_s : s = (a + b + c) / 2)
  (h_inradius : r = s / ((a + b + c) / 2)) :
  let original_circle_area := π * r^2
  let smaller_circles_area := π * r^2 * ((s - a)^2 + (s - b)^2 + (s - c)^2) / s^2
  original_circle_area + smaller_circles_area = π * r^2 * (a^2 + b^2 + c^2) / s^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_inscribed_circle_areas_l726_72645


namespace NUMINAMATH_CALUDE_interview_probability_correct_l726_72600

structure TouristGroup where
  total : ℕ
  outside_fraction : ℚ
  inside_fraction : ℚ
  gold_fraction : ℚ
  silver_fraction : ℚ

def interview_probability (group : TouristGroup) : ℚ × ℚ :=
  let outside := (group.total : ℚ) * group.outside_fraction
  let inside := (group.total : ℚ) * group.inside_fraction
  let gold := outside * group.gold_fraction
  let silver := inside * group.silver_fraction
  let no_card := group.total - (gold + silver)
  let prob_one_silver := (silver * (group.total - silver)) / ((group.total * (group.total - 1)) / 2)
  let prob_equal := (((no_card * (no_card - 1)) / 2) + gold * silver) / ((group.total * (group.total - 1)) / 2)
  (prob_one_silver, prob_equal)

theorem interview_probability_correct (group : TouristGroup) 
  (h1 : group.total = 36)
  (h2 : group.outside_fraction = 3/4)
  (h3 : group.inside_fraction = 1/4)
  (h4 : group.gold_fraction = 1/3)
  (h5 : group.silver_fraction = 2/3) :
  interview_probability group = (2/7, 44/105) := by
  sorry

end NUMINAMATH_CALUDE_interview_probability_correct_l726_72600


namespace NUMINAMATH_CALUDE_factor_calculation_l726_72641

theorem factor_calculation : ∃ (f : ℚ), 
  let initial_number := 10
  let doubled_plus_eight := 2 * initial_number + 8
  f * doubled_plus_eight = 84 ∧ f = 3 := by sorry

end NUMINAMATH_CALUDE_factor_calculation_l726_72641


namespace NUMINAMATH_CALUDE_trajectory_equation_l726_72665

-- Define the point M
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition for point M
def satisfiesCondition (M : Point) : Prop :=
  Real.sqrt ((M.y + 5)^2 + M.x^2) - Real.sqrt ((M.y - 5)^2 + M.x^2) = 8

-- Define the trajectory equation
def isOnTrajectory (M : Point) : Prop :=
  M.y^2 / 16 - M.x^2 / 9 = 1 ∧ M.y > 0

-- Theorem statement
theorem trajectory_equation (M : Point) :
  satisfiesCondition M → isOnTrajectory M :=
by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l726_72665


namespace NUMINAMATH_CALUDE_simplify_expression_l726_72614

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^2 + y^2)⁻¹ * (x⁻¹ + y⁻¹) = (x^3*y + x*y^3)⁻¹ * (x + y) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l726_72614


namespace NUMINAMATH_CALUDE_plane_perpendicularity_condition_l726_72655

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the properties and relations
variable (subset : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity_condition 
  (α β : Plane) (a : Line) 
  (h_subset : subset a α) :
  (∀ (a : Line), subset a α → perpendicular_line_plane a β → perpendicular_plane_plane α β) ∧ 
  (∃ (a : Line), subset a α ∧ perpendicular_plane_plane α β ∧ ¬perpendicular_line_plane a β) :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_condition_l726_72655


namespace NUMINAMATH_CALUDE_polynomial_sum_l726_72648

-- Define the polynomials
def f (x : ℝ) : ℝ := 2*x^3 - 4*x^2 + 2*x - 5
def g (x : ℝ) : ℝ := -3*x^2 + 4*x - 7
def h (x : ℝ) : ℝ := 6*x^3 + x^2 + 3*x + 2

-- State the theorem
theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = 8*x^3 - 6*x^2 + 9*x - 10 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l726_72648


namespace NUMINAMATH_CALUDE_fraction_subtraction_equals_two_l726_72631

theorem fraction_subtraction_equals_two (a : ℝ) (h : a ≠ 1) :
  (2 * a) / (a - 1) - 2 / (a - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equals_two_l726_72631


namespace NUMINAMATH_CALUDE_number_puzzle_l726_72625

theorem number_puzzle : ∃ x : ℝ, (x / 5 + 6 = 65) ∧ (x = 295) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l726_72625


namespace NUMINAMATH_CALUDE_g_max_value_l726_72639

/-- The function g(x) defined for x > 0 -/
noncomputable def g (x : ℝ) : ℝ := x * Real.log (1 + 1/x) + (1/x) * Real.log (1 + x)

/-- Theorem stating that the maximum value of g(x) for x > 0 is 2ln2 -/
theorem g_max_value : ∃ (M : ℝ), M = 2 * Real.log 2 ∧ ∀ x > 0, g x ≤ M :=
sorry

end NUMINAMATH_CALUDE_g_max_value_l726_72639


namespace NUMINAMATH_CALUDE_two_digit_sums_of_six_powers_of_two_l726_72629

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_sum_of_six_powers_of_two (n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧
    n = 2^0 + 2^a + 2^b + 2^c + 2^d + 2^e + 2^f

theorem two_digit_sums_of_six_powers_of_two :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, is_two_digit n ∧ is_sum_of_six_powers_of_two n) ∧
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_two_digit_sums_of_six_powers_of_two_l726_72629


namespace NUMINAMATH_CALUDE_christmas_decorations_distribution_l726_72602

/-- The number of decorations in each box -/
def decorations_per_box : ℕ := 10

/-- The total number of decorations handed out -/
def total_decorations : ℕ := 120

/-- The number of families who received a box of decorations -/
def num_families : ℕ := 11

theorem christmas_decorations_distribution :
  decorations_per_box * (num_families + 1) = total_decorations :=
sorry

end NUMINAMATH_CALUDE_christmas_decorations_distribution_l726_72602


namespace NUMINAMATH_CALUDE_card_arrangement_count_l726_72650

/-- Represents a board with a given number of cells -/
structure Board :=
  (cells : Nat)

/-- Represents a set of cards with a given count -/
structure CardSet :=
  (count : Nat)

/-- Calculates the number of possible arrangements of cards on a board -/
def possibleArrangements (board : Board) (cards : CardSet) : Nat :=
  board.cells - cards.count + 1

/-- The theorem to be proved -/
theorem card_arrangement_count :
  let board := Board.mk 1994
  let cards := CardSet.mk 1000
  let arrangements := possibleArrangements board cards
  arrangements = 995 ∧ arrangements < 500000 := by
  sorry

end NUMINAMATH_CALUDE_card_arrangement_count_l726_72650


namespace NUMINAMATH_CALUDE_player1_receives_57_coins_l726_72682

/-- Represents the number of players and sectors on the table -/
def n : ℕ := 9

/-- Represents the total number of rotations -/
def total_rotations : ℕ := 11

/-- Represents the coins received by player 4 -/
def player4_coins : ℕ := 90

/-- Represents the coins received by player 8 -/
def player8_coins : ℕ := 35

/-- Represents the coins received by player 1 -/
def player1_coins : ℕ := 57

/-- Theorem stating that given the conditions, player 1 receives 57 coins -/
theorem player1_receives_57_coins :
  n = 9 →
  total_rotations = 11 →
  player4_coins = 90 →
  player8_coins = 35 →
  player1_coins = 57 :=
by sorry

end NUMINAMATH_CALUDE_player1_receives_57_coins_l726_72682


namespace NUMINAMATH_CALUDE_worst_player_is_niece_l726_72666

-- Define the players
inductive Player
| Grandmother
| Niece
| Grandson
| SonInLaw

-- Define the sex of a player
inductive Sex
| Male
| Female

-- Define the generation of a player
inductive Generation
| Old
| Middle
| Young

-- Function to determine the sex of a player
def sex : Player → Sex
| Player.Grandmother => Sex.Female
| Player.Niece => Sex.Female
| Player.Grandson => Sex.Male
| Player.SonInLaw => Sex.Male

-- Function to determine the generation of a player
def generation : Player → Generation
| Player.Grandmother => Generation.Old
| Player.Niece => Generation.Young
| Player.Grandson => Generation.Young
| Player.SonInLaw => Generation.Middle

-- Function to determine if two players are cousins
def areCousins : Player → Player → Bool
| Player.Niece, Player.Grandson => true
| Player.Grandson, Player.Niece => true
| _, _ => false

-- Theorem statement
theorem worst_player_is_niece :
  ∀ (worst best : Player),
  (∃ cousin : Player, areCousins worst cousin ∧ sex cousin ≠ sex best) →
  generation worst ≠ generation best →
  worst = Player.Niece :=
by sorry

end NUMINAMATH_CALUDE_worst_player_is_niece_l726_72666


namespace NUMINAMATH_CALUDE_birds_in_tree_l726_72662

theorem birds_in_tree (initial_birds : ℕ) (new_birds : ℕ) (total_birds : ℕ) : 
  initial_birds = 14 → new_birds = 21 → total_birds = initial_birds + new_birds → total_birds = 35 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l726_72662


namespace NUMINAMATH_CALUDE_even_function_sine_condition_l726_72676

theorem even_function_sine_condition 
  (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) :
  (∀ x : ℝ, A * Real.sin (ω * x + φ) = A * Real.sin (ω * (-x) + φ)) ↔ 
  ∃ k : ℤ, φ = k * π + π / 2 := by
sorry

end NUMINAMATH_CALUDE_even_function_sine_condition_l726_72676


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l726_72632

/-- The difference between the larger and smaller x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_difference : ∃ (a c : ℝ),
  (3 * a^2 - 6 * a + 2 = -2 * a^2 - 4 * a + 3) ∧
  (3 * c^2 - 6 * c + 2 = -2 * c^2 - 4 * c + 3) ∧
  (c ≥ a) ∧
  (c - a = 2 * Real.sqrt 6 / 5) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l726_72632


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l726_72612

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line
def line_l : Set (ℝ × ℝ) := {p | p.1 = -3}

-- Define point A
def point_A : ℝ × ℝ := (3, 0)

-- Define the properties of the moving circle
def is_valid_circle (c : Circle) : Prop :=
  (c.center.1 - 3)^2 + c.center.2^2 = c.radius^2 ∧  -- passes through A(3,0)
  c.center.1 + c.radius = -3                        -- tangent to x = -3

-- Theorem statement
theorem trajectory_is_parabola :
  ∀ c : Circle, is_valid_circle c →
  ∃ x y : ℝ, c.center = (x, y) ∧ y^2 = 12 * x :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l726_72612


namespace NUMINAMATH_CALUDE_length_AB_l726_72681

/-- Parabola C: y^2 = 8x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 8*x

/-- Line l: y = (√3/3)(x-2) -/
def line_l (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - 2)

/-- A and B are intersection points of C and l -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola_C A.1 A.2 ∧ line_l A.1 A.2 ∧
  parabola_C B.1 B.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

/-- The length of AB is 32 -/
theorem length_AB (A B : ℝ × ℝ) (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 32 := by sorry

end NUMINAMATH_CALUDE_length_AB_l726_72681


namespace NUMINAMATH_CALUDE_planes_parallel_from_parallel_perp_lines_l726_72637

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_from_parallel_perp_lines 
  (m n : Line) (α β : Plane) :
  m ≠ n →
  parallel m n →
  perp_line_plane m α →
  perp_line_plane n β →
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_from_parallel_perp_lines_l726_72637


namespace NUMINAMATH_CALUDE_val_coins_value_l726_72672

/-- Calculates the total value of Val's coins given the initial number of nickels and the number of additional nickels found. -/
def total_value (initial_nickels : ℕ) (found_nickels : ℕ) : ℚ :=
  let total_nickels := initial_nickels + found_nickels
  let dimes := 3 * initial_nickels
  let quarters := 2 * dimes
  let nickel_value := (5 : ℚ) / 100
  let dime_value := (10 : ℚ) / 100
  let quarter_value := (25 : ℚ) / 100
  (total_nickels : ℚ) * nickel_value + (dimes : ℚ) * dime_value + (quarters : ℚ) * quarter_value

theorem val_coins_value :
  total_value 20 40 = 39 := by
  sorry

end NUMINAMATH_CALUDE_val_coins_value_l726_72672


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l726_72647

/-- Given a geometric sequence of positive integers where the first term is 5 and the fourth term is 405,
    prove that the fifth term is 405. -/
theorem fifth_term_of_geometric_sequence (a : ℕ+) (r : ℕ+) : 
  a = 5 → a * r^3 = 405 → a * r^4 = 405 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l726_72647


namespace NUMINAMATH_CALUDE_negative_sqrt_six_squared_equals_six_l726_72695

theorem negative_sqrt_six_squared_equals_six : (-Real.sqrt 6)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_six_squared_equals_six_l726_72695


namespace NUMINAMATH_CALUDE_perfect_squares_difference_l726_72696

theorem perfect_squares_difference (m : ℕ+) : 
  (∃ a : ℕ, m - 4 = a^2) ∧ (∃ b : ℕ, m + 5 = b^2) → m = 20 ∨ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_difference_l726_72696


namespace NUMINAMATH_CALUDE_parcel_boxes_count_l726_72617

/-- Represents the position of a parcel in a rectangular arrangement of boxes -/
structure ParcelPosition where
  left : Nat
  right : Nat
  front : Nat
  back : Nat

/-- Calculates the total number of parcel boxes given the position of a specific parcel -/
def totalParcelBoxes (pos : ParcelPosition) : Nat :=
  (pos.left + pos.right - 1) * (pos.front + pos.back - 1)

/-- Theorem stating that given the specific parcel position, the total number of boxes is 399 -/
theorem parcel_boxes_count (pos : ParcelPosition) 
  (h_left : pos.left = 7)
  (h_right : pos.right = 13)
  (h_front : pos.front = 8)
  (h_back : pos.back = 14) : 
  totalParcelBoxes pos = 399 := by
  sorry

#eval totalParcelBoxes ⟨7, 13, 8, 14⟩

end NUMINAMATH_CALUDE_parcel_boxes_count_l726_72617


namespace NUMINAMATH_CALUDE_carnation_bouquet_combinations_l726_72667

def distribute_carnations (total : ℕ) (types : ℕ) (extras : ℕ) : ℕ :=
  Nat.choose (extras + types - 1) (types - 1)

theorem carnation_bouquet_combinations :
  distribute_carnations 5 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_carnation_bouquet_combinations_l726_72667


namespace NUMINAMATH_CALUDE_complex_equation_solution_l726_72628

theorem complex_equation_solution :
  ∀ z : ℂ, (Complex.I - 1) * z = 2 → z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l726_72628


namespace NUMINAMATH_CALUDE_hua_luogeng_birthday_factorization_l726_72646

theorem hua_luogeng_birthday_factorization :
  (1163 : ℕ).Prime ∧ ¬(16424 : ℕ).Prime :=
by
  have h : 19101112 = 1163 * 16424 := by rfl
  sorry

end NUMINAMATH_CALUDE_hua_luogeng_birthday_factorization_l726_72646


namespace NUMINAMATH_CALUDE_max_profit_month_and_value_l726_72652

def f (x : ℕ) : ℝ := -3 * x^2 + 40 * x

def q (x : ℕ) : ℝ := 150 + 2 * x

def profit (x : ℕ) : ℝ := (185 - q x) * f x

theorem max_profit_month_and_value :
  ∃ (x : ℕ), 1 ≤ x ∧ x ≤ 12 ∧
  (∀ (y : ℕ), 1 ≤ y ∧ y ≤ 12 → profit y ≤ profit x) ∧
  x = 5 ∧ profit x = 3125 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_month_and_value_l726_72652


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l726_72692

/-- Given a paint mixture ratio of 3:2:4 for blue:green:white paint, 
    if 12 quarts of white paint are used, then 6 quarts of green paint should be used. -/
theorem paint_mixture_ratio (blue green white : ℚ) : 
  blue / green = 3 / 2 ∧ 
  green / white = 2 / 4 ∧ 
  white = 12 → 
  green = 6 := by
sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l726_72692


namespace NUMINAMATH_CALUDE_count_wings_l726_72640

/-- The number of planes in the air exhibition -/
def num_planes : ℕ := 54

/-- The number of wings per plane -/
def wings_per_plane : ℕ := 2

/-- The total number of wings counted -/
def total_wings : ℕ := num_planes * wings_per_plane

theorem count_wings : total_wings = 108 := by
  sorry

end NUMINAMATH_CALUDE_count_wings_l726_72640


namespace NUMINAMATH_CALUDE_omega_range_l726_72609

/-- Given a function f(x) = 2sin(ωx) with ω > 0, if f(x) has a minimum value of -2 
    on the interval [-π/3, π/4], then 0 < ω ≤ 3/2 -/
theorem omega_range (ω : ℝ) (h1 : ω > 0) : 
  (∀ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω * x) ≥ -2) →
  (∃ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω * x) = -2) →
  0 < ω ∧ ω ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_omega_range_l726_72609


namespace NUMINAMATH_CALUDE_cross_section_area_fraction_l726_72670

theorem cross_section_area_fraction (r : ℝ) (r_pos : r > 0) : 
  let sphere_surface_area := 4 * Real.pi * r^2
  let cross_section_radius := r / 2
  let cross_section_area := Real.pi * cross_section_radius^2
  cross_section_area / sphere_surface_area = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_fraction_l726_72670


namespace NUMINAMATH_CALUDE_board_cut_multiple_l726_72606

/-- Given a board of 120 cm cut into two pieces, where the shorter piece is 35 cm
    and the longer piece is 15 cm longer than m times the shorter piece, m must equal 2. -/
theorem board_cut_multiple (m : ℝ) : 
  (35 : ℝ) + (m * 35 + 15) = 120 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_multiple_l726_72606


namespace NUMINAMATH_CALUDE_chessboard_symmetry_l726_72605

-- Define a chessboard
structure Chessboard :=
  (ranks : Fin 8)
  (files : Fin 8)

-- Define a chess square
structure Square :=
  (file : Char)
  (rank : Nat)

-- Define symmetry on the chessboard
def symmetric (s1 s2 : Square) (b : Chessboard) : Prop :=
  s1.file = s2.file ∧ s1.rank + s2.rank = 9

-- Define the line of symmetry
def lineOfSymmetry (b : Chessboard) : Prop :=
  ∀ (s1 s2 : Square), symmetric s1 s2 b → (s1.rank = 4 ∧ s2.rank = 5) ∨ (s1.rank = 5 ∧ s2.rank = 4)

-- Theorem statement
theorem chessboard_symmetry (b : Chessboard) :
  lineOfSymmetry b ∧
  symmetric (Square.mk 'e' 2) (Square.mk 'e' 7) b ∧
  symmetric (Square.mk 'h' 5) (Square.mk 'h' 4) b :=
sorry

end NUMINAMATH_CALUDE_chessboard_symmetry_l726_72605


namespace NUMINAMATH_CALUDE_marly_soup_containers_l726_72616

/-- The number of containers needed to store Marly's soup -/
def containers_needed (milk chicken_stock pureed_vegetables other_ingredients container_capacity : ℚ) : ℕ :=
  let total_soup := milk + chicken_stock + pureed_vegetables + other_ingredients
  (total_soup / container_capacity).ceil.toNat

/-- Proof that Marly needs 28 containers for his soup -/
theorem marly_soup_containers :
  containers_needed 15 (3 * 15) 5 4 (5/2) = 28 := by
  sorry

end NUMINAMATH_CALUDE_marly_soup_containers_l726_72616


namespace NUMINAMATH_CALUDE_random_walk_properties_l726_72626

/-- Represents a random walk on a line -/
structure RandomWalk where
  a : ℕ  -- number of steps to the right
  b : ℕ  -- number of steps to the left
  h : a > b

/-- The maximum possible range of the random walk -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of the random walk -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences achieving the maximum range -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

theorem random_walk_properties (w : RandomWalk) :
  (max_range w = w.a) ∧
  (min_range w = w.a - w.b) ∧
  (max_range_sequences w = w.b + 1) := by
  sorry

end NUMINAMATH_CALUDE_random_walk_properties_l726_72626


namespace NUMINAMATH_CALUDE_triangle_medians_theorem_l726_72654

/-- Given a triangle with side lengths a, b, and c, and orthogonal medians m_a and m_b -/
def Triangle (a b c : ℝ) (m_a m_b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  m_a = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2) ∧
  m_b = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2) ∧
  m_a * m_b = 0  -- orthogonality condition

theorem triangle_medians_theorem {a b c m_a m_b : ℝ} (h : Triangle a b c m_a m_b) :
  let m_c := (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)
  -- 1. The medians form a right-angled triangle
  m_a^2 + m_b^2 = m_c^2 ∧
  -- 2. The inequality holds
  5*(a^2 + b^2 - c^2) ≥ 8*a*b :=
by sorry

end NUMINAMATH_CALUDE_triangle_medians_theorem_l726_72654


namespace NUMINAMATH_CALUDE_third_bouquet_carnations_l726_72611

/-- Theorem: Given three bouquets of carnations with specific conditions, 
    the third bouquet contains 13 carnations. -/
theorem third_bouquet_carnations 
  (total_bouquets : ℕ)
  (first_bouquet : ℕ)
  (second_bouquet : ℕ)
  (average_carnations : ℕ)
  (h1 : total_bouquets = 3)
  (h2 : first_bouquet = 9)
  (h3 : second_bouquet = 14)
  (h4 : average_carnations = 12)
  (h5 : average_carnations * total_bouquets = first_bouquet + second_bouquet + (total_bouquets - 2)) :
  total_bouquets - 2 = 13 := by
  sorry


end NUMINAMATH_CALUDE_third_bouquet_carnations_l726_72611


namespace NUMINAMATH_CALUDE_one_fourth_of_8_2_l726_72633

theorem one_fourth_of_8_2 : (8.2 : ℚ) / 4 = 41 / 20 := by sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_2_l726_72633


namespace NUMINAMATH_CALUDE_rounding_proof_l726_72624

def base : ℚ := 1003 / 1000

def power : ℕ := 4

def exact_result : ℚ := base ^ power

def rounded_result : ℚ := 1012 / 1000

def decimal_places : ℕ := 3

theorem rounding_proof : 
  (round (exact_result * 10^decimal_places) / 10^decimal_places) = rounded_result := by
  sorry

end NUMINAMATH_CALUDE_rounding_proof_l726_72624


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l726_72671

theorem smallest_angle_in_triangle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  max a (max b c) = 120 →  -- The largest angle is 120°
  b / c = 3 / 2 →  -- Ratio of the other two angles is 3:2
  b > c →  -- Ensure b is the middle angle and c is the smallest
  c = 24 :=  -- The smallest angle is 24°
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l726_72671


namespace NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l726_72604

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line in 3D space
  -- This is a placeholder structure

/-- Two lines are skew if they are not coplanar -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- Two lines have no common point if they don't intersect -/
def have_no_common_point (l1 l2 : Line3D) : Prop :=
  sorry

/-- Theorem stating that "are_skew" is a sufficient but not necessary condition for "have_no_common_point" -/
theorem skew_lines_sufficient_not_necessary :
  ∃ (l1 l2 l3 l4 : Line3D),
    (are_skew l1 l2 → have_no_common_point l1 l2) ∧
    (have_no_common_point l3 l4 ∧ ¬are_skew l3 l4) :=
  sorry

end NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l726_72604


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l726_72635

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 13) 
  (sum_prod_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1027 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l726_72635


namespace NUMINAMATH_CALUDE_train_speed_calculation_l726_72664

-- Define the given constants
def train_length : ℝ := 160
def bridge_length : ℝ := 215
def crossing_time : ℝ := 30

-- Define the speed conversion factor
def m_per_s_to_km_per_hr : ℝ := 3.6

-- Theorem statement
theorem train_speed_calculation :
  let total_distance := train_length + bridge_length
  let speed_m_per_s := total_distance / crossing_time
  let speed_km_per_hr := speed_m_per_s * m_per_s_to_km_per_hr
  speed_km_per_hr = 45 := by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l726_72664


namespace NUMINAMATH_CALUDE_vector_equality_implies_norm_equality_l726_72669

theorem vector_equality_implies_norm_equality 
  {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] 
  (a b : n) (ha : a ≠ 0) (hb : b ≠ 0) :
  a + 2 • b = 0 → ‖a - b‖ = ‖a‖ + ‖b‖ := by
sorry

end NUMINAMATH_CALUDE_vector_equality_implies_norm_equality_l726_72669


namespace NUMINAMATH_CALUDE_nonagon_area_theorem_l726_72699

/-- Represents a right triangle with regular nonagons on its sides -/
structure RightTriangleWithNonagons where
  /-- Length of the hypotenuse -/
  a : ℝ
  /-- Length of one cathetus -/
  b : ℝ
  /-- Length of the other cathetus -/
  c : ℝ
  /-- Area of the nonagon on the hypotenuse -/
  A₁ : ℝ
  /-- Area of the nonagon on one cathetus -/
  A₂ : ℝ
  /-- Area of the nonagon on the other cathetus -/
  A₃ : ℝ
  /-- The triangle is a right triangle -/
  right_triangle : a^2 = b^2 + c^2
  /-- The areas of nonagons are proportional to the squares of the sides -/
  proportional_areas : A₁ / a^2 = A₂ / b^2 ∧ A₁ / a^2 = A₃ / c^2

/-- The main theorem -/
theorem nonagon_area_theorem (t : RightTriangleWithNonagons) 
    (h₁ : t.A₁ = 2019) (h₂ : t.A₂ = 1602) : t.A₃ = 417 := by
  sorry


end NUMINAMATH_CALUDE_nonagon_area_theorem_l726_72699


namespace NUMINAMATH_CALUDE_spherical_triangle_smallest_angle_l726_72678

/-- 
Theorem: In a spherical triangle where the interior angles are in a 4:5:6 ratio 
and their sum is 270 degrees, the smallest angle measures 72 degrees.
-/
theorem spherical_triangle_smallest_angle 
  (a b c : ℝ) 
  (ratio : a = 4 * (b / 5) ∧ b = 5 * (c / 6)) 
  (sum_270 : a + b + c = 270) : 
  a = 72 := by
sorry

end NUMINAMATH_CALUDE_spherical_triangle_smallest_angle_l726_72678


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l726_72621

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := i / (1 + i)
  Complex.im z = (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l726_72621


namespace NUMINAMATH_CALUDE_correct_calculation_result_l726_72649

theorem correct_calculation_result (x : ℚ) : 
  (x * 6 = 96) → (x / 8 = 2) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_result_l726_72649


namespace NUMINAMATH_CALUDE_inscribed_hexagon_radius_theorem_l726_72680

/-- A hexagon inscribed in a circle with radius R, where three consecutive sides are equal to a
    and the other three consecutive sides are equal to b. -/
structure InscribedHexagon (R a b : ℝ) : Prop where
  radius_positive : R > 0
  side_a_positive : a > 0
  side_b_positive : b > 0
  three_sides_a : ∃ (AB BC CD : ℝ), AB = a ∧ BC = a ∧ CD = a
  three_sides_b : ∃ (DE EF FA : ℝ), DE = b ∧ EF = b ∧ FA = b

/-- The theorem stating the relationship between the radius R and sides a and b of the inscribed hexagon. -/
theorem inscribed_hexagon_radius_theorem (R a b : ℝ) (h : InscribedHexagon R a b) :
  R^2 = (a^2 + b^2 + a*b) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_radius_theorem_l726_72680


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l726_72644

open Set

def M : Set ℝ := {x | x^2 - 4*x < 0}
def N : Set ℝ := {x | |x| ≤ 2}

theorem union_of_M_and_N : M ∪ N = Icc (-2) 4 := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l726_72644


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l726_72685

theorem sqrt_equation_solution (a : ℝ) (h : a ≥ -1/4) :
  ∃ x : ℝ, x ≥ 0 ∧ Real.sqrt (a + Real.sqrt (a + x)) = x ∧ x = (1 + Real.sqrt (1 + 4*a)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l726_72685


namespace NUMINAMATH_CALUDE_sine_cosine_roots_product_l726_72686

theorem sine_cosine_roots_product (α β a b c d : Real) : 
  (∀ x, x^2 - a*x + b = 0 ↔ x = Real.sin α ∨ x = Real.sin β) →
  (∀ x, x^2 - c*x + d = 0 ↔ x = Real.cos α ∨ x = Real.cos β) →
  c * d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_roots_product_l726_72686


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l726_72693

/-- The lateral surface area of a cone with base radius 2 cm and slant height 5 cm is 10π cm². -/
theorem cone_lateral_surface_area : 
  let r : ℝ := 2  -- radius in cm
  let l : ℝ := 5  -- slant height in cm
  let lateral_area := (1/2) * l * (2 * Real.pi * r)
  lateral_area = 10 * Real.pi
  := by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l726_72693


namespace NUMINAMATH_CALUDE_estimate_fish_population_l726_72688

/-- Estimates the number of fish in a pond using the catch-mark-recapture method. -/
theorem estimate_fish_population (initial_catch : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  initial_catch > 0 →
  second_catch > 0 →
  marked_in_second > 0 →
  marked_in_second ≤ second_catch →
  marked_in_second ≤ initial_catch →
  (initial_catch * second_catch) / marked_in_second = 1200 →
  ∃ (estimated_population : ℕ), estimated_population = 1200 :=
by sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l726_72688


namespace NUMINAMATH_CALUDE_greatest_integer_a_l726_72653

theorem greatest_integer_a : ∀ a : ℤ,
  (∃ x : ℤ, (x - a) * (x - 7) + 3 = 0) →
  a ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_a_l726_72653


namespace NUMINAMATH_CALUDE_no_solution_for_specific_p_range_l726_72607

theorem no_solution_for_specific_p_range (p : ℝ) (h : 4/3 < p ∧ p < 2) :
  ¬∃ x : ℝ, Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_specific_p_range_l726_72607


namespace NUMINAMATH_CALUDE_cubic_root_problem_l726_72643

/-- A monic cubic polynomial -/
def MonicCubic (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

theorem cubic_root_problem (r : ℝ) (f g : ℝ → ℝ) 
    (hf : ∃ a b c, f = MonicCubic a b c)
    (hg : ∃ a b c, g = MonicCubic a b c)
    (hf_roots : f (r + 2) = 0 ∧ f (r + 4) = 0)
    (hg_roots : g (r + 3) = 0 ∧ g (r + 5) = 0)
    (h_diff : ∀ x, f x - g x = 2*r + 1) :
  r = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_problem_l726_72643


namespace NUMINAMATH_CALUDE_largest_number_l726_72663

theorem largest_number : ∀ (a b c : ℝ), 
  a = -12.4 → b = -1.23 → c = -0.13 → 
  (0 ≥ a) ∧ (0 ≥ b) ∧ (0 ≥ c) ∧ (0 ≥ 0) :=
by
  sorry

#check largest_number

end NUMINAMATH_CALUDE_largest_number_l726_72663


namespace NUMINAMATH_CALUDE_sqrt_12_minus_n_integer_l726_72636

theorem sqrt_12_minus_n_integer (n : ℕ) : 
  (∃ k : ℕ, k^2 = 12 - n) → n ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_12_minus_n_integer_l726_72636


namespace NUMINAMATH_CALUDE_turkey_roasting_problem_l726_72601

/-- Represents the turkey roasting problem --/
structure TurkeyRoasting where
  turkeyWeight : ℕ
  roastingTimePerPound : ℕ
  roastingStartTime : ℕ
  dinnerTime : ℕ

/-- Calculates the maximum number of turkeys that can be roasted --/
def maxTurkeys (tr : TurkeyRoasting) : ℕ :=
  let totalRoastingTime := tr.dinnerTime - tr.roastingStartTime
  let roastingTimePerTurkey := tr.turkeyWeight * tr.roastingTimePerPound
  totalRoastingTime / roastingTimePerTurkey

/-- Theorem statement for the turkey roasting problem --/
theorem turkey_roasting_problem :
  let tr : TurkeyRoasting := {
    turkeyWeight := 16,
    roastingTimePerPound := 15,
    roastingStartTime := 10 * 60,  -- 10:00 am in minutes
    dinnerTime := 18 * 60  -- 6:00 pm in minutes
  }
  maxTurkeys tr = 2 := by
  sorry


end NUMINAMATH_CALUDE_turkey_roasting_problem_l726_72601


namespace NUMINAMATH_CALUDE_max_b_value_l726_72661

/-- Given two functions f and g with a common point and equal tangents, 
    prove the maximum value of b -/
theorem max_b_value (a : ℝ) (h_a : a > 0) :
  let f := fun x : ℝ => (1/2) * x^2 + 2*a*x
  let g := fun x b : ℝ => 3*a^2 * Real.log x + b
  ∃ (x₀ b : ℝ), 
    (f x₀ = g x₀ b) ∧ 
    (deriv f x₀ = deriv (fun x => g x b) x₀) →
    (∀ b' : ℝ, ∃ x : ℝ, f x = g x b' → b' ≤ (3/2) * Real.exp ((2:ℝ)/3)) :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l726_72661


namespace NUMINAMATH_CALUDE_base6_addition_theorem_l726_72679

/-- Converts a base 6 number to base 10 --/
def base6_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 6 --/
def base10_to_base6 (n : ℕ) : ℕ := sorry

/-- Addition in base 6 --/
def add_base6 (a b : ℕ) : ℕ :=
  base10_to_base6 (base6_to_base10 a + base6_to_base10 b)

theorem base6_addition_theorem :
  add_base6 1254 3452 = 5150 := by sorry

end NUMINAMATH_CALUDE_base6_addition_theorem_l726_72679


namespace NUMINAMATH_CALUDE_division_of_mixed_number_by_fraction_l726_72642

theorem division_of_mixed_number_by_fraction :
  (3 : ℚ) / 2 / ((5 : ℚ) / 6) = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_division_of_mixed_number_by_fraction_l726_72642


namespace NUMINAMATH_CALUDE_equation_equivalence_l726_72615

theorem equation_equivalence (y : ℝ) (Q : ℝ) (h : 4 * (5 * y + 3 * Real.pi) = Q) :
  8 * (10 * y + 6 * Real.pi + 2 * Real.sqrt 3) = 4 * Q + 16 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l726_72615


namespace NUMINAMATH_CALUDE_triangle_identity_l726_72660

/-- For any triangle with sides a, b, c, circumradius R, and altitude CH from vertex C to side AB,
    the identity (a² + b² - c²) / (ab) = CH / R holds. -/
theorem triangle_identity (a b c R CH : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hR : R > 0) (hCH : CH > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + b^2 - c^2) / (a * b) = CH / R := by
  sorry

end NUMINAMATH_CALUDE_triangle_identity_l726_72660


namespace NUMINAMATH_CALUDE_sara_movie_purchase_cost_l726_72622

/-- The amount Sara spent on buying a movie, given her other movie-related expenses --/
theorem sara_movie_purchase_cost (ticket_price : ℝ) (ticket_count : ℕ) 
  (rental_cost : ℝ) (total_spent : ℝ) (h1 : ticket_price = 10.62) 
  (h2 : ticket_count = 2) (h3 : rental_cost = 1.59) (h4 : total_spent = 36.78) : 
  total_spent - (ticket_price * ↑ticket_count + rental_cost) = 13.95 := by
  sorry

end NUMINAMATH_CALUDE_sara_movie_purchase_cost_l726_72622


namespace NUMINAMATH_CALUDE_inverse_abs_is_geometric_sequence_preserving_l726_72627

/-- A function is geometric sequence-preserving if it transforms any non-constant
    geometric sequence into another geometric sequence. -/
def IsGeometricSequencePreserving (f : ℝ → ℝ) : Prop :=
  ∀ (a : ℕ → ℝ) (q : ℝ),
    (∀ n, a n ≠ 0) →
    (∀ n, a (n + 1) = q * a n) →
    q ≠ 1 →
    ∃ r : ℝ, r ≠ 1 ∧ ∀ n, f (a (n + 1)) = r * f (a n)

/-- The function f(x) = 1/|x| is geometric sequence-preserving. -/
theorem inverse_abs_is_geometric_sequence_preserving :
    IsGeometricSequencePreserving (fun x ↦ 1 / |x|) := by
  sorry


end NUMINAMATH_CALUDE_inverse_abs_is_geometric_sequence_preserving_l726_72627


namespace NUMINAMATH_CALUDE_length_of_AB_l726_72689

-- Define the circle Γ
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 3}

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - m * p.2 - 1 = 0}
def l₂ (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | m * p.1 + p.2 - m = 0}

-- Define the intersection points
def A (m : ℝ) : ℝ × ℝ := sorry
def B (m : ℝ) : ℝ × ℝ := sorry
def C (m : ℝ) : ℝ × ℝ := sorry
def D (m : ℝ) : ℝ × ℝ := sorry

-- State the theorem
theorem length_of_AB (m : ℝ) : 
  A m ∈ Γ ∧ A m ∈ l₁ m ∧
  B m ∈ Γ ∧ B m ∈ l₂ m ∧
  C m ∈ Γ ∧ C m ∈ l₁ m ∧
  D m ∈ Γ ∧ D m ∈ l₂ m ∧
  (A m).2 > 0 ∧ (B m).2 > 0 ∧
  (C m).2 < 0 ∧ (D m).2 < 0 ∧
  (D m).2 - (C m).2 = (D m).1 - (C m).1 →
  Real.sqrt ((A m).1 - (B m).1)^2 + ((A m).2 - (B m).2)^2 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AB_l726_72689


namespace NUMINAMATH_CALUDE_second_company_can_hire_three_geniuses_l726_72668

/-- Represents a programmer --/
structure Programmer where
  id : Nat

/-- Represents a genius programmer --/
structure Genius extends Programmer

/-- Represents the hiring game between two companies --/
structure HiringGame where
  programmers : List Programmer
  geniuses : List Genius
  acquaintances : List (Programmer × Programmer)

/-- Represents a company's hiring strategy --/
structure HiringStrategy where
  nextHire : List Programmer → List Programmer → Option Programmer

/-- The result of the hiring game --/
inductive GameResult
  | FirstCompanyWins
  | SecondCompanyWins

/-- Simulates the hiring game given two strategies --/
def playGame (game : HiringGame) (strategy1 strategy2 : HiringStrategy) : GameResult :=
  sorry

/-- Theorem stating that there exists a winning strategy for the second company --/
theorem second_company_can_hire_three_geniuses :
  ∃ (game : HiringGame) (strategy : HiringStrategy),
    (game.geniuses.length = 4) →
    ∀ (opponent_strategy : HiringStrategy),
      playGame game opponent_strategy strategy = GameResult.SecondCompanyWins :=
sorry

end NUMINAMATH_CALUDE_second_company_can_hire_three_geniuses_l726_72668


namespace NUMINAMATH_CALUDE_boyds_boy_friends_percentage_l726_72618

theorem boyds_boy_friends_percentage 
  (julian_total_friends : ℕ)
  (julian_boys_percentage : ℚ)
  (julian_girls_percentage : ℚ)
  (boyd_total_friends : ℕ)
  (h1 : julian_total_friends = 80)
  (h2 : julian_boys_percentage = 60 / 100)
  (h3 : julian_girls_percentage = 40 / 100)
  (h4 : julian_boys_percentage + julian_girls_percentage = 1)
  (h5 : boyd_total_friends = 100)
  (h6 : (julian_girls_percentage * julian_total_friends : ℚ) * 2 = boyd_total_friends - (boyd_total_friends - (julian_girls_percentage * julian_total_friends : ℚ) * 2)) :
  (boyd_total_friends - (julian_girls_percentage * julian_total_friends : ℚ) * 2) / boyd_total_friends = 36 / 100 := by
  sorry

end NUMINAMATH_CALUDE_boyds_boy_friends_percentage_l726_72618


namespace NUMINAMATH_CALUDE_sams_balloons_l726_72613

theorem sams_balloons (fred_balloons : ℕ) (mary_balloons : ℕ) (total_balloons : ℕ) 
  (h1 : fred_balloons = 5)
  (h2 : mary_balloons = 7)
  (h3 : total_balloons = 18) :
  total_balloons - fred_balloons - mary_balloons = 6 :=
by sorry

end NUMINAMATH_CALUDE_sams_balloons_l726_72613


namespace NUMINAMATH_CALUDE_savings_percentage_l726_72659

def monthly_salary : ℝ := 1000
def savings_after_increase : ℝ := 175
def expense_increase_rate : ℝ := 0.10

theorem savings_percentage :
  ∃ (savings_rate : ℝ),
    savings_rate * monthly_salary = monthly_salary - (monthly_salary - savings_rate * monthly_salary) * (1 + expense_increase_rate) ∧
    savings_rate * monthly_salary = savings_after_increase ∧
    savings_rate = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_savings_percentage_l726_72659


namespace NUMINAMATH_CALUDE_factorization_equality_l726_72673

theorem factorization_equality (x : ℝ) :
  (x^2 - x - 6) * (x^2 + 3*x - 4) + 24 =
  (x + 3) * (x - 2) * (x + (1 + Real.sqrt 33) / 2) * (x + (1 - Real.sqrt 33) / 2) := by
sorry

end NUMINAMATH_CALUDE_factorization_equality_l726_72673


namespace NUMINAMATH_CALUDE_quadratic_equation_root_difference_l726_72658

theorem quadratic_equation_root_difference (k : ℝ) : 
  (∃ a b : ℝ, 3 * a^2 + 2 * a + k = 0 ∧ 
              3 * b^2 + 2 * b + k = 0 ∧ 
              |a - b| = (a^2 + b^2).sqrt) ↔ 
  (k = 0 ∨ k = -4/15) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_difference_l726_72658


namespace NUMINAMATH_CALUDE_pages_left_after_three_weeks_l726_72697

-- Define the structure for a book
structure Book where
  totalPages : ℕ
  pagesRead : ℕ
  pagesPerDay : ℕ

-- Define Elliot's books
def book1 : Book := ⟨512, 194, 30⟩
def book2 : Book := ⟨298, 0, 20⟩
def book3 : Book := ⟨365, 50, 25⟩
def book4 : Book := ⟨421, 0, 15⟩

-- Define the number of days
def days : ℕ := 21

-- Function to calculate pages left after reading
def pagesLeftAfterReading (b : Book) (days : ℕ) : ℕ :=
  max 0 (b.totalPages - b.pagesRead - b.pagesPerDay * days)

-- Theorem statement
theorem pages_left_after_three_weeks :
  pagesLeftAfterReading book1 days + 
  pagesLeftAfterReading book2 days + 
  pagesLeftAfterReading book3 days + 
  pagesLeftAfterReading book4 days = 106 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_after_three_weeks_l726_72697


namespace NUMINAMATH_CALUDE_awards_sum_is_80_l726_72683

/-- The number of awards won by Scott -/
def scott_awards : ℕ := 4

/-- The number of awards won by Jessie -/
def jessie_awards : ℕ := 3 * scott_awards

/-- The number of awards won by the rival athlete -/
def rival_awards : ℕ := 2 * jessie_awards

/-- The number of awards won by Brad -/
def brad_awards : ℕ := (5 * rival_awards) / 3

/-- The total number of awards won by all four athletes -/
def total_awards : ℕ := scott_awards + jessie_awards + rival_awards + brad_awards

theorem awards_sum_is_80 : total_awards = 80 := by
  sorry

end NUMINAMATH_CALUDE_awards_sum_is_80_l726_72683


namespace NUMINAMATH_CALUDE_euler_number_proof_l726_72634

def gauss_number : ℂ := Complex.mk 6 4

theorem euler_number_proof (product : ℂ) (h1 : product = Complex.mk 48 (-18)) :
  ∃ (euler_number : ℂ), euler_number * gauss_number = product ∧ euler_number = Complex.mk 4 (-6) := by
  sorry

end NUMINAMATH_CALUDE_euler_number_proof_l726_72634


namespace NUMINAMATH_CALUDE_logarithm_equality_l726_72690

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_equality :
  lg (25 / 16) - 2 * lg (5 / 9) + lg (32 / 81) = lg 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equality_l726_72690


namespace NUMINAMATH_CALUDE_angle_around_point_l726_72630

theorem angle_around_point (y : ℝ) : 
  210 + 3 * y = 360 → y = 50 := by sorry

end NUMINAMATH_CALUDE_angle_around_point_l726_72630


namespace NUMINAMATH_CALUDE_boat_reachable_area_l726_72620

/-- Represents the speed of the boat in miles per hour -/
structure BoatSpeed where
  river : ℝ
  land : ℝ

/-- Calculates the area reachable by the boat given its speed and time limit -/
def reachable_area (speed : BoatSpeed) (time_limit : ℝ) : ℝ :=
  sorry

theorem boat_reachable_area :
  let speed : BoatSpeed := { river := 40, land := 10 }
  let time_limit : ℝ := 12 / 60 -- 12 minutes in hours
  reachable_area speed time_limit = 232 * π / 6 := by
  sorry

#eval (232 + 6 : Nat)

end NUMINAMATH_CALUDE_boat_reachable_area_l726_72620


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l726_72674

theorem parallelogram_side_length
  (s : ℝ)
  (side1 : ℝ)
  (side2 : ℝ)
  (angle : ℝ)
  (area : ℝ)
  (h1 : side1 = s)
  (h2 : side2 = 3 * s)
  (h3 : angle = π / 3)  -- 60 degrees in radians
  (h4 : area = 27 * Real.sqrt 3)
  (h5 : area = side1 * side2 * Real.sin angle) :
  s = 3 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l726_72674


namespace NUMINAMATH_CALUDE_fraction_simplification_and_rationalization_l726_72691

theorem fraction_simplification_and_rationalization :
  (3 : ℝ) / (Real.sqrt 75 + Real.sqrt 48 + Real.sqrt 12) = Real.sqrt 3 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_and_rationalization_l726_72691


namespace NUMINAMATH_CALUDE_number_puzzle_l726_72698

theorem number_puzzle (x y : ℝ) : x = 33 → (x / 4) + y = 15 → y = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l726_72698


namespace NUMINAMATH_CALUDE_magic_box_solution_l726_72623

-- Define the magic box function
def magicBox (a b : ℝ) : ℝ := a^2 + b - 1

-- State the theorem
theorem magic_box_solution :
  ∀ m : ℝ, magicBox m (-2*m) = 2 → m = 3 ∨ m = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_magic_box_solution_l726_72623


namespace NUMINAMATH_CALUDE_max_isosceles_triangles_2017gon_l726_72638

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- Represents a triangulation of a regular polygon -/
structure Triangulation (n : ℕ) where
  polygon : RegularPolygon n
  num_diagonals : ℕ
  num_triangles : ℕ
  diagonals_non_intersecting : Bool
  triangle_count_valid : num_triangles = n - 2

/-- Counts the maximum number of isosceles triangles in a given triangulation -/
def max_isosceles_triangles (t : Triangulation 2017) : ℕ :=
  sorry

/-- The main theorem stating the maximum number of isosceles triangles -/
theorem max_isosceles_triangles_2017gon :
  ∀ (t : Triangulation 2017),
    t.num_diagonals = 2014 ∧ 
    t.diagonals_non_intersecting = true →
    max_isosceles_triangles t = 2010 :=
  sorry

end NUMINAMATH_CALUDE_max_isosceles_triangles_2017gon_l726_72638


namespace NUMINAMATH_CALUDE_homework_problem_l726_72656

theorem homework_problem (p t : ℕ) : 
  p > 0 → 
  t > 0 → 
  p ≥ 15 → 
  3 * p - 5 ≥ 20 → 
  p * t = (3 * p - 5) * (t - 3) → 
  p * t = 100 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_l726_72656
