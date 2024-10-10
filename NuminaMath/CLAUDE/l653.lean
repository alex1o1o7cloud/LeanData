import Mathlib

namespace min_value_theorem_min_value_is_two_min_value_achieved_l653_65364

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x + y) / (x*y) = 7/2 + Real.sqrt 6) : 
  ∀ a b : ℝ, a > 0 → b > 0 → (2*a + b) / (a*b) = 7/2 + Real.sqrt 6 → x + 3*y ≤ a + 3*b :=
sorry

theorem min_value_is_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x + y) / (x*y) = 7/2 + Real.sqrt 6) : 
  x + 3*y ≥ 2 :=
sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x + y) / (x*y) = 7/2 + Real.sqrt 6) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (2*a + b) / (a*b) = 7/2 + Real.sqrt 6 ∧ a + 3*b = 2 :=
sorry

end min_value_theorem_min_value_is_two_min_value_achieved_l653_65364


namespace infinitely_many_n_congruent_to_sum_of_digits_l653_65322

/-- Sum of digits in base r -/
def S_r (r : ℕ) (n : ℕ) : ℕ := sorry

/-- There are infinitely many n such that S_r(n) ≡ n (mod p) -/
theorem infinitely_many_n_congruent_to_sum_of_digits 
  (r : ℕ) (p : ℕ) (hr : r > 1) (hp : Nat.Prime p) :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ k, S_r r (f k) ≡ f k [MOD p] := by sorry

end infinitely_many_n_congruent_to_sum_of_digits_l653_65322


namespace functional_equation_equivalence_l653_65341

theorem functional_equation_equivalence (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f x + f y) ↔
  (∀ x y : ℝ, f (x + y + x * y) = f x + f y + f (x * y)) :=
by sorry

end functional_equation_equivalence_l653_65341


namespace identical_permutations_of_increasing_sum_l653_65326

/-- A strictly increasing finite sequence of real numbers -/
def StrictlyIncreasingSeq (a : Fin n → ℝ) : Prop :=
  ∀ i j : Fin n, i < j → a i < a j

/-- A permutation of indices -/
def IsPermutation (σ : Fin n → Fin n) : Prop :=
  Function.Bijective σ

theorem identical_permutations_of_increasing_sum
  (a : Fin n → ℝ) (σ : Fin n → Fin n)
  (h_inc : StrictlyIncreasingSeq a)
  (h_perm : IsPermutation σ)
  (h_sum_inc : StrictlyIncreasingSeq (fun i => a i + a (σ i))) :
  ∀ i, a i = a (σ i) := by
sorry

end identical_permutations_of_increasing_sum_l653_65326


namespace two_distinct_real_roots_l653_65357

def polynomial (a x : ℝ) : ℝ := x^4 + 3*a*x^3 + a*(1-5*a^2)*x - 3*a^4 + a^2 + 1

theorem two_distinct_real_roots (a : ℝ) :
  (∃ x : ℝ, polynomial a x = 0) ∧ 
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ polynomial a x₁ = 0 ∧ polynomial a x₂ = 0) →
  a = 2 * Real.sqrt 26 / 13 ∨ a = -2 * Real.sqrt 26 / 13 := by
  sorry

end two_distinct_real_roots_l653_65357


namespace translation_theorem_l653_65308

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line horizontally and vertically -/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept - l.slope * dx + dy }

/-- The original line y = 2x - 3 -/
def original_line : Line :=
  { slope := 2,
    intercept := -3 }

/-- The amount of horizontal translation -/
def right_shift : ℝ := 2

/-- The amount of vertical translation -/
def up_shift : ℝ := 1

/-- The expected resulting line after translation -/
def expected_result : Line :=
  { slope := 2,
    intercept := -6 }

theorem translation_theorem :
  translate original_line right_shift up_shift = expected_result := by
  sorry

end translation_theorem_l653_65308


namespace age_problem_l653_65315

/-- Given the ages of Matt, Kaylee, Bella, and Alex, prove that they satisfy the given conditions -/
theorem age_problem (matt_age kaylee_age bella_age alex_age : ℕ) : 
  matt_age = 5 ∧ 
  kaylee_age + 7 = 3 * matt_age ∧ 
  kaylee_age + bella_age = matt_age + 9 ∧ 
  bella_age = alex_age + 3 →
  kaylee_age = 8 ∧ matt_age = 5 ∧ bella_age = 6 ∧ alex_age = 3 := by
sorry

end age_problem_l653_65315


namespace max_distance_M_to_N_l653_65379

-- Define the circles and point
def circle_M (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*a*x - 2*a*y + 2*a^2 - 2 = 0
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 18
def point_N : ℝ × ℝ := (1, 2)

-- Define the theorem
theorem max_distance_M_to_N :
  ∀ a : ℝ,
  (∀ x y : ℝ, ∃ x' y' : ℝ, circle_M a x' y' ∧ circle_O x' y') →
  ∃ a_max : ℝ,
    (∀ a' : ℝ, (∃ x y : ℝ, circle_M a' x y ∧ circle_O x y) →
      Real.sqrt ((a' - point_N.1)^2 + (a' - point_N.2)^2) ≤ Real.sqrt ((a_max - point_N.1)^2 + (a_max - point_N.2)^2)) ∧
    Real.sqrt ((a_max - point_N.1)^2 + (a_max - point_N.2)^2) = Real.sqrt 13 :=
by sorry

end max_distance_M_to_N_l653_65379


namespace quadratic_intersection_theorem_l653_65398

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + m

-- Define the condition for three intersection points
def has_three_intersections (m : ℝ) : Prop :=
  m ≠ 0 ∧ ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  ((f m x₁ = 0 ∧ x₁ ≠ 0) ∨ (x₁ = 0 ∧ f m 0 = m)) ∧
  ((f m x₂ = 0 ∧ x₂ ≠ 0) ∨ (x₂ = 0 ∧ f m 0 = m)) ∧
  ((f m x₃ = 0 ∧ x₃ ≠ 0) ∨ (x₃ = 0 ∧ f m 0 = m))

-- Define the circle passing through the three intersection points
def circle_through_intersections (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - (m + 1)*y + m = 0

-- The main theorem
theorem quadratic_intersection_theorem (m : ℝ) :
  has_three_intersections m →
  (m < 4 ∧
   circle_through_intersections m 0 1 ∧
   circle_through_intersections m (-4) 1) :=
by sorry

end quadratic_intersection_theorem_l653_65398


namespace dog_weight_ratio_l653_65354

theorem dog_weight_ratio (chihuahua pitbull great_dane : ℝ) : 
  chihuahua + pitbull + great_dane = 439 →
  great_dane = 307 →
  great_dane = 3 * pitbull + 10 →
  pitbull / chihuahua = 3 := by
  sorry

end dog_weight_ratio_l653_65354


namespace remainder_sum_l653_65347

theorem remainder_sum (n : ℤ) (h : n % 12 = 5) : (n % 4 + n % 3) = 3 := by
  sorry

end remainder_sum_l653_65347


namespace min_value_theorem_l653_65337

/-- The minimum value of (x₁² + x₂²) / (x₁ - x₂) given the conditions -/
theorem min_value_theorem (a c m n x₁ x₂ : ℝ) : 
  (2 * a * m + (a + c) * n + 2 * c = 0) →  -- line passes through (m, n)
  (x₁ + x₂ + m + n = 15) →                 -- sum condition
  (x₁ > x₂) →                              -- ordering condition
  (∀ y₁ y₂ : ℝ, (y₁ + y₂ + m + n = 15) → (y₁ > y₂) → 
    (x₁^2 + x₂^2) / (x₁ - x₂) ≤ (y₁^2 + y₂^2) / (y₁ - y₂)) →
  (x₁^2 + x₂^2) / (x₁ - x₂) = 16 :=
by sorry

end min_value_theorem_l653_65337


namespace prob_win_series_4_1_l653_65340

/-- Represents the location of a game -/
inductive GameLocation
  | Home
  | Away

/-- Represents the schedule of games for Team A -/
def schedule : List GameLocation :=
  [GameLocation.Home, GameLocation.Home, GameLocation.Away, GameLocation.Away, 
   GameLocation.Home, GameLocation.Away, GameLocation.Home]

/-- Probability of Team A winning a home game -/
def probWinHome : ℝ := 0.6

/-- Probability of Team A winning an away game -/
def probWinAway : ℝ := 0.5

/-- Calculates the probability of Team A winning a game based on its location -/
def probWin (loc : GameLocation) : ℝ :=
  match loc with
  | GameLocation.Home => probWinHome
  | GameLocation.Away => probWinAway

/-- Calculates the probability of a specific game outcome for Team A -/
def probOutcome (outcomes : List Bool) : ℝ :=
  List.zipWith (fun o l => if o then probWin l else 1 - probWin l) outcomes schedule
  |> List.prod

/-- Theorem: The probability of Team A winning the series with a 4:1 score is 0.18 -/
theorem prob_win_series_4_1 : 
  (probOutcome [false, true, true, true, true] +
   probOutcome [true, false, true, true, true] +
   probOutcome [true, true, false, true, true] +
   probOutcome [true, true, true, false, true]) = 0.18 := by
  sorry


end prob_win_series_4_1_l653_65340


namespace three_people_selection_l653_65392

-- Define the number of people in the group
def n : ℕ := 30

-- Define the number of enemies each person has
def enemies_per_person : ℕ := 6

-- Define the function to calculate the number of ways to select 3 people
-- such that any two of them are either friends or enemies
def select_three_people (n : ℕ) (enemies_per_person : ℕ) : ℕ :=
  -- The actual calculation is not implemented, as per instructions
  sorry

-- The theorem to prove
theorem three_people_selection :
  select_three_people n enemies_per_person = 1990 := by
  sorry

end three_people_selection_l653_65392


namespace time_to_finish_game_l653_65339

/-- Calculates the time to finish a game given initial and increased play times --/
theorem time_to_finish_game 
  (initial_hours_per_day : ℝ)
  (initial_days : ℝ)
  (completion_percentage : ℝ)
  (increased_hours_per_day : ℝ) :
  initial_hours_per_day = 4 →
  initial_days = 14 →
  completion_percentage = 0.4 →
  increased_hours_per_day = 7 →
  (initial_days * initial_hours_per_day * (1 / completion_percentage) - 
   initial_days * initial_hours_per_day) / increased_hours_per_day = 12 := by
sorry

end time_to_finish_game_l653_65339


namespace cindy_crayons_count_l653_65303

/-- The number of crayons Karen has -/
def karen_crayons : ℕ := 639

/-- The difference between Karen's and Cindy's crayons -/
def difference : ℕ := 135

/-- The number of crayons Cindy has -/
def cindy_crayons : ℕ := karen_crayons - difference

theorem cindy_crayons_count : cindy_crayons = 504 := by
  sorry

end cindy_crayons_count_l653_65303


namespace students_in_two_courses_l653_65302

theorem students_in_two_courses 
  (total : ℕ) 
  (math : ℕ) 
  (chinese : ℕ) 
  (international : ℕ) 
  (all_three : ℕ) 
  (none : ℕ) 
  (h1 : total = 400) 
  (h2 : math = 169) 
  (h3 : chinese = 158) 
  (h4 : international = 145) 
  (h5 : all_three = 30) 
  (h6 : none = 20) : 
  ∃ (two_courses : ℕ), 
    two_courses = 32 ∧ 
    total = math + chinese + international - two_courses - 2 * all_three + none :=
by sorry

end students_in_two_courses_l653_65302


namespace valentines_count_l653_65393

theorem valentines_count (initial : ℕ) (given_away : ℕ) (received : ℕ) : 
  initial = 60 → given_away = 16 → received = 5 → 
  initial - given_away + received = 49 := by
  sorry

end valentines_count_l653_65393


namespace equation_system_solution_l653_65327

def solution_set (a b c x y z : ℝ) : Prop :=
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = c) ∨
  (x = a ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = b ∧ z = 0)

theorem equation_system_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∀ x y z : ℝ,
    (a * x + b * y = (x - y)^2 ∧
     b * y + c * z = (y - z)^2 ∧
     c * z + a * x = (z - x)^2) ↔
    solution_set a b c x y z :=
by sorry

end equation_system_solution_l653_65327


namespace alpha_plus_two_beta_eq_pi_over_four_l653_65397

theorem alpha_plus_two_beta_eq_pi_over_four 
  (α β : Real) 
  (acute_α : 0 < α ∧ α < π / 2) 
  (acute_β : 0 < β ∧ β < π / 2) 
  (tan_α : Real.tan α = 1 / 7) 
  (sin_β : Real.sin β = Real.sqrt 10 / 10) : 
  α + 2 * β = π / 4 := by
sorry

end alpha_plus_two_beta_eq_pi_over_four_l653_65397


namespace cyclic_sum_inequality_l653_65377

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a - b) / (b + c) + (b - c) / (c + d) + (c - d) / (d + a) + (d - a) / (a + b) ≥ 0 ∧
  ((a - b) / (b + c) + (b - c) / (c + d) + (c - d) / (d + a) + (d - a) / (a + b) = 0 ↔ a = c ∧ b = d) :=
by sorry

end cyclic_sum_inequality_l653_65377


namespace prob_ratio_l653_65316

/-- Represents the total number of cards in the box -/
def total_cards : ℕ := 50

/-- Represents the number of distinct card numbers -/
def distinct_numbers : ℕ := 10

/-- Represents the number of cards for each number -/
def cards_per_number : ℕ := 5

/-- Represents the number of cards drawn -/
def cards_drawn : ℕ := 5

/-- Calculates the probability of drawing five cards with the same number -/
def prob_same_number : ℚ :=
  (distinct_numbers : ℚ) / (total_cards.choose cards_drawn : ℚ)

/-- Calculates the probability of drawing four cards of one number and one card of a different number -/
def prob_four_and_one : ℚ :=
  ((distinct_numbers : ℚ) * (distinct_numbers - 1 : ℚ) * (cards_per_number : ℚ) * (cards_per_number : ℚ)) /
  (total_cards.choose cards_drawn : ℚ)

/-- Theorem stating the ratio of probabilities -/
theorem prob_ratio :
  prob_four_and_one / prob_same_number = 225 := by sorry

end prob_ratio_l653_65316


namespace area_ratio_EFWZ_ZWGH_l653_65353

-- Define the points
variable (E F G H O Q Z W : ℝ × ℝ)

-- Define the lengths
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the conditions
axiom EF_eq_EO : length E F = length E O
axiom EO_eq_OG : length E O = length O G
axiom OG_eq_GH : length O G = length G H
axiom EF_eq_12 : length E F = 12
axiom FG_eq_18 : length F G = 18
axiom EH_eq_18 : length E H = 18
axiom OH_eq_18 : length O H = 18

-- Define Q as the point on FG such that OQ is perpendicular to FG
axiom Q_on_FG : sorry
axiom OQ_perp_FG : sorry

-- Define Z as midpoint of EF
axiom Z_midpoint_EF : sorry

-- Define W as midpoint of GH
axiom W_midpoint_GH : sorry

-- Define the area function for trapezoids
def area_trapezoid (A B C D : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio_EFWZ_ZWGH : 
  area_trapezoid E F W Z = area_trapezoid Z W G H := by sorry

end area_ratio_EFWZ_ZWGH_l653_65353


namespace point_B_in_fourth_quadrant_l653_65388

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Determines if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Given point A in the second quadrant, prove that point B is in the fourth quadrant -/
theorem point_B_in_fourth_quadrant (m n : ℝ) (h : isInSecondQuadrant ⟨m, n⟩) :
  isInFourthQuadrant ⟨2*n - m, -n + m⟩ := by
  sorry

end point_B_in_fourth_quadrant_l653_65388


namespace isosceles_triangle_angle_relation_l653_65345

theorem isosceles_triangle_angle_relation (A B C C₁ C₂ θ : Real) :
  -- Isosceles triangle condition
  A = B →
  -- Altitude divides angle C into C₁ and C₂
  A + C₁ = 90 →
  B + C₂ = 90 →
  -- External angle θ
  θ = 30 →
  θ = A + B →
  -- Conclusion
  C₁ = 75 ∧ C₂ = 75 := by
  sorry

end isosceles_triangle_angle_relation_l653_65345


namespace number_puzzle_l653_65335

theorem number_puzzle : ∃ x : ℤ, x + 3*12 + 3*13 + 3*16 = 134 ∧ x = 11 := by
  sorry

end number_puzzle_l653_65335


namespace third_face_area_is_60_l653_65328

/-- Represents a cuboidal box with given dimensions -/
structure CuboidalBox where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The area of the first adjacent face -/
def first_face_area (box : CuboidalBox) : ℝ := box.length * box.width

/-- The area of the second adjacent face -/
def second_face_area (box : CuboidalBox) : ℝ := box.width * box.height

/-- The area of the third adjacent face -/
def third_face_area (box : CuboidalBox) : ℝ := box.length * box.height

/-- The volume of the box -/
def volume (box : CuboidalBox) : ℝ := box.length * box.width * box.height

/-- Theorem stating the area of the third face given the conditions -/
theorem third_face_area_is_60 (box : CuboidalBox) 
  (h1 : first_face_area box = 120)
  (h2 : second_face_area box = 72)
  (h3 : volume box = 720) :
  third_face_area box = 60 := by
  sorry


end third_face_area_is_60_l653_65328


namespace total_triangles_in_pattern_l653_65306

/-- Represents the hexagonal pattern with triangular subdivisions -/
structure HexagonalPattern :=
  (small_triangles : ℕ)
  (medium_triangles : ℕ)
  (large_triangles : ℕ)
  (extra_large_triangles : ℕ)

/-- The total number of triangles in the hexagonal pattern -/
def total_triangles (pattern : HexagonalPattern) : ℕ :=
  pattern.small_triangles + pattern.medium_triangles + pattern.large_triangles + pattern.extra_large_triangles

/-- The specific hexagonal pattern described in the problem -/
def given_pattern : HexagonalPattern :=
  { small_triangles := 10,
    medium_triangles := 6,
    large_triangles := 3,
    extra_large_triangles := 1 }

theorem total_triangles_in_pattern :
  total_triangles given_pattern = 20 := by
  sorry

end total_triangles_in_pattern_l653_65306


namespace equal_potato_distribution_l653_65378

theorem equal_potato_distribution (total_potatoes : ℕ) (family_members : ℕ) 
  (h1 : total_potatoes = 60) (h2 : family_members = 6) :
  total_potatoes / family_members = 10 := by
  sorry

end equal_potato_distribution_l653_65378


namespace total_sum_is_120_rupees_l653_65334

/-- Represents the division of money among three people -/
structure MoneyDivision where
  a_share : ℕ  -- A's share in paisa per rupee
  b_share : ℕ  -- B's share in paisa per rupee
  c_share : ℕ  -- C's share in paisa per rupee

/-- The given problem setup -/
def problem_setup : MoneyDivision :=
  { a_share := 0,  -- We don't know A's exact share, so we leave it as 0
    b_share := 65,
    c_share := 40 }

/-- Theorem stating the total sum of money -/
theorem total_sum_is_120_rupees (md : MoneyDivision) 
  (h1 : md.b_share = 65)
  (h2 : md.c_share = 40)
  (h3 : md.a_share + md.b_share + md.c_share = 100)  -- Total per rupee is 100 paisa
  (h4 : md.c_share * 120 = 4800)  -- C's share is Rs. 48 (4800 paisa)
  : (4800 / md.c_share) * 100 = 12000 := by
  sorry

#check total_sum_is_120_rupees

end total_sum_is_120_rupees_l653_65334


namespace polynomial_simplification_l653_65384

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + 5 * x^3 + x + 20) - (x^6 + 4 * x^5 - 2 * x^4 + x^3 + 15) =
  x^6 - x^5 + 3 * x^4 + 4 * x^3 + x + 5 := by
sorry

end polynomial_simplification_l653_65384


namespace marbles_in_container_l653_65338

/-- Given that a container with volume 24 cm³ holds 75 marbles, 
    prove that a container with volume 72 cm³ holds 225 marbles, 
    assuming the number of marbles is proportional to the volume. -/
theorem marbles_in_container (v₁ v₂ : ℝ) (m₁ m₂ : ℕ) 
  (h₁ : v₁ = 24) (h₂ : v₂ = 72) (h₃ : m₁ = 75) 
  (h₄ : v₁ * m₂ = v₂ * m₁) : m₂ = 225 := by
  sorry

end marbles_in_container_l653_65338


namespace only_solution_is_three_l653_65367

def sum_of_digits (n : ℕ) : ℕ :=
  sorry

theorem only_solution_is_three :
  ∃! n : ℕ, sum_of_digits (5^n) = 2^n ∧ n = 3 :=
sorry

end only_solution_is_three_l653_65367


namespace jemma_grasshopper_count_l653_65363

/-- The number of grasshoppers Jemma found on the African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The number of dozens of baby grasshoppers Jemma found on the grass -/
def dozens_of_baby_grasshoppers : ℕ := 2

/-- The number of grasshoppers in a dozen -/
def grasshoppers_per_dozen : ℕ := 12

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := grasshoppers_on_plant + dozens_of_baby_grasshoppers * grasshoppers_per_dozen

theorem jemma_grasshopper_count : total_grasshoppers = 31 := by
  sorry

end jemma_grasshopper_count_l653_65363


namespace field_trip_students_l653_65349

theorem field_trip_students (van_capacity : ℕ) (num_vans : ℕ) (num_adults : ℕ) 
  (h1 : van_capacity = 9)
  (h2 : num_vans = 6)
  (h3 : num_adults = 14) :
  num_vans * van_capacity - num_adults = 40 := by
  sorry

end field_trip_students_l653_65349


namespace fenced_area_calculation_l653_65332

theorem fenced_area_calculation (length width cutout_side : ℕ) : 
  length = 20 → width = 18 → cutout_side = 4 →
  (length * width) - (cutout_side * cutout_side) = 344 := by
sorry

end fenced_area_calculation_l653_65332


namespace arithmetic_square_root_of_sqrt_16_l653_65360

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end arithmetic_square_root_of_sqrt_16_l653_65360


namespace remainder_of_repeated_sequence_l653_65399

/-- The sequence of digits that is repeated to form the number -/
def digit_sequence : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- The number of digits in the large number -/
def total_digits : Nat := 2012

/-- The theorem stating that the remainder when the 2012-digit number
    formed by repeating the sequence 1, 2, 3, 4, 5, 6, 7, 8, 9
    is divided by 9 is equal to 6 -/
theorem remainder_of_repeated_sequence :
  (List.sum (List.take (total_digits % digit_sequence.length) digit_sequence)) % 9 = 6 := by
  sorry

end remainder_of_repeated_sequence_l653_65399


namespace sufficient_but_not_necessary_l653_65382

-- Define the proposition
def proposition (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a*x - 4*a ≤ 0

-- Define the condition
def condition (a : ℝ) : Prop :=
  -16 ≤ a ∧ a ≤ 0

-- Theorem statement
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, condition a → ¬proposition a) ∧
  ¬(∀ a : ℝ, ¬proposition a → condition a) :=
sorry

end sufficient_but_not_necessary_l653_65382


namespace triangle_max_area_l653_65309

theorem triangle_max_area (a b : ℝ) (h1 : a + b = 4) (h2 : 0 < a) (h3 : 0 < b) : 
  (1/2 : ℝ) * a * b * Real.sin (π/6) ≤ 1 :=
sorry

end triangle_max_area_l653_65309


namespace inverse_variation_problem_l653_65375

theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (3 * x^2 * y = 3 * 3^2 * 15) → (y = 6750) → (x = Real.sqrt 2 / 10) := by
  sorry

end inverse_variation_problem_l653_65375


namespace functional_equation_solutions_l653_65369

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x - f y) = (x - y)^2 * f (x + y)

/-- The theorem stating the possible forms of functions satisfying the equation. -/
theorem functional_equation_solutions :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
    (∀ x, f x = 0) ∨ (∀ x, f x = x^2) ∨ (∀ x, f x = -x^2) := by
  sorry

end functional_equation_solutions_l653_65369


namespace shortest_side_of_right_triangle_l653_65314

theorem shortest_side_of_right_triangle (a b c : ℝ) : 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → a ≤ b ∧ a ≤ c := by
  sorry

end shortest_side_of_right_triangle_l653_65314


namespace hyperbola_sum_theorem_l653_65386

def F₁ : ℝ × ℝ := (2, -1)
def F₂ : ℝ × ℝ := (2, 3)

def is_on_hyperbola (P : ℝ × ℝ) : Prop :=
  abs (dist P F₁ - dist P F₂) = 2

def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

theorem hyperbola_sum_theorem (h k a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x y, hyperbola_equation x y h k a b ↔ is_on_hyperbola (x, y)) :
  h + k + a + b = 4 + Real.sqrt 3 := by
  sorry

end hyperbola_sum_theorem_l653_65386


namespace total_discount_calculation_l653_65372

theorem total_discount_calculation (tshirt_price jeans_price : ℝ)
  (tshirt_discount jeans_discount : ℝ) :
  tshirt_price = 25 →
  jeans_price = 75 →
  tshirt_discount = 0.3 →
  jeans_discount = 0.1 →
  tshirt_price * tshirt_discount + jeans_price * jeans_discount = 15 := by
  sorry

end total_discount_calculation_l653_65372


namespace min_sum_of_distances_l653_65329

/-- The curve on which point P moves -/
def curve (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The first line l₁ -/
def line1 (y : ℝ) : Prop := y = 2

/-- The second line l₂ -/
def line2 (x : ℝ) : Prop := x = -1

/-- The distance from a point (x, y) to line1 -/
def dist_to_line1 (y : ℝ) : ℝ := |y - 2|

/-- The distance from a point (x, y) to line2 -/
def dist_to_line2 (x : ℝ) : ℝ := |x + 1|

/-- The sum of distances from a point (x, y) to both lines -/
def sum_of_distances (x y : ℝ) : ℝ := dist_to_line1 y + dist_to_line2 x

/-- The theorem stating the minimum value of the sum of distances -/
theorem min_sum_of_distances :
  ∃ (min : ℝ), min = 4 - Real.sqrt 2 ∧
  ∀ (x y : ℝ), curve x y → sum_of_distances x y ≥ min :=
sorry

end min_sum_of_distances_l653_65329


namespace inequality_proof_l653_65331

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) :
  a * b < 1 ∧ 1 < (a^2 + b^2) / 2 := by
  sorry

end inequality_proof_l653_65331


namespace symmetric_point_example_l653_65394

/-- Given a line ax + by + c = 0 and two points P and Q, this function checks if Q is symmetric to P with respect to the line. -/
def is_symmetric_point (a b c : ℝ) (px py qx qy : ℝ) : Prop :=
  let mx := (px + qx) / 2
  let my := (py + qy) / 2
  (a * mx + b * my + c = 0) ∧ (a * (qx - px) + b * (qy - py) = 0)

/-- The point (3, 2) is symmetric to (-1, -2) with respect to the line x + y = 1 -/
theorem symmetric_point_example : is_symmetric_point 1 1 (-1) (-1) (-2) 3 2 := by
  sorry

end symmetric_point_example_l653_65394


namespace snowdrift_final_depth_l653_65343

/-- Calculates the final depth of a snowdrift after four days of weather events. -/
def snowdrift_depth (initial_depth : ℝ) (day2_melt_fraction : ℝ) (day3_snow : ℝ) (day4_snow : ℝ) : ℝ :=
  ((initial_depth * (1 - day2_melt_fraction)) + day3_snow) + day4_snow

/-- Theorem stating that given specific weather conditions over four days,
    the final depth of a snowdrift will be 34 inches. -/
theorem snowdrift_final_depth :
  snowdrift_depth 20 0.5 6 18 = 34 := by
  sorry

end snowdrift_final_depth_l653_65343


namespace sequence_formula_l653_65342

theorem sequence_formula (a : ℕ → ℚ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, n ≥ 1 → (n + 1) * a n = 2 * n * a (n + 1)) :
    ∀ n : ℕ, n ≥ 1 → a n = n / (2^(n-1)) := by sorry

end sequence_formula_l653_65342


namespace negation_of_existence_negation_of_quadratic_inequality_l653_65387

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x - 1 ≥ 0) ↔ (∀ x : ℝ, x^2 + x - 1 < 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l653_65387


namespace triangle_theorem_l653_65324

-- Define a triangle with side lengths and angles
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.A = π/3)  -- A = 60° in radians
  (h2 : t.a = Real.sqrt 13)
  (h3 : t.b = 1) :
  t.c = 4 ∧ (t.a + t.b + t.c) / (Real.sin t.A + Real.sin t.B + Real.sin t.C) = 2 * Real.sqrt 39 / 3 := by
  sorry


end triangle_theorem_l653_65324


namespace triangle_side_length_l653_65383

theorem triangle_side_length (A B C : ℝ) (AC : ℝ) (sinA sinB : ℝ) (cosC : ℝ) :
  AC = 3 →
  3 * sinA = 2 * sinB →
  cosC = 1 / 4 →
  ∃ (AB : ℝ), AB = Real.sqrt 10 := by
  sorry

end triangle_side_length_l653_65383


namespace total_animals_hunted_l653_65323

/- Define the number of animals hunted by each person -/
def sam_hunt : ℕ := 6

def rob_hunt : ℕ := sam_hunt / 2

def rob_sam_total : ℕ := sam_hunt + rob_hunt

def mark_hunt : ℕ := rob_sam_total / 3

def peter_hunt : ℕ := 3 * mark_hunt

/- Theorem to prove -/
theorem total_animals_hunted : sam_hunt + rob_hunt + mark_hunt + peter_hunt = 21 := by
  sorry

end total_animals_hunted_l653_65323


namespace pirate_digging_time_pirate_digging_time_proof_l653_65350

/-- Calculates the time needed to dig up a buried treasure after natural events --/
theorem pirate_digging_time (initial_depth : ℝ) (initial_time : ℝ) 
  (storm_factor : ℝ) (tsunami_sand : ℝ) (earthquake_sand : ℝ) (mudslide_sand : ℝ) 
  (speed_change : ℝ) : ℝ :=
  let initial_speed := initial_depth / initial_time
  let new_speed := initial_speed * (1 - speed_change)
  let final_depth := initial_depth * storm_factor + tsunami_sand + earthquake_sand + mudslide_sand
  final_depth / new_speed

/-- Proves that the time to dig up the treasure is approximately 6.56 hours --/
theorem pirate_digging_time_proof :
  ∃ ε > 0, |pirate_digging_time 8 4 0.5 2 1.5 3 0.2 - 6.56| < ε :=
by
  sorry

end pirate_digging_time_pirate_digging_time_proof_l653_65350


namespace perpendicular_bisector_equation_equal_distances_m_value_l653_65361

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def A : Point := { x := -3, y := -4 }
def B : Point := { x := 6, y := 3 }

def perpendicular_bisector (p1 p2 : Point) : Line := sorry

def distance_to_line (p : Point) (l : Line) : ℝ := sorry

theorem perpendicular_bisector_equation :
  perpendicular_bisector A B = { a := 9, b := 7, c := -10 } := by sorry

theorem equal_distances_m_value (m : ℝ) :
  let l : Line := { a := 1, b := m, c := 1 }
  distance_to_line A l = distance_to_line B l → m = 5 := by sorry

end perpendicular_bisector_equation_equal_distances_m_value_l653_65361


namespace pizza_order_l653_65346

theorem pizza_order (total_slices : ℕ) (slices_per_pizza : ℕ) (h1 : total_slices = 14) (h2 : slices_per_pizza = 2) :
  total_slices / slices_per_pizza = 7 := by
  sorry

end pizza_order_l653_65346


namespace absolute_value_equation_solution_l653_65359

theorem absolute_value_equation_solution (x : ℝ) : 
  |24 / x + 4| = 4 → x = -3 := by sorry

end absolute_value_equation_solution_l653_65359


namespace sequence_sum_problem_l653_65318

def S (n : ℕ) : ℤ :=
  (-1)^(n+1) * (4*n - 3) + if n > 1 then S (n-1) else 0

theorem sequence_sum_problem : S 15 + S 22 - S 31 = -76 := by
  sorry

end sequence_sum_problem_l653_65318


namespace sum_greater_than_four_necessary_not_sufficient_l653_65336

theorem sum_greater_than_four_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, (a > 1 ∧ b > 3) → a + b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ ¬(a > 1 ∧ b > 3)) :=
by sorry

end sum_greater_than_four_necessary_not_sufficient_l653_65336


namespace problem_statement_l653_65320

theorem problem_statement (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum_squares : x^2 + y^2 + z^2 = 6) :
  -- Part 1: Maximum value of x + 2y + z is 6
  (∃ (max : ℝ), max = 6 ∧ x + 2*y + z ≤ max ∧ 
    ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
      x₀^2 + y₀^2 + z₀^2 = 6 ∧ x₀ + 2*y₀ + z₀ = max) ∧
  -- Part 2: If |a+1| - 2a ≥ x + 2y + z for all valid x, y, z, then a ≤ -7/3
  (∀ (a : ℝ), (∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 → 
    x'^2 + y'^2 + z'^2 = 6 → |a + 1| - 2*a ≥ x' + 2*y' + z') → a ≤ -7/3) :=
by sorry

end problem_statement_l653_65320


namespace quadratic_distinct_roots_l653_65368

/-- 
For a quadratic equation x^2 - 4x + m - 1 = 0, 
if it has two distinct real roots, then m < 5.
-/
theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 - 4*x + m - 1 = 0 ∧ 
    y^2 - 4*y + m - 1 = 0) → 
  m < 5 := by
  sorry


end quadratic_distinct_roots_l653_65368


namespace student_heights_average_l653_65348

theorem student_heights_average :
  ∀ (h1 h2 h3 h4 : ℝ),
    h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4 →
    max h1 (max h2 (max h3 h4)) = 152 →
    min h1 (min h2 (min h3 h4)) = 137 →
    ∃ (avg : ℝ), avg = 145 ∧ (h1 + h2 + h3 + h4) / 4 = avg :=
by sorry

end student_heights_average_l653_65348


namespace circle_equation_specific_l653_65371

/-- The equation of a circle with center (h, k) and radius r -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The equation of a circle with center (2, -3) and radius 4 -/
theorem circle_equation_specific : 
  CircleEquation 2 (-3) 4 x y ↔ (x - 2)^2 + (y + 3)^2 = 16 :=
by sorry

end circle_equation_specific_l653_65371


namespace quadratic_equation_properties_l653_65301

/-- The quadratic equation x^2 + (2k+1)x + k^2 + 1 = 0 has two distinct real roots -/
def has_distinct_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + (2*k+1)*x₁ + k^2 + 1 = 0 ∧ x₂^2 + (2*k+1)*x₂ + k^2 + 1 = 0

/-- The product of the roots of the quadratic equation is 5 -/
def roots_product_is_5 (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ * x₂ = 5 ∧ x₁^2 + (2*k+1)*x₁ + k^2 + 1 = 0 ∧ x₂^2 + (2*k+1)*x₂ + k^2 + 1 = 0

theorem quadratic_equation_properties :
  (∀ k : ℝ, has_distinct_roots k ↔ k > 3/4) ∧
  (∀ k : ℝ, roots_product_is_5 k → k = 2) := by
  sorry

end quadratic_equation_properties_l653_65301


namespace polynomial_roots_l653_65300

theorem polynomial_roots : 
  let p : ℝ → ℝ := λ x => 3*x^4 + 17*x^3 - 32*x^2 - 12*x
  (p 0 = 0) ∧ 
  (p (-1/2) = 0) ∧ 
  (p (4/3) = 0) ∧ 
  (p (-3) = 0) :=
by sorry

end polynomial_roots_l653_65300


namespace smallest_n_satisfying_congruences_l653_65330

theorem smallest_n_satisfying_congruences : 
  ∃ n : ℕ, n > 20 ∧ n % 6 = 4 ∧ n % 7 = 5 ∧ 
  ∀ m : ℕ, m > 20 ∧ m % 6 = 4 ∧ m % 7 = 5 → n ≤ m :=
by
  use 40
  sorry

end smallest_n_satisfying_congruences_l653_65330


namespace line_equation_sum_l653_65321

/-- Given a line with slope -3 passing through the point (5, 2),
    prove that m + b = 14 where y = mx + b is the equation of the line. -/
theorem line_equation_sum (m b : ℝ) : 
  m = -3 → 
  2 = m * 5 + b → 
  m + b = 14 := by
  sorry

end line_equation_sum_l653_65321


namespace min_value_a5_plus_a6_l653_65307

/-- A positive arithmetic geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem min_value_a5_plus_a6 (a : ℕ → ℝ) 
    (h_seq : ArithmeticGeometricSequence a) 
    (h_cond : a 4 + a 3 - a 2 - a 1 = 1) :
    ∃ (min_val : ℝ), min_val = 4 ∧ 
    (∀ a5 a6, a 5 = a5 → a 6 = a6 → a5 + a6 ≥ min_val) ∧
    (∃ a5 a6, a 5 = a5 ∧ a 6 = a6 ∧ a5 + a6 = min_val) := by
  sorry

end min_value_a5_plus_a6_l653_65307


namespace samantha_birth_year_proof_l653_65313

/-- The year when the first AMC 8 was held -/
def first_amc8_year : ℕ := 1985

/-- The number of the AMC 8 in which Samantha participated -/
def samantha_amc8_number : ℕ := 7

/-- Samantha's age when she participated in the AMC 8 -/
def samantha_age_at_amc8 : ℕ := 12

/-- The year when Samantha was born -/
def samantha_birth_year : ℕ := 1979

theorem samantha_birth_year_proof :
  samantha_birth_year = first_amc8_year + (samantha_amc8_number - 1) - samantha_age_at_amc8 :=
by sorry

end samantha_birth_year_proof_l653_65313


namespace perpendicular_parallel_perpendicular_l653_65376

-- Define a type for lines in 3D space
structure Line3D where
  -- You might represent a line by a point and a direction vector
  -- But for this abstract proof, we don't need to specify the internals
  mk :: 

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define parallelism for lines
def parallel (l1 l2 : Line3D) : Prop := sorry

-- The theorem to prove
theorem perpendicular_parallel_perpendicular 
  (l1 l2 l3 : Line3D) : 
  perpendicular l1 l2 → parallel l2 l3 → perpendicular l1 l3 := by
  sorry

end perpendicular_parallel_perpendicular_l653_65376


namespace square_difference_division_eleven_l653_65317

theorem square_difference_division_eleven : (121^2 - 110^2) / 11 = 231 := by
  sorry

end square_difference_division_eleven_l653_65317


namespace correct_number_of_seasons_l653_65310

/-- Represents the number of seasons in a TV show. -/
def num_seasons : ℕ := 5

/-- Cost per episode for the first season. -/
def first_season_cost : ℕ := 100000

/-- Cost per episode for other seasons. -/
def other_season_cost : ℕ := 200000

/-- Number of episodes in the first season. -/
def first_season_episodes : ℕ := 12

/-- Number of episodes in middle seasons. -/
def middle_season_episodes : ℕ := 18

/-- Number of episodes in the last season. -/
def last_season_episodes : ℕ := 24

/-- Total cost of producing all episodes. -/
def total_cost : ℕ := 16800000

/-- Theorem stating that the number of seasons is correct given the conditions. -/
theorem correct_number_of_seasons :
  (first_season_cost * first_season_episodes) +
  ((num_seasons - 2) * other_season_cost * middle_season_episodes) +
  (other_season_cost * last_season_episodes) = total_cost :=
sorry

end correct_number_of_seasons_l653_65310


namespace system_of_inequalities_solution_l653_65358

theorem system_of_inequalities_solution (x : ℝ) :
  (x < x / 5 + 4 ∧ 4 * x + 1 > 3 * (2 * x - 1)) → x < 2 := by
  sorry

end system_of_inequalities_solution_l653_65358


namespace intersection_of_p_and_q_when_a_is_one_range_of_a_for_not_p_sufficient_for_not_q_l653_65305

/-- Proposition p: x^2 - 5ax + 4a^2 < 0, where a > 0 -/
def p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0 ∧ a > 0

/-- Proposition q: x^2 - 2x - 8 ≤ 0 and x^2 + 3x - 10 > 0 -/
def q (x : ℝ) : Prop := x^2 - 2*x - 8 ≤ 0 ∧ x^2 + 3*x - 10 > 0

/-- The solution set of p -/
def solution_set_p (a : ℝ) : Set ℝ := {x | p x a}

/-- The solution set of q -/
def solution_set_q : Set ℝ := {x | q x}

theorem intersection_of_p_and_q_when_a_is_one :
  (solution_set_p 1) ∩ solution_set_q = Set.Ioo 2 4 := by sorry

theorem range_of_a_for_not_p_sufficient_for_not_q :
  {a : ℝ | ∀ x, ¬(p x a) → ¬(q x)} = Set.Icc 1 2 := by sorry

end intersection_of_p_and_q_when_a_is_one_range_of_a_for_not_p_sufficient_for_not_q_l653_65305


namespace circle_equation_l653_65389

/-- A circle with center on the x-axis, passing through the origin, and tangent to the line y = 4 has the general equation x^2 + y^2 ± 8x = 0. -/
theorem circle_equation (x y : ℝ) : 
  ∃ (a : ℝ), 
    (∀ (x₀ y₀ : ℝ), (x₀ - a)^2 + y₀^2 = 16 → y₀ ≤ 4) ∧  -- Circle is tangent to y = 4
    ((0 - a)^2 + 0^2 = 16) ∧                             -- Circle passes through origin
    (a^2 = 16) →                                         -- Center is on x-axis at distance 4 from origin
  (x^2 + y^2 + 8*x = 0 ∨ x^2 + y^2 - 8*x = 0) :=
by sorry

end circle_equation_l653_65389


namespace theater_attendance_l653_65312

theorem theater_attendance (adult_price child_price total_people total_revenue : ℕ)
  (h1 : adult_price = 8)
  (h2 : child_price = 1)
  (h3 : total_people = 22)
  (h4 : total_revenue = 50) :
  ∃ (adults children : ℕ),
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_revenue ∧
    children = 18 := by
  sorry

end theater_attendance_l653_65312


namespace NaHSO3_moles_required_l653_65351

/-- Represents the balanced chemical equation for the reaction -/
structure ChemicalEquation :=
  (reactants : List String)
  (products : List String)

/-- Represents the stoichiometric coefficient of a substance in a reaction -/
def stoichiometricCoefficient (equation : ChemicalEquation) (substance : String) : ℕ :=
  if substance ∈ equation.reactants ∨ substance ∈ equation.products then 1 else 0

/-- The chemical equation for the given reaction -/
def reactionEquation : ChemicalEquation :=
  { reactants := ["NaHSO3", "HCl"],
    products := ["SO2", "H2O", "NaCl"] }

/-- Theorem stating the number of moles of NaHSO3 required to form 2 moles of SO2 -/
theorem NaHSO3_moles_required :
  let NaHSO3_coeff := stoichiometricCoefficient reactionEquation "NaHSO3"
  let SO2_coeff := stoichiometricCoefficient reactionEquation "SO2"
  let SO2_moles_formed := 2
  NaHSO3_coeff * SO2_moles_formed / SO2_coeff = 2 := by
  sorry

end NaHSO3_moles_required_l653_65351


namespace fixed_point_of_exponential_function_l653_65311

/-- The function f(x) = a^(x-1) + 3 passes through the point (1, 4) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by
  sorry

end fixed_point_of_exponential_function_l653_65311


namespace exists_number_with_digit_sum_div_11_l653_65362

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem exists_number_with_digit_sum_div_11 (N : ℕ) : 
  ∃ k ∈ Finset.range 39, 11 ∣ sum_of_digits (N + k) := by sorry

end exists_number_with_digit_sum_div_11_l653_65362


namespace win_sector_area_l653_65366

/-- Given a circular spinner with radius 8 cm and a probability of winning 3/8,
    the area of the WIN sector is 24π square centimeters. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/8) :
  p * π * r^2 = 24 * π := by
sorry

end win_sector_area_l653_65366


namespace absolute_value_equation_solution_l653_65304

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 3 * y = 12 :=
by
  -- The unique solution is y = 2
  use 2
  constructor
  · -- Prove that y = 2 satisfies the equation
    simp
    norm_num
  · -- Prove uniqueness
    intro z hz
    -- Proof goes here
    sorry

#check absolute_value_equation_solution

end absolute_value_equation_solution_l653_65304


namespace power_of_seven_mod_2000_l653_65325

theorem power_of_seven_mod_2000 : 7^2023 % 2000 = 1849 := by
  sorry

end power_of_seven_mod_2000_l653_65325


namespace cut_rectangle_properties_l653_65373

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the result of cutting a smaller rectangle from a larger one -/
structure CutRectangle where
  original : Rectangle
  cut : Rectangle

/-- The resulting figure after cutting -/
def resultingFigure (cr : CutRectangle) : ℝ := area cr.original - area cr.cut

theorem cut_rectangle_properties (R : Rectangle) (S : CutRectangle) 
    (h1 : S.original = R) 
    (h2 : area S.cut > 0) 
    (h3 : S.cut.length < R.length ∧ S.cut.width < R.width) :
  resultingFigure S < area R ∧ perimeter R = perimeter S.original :=
by sorry

end cut_rectangle_properties_l653_65373


namespace definite_integral_exp_plus_2x_l653_65365

theorem definite_integral_exp_plus_2x : 
  ∫ x in (-1)..1, (Real.exp x + 2 * x) = Real.exp 1 - Real.exp (-1) := by sorry

end definite_integral_exp_plus_2x_l653_65365


namespace geometric_sequence_ratio_l653_65385

/-- Given a geometric sequence {a_n} with all positive terms, where 3a_1, (1/2)a_3, 2a_2 form an arithmetic sequence, (a_11 + a_13) / (a_8 + a_10) = 27. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∃ q : ℝ, ∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  (3 * a 1 - (1/2) * a 3 = (1/2) * a 3 - 2 * a 2) →  -- arithmetic sequence condition
  (a 11 + a 13) / (a 8 + a 10) = 27 := by
sorry

end geometric_sequence_ratio_l653_65385


namespace difference_of_squares_153_147_l653_65344

theorem difference_of_squares_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end difference_of_squares_153_147_l653_65344


namespace t_equality_l653_65381

theorem t_equality (t : ℝ) : t = 1 / (1 - 2^(1/4)) → t = -(1 + 2^(1/4)) * (1 + 2^(1/2)) := by
  sorry

end t_equality_l653_65381


namespace breakfast_cost_theorem_l653_65333

/-- Represents the cost of breakfast items and the special offer. -/
structure BreakfastPrices where
  toast : ℚ
  egg : ℚ
  coffee : ℚ
  orange_juice : ℚ
  special_offer : ℚ

/-- Represents an individual's breakfast order. -/
structure BreakfastOrder where
  toast : ℕ
  egg : ℕ
  coffee : ℕ
  orange_juice : ℕ

/-- Calculates the cost of a breakfast order given the prices. -/
def orderCost (prices : BreakfastPrices) (order : BreakfastOrder) : ℚ :=
  prices.toast * order.toast +
  prices.egg * order.egg +
  (if order.coffee ≥ 2 then prices.special_offer else prices.coffee * order.coffee) +
  prices.orange_juice * order.orange_juice

/-- Calculates the total cost of all breakfast orders with service charge. -/
def totalCost (prices : BreakfastPrices) (orders : List BreakfastOrder) (serviceCharge : ℚ) : ℚ :=
  let subtotal := (orders.map (orderCost prices)).sum
  subtotal + subtotal * serviceCharge

/-- Theorem stating that the total breakfast cost is £48.40. -/
theorem breakfast_cost_theorem (prices : BreakfastPrices)
    (dale andrew melanie kevin : BreakfastOrder) :
    prices.toast = 1 →
    prices.egg = 3 →
    prices.coffee = 2 →
    prices.orange_juice = 3/2 →
    prices.special_offer = 7/2 →
    dale = { toast := 2, egg := 2, coffee := 1, orange_juice := 0 } →
    andrew = { toast := 1, egg := 2, coffee := 0, orange_juice := 1 } →
    melanie = { toast := 3, egg := 1, coffee := 0, orange_juice := 2 } →
    kevin = { toast := 4, egg := 3, coffee := 2, orange_juice := 0 } →
    totalCost prices [dale, andrew, melanie, kevin] (1/10) = 484/10 := by
  sorry


end breakfast_cost_theorem_l653_65333


namespace derek_average_increase_l653_65395

def derek_scores : List ℝ := [92, 86, 89, 94, 91]

theorem derek_average_increase :
  let first_three := derek_scores.take 3
  let all_five := derek_scores
  (all_five.sum / all_five.length) - (first_three.sum / first_three.length) = 1.4 := by
  sorry

end derek_average_increase_l653_65395


namespace parabola_directrix_l653_65396

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the points
def origin : ℝ × ℝ := (0, 0)
def point_D : ℝ × ℝ := (1, 2)

-- Define the perpendicularity condition
def perpendicular (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  (p2.1 - p1.1) * (p4.1 - p3.1) + (p2.2 - p1.2) * (p4.2 - p3.2) = 0

-- State the theorem
theorem parabola_directrix (p : ℝ) (A B : ℝ × ℝ) :
  parabola p A.1 A.2 ∧ 
  parabola p B.1 B.2 ∧ 
  perpendicular origin A origin B ∧
  perpendicular origin point_D A B →
  ∃ (x : ℝ), x = -5/4 ∧ ∀ (y : ℝ), parabola p x y → x = -p/2 :=
sorry

end parabola_directrix_l653_65396


namespace sampling_consistency_l653_65370

def systematic_sampling (n : ℕ) (k : ℕ) (i : ℕ) : Prop :=
  ∃ (r : ℕ), i = r * k ∧ r ≤ n / k

theorem sampling_consistency 
  (total : ℕ) (sample_size : ℕ) (selected : ℕ) (h_total : total = 800) (h_sample : sample_size = 50)
  (h_selected : selected = 39) (h_interval : total / sample_size = 16) :
  systematic_sampling total (total / sample_size) selected → 
  systematic_sampling total (total / sample_size) 7 :=
by sorry

end sampling_consistency_l653_65370


namespace quadratic_roots_sum_l653_65352

theorem quadratic_roots_sum (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 2023 = 0 → x₂^2 + x₂ - 2023 = 0 → x₁^2 + 2*x₁ + x₂ = 2022 := by
  sorry

end quadratic_roots_sum_l653_65352


namespace quadratic_real_roots_k_range_l653_65355

/-- Given a quadratic equation x^2 + 2x - k = 0 with real roots, prove that k ≥ -1 -/
theorem quadratic_real_roots_k_range (k : ℝ) :
  (∃ x : ℝ, x^2 + 2*x - k = 0) →
  k ≥ -1 := by
  sorry

end quadratic_real_roots_k_range_l653_65355


namespace statements_correctness_l653_65374

theorem statements_correctness : 
  (∃! n : ℕ, n = 3 ∧ 
    (2^3 = 8) ∧ 
    (∀ r : ℚ, ∃ s : ℚ, s < r) ∧ 
    (∀ x : ℝ, x + x = 0 → x = 0) ∧ 
    (Real.sqrt ((-4)^2) ≠ 4) ∧ 
    (∃ x : ℝ, x ≠ 1 ∧ 1 / x ≠ 1)) :=
by sorry

end statements_correctness_l653_65374


namespace range_of_m_l653_65356

/-- Given two predicates p and q on real numbers, where p states that there exists a real x such that
    mx² + 1 ≤ 0, and q states that for all real x, x² + mx + 1 > 0, if the disjunction of p and q
    is false, then m is greater than or equal to 2. -/
theorem range_of_m (m : ℝ) : 
  let p := ∃ x : ℝ, m * x^2 + 1 ≤ 0
  let q := ∀ x : ℝ, x^2 + m * x + 1 > 0
  ¬(p ∨ q) → m ≥ 2 := by
  sorry

end range_of_m_l653_65356


namespace distinct_remainders_l653_65391

theorem distinct_remainders (n : ℕ+) :
  ∀ (i j : Fin n), i ≠ j →
    (2 * i.val + 1) ^ (2 * i.val + 1) % (2 ^ n.val) ≠
    (2 * j.val + 1) ^ (2 * j.val + 1) % (2 ^ n.val) := by
  sorry

end distinct_remainders_l653_65391


namespace circle_area_when_six_reciprocal_circumference_equals_diameter_l653_65380

/-- Given a circle where six times the reciprocal of its circumference equals its diameter, the area of the circle is 3/2 -/
theorem circle_area_when_six_reciprocal_circumference_equals_diameter (r : ℝ) (h : 6 * (1 / (2 * Real.pi * r)) = 2 * r) : π * r^2 = 3/2 := by
  sorry

end circle_area_when_six_reciprocal_circumference_equals_diameter_l653_65380


namespace complex_division_result_l653_65319

theorem complex_division_result (z : ℂ) (h : z = 1 - Complex.I * Real.sqrt 3) : 
  4 / z = 1 + Complex.I * Real.sqrt 3 := by
sorry

end complex_division_result_l653_65319


namespace train_speed_l653_65390

/-- Proves that a train with given passing times has a specific speed -/
theorem train_speed (pole_passing_time : ℝ) (stationary_train_length : ℝ) (stationary_train_passing_time : ℝ) :
  pole_passing_time = 10 →
  stationary_train_length = 300 →
  stationary_train_passing_time = 40 →
  ∃ (train_length : ℝ),
    train_length > 0 ∧
    train_length / pole_passing_time = (train_length + stationary_train_length) / stationary_train_passing_time ∧
    train_length / pole_passing_time = 10 :=
by sorry

end train_speed_l653_65390
