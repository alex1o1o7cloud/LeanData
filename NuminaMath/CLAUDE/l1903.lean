import Mathlib

namespace NUMINAMATH_CALUDE_z_purely_imaginary_z_in_fourth_quadrant_l1903_190331

/-- Definition of the complex number z as a function of m -/
def z (m : ℝ) : ℂ := Complex.mk (2*m^2 - 7*m + 6) (m^2 - m - 2)

/-- z is purely imaginary iff m = 3/2 -/
theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * (z m).im ↔ m = 3/2 := by
  sorry

/-- z is in the fourth quadrant iff -1 < m < 3/2 -/
theorem z_in_fourth_quadrant (m : ℝ) : 
  (z m).re > 0 ∧ (z m).im < 0 ↔ -1 < m ∧ m < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_z_in_fourth_quadrant_l1903_190331


namespace NUMINAMATH_CALUDE_distance_n_n_l1903_190387

/-- The distance function for a point (a,b) on the polygonal path -/
def distance (a b : ℕ) : ℕ := sorry

/-- The theorem stating that the distance of (n,n) is n^2 + n -/
theorem distance_n_n (n : ℕ) : distance n n = n^2 + n := by sorry

end NUMINAMATH_CALUDE_distance_n_n_l1903_190387


namespace NUMINAMATH_CALUDE_stratified_sampling_l1903_190337

theorem stratified_sampling (total_employees : ℕ) (employees_over_30 : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 49)
  (h2 : employees_over_30 = 14)
  (h3 : sample_size = 7) :
  ↑employees_over_30 / ↑total_employees * ↑sample_size = 2 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l1903_190337


namespace NUMINAMATH_CALUDE_inequality_proof_l1903_190329

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / (x^4 + y^2) + y / (x^2 + y^4) ≤ 1 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1903_190329


namespace NUMINAMATH_CALUDE_almost_square_quotient_l1903_190368

/-- Definition of an almost square -/
def AlmostSquare (k : ℕ) : Prop := ∃ n : ℕ, k = n * (n + 1)

/-- Theorem: Every almost square can be expressed as a quotient of two almost squares -/
theorem almost_square_quotient (n : ℕ) : 
  ∃ a b : ℕ, AlmostSquare a ∧ AlmostSquare b ∧ n * (n + 1) = a / b := by
  sorry

end NUMINAMATH_CALUDE_almost_square_quotient_l1903_190368


namespace NUMINAMATH_CALUDE_square_difference_65_35_l1903_190388

theorem square_difference_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_65_35_l1903_190388


namespace NUMINAMATH_CALUDE_sphere_volume_to_surface_area_l1903_190320

theorem sphere_volume_to_surface_area :
  ∀ (r : ℝ), 
    (4 / 3 : ℝ) * π * r^3 = 4 * Real.sqrt 3 * π → 
    4 * π * r^2 = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_to_surface_area_l1903_190320


namespace NUMINAMATH_CALUDE_chandler_saves_for_bike_l1903_190361

/-- The number of weeks needed for Chandler to save enough money to buy the mountain bike -/
def weeks_to_save : ℕ → Prop :=
  λ w => 
    let bike_cost : ℕ := 600
    let birthday_money : ℕ := 60 + 40 + 20
    let weekly_earnings : ℕ := 20
    let weekly_expenses : ℕ := 4
    let weekly_savings : ℕ := weekly_earnings - weekly_expenses
    birthday_money + w * weekly_savings = bike_cost

theorem chandler_saves_for_bike : weeks_to_save 30 := by
  sorry

end NUMINAMATH_CALUDE_chandler_saves_for_bike_l1903_190361


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1903_190365

/-- Given a point A in a Cartesian coordinate system with coordinates (-2, -3),
    its coordinates with respect to the origin are also (-2, -3). -/
theorem point_coordinates_wrt_origin :
  ∀ (A : ℝ × ℝ), A = (-2, -3) → A = (-2, -3) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1903_190365


namespace NUMINAMATH_CALUDE_positive_terms_count_l1903_190317

/-- The number of positive terms in an arithmetic sequence with general term a_n = 90 - 2n -/
theorem positive_terms_count : ∃ k : ℕ, k = 44 ∧ 
  ∀ n : ℕ+, (90 : ℝ) - 2 * (n : ℝ) > 0 ↔ n ≤ k := by sorry

end NUMINAMATH_CALUDE_positive_terms_count_l1903_190317


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l1903_190367

theorem smallest_positive_integer_congruence :
  ∃ (y : ℕ), y > 0 ∧ y + 3721 ≡ 803 [ZMOD 17] ∧
  ∀ (z : ℕ), z > 0 ∧ z + 3721 ≡ 803 [ZMOD 17] → y ≤ z ∧ y = 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l1903_190367


namespace NUMINAMATH_CALUDE_petes_number_l1903_190339

theorem petes_number : ∃ x : ℚ, 5 * (3 * x + 15) = 200 ∧ x = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l1903_190339


namespace NUMINAMATH_CALUDE_remainder_theorem_l1903_190374

theorem remainder_theorem (x y u v : ℕ) (hx : x > 0) (hy : y > 0) (hv : v < y) (hxdiv : x = u * y + v) :
  (x + 3 * u * y + 4) % y = (v + 4) % y :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1903_190374


namespace NUMINAMATH_CALUDE_all_positive_integers_l1903_190363

def is_valid_set (A : Set ℕ) : Prop :=
  1 ∈ A ∧
  ∃ k : ℕ, k ≠ 1 ∧ k ∈ A ∧
  ∀ m n : ℕ, m ∈ A → n ∈ A → m ≠ n →
    ((m + 1) / (Nat.gcd (m + 1) (n + 1))) ∈ A

theorem all_positive_integers (A : Set ℕ) :
  is_valid_set A → A = {n : ℕ | n > 0} :=
by sorry

end NUMINAMATH_CALUDE_all_positive_integers_l1903_190363


namespace NUMINAMATH_CALUDE_largest_among_decimals_l1903_190383

theorem largest_among_decimals :
  let a := 0.989
  let b := 0.997
  let c := 0.991
  let d := 0.999
  let e := 0.990
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end NUMINAMATH_CALUDE_largest_among_decimals_l1903_190383


namespace NUMINAMATH_CALUDE_regular_nonagon_diagonal_relation_l1903_190389

/-- Regular nonagon -/
structure RegularNonagon where
  /-- Length of a side -/
  a : ℝ
  /-- Length of the shortest diagonal -/
  b : ℝ
  /-- Length of the longest diagonal -/
  d : ℝ
  /-- a is positive -/
  a_pos : a > 0

/-- Theorem: In a regular nonagon, d^2 = a^2 + ab + b^2 -/
theorem regular_nonagon_diagonal_relation (N : RegularNonagon) : N.d^2 = N.a^2 + N.a*N.b + N.b^2 := by
  sorry

end NUMINAMATH_CALUDE_regular_nonagon_diagonal_relation_l1903_190389


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_relationship_l1903_190321

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (skew_lines : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_relationship
  (a b : Line) (α : Plane)
  (h1 : parallel_line_plane a α)
  (h2 : contained_in_plane b α) :
  parallel_lines a b ∨ skew_lines a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_relationship_l1903_190321


namespace NUMINAMATH_CALUDE_park_length_l1903_190340

theorem park_length (perimeter breadth : ℝ) (h1 : perimeter = 1000) (h2 : breadth = 200) :
  let length := (perimeter - 2 * breadth) / 2
  length = 300 :=
by
  sorry

#check park_length

end NUMINAMATH_CALUDE_park_length_l1903_190340


namespace NUMINAMATH_CALUDE_derek_age_l1903_190336

/-- Given the ages of Uncle Bob, Evan, and Derek, prove Derek's age -/
theorem derek_age (uncle_bob_age : ℕ) (evan_age : ℕ) (derek_age : ℕ) : 
  uncle_bob_age = 60 →
  evan_age = 2 * uncle_bob_age / 3 →
  derek_age = evan_age - 10 →
  derek_age = 30 := by
sorry

end NUMINAMATH_CALUDE_derek_age_l1903_190336


namespace NUMINAMATH_CALUDE_inequality_holds_function_increasing_l1903_190364

theorem inequality_holds (x : ℝ) (h : x ≥ 1) : x / 2 ≥ (x - 1) / (x + 1) := by
  sorry

theorem function_increasing (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hxy : x < y) :
  (x / 2 - (x - 1) / (x + 1)) < (y / 2 - (y - 1) / (y + 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_function_increasing_l1903_190364


namespace NUMINAMATH_CALUDE_kevin_savings_exceeds_ten_l1903_190397

def kevin_savings (n : ℕ) : ℚ :=
  2 * (3^n - 1) / (3 - 1)

theorem kevin_savings_exceeds_ten :
  ∃ n : ℕ, kevin_savings n > 1000 ∧ ∀ m : ℕ, m < n → kevin_savings m ≤ 1000 :=
by sorry

end NUMINAMATH_CALUDE_kevin_savings_exceeds_ten_l1903_190397


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l1903_190357

theorem complementary_angles_ratio (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Angles are positive
  a + b = 90 ∧     -- Angles are complementary
  a / b = 5 / 4 →  -- Ratio of angles is 5:4
  a = 50 :=        -- Larger angle is 50°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l1903_190357


namespace NUMINAMATH_CALUDE_power_of_power_of_two_l1903_190324

theorem power_of_power_of_two : (2^2)^(2^2) = 256 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_of_two_l1903_190324


namespace NUMINAMATH_CALUDE_card_probability_l1903_190332

theorem card_probability : 
  let total_cards : ℕ := 52
  let cards_per_suit : ℕ := 13
  let top_cards : ℕ := 4
  let favorable_suits : ℕ := 2  -- spades and clubs

  let favorable_outcomes : ℕ := favorable_suits * (cards_per_suit.descFactorial top_cards)
  let total_outcomes : ℕ := total_cards.descFactorial top_cards

  (favorable_outcomes : ℚ) / total_outcomes = 286 / 54145 := by sorry

end NUMINAMATH_CALUDE_card_probability_l1903_190332


namespace NUMINAMATH_CALUDE_polynomial_xy_coefficient_l1903_190304

theorem polynomial_xy_coefficient (k : ℝ) : 
  (∀ x y : ℝ, x^2 - 3*k*x*y - 3*y^2 + 6*x*y - 8 = x^2 + (-3*k + 6)*x*y - 3*y^2 - 8) →
  (-3*k + 6 = 0) →
  k = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_xy_coefficient_l1903_190304


namespace NUMINAMATH_CALUDE_largest_number_l1903_190353

theorem largest_number (a b c d : ℝ) (h1 : a = 3) (h2 : b = -7) (h3 : c = 0) (h4 : d = 1/9) :
  a = max a (max b (max c d)) :=
sorry

end NUMINAMATH_CALUDE_largest_number_l1903_190353


namespace NUMINAMATH_CALUDE_at_most_two_sides_equal_longest_diagonal_l1903_190308

/-- A convex polygon -/
structure ConvexPolygon where
  -- We don't need to define the full structure, just declare it exists
  mk :: 

/-- The longest diagonal of a convex polygon -/
def longest_diagonal (p : ConvexPolygon) : ℝ := sorry

/-- A side of a convex polygon -/
def side (p : ConvexPolygon) : ℝ := sorry

/-- The number of sides in a convex polygon that are equal to the longest diagonal -/
def num_sides_equal_to_longest_diagonal (p : ConvexPolygon) : ℕ := sorry

/-- Theorem: At most two sides of a convex polygon can be equal to its longest diagonal -/
theorem at_most_two_sides_equal_longest_diagonal (p : ConvexPolygon) :
  num_sides_equal_to_longest_diagonal p ≤ 2 := by sorry

end NUMINAMATH_CALUDE_at_most_two_sides_equal_longest_diagonal_l1903_190308


namespace NUMINAMATH_CALUDE_bird_count_l1903_190398

theorem bird_count (total_wings : ℕ) (wings_per_bird : ℕ) (h1 : total_wings = 26) (h2 : wings_per_bird = 2) :
  total_wings / wings_per_bird = 13 := by
  sorry

end NUMINAMATH_CALUDE_bird_count_l1903_190398


namespace NUMINAMATH_CALUDE_technicians_count_l1903_190312

/-- The number of technicians in a workshop with given salary conditions -/
def num_technicians (total_workers : ℕ) (avg_salary_all : ℚ) (avg_salary_tech : ℚ) (avg_salary_rest : ℚ) : ℕ :=
  7

/-- Theorem stating that the number of technicians is 7 under the given conditions -/
theorem technicians_count :
  let total_workers : ℕ := 21
  let avg_salary_all : ℚ := 8000
  let avg_salary_tech : ℚ := 12000
  let avg_salary_rest : ℚ := 6000
  num_technicians total_workers avg_salary_all avg_salary_tech avg_salary_rest = 7 := by
  sorry

#check technicians_count

end NUMINAMATH_CALUDE_technicians_count_l1903_190312


namespace NUMINAMATH_CALUDE_integral_reciprocal_plus_one_l1903_190373

theorem integral_reciprocal_plus_one : ∫ x in (0:ℝ)..1, 1 / (1 + x) = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_plus_one_l1903_190373


namespace NUMINAMATH_CALUDE_danny_found_fifty_caps_l1903_190323

/-- The number of wrappers Danny found at the park -/
def wrappers_found : ℕ := 46

/-- The total number of bottle caps Danny has now -/
def total_caps : ℕ := 21

/-- The total number of wrappers Danny has now -/
def total_wrappers : ℕ := 52

/-- The difference between bottle caps and wrappers found at the park -/
def cap_wrapper_difference : ℕ := 4

/-- The number of bottle caps Danny found at the park -/
def caps_found : ℕ := wrappers_found + cap_wrapper_difference

theorem danny_found_fifty_caps : caps_found = 50 := by
  sorry

end NUMINAMATH_CALUDE_danny_found_fifty_caps_l1903_190323


namespace NUMINAMATH_CALUDE_day_crew_loading_fraction_l1903_190362

theorem day_crew_loading_fraction (D : ℚ) (Wd : ℚ) : 
  D > 0 → Wd > 0 →
  (D * Wd) / ((D * Wd) + ((3/4 * D) * (4/9 * Wd))) = 3/4 := by
sorry

end NUMINAMATH_CALUDE_day_crew_loading_fraction_l1903_190362


namespace NUMINAMATH_CALUDE_soccer_league_games_l1903_190335

/-- The number of teams in the soccer league -/
def num_teams : ℕ := 12

/-- The number of times each pair of teams plays against each other -/
def games_per_pair : ℕ := 4

/-- The total number of games played in the season -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2 * games_per_pair

theorem soccer_league_games :
  total_games = 264 :=
sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1903_190335


namespace NUMINAMATH_CALUDE_stirling_bounds_l1903_190379

-- Define e as the limit of (1 + 1/n)^n as n approaches infinity
noncomputable def e : ℝ := Real.exp 1

-- State the theorem
theorem stirling_bounds (n : ℕ) (h : n > 6) :
  (n / e : ℝ)^n < n! ∧ (n! : ℝ) < n * (n / e)^n :=
sorry

end NUMINAMATH_CALUDE_stirling_bounds_l1903_190379


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_three_l1903_190316

def A : Set ℝ := {2, 3}
def B (a : ℝ) : Set ℝ := {1, 2, a}

theorem subset_implies_a_equals_three (h : A ⊆ B a) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_three_l1903_190316


namespace NUMINAMATH_CALUDE_winning_pair_probability_l1903_190385

/-- Represents the color of a card -/
inductive Color
| Blue
| Purple

/-- Represents the letter on a card -/
inductive Letter
| A | B | C | D | E | F

/-- Represents a card with a color and a letter -/
structure Card where
  color : Color
  letter : Letter

/-- The deck of cards -/
def deck : List Card := sorry

/-- Checks if two cards form a winning pair -/
def is_winning_pair (c1 c2 : Card) : Bool := sorry

/-- Calculates the probability of drawing a winning pair -/
def probability_winning_pair : ℚ := sorry

/-- Theorem stating the probability of drawing a winning pair -/
theorem winning_pair_probability : 
  probability_winning_pair = 29 / 45 := by sorry

end NUMINAMATH_CALUDE_winning_pair_probability_l1903_190385


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1903_190341

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (x - 14) = 2) ∧ (x = 18) :=
by
  sorry

#check sqrt_equation_solution

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1903_190341


namespace NUMINAMATH_CALUDE_power_sum_inequality_l1903_190394

theorem power_sum_inequality (a b c : ℝ) (m : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hm : 0 < m) :
  (a + b)^m + (b + c)^m + (c + a)^m ≤ 2^m * (a^m + b^m + c^m) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l1903_190394


namespace NUMINAMATH_CALUDE_probability_sum_11_three_dice_l1903_190351

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The target sum we're looking for -/
def targetSum : ℕ := 11

/-- The number of dice being rolled -/
def numDice : ℕ := 3

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of ways to roll a sum of 11 with three dice -/
def favorableOutcomes : ℕ := 24

/-- The probability of rolling a sum of 11 with three standard six-sided dice is 1/9 -/
theorem probability_sum_11_three_dice : 
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_sum_11_three_dice_l1903_190351


namespace NUMINAMATH_CALUDE_complex_number_equality_l1903_190358

theorem complex_number_equality (b : ℝ) : 
  (Complex.re ((1 + b * Complex.I) / (1 - Complex.I)) = 
   Complex.im ((1 + b * Complex.I) / (1 - Complex.I))) → b = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1903_190358


namespace NUMINAMATH_CALUDE_band_members_minimum_l1903_190391

theorem band_members_minimum (n : ℕ) : n = 165 ↔ 
  n > 0 ∧ 
  n % 6 = 3 ∧ 
  n % 8 = 5 ∧ 
  n % 9 = 7 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 3 → m % 8 = 5 → m % 9 = 7 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_band_members_minimum_l1903_190391


namespace NUMINAMATH_CALUDE_projectile_height_l1903_190303

theorem projectile_height (t : ℝ) : 
  t > 0 ∧ -16 * t^2 + 80 * t = 36 ∧ 
  (∀ s, s > 0 ∧ -16 * s^2 + 80 * s = 36 → t ≤ s) → 
  t = 0.5 := by sorry

end NUMINAMATH_CALUDE_projectile_height_l1903_190303


namespace NUMINAMATH_CALUDE_line_equation_proof_l1903_190322

theorem line_equation_proof (x y : ℝ) :
  let point_A : ℝ × ℝ := (1, 3)
  let slope_reference : ℝ := -4
  let slope_line : ℝ := slope_reference / 3
  (4 * x + 3 * y - 13 = 0) ↔
    (y - point_A.2 = slope_line * (x - point_A.1) ∧
     slope_line = slope_reference / 3) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1903_190322


namespace NUMINAMATH_CALUDE_vector_collinearity_l1903_190369

theorem vector_collinearity (k : ℝ) : 
  let a : ℝ × ℝ := (1, 0)
  let b : ℝ × ℝ := (0, 1)
  let v1 : ℝ × ℝ := (2 * a.1 - 3 * b.1, 2 * a.2 - 3 * b.2)
  let v2 : ℝ × ℝ := (k * a.1 + 6 * b.1, k * a.2 + 6 * b.2)
  (∃ (t : ℝ), v1 = (t * v2.1, t * v2.2)) → k = -4 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1903_190369


namespace NUMINAMATH_CALUDE_first_class_students_l1903_190372

/-- Represents the number of students in the first class -/
def x : ℕ := sorry

/-- The average mark of the first class -/
def avg_first : ℝ := 40

/-- The number of students in the second class -/
def students_second : ℕ := 50

/-- The average mark of the second class -/
def avg_second : ℝ := 70

/-- The average mark of all students combined -/
def avg_total : ℝ := 58.75

/-- Theorem stating that the number of students in the first class is 30 -/
theorem first_class_students : 
  (x * avg_first + students_second * avg_second) / (x + students_second) = avg_total → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_first_class_students_l1903_190372


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1903_190346

theorem complex_fraction_simplification : 
  let i : ℂ := Complex.I
  (1 - i) / (1 + i) = -i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1903_190346


namespace NUMINAMATH_CALUDE_min_marked_cells_7x7_l1903_190349

/-- Represents a grid with dimensions (2n-1) x (2n-1) -/
def Grid (n : ℕ) := Fin (2*n - 1) → Fin (2*n - 1) → Bool

/-- Checks if a 1 x 4 strip contains a marked cell -/
def stripContainsMarked (g : Grid 4) (start_row start_col : Fin 7) (isHorizontal : Bool) : Prop :=
  ∃ i : Fin 4, g (if isHorizontal then start_row else start_row + i) 
               (if isHorizontal then start_col + i else start_col) = true

/-- A valid marking satisfies the strip condition for all strips -/
def isValidMarking (g : Grid 4) : Prop :=
  ∀ row col : Fin 7, ∀ isHorizontal : Bool, 
    stripContainsMarked g row col isHorizontal

/-- Counts the number of marked cells in a grid -/
def countMarked (g : Grid 4) : ℕ :=
  (Finset.univ.filter (λ x : Fin 7 × Fin 7 => g x.1 x.2)).card

/-- Main theorem: The minimum number of marked cells in a valid 7x7 grid marking is 12 -/
theorem min_marked_cells_7x7 :
  (∃ g : Grid 4, isValidMarking g ∧ countMarked g = 12) ∧
  (∀ g : Grid 4, isValidMarking g → countMarked g ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_min_marked_cells_7x7_l1903_190349


namespace NUMINAMATH_CALUDE_factor_expression_l1903_190307

theorem factor_expression (x : ℝ) : 63 * x^2 + 54 = 9 * (7 * x^2 + 6) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1903_190307


namespace NUMINAMATH_CALUDE_quadratic_product_zero_l1903_190359

/-- Given a quadratic polynomial f(x) = ax^2 + bx + c, 
    if f((a - b - c)/(2a)) = 0 and f((c - a - b)/(2a)) = 0, 
    then f(-1) * f(1) = 0 -/
theorem quadratic_product_zero 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : f ((a - b - c) / (2 * a)) = 0)
  (h3 : f ((c - a - b) / (2 * a)) = 0)
  : f (-1) * f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_product_zero_l1903_190359


namespace NUMINAMATH_CALUDE_quadratic_root_expression_l1903_190342

theorem quadratic_root_expression (x : ℝ) : 
  x > 0 ∧ x^2 - 10*x - 10 = 0 → 1/20 * x^4 - 6*x^2 - 45 = -50 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_expression_l1903_190342


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l1903_190311

theorem cubic_equation_solutions :
  ∀ x y z : ℤ,
  (x + y + z = 2 ∧ x^3 + y^3 + z^3 = -10) ↔
  ((x, y, z) = (3, 3, -4) ∨ (x, y, z) = (3, -4, 3) ∨ (x, y, z) = (-4, 3, 3)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l1903_190311


namespace NUMINAMATH_CALUDE_solution_5tuples_l1903_190390

theorem solution_5tuples :
  {t : ℕ × ℕ × ℕ × ℕ × ℕ | 
    let (a, b, c, d, n) := t
    (a + b + c + d = 100) ∧
    (n > 0) ∧
    (a + n = b - n) ∧
    (b - n = c * n) ∧
    (c * n = d / n)} =
  {(24, 26, 25, 25, 1), (12, 20, 4, 64, 4), (0, 18, 1, 81, 9)} :=
by sorry

end NUMINAMATH_CALUDE_solution_5tuples_l1903_190390


namespace NUMINAMATH_CALUDE_second_number_value_l1903_190309

theorem second_number_value (x y z : ℝ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 4 / 7) :
  y = 240 / 7 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l1903_190309


namespace NUMINAMATH_CALUDE_book_pages_count_l1903_190330

/-- Calculates the total number of pages in a book given the number of pages read, left to read, and skipped. -/
def totalPages (pagesRead : ℕ) (pagesLeft : ℕ) (pagesSkipped : ℕ) : ℕ :=
  pagesRead + pagesLeft + pagesSkipped

/-- Proves that for the given numbers of pages read, left to read, and skipped, the total number of pages in the book is 372. -/
theorem book_pages_count : totalPages 125 231 16 = 372 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l1903_190330


namespace NUMINAMATH_CALUDE_record_listening_time_l1903_190360

/-- The number of days required to listen to a record collection --/
def days_to_listen (initial_records : ℕ) (gift_records : ℕ) (purchased_records : ℕ) (days_per_record : ℕ) : ℕ :=
  (initial_records + gift_records + purchased_records) * days_per_record

/-- Theorem: Given the initial conditions, it takes 100 days to listen to the entire record collection --/
theorem record_listening_time : days_to_listen 8 12 30 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_record_listening_time_l1903_190360


namespace NUMINAMATH_CALUDE_square_six_z_minus_five_l1903_190381

theorem square_six_z_minus_five (z : ℝ) (hz : 3 * z^2 + 2 * z = 5 * z + 11) : 
  (6 * z - 5)^2 = 141 := by
  sorry

end NUMINAMATH_CALUDE_square_six_z_minus_five_l1903_190381


namespace NUMINAMATH_CALUDE_sequence_100th_term_l1903_190306

theorem sequence_100th_term (a : ℕ → ℕ) (h1 : a 1 = 2) 
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) : 
  a 100 = 9902 := by
  sorry

end NUMINAMATH_CALUDE_sequence_100th_term_l1903_190306


namespace NUMINAMATH_CALUDE_grid_paths_6x5_l1903_190305

/-- The number of paths from (0,0) to (m,n) on a grid, moving only right and up -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The dimensions of our grid -/
def gridWidth : ℕ := 6
def gridHeight : ℕ := 5

theorem grid_paths_6x5 : 
  gridPaths gridWidth gridHeight = 462 := by sorry

end NUMINAMATH_CALUDE_grid_paths_6x5_l1903_190305


namespace NUMINAMATH_CALUDE_ab_value_l1903_190350

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 39) : a * b = 18 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1903_190350


namespace NUMINAMATH_CALUDE_count_two_digit_primes_ending_in_3_l1903_190384

def two_digit_primes_ending_in_3 : List Nat := [13, 23, 33, 43, 53, 63, 73, 83, 93]

theorem count_two_digit_primes_ending_in_3 : 
  (two_digit_primes_ending_in_3.filter Nat.Prime).length = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_two_digit_primes_ending_in_3_l1903_190384


namespace NUMINAMATH_CALUDE_min_distance_ab_value_l1903_190327

theorem min_distance_ab_value (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a^2 - a*b + 1 = 0) (hcd : c^2 + d^2 = 1) :
  let f := fun (x y : ℝ) => (a - x)^2 + (b - y)^2
  ∃ (m : ℝ), (∀ x y, c^2 + d^2 = 1 → f x y ≥ m) ∧ 
             (a * b = Real.sqrt 2 / 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_ab_value_l1903_190327


namespace NUMINAMATH_CALUDE_independence_test_relationship_l1903_190354

-- Define the random variable K²
def K_squared : ℝ → ℝ := sorry

-- Define the probability of judging variables as related
def prob_related : ℝ → ℝ := sorry

-- Define the test of independence
def test_of_independence : (ℝ → ℝ) → (ℝ → ℝ) → Prop := sorry

-- Theorem statement
theorem independence_test_relationship :
  ∀ (x y : ℝ), x > y →
  test_of_independence K_squared prob_related →
  prob_related (K_squared x) < prob_related (K_squared y) :=
sorry

end NUMINAMATH_CALUDE_independence_test_relationship_l1903_190354


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1903_190378

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 2 / x) ↔ x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1903_190378


namespace NUMINAMATH_CALUDE_contrapositive_example_l1903_190338

theorem contrapositive_example (a b : ℝ) : 
  (∀ a b, a = 0 → a * b = 0) ↔ (∀ a b, a * b ≠ 0 → a ≠ 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l1903_190338


namespace NUMINAMATH_CALUDE_doll_ratio_l1903_190302

/-- The ratio of Dina's dolls to Ivy's dolls is 2:1 -/
theorem doll_ratio : 
  ∀ (ivy_dolls : ℕ) (dina_dolls : ℕ),
  (2 : ℚ) / 3 * ivy_dolls = 20 →
  dina_dolls = 60 →
  (dina_dolls : ℚ) / ivy_dolls = 2 := by
  sorry

end NUMINAMATH_CALUDE_doll_ratio_l1903_190302


namespace NUMINAMATH_CALUDE_log_23_between_consecutive_integers_sum_l1903_190376

theorem log_23_between_consecutive_integers_sum : ∃ (c d : ℤ), 
  c + 1 = d ∧ 
  (c : ℝ) < Real.log 23 / Real.log 10 ∧ 
  Real.log 23 / Real.log 10 < (d : ℝ) ∧ 
  c + d = 3 := by sorry

end NUMINAMATH_CALUDE_log_23_between_consecutive_integers_sum_l1903_190376


namespace NUMINAMATH_CALUDE_long_distance_call_cost_per_minute_l1903_190375

/-- Calculates the cost per minute of a long distance call given the initial card value,
    call duration, and remaining credit. -/
def cost_per_minute (initial_value : ℚ) (call_duration : ℚ) (remaining_credit : ℚ) : ℚ :=
  (initial_value - remaining_credit) / call_duration

/-- Proves that the cost per minute for long distance calls is $0.16 given the specified conditions. -/
theorem long_distance_call_cost_per_minute :
  let initial_value : ℚ := 30
  let call_duration : ℚ := 22
  let remaining_credit : ℚ := 26.48
  cost_per_minute initial_value call_duration remaining_credit = 0.16 := by
  sorry

#eval cost_per_minute 30 22 26.48

end NUMINAMATH_CALUDE_long_distance_call_cost_per_minute_l1903_190375


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1903_190328

/-- Prove that the given expression evaluates to 1/5 -/
theorem complex_fraction_evaluation :
  (⌈(19 / 6 : ℚ) - ⌈(34 / 21 : ℚ)⌉⌉ : ℚ) / (⌈(34 / 6 : ℚ) + ⌈(6 * 19 / 34 : ℚ)⌉⌉ : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1903_190328


namespace NUMINAMATH_CALUDE_dog_grouping_theorem_l1903_190313

/-- The number of ways to divide 12 dogs into specified groups -/
def dog_grouping_count : ℕ := sorry

/-- Total number of dogs -/
def total_dogs : ℕ := 12

/-- Size of the first group (including Fluffy) -/
def group1_size : ℕ := 3

/-- Size of the second group (including Nipper) -/
def group2_size : ℕ := 5

/-- Size of the third group (including Spot) -/
def group3_size : ℕ := 4

/-- Theorem stating the correct number of ways to group the dogs -/
theorem dog_grouping_theorem : 
  dog_grouping_count = 20160 ∧
  total_dogs = group1_size + group2_size + group3_size ∧
  group1_size = 3 ∧
  group2_size = 5 ∧
  group3_size = 4 := by sorry

end NUMINAMATH_CALUDE_dog_grouping_theorem_l1903_190313


namespace NUMINAMATH_CALUDE_sum_of_complex_roots_l1903_190352

theorem sum_of_complex_roots (a₁ a₂ a₃ : ℂ)
  (h1 : a₁^2 + a₂^2 + a₃^2 = 0)
  (h2 : a₁^3 + a₂^3 + a₃^3 = 0)
  (h3 : a₁^4 + a₂^4 + a₃^4 = 0) :
  a₁ + a₂ + a₃ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_complex_roots_l1903_190352


namespace NUMINAMATH_CALUDE_min_production_to_meet_demand_l1903_190319

/-- Total market demand function -/
def f (x : ℕ) : ℕ := x * (x + 1) * (35 - 2 * x)

/-- Monthly demand function -/
def g (x : ℕ) : ℤ := f x - f (x - 1)

/-- The range of valid month numbers -/
def valid_months : Set ℕ := {x | 1 ≤ x ∧ x ≤ 12}

theorem min_production_to_meet_demand :
  ∃ (a : ℕ), (∀ x ∈ valid_months, (g x : ℝ) ≤ a) ∧
  (∀ b : ℕ, (∀ x ∈ valid_months, (g x : ℝ) ≤ b) → a ≤ b) ∧
  a = 171 := by
  sorry

end NUMINAMATH_CALUDE_min_production_to_meet_demand_l1903_190319


namespace NUMINAMATH_CALUDE_bird_watching_ratio_l1903_190347

/-- Given the conditions of Camille's bird watching, prove the ratio of robins to cardinals -/
theorem bird_watching_ratio :
  ∀ (cardinals blue_jays sparrows robins : ℕ),
    cardinals = 3 →
    blue_jays = 2 * cardinals →
    sparrows = 3 * cardinals + 1 →
    cardinals + blue_jays + sparrows + robins = 31 →
    robins / cardinals = 4 := by
  sorry

#check bird_watching_ratio

end NUMINAMATH_CALUDE_bird_watching_ratio_l1903_190347


namespace NUMINAMATH_CALUDE_problem_solution_l1903_190310

-- Define a function to check if a number is square-free
def is_square_free (n : ℕ) : Prop :=
  ∀ (p : ℕ), Nat.Prime p → (p * p ∣ n) → p = 1

-- Define the condition for the problem
def satisfies_condition (p : ℕ) : Prop :=
  Nat.Prime p ∧ p ≥ 3 ∧
  ∀ (q : ℕ), Nat.Prime q → q < p →
    is_square_free (p - p / q * q)

-- State the theorem
theorem problem_solution :
  {p : ℕ | satisfies_condition p} = {3, 5, 7, 13} :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1903_190310


namespace NUMINAMATH_CALUDE_triangle_max_area_l1903_190396

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * cos(B) + b * cos(A) = √3 and the area of its circumcircle is π,
    then the maximum area of triangle ABC is 3√3/4. -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.cos B + b * Real.cos A = Real.sqrt 3 →
  (π * (a / (2 * Real.sin A))^2) = π →
  ∃ (S : ℝ), S = (1/2) * a * b * Real.sin C ∧
              S ≤ (3 * Real.sqrt 3) / 4 ∧
              (∀ (S' : ℝ), S' = (1/2) * a * b * Real.sin C → S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1903_190396


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l1903_190333

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (distinct : Line → Line → Prop)
variable (nonCoincident : Plane → Plane → Prop)
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (m : Line) (α β : Plane)
  (h1 : nonCoincident α β)
  (h2 : perpendicular m α)
  (h3 : parallel m β) :
  planePerp α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l1903_190333


namespace NUMINAMATH_CALUDE_new_vessel_capacity_l1903_190395

/-- Given two vessels with different alcohol concentrations, prove the capacity of a new vessel -/
theorem new_vessel_capacity
  (vessel1_capacity : ℝ) (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ) (vessel2_alcohol_percentage : ℝ)
  (total_liquid : ℝ) (new_concentration : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percentage = 0.3)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_alcohol_percentage = 0.45)
  (h5 : total_liquid = 8)
  (h6 : new_concentration = 0.33) :
  (vessel1_capacity * vessel1_alcohol_percentage + vessel2_capacity * vessel2_alcohol_percentage) / new_concentration = 10 := by
sorry

end NUMINAMATH_CALUDE_new_vessel_capacity_l1903_190395


namespace NUMINAMATH_CALUDE_grace_earnings_l1903_190392

theorem grace_earnings (weekly_charge : ℕ) (payment_interval : ℕ) (total_weeks : ℕ) (total_earnings : ℕ) : 
  weekly_charge = 300 →
  payment_interval = 2 →
  total_weeks = 6 →
  total_earnings = 1800 →
  total_weeks * weekly_charge = total_earnings :=
by
  sorry

end NUMINAMATH_CALUDE_grace_earnings_l1903_190392


namespace NUMINAMATH_CALUDE_stair_climbing_time_l1903_190386

theorem stair_climbing_time (n : ℕ) (a d : ℝ) (h : n = 7 ∧ a = 25 ∧ d = 10) :
  (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 385 :=
by sorry

end NUMINAMATH_CALUDE_stair_climbing_time_l1903_190386


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1903_190399

theorem imaginary_part_of_z : Complex.im ((1 - Complex.I) / Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1903_190399


namespace NUMINAMATH_CALUDE_current_trees_count_l1903_190380

/-- The number of dogwood trees currently in the park -/
def current_trees : ℕ := sorry

/-- The number of trees to be planted today -/
def trees_today : ℕ := 3

/-- The number of trees to be planted tomorrow -/
def trees_tomorrow : ℕ := 2

/-- The total number of trees after planting -/
def total_trees : ℕ := 12

/-- Proof that the current number of trees is 7 -/
theorem current_trees_count : current_trees = 7 := by
  sorry

end NUMINAMATH_CALUDE_current_trees_count_l1903_190380


namespace NUMINAMATH_CALUDE_f_bounded_implies_k_eq_three_l1903_190300

/-- The function f(x) = -4x³ + kx --/
def f (k : ℝ) (x : ℝ) : ℝ := -4 * x^3 + k * x

/-- The theorem stating that if f(x) ≤ 1 for all x in [-1, 1], then k = 3 --/
theorem f_bounded_implies_k_eq_three (k : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f k x ≤ 1) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_bounded_implies_k_eq_three_l1903_190300


namespace NUMINAMATH_CALUDE_expression_value_l1903_190393

theorem expression_value :
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 1
  x^2 * y * z - x * y * z^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1903_190393


namespace NUMINAMATH_CALUDE_polynomial_identity_solution_l1903_190382

theorem polynomial_identity_solution :
  ∀ (a b c : ℝ),
    (∀ x : ℝ, x^3 - a*x^2 + b*x - c = (x-a)*(x-b)*(x-c))
    ↔ 
    (a = -1 ∧ b = -1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identity_solution_l1903_190382


namespace NUMINAMATH_CALUDE_find_number_l1903_190377

theorem find_number (x y N : ℝ) (h1 : x / (2 * y) = N / 2) (h2 : (7 * x + 2 * y) / (x - 2 * y) = 23) :
  N = 3 := by
sorry

end NUMINAMATH_CALUDE_find_number_l1903_190377


namespace NUMINAMATH_CALUDE_x_plus_y_value_l1903_190366

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 2010)
  (eq2 : x + 2010 * Real.sin y = 2009)
  (y_range : π / 2 ≤ y ∧ y ≤ π) :
  x + y = 2011 + π := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l1903_190366


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1903_190370

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0) ↔ (a < 0 ∨ a ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1903_190370


namespace NUMINAMATH_CALUDE_greatest_k_inequality_l1903_190343

theorem greatest_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 > b*c) :
  (a^2 - b*c)^2 ≥ 4*(b^2 - c*a)*(c^2 - a*b) := by
  sorry

end NUMINAMATH_CALUDE_greatest_k_inequality_l1903_190343


namespace NUMINAMATH_CALUDE_largest_base7_3digit_in_decimal_l1903_190326

/-- The largest three-digit number in base 7 -/
def largest_base7_3digit : ℕ := 6 * 7^2 + 6 * 7^1 + 6 * 7^0

/-- Converts a base 7 number to decimal -/
def base7_to_decimal (n : ℕ) : ℕ := n

theorem largest_base7_3digit_in_decimal :
  base7_to_decimal largest_base7_3digit = 342 := by sorry

end NUMINAMATH_CALUDE_largest_base7_3digit_in_decimal_l1903_190326


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1903_190345

-- Define vectors a and b
def a : ℝ × ℝ := (-3, 4)
def b : ℝ × ℝ := (2, -1)

-- Define the perpendicularity condition
def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Theorem statement
theorem perpendicular_vectors_x_value :
  ∃ x : ℝ, is_perpendicular (a.1 - x * b.1, a.2 - x * b.2) (a.1 - b.1, a.2 - b.2) ∧ x = -7/3 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1903_190345


namespace NUMINAMATH_CALUDE_angle_at_seven_l1903_190301

/-- The number of parts the clock face is divided into -/
def clock_parts : ℕ := 12

/-- The angle of each part of the clock face in degrees -/
def part_angle : ℝ := 30

/-- The time in hours -/
def time : ℝ := 7

/-- The angle between the hour hand and the minute hand at a given time -/
def angle_between (t : ℝ) : ℝ := sorry

theorem angle_at_seven : angle_between time = 150 := by sorry

end NUMINAMATH_CALUDE_angle_at_seven_l1903_190301


namespace NUMINAMATH_CALUDE_pencils_remaining_l1903_190371

/-- The number of pencils left in a box after some are taken -/
def pencils_left (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Theorem: Given 79 initial pencils and 4 taken, 75 pencils are left -/
theorem pencils_remaining : pencils_left 79 4 = 75 := by
  sorry

end NUMINAMATH_CALUDE_pencils_remaining_l1903_190371


namespace NUMINAMATH_CALUDE_opposite_direction_time_calculation_l1903_190318

/-- Given two people moving in opposite directions from the same starting point,
    calculate the time taken to reach a specific distance between them. -/
theorem opposite_direction_time_calculation 
  (speed1 : ℝ) (speed2 : ℝ) (distance : ℝ) 
  (h1 : speed1 = 2) 
  (h2 : speed2 = 3) 
  (h3 : distance = 20) : 
  distance / (speed1 + speed2) = 4 := by
  sorry

#check opposite_direction_time_calculation

end NUMINAMATH_CALUDE_opposite_direction_time_calculation_l1903_190318


namespace NUMINAMATH_CALUDE_negative_expression_l1903_190344

/-- Given real numbers U, V, W, X, and Y with the following properties:
    U and W are negative,
    V and Y are positive,
    X is near zero (small in absolute value),
    prove that U - V is negative. -/
theorem negative_expression (U V W X Y : ℝ) 
  (hU : U < 0) (hW : W < 0) 
  (hV : V > 0) (hY : Y > 0) 
  (hX : ∃ ε > 0, abs X < ε ∧ ε < 1) : 
  U - V < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_expression_l1903_190344


namespace NUMINAMATH_CALUDE_cells_covered_by_two_squares_l1903_190315

/-- Represents a square on a graph paper --/
structure Square where
  size : ℕ
  position : ℕ × ℕ

/-- Represents the configuration of squares on the graph paper --/
def SquareConfiguration := List Square

/-- Counts the number of cells covered by exactly two squares in a given configuration --/
def countCellsCoveredByTwoSquares (config : SquareConfiguration) : ℕ :=
  sorry

/-- The specific configuration of squares from the problem --/
def problemConfiguration : SquareConfiguration :=
  [{ size := 5, position := (0, 0) },
   { size := 5, position := (3, 0) },
   { size := 5, position := (3, 3) }]

theorem cells_covered_by_two_squares :
  countCellsCoveredByTwoSquares problemConfiguration = 13 := by
  sorry

end NUMINAMATH_CALUDE_cells_covered_by_two_squares_l1903_190315


namespace NUMINAMATH_CALUDE_fourth_fifth_sum_l1903_190348

/-- An arithmetic sequence with given properties -/
def arithmeticSequence (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ a 3 = 17 ∧ a 6 = 32 ∧ ∀ n, a (n + 1) - a n = a 2 - a 1

theorem fourth_fifth_sum (a : ℕ → ℕ) (h : arithmeticSequence a) : a 4 + a 5 = 55 := by
  sorry

end NUMINAMATH_CALUDE_fourth_fifth_sum_l1903_190348


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l1903_190325

-- Define the quadratic equation
def quadratic_equation (s t x : ℝ) : Prop :=
  s * x^2 + t * x + s - 1 = 0

-- Define the existence of a real root
def has_real_root (s t : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation s t x

-- Main theorem
theorem quadratic_real_root_condition (s : ℝ) :
  (s ≠ 0 ∧ ∀ t : ℝ, has_real_root s t) ↔ (0 < s ∧ s ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l1903_190325


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l1903_190314

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 10) :
  (⌊y⌋ - ⌈x⌉ : ℤ) ≤ 5 ∧ ∃ (x' y' : ℝ), 3 < x' ∧ x' < 6 ∧ 6 < y' ∧ y' < 10 ∧ ⌊y'⌋ - ⌈x'⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l1903_190314


namespace NUMINAMATH_CALUDE_farm_horses_and_cows_l1903_190355

theorem farm_horses_and_cows (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 3 * initial_cows →
  (initial_horses - 15) * 3 = 5 * (initial_cows + 15) →
  initial_horses - 15 - (initial_cows + 15) = 30 := by
  sorry

end NUMINAMATH_CALUDE_farm_horses_and_cows_l1903_190355


namespace NUMINAMATH_CALUDE_total_passengers_four_trips_l1903_190334

/-- Calculates the total number of passengers transported in multiple round trips -/
def total_passengers (passengers_one_way : ℕ) (passengers_return : ℕ) (num_round_trips : ℕ) : ℕ :=
  (passengers_one_way + passengers_return) * num_round_trips

/-- Theorem stating that the total number of passengers transported in 4 round trips is 640 -/
theorem total_passengers_four_trips :
  total_passengers 100 60 4 = 640 := by
  sorry

end NUMINAMATH_CALUDE_total_passengers_four_trips_l1903_190334


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1903_190356

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 8*x + 1 = 0 ↔ (x - 4)^2 = 15 := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1903_190356
