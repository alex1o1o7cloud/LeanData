import Mathlib

namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l369_36953

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l369_36953


namespace NUMINAMATH_CALUDE_no_four_integers_product_square_l369_36955

theorem no_four_integers_product_square : ¬∃ (a b c d : ℕ+), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  (∃ (m : ℕ), (a * b + 2006 : ℕ) = m^2) ∧
  (∃ (n : ℕ), (a * c + 2006 : ℕ) = n^2) ∧
  (∃ (p : ℕ), (a * d + 2006 : ℕ) = p^2) ∧
  (∃ (q : ℕ), (b * c + 2006 : ℕ) = q^2) ∧
  (∃ (r : ℕ), (b * d + 2006 : ℕ) = r^2) ∧
  (∃ (s : ℕ), (c * d + 2006 : ℕ) = s^2) :=
by
  sorry

end NUMINAMATH_CALUDE_no_four_integers_product_square_l369_36955


namespace NUMINAMATH_CALUDE_joseph_total_distance_l369_36961

/-- Joseph's daily running distance in meters -/
def daily_distance : ℕ := 900

/-- Number of days Joseph ran -/
def days_run : ℕ := 3

/-- Total distance Joseph ran -/
def total_distance : ℕ := daily_distance * days_run

/-- Theorem: Joseph's total running distance is 2700 meters -/
theorem joseph_total_distance : total_distance = 2700 := by
  sorry

end NUMINAMATH_CALUDE_joseph_total_distance_l369_36961


namespace NUMINAMATH_CALUDE_liam_fourth_week_l369_36975

/-- A sequence of four numbers representing chapters read each week -/
def ChapterSequence := Fin 4 → ℕ

/-- The properties of Liam's reading sequence -/
def IsLiamSequence (s : ChapterSequence) : Prop :=
  (∀ i : Fin 3, s (i + 1) = s i + 3) ∧
  (s 0 + s 1 + s 2 + s 3 = 50)

/-- Theorem stating that the fourth number in Liam's sequence is 17 -/
theorem liam_fourth_week (s : ChapterSequence) (h : IsLiamSequence s) : s 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_liam_fourth_week_l369_36975


namespace NUMINAMATH_CALUDE_signboard_white_area_l369_36994

/-- Represents the dimensions of a letter stroke -/
structure StrokeDimensions where
  width : ℝ
  height : ℝ

/-- Represents a letter on the signboard -/
inductive Letter
| L
| A
| S
| T

/-- Calculates the area of a letter based on its strokes -/
def letterArea (letter : Letter) : ℝ :=
  match letter with
  | Letter.L => 9
  | Letter.A => 7.5
  | Letter.S => 13
  | Letter.T => 9

/-- Represents the signboard -/
structure Signboard where
  width : ℝ
  height : ℝ
  word : List Letter
  strokeWidth : ℝ

def signboard : Signboard :=
  { width := 6
  , height := 18
  , word := [Letter.L, Letter.A, Letter.S, Letter.T]
  , strokeWidth := 1 }

/-- Calculates the total area of the signboard -/
def totalArea (s : Signboard) : ℝ :=
  s.width * s.height

/-- Calculates the area covered by the letters -/
def coveredArea (s : Signboard) : ℝ :=
  s.word.map letterArea |> List.sum

/-- Calculates the white area remaining on the signboard -/
def whiteArea (s : Signboard) : ℝ :=
  totalArea s - coveredArea s

/-- Theorem stating that the white area of the given signboard is 69.5 square units -/
theorem signboard_white_area :
  whiteArea signboard = 69.5 := by
  sorry

end NUMINAMATH_CALUDE_signboard_white_area_l369_36994


namespace NUMINAMATH_CALUDE_min_value_on_circle_l369_36995

theorem min_value_on_circle (x y : ℝ) (h : (x - 2)^2 + (y - 2)^2 = 1) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a - 2)^2 + (b - 2)^2 = 1 → a^2 + b^2 ≥ m) ∧
  (m = 9 - 4 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l369_36995


namespace NUMINAMATH_CALUDE_coefficient_x7y2_is_20_l369_36947

/-- The coefficient of x^7y^2 in the expansion of (x-y)(x+y)^8 -/
def coefficient_x7y2 : ℕ :=
  (Nat.choose 8 2) - (Nat.choose 8 1)

/-- Theorem: The coefficient of x^7y^2 in the expansion of (x-y)(x+y)^8 is 20 -/
theorem coefficient_x7y2_is_20 : coefficient_x7y2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x7y2_is_20_l369_36947


namespace NUMINAMATH_CALUDE_geometric_series_sum_l369_36913

/-- The limiting sum of a geometric series with first term 6 and common ratio -2/5 is 30/7 -/
theorem geometric_series_sum : 
  let a : ℚ := 6
  let r : ℚ := -2/5
  let s : ℚ := a / (1 - r)
  s = 30/7 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l369_36913


namespace NUMINAMATH_CALUDE_max_ab_given_extremum_l369_36925

/-- Given positive real numbers a and b, and a function f with an extremum at x = 1,
    the maximum value of ab is 9. -/
theorem max_ab_given_extremum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := fun x => 4 * x^3 - a * x^2 - 2 * b * x + 2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  (a * b ≤ 9) ∧ (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = 9 ∧
    let f₀ := fun x => 4 * x^3 - a₀ * x^2 - 2 * b₀ * x + 2
    ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f₀ x ≤ f₀ 1 ∨ f₀ x ≥ f₀ 1) :=
by sorry


end NUMINAMATH_CALUDE_max_ab_given_extremum_l369_36925


namespace NUMINAMATH_CALUDE_lcm_count_l369_36936

theorem lcm_count : 
  ∃! (n : ℕ), n > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ k ∈ S, k > 0 ∧ Nat.lcm (9^9) (Nat.lcm (12^12) k) = 18^18) ∧
    (∀ k ∉ S, k > 0 → Nat.lcm (9^9) (Nat.lcm (12^12) k) ≠ 18^18)) :=
sorry

end NUMINAMATH_CALUDE_lcm_count_l369_36936


namespace NUMINAMATH_CALUDE_equation_solution_l369_36902

theorem equation_solution (x : ℝ) : x > 0 → (5 * x^(1/4) - 3 * (x / x^(3/4)) = 10 + x^(1/4)) ↔ x = 10000 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l369_36902


namespace NUMINAMATH_CALUDE_trapezoid_with_equal_angles_l369_36968

-- Define a trapezoid
structure Trapezoid :=
  (is_quadrilateral : Bool)
  (has_parallel_sides : Bool)
  (has_nonparallel_sides : Bool)

-- Define properties of a trapezoid
def Trapezoid.is_isosceles (t : Trapezoid) : Prop := sorry
def Trapezoid.is_right_angled (t : Trapezoid) : Prop := sorry
def Trapezoid.has_two_equal_angles (t : Trapezoid) : Prop := sorry

-- Theorem statement
theorem trapezoid_with_equal_angles 
  (t : Trapezoid) 
  (h1 : t.is_quadrilateral = true) 
  (h2 : t.has_parallel_sides = true) 
  (h3 : t.has_nonparallel_sides = true) 
  (h4 : t.has_two_equal_angles) : 
  t.is_isosceles ∨ t.is_right_angled := sorry

end NUMINAMATH_CALUDE_trapezoid_with_equal_angles_l369_36968


namespace NUMINAMATH_CALUDE_gum_cost_1000_l369_36945

/-- The cost of buying a given number of pieces of gum, considering bulk discount --/
def gumCost (pieces : ℕ) : ℚ :=
  let baseCost := 2 * pieces
  let discountedCost := if pieces > 500 then baseCost * (9/10) else baseCost
  discountedCost / 100

theorem gum_cost_1000 :
  gumCost 1000 = 18 := by sorry

end NUMINAMATH_CALUDE_gum_cost_1000_l369_36945


namespace NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l369_36948

theorem quadratic_roots_real_and_equal 
  (a k : ℝ) 
  (ha : a > 0) 
  (hk : k > 0) 
  (h_discriminant : (6 * Real.sqrt k) ^ 2 - 4 * a * (18 * k) = 0) : 
  ∃ x : ℝ, ∀ y : ℝ, a * y ^ 2 - 6 * y * Real.sqrt k + 18 * k = 0 ↔ y = x :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l369_36948


namespace NUMINAMATH_CALUDE_expand_and_simplify_expression_l369_36923

theorem expand_and_simplify_expression (x : ℝ) : 
  6 * (x - 7) * (2 * x + 15) + (3 * x - 4) * (x + 5) = 15 * x^2 + 17 * x - 650 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_expression_l369_36923


namespace NUMINAMATH_CALUDE_viju_aju_age_ratio_l369_36978

/-- Given that Viju's age 5 years ago was 16 and that four years from now, 
    the ratio of ages of Viju to Aju will be 5:2, 
    prove that the present age ratio of Viju to Aju is 7:2. -/
theorem viju_aju_age_ratio :
  ∀ (viju_age aju_age : ℕ),
    viju_age - 5 = 16 →
    (viju_age + 4) * 2 = (aju_age + 4) * 5 →
    ∃ (k : ℕ), k > 0 ∧ viju_age = 7 * k ∧ aju_age = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_viju_aju_age_ratio_l369_36978


namespace NUMINAMATH_CALUDE_min_teams_in_tournament_l369_36982

/-- Represents a football team in the tournament -/
structure Team where
  wins : Nat
  draws : Nat
  losses : Nat

/-- Calculates the score of a team -/
def score (t : Team) : Nat := 3 * t.wins + t.draws

/-- Represents a football tournament -/
structure Tournament where
  teams : List Team
  /-- Each team plays against every other team once -/
  matches_played : ∀ t ∈ teams, t.wins + t.draws + t.losses = teams.length - 1
  /-- There exists a team with the highest score -/
  highest_scorer : ∃ t ∈ teams, ∀ t' ∈ teams, t ≠ t' → score t > score t'
  /-- The highest scoring team has the fewest wins -/
  fewest_wins : ∃ t ∈ teams, (∀ t' ∈ teams, score t ≥ score t') ∧ 
                              (∀ t' ∈ teams, t ≠ t' → t.wins < t'.wins)

/-- The minimum number of teams in a valid tournament is 8 -/
theorem min_teams_in_tournament : 
  ∀ t : Tournament, t.teams.length ≥ 8 ∧ 
  (∃ t' : Tournament, t'.teams.length = 8) := by sorry

end NUMINAMATH_CALUDE_min_teams_in_tournament_l369_36982


namespace NUMINAMATH_CALUDE_final_statue_weight_approx_l369_36919

/-- The weight of the final statue given the initial weights and removal percentages --/
def final_statue_weight (initial_marble : ℝ) (initial_granite : ℝ) 
  (marble_removal1 : ℝ) (marble_removal2 : ℝ) (marble_removal3 : ℝ) 
  (granite_removal1 : ℝ) (granite_removal2 : ℝ) 
  (marble_removal_final : ℝ) (granite_removal_final : ℝ) : ℝ :=
  let remaining_marble1 := initial_marble * (1 - marble_removal1)
  let remaining_marble2 := remaining_marble1 * (1 - marble_removal2)
  let remaining_marble3 := remaining_marble2 * (1 - marble_removal3)
  let final_marble := remaining_marble3 * (1 - marble_removal_final)
  
  let remaining_granite1 := initial_granite * (1 - granite_removal1)
  let remaining_granite2 := remaining_granite1 * (1 - granite_removal2)
  let final_granite := remaining_granite2 * (1 - granite_removal_final)
  
  final_marble + final_granite

/-- The final weight of the statue is approximately 119.0826 kg --/
theorem final_statue_weight_approx :
  ∃ ε > 0, ε < 0.0001 ∧ 
  |final_statue_weight 225 65 0.32 0.22 0.15 0.40 0.25 0.10 0.05 - 119.0826| < ε :=
sorry

end NUMINAMATH_CALUDE_final_statue_weight_approx_l369_36919


namespace NUMINAMATH_CALUDE_church_attendance_l369_36909

theorem church_attendance (total people : ℕ) (children : ℕ) (female_adults : ℕ) :
  total = 200 →
  children = 80 →
  female_adults = 60 →
  total = children + female_adults + (total - children - female_adults) →
  total - children - female_adults = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_church_attendance_l369_36909


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l369_36930

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x - 3
  ∃ x1 x2 : ℝ, x1 = 3 + 2*Real.sqrt 3 ∧ 
             x2 = 3 - 2*Real.sqrt 3 ∧ 
             f x1 = 0 ∧ f x2 = 0 ∧
             ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l369_36930


namespace NUMINAMATH_CALUDE_extracurricular_materials_choice_l369_36910

/-- The number of ways to choose r items from n items -/
def choose (n : ℕ) (r : ℕ) : ℕ := Nat.choose n r

/-- The number of ways to arrange r items from n items -/
def arrange (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The total number of extracurricular reading materials -/
def totalMaterials : ℕ := 6

/-- The number of materials each student chooses -/
def materialsPerStudent : ℕ := 2

/-- The number of common materials between students -/
def commonMaterials : ℕ := 1

theorem extracurricular_materials_choice :
  (choose totalMaterials commonMaterials) *
  (arrange (totalMaterials - commonMaterials) (materialsPerStudent - commonMaterials)) = 120 := by
  sorry


end NUMINAMATH_CALUDE_extracurricular_materials_choice_l369_36910


namespace NUMINAMATH_CALUDE_circles_relation_l369_36954

theorem circles_relation (a b c : ℝ) :
  (∃ x : ℝ, x^2 - 2*a*x + b^2 = c*(b - a) ∧ 
   ∀ y : ℝ, y^2 - 2*a*y + b^2 = c*(b - a) → y = x) →
  (a = b ∨ c = a + b) :=
by sorry

end NUMINAMATH_CALUDE_circles_relation_l369_36954


namespace NUMINAMATH_CALUDE_cat_mouse_position_after_299_moves_l369_36917

/-- Represents the four rooms for the cat --/
inductive CatRoom
| TopLeft
| TopRight
| BottomRight
| BottomLeft

/-- Represents the eight segments for the mouse --/
inductive MouseSegment
| TopLeft
| TopMiddle
| TopRight
| RightMiddle
| BottomRight
| BottomMiddle
| BottomLeft
| LeftMiddle

/-- Function to determine cat's position after n moves --/
def catPosition (n : ℕ) : CatRoom :=
  match (n - n / 100) % 4 with
  | 0 => CatRoom.TopLeft
  | 1 => CatRoom.TopRight
  | 2 => CatRoom.BottomRight
  | _ => CatRoom.BottomLeft

/-- Function to determine mouse's position after n moves --/
def mousePosition (n : ℕ) : MouseSegment :=
  match n % 8 with
  | 0 => MouseSegment.TopLeft
  | 1 => MouseSegment.TopMiddle
  | 2 => MouseSegment.TopRight
  | 3 => MouseSegment.RightMiddle
  | 4 => MouseSegment.BottomRight
  | 5 => MouseSegment.BottomMiddle
  | 6 => MouseSegment.BottomLeft
  | _ => MouseSegment.LeftMiddle

theorem cat_mouse_position_after_299_moves :
  catPosition 299 = CatRoom.TopLeft ∧
  mousePosition 299 = MouseSegment.RightMiddle :=
by sorry

end NUMINAMATH_CALUDE_cat_mouse_position_after_299_moves_l369_36917


namespace NUMINAMATH_CALUDE_balanced_sequence_equality_l369_36906

/-- A sequence of five real numbers is balanced if, when any one number is removed, 
    the remaining four can be divided into two groups of two numbers each 
    such that the sum of one group equals the sum of the other group. -/
def IsBalanced (a b c d e : ℝ) : Prop :=
  (b + c = d + e) ∧ (a + c = d + e) ∧ (a + b = d + e) ∧
  (a + c = b + e) ∧ (a + d = b + e)

theorem balanced_sequence_equality (a b c d e : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h_balanced : IsBalanced a b c d e)
  (h_sum1 : e + c = b + d)
  (h_sum2 : e + a = c + d) :
  a = b ∧ b = c ∧ c = d ∧ d = e := by
  sorry

end NUMINAMATH_CALUDE_balanced_sequence_equality_l369_36906


namespace NUMINAMATH_CALUDE_circle_equation_l369_36951

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def satisfiesConditions (c : Circle) : Prop :=
  let (a, b) := c.center
  -- Condition 1: chord on y-axis has length 2
  c.radius^2 = a^2 + 1 ∧
  -- Condition 2: ratio of arc lengths divided by x-axis is 3:1
  (c.radius^2 = 2 * b^2) ∧
  -- Condition 3: distance from center to line x - 2y = 0 is √5/5
  |a - 2*b| / Real.sqrt 5 = Real.sqrt 5 / 5

-- Theorem statement
theorem circle_equation (c : Circle) :
  satisfiesConditions c →
  ((∃ x y, (x + 1)^2 + (y + 1)^2 = 2) ∨ (∃ x y, (x - 1)^2 + (y - 1)^2 = 2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l369_36951


namespace NUMINAMATH_CALUDE_xiao_ming_brother_age_l369_36908

def is_multiple_of_19 (year : ℕ) : Prop := ∃ k : ℕ, year = 19 * k

def has_repeated_digits (year : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ (∃ i j : ℕ, i ≠ j ∧ (year / 10^i) % 10 = d ∧ (year / 10^j) % 10 = d)

def first_non_repeating_year (birth_year : ℕ) (target_year : ℕ) : Prop :=
  ¬(has_repeated_digits target_year) ∧
  ∀ y : ℕ, birth_year ≤ y ∧ y < target_year → has_repeated_digits y

theorem xiao_ming_brother_age :
  ∀ birth_year : ℕ,
    is_multiple_of_19 birth_year →
    first_non_repeating_year birth_year 2013 →
    2013 - birth_year = 18 :=
by sorry

end NUMINAMATH_CALUDE_xiao_ming_brother_age_l369_36908


namespace NUMINAMATH_CALUDE_circles_equal_radii_l369_36972

/-- Proves that the radii of circles A, B, and C are equal -/
theorem circles_equal_radii (r_A : ℝ) (d_B : ℝ) (c_C : ℝ) : 
  r_A = 5 → d_B = 10 → c_C = 10 * Real.pi → r_A = d_B / 2 ∧ r_A = c_C / (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_circles_equal_radii_l369_36972


namespace NUMINAMATH_CALUDE_least_integer_proof_l369_36914

/-- The least positive integer divisible by all numbers from 1 to 22 and 25 to 30 -/
def least_integer : ℕ := 1237834741500

/-- The set of divisors from 1 to 30, excluding 23 and 24 -/
def divisors : Set ℕ := {n : ℕ | n ∈ Finset.range 31 ∧ n ≠ 23 ∧ n ≠ 24}

theorem least_integer_proof :
  (∀ n ∈ divisors, least_integer % n = 0) ∧
  (∀ m : ℕ, m < least_integer →
    ∃ k ∈ divisors, m % k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_least_integer_proof_l369_36914


namespace NUMINAMATH_CALUDE_problem_solution_l369_36938

theorem problem_solution (x y z t : ℝ) 
  (eq1 : x = y^2 - 16*x^2)
  (eq2 : y = z^2 - 4*x^2)
  (eq3 : z = t^2 - x^2)
  (eq4 : t = x - 1) :
  x = 1/9 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l369_36938


namespace NUMINAMATH_CALUDE_f_of_5_equals_15_l369_36934

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem f_of_5_equals_15 : f 5 = 15 := by sorry

end NUMINAMATH_CALUDE_f_of_5_equals_15_l369_36934


namespace NUMINAMATH_CALUDE_third_group_size_l369_36963

theorem third_group_size (total : ℕ) (first_fraction : ℚ) (second_fraction : ℚ)
  (h_total : total = 45)
  (h_first : first_fraction = 1 / 3)
  (h_second : second_fraction = 2 / 5)
  : total - (total * first_fraction).floor - (total * second_fraction).floor = 12 :=
by sorry

end NUMINAMATH_CALUDE_third_group_size_l369_36963


namespace NUMINAMATH_CALUDE_cos_seven_pi_fourth_l369_36937

theorem cos_seven_pi_fourth : Real.cos (7 * π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_fourth_l369_36937


namespace NUMINAMATH_CALUDE_chair_rows_theorem_l369_36991

/-- Given a total number of chairs and chairs per row, calculates the number of rows -/
def calculate_rows (total_chairs : ℕ) (chairs_per_row : ℕ) : ℕ :=
  total_chairs / chairs_per_row

/-- Theorem stating that for 432 total chairs and 16 chairs per row, there are 27 rows -/
theorem chair_rows_theorem :
  calculate_rows 432 16 = 27 := by
  sorry

end NUMINAMATH_CALUDE_chair_rows_theorem_l369_36991


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l369_36926

theorem triangle_side_lengths 
  (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = c + 2) 
  (h3 : Real.sin (Real.arcsin (Real.sqrt 3 / 2)) = Real.sqrt 3 / 2) : 
  a = 7 ∧ b = 5 ∧ c = 3 := by
  sorry

#check triangle_side_lengths

end NUMINAMATH_CALUDE_triangle_side_lengths_l369_36926


namespace NUMINAMATH_CALUDE_square_diff_product_l369_36956

theorem square_diff_product (x y : ℝ) (hx : x = Real.sqrt 3 + 1) (hy : y = Real.sqrt 3 - 1) :
  x^2 * y - x * y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_product_l369_36956


namespace NUMINAMATH_CALUDE_trajectory_equation_l369_36911

/-- The equation of the trajectory of point P in the xOy plane, given point A at (0,0,4) and |PA| = 5 -/
theorem trajectory_equation :
  ∀ (x y : ℝ),
  let P : ℝ × ℝ × ℝ := (x, y, 0)
  let A : ℝ × ℝ × ℝ := (0, 0, 4)
  (x^2 + y^2 + (0 - 4)^2 = 5^2) →
  (x^2 + y^2 = 9) :=
by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l369_36911


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l369_36974

/-- Given a bag with white and red balls, prove that the probability of drawing a white ball equals 2/6 -/
theorem probability_of_white_ball (b : ℕ) : 
  let white_balls := b - 4
  let red_balls := b + 46
  let total_balls := white_balls + red_balls
  let prob_white := white_balls / total_balls
  prob_white = 2 / 6 := by
sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l369_36974


namespace NUMINAMATH_CALUDE_complex_subtraction_l369_36966

theorem complex_subtraction (z₁ z₂ : ℂ) (h₁ : z₁ = 3 + I) (h₂ : z₂ = 2 - I) :
  z₁ - z₂ = 1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l369_36966


namespace NUMINAMATH_CALUDE_smallest_positive_difference_l369_36987

/-- Vovochka's method of adding two three-digit numbers -/
def vovochka_sum (a b c d e f : Nat) : Nat :=
  (a + d) * 1000 + (b + e) * 100 + (c + f)

/-- The correct sum of two three-digit numbers -/
def correct_sum (a b c d e f : Nat) : Nat :=
  (a + d) * 100 + (b + e) * 10 + (c + f)

/-- The difference between Vovochka's sum and the correct sum -/
def sum_difference (a b c d e f : Nat) : Int :=
  (vovochka_sum a b c d e f) - (correct_sum a b c d e f)

theorem smallest_positive_difference :
  ∀ a b c d e f : Nat,
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 →
  sum_difference a b c d e f ≠ 0 →
  1800 ≤ |sum_difference a b c d e f| :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_difference_l369_36987


namespace NUMINAMATH_CALUDE_trapezoid_properties_l369_36977

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  side1 : ℝ
  side2 : ℝ
  diagonal_is_bisector : Bool

/-- Properties of the specific trapezoid in the problem -/
def problem_trapezoid : IsoscelesTrapezoid :=
  { side1 := 6
  , side2 := 6.25
  , diagonal_is_bisector := true }

/-- The length of the diagonal from the acute angle vertex -/
def diagonal_length (t : IsoscelesTrapezoid) : ℝ := sorry

/-- The area of the trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating the properties of the specific trapezoid -/
theorem trapezoid_properties :
  let t := problem_trapezoid
  abs (diagonal_length t - 10.423) < 0.001 ∧
  abs (trapezoid_area t - 32) < 0.001 := by sorry

end NUMINAMATH_CALUDE_trapezoid_properties_l369_36977


namespace NUMINAMATH_CALUDE_cara_arrangements_l369_36900

def num_friends : ℕ := 7

def arrangements (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

theorem cara_arrangements :
  arrangements num_friends = 6 :=
sorry

end NUMINAMATH_CALUDE_cara_arrangements_l369_36900


namespace NUMINAMATH_CALUDE_intersection_point_l369_36904

def line1 (x y : ℚ) : Prop := y = 3 * x + 1
def line2 (x y : ℚ) : Prop := y + 1 = -7 * x

theorem intersection_point :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (-1/5, 2/5) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l369_36904


namespace NUMINAMATH_CALUDE_second_number_value_l369_36999

theorem second_number_value (A B C : ℝ) 
  (sum_eq : A + B + C = 98)
  (ratio_AB : A / B = 2 / 3)
  (ratio_BC : B / C = 5 / 8)
  (pos_A : A > 0)
  (pos_B : B > 0)
  (pos_C : C > 0) : 
  B = 30 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l369_36999


namespace NUMINAMATH_CALUDE_fraction_product_equality_l369_36997

theorem fraction_product_equality : (2 : ℚ) / 3 * 4 / 7 * 9 / 11 = 24 / 77 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l369_36997


namespace NUMINAMATH_CALUDE_base4_calculation_l369_36944

/-- Represents a number in base 4 --/
def Base4 : Type := ℕ

/-- Multiplication operation for base 4 numbers --/
def mul_base4 : Base4 → Base4 → Base4 := sorry

/-- Division operation for base 4 numbers --/
def div_base4 : Base4 → Base4 → Base4 := sorry

/-- Conversion from decimal to base 4 --/
def to_base4 (n : ℕ) : Base4 := sorry

/-- Conversion from base 4 to decimal --/
def from_base4 (n : Base4) : ℕ := sorry

theorem base4_calculation :
  let a := to_base4 203
  let b := to_base4 21
  let c := to_base4 3
  let result := to_base4 110320
  mul_base4 (div_base4 a c) b = result := by sorry

end NUMINAMATH_CALUDE_base4_calculation_l369_36944


namespace NUMINAMATH_CALUDE_union_A_B_range_of_a_l369_36970

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | 1 < x ∧ x ≤ 8} := by sorry

-- Theorem for the range of a when A ∩ C is nonempty
theorem range_of_a (a : ℝ) : (A ∩ C a).Nonempty → a < 8 := by sorry

end NUMINAMATH_CALUDE_union_A_B_range_of_a_l369_36970


namespace NUMINAMATH_CALUDE_goose_eggs_count_l369_36912

/-- The number of goose eggs laid at a pond -/
def total_eggs : ℕ := 400

/-- The fraction of eggs that hatched -/
def hatch_rate : ℚ := 1/2

/-- The fraction of hatched geese that survived the first month -/
def first_month_survival_rate : ℚ := 3/4

/-- The fraction of geese that survived the first month but did not survive the first year -/
def first_year_death_rate : ℚ := 3/5

/-- The number of geese that survived the first year -/
def survived_first_year : ℕ := 120

theorem goose_eggs_count :
  (total_eggs : ℚ) * hatch_rate * first_month_survival_rate * (1 - first_year_death_rate) = survived_first_year :=
sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l369_36912


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l369_36933

/-- Calculate the length of a bridge given train parameters and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 200 →
  train_speed_kmh = 60 →
  crossing_time = 45 →
  ∃ bridge_length : ℝ,
    (bridge_length ≥ 550) ∧ 
    (bridge_length ≤ 551) ∧
    (train_speed_kmh * 1000 / 3600 * crossing_time = train_length + bridge_length) :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l369_36933


namespace NUMINAMATH_CALUDE_max_value_inequality_l369_36946

theorem max_value_inequality (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (a_ge : a ≥ -1)
  (b_ge : b ≥ -2)
  (c_ge : c ≥ -4)
  (abc_nonneg : a * b * c ≥ 0) :
  Real.sqrt (4 * a + 4) + Real.sqrt (4 * b + 8) + Real.sqrt (4 * c + 16) ≤ 2 * Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l369_36946


namespace NUMINAMATH_CALUDE_other_solution_of_quadratic_equation_l369_36924

theorem other_solution_of_quadratic_equation :
  let f (x : ℚ) := 77 * x^2 + 35 - (125 * x - 14)
  ∃ (x : ℚ), x ≠ 8/11 ∧ f x = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_of_quadratic_equation_l369_36924


namespace NUMINAMATH_CALUDE_sum_reciprocal_equals_two_max_weighted_sum_reciprocal_l369_36918

-- Define the variables and conditions
variable (a b x y : ℝ)
variable (ha : a > 1)
variable (hb : b > 1)
variable (hx : a^x = 2)
variable (hy : b^y = 2)

-- Theorem 1
theorem sum_reciprocal_equals_two (hab : a * b = 4) :
  1 / x + 1 / y = 2 := by sorry

-- Theorem 2
theorem max_weighted_sum_reciprocal (hab : a^2 + b = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b x y : ℝ), a > 1 → b > 1 → a^x = 2 → b^y = 2 → a^2 + b = 8 →
    2 / x + 1 / y ≤ m := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_equals_two_max_weighted_sum_reciprocal_l369_36918


namespace NUMINAMATH_CALUDE_tournament_dominance_chain_l369_36916

/-- Represents a round-robin tournament with 8 players -/
structure Tournament :=
  (players : Finset (Fin 8))
  (defeated : Fin 8 → Fin 8 → Prop)
  (round_robin : ∀ i j, i ≠ j → (defeated i j ∨ defeated j i))
  (asymmetric : ∀ i j, defeated i j → ¬ defeated j i)

/-- The main theorem to be proved -/
theorem tournament_dominance_chain (t : Tournament) :
  ∃ (a b c d : Fin 8),
    a ∈ t.players ∧ b ∈ t.players ∧ c ∈ t.players ∧ d ∈ t.players ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    t.defeated a b ∧ t.defeated a c ∧ t.defeated a d ∧
    t.defeated b c ∧ t.defeated b d ∧
    t.defeated c d :=
sorry

end NUMINAMATH_CALUDE_tournament_dominance_chain_l369_36916


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_prism_l369_36993

/-- The surface area of a sphere circumscribing a right square prism -/
theorem sphere_surface_area_of_prism (base_edge : ℝ) (height : ℝ) 
  (h_base : base_edge = 2) (h_height : height = 3) :
  4 * π * ((base_edge^2 + base_edge^2 + height^2).sqrt / 2)^2 = 17 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_prism_l369_36993


namespace NUMINAMATH_CALUDE_product_divisible_by_5_probability_l369_36920

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The probability that the product of the numbers rolled is divisible by 5 -/
def prob_divisible_by_5 : ℚ := 144495 / 262144

/-- Theorem stating the probability of the product being divisible by 5 -/
theorem product_divisible_by_5_probability :
  (1 : ℚ) - (1 - 1 / num_sides) ^ num_dice = prob_divisible_by_5 := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_5_probability_l369_36920


namespace NUMINAMATH_CALUDE_final_digit_is_two_l369_36967

/-- Represents the state of the board with counts of 0s, 1s, and 2s -/
structure BoardState where
  zeros : ℕ
  ones : ℕ
  twos : ℕ

/-- Represents a valid operation on the board -/
inductive Operation
  | erase_zero_one_add_two
  | erase_one_two_add_zero
  | erase_zero_two_add_one

/-- Applies an operation to a board state -/
def apply_operation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.erase_zero_one_add_two => 
      ⟨state.zeros - 1, state.ones - 1, state.twos + 1⟩
  | Operation.erase_one_two_add_zero => 
      ⟨state.zeros + 1, state.ones - 1, state.twos - 1⟩
  | Operation.erase_zero_two_add_one => 
      ⟨state.zeros - 1, state.ones + 1, state.twos - 1⟩

/-- Checks if the board state has only one digit remaining -/
def has_one_digit (state : BoardState) : Prop :=
  (state.zeros = 1 ∧ state.ones = 0 ∧ state.twos = 0) ∨
  (state.zeros = 0 ∧ state.ones = 1 ∧ state.twos = 0) ∨
  (state.zeros = 0 ∧ state.ones = 0 ∧ state.twos = 1)

/-- The main theorem to prove -/
theorem final_digit_is_two 
  (initial : BoardState) 
  (operations : List Operation) 
  (h_final : has_one_digit (operations.foldl apply_operation initial)) :
  (operations.foldl apply_operation initial).twos = 1 :=
sorry

end NUMINAMATH_CALUDE_final_digit_is_two_l369_36967


namespace NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_l369_36907

def M (a : ℝ) : Set ℝ := {1, a}
def N : Set ℝ := {-1, 0, 1}

theorem a_zero_sufficient_not_necessary :
  (∀ a : ℝ, a = 0 → M a ⊆ N) ∧
  (∃ a : ℝ, a ≠ 0 ∧ M a ⊆ N) :=
by sorry

end NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_l369_36907


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l369_36979

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → a ∈ Set.Icc (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l369_36979


namespace NUMINAMATH_CALUDE_sugar_price_increase_vs_inflation_sugar_price_increase_specific_l369_36981

/-- The percentage by which the rate of increase of sugar price exceeds inflation --/
theorem sugar_price_increase_vs_inflation (initial_price final_price : ℝ) 
  (inflation_rate : ℝ) (years : ℕ) : ℝ :=
  let total_sugar_increase := (final_price - initial_price) / initial_price * 100
  let total_inflation := ((1 + inflation_rate / 100) ^ years - 1) * 100
  total_sugar_increase - total_inflation

/-- Given specific values, prove that the difference is approximately 6.81% --/
theorem sugar_price_increase_specific :
  let initial_price : ℝ := 25
  let final_price : ℝ := 33.0625
  let inflation_rate : ℝ := 12
  let years : ℕ := 2
  abs (sugar_price_increase_vs_inflation initial_price final_price inflation_rate years - 6.81) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_sugar_price_increase_vs_inflation_sugar_price_increase_specific_l369_36981


namespace NUMINAMATH_CALUDE_max_value_constraint_l369_36965

theorem max_value_constraint (x y : ℝ) 
  (h1 : |x - y| ≤ 2) 
  (h2 : |3*x + y| ≤ 6) : 
  x^2 + y^2 ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l369_36965


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_l369_36950

-- Define the value of a trillion
def trillion : ℝ := 10^12

-- Define the GDP value in trillions
def gdp_trillions : ℝ := 121

-- Theorem statement
theorem gdp_scientific_notation :
  gdp_trillions * trillion = 1.21 * 10^14 := by
  sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_l369_36950


namespace NUMINAMATH_CALUDE_ascending_order_l369_36964

-- Define the variables
def a : ℕ := 2^55
def b : ℕ := 3^44
def c : ℕ := 5^33
def d : ℕ := 6^22

-- Theorem stating the ascending order
theorem ascending_order : a < d ∧ d < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_ascending_order_l369_36964


namespace NUMINAMATH_CALUDE_smallest_base_for_61_digits_l369_36990

theorem smallest_base_for_61_digits : ∃ (b : ℕ), b > 1 ∧ 
  (∀ (n : ℕ), n > 1 → n < b → (Nat.log 10 (n^200) + 1 < 61)) ∧ 
  (Nat.log 10 (b^200) + 1 = 61) := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_61_digits_l369_36990


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l369_36931

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ) ^ (x^2 - 6*x + 8) = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l369_36931


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l369_36959

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l369_36959


namespace NUMINAMATH_CALUDE_min_trees_triangular_plot_l369_36903

/-- Given a triangular plot with 5 trees planted on each side, 
    the minimum number of trees that can be planted is 12. -/
theorem min_trees_triangular_plot : 
  ∀ (trees_per_side : ℕ), 
  trees_per_side = 5 → 
  (∃ (min_trees : ℕ), 
    min_trees = 12 ∧ 
    ∀ (total_trees : ℕ), 
      (total_trees ≥ min_trees ∧ 
       ∃ (trees_on_edges : ℕ), 
         trees_on_edges = total_trees - 3 ∧ 
         trees_on_edges % 3 = 0 ∧ 
         trees_on_edges / 3 + 1 = trees_per_side)) :=
by sorry

end NUMINAMATH_CALUDE_min_trees_triangular_plot_l369_36903


namespace NUMINAMATH_CALUDE_hilt_pies_theorem_l369_36928

/-- The total number of pies Mrs. Hilt needs to bake -/
def total_pies (pecan_pies apple_pies : ℝ) (factor : ℝ) : ℝ :=
  (pecan_pies + apple_pies) * factor

/-- Theorem: Given the initial number of pecan pies (16.0) and apple pies (14.0),
    and a multiplication factor (5.0), the total number of pies Mrs. Hilt
    needs to bake is 150.0. -/
theorem hilt_pies_theorem :
  total_pies 16.0 14.0 5.0 = 150.0 := by
  sorry

end NUMINAMATH_CALUDE_hilt_pies_theorem_l369_36928


namespace NUMINAMATH_CALUDE_three_numbers_sum_l369_36998

theorem three_numbers_sum (s : ℕ) :
  let A := Finset.range (4 * s) 
  ∀ (S : Finset ℕ), S ⊆ A → S.card = 2 * s + 2 →
    ∃ (x y z : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x + y = 2 * z :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l369_36998


namespace NUMINAMATH_CALUDE_min_f_correct_a_range_condition_l369_36943

noncomputable section

def f (a x : ℝ) : ℝ := x - (a + 1) * Real.log x - a / x

def g (x : ℝ) : ℝ := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

def min_f (a : ℝ) : ℝ :=
  if a ≤ 1 then 1 - a
  else if a < Real.exp 1 then a - (a + 1) * Real.log a - 1
  else Real.exp 1 - (a + 1) - a / Real.exp 1

theorem min_f_correct (a : ℝ) :
  ∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ min_f a := by sorry

theorem a_range_condition (a : ℝ) :
  a < 1 →
  (∃ x₁ ∈ Set.Icc (Real.exp 1) (Real.exp 2),
    ∀ x₂ ∈ Set.Icc (-2) 0, f a x₁ < g x₂) →
  (Real.exp 2 - 2 * Real.exp 1) / (Real.exp 1 + 1) < a := by sorry

end

end NUMINAMATH_CALUDE_min_f_correct_a_range_condition_l369_36943


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_fourths_l369_36940

theorem greatest_integer_less_than_negative_seventeen_fourths :
  ⌊-17/4⌋ = -5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_fourths_l369_36940


namespace NUMINAMATH_CALUDE_cuboid_length_problem_l369_36989

/-- The surface area of a cuboid given its length, width, and height -/
def cuboidSurfaceArea (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: The length of a cuboid with surface area 700 m², breadth 14 m, and height 7 m is 12 m -/
theorem cuboid_length_problem :
  ∃ (l : ℝ), cuboidSurfaceArea l 14 7 = 700 ∧ l = 12 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_length_problem_l369_36989


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l369_36927

noncomputable section

variables (a b c : ℝ) (f : ℝ → ℝ)

def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_properties
  (h1 : quadratic_function f)
  (h2 : f 0 = 2)
  (h3 : ∀ x, f (x + 1) - f x = 2 * x - 1) :
  (∀ x, f x = x^2 - 2*x + 2) ∧
  (∀ x, x > 1 → (deriv f) x > 0) ∧
  (∀ x, x < 1 → (deriv f) x < 0) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≤ 5) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = 5) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = 1) :=
by
  sorry

end

end NUMINAMATH_CALUDE_quadratic_function_properties_l369_36927


namespace NUMINAMATH_CALUDE_salary_change_percentage_l369_36996

theorem salary_change_percentage (S : ℝ) (x : ℝ) (h : S > 0) :
  S * (1 - x / 100) * (1 + x / 100) = 0.75 * S → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l369_36996


namespace NUMINAMATH_CALUDE_triangle_inequality_triangle_equality_l369_36985

/-- The area of a triangle with sides a, b, c -/
noncomputable def A (a b c : ℝ) : ℝ := sorry

/-- Function f as defined in the problem -/
noncomputable def f (a b c : ℝ) : ℝ := Real.sqrt (A a b c)

/-- The main theorem -/
theorem triangle_inequality (a b c a' b' c' : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (ha' : 0 < a') (hb' : 0 < b') (hc' : 0 < c') :
    f a b c + f a' b' c' ≤ f (a + a') (b + b') (c + c') :=
  sorry

/-- Condition for equality -/
theorem triangle_equality (a b c a' b' c' : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (ha' : 0 < a') (hb' : 0 < b') (hc' : 0 < c') :
    f a b c + f a' b' c' = f (a + a') (b + b') (c + c') ↔ a / a' = b / b' ∧ b / b' = c / c' :=
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_triangle_equality_l369_36985


namespace NUMINAMATH_CALUDE_infinite_primes_with_property_l369_36915

theorem infinite_primes_with_property : 
  ∃ (S : Set Nat), 
    (∀ p ∈ S, Nat.Prime p) ∧ 
    (Set.Infinite S) ∧ 
    (∀ p ∈ S, ∃ n : Nat, ¬(n ∣ (p - 1)) ∧ (p ∣ (Nat.factorial n + 1))) := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_with_property_l369_36915


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l369_36988

theorem at_least_one_not_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l369_36988


namespace NUMINAMATH_CALUDE_log_sum_sqrt_equals_sqrt_thirteen_sixths_l369_36922

theorem log_sum_sqrt_equals_sqrt_thirteen_sixths :
  Real.sqrt (Real.log 8 / Real.log 4 + Real.log 4 / Real.log 8) = Real.sqrt (13 / 6) := by
  sorry

end NUMINAMATH_CALUDE_log_sum_sqrt_equals_sqrt_thirteen_sixths_l369_36922


namespace NUMINAMATH_CALUDE_dietitian_excess_calories_l369_36980

/-- Calculates the excess calories consumed given the total lunch calories and the fraction eaten -/
def excess_calories (total_calories : ℕ) (fraction_eaten : ℚ) (recommended_calories : ℕ) : ℤ :=
  ⌊(fraction_eaten * total_calories : ℚ)⌋ - recommended_calories

/-- Proves that eating 3/4 of a 40-calorie lunch exceeds the recommended 25 calories by 5 -/
theorem dietitian_excess_calories :
  excess_calories 40 (3/4 : ℚ) 25 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dietitian_excess_calories_l369_36980


namespace NUMINAMATH_CALUDE_factor_expression_l369_36969

theorem factor_expression (x : ℝ) : 72 * x^3 - 250 * x^7 = 2 * x^3 * (36 - 125 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l369_36969


namespace NUMINAMATH_CALUDE_tangent_line_equation_l369_36962

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 3

theorem tangent_line_equation :
  let P : ℝ × ℝ := (1, f 1)
  let m : ℝ := (3 * P.1^2 - 1)  -- Derivative of f at x = 1
  (2 : ℝ) * x - y + 1 = 0 ↔ y - P.2 = m * (x - P.1) := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l369_36962


namespace NUMINAMATH_CALUDE_condition_relationship_l369_36935

theorem condition_relationship : 
  ∀ x : ℝ, (x > 3 → x > 2) ∧ ¬(x > 2 → x > 3) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l369_36935


namespace NUMINAMATH_CALUDE_hannah_cookies_sold_l369_36905

/-- Proves that Hannah sold 40 cookies given the conditions of the problem -/
theorem hannah_cookies_sold : ℕ :=
  let cookie_price : ℚ := 8 / 10
  let cupcake_price : ℚ := 2
  let cupcakes_sold : ℕ := 30
  let spoon_set_price : ℚ := 13 / 2
  let spoon_sets_bought : ℕ := 2
  let money_left : ℚ := 79

  let cookies_sold : ℕ := 40

  have h1 : cookie_price * cookies_sold + cupcake_price * cupcakes_sold = 
            spoon_set_price * spoon_sets_bought + money_left := by sorry

  cookies_sold


end NUMINAMATH_CALUDE_hannah_cookies_sold_l369_36905


namespace NUMINAMATH_CALUDE_license_plate_increase_l369_36986

theorem license_plate_increase : 
  let old_plates := 26 * 10^4
  let new_plates := 26^3 * 10^3
  new_plates / old_plates = 26^2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_increase_l369_36986


namespace NUMINAMATH_CALUDE_quarter_power_equality_l369_36992

theorem quarter_power_equality (x : ℝ) : (1 / 4 : ℝ) ^ x = 0.25 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quarter_power_equality_l369_36992


namespace NUMINAMATH_CALUDE_f_composition_equals_one_over_e_l369_36958

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else Real.log x

theorem f_composition_equals_one_over_e :
  f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_f_composition_equals_one_over_e_l369_36958


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l369_36942

/-- A parallelogram with consecutive side lengths 12, 5y-3, 3x+2, and 9 has x+y equal to 86/15 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  (3*x + 2 = 12) → (5*y - 3 = 9) → x + y = 86/15 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l369_36942


namespace NUMINAMATH_CALUDE_exam_students_count_l369_36929

theorem exam_students_count (total_average : ℝ) (excluded_average : ℝ) (new_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 40) 
  (h3 : new_average = 90) (h4 : excluded_count = 5) : 
  ∃ (n : ℕ), n = 25 ∧ 
    (n : ℝ) * total_average = 
      ((n - excluded_count) : ℝ) * new_average + (excluded_count : ℝ) * excluded_average :=
by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l369_36929


namespace NUMINAMATH_CALUDE_relay_race_distance_per_member_l369_36976

theorem relay_race_distance_per_member 
  (total_distance : ℕ) (team_members : ℕ) 
  (h1 : total_distance = 150) 
  (h2 : team_members = 5) : 
  total_distance / team_members = 30 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_distance_per_member_l369_36976


namespace NUMINAMATH_CALUDE_election_percentage_l369_36960

theorem election_percentage (total_votes : ℕ) (vote_difference : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 5500 →
  vote_difference = 1650 →
  candidate_percentage = 35 / 100 →
  (candidate_percentage * total_votes : ℚ) + 
  (candidate_percentage * total_votes : ℚ) + vote_difference = total_votes :=
by sorry

end NUMINAMATH_CALUDE_election_percentage_l369_36960


namespace NUMINAMATH_CALUDE_speed_ratio_l369_36949

/-- Two people walk in opposite directions for 1 hour and swap destinations -/
structure WalkProblem where
  /-- Speed of person A in km/h -/
  v₁ : ℝ
  /-- Speed of person B in km/h -/
  v₂ : ℝ
  /-- Both speeds are positive -/
  h₁ : v₁ > 0
  h₂ : v₂ > 0
  /-- Person A reaches B's destination 35 minutes after B reaches A's destination -/
  h₃ : v₂ / v₁ - v₁ / v₂ = 35 / 60

/-- The ratio of speeds is 3:4 -/
theorem speed_ratio (w : WalkProblem) : w.v₁ / w.v₂ = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_l369_36949


namespace NUMINAMATH_CALUDE_cubic_trinomial_degree_l369_36952

theorem cubic_trinomial_degree (n : ℕ) : 
  (∃ (p : Polynomial ℝ), p = X^n - 5*X + 4 ∧ Polynomial.degree p = 3) → n = 3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_trinomial_degree_l369_36952


namespace NUMINAMATH_CALUDE_collection_for_37_members_l369_36984

/-- Calculates the total collection amount in rupees for a group of students -/
def total_collection_rupees (num_members : ℕ) (paise_per_rupee : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / paise_per_rupee

/-- Proves that the total collection for 37 members is 13.69 rupees -/
theorem collection_for_37_members :
  total_collection_rupees 37 100 = 13.69 := by
  sorry

#eval total_collection_rupees 37 100

end NUMINAMATH_CALUDE_collection_for_37_members_l369_36984


namespace NUMINAMATH_CALUDE_oil_bill_ratio_l369_36939

/-- The oil bill problem -/
theorem oil_bill_ratio (january_bill : ℝ) (february_bill : ℝ) : 
  january_bill = 119.99999999999994 →
  february_bill / january_bill = 3 / 2 →
  (february_bill + 20) / january_bill = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_oil_bill_ratio_l369_36939


namespace NUMINAMATH_CALUDE_quarter_percent_of_160_l369_36973

theorem quarter_percent_of_160 : (1 / 4 : ℚ) / 100 * 160 = (0.4 : ℚ) := by sorry

end NUMINAMATH_CALUDE_quarter_percent_of_160_l369_36973


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_attained_l369_36932

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 6) : 
  (9 / x + 16 / y + 25 / z) ≥ 24 := by
  sorry

theorem min_value_attained (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 6) : 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ + y₀ + z₀ = 6 ∧ 
  (9 / x₀ + 16 / y₀ + 25 / z₀) = 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_attained_l369_36932


namespace NUMINAMATH_CALUDE_club_membership_l369_36971

theorem club_membership (total : ℕ) (difference : ℕ) (first_year : ℕ) : 
  total = 128 →
  difference = 12 →
  first_year = total / 2 + difference / 2 →
  first_year = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_club_membership_l369_36971


namespace NUMINAMATH_CALUDE_range_of_a_l369_36983

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l369_36983


namespace NUMINAMATH_CALUDE_square_root_81_l369_36941

theorem square_root_81 : ∀ (x : ℝ), x^2 = 81 ↔ x = 9 ∨ x = -9 := by sorry

end NUMINAMATH_CALUDE_square_root_81_l369_36941


namespace NUMINAMATH_CALUDE_solution_correctness_l369_36921

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + Real.sqrt (x + 2*y) - 2*y = 7/2
def equation2 (x y : ℝ) : Prop := x^2 + x + 2*y - 4*y^2 = 27/2

-- State the theorem
theorem solution_correctness : 
  equation1 (19/4) (17/8) ∧ equation2 (19/4) (17/8) := by sorry

end NUMINAMATH_CALUDE_solution_correctness_l369_36921


namespace NUMINAMATH_CALUDE_smallest_fourth_power_b_l369_36957

theorem smallest_fourth_power_b : ∃ (n : ℕ), 
  (7 + 7 * 18 + 7 * 18^2 = n^4) ∧ 
  (∀ (b : ℕ), b > 0 → b < 18 → ¬∃ (m : ℕ), 7 + 7 * b + 7 * b^2 = m^4) := by
  sorry

end NUMINAMATH_CALUDE_smallest_fourth_power_b_l369_36957


namespace NUMINAMATH_CALUDE_gcd_product_equivalence_l369_36901

theorem gcd_product_equivalence (a m n : ℤ) : 
  Int.gcd a (m * n) = 1 ↔ Int.gcd a m = 1 ∧ Int.gcd a n = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_product_equivalence_l369_36901
