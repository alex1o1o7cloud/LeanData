import Mathlib

namespace river_crossing_l2639_263963

/-- Represents a river with islands -/
structure River :=
  (width : ℝ)
  (islandsPerimeter : ℝ)

/-- Theorem stating that it's possible to cross the river in less than 3 meters -/
theorem river_crossing (r : River) 
  (h_width : r.width = 1)
  (h_perimeter : r.islandsPerimeter = 8) : 
  ∃ (path : ℝ), path < 3 ∧ path ≥ r.width :=
sorry

end river_crossing_l2639_263963


namespace triangle_special_case_triangle_inequality_l2639_263967

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ
  hpos : 0 < a ∧ 0 < b ∧ 0 < c
  htri : a + b > c ∧ b + c > a ∧ c + a > b

-- Part (a)
theorem triangle_special_case (t : Triangle) 
  (h : 6 * t.area = 2 * t.a^2 + t.b * t.c) : 
  t.b = t.c ∧ t.b = Real.sqrt (5/2) * t.a :=
sorry

-- Part (b)
theorem triangle_inequality (t : Triangle) :
  3 * t.a^2 + 3 * t.b^2 - t.c^2 ≥ 4 * Real.sqrt 3 * t.area :=
sorry

end triangle_special_case_triangle_inequality_l2639_263967


namespace factorization_of_x2y_plus_xy2_l2639_263974

theorem factorization_of_x2y_plus_xy2 (x y : ℝ) : x^2*y + x*y^2 = x*y*(x + y) := by
  sorry

end factorization_of_x2y_plus_xy2_l2639_263974


namespace triangle_min_perimeter_l2639_263949

theorem triangle_min_perimeter (a b x : ℕ) (ha : a = 36) (hb : b = 50) : 
  (a + b > x ∧ a + x > b ∧ b + x > a) → (a + b + x ≥ 101) := by
  sorry

end triangle_min_perimeter_l2639_263949


namespace union_when_a_neg_two_subset_condition_l2639_263922

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a^2 + 1}

-- Statement for part (i)
theorem union_when_a_neg_two :
  A (-2) ∪ B (-2) = {x : ℝ | -5 < x ∧ x < 5} := by sorry

-- Statement for part (ii)
theorem subset_condition :
  ∀ a : ℝ, B a ⊆ A a ↔ a ∈ ({x : ℝ | 1 ≤ x ∧ x ≤ 3} ∪ {-1}) := by sorry

end union_when_a_neg_two_subset_condition_l2639_263922


namespace brick_length_proof_l2639_263910

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ :=
  d.length * d.width * d.height

theorem brick_length_proof (wall : Dimensions) (brick : Dimensions) (num_bricks : ℝ) :
  wall.length = 8 →
  wall.width = 6 →
  wall.height = 0.02 →
  brick.length = 0.11 →
  brick.width = 0.05 →
  brick.height = 0.06 →
  num_bricks = 2909.090909090909 →
  volume wall / volume brick = num_bricks →
  brick.length = 0.11 := by
  sorry

end brick_length_proof_l2639_263910


namespace vector_sum_l2639_263944

/-- Given two plane vectors a and b, prove their sum is (0, 1) -/
theorem vector_sum (a b : ℝ × ℝ) (h1 : a = (1, -1)) (h2 : b = (-1, 2)) :
  a + b = (0, 1) := by sorry

end vector_sum_l2639_263944


namespace marshmallow_challenge_l2639_263918

/-- The number of marshmallows Haley can hold -/
def haley_marshmallows : ℕ := sorry

/-- The number of marshmallows Michael can hold -/
def michael_marshmallows : ℕ := 3 * haley_marshmallows

/-- The number of marshmallows Brandon can hold -/
def brandon_marshmallows : ℕ := michael_marshmallows / 2

/-- The total number of marshmallows all three kids can hold -/
def total_marshmallows : ℕ := 44

theorem marshmallow_challenge : 
  haley_marshmallows + michael_marshmallows + brandon_marshmallows = total_marshmallows ∧ 
  haley_marshmallows = 8 := by sorry

end marshmallow_challenge_l2639_263918


namespace simplify_expression_l2639_263960

theorem simplify_expression : (7^5 + 2^7) * (2^3 - (-1)^3)^8 = 729000080835 := by
  sorry

end simplify_expression_l2639_263960


namespace division_multiplication_identity_l2639_263902

theorem division_multiplication_identity (a : ℝ) (h : a ≠ 0) : 1 / a * a = 1 := by
  sorry

end division_multiplication_identity_l2639_263902


namespace sufficient_condition_increasing_f_increasing_on_interval_l2639_263938

/-- A sufficient condition for f(x) = x^2 + 2ax + 1 to be increasing on (1, +∞) -/
theorem sufficient_condition_increasing (a : ℝ) (h : a = -1) :
  ∀ x y, 1 < x → x < y → x^2 + 2*a*x + 1 < y^2 + 2*a*y + 1 := by
  sorry

/-- Definition of the function f(x) = x^2 + 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

/-- The function f is increasing on (1, +∞) when a = -1 -/
theorem f_increasing_on_interval (a : ℝ) (h : a = -1) :
  StrictMonoOn (f a) (Set.Ioi 1) := by
  sorry

end sufficient_condition_increasing_f_increasing_on_interval_l2639_263938


namespace player1_can_achieve_6_player2_can_prevent_above_6_max_achievable_sum_is_6_l2639_263976

/-- Represents a cell on the 5x5 board -/
inductive Cell
| mk (row : Fin 5) (col : Fin 5)

/-- Represents the state of a cell (Empty, Marked by Player 1, or Marked by Player 2) -/
inductive CellState
| Empty
| Player1
| Player2

/-- Represents the game board -/
def Board := Cell → CellState

/-- Checks if a given 3x3 sub-square is valid on the 5x5 board -/
def isValid3x3Square (topLeft : Cell) : Prop :=
  ∃ (r c : Fin 3), topLeft = Cell.mk r c

/-- Computes the sum of a 3x3 sub-square -/
def subSquareSum (b : Board) (topLeft : Cell) : ℕ :=
  sorry

/-- The maximum sum of any 3x3 sub-square on the board -/
def maxSubSquareSum (b : Board) : ℕ :=
  sorry

/-- A strategy for Player 1 -/
def Player1Strategy := Board → Cell

/-- A strategy for Player 2 -/
def Player2Strategy := Board → Cell

/-- Simulates a game given strategies for both players -/
def playGame (s1 : Player1Strategy) (s2 : Player2Strategy) : Board :=
  sorry

/-- Theorem stating that Player 1 can always achieve a maximum 3x3 sub-square sum of at least 6 -/
theorem player1_can_achieve_6 :
  ∃ (s1 : Player1Strategy), ∀ (s2 : Player2Strategy),
    maxSubSquareSum (playGame s1 s2) ≥ 6 :=
  sorry

/-- Theorem stating that Player 2 can always prevent the maximum 3x3 sub-square sum from exceeding 6 -/
theorem player2_can_prevent_above_6 :
  ∃ (s2 : Player2Strategy), ∀ (s1 : Player1Strategy),
    maxSubSquareSum (playGame s1 s2) ≤ 6 :=
  sorry

/-- Main theorem combining the above results -/
theorem max_achievable_sum_is_6 :
  (∃ (s1 : Player1Strategy), ∀ (s2 : Player2Strategy),
    maxSubSquareSum (playGame s1 s2) ≥ 6) ∧
  (∃ (s2 : Player2Strategy), ∀ (s1 : Player1Strategy),
    maxSubSquareSum (playGame s1 s2) ≤ 6) :=
  sorry

end player1_can_achieve_6_player2_can_prevent_above_6_max_achievable_sum_is_6_l2639_263976


namespace unique_solution_l2639_263959

theorem unique_solution : ∀ a b c : ℕ, 2^a + 9^b = 2 * 5^c + 5 ↔ a = 1 ∧ b = 0 ∧ c = 0 := by
  sorry

end unique_solution_l2639_263959


namespace tournament_matches_l2639_263937

/-- A tournament with the given rules --/
structure Tournament :=
  (num_players : ℕ)
  (num_players_per_match : ℕ)
  (points_per_match : Fin 3 → ℕ)
  (eliminated_per_match : ℕ)

/-- The number of matches played in a tournament --/
def num_matches (t : Tournament) : ℕ :=
  t.num_players - 1

/-- The theorem stating the number of matches in the specific tournament --/
theorem tournament_matches :
  ∀ t : Tournament,
    t.num_players = 999 ∧
    t.num_players_per_match = 3 ∧
    t.points_per_match 0 = 2 ∧
    t.points_per_match 1 = 1 ∧
    t.points_per_match 2 = 0 ∧
    t.eliminated_per_match = 1 →
    num_matches t = 998 :=
by sorry

end tournament_matches_l2639_263937


namespace time_after_850_hours_l2639_263903

/-- Represents a time on a 12-hour clock -/
structure Time12Hour where
  hour : Nat
  minute : Nat
  period : Bool  -- false for AM, true for PM
  h_valid : hour ≥ 1 ∧ hour ≤ 12
  m_valid : minute ≥ 0 ∧ minute < 60

/-- Adds hours to a given time on a 12-hour clock -/
def addHours (t : Time12Hour) (h : Nat) : Time12Hour :=
  sorry

theorem time_after_850_hours : 
  let start_time := Time12Hour.mk 3 15 true (by norm_num) (by norm_num)
  let end_time := Time12Hour.mk 1 15 false (by norm_num) (by norm_num)
  addHours start_time 850 = end_time := by sorry

end time_after_850_hours_l2639_263903


namespace BRICS_is_set_closeToZero_is_not_l2639_263993

-- Define a type for countries
structure Country where
  name : String

-- Define the BRICS summit participants
def BRICS2016Participants : Set Country := sorry

-- Define a property for real numbers "close to 0"
def closeToZero (x : ℝ) : Prop := sorry

theorem BRICS_is_set_closeToZero_is_not :
  (∃ (S : Set Country), S = BRICS2016Participants) ∧
  (¬ ∃ (T : Set ℝ), ∀ x, x ∈ T ↔ closeToZero x) :=
sorry

end BRICS_is_set_closeToZero_is_not_l2639_263993


namespace chocolate_milk_total_ounces_l2639_263924

-- Define the ingredients per glass
def milk_per_glass : ℚ := 6
def syrup_per_glass : ℚ := 1.5
def cream_per_glass : ℚ := 0.5

-- Define the total available ingredients
def total_milk : ℚ := 130
def total_syrup : ℚ := 60
def total_cream : ℚ := 25

-- Define the size of each glass
def glass_size : ℚ := 8

-- Theorem to prove
theorem chocolate_milk_total_ounces :
  let max_glasses := min (total_milk / milk_per_glass) 
                         (min (total_syrup / syrup_per_glass) (total_cream / cream_per_glass))
  let full_glasses := ⌊max_glasses⌋
  full_glasses * glass_size = 168 := by
sorry

end chocolate_milk_total_ounces_l2639_263924


namespace shaded_area_ratio_l2639_263940

theorem shaded_area_ratio : 
  ∀ (r₁ r₂ r₃ r₄ : ℝ), 
    r₁ = 1 → r₂ = 2 → r₃ = 3 → r₄ = 4 →
    (π * r₁^2 + π * r₃^2 - π * r₂^2) / (π * r₄^2) = 3 / 8 := by
  sorry

end shaded_area_ratio_l2639_263940


namespace four_point_equal_inradii_congruent_triangles_l2639_263991

-- Define a type for points in a plane
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a type for triangles
structure Triangle :=
  (a b c : Point)

-- Define a function to check if three points are collinear
def collinear (p q r : Point) : Prop := sorry

-- Define a function to calculate the inradius of a triangle
def inradius (t : Triangle) : ℝ := sorry

-- Define a function to check if two triangles are congruent
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Main theorem
theorem four_point_equal_inradii_congruent_triangles 
  (A B C D : Point) : 
  (¬collinear A B C ∧ ¬collinear A B D ∧ ¬collinear A C D ∧ ¬collinear B C D) →
  (inradius ⟨A, B, C⟩ = inradius ⟨A, B, D⟩) →
  (inradius ⟨A, B, C⟩ = inradius ⟨A, C, D⟩) →
  (inradius ⟨A, B, C⟩ = inradius ⟨B, C, D⟩) →
  (congruent ⟨A, B, C⟩ ⟨A, B, D⟩) ∧ 
  (congruent ⟨A, B, C⟩ ⟨A, C, D⟩) ∧ 
  (congruent ⟨A, B, C⟩ ⟨B, C, D⟩) :=
by sorry

end four_point_equal_inradii_congruent_triangles_l2639_263991


namespace hapok_guarantee_l2639_263978

/-- Represents the coin division game between Hapok and Glazok -/
structure CoinGame where
  totalCoins : Nat
  maxHandfuls : Nat

/-- Represents a strategy for Hapok -/
structure Strategy where
  coinsPerHandful : Nat

/-- Calculates the minimum number of coins Hapok can guarantee with a given strategy -/
def guaranteedCoins (game : CoinGame) (strategy : Strategy) : Nat :=
  let fullHandfuls := game.totalCoins / strategy.coinsPerHandful
  let remainingCoins := game.totalCoins % strategy.coinsPerHandful
  if fullHandfuls ≥ 2 * game.maxHandfuls - 1 then
    (game.maxHandfuls - 1) * strategy.coinsPerHandful + remainingCoins
  else
    (fullHandfuls - game.maxHandfuls) * strategy.coinsPerHandful

/-- Theorem stating that Hapok can guarantee at least 46 coins -/
theorem hapok_guarantee (game : CoinGame) (strategy : Strategy) :
  game.totalCoins = 100 →
  game.maxHandfuls = 9 →
  strategy.coinsPerHandful = 6 →
  guaranteedCoins game strategy ≥ 46 := by
  sorry

#eval guaranteedCoins { totalCoins := 100, maxHandfuls := 9 } { coinsPerHandful := 6 }

end hapok_guarantee_l2639_263978


namespace expression_equality_l2639_263989

theorem expression_equality : 12 + 5*(4-9)^2 - 3 = 134 := by sorry

end expression_equality_l2639_263989


namespace opposite_sides_line_range_l2639_263909

theorem opposite_sides_line_range (a : ℝ) : 
  (0 + 0 < a ∧ a < 1 + 1) ∨ (0 + 0 > a ∧ a > 1 + 1) ↔ a < 0 ∨ a > 2 := by
sorry

end opposite_sides_line_range_l2639_263909


namespace prime_power_minus_cube_eq_one_l2639_263964

theorem prime_power_minus_cube_eq_one (p : ℕ) (hp : Prime p) :
  ∀ x y : ℕ, x > 0 → y > 0 → p^x - y^3 = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) :=
sorry

end prime_power_minus_cube_eq_one_l2639_263964


namespace largest_class_size_l2639_263923

/-- Represents the number of students in the largest class of a school -/
def largest_class (n : ℕ) : ℕ := n

/-- Represents the total number of students in the school -/
def total_students (n : ℕ) : ℕ := 
  (largest_class n) + 
  (largest_class n - 2) + 
  (largest_class n - 4) + 
  (largest_class n - 6) + 
  (largest_class n - 8)

/-- Theorem stating that the largest class has 25 students -/
theorem largest_class_size : 
  (total_students 25 = 105) ∧ (largest_class 25 = 25) := by
  sorry

#check largest_class_size

end largest_class_size_l2639_263923


namespace ratio_calculation_l2639_263985

theorem ratio_calculation (X Y Z : ℚ) (h : X / Y = 3 / 2 ∧ Y / Z = 1 / 3) :
  (4 * X + 3 * Y) / (5 * Z - 2 * X) = 3 / 4 := by
  sorry

end ratio_calculation_l2639_263985


namespace square_area_from_diagonal_l2639_263979

theorem square_area_from_diagonal (d : ℝ) (h : d = 40) : 
  (d^2 / 2) = 800 := by sorry

#check square_area_from_diagonal

end square_area_from_diagonal_l2639_263979


namespace binary_to_octal_conversion_l2639_263920

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_natural (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to its octal representation as a list of digits. -/
def natural_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: natural_to_octal (n / 8)

/-- The binary representation of the number to be converted. -/
def binary_number : List Bool := [true, true, true, false, false]

/-- The expected octal representation. -/
def expected_octal : List ℕ := [4, 3]

theorem binary_to_octal_conversion :
  natural_to_octal (binary_to_natural binary_number) = expected_octal := by
  sorry

#eval binary_to_natural binary_number
#eval natural_to_octal (binary_to_natural binary_number)

end binary_to_octal_conversion_l2639_263920


namespace four_digit_multiple_of_65_l2639_263961

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_number (n : ℕ) : ℕ :=
  let d := n % 10
  let c := (n / 10) % 10
  let b := (n / 100) % 10
  let a := n / 1000
  1000 * d + 100 * c + 10 * b + a

theorem four_digit_multiple_of_65 :
  ∃! n : ℕ, is_four_digit n ∧ 
            65 ∣ n ∧ 
            65 ∣ (reverse_number n) ∧
            n = 5005 := by
  sorry

end four_digit_multiple_of_65_l2639_263961


namespace sum_of_digits_82_l2639_263996

theorem sum_of_digits_82 :
  ∀ (tens ones : ℕ),
    tens * 10 + ones = 82 →
    tens - ones = 6 →
    tens + ones = 10 :=
by sorry

end sum_of_digits_82_l2639_263996


namespace parabola_standard_equation_l2639_263969

/-- A parabola with its focus on the line x-2y+2=0 has a standard equation of either x^2 = 4y or y^2 = -8x -/
theorem parabola_standard_equation (F : ℝ × ℝ) :
  (F.1 - 2 * F.2 + 2 = 0) →
  (∃ (x y : ℝ → ℝ), (∀ t, x t ^ 2 = 4 * y t) ∨ (∀ t, y t ^ 2 = -8 * x t)) :=
by sorry

end parabola_standard_equation_l2639_263969


namespace integer_roots_of_cubic_l2639_263951

theorem integer_roots_of_cubic (a : ℤ) : 
  (∃ x : ℤ, x^3 + 3*x^2 + a*x + 8 = 0) ↔ 
  a ∈ ({-89, -39, -30, -14, -12, -6, -2, 10} : Set ℤ) := by
  sorry

end integer_roots_of_cubic_l2639_263951


namespace angle_complement_problem_l2639_263962

theorem angle_complement_problem (x : ℝ) : 
  x + 2 * (4 * x + 10) = 90 → x = 70 / 9 :=
by sorry

end angle_complement_problem_l2639_263962


namespace distribute_students_count_l2639_263908

/-- The number of ways to distribute 4 students among 3 universities --/
def distribute_students : ℕ :=
  -- We define the function without implementation
  sorry

/-- Theorem stating that the number of ways to distribute 4 students
    among 3 universities, with each university receiving at least 1 student,
    is equal to 36 --/
theorem distribute_students_count :
  distribute_students = 36 := by
  sorry

end distribute_students_count_l2639_263908


namespace f_properties_l2639_263934

/-- The function f(x) = mx^2 + (1-3m)x - 4 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (1 - 3*m) * x - 4

theorem f_properties :
  -- Part I
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f 1 x ≤ 4 ∧ f 1 x ≥ -5) ∧
  (∃ x₁ ∈ Set.Icc (-2 : ℝ) 2, f 1 x₁ = 4) ∧
  (∃ x₂ ∈ Set.Icc (-2 : ℝ) 2, f 1 x₂ = -5) ∧

  -- Part II (simplified representation of the solution sets)
  (∀ m : ℝ, ∃ S : Set ℝ, ∀ x : ℝ, f m x > -1 ↔ x ∈ S) ∧

  -- Part III
  (∀ m < 0, (∃ x₀ > 1, f m x₀ > 0) → m < -1 ∨ (-1/9 < m ∧ m < 0)) :=
by sorry

end f_properties_l2639_263934


namespace pascal_triangle_61_row_third_number_l2639_263928

theorem pascal_triangle_61_row_third_number : 
  let n : ℕ := 60  -- The row number (61 numbers means it's the 60th row, 0-indexed)
  let k : ℕ := 2   -- The position of the number we're interested in (3rd number, 0-indexed)
  Nat.choose n k = 1770 := by
sorry

end pascal_triangle_61_row_third_number_l2639_263928


namespace dorokhov_vacation_cost_l2639_263965

/-- Represents a travel agency with its pricing structure -/
structure TravelAgency where
  name : String
  under_age_price : ℕ
  over_age_price : ℕ
  age_threshold : ℕ
  discount_or_commission : ℚ
  is_discount : Bool

/-- Calculates the total cost for a family's vacation package -/
def calculate_cost (agency : TravelAgency) (num_adults num_children : ℕ) (child_age : ℕ) : ℚ :=
  let base_cost := 
    if child_age < agency.age_threshold
    then agency.under_age_price * num_children + agency.over_age_price * num_adults
    else agency.over_age_price * (num_adults + num_children)
  let adjustment := base_cost * agency.discount_or_commission
  if agency.is_discount
  then base_cost - adjustment
  else base_cost + adjustment

/-- The Dorokhov family vacation problem -/
theorem dorokhov_vacation_cost : 
  let globus : TravelAgency := {
    name := "Globus",
    under_age_price := 11200,
    over_age_price := 25400,
    age_threshold := 5,
    discount_or_commission := 2 / 100,
    is_discount := true
  }
  let around_world : TravelAgency := {
    name := "Around the World",
    under_age_price := 11400,
    over_age_price := 23500,
    age_threshold := 6,
    discount_or_commission := 1 / 100,
    is_discount := false
  }
  let globus_cost := calculate_cost globus 2 1 5
  let around_world_cost := calculate_cost around_world 2 1 5
  min globus_cost around_world_cost = 58984 := by sorry

end dorokhov_vacation_cost_l2639_263965


namespace expression_evaluation_l2639_263983

theorem expression_evaluation :
  let a : ℚ := -1/3
  (3*a - 1)^2 + 3*a*(3*a + 2) = 3 := by sorry

end expression_evaluation_l2639_263983


namespace prime_relation_l2639_263926

theorem prime_relation (p q : ℕ) : 
  Nat.Prime p ∧ 
  p = Nat.minFac (Nat.minFac 2) ∧ 
  q = 13 * p + 3 ∧ 
  Nat.Prime q → 
  q = 29 := by sorry

end prime_relation_l2639_263926


namespace number_of_mappings_l2639_263911

/-- Given two finite sets A and B, where |A| = n and |B| = k, this function
    represents the number of order-preserving surjective mappings from A to B. -/
def orderPreservingSurjections (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- The sets A and B -/
def A : Set ℝ := {a | ∃ i : Fin 60, a = i}
def B : Set ℝ := {b | ∃ i : Fin 25, b = i}

/-- The mapping f from A to B -/
def f : A → B := sorry

/-- f is surjective -/
axiom f_surjective : Function.Surjective f

/-- f preserves order -/
axiom f_order_preserving :
  ∀ (a₁ a₂ : A), (a₁ : ℝ) ≤ (a₂ : ℝ) → (f a₁ : ℝ) ≥ (f a₂ : ℝ)

/-- The main theorem: The number of such mappings is C₅₉²⁴ -/
theorem number_of_mappings :
  orderPreservingSurjections 60 25 = Nat.choose 59 24 := by sorry

end number_of_mappings_l2639_263911


namespace complex_cubic_sum_ratio_l2639_263900

theorem complex_cubic_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10)
  (h_eq : 2 * ((x - y)^2 + (x - z)^2 + (y - z)^2) = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 11 / 2 := by
  sorry

end complex_cubic_sum_ratio_l2639_263900


namespace power_division_l2639_263905

theorem power_division : 2^12 / 8^3 = 8 := by sorry

end power_division_l2639_263905


namespace harmonio_theorem_l2639_263956

/-- Represents the student population at Harmonio Middle School -/
structure School where
  total : ℝ
  enjoy_singing : ℝ
  admit_liking : ℝ
  dislike_consistent : ℝ

/-- Conditions for Harmonio Middle School -/
def harmonio_conditions (s : School) : Prop :=
  s.total > 0 ∧
  s.enjoy_singing = 0.7 * s.total ∧
  s.admit_liking = 0.75 * s.enjoy_singing ∧
  s.dislike_consistent = 0.8 * (s.total - s.enjoy_singing)

/-- Theorem statement for the problem -/
theorem harmonio_theorem (s : School) (h : harmonio_conditions s) :
  let claim_dislike := s.dislike_consistent + (s.enjoy_singing - s.admit_liking)
  (s.enjoy_singing - s.admit_liking) / claim_dislike = 0.4217 := by
  sorry


end harmonio_theorem_l2639_263956


namespace vector_difference_magnitude_l2639_263980

theorem vector_difference_magnitude (a b : ℝ × ℝ × ℝ) :
  a = (2, 3, -1) → b = (-2, 1, 3) → ‖a - b‖ = 6 := by
  sorry

end vector_difference_magnitude_l2639_263980


namespace sticker_distribution_ways_l2639_263917

/-- The number of ways to distribute stickers across sheets of paper -/
def distribute_stickers (total_stickers : ℕ) (total_sheets : ℕ) : ℕ :=
  Nat.choose (total_stickers - total_sheets + total_sheets - 1) (total_sheets - 1)

/-- Theorem: There are 126 ways to distribute 10 stickers across 5 sheets -/
theorem sticker_distribution_ways :
  distribute_stickers 10 5 = 126 := by
  sorry

#eval distribute_stickers 10 5

end sticker_distribution_ways_l2639_263917


namespace marys_nickels_l2639_263933

/-- Given Mary's initial nickels and the additional nickels from her dad,
    calculate the total number of nickels Mary has now. -/
theorem marys_nickels (initial : ℕ) (additional : ℕ) : 
  initial = 7 → additional = 5 → initial + additional = 12 := by
  sorry

end marys_nickels_l2639_263933


namespace min_value_of_sum_of_squares_l2639_263958

theorem min_value_of_sum_of_squares (x y : ℝ) (h : 4 * x^2 + 4 * x * y + 7 * y^2 = 3) :
  ∃ (m : ℝ), m = 3/8 ∧ x^2 + y^2 ≥ m ∧ ∃ (x₀ y₀ : ℝ), 4 * x₀^2 + 4 * x₀ * y₀ + 7 * y₀^2 = 3 ∧ x₀^2 + y₀^2 = m :=
by sorry

end min_value_of_sum_of_squares_l2639_263958


namespace fourth_root_63504000_l2639_263907

theorem fourth_root_63504000 : 
  (63504000 : ℝ)^(1/4) = 2 * (2 : ℝ)^(1/2) * (3 : ℝ)^(1/2) * (11 : ℝ)^(1/4) * 10^(3/4) := by
  sorry

end fourth_root_63504000_l2639_263907


namespace complex_multiplication_l2639_263931

def A : ℂ := 6 - 2 * Complex.I
def M : ℂ := -3 + 4 * Complex.I
def S : ℂ := 2 * Complex.I
def P : ℂ := 3
def C : ℂ := 1 + Complex.I

theorem complex_multiplication :
  (A - M + S - P) * C = 10 + 2 * Complex.I :=
by sorry

end complex_multiplication_l2639_263931


namespace january_salary_l2639_263946

/-- Represents the monthly salary for a person -/
structure MonthlySalary where
  january : ℕ
  february : ℕ
  march : ℕ
  april : ℕ
  may : ℕ

/-- The average salary calculation is correct -/
def average_salary_correct (s : MonthlySalary) : Prop :=
  (s.january + s.february + s.march + s.april) / 4 = 8000 ∧
  (s.february + s.march + s.april + s.may) / 4 = 9500

/-- The salary for May is 6500 -/
def may_salary_correct (s : MonthlySalary) : Prop :=
  s.may = 6500

/-- The theorem stating that given the conditions, the salary for January is 500 -/
theorem january_salary (s : MonthlySalary) 
  (h1 : average_salary_correct s) 
  (h2 : may_salary_correct s) : 
  s.january = 500 := by
  sorry

end january_salary_l2639_263946


namespace pythagorean_triple_5_12_13_l2639_263968

theorem pythagorean_triple_5_12_13 : 5^2 + 12^2 = 13^2 := by
  sorry

end pythagorean_triple_5_12_13_l2639_263968


namespace quadratic_factorization_l2639_263914

/-- A quadratic expression can be factored completely if and only if its discriminant is a perfect square. -/
def is_factorable (a b c : ℝ) : Prop :=
  ∃ k : ℤ, (b^2 - 4*a*c : ℝ) = (k : ℝ)^2

theorem quadratic_factorization (m : ℝ) :
  (is_factorable 1 (3 - m) 25) → (m = -7 ∨ m = 13) :=
by sorry

end quadratic_factorization_l2639_263914


namespace bacteria_exceeds_200_on_day_4_l2639_263988

-- Define the bacteria population function
def bacteria_population (initial_population : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial_population * growth_factor ^ days

-- Theorem statement
theorem bacteria_exceeds_200_on_day_4 :
  let initial_population := 5
  let growth_factor := 3
  let threshold := 200
  (∀ d : ℕ, d < 4 → bacteria_population initial_population growth_factor d ≤ threshold) ∧
  (bacteria_population initial_population growth_factor 4 > threshold) :=
by sorry

end bacteria_exceeds_200_on_day_4_l2639_263988


namespace derivative_f_l2639_263929

noncomputable def f (x : ℝ) : ℝ := (1 / (4 * Real.sqrt 5)) * Real.log ((2 + Real.sqrt 5 * Real.tanh x) / (2 - Real.sqrt 5 * Real.tanh x))

theorem derivative_f (x : ℝ) : 
  deriv f x = 1 / (4 - Real.sinh x ^ 2) :=
sorry

end derivative_f_l2639_263929


namespace consecutive_numbers_sum_product_l2639_263997

/-- Given four consecutive natural numbers x-1, x, x+1, and x+2, 
    if the product of their sum and the sum of their squares 
    equals three times the sum of their cubes, then x = 5. -/
theorem consecutive_numbers_sum_product (x : ℕ) : 
  (x - 1 + x + (x + 1) + (x + 2)) * 
  ((x - 1)^2 + x^2 + (x + 1)^2 + (x + 2)^2) = 
  3 * ((x - 1)^3 + x^3 + (x + 1)^3 + (x + 2)^3) → 
  x = 5 := by
  sorry

#check consecutive_numbers_sum_product

end consecutive_numbers_sum_product_l2639_263997


namespace probability_at_least_half_even_dice_l2639_263921

theorem probability_at_least_half_even_dice (dice : Nat) (p_even : ℝ) :
  dice = 4 →
  p_even = 1/2 →
  let p_two_even := Nat.choose dice 2 * p_even^2 * (1 - p_even)^2
  let p_three_even := Nat.choose dice 3 * p_even^3 * (1 - p_even)
  let p_four_even := p_even^4
  p_two_even + p_three_even + p_four_even = 11/16 := by
sorry

end probability_at_least_half_even_dice_l2639_263921


namespace simplify_fraction_l2639_263939

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end simplify_fraction_l2639_263939


namespace geometric_sequence_sum_l2639_263916

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 4 + 2 * a 8 * a 5 + a 9 * a 7 = 36 →
  a 5 + a 8 = 6 := by
sorry

end geometric_sequence_sum_l2639_263916


namespace a_bounds_l2639_263906

def a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => a n + (1 / (n+1)^2) * (a n)^2

theorem a_bounds (n : ℕ) : (n+1)/(n+2) < a n ∧ a n < n+1 := by
  sorry

end a_bounds_l2639_263906


namespace cubic_inequality_solution_l2639_263970

theorem cubic_inequality_solution (x : ℝ) :
  x^3 + x^2 - 7*x + 6 < 0 ↔ -3 < x ∧ x < 1 ∨ 1 < x ∧ x < 2 := by sorry

end cubic_inequality_solution_l2639_263970


namespace f_properties_l2639_263927

-- Define the function f
def f (a b x : ℝ) : ℝ := 3 * a * x^2 - 2 * (a + b) * x + b

-- State the theorem
theorem f_properties (a b : ℝ) (ha : a > 0) :
  -- Part I
  (b = 1/2 ∧ 
   ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 ∧ 
   f a (1/2) x₁ = |x₁ - 1/2| ∧ 
   f a (1/2) x₂ = |x₂ - 1/2|) →
  a ≥ 1 ∧
  -- Part II
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f a b 0| ≤ 2 ∧ |f a b 1| ≤ 2) →
  ∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f a b x| ≤ M :=
by
  sorry

end f_properties_l2639_263927


namespace E_parity_l2639_263982

def E : ℕ → ℤ
  | 0 => 2
  | 1 => 3
  | 2 => 4
  | n + 3 => E (n + 2) + 2 * E (n + 1) - E n

theorem E_parity : (E 10 % 2 = 1) ∧ (E 11 % 2 = 0) ∧ (E 12 % 2 = 1) := by
  sorry

end E_parity_l2639_263982


namespace only_two_digit_divisor_with_remainder_four_l2639_263973

theorem only_two_digit_divisor_with_remainder_four (d : ℕ) : 
  d > 0 ∧ d ≥ 10 ∧ d ≤ 99 ∧ 143 % d = 4 → d = 139 :=
by sorry

end only_two_digit_divisor_with_remainder_four_l2639_263973


namespace josh_spending_l2639_263915

def initial_amount : ℚ := 9
def drink_cost : ℚ := 1.75
def final_amount : ℚ := 6

theorem josh_spending (amount_spent_after_drink : ℚ) : 
  initial_amount - drink_cost - amount_spent_after_drink = final_amount → 
  amount_spent_after_drink = 1.25 := by
sorry

end josh_spending_l2639_263915


namespace complex_magnitude_problem_l2639_263932

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs z = 1)
  (h2 : Complex.abs w = 2)
  (h3 : Complex.arg z = Real.pi / 2)
  (h4 : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = (Real.sqrt (5 + 2 * Real.sqrt 2)) / (2 * Real.sqrt 2) := by
  sorry

end complex_magnitude_problem_l2639_263932


namespace unique_solution_l2639_263987

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x, x > 0 → f x > 0}

/-- The functional equation that f must satisfy -/
def SatisfiesEquation (f : PositiveRealFunction) : Prop :=
  ∀ x y, x > 0 → y > 0 → f.val (x + f.val y) = f.val (x + y) + f.val y

/-- The theorem stating that the only solution is f(x) = 2x -/
theorem unique_solution (f : PositiveRealFunction) (h : SatisfiesEquation f) :
  ∀ x, x > 0 → f.val x = 2 * x :=
sorry

end unique_solution_l2639_263987


namespace percentage_increase_l2639_263945

theorem percentage_increase (original : ℝ) (new : ℝ) : 
  original = 50 → new = 80 → (new - original) / original * 100 = 60 := by
  sorry

end percentage_increase_l2639_263945


namespace largest_c_for_g_range_two_l2639_263992

/-- The function g(x) defined as x^2 - 5x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 - 5*x + c

/-- Theorem stating that the largest value of c such that 2 is in the range of g(x) is 33/4 -/
theorem largest_c_for_g_range_two :
  ∃ (c_max : ℝ), c_max = 33/4 ∧
  (∀ c : ℝ, (∃ x : ℝ, g c x = 2) → c ≤ c_max) ∧
  (∃ x : ℝ, g c_max x = 2) :=
sorry

end largest_c_for_g_range_two_l2639_263992


namespace product_evaluation_l2639_263966

theorem product_evaluation (a : ℕ) (h : a = 7) : 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 5040 := by
  sorry

end product_evaluation_l2639_263966


namespace number_pairs_sum_800_or_400_l2639_263977

theorem number_pairs_sum_800_or_400 (x y : ℤ) (A : ℤ) (h1 : x ≥ y) (h2 : A = 800 ∨ A = 400) 
  (h3 : (x + y) + (x - y) + x * y + x / y = A) :
  (A = 800 ∧ ((x = 38 ∧ y = 19) ∨ (x = -42 ∧ y = -21) ∨ (x = 36 ∧ y = 9) ∨ 
              (x = -44 ∧ y = -11) ∨ (x = 40 ∧ y = 4) ∨ (x = -60 ∧ y = -6) ∨ 
              (x = 20 ∧ y = 1) ∨ (x = -60 ∧ y = -3))) ∨
  (A = 400 ∧ ((x = 19 ∧ y = 19) ∨ (x = -21 ∧ y = -21) ∨ (x = 36 ∧ y = 9) ∨ 
              (x = -44 ∧ y = -11) ∨ (x = 64 ∧ y = 4) ∨ (x = -96 ∧ y = -6) ∨ 
              (x = 75 ∧ y = 3) ∨ (x = -125 ∧ y = -5) ∨ (x = 100 ∧ y = 1) ∨ 
              (x = -300 ∧ y = -3))) :=
by sorry

end number_pairs_sum_800_or_400_l2639_263977


namespace coefficient_of_x8_in_expansion_l2639_263953

/-- The coefficient of x^8 in the expansion of (1 + 3x - 2x^2)^5 is -720 -/
theorem coefficient_of_x8_in_expansion : 
  let p : Polynomial ℤ := 1 + 3 * X - 2 * X^2
  let coeff := (p^5).coeff 8
  coeff = -720 := by sorry

end coefficient_of_x8_in_expansion_l2639_263953


namespace calculation_result_l2639_263935

theorem calculation_result : 12.05 * 5.4 + 0.6 = 65.67 := by
  sorry

end calculation_result_l2639_263935


namespace difference_c_minus_a_l2639_263955

/-- Given that the average of a and b is 40, and the average of b and c is 60,
    prove that the difference between c and a is 40. -/
theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 40)
  (h2 : (b + c) / 2 = 60) :
  c - a = 40 := by
  sorry

end difference_c_minus_a_l2639_263955


namespace min_value_a_l2639_263995

theorem min_value_a (a b c d : ℕ+) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 2004)
  (h3 : a^2 - b^2 + c^2 - d^2 = 2004) :
  ∀ x : ℕ+, (x > b ∧ b > c ∧ c > d ∧ 
             x + b + c + d = 2004 ∧ 
             x^2 - b^2 + c^2 - d^2 = 2004) → 
    x ≥ 503 :=
by sorry

end min_value_a_l2639_263995


namespace triangle_inequality_l2639_263912

theorem triangle_inequality (a b c : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_triangle : a^2 + b^2 ≥ c^2 ∧ b^2 + c^2 ≥ a^2 ∧ c^2 + a^2 ≥ b^2) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) := by
  sorry

end triangle_inequality_l2639_263912


namespace xiaotong_message_forwarding_l2639_263901

theorem xiaotong_message_forwarding :
  ∃ (x : ℕ), x > 0 ∧ 1 + x + x^2 = 91 :=
by sorry

end xiaotong_message_forwarding_l2639_263901


namespace am_gm_inequality_and_equality_condition_l2639_263957

theorem am_gm_inequality_and_equality_condition (x : ℝ) (h : x > 0) :
  x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by sorry

end am_gm_inequality_and_equality_condition_l2639_263957


namespace roberts_journey_distance_l2639_263942

/-- Represents the time in hours for each leg of Robert's journey -/
structure JourneyTimes where
  ab : ℝ
  bc : ℝ
  ca : ℝ

/-- Calculates the total distance of Robert's journey -/
def totalDistance (times : JourneyTimes) : ℝ :=
  let adjustedTime := times.ab + times.bc + times.ca - 1.5
  90 * adjustedTime

/-- Theorem stating that the total distance of Robert's journey is 1305 miles -/
theorem roberts_journey_distance (times : JourneyTimes) 
  (h1 : times.ab = 6)
  (h2 : times.bc = 5.5)
  (h3 : times.ca = 4.5) : 
  totalDistance times = 1305 := by
  sorry

#eval totalDistance { ab := 6, bc := 5.5, ca := 4.5 }

end roberts_journey_distance_l2639_263942


namespace room_dimension_proof_l2639_263950

/-- Proves that given the room dimensions and whitewashing costs, the unknown dimension is 15 feet -/
theorem room_dimension_proof (x : ℝ) : 
  let room_length : ℝ := 25
  let room_height : ℝ := 12
  let door_area : ℝ := 6 * 3
  let window_area : ℝ := 4 * 3
  let num_windows : ℕ := 3
  let whitewash_cost_per_sqft : ℝ := 3
  let total_cost : ℝ := 2718
  let wall_area : ℝ := 2 * (room_length * room_height) + 2 * (x * room_height)
  let non_whitewash_area : ℝ := door_area + num_windows * window_area
  let whitewash_area : ℝ := wall_area - non_whitewash_area
  whitewash_area * whitewash_cost_per_sqft = total_cost → x = 15 :=
by
  sorry


end room_dimension_proof_l2639_263950


namespace census_contradiction_l2639_263990

/-- Represents a family in the house -/
structure Family where
  boys : ℕ
  girls : ℕ

/-- The census data for the house -/
structure CensusData where
  families : List Family

/-- Conditions from the problem -/
def ValidCensus (data : CensusData) : Prop :=
  ∀ f ∈ data.families,
    (f.boys > 0 → f.girls > 0) ∧  -- Every boy has a sister
    (f.boys + f.girls > 0)  -- No families without children

/-- Total number of boys in the house -/
def TotalBoys (data : CensusData) : ℕ :=
  (data.families.map (λ f => f.boys)).sum

/-- Total number of girls in the house -/
def TotalGirls (data : CensusData) : ℕ :=
  (data.families.map (λ f => f.girls)).sum

/-- Total number of children in the house -/
def TotalChildren (data : CensusData) : ℕ :=
  TotalBoys data + TotalGirls data

/-- Total number of adults in the house -/
def TotalAdults (data : CensusData) : ℕ :=
  2 * data.families.length

/-- The main theorem to prove -/
theorem census_contradiction (data : CensusData) 
  (h_valid : ValidCensus data)
  (h_more_boys : TotalBoys data > TotalGirls data) :
  TotalChildren data > TotalAdults data :=
sorry

end census_contradiction_l2639_263990


namespace lamp_probability_l2639_263972

theorem lamp_probability : Real → Prop :=
  fun p =>
    let total_length : Real := 6
    let min_distance : Real := 2
    p = (total_length - 2 * min_distance) / total_length

#check lamp_probability (1/3)

end lamp_probability_l2639_263972


namespace line_segment_endpoint_l2639_263954

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  ((x - 2)^2 + 3^2)^(1/2) = 6 → 
  x = 2 + 3 * Real.sqrt 3 :=
by sorry

end line_segment_endpoint_l2639_263954


namespace tank_insulation_cost_l2639_263925

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the cost of insulating a rectangular tank -/
def insulationCost (l w h costPerSqFt : ℝ) : ℝ :=
  surfaceArea l w h * costPerSqFt

theorem tank_insulation_cost :
  let l : ℝ := 6
  let w : ℝ := 3
  let h : ℝ := 2
  let costPerSqFt : ℝ := 20
  insulationCost l w h costPerSqFt = 1440 := by
  sorry

#eval insulationCost 6 3 2 20

end tank_insulation_cost_l2639_263925


namespace right_triangle_with_hypotenuse_65_l2639_263952

theorem right_triangle_with_hypotenuse_65 :
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25 :=          -- Shorter leg length
by
  sorry

end right_triangle_with_hypotenuse_65_l2639_263952


namespace problem_solution_l2639_263998

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - x + 2

def tangent_line (x y : ℝ) : Prop := x + y - 2 = 0

theorem problem_solution (m : ℝ) (h_m : m > 0) :
  (∀ x : ℝ, x > 0 → x < 1/Real.exp 1 → (deriv f) x < 0) ∧
  (m > 1/Real.exp 1 → ∀ x : ℝ, 1/Real.exp 1 < x → x < m → (deriv f) x > 0) ∧
  (∀ x : ℝ, x > 0 → 2 * f x ≤ (deriv (g (-2))) x + 2) ∧
  (∀ a : ℝ, a ≥ -2 → ∀ x : ℝ, x > 0 → 2 * f x ≤ (deriv (g a)) x + 2) ∧
  tangent_line (0 : ℝ) (g 1 0) :=
sorry

end problem_solution_l2639_263998


namespace power_relation_l2639_263984

theorem power_relation (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 9) : x^(3*a - b) = 8/9 := by
  sorry

end power_relation_l2639_263984


namespace sourball_theorem_l2639_263904

def sourball_problem (nellie jacob lana bucket_total : ℕ) : Prop :=
  nellie = 12 ∧
  jacob = nellie / 2 ∧
  lana = jacob - 3 ∧
  bucket_total = 30 ∧
  let total_eaten := nellie + jacob + lana
  let remaining := bucket_total - total_eaten
  remaining / 3 = 3

theorem sourball_theorem :
  ∃ (nellie jacob lana bucket_total : ℕ),
    sourball_problem nellie jacob lana bucket_total :=
by
  sorry

end sourball_theorem_l2639_263904


namespace range_of_ab_plus_a_plus_b_l2639_263947

def f (x : ℝ) := |x^2 + 2*x - 1|

theorem range_of_ab_plus_a_plus_b 
  (a b : ℝ) 
  (h1 : a < b) 
  (h2 : b < -1) 
  (h3 : f a = f b) :
  ∀ y : ℝ, (∃ x : ℝ, a < x ∧ x < b ∧ y = a*b + a + b) → -1 < y ∧ y < 1 :=
by sorry

end range_of_ab_plus_a_plus_b_l2639_263947


namespace largest_valid_domain_l2639_263943

def is_valid_domain (S : Set ℝ) : Prop :=
  ∃ g : ℝ → ℝ, 
    (∀ x ∈ S, (1 / x) ∈ S) ∧ 
    (∀ x ∈ S, g x + g (1 / x) = x^2)

theorem largest_valid_domain : 
  is_valid_domain {-1, 1} ∧ 
  ∀ S : Set ℝ, is_valid_domain S → S ⊆ {-1, 1} :=
sorry

end largest_valid_domain_l2639_263943


namespace regular_polygon_144_degree_angles_has_10_sides_l2639_263941

/-- A regular polygon with interior angles of 144 degrees has 10 sides. -/
theorem regular_polygon_144_degree_angles_has_10_sides :
  ∀ n : ℕ,
  n > 2 →
  (180 * (n - 2) : ℝ) = 144 * n →
  n = 10 :=
by sorry

end regular_polygon_144_degree_angles_has_10_sides_l2639_263941


namespace odd_log_properties_l2639_263948

noncomputable section

-- Define the logarithm function with base a
def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := log a x

-- Theorem statement
theorem odd_log_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- Part 1: The value of m
  (∀ x > 0, log a x + log a (-x) = 0) →
  -- Part 2: The derivative of f
  (∀ x ≠ 0, deriv (f a) x = (Real.log a)⁻¹ / x) ∧
  -- Part 3: The value of a given the range condition
  (∀ x ∈ Set.Ioo 1 (a - 2), f a x ∈ Set.Ioi 1) →
  a = 2 + Real.sqrt 5 := by
sorry

end

end odd_log_properties_l2639_263948


namespace problem_solution_l2639_263919

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 6)
  (h_eq2 : y + 1 / x = 15) :
  z + 1 / y = 23 / 89 := by
sorry

end problem_solution_l2639_263919


namespace hyperbola_eccentricity_l2639_263994

/-- Given a hyperbola mx^2 + 5y^2 = 5m with eccentricity e = 2, prove that m = -15 -/
theorem hyperbola_eccentricity (m : ℝ) : 
  (∃ (x y : ℝ), m*x^2 + 5*y^2 = 5*m) → -- Hyperbola equation
  (∃ (e : ℝ), e = 2 ∧ e^2 = 1 - m/5) → -- Eccentricity definition
  m = -15 := by
sorry

end hyperbola_eccentricity_l2639_263994


namespace fractional_exponent_simplification_l2639_263913

theorem fractional_exponent_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a ^ (2 * b ^ (1/4))) / (a ^ (1/2) * b ^ (1/4)) = a ^ (3/2) := by
  sorry

end fractional_exponent_simplification_l2639_263913


namespace pharmacist_weights_impossibility_l2639_263936

theorem pharmacist_weights_impossibility :
  ¬∃ (w₁ w₂ w₃ : ℝ),
    w₁ < 90 ∧ w₂ < 90 ∧ w₃ < 90 ∧
    w₁ + w₂ + w₃ = 100 ∧
    w₁ + 2*w₂ + w₃ = 101 ∧
    w₁ + w₂ + 2*w₃ = 102 :=
sorry

end pharmacist_weights_impossibility_l2639_263936


namespace coefficient_x_cubed_in_product_l2639_263986

/-- The coefficient of x^3 in the product of two specific polynomials -/
theorem coefficient_x_cubed_in_product : ∃ (p q : Polynomial ℤ),
  p = 3 * X^3 + 2 * X^2 + 4 * X + 5 ∧
  q = 4 * X^3 + 6 * X^2 + 5 * X + 2 ∧
  (p * q).coeff 3 = 10 := by
  sorry

end coefficient_x_cubed_in_product_l2639_263986


namespace special_fraction_equality_l2639_263930

theorem special_fraction_equality (a b : ℝ) 
  (h : a / (1 + a) + b / (1 + b) = 1) : 
  a / (1 + b^2) - b / (1 + a^2) = a - b := by
  sorry

end special_fraction_equality_l2639_263930


namespace correct_allocation_count_l2639_263971

def num_volunteers : ℕ := 4
def num_events : ℕ := 3

def allocation_schemes (n_volunteers : ℕ) (n_events : ℕ) : ℕ :=
  if n_volunteers < n_events then 0
  else (n_events.factorial * n_events^(n_volunteers - n_events))

theorem correct_allocation_count :
  allocation_schemes num_volunteers num_events = 18 :=
sorry

end correct_allocation_count_l2639_263971


namespace arithmetic_sequence_property_l2639_263999

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h1 : a 3 + a 8 = 10) : 3 * a 5 + a 7 = 20 := by
  sorry

end arithmetic_sequence_property_l2639_263999


namespace arithmetic_sequence_68th_term_l2639_263981

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first_term : ℕ
  term_21 : ℕ

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℕ :=
  seq.first_term + (n - 1) * ((seq.term_21 - seq.first_term) / 20)

/-- Theorem stating that the 68th term of the given arithmetic sequence is 204 -/
theorem arithmetic_sequence_68th_term
  (seq : ArithmeticSequence)
  (h1 : seq.first_term = 3)
  (h2 : seq.term_21 = 63) :
  nth_term seq 68 = 204 := by
  sorry


end arithmetic_sequence_68th_term_l2639_263981


namespace absolute_value_equation_roots_l2639_263975

theorem absolute_value_equation_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ ≠ x₂) ∧ 
  (|x₁|^2 + |x₁| - 12 = 0) ∧ 
  (|x₂|^2 + |x₂| - 12 = 0) ∧
  (x₁ + x₂ = 0) ∧
  (∀ x : ℝ, |x|^2 + |x| - 12 = 0 → (x = x₁ ∨ x = x₂)) := by
  sorry

end absolute_value_equation_roots_l2639_263975
