import Mathlib

namespace NUMINAMATH_CALUDE_grid_tiling_condition_l1203_120308

/-- Represents a tile type that can cover a 2x2 or larger area of a grid -/
structure Tile :=
  (width : ℕ)
  (height : ℕ)
  (valid : width ≥ 2 ∧ height ≥ 2)

/-- Represents the set of 6 available tile types -/
def TileSet : Set Tile := sorry

/-- Predicate to check if a grid can be tiled with the given tile set -/
def canBeTiled (m n : ℕ) (tiles : Set Tile) : Prop := sorry

/-- Main theorem: A rectangular grid can be tiled iff 4 divides m or n, and neither is 1 -/
theorem grid_tiling_condition (m n : ℕ) :
  canBeTiled m n TileSet ↔ (4 ∣ m ∨ 4 ∣ n) ∧ m ≠ 1 ∧ n ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_grid_tiling_condition_l1203_120308


namespace NUMINAMATH_CALUDE_special_line_equation_l1203_120351

/-- A line passing through (6, -2) with x-intercept 1 greater than y-intercept -/
structure SpecialLine where
  /-- The equation of the line in the form ax + by + c = 0 -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- The line passes through (6, -2) -/
  point_condition : a * 6 + b * (-2) + c = 0
  /-- The x-intercept is 1 greater than the y-intercept -/
  intercept_condition : -c/a = -c/b + 1

/-- The equation of the special line is either x + 2y - 2 = 0 or 2x + 3y - 6 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.a = 1 ∧ l.b = 2 ∧ l.c = -2) ∨ (l.a = 2 ∧ l.b = 3 ∧ l.c = -6) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l1203_120351


namespace NUMINAMATH_CALUDE_daisy_toys_count_l1203_120310

def monday_toys : ℕ := 5
def tuesday_toys_left : ℕ := 3
def tuesday_toys_bought : ℕ := 3
def wednesday_toys_bought : ℕ := 5

def total_toys : ℕ := monday_toys + (monday_toys - tuesday_toys_left) + tuesday_toys_bought + wednesday_toys_bought

theorem daisy_toys_count : total_toys = 15 := by sorry

end NUMINAMATH_CALUDE_daisy_toys_count_l1203_120310


namespace NUMINAMATH_CALUDE_laborer_wage_calculation_l1203_120391

/-- Proves that the daily wage for a laborer is 2.00 rupees given the problem conditions --/
theorem laborer_wage_calculation (total_days : ℕ) (absent_days : ℕ) (fine_per_day : ℚ) (total_received : ℚ) :
  total_days = 25 →
  absent_days = 5 →
  fine_per_day = 1/2 →
  total_received = 75/2 →
  ∃ (daily_wage : ℚ), 
    daily_wage * (total_days - absent_days : ℚ) - (fine_per_day * absent_days) = total_received ∧
    daily_wage = 2 := by
  sorry

#eval (2 : ℚ)

end NUMINAMATH_CALUDE_laborer_wage_calculation_l1203_120391


namespace NUMINAMATH_CALUDE_min_balls_correct_l1203_120367

/-- The minimum number of balls that satisfies the given conditions -/
def min_balls : ℕ := 24

/-- The number of white balls -/
def white_balls : ℕ := min_balls / 3

/-- The number of black balls -/
def black_balls : ℕ := 2 * white_balls

/-- The number of pairs of different colors -/
def different_color_pairs : ℕ := min_balls / 4

/-- The number of pairs of the same color -/
def same_color_pairs : ℕ := 3 * different_color_pairs

theorem min_balls_correct :
  (black_balls = 2 * white_balls) ∧
  (black_balls + white_balls = min_balls) ∧
  (same_color_pairs = 3 * different_color_pairs) ∧
  (same_color_pairs + different_color_pairs = min_balls) ∧
  (∀ n : ℕ, n < min_balls → ¬(
    (2 * (n / 3) = n - (n / 3)) ∧
    (3 * (n / 4) = n - (n / 4))
  )) := by
  sorry

#eval min_balls

end NUMINAMATH_CALUDE_min_balls_correct_l1203_120367


namespace NUMINAMATH_CALUDE_product_not_divisible_by_sum_l1203_120342

theorem product_not_divisible_by_sum (a b : ℕ) (h : a + b = 201) : ¬(201 ∣ (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_product_not_divisible_by_sum_l1203_120342


namespace NUMINAMATH_CALUDE_nPointedStar_interiorAngleSum_l1203_120390

/-- Represents an n-pointed star formed from an n-sided convex polygon -/
structure NPointedStar where
  n : ℕ
  h_n : n ≥ 6

/-- The sum of interior angles at the vertices of an n-pointed star -/
def interiorAngleSum (star : NPointedStar) : ℝ :=
  180 * (star.n - 2)

/-- Theorem: The sum of interior angles at the vertices of an n-pointed star
    formed by extending every third side of an n-sided convex polygon (n ≥ 6)
    is equal to 180°(n-2) -/
theorem nPointedStar_interiorAngleSum (star : NPointedStar) :
  interiorAngleSum star = 180 * (star.n - 2) := by
  sorry

end NUMINAMATH_CALUDE_nPointedStar_interiorAngleSum_l1203_120390


namespace NUMINAMATH_CALUDE_cards_taken_away_l1203_120327

theorem cards_taken_away (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 76)
  (h2 : final_cards = 17) :
  initial_cards - final_cards = 59 := by
  sorry

end NUMINAMATH_CALUDE_cards_taken_away_l1203_120327


namespace NUMINAMATH_CALUDE_complex_number_calculation_l1203_120325

theorem complex_number_calculation : 
  (Complex.mk 1 3) * (Complex.mk 2 (-4)) + (Complex.mk 2 5) * (Complex.mk 2 (-1)) = Complex.mk 13 10 := by
sorry

end NUMINAMATH_CALUDE_complex_number_calculation_l1203_120325


namespace NUMINAMATH_CALUDE_cosine_shift_l1203_120381

theorem cosine_shift (x : ℝ) :
  let f (x : ℝ) := 3 * Real.cos (1/2 * x - π/3)
  let period := 4 * π
  let shift := period / 8
  let g (x : ℝ) := f (x + shift)
  g x = 3 * Real.cos (1/2 * x - π/12) := by
  sorry

end NUMINAMATH_CALUDE_cosine_shift_l1203_120381


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1203_120302

theorem gcd_of_specific_numbers : Nat.gcd 55555555 111111111 = 11111111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1203_120302


namespace NUMINAMATH_CALUDE_sum_of_absolute_values_zero_l1203_120345

theorem sum_of_absolute_values_zero (a b : ℝ) : 
  |a + 2| + |b - 7| = 0 → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_values_zero_l1203_120345


namespace NUMINAMATH_CALUDE_distinct_cube_constructions_l1203_120397

/-- The group of rotational symmetries of a cube -/
def CubeRotationGroup : Type := Unit

/-- The number of elements in the cube rotation group -/
def CubeRotationGroup.order : ℕ := 24

/-- The number of ways to place 5 white cubes in a 2x2x2 cube -/
def WhiteCubePlacements : ℕ := Nat.choose 8 5

/-- The number of fixed points under the identity rotation -/
def FixedPointsUnderIdentity : ℕ := WhiteCubePlacements

/-- The number of fixed points under all non-identity rotations -/
def FixedPointsUnderNonIdentity : ℕ := 0

/-- The total number of fixed points under all rotations -/
def TotalFixedPoints : ℕ := FixedPointsUnderIdentity + 23 * FixedPointsUnderNonIdentity

theorem distinct_cube_constructions :
  (TotalFixedPoints : ℚ) / CubeRotationGroup.order = 7 / 3 := by sorry

end NUMINAMATH_CALUDE_distinct_cube_constructions_l1203_120397


namespace NUMINAMATH_CALUDE_rational_expression_value_l1203_120368

theorem rational_expression_value (a b c d m : ℚ) : 
  a ≠ 0 ∧ 
  a + b = 0 ∧ 
  c * d = 1 ∧ 
  (m = -5 ∨ m = 1) → 
  |m| - a/b + (a+b)/2020 - c*d = 1 ∨ |m| - a/b + (a+b)/2020 - c*d = 5 :=
by sorry

end NUMINAMATH_CALUDE_rational_expression_value_l1203_120368


namespace NUMINAMATH_CALUDE_playstation_value_l1203_120349

theorem playstation_value (computer_cost accessories_cost out_of_pocket : ℝ) 
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : out_of_pocket = 580) : 
  ∃ (playstation_value : ℝ), 
    playstation_value = 400 ∧ 
    computer_cost + accessories_cost = out_of_pocket + playstation_value * 0.8 := by
  sorry

end NUMINAMATH_CALUDE_playstation_value_l1203_120349


namespace NUMINAMATH_CALUDE_one_greater_than_negative_two_l1203_120330

theorem one_greater_than_negative_two : 1 > -2 := by
  sorry

end NUMINAMATH_CALUDE_one_greater_than_negative_two_l1203_120330


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1203_120378

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^3 + 7) * (3 * x + 4) = 15 * x^4 + 20 * x^3 + 21 * x + 28 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1203_120378


namespace NUMINAMATH_CALUDE_final_deficit_is_twelve_l1203_120311

/-- Calculates the final score difference for Liz's basketball game --/
def final_score_difference (initial_deficit : ℕ) 
  (liz_free_throws liz_threes liz_jumps liz_and_ones : ℕ)
  (taylor_threes taylor_jumps : ℕ)
  (opp1_threes : ℕ)
  (opp2_jumps opp2_free_throws : ℕ)
  (opp3_jumps opp3_threes : ℕ) : ℤ :=
  let liz_score := liz_free_throws + 3 * liz_threes + 2 * liz_jumps + 3 * liz_and_ones
  let taylor_score := 3 * taylor_threes + 2 * taylor_jumps
  let opp1_score := 3 * opp1_threes
  let opp2_score := 2 * opp2_jumps + opp2_free_throws
  let opp3_score := 2 * opp3_jumps + 3 * opp3_threes
  let team_score_diff := (liz_score + taylor_score) - (opp1_score + opp2_score + opp3_score)
  initial_deficit - team_score_diff

theorem final_deficit_is_twelve :
  final_score_difference 25 5 4 5 1 2 3 4 4 2 2 1 = 12 := by
  sorry

end NUMINAMATH_CALUDE_final_deficit_is_twelve_l1203_120311


namespace NUMINAMATH_CALUDE_basketball_game_score_l1203_120369

/-- Represents the score of a team in a basketball game -/
structure Score :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if a sequence is geometric with common ratio r -/
def isGeometric (s : Score) (r : ℚ) : Prop :=
  s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if a sequence is arithmetic with common difference d -/
def isArithmetic (s : Score) (d : ℕ) : Prop :=
  s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- The main theorem -/
theorem basketball_game_score 
  (sharks lions : Score) 
  (r : ℚ) 
  (d : ℕ) : 
  sharks.q1 = lions.q1 →  -- Tied at first quarter
  isGeometric sharks r →  -- Sharks scored in geometric sequence
  isArithmetic lions d →  -- Lions scored in arithmetic sequence
  (sharks.q1 + sharks.q2 + sharks.q3 + sharks.q4) = 
    (lions.q1 + lions.q2 + lions.q3 + lions.q4 + 2) →  -- Sharks won by 2 points
  sharks.q1 + sharks.q2 + sharks.q3 + sharks.q4 ≤ 120 →  -- Sharks' total ≤ 120
  lions.q1 + lions.q2 + lions.q3 + lions.q4 ≤ 120 →  -- Lions' total ≤ 120
  sharks.q1 + sharks.q2 + lions.q1 + lions.q2 = 45  -- First half total is 45
  := by sorry

end NUMINAMATH_CALUDE_basketball_game_score_l1203_120369


namespace NUMINAMATH_CALUDE_nested_root_simplification_l1203_120389

theorem nested_root_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (y * Real.sqrt (y^3 * Real.sqrt (y^5))) = (y^15)^(1/8) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l1203_120389


namespace NUMINAMATH_CALUDE_bookshelf_selections_l1203_120383

/-- Represents a bookshelf with three layers -/
structure Bookshelf :=
  (layer1 : ℕ)
  (layer2 : ℕ)
  (layer3 : ℕ)

/-- The total number of books in the bookshelf -/
def total_books (b : Bookshelf) : ℕ :=
  b.layer1 + b.layer2 + b.layer3

/-- The number of ways to select one book from each layer -/
def ways_to_select_from_each_layer (b : Bookshelf) : ℕ :=
  b.layer1 * b.layer2 * b.layer3

/-- Our specific bookshelf instance -/
def our_bookshelf : Bookshelf :=
  ⟨6, 5, 4⟩

theorem bookshelf_selections (b : Bookshelf) :
  (total_books b = 15) ∧
  (ways_to_select_from_each_layer b = 120) :=
sorry

end NUMINAMATH_CALUDE_bookshelf_selections_l1203_120383


namespace NUMINAMATH_CALUDE_weight_of_A_l1203_120337

theorem weight_of_A (A B C D : ℝ) : 
  (A + B + C) / 3 = 84 →
  (A + B + C + D) / 4 = 80 →
  (B + C + D + (D + 8)) / 4 = 79 →
  A = 80 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_A_l1203_120337


namespace NUMINAMATH_CALUDE_malvina_money_l1203_120365

theorem malvina_money (m n : ℕ) : 
  m + n < 40 →
  n < 8 * m →
  n ≥ 4 * m + 15 →
  n = 31 :=
by sorry

end NUMINAMATH_CALUDE_malvina_money_l1203_120365


namespace NUMINAMATH_CALUDE_future_age_comparison_l1203_120304

/-- Represents the age difference between Martha and Ellen in years -/
def AgeDifference : ℕ → Prop :=
  fun x => 32 = 2 * (10 + x)

/-- Proves that the number of years into the future when Martha's age is twice Ellen's age is 6 -/
theorem future_age_comparison : ∃ (x : ℕ), AgeDifference x ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_future_age_comparison_l1203_120304


namespace NUMINAMATH_CALUDE_height_percentage_difference_l1203_120373

theorem height_percentage_difference (height_A height_B : ℝ) :
  height_B = height_A * (1 + 0.42857142857142854) →
  (height_B - height_A) / height_B * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_height_percentage_difference_l1203_120373


namespace NUMINAMATH_CALUDE_cubic_integer_bound_l1203_120363

theorem cubic_integer_bound (a b c d : ℝ) (ha : a > 4/3) :
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ |a * x^3 + b * x^2 + c * x + d| ≤ 1) ∧ Finset.card S ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_integer_bound_l1203_120363


namespace NUMINAMATH_CALUDE_welders_left_correct_l1203_120339

/-- The number of welders who left after the first day -/
def welders_who_left : ℕ := 9

/-- The initial number of welders -/
def initial_welders : ℕ := 12

/-- The number of days to complete the order with all welders -/
def initial_days : ℕ := 3

/-- The number of additional days needed after some welders left -/
def additional_days : ℕ := 8

theorem welders_left_correct :
  ∃ (r : ℝ), r > 0 ∧
  initial_welders * r * initial_days = (initial_welders - welders_who_left) * r * (1 + additional_days) :=
by sorry

end NUMINAMATH_CALUDE_welders_left_correct_l1203_120339


namespace NUMINAMATH_CALUDE_sqrt_equality_l1203_120301

theorem sqrt_equality (n k : ℕ) (h : n + 1 = k^2) :
  Real.sqrt (k + k / n) = k * Real.sqrt (k / n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_l1203_120301


namespace NUMINAMATH_CALUDE_farmer_max_profit_l1203_120335

/-- Represents the farmer's problem of maximizing profit given land and budget constraints -/
theorem farmer_max_profit (total_land : ℝ) (rice_yield peanut_yield : ℝ) 
  (rice_cost peanut_cost : ℝ) (rice_price peanut_price : ℝ) (budget : ℝ) :
  total_land = 2 →
  rice_yield = 6000 →
  peanut_yield = 1500 →
  rice_cost = 3600 →
  peanut_cost = 1200 →
  rice_price = 3 →
  peanut_price = 5 →
  budget = 6000 →
  ∃ (rice_area peanut_area : ℝ),
    rice_area = 1.5 ∧
    peanut_area = 0.5 ∧
    rice_area + peanut_area ≤ total_land ∧
    rice_cost * rice_area + peanut_cost * peanut_area ≤ budget ∧
    ∀ (x y : ℝ),
      x + y ≤ total_land →
      rice_cost * x + peanut_cost * y ≤ budget →
      (rice_price * rice_yield - rice_cost) * x + (peanut_price * peanut_yield - peanut_cost) * y ≤
      (rice_price * rice_yield - rice_cost) * rice_area + (peanut_price * peanut_yield - peanut_cost) * peanut_area :=
by
  sorry


end NUMINAMATH_CALUDE_farmer_max_profit_l1203_120335


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l1203_120322

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 130) (h2 : Nat.gcd a c = 770) :
  ∃ (b' c' : ℕ+), Nat.gcd a b' = 130 ∧ Nat.gcd a c' = 770 ∧ 
  Nat.gcd b' c' = 10 ∧ ∀ (b'' c'' : ℕ+), Nat.gcd a b'' = 130 → Nat.gcd a c'' = 770 → 
  Nat.gcd b'' c'' ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l1203_120322


namespace NUMINAMATH_CALUDE_largest_common_divisor_under_60_l1203_120300

theorem largest_common_divisor_under_60 : 
  ∃ (n : ℕ), n ∣ 456 ∧ n ∣ 108 ∧ n < 60 ∧ 
  ∀ (m : ℕ), m ∣ 456 → m ∣ 108 → m < 60 → m ≤ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_under_60_l1203_120300


namespace NUMINAMATH_CALUDE_pump_problem_l1203_120338

theorem pump_problem (x y : ℝ) 
  (h1 : x / 4 + y / 12 = 11)  -- Four pumps fill first tanker and 1/3 of second in 11 hours
  (h2 : x / 3 + y / 4 = 18)   -- Three pumps fill first tanker, one fills 1/4 of second in 18 hours
  : y / 3 = 8 :=              -- Three pumps fill second tanker in 8 hours
by sorry

end NUMINAMATH_CALUDE_pump_problem_l1203_120338


namespace NUMINAMATH_CALUDE_frog_distribution_l1203_120332

/-- Represents the three lakes in the problem -/
inductive Lake
| Crystal
| Lassie
| Emerald

/-- Represents the three frog species in the problem -/
inductive Species
| A
| B
| C

/-- The number of frogs of a given species in a given lake -/
def frog_count (l : Lake) (s : Species) : ℕ :=
  match l, s with
  | Lake.Lassie, Species.A => 45
  | Lake.Lassie, Species.B => 35
  | Lake.Lassie, Species.C => 25
  | Lake.Crystal, Species.A => 36
  | Lake.Crystal, Species.B => 39
  | Lake.Crystal, Species.C => 25
  | Lake.Emerald, Species.A => 59
  | Lake.Emerald, Species.B => 70
  | Lake.Emerald, Species.C => 38

/-- The total number of frogs of a given species across all lakes -/
def total_frogs (s : Species) : ℕ :=
  (frog_count Lake.Crystal s) + (frog_count Lake.Lassie s) + (frog_count Lake.Emerald s)

theorem frog_distribution :
  (total_frogs Species.A = 140) ∧
  (total_frogs Species.B = 144) ∧
  (total_frogs Species.C = 88) :=
by sorry


end NUMINAMATH_CALUDE_frog_distribution_l1203_120332


namespace NUMINAMATH_CALUDE_abs_function_symmetric_about_y_axis_l1203_120366

def f (x : ℝ) : ℝ := |x|

theorem abs_function_symmetric_about_y_axis :
  ∀ x : ℝ, f (-x) = f x :=
by
  sorry

end NUMINAMATH_CALUDE_abs_function_symmetric_about_y_axis_l1203_120366


namespace NUMINAMATH_CALUDE_orange_bucket_difference_l1203_120313

theorem orange_bucket_difference (bucket1 bucket2 bucket3 total : ℕ) : 
  bucket1 = 22 →
  bucket2 = bucket1 + 17 →
  bucket3 < bucket2 →
  total = bucket1 + bucket2 + bucket3 →
  total = 89 →
  bucket2 - bucket3 = 11 := by
sorry

end NUMINAMATH_CALUDE_orange_bucket_difference_l1203_120313


namespace NUMINAMATH_CALUDE_max_leftover_candy_l1203_120319

theorem max_leftover_candy (y : ℕ) : ∃ (q r : ℕ), y = 6 * q + r ∧ r < 6 ∧ r ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_candy_l1203_120319


namespace NUMINAMATH_CALUDE_least_k_for_subset_sum_l1203_120317

theorem least_k_for_subset_sum (n : ℕ) :
  let k := if n % 2 = 1 then 2 * n else n + 1
  ∀ (A : Finset ℕ), A.card ≥ k →
    ∃ (S : Finset ℕ), S ⊆ A ∧ S.card % 2 = 0 ∧ (S.sum id) % n = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_k_for_subset_sum_l1203_120317


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l1203_120379

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a rectangular pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from the pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  (area pan.panDimensions) / (area pan.pieceDimensions)

theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 24, width := 15 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 60 := by sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l1203_120379


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1203_120356

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x - 6 > 0) ↔ (∃ x : ℝ, x^2 + 2*x - 6 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1203_120356


namespace NUMINAMATH_CALUDE_band_competition_l1203_120328

theorem band_competition (flute trumpet trombone drummer clarinet french_horn : ℕ) : 
  trumpet = 3 * flute ∧ 
  trombone = trumpet - 8 ∧ 
  drummer = trombone + 11 ∧ 
  clarinet = 2 * flute ∧ 
  french_horn = trombone + 3 ∧ 
  flute + trumpet + trombone + drummer + clarinet + french_horn = 65 → 
  flute = 6 := by sorry

end NUMINAMATH_CALUDE_band_competition_l1203_120328


namespace NUMINAMATH_CALUDE_reciprocal_product_theorem_l1203_120370

theorem reciprocal_product_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 6 * x * y) : (1 / x) * (1 / y) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_product_theorem_l1203_120370


namespace NUMINAMATH_CALUDE_quadratic_roots_complex_l1203_120331

theorem quadratic_roots_complex (x : ℂ) : 
  x^2 + 6*x + 13 = 0 ↔ (x + 3*I) * (x - 3*I) = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_complex_l1203_120331


namespace NUMINAMATH_CALUDE_kolya_walking_speed_l1203_120399

/-- Represents the scenario of Kolya's journey to the store -/
structure JourneyScenario where
  total_distance : ℝ
  initial_speed : ℝ
  doubled_speed : ℝ
  store_closing_time : ℝ

/-- Calculates Kolya's walking speed given a JourneyScenario -/
def calculate_walking_speed (scenario : JourneyScenario) : ℝ :=
  -- The actual calculation would go here
  sorry

/-- Theorem stating that Kolya's walking speed is 20/3 km/h -/
theorem kolya_walking_speed (scenario : JourneyScenario) 
  (h1 : scenario.initial_speed = 10)
  (h2 : scenario.doubled_speed = 2 * scenario.initial_speed)
  (h3 : scenario.store_closing_time = scenario.total_distance / scenario.initial_speed)
  (h4 : scenario.total_distance > 0) :
  calculate_walking_speed scenario = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_kolya_walking_speed_l1203_120399


namespace NUMINAMATH_CALUDE_susans_remaining_money_is_830_02_l1203_120385

/-- Calculates Susan's remaining money after expenses --/
def susans_remaining_money (swimming_earnings babysitting_earnings online_earnings_euro : ℚ)
  (exchange_rate tax_rate clothes_percent books_percent gifts_percent : ℚ) : ℚ :=
  let online_earnings_dollar := online_earnings_euro * exchange_rate
  let total_earnings := swimming_earnings + babysitting_earnings + online_earnings_dollar
  let tax_amount := online_earnings_dollar * tax_rate
  let after_tax := online_earnings_dollar - tax_amount
  let clothes_spend := total_earnings * clothes_percent
  let after_clothes := total_earnings - clothes_spend
  let books_spend := after_clothes * books_percent
  let after_books := after_clothes - books_spend
  let gifts_spend := after_books * gifts_percent
  after_books - gifts_spend

/-- Theorem stating that Susan's remaining money is $830.02 --/
theorem susans_remaining_money_is_830_02 :
  susans_remaining_money 1000 500 300 1.20 0.02 0.30 0.25 0.15 = 830.02 := by
  sorry

end NUMINAMATH_CALUDE_susans_remaining_money_is_830_02_l1203_120385


namespace NUMINAMATH_CALUDE_complex_cut_cube_edges_l1203_120341

/-- A cube with complex cuts at each vertex -/
structure ComplexCutCube where
  /-- The number of vertices in the original cube -/
  originalVertices : Nat
  /-- The number of edges in the original cube -/
  originalEdges : Nat
  /-- The number of cuts per vertex -/
  cutsPerVertex : Nat
  /-- The number of new edges introduced per vertex due to cuts -/
  newEdgesPerVertex : Nat

/-- Theorem stating that a cube with complex cuts results in 60 edges -/
theorem complex_cut_cube_edges (c : ComplexCutCube) 
  (h1 : c.originalVertices = 8)
  (h2 : c.originalEdges = 12)
  (h3 : c.cutsPerVertex = 2)
  (h4 : c.newEdgesPerVertex = 6) : 
  c.originalEdges + c.originalVertices * c.newEdgesPerVertex = 60 := by
  sorry

/-- The total number of edges in a complex cut cube is 60 -/
def total_edges (c : ComplexCutCube) : Nat :=
  c.originalEdges + c.originalVertices * c.newEdgesPerVertex

#check complex_cut_cube_edges

end NUMINAMATH_CALUDE_complex_cut_cube_edges_l1203_120341


namespace NUMINAMATH_CALUDE_inequality_proof_l1203_120340

theorem inequality_proof (u v w : ℝ) 
  (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h : u + v + w + Real.sqrt (u * v * w) = 4) :
  Real.sqrt (v * w / u) + Real.sqrt (u * w / v) + Real.sqrt (u * v / w) ≥ u + v + w :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1203_120340


namespace NUMINAMATH_CALUDE_dance_partners_exist_l1203_120386

variable {Boys Girls : Type}
variable (danced : Boys → Girls → Prop)

theorem dance_partners_exist
  (h1 : ∀ b : Boys, ∃ g : Girls, ¬danced b g)
  (h2 : ∀ g : Girls, ∃ b : Boys, danced b g) :
  ∃ (g g' : Boys) (f f' : Girls),
    danced g f ∧ ¬danced g f' ∧ danced g' f' ∧ ¬danced g' f :=
by sorry

end NUMINAMATH_CALUDE_dance_partners_exist_l1203_120386


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_l1203_120371

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_imply_x (x : ℝ) :
  let a : ℝ × ℝ := (1, 2*x + 1)
  let b : ℝ × ℝ := (2, 3)
  parallel a b → x = 1/4 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_l1203_120371


namespace NUMINAMATH_CALUDE_stamp_collection_theorem_l1203_120376

def stamp_collection_value (total_stamps : ℕ) (sample_stamps : ℕ) (sample_value : ℕ) (bonus_per_set : ℕ) : ℕ :=
  let stamp_value : ℕ := sample_value / sample_stamps
  let total_value : ℕ := total_stamps * stamp_value
  let complete_sets : ℕ := total_stamps / sample_stamps
  let bonus : ℕ := complete_sets * bonus_per_set
  total_value + bonus

theorem stamp_collection_theorem :
  stamp_collection_value 21 7 28 5 = 99 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_theorem_l1203_120376


namespace NUMINAMATH_CALUDE_not_necessary_nor_sufficient_condition_l1203_120343

theorem not_necessary_nor_sufficient_condition (x : ℝ) :
  ¬((-2 < x ∧ x < 1) → (|x| > 1)) ∧ ¬((|x| > 1) → (-2 < x ∧ x < 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_necessary_nor_sufficient_condition_l1203_120343


namespace NUMINAMATH_CALUDE_main_theorem_l1203_120396

/-- A function with the property that (f x + y) * (f y + x) > 0 implies f x + y = f y + x -/
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x + y) * (f y + x) > 0 → f x + y = f y + x

/-- The main theorem: if f has the property, then f x + y ≤ f y + x whenever x > y -/
theorem main_theorem (f : ℝ → ℝ) (hf : has_property f) :
  ∀ x y : ℝ, x > y → f x + y ≤ f y + x :=
sorry

end NUMINAMATH_CALUDE_main_theorem_l1203_120396


namespace NUMINAMATH_CALUDE_figure_area_is_74_l1203_120305

/-- Represents the dimensions of the composite rectangular figure -/
structure FigureDimensions where
  height : ℕ
  width1 : ℕ
  width2 : ℕ
  width3 : ℕ
  height2 : ℕ
  height3 : ℕ

/-- Calculates the area of the composite rectangular figure -/
def calculateArea (d : FigureDimensions) : ℕ :=
  d.height * d.width1 + 
  (d.height - d.height2) * d.width2 +
  d.height2 * d.width2 +
  (d.height - d.height3) * d.width3

/-- Theorem stating that the area of the figure with given dimensions is 74 square units -/
theorem figure_area_is_74 (d : FigureDimensions) 
  (h1 : d.height = 7)
  (h2 : d.width1 = 6)
  (h3 : d.width2 = 4)
  (h4 : d.width3 = 5)
  (h5 : d.height2 = 2)
  (h6 : d.height3 = 6) :
  calculateArea d = 74 := by
  sorry

#eval calculateArea { height := 7, width1 := 6, width2 := 4, width3 := 5, height2 := 2, height3 := 6 }

end NUMINAMATH_CALUDE_figure_area_is_74_l1203_120305


namespace NUMINAMATH_CALUDE_intersection_when_m_is_one_intersection_equals_A_iff_l1203_120309

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | (x - 2*m + 1)*(x - m + 2) < 0}
def B : Set ℝ := {x | 1 ≤ x + 1 ∧ x + 1 ≤ 4}

-- Theorem 1: When m = 1, A ∩ B = {x | 0 ≤ x < 1}
theorem intersection_when_m_is_one :
  A 1 ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

-- Theorem 2: A ∩ B = A if and only if m ∈ {-1, 2}
theorem intersection_equals_A_iff :
  ∀ m : ℝ, A m ∩ B = A m ↔ m = -1 ∨ m = 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_one_intersection_equals_A_iff_l1203_120309


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1203_120392

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Area formula
  b^2 / (3 * Real.sin B) = (1/2) * a * c * Real.sin B →
  -- Given condition
  6 * Real.cos A * Real.cos C = 1 →
  -- Given side length
  b = 3 →
  -- Conclusion
  B = π/3 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1203_120392


namespace NUMINAMATH_CALUDE_eight_monkeys_eat_fortyeight_bananas_l1203_120350

/-- Given a rate at which monkeys eat bananas, calculate the number of monkeys needed to eat a certain number of bananas -/
def monkeys_needed (initial_monkeys initial_bananas target_bananas : ℕ) : ℕ :=
  initial_monkeys

/-- Theorem: Given that 8 monkeys take 8 minutes to eat 8 bananas, 8 monkeys are needed to eat 48 bananas -/
theorem eight_monkeys_eat_fortyeight_bananas :
  monkeys_needed 8 8 48 = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_monkeys_eat_fortyeight_bananas_l1203_120350


namespace NUMINAMATH_CALUDE_sequence_properties_l1203_120362

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

theorem sequence_properties
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (ha_cond : 2 * a 5 - a 3 = 3)
  (hb_2 : b 2 = 1)
  (hb_4 : b 4 = 4) :
  a 7 = 3 ∧ b 6 = 16 ∧ (∃ q : ℝ, (q = 2 ∨ q = -2) ∧ ∀ n : ℕ, b (n + 1) = b n * q) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1203_120362


namespace NUMINAMATH_CALUDE_shaded_area_is_two_thirds_l1203_120359

/-- Square PQRS with shaded regions -/
structure ShadedSquare where
  /-- Side length of the square PQRS -/
  side_length : ℝ
  /-- Side length of the first shaded square region -/
  first_region : ℝ
  /-- Side length of the outer square in the second shaded region -/
  second_region_outer : ℝ
  /-- Side length of the inner square in the second shaded region -/
  second_region_inner : ℝ
  /-- Side length of the outer square in the third shaded region -/
  third_region_outer : ℝ
  /-- Side length of the inner square in the third shaded region -/
  third_region_inner : ℝ

/-- Theorem stating that the shaded area is 2/3 of the total area -/
theorem shaded_area_is_two_thirds (sq : ShadedSquare)
    (h1 : sq.side_length = 6)
    (h2 : sq.first_region = 1)
    (h3 : sq.second_region_outer = 4)
    (h4 : sq.second_region_inner = 2)
    (h5 : sq.third_region_outer = 6)
    (h6 : sq.third_region_inner = 5) :
    (sq.first_region^2 + (sq.second_region_outer^2 - sq.second_region_inner^2) +
     (sq.third_region_outer^2 - sq.third_region_inner^2)) / sq.side_length^2 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_two_thirds_l1203_120359


namespace NUMINAMATH_CALUDE_both_parents_single_eyelids_sufficient_not_necessary_l1203_120355

-- Define the possible genotypes
inductive Genotype
  | AA
  | Aa
  | aa

-- Define the phenotype (eyelid type)
inductive Phenotype
  | Double
  | Single

-- Function to determine phenotype from genotype
def phenotype (g : Genotype) : Phenotype :=
  match g with
  | Genotype.AA => Phenotype.Double
  | Genotype.Aa => Phenotype.Double
  | Genotype.aa => Phenotype.Single

-- Function to model gene inheritance
def inheritGene (parent1 : Genotype) (parent2 : Genotype) : Genotype :=
  sorry

-- Define what it means for both parents to have single eyelids
def bothParentsSingleEyelids (parent1 : Genotype) (parent2 : Genotype) : Prop :=
  phenotype parent1 = Phenotype.Single ∧ phenotype parent2 = Phenotype.Single

-- Define what it means for a child to have single eyelids
def childSingleEyelids (child : Genotype) : Prop :=
  phenotype child = Phenotype.Single

-- Theorem stating that "both parents have single eyelids" is sufficient but not necessary
theorem both_parents_single_eyelids_sufficient_not_necessary :
  (∀ (parent1 parent2 : Genotype),
    bothParentsSingleEyelids parent1 parent2 →
    childSingleEyelids (inheritGene parent1 parent2)) ∧
  (∃ (parent1 parent2 : Genotype),
    childSingleEyelids (inheritGene parent1 parent2) ∧
    ¬bothParentsSingleEyelids parent1 parent2) :=
  sorry

end NUMINAMATH_CALUDE_both_parents_single_eyelids_sufficient_not_necessary_l1203_120355


namespace NUMINAMATH_CALUDE_sum_surface_areas_of_cut_cube_l1203_120377

/-- The sum of surface areas of cuboids resulting from cutting a unit cube -/
theorem sum_surface_areas_of_cut_cube : 
  let n : ℕ := 4  -- number of divisions per side
  let num_cuboids : ℕ := n^3
  let side_length : ℚ := 1 / n
  let surface_area_one_cuboid : ℚ := 6 * side_length^2
  surface_area_one_cuboid * num_cuboids = 24 := by sorry

end NUMINAMATH_CALUDE_sum_surface_areas_of_cut_cube_l1203_120377


namespace NUMINAMATH_CALUDE_four_solutions_gg_eq_3_l1203_120388

def g (x : ℝ) : ℝ := x^2 - 4*x + 3

def domain (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5

theorem four_solutions_gg_eq_3 :
  ∃! (s : Finset ℝ), s.card = 4 ∧ 
  (∀ x ∈ s, domain x ∧ g (g x) = 3) ∧
  (∀ x, domain x → g (g x) = 3 → x ∈ s) :=
sorry

end NUMINAMATH_CALUDE_four_solutions_gg_eq_3_l1203_120388


namespace NUMINAMATH_CALUDE_max_intersections_three_polygons_l1203_120352

/-- Represents a convex polygon with a given number of sides -/
structure ConvexPolygon where
  sides : ℕ

/-- Theorem stating the maximum number of intersections among three convex polygons -/
theorem max_intersections_three_polygons
  (P1 P2 P3 : ConvexPolygon)
  (h1 : P1.sides ≤ P2.sides)
  (h2 : P2.sides ≤ P3.sides)
  (h_no_shared_segments : True)  -- Represents the condition that polygons don't share line segments
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_three_polygons_l1203_120352


namespace NUMINAMATH_CALUDE_correct_purchase_ways_l1203_120318

/-- The number of oreo flavors available -/
def num_oreo_flavors : ℕ := 6

/-- The number of milk flavors available -/
def num_milk_flavors : ℕ := 4

/-- The total number of products they purchase collectively -/
def total_products : ℕ := 3

/-- Function to calculate the number of ways Alpha and Beta can purchase products -/
def purchase_ways : ℕ := sorry

/-- Theorem stating the correct number of ways to purchase products -/
theorem correct_purchase_ways : purchase_ways = 656 := by sorry

end NUMINAMATH_CALUDE_correct_purchase_ways_l1203_120318


namespace NUMINAMATH_CALUDE_vector_equality_sufficient_not_necessary_l1203_120382

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_equality_sufficient_not_necessary :
  (∀ a b : V, a = b → (‖a‖ = ‖b‖ ∧ parallel a b)) ∧
  (∃ a b : V, ‖a‖ = ‖b‖ ∧ parallel a b ∧ a ≠ b) :=
sorry

end NUMINAMATH_CALUDE_vector_equality_sufficient_not_necessary_l1203_120382


namespace NUMINAMATH_CALUDE_complex_modulus_l1203_120316

theorem complex_modulus (z : ℂ) (h : z * (1 - Complex.I) = -Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_l1203_120316


namespace NUMINAMATH_CALUDE_solution_value_l1203_120360

/-- 
If (1, k) is a solution to the equation 2x + y = 6, then k = 4.
-/
theorem solution_value (k : ℝ) : (2 * 1 + k = 6) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1203_120360


namespace NUMINAMATH_CALUDE_simplify_expression_l1203_120372

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 - b^3 = a - b) :
  a/b - b/a + 1/(a*b) = -1 + 2/(a*b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1203_120372


namespace NUMINAMATH_CALUDE_shaded_shapes_area_l1203_120357

/-- Represents a point on a grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a shape on the grid -/
structure GridShape where
  vertices : List GridPoint

/-- The grid size -/
def gridSize : ℕ := 7

/-- Function to calculate the area of a shape on the grid -/
def calculateArea (shape : GridShape) : ℚ :=
  sorry

/-- The newly designed shaded shapes on the grid -/
def shadedShapes : List GridShape :=
  sorry

/-- Theorem stating that the total area of the shaded shapes is 3 -/
theorem shaded_shapes_area :
  (shadedShapes.map calculateArea).sum = 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_shapes_area_l1203_120357


namespace NUMINAMATH_CALUDE_sum_of_first_and_last_l1203_120353

/-- A sequence of eight terms -/
structure EightTermSequence where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ
  U : ℝ
  V : ℝ
  W : ℝ

/-- The sum of any four consecutive terms is 40 -/
def consecutive_sum_40 (seq : EightTermSequence) : Prop :=
  seq.P + seq.Q + seq.R + seq.S = 40 ∧
  seq.Q + seq.R + seq.S + seq.T = 40 ∧
  seq.R + seq.S + seq.T + seq.U = 40 ∧
  seq.S + seq.T + seq.U + seq.V = 40 ∧
  seq.T + seq.U + seq.V + seq.W = 40

theorem sum_of_first_and_last (seq : EightTermSequence) 
  (h1 : seq.S = 10)
  (h2 : consecutive_sum_40 seq) : 
  seq.P + seq.W = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_and_last_l1203_120353


namespace NUMINAMATH_CALUDE_equal_expressions_condition_l1203_120320

theorem equal_expressions_condition (a b c : ℝ) :
  a + b + c = (a + b) * (a + c) ↔ a = 1 ∧ b = 1 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_expressions_condition_l1203_120320


namespace NUMINAMATH_CALUDE_baker_flour_remaining_l1203_120321

/-- A baker's recipe requires 3 eggs for every 2 cups of flour. -/
def recipe_ratio : ℚ := 3 / 2

/-- The number of eggs needed to use up all remaining flour. -/
def eggs_needed : ℕ := 9

/-- Calculates the number of cups of flour remaining given the recipe ratio and eggs needed. -/
def flour_remaining (ratio : ℚ) (eggs : ℕ) : ℚ := (eggs : ℚ) / ratio

theorem baker_flour_remaining :
  flour_remaining recipe_ratio eggs_needed = 6 := by
  sorry

end NUMINAMATH_CALUDE_baker_flour_remaining_l1203_120321


namespace NUMINAMATH_CALUDE_vasya_initial_larger_l1203_120336

/-- Represents the initial investments and profit rates for Vasya and Petya --/
structure InvestmentScenario where
  vasya_initial : ℝ
  petya_initial : ℝ
  vasya_rate : ℝ
  petya_rate : ℝ
  exchange_rate_increase : ℝ

/-- Calculates the profit for a given initial investment and rate --/
def profit (initial : ℝ) (rate : ℝ) : ℝ := initial * rate

/-- Calculates Petya's effective rate considering exchange rate increase --/
def petya_effective_rate (petya_rate : ℝ) (exchange_rate_increase : ℝ) : ℝ :=
  1 + petya_rate + exchange_rate_increase + petya_rate * exchange_rate_increase

/-- Theorem stating that Vasya's initial investment is larger given equal profits --/
theorem vasya_initial_larger (scenario : InvestmentScenario) 
  (h1 : scenario.vasya_rate = 0.20)
  (h2 : scenario.petya_rate = 0.10)
  (h3 : scenario.exchange_rate_increase = 0.095)
  (h4 : profit scenario.vasya_initial scenario.vasya_rate = 
        profit scenario.petya_initial (petya_effective_rate scenario.petya_rate scenario.exchange_rate_increase)) :
  scenario.vasya_initial > scenario.petya_initial := by
  sorry


end NUMINAMATH_CALUDE_vasya_initial_larger_l1203_120336


namespace NUMINAMATH_CALUDE_shoe_repair_time_calculation_l1203_120398

/-- Given the total time spent on repairing shoes and the time required to replace buckles,
    calculate the time needed to even out the heel for each shoe. -/
theorem shoe_repair_time_calculation 
  (total_time : ℝ)
  (buckle_time : ℝ)
  (h_total : total_time = 30)
  (h_buckle : buckle_time = 5)
  : (total_time - buckle_time) / 2 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_shoe_repair_time_calculation_l1203_120398


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_two_l1203_120387

/-- A function f is even if f(-x) = f(x) for all x in ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = (x+a)(x-2) -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ (x + a) * (x - 2)

/-- If f(x) = (x+a)(x-2) is an even function, then a = 2 -/
theorem even_function_implies_a_eq_two :
  ∀ a : ℝ, IsEven (f a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_two_l1203_120387


namespace NUMINAMATH_CALUDE_fraction_unchanged_l1203_120306

theorem fraction_unchanged (x y : ℝ) (h : y ≠ 2*x) : 
  (3*(3*x)) / (2*(3*x) - 3*y) = (3*x) / (2*x - y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l1203_120306


namespace NUMINAMATH_CALUDE_construction_delay_l1203_120314

/-- Represents the construction project with given parameters -/
structure ConstructionProject where
  totalDays : ℕ
  initialWorkers : ℕ
  additionalWorkers : ℕ
  additionalWorkersStartDay : ℕ

/-- Calculates the total work units completed in the project -/
def totalWorkUnits (project : ConstructionProject) : ℕ :=
  project.initialWorkers * project.totalDays +
  project.additionalWorkers * (project.totalDays - project.additionalWorkersStartDay)

/-- Calculates the number of days needed to complete the work with only initial workers -/
def daysNeededWithoutAdditionalWorkers (project : ConstructionProject) : ℕ :=
  (totalWorkUnits project) / project.initialWorkers

/-- Theorem: The project will be 90 days behind schedule without additional workers -/
theorem construction_delay (project : ConstructionProject) 
  (h1 : project.totalDays = 100)
  (h2 : project.initialWorkers = 100)
  (h3 : project.additionalWorkers = 100)
  (h4 : project.additionalWorkersStartDay = 10) :
  daysNeededWithoutAdditionalWorkers project - project.totalDays = 90 := by
  sorry

end NUMINAMATH_CALUDE_construction_delay_l1203_120314


namespace NUMINAMATH_CALUDE_fraction_power_four_l1203_120364

theorem fraction_power_four : (5 / 6 : ℚ) ^ 4 = 625 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_four_l1203_120364


namespace NUMINAMATH_CALUDE_max_value_of_f_l1203_120384

-- Define the function to be maximized
def f (a b c : ℝ) : ℝ := a * b + b * c + 2 * a * c

-- State the theorem
theorem max_value_of_f :
  ∀ a b c : ℝ,
  a ≥ 0 → b ≥ 0 → c ≥ 0 →
  a + b + c = 1 →
  f a b c ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1203_120384


namespace NUMINAMATH_CALUDE_jenny_ran_distance_l1203_120329

-- Define the distance Jenny walked
def distance_walked : ℝ := 0.4

-- Define the additional distance Jenny ran compared to what she walked
def additional_distance_ran : ℝ := 0.2

-- Theorem: The distance Jenny ran is equal to 0.6 miles
theorem jenny_ran_distance : distance_walked + additional_distance_ran = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_jenny_ran_distance_l1203_120329


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1203_120334

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 22 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1203_120334


namespace NUMINAMATH_CALUDE_net_profit_calculation_l1203_120333

def calculate_net_profit (basil_seed_cost mint_seed_cost zinnia_seed_cost : ℚ)
  (potting_soil_cost packaging_cost : ℚ)
  (sellers_fee_rate sales_tax_rate : ℚ)
  (basil_yield mint_yield zinnia_yield : ℕ)
  (basil_germination mint_germination zinnia_germination : ℚ)
  (healthy_basil_price healthy_mint_price healthy_zinnia_price : ℚ)
  (small_basil_price small_mint_price small_zinnia_price : ℚ)
  (healthy_basil_sold small_basil_sold : ℕ)
  (healthy_mint_sold small_mint_sold : ℕ)
  (healthy_zinnia_sold small_zinnia_sold : ℕ) : ℚ :=
  let total_revenue := 
    healthy_basil_price * healthy_basil_sold + small_basil_price * small_basil_sold +
    healthy_mint_price * healthy_mint_sold + small_mint_price * small_mint_sold +
    healthy_zinnia_price * healthy_zinnia_sold + small_zinnia_price * small_zinnia_sold
  let total_expenses := 
    basil_seed_cost + mint_seed_cost + zinnia_seed_cost + potting_soil_cost + packaging_cost
  let sellers_fee := sellers_fee_rate * total_revenue
  let sales_tax := sales_tax_rate * total_revenue
  total_revenue - total_expenses - sellers_fee - sales_tax

theorem net_profit_calculation : 
  calculate_net_profit 2 3 7 15 5 (1/10) (1/20)
    20 15 10 (4/5) (3/4) (7/10)
    5 6 10 3 4 7
    12 8 10 4 5 2 = 158.4 := by sorry

end NUMINAMATH_CALUDE_net_profit_calculation_l1203_120333


namespace NUMINAMATH_CALUDE_shark_percentage_is_25_l1203_120303

/-- Represents the count of fish on day one -/
def day_one_count : ℕ := 15

/-- Represents the multiplier for day two's count relative to day one -/
def day_two_multiplier : ℕ := 3

/-- Represents the total number of sharks counted over two days -/
def total_sharks : ℕ := 15

/-- Calculates the total number of fish counted over two days -/
def total_fish : ℕ := day_one_count + day_one_count * day_two_multiplier

/-- Represents the percentage of sharks among the counted fish -/
def shark_percentage : ℚ := (total_sharks : ℚ) / (total_fish : ℚ) * 100

theorem shark_percentage_is_25 : shark_percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_shark_percentage_is_25_l1203_120303


namespace NUMINAMATH_CALUDE_parking_lot_theorem_l1203_120375

/-- A multi-story parking lot with equal-sized levels -/
structure ParkingLot where
  total_spaces : ℕ
  num_levels : ℕ
  cars_on_one_level : ℕ

/-- Calculates the number of additional cars that can fit on one level -/
def additional_cars (p : ParkingLot) : ℕ :=
  (p.total_spaces / p.num_levels) - p.cars_on_one_level

theorem parking_lot_theorem (p : ParkingLot) 
  (h1 : p.total_spaces = 425)
  (h2 : p.num_levels = 5)
  (h3 : p.cars_on_one_level = 23) :
  additional_cars p = 62 := by
  sorry

#eval additional_cars { total_spaces := 425, num_levels := 5, cars_on_one_level := 23 }

end NUMINAMATH_CALUDE_parking_lot_theorem_l1203_120375


namespace NUMINAMATH_CALUDE_band_gigs_theorem_l1203_120315

/-- Represents a band with its members and earnings -/
structure Band where
  members : ℕ
  earnings_per_member_per_gig : ℕ
  total_earnings : ℕ

/-- Calculates the number of gigs played by a band -/
def gigs_played (b : Band) : ℕ :=
  b.total_earnings / (b.members * b.earnings_per_member_per_gig)

/-- Theorem stating that for a band with 4 members, $20 earnings per member per gig,
    and $400 total earnings, the number of gigs played is 5 -/
theorem band_gigs_theorem (b : Band) 
    (h1 : b.members = 4)
    (h2 : b.earnings_per_member_per_gig = 20)
    (h3 : b.total_earnings = 400) :
    gigs_played b = 5 := by
  sorry

end NUMINAMATH_CALUDE_band_gigs_theorem_l1203_120315


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1203_120326

/-- A geometric sequence with a_3 = 8 * a_6 has S_4 / S_2 = 5/4 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- geometric sequence condition
  (∀ n, S n = (a 1 * (1 - (a 2 / a 1)^n)) / (1 - (a 2 / a 1))) →  -- sum formula
  a 3 = 8 * a 6 →  -- given condition
  S 4 / S 2 = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1203_120326


namespace NUMINAMATH_CALUDE_john_puppy_profit_l1203_120358

/-- Calculates the profit from selling puppies given the initial conditions --/
def puppy_profit (initial_puppies : ℕ) (sale_price : ℕ) (stud_fee : ℕ) : ℕ :=
  let remaining_after_giving_away := initial_puppies / 2
  let remaining_after_keeping_one := remaining_after_giving_away - 1
  let total_sales := remaining_after_keeping_one * sale_price
  total_sales - stud_fee

/-- Proves that John's profit from selling puppies is $1500 --/
theorem john_puppy_profit : puppy_profit 8 600 300 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_john_puppy_profit_l1203_120358


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_four_sqrt_three_l1203_120380

theorem sqrt_sum_equals_four_sqrt_three :
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_four_sqrt_three_l1203_120380


namespace NUMINAMATH_CALUDE_diorama_building_time_l1203_120361

/-- Represents the time spent on the diorama project -/
structure DioramaTime where
  planning : ℕ  -- Planning time in minutes
  building : ℕ  -- Building time in minutes

/-- Defines the conditions of the diorama project -/
def validDioramaTime (t : DioramaTime) : Prop :=
  t.building = 3 * t.planning - 5 ∧
  t.building + t.planning = 67

/-- Theorem stating that the building time is 49 minutes -/
theorem diorama_building_time :
  ∀ t : DioramaTime, validDioramaTime t → t.building = 49 :=
by
  sorry


end NUMINAMATH_CALUDE_diorama_building_time_l1203_120361


namespace NUMINAMATH_CALUDE_sin_alpha_plus_5pi_12_l1203_120323

theorem sin_alpha_plus_5pi_12 (α : Real) (h1 : 0 < α) (h2 : α < π / 2)
  (h3 : Real.cos α - Real.sin α = 2 * Real.sqrt 2 / 3) :
  Real.sin (α + 5 * π / 12) = (2 + Real.sqrt 15) / 6 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_5pi_12_l1203_120323


namespace NUMINAMATH_CALUDE_factor_81_minus_4y4_l1203_120374

theorem factor_81_minus_4y4 (y : ℝ) : 81 - 4 * y^4 = (9 + 2 * y^2) * (9 - 2 * y^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_81_minus_4y4_l1203_120374


namespace NUMINAMATH_CALUDE_problem_solution_l1203_120393

theorem problem_solution : ∃! x : ℝ, 3 ∈ ({x + 2, x^2 + 2*x} : Set ℝ) ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1203_120393


namespace NUMINAMATH_CALUDE_james_room_area_l1203_120324

/-- Calculates the total area of rooms given initial dimensions and modifications --/
def total_area (initial_length initial_width increase : ℕ) : ℕ :=
  let new_length := initial_length + increase
  let new_width := initial_width + increase
  let single_room_area := new_length * new_width
  4 * single_room_area + 2 * single_room_area

/-- Theorem stating the total area for the given problem --/
theorem james_room_area :
  total_area 13 18 2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_james_room_area_l1203_120324


namespace NUMINAMATH_CALUDE_no_collision_after_jumps_l1203_120354

/-- Represents the position of a grasshopper -/
structure Position where
  x : Int
  y : Int

/-- Represents the state of the system with four grasshoppers -/
structure GrasshopperSystem where
  positions : Fin 4 → Position

/-- Performs a symmetric jump for one grasshopper -/
def symmetricJump (system : GrasshopperSystem) (jumper : Fin 4) : GrasshopperSystem :=
  sorry

/-- Checks if any two grasshoppers occupy the same position -/
def hasCollision (system : GrasshopperSystem) : Bool :=
  sorry

/-- Initial configuration of the grasshoppers on a square -/
def initialSquare : GrasshopperSystem :=
  { positions := λ i => match i with
    | 0 => ⟨0, 0⟩
    | 1 => ⟨0, 1⟩
    | 2 => ⟨1, 1⟩
    | 3 => ⟨1, 0⟩ }

theorem no_collision_after_jumps :
  ∀ (jumps : List (Fin 4)), ¬(hasCollision (jumps.foldl symmetricJump initialSquare)) :=
  sorry

end NUMINAMATH_CALUDE_no_collision_after_jumps_l1203_120354


namespace NUMINAMATH_CALUDE_arcsin_sin_eq_x_div_3_l1203_120307

theorem arcsin_sin_eq_x_div_3 :
  ∃! x : ℝ, x ∈ Set.Icc (-3 * Real.pi / 2) (3 * Real.pi / 2) ∧ 
    Real.arcsin (Real.sin x) = x / 3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sin_eq_x_div_3_l1203_120307


namespace NUMINAMATH_CALUDE_triangle_properties_l1203_120346

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the area and cosine of angle ADC where D is the midpoint of BC. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  c = 4 →
  b = 3 →
  A = π / 3 →
  let S := (1 / 2) * b * c * Real.sin A
  let cos_ADC := (7 * Real.sqrt 481) / 481
  (S = 3 * Real.sqrt 3 ∧ cos_ADC = (7 * Real.sqrt 481) / 481) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1203_120346


namespace NUMINAMATH_CALUDE_range_of_m_l1203_120348

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → 
  -3 ≤ m ∧ m ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1203_120348


namespace NUMINAMATH_CALUDE_square_side_length_equal_perimeter_l1203_120312

theorem square_side_length_equal_perimeter (rectangle_length rectangle_width : ℝ) 
  (h1 : rectangle_length = 10)
  (h2 : rectangle_width = 8) : 
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  let square_side_length := rectangle_perimeter / 4
  square_side_length = 9 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_equal_perimeter_l1203_120312


namespace NUMINAMATH_CALUDE_investment_plans_count_l1203_120347

theorem investment_plans_count (n_projects : ℕ) (n_cities : ℕ) (max_per_city : ℕ) : 
  n_projects = 3 → n_cities = 5 → max_per_city = 2 →
  (Nat.choose n_cities 3 * Nat.factorial 3 + 
   Nat.choose n_cities 1 * Nat.choose (n_cities - 1) 1 * 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_investment_plans_count_l1203_120347


namespace NUMINAMATH_CALUDE_number_of_amateurs_l1203_120394

/-- The number of chess amateurs in the tournament -/
def n : ℕ := sorry

/-- The number of other amateurs each amateur plays with -/
def games_per_amateur : ℕ := 4

/-- The total number of possible games in the tournament -/
def total_games : ℕ := 10

/-- Theorem stating the number of chess amateurs in the tournament -/
theorem number_of_amateurs :
  n = 5 ∧
  games_per_amateur = 4 ∧
  total_games = 10 ∧
  n.choose 2 = total_games :=
sorry

end NUMINAMATH_CALUDE_number_of_amateurs_l1203_120394


namespace NUMINAMATH_CALUDE_inequality_holds_iff_c_equals_one_l1203_120395

theorem inequality_holds_iff_c_equals_one (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (∃ c : ℝ, c > 0 ∧ ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (x^3 * y + y^3 * z + z^3 * x) / (x + y + z) + 4 * c / (x * y * z) ≥ 2 * c + 2) ↔
  (∃ c : ℝ, c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_c_equals_one_l1203_120395


namespace NUMINAMATH_CALUDE_star_removal_theorem_l1203_120344

/-- Represents a 2n × 2n table with stars -/
structure StarTable (n : ℕ) where
  stars : Finset ((Fin (2*n)) × (Fin (2*n)))
  star_count : stars.card = 3*n

/-- Represents a selection of rows and columns -/
structure Selection (n : ℕ) where
  rows : Finset (Fin (2*n))
  columns : Finset (Fin (2*n))
  row_count : rows.card = n
  column_count : columns.card = n

/-- Predicate to check if a star is removed by a selection -/
def is_removed (star : (Fin (2*n)) × (Fin (2*n))) (sel : Selection n) : Prop :=
  star.1 ∈ sel.rows ∨ star.2 ∈ sel.columns

/-- Theorem: For any 2n × 2n table with 3n stars, there exists a selection
    of n rows and n columns that removes all stars -/
theorem star_removal_theorem (n : ℕ) (table : StarTable n) :
  ∃ (sel : Selection n), ∀ star ∈ table.stars, is_removed star sel :=
sorry

end NUMINAMATH_CALUDE_star_removal_theorem_l1203_120344
