import Mathlib

namespace die_roll_probability_l3004_300456

theorem die_roll_probability : 
  let p : ℝ := 1 / 2  -- probability of rolling an even number on a single die
  let n : ℕ := 8      -- number of rolls
  1 - (1 - p) ^ n = 255 / 256 :=
by sorry

end die_roll_probability_l3004_300456


namespace sin_cube_identity_l3004_300450

theorem sin_cube_identity (θ : ℝ) :
  ∃! (c d : ℝ), ∀ θ, Real.sin θ ^ 3 = c * Real.sin (3 * θ) + d * Real.sin θ :=
by
  -- The unique pair (c, d) is (-1/4, 3/4)
  sorry

end sin_cube_identity_l3004_300450


namespace square_root_of_x_plus_y_l3004_300499

theorem square_root_of_x_plus_y (x y : ℝ) : 
  (Real.sqrt (3 - x) + Real.sqrt (x - 3) + 1 = y) → 
  Real.sqrt (x + y) = 2 := by
sorry

end square_root_of_x_plus_y_l3004_300499


namespace adjacent_nonadjacent_probability_l3004_300440

def num_students : ℕ := 5

def total_arrangements : ℕ := num_students.factorial

def valid_arrangements : ℕ := 24

theorem adjacent_nonadjacent_probability :
  (valid_arrangements : ℚ) / total_arrangements = 1 / 5 := by sorry

end adjacent_nonadjacent_probability_l3004_300440


namespace intersection_when_a_is_one_range_of_a_when_B_subset_complement_A_l3004_300468

-- Define the sets A and B
def A : Set ℝ := {x | (1 + x) / (2 - x) > 0}
def B (a : ℝ) : Set ℝ := {x | (a * x - 1) * (x + 2) ≥ 0}

-- Theorem 1: When a = 1, A ∩ B = {x | 1 ≤ x < 2}
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

-- Theorem 2: When B ⊆ ℝ\A, the range of a is 0 < a ≤ 1/2
theorem range_of_a_when_B_subset_complement_A :
  ∀ a : ℝ, (0 < a ∧ B a ⊆ (Set.univ \ A)) ↔ (0 < a ∧ a ≤ 1/2) := by sorry

end intersection_when_a_is_one_range_of_a_when_B_subset_complement_A_l3004_300468


namespace vectors_are_coplanar_l3004_300489

open Real
open EuclideanSpace

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define the vectors
variable (MA MB MC : V)

-- State the theorem
theorem vectors_are_coplanar 
  (h_noncollinear : ¬ ∃ (k : ℝ), MA = k • MB)
  (h_MC_def : MC = 5 • MA - 3 • MB) :
  ∃ (a b c : ℝ), a • MA + b • MB + c • MC = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :=
sorry

end vectors_are_coplanar_l3004_300489


namespace max_integers_with_pairwise_common_divisor_and_coprime_triples_l3004_300407

theorem max_integers_with_pairwise_common_divisor_and_coprime_triples :
  (∃ (n : ℕ) (a : Fin n → ℕ), n ≥ 3 ∧
    (∀ i, a i < 5000) ∧
    (∀ i j, i ≠ j → ∃ d > 1, d ∣ a i ∧ d ∣ a j) ∧
    (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → Nat.gcd (a i) (Nat.gcd (a j) (a k)) = 1)) →
  (∀ (n : ℕ) (a : Fin n → ℕ), n ≥ 3 →
    (∀ i, a i < 5000) →
    (∀ i j, i ≠ j → ∃ d > 1, d ∣ a i ∧ d ∣ a j) →
    (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → Nat.gcd (a i) (Nat.gcd (a j) (a k)) = 1) →
    n ≤ 4) :=
by sorry

end max_integers_with_pairwise_common_divisor_and_coprime_triples_l3004_300407


namespace penalty_kicks_count_l3004_300446

theorem penalty_kicks_count (total_players : ℕ) (goalies : ℕ) 
  (h1 : total_players = 25) 
  (h2 : goalies = 4) 
  (h3 : goalies ≤ total_players) : 
  goalies * (total_players - 1) = 96 := by
  sorry

end penalty_kicks_count_l3004_300446


namespace line_rotation_theorem_l3004_300465

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rotates a line counterclockwise around a point --/
def rotateLine (l : Line) (θ : ℝ) (p : ℝ × ℝ) : Line :=
  sorry

/-- Finds the intersection of a line with the x-axis --/
def xAxisIntersection (l : Line) : ℝ × ℝ :=
  sorry

theorem line_rotation_theorem (l : Line) :
  l.a = 2 ∧ l.b = -1 ∧ l.c = -4 →
  let p := xAxisIntersection l
  let l' := rotateLine l (π/4) p
  l'.a = 3 ∧ l'.b = 1 ∧ l'.c = -6 :=
sorry

end line_rotation_theorem_l3004_300465


namespace strip_length_is_four_l3004_300430

/-- The length of each square in the strip -/
def square_length : ℚ := 2/3

/-- The number of squares in the strip -/
def num_squares : ℕ := 6

/-- The total length of the strip -/
def strip_length : ℚ := square_length * num_squares

/-- Theorem: The strip composed of 6 squares, each with length 2/3, has a total length of 4 -/
theorem strip_length_is_four : strip_length = 4 := by
  sorry

end strip_length_is_four_l3004_300430


namespace fayes_age_l3004_300405

/-- Given the ages of Diana, Eduardo, Chad, Faye, and Greg, prove Faye's age --/
theorem fayes_age 
  (D E C F G : ℕ) -- Ages of Diana, Eduardo, Chad, Faye, and Greg
  (h1 : D = E - 2)
  (h2 : C = E + 3)
  (h3 : F = C - 1)
  (h4 : D = 16)
  (h5 : G = D - 5) :
  F = 20 := by
  sorry

end fayes_age_l3004_300405


namespace eight_book_distribution_l3004_300462

/-- The number of ways to distribute identical books between a library and being checked out -/
def distribute_books (total : ℕ) : ℕ :=
  if total < 2 then 0 else total - 1

/-- Theorem: For 8 identical books, there are 7 ways to distribute them between a library and being checked out, with at least one book in each location -/
theorem eight_book_distribution :
  distribute_books 8 = 7 := by
  sorry

end eight_book_distribution_l3004_300462


namespace complex_subtraction_l3004_300422

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 2 + 3*I) :
  a - 3*b = -1 - 12*I := by
  sorry

end complex_subtraction_l3004_300422


namespace nut_count_theorem_l3004_300418

def total_pistachios : ℕ := 80
def total_almonds : ℕ := 60
def total_cashews : ℕ := 40

def pistachio_shell_ratio : ℚ := 95 / 100
def pistachio_opened_ratio : ℚ := 75 / 100

def almond_shell_ratio : ℚ := 90 / 100
def almond_cracked_ratio : ℚ := 80 / 100

def cashew_shell_ratio : ℚ := 85 / 100
def cashew_salted_ratio : ℚ := 70 / 100
def cashew_opened_ratio : ℚ := 60 / 100

theorem nut_count_theorem :
  let pistachios_opened := ⌊(total_pistachios : ℚ) * pistachio_shell_ratio * pistachio_opened_ratio⌋
  let almonds_cracked := ⌊(total_almonds : ℚ) * almond_shell_ratio * almond_cracked_ratio⌋
  let cashews_opened := ⌊(total_cashews : ℚ) * cashew_shell_ratio * cashew_opened_ratio⌋
  let total_opened_cracked := pistachios_opened + almonds_cracked + cashews_opened
  let shelled_salted_cashews := ⌊(total_cashews : ℚ) * cashew_shell_ratio * cashew_salted_ratio⌋
  total_opened_cracked = 120 ∧ shelled_salted_cashews = 23 := by
  sorry

end nut_count_theorem_l3004_300418


namespace zachary_pushup_count_l3004_300415

/-- The number of push-ups David did -/
def david_pushups : ℕ := 44

/-- The difference between Zachary's and David's push-ups -/
def pushup_difference : ℕ := 7

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := david_pushups + pushup_difference

theorem zachary_pushup_count : zachary_pushups = 51 := by
  sorry

end zachary_pushup_count_l3004_300415


namespace total_turtles_is_30_l3004_300449

/-- The number of turtles Kristen has -/
def kristens_turtles : ℕ := 12

/-- The number of turtles Kris has -/
def kris_turtles : ℕ := kristens_turtles / 4

/-- The number of turtles Trey has -/
def treys_turtles : ℕ := 5 * kris_turtles

/-- The total number of turtles -/
def total_turtles : ℕ := kristens_turtles + kris_turtles + treys_turtles

theorem total_turtles_is_30 : total_turtles = 30 := by
  sorry

end total_turtles_is_30_l3004_300449


namespace linear_function_proof_l3004_300410

/-- A linear function passing through two points -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := fun x ↦ k * x + b

/-- The linear function passes through the point (3,1) -/
def PassesThrough3_1 (k b : ℝ) : Prop := LinearFunction k b 3 = 1

/-- The linear function passes through the point (2,0) -/
def PassesThrough2_0 (k b : ℝ) : Prop := LinearFunction k b 2 = 0

theorem linear_function_proof (k b : ℝ) 
  (h1 : PassesThrough3_1 k b) (h2 : PassesThrough2_0 k b) :
  (∀ x, LinearFunction k b x = x - 2) ∧ (LinearFunction k b 6 = 4) := by
  sorry

end linear_function_proof_l3004_300410


namespace derivative_sin_minus_x_cos_l3004_300438

theorem derivative_sin_minus_x_cos (x : ℝ) :
  deriv (λ x => Real.sin x - x * Real.cos x) x = x * Real.sin x := by
  sorry

end derivative_sin_minus_x_cos_l3004_300438


namespace third_divisor_is_seventeen_l3004_300423

theorem third_divisor_is_seventeen : ∃ (d : ℕ), d = 17 ∧ d > 11 ∧ 
  (3374 % 9 = 8) ∧ (3374 % 11 = 8) ∧ (3374 % d = 8) ∧
  (∀ (x : ℕ), x > 11 ∧ x < d → (3374 % x ≠ 8)) :=
by sorry

end third_divisor_is_seventeen_l3004_300423


namespace perimeter_comparison_l3004_300455

-- Define a structure for rectangular parallelepiped
structure RectangularParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  positive_dimensions : 0 < length ∧ 0 < width ∧ 0 < height

-- Define a function to calculate the perimeter of a rectangular parallelepiped
def perimeter (p : RectangularParallelepiped) : ℝ :=
  4 * (p.length + p.width + p.height)

-- Define what it means for one parallelepiped to be contained within another
def contained_within (p q : RectangularParallelepiped) : Prop :=
  p.length ≤ q.length ∧ p.width ≤ q.width ∧ p.height ≤ q.height

-- Theorem statement
theorem perimeter_comparison 
  (p q : RectangularParallelepiped) 
  (h : contained_within p q) : 
  perimeter p ≤ perimeter q :=
sorry

end perimeter_comparison_l3004_300455


namespace height_of_cylinder_A_l3004_300434

/-- Theorem: Height of Cylinder A given volume ratio with Cylinder B -/
theorem height_of_cylinder_A (r_A r_B h_B : ℝ) 
  (h_circum_A : 2 * Real.pi * r_A = 8)
  (h_circum_B : 2 * Real.pi * r_B = 10)
  (h_height_B : h_B = 8)
  (h_volume_ratio : Real.pi * r_A^2 * (7 : ℝ) = 0.5600000000000001 * Real.pi * r_B^2 * h_B) :
  ∃ h_A : ℝ, h_A = 7 ∧ Real.pi * r_A^2 * h_A = 0.5600000000000001 * Real.pi * r_B^2 * h_B := by
  sorry

end height_of_cylinder_A_l3004_300434


namespace fraction_addition_l3004_300408

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l3004_300408


namespace ones_digit_of_3_to_52_l3004_300459

theorem ones_digit_of_3_to_52 : (3^52 : ℕ) % 10 = 1 := by sorry

end ones_digit_of_3_to_52_l3004_300459


namespace factorization_2x2_minus_8_factorization_ax2_minus_2ax_plus_a_l3004_300420

-- Factorization of 2x^2 - 8
theorem factorization_2x2_minus_8 (x : ℝ) :
  2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by sorry

-- Factorization of ax^2 - 2ax + a
theorem factorization_ax2_minus_2ax_plus_a (x a : ℝ) (ha : a ≠ 0) :
  a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by sorry

end factorization_2x2_minus_8_factorization_ax2_minus_2ax_plus_a_l3004_300420


namespace sum_abcd_is_negative_six_l3004_300441

theorem sum_abcd_is_negative_six 
  (a b c d : ℤ) 
  (h : a + 1 = b + 2 ∧ b + 2 = c + 3 ∧ c + 3 = d + 4 ∧ d + 4 = a + b + c + d + 7) : 
  a + b + c + d = -6 := by
sorry

end sum_abcd_is_negative_six_l3004_300441


namespace polynomial_positivity_l3004_300470

theorem polynomial_positivity (P : ℕ → ℝ) 
  (h0 : P 0 > 0)
  (h1 : P 1 > P 0)
  (h2 : P 2 > 2 * P 1 - P 0)
  (h3 : P 3 > 3 * P 2 - 3 * P 1 + P 0)
  (h4 : ∀ n : ℕ, P (n + 4) > 4 * P (n + 3) - 6 * P (n + 2) + 4 * P (n + 1) - P n) :
  ∀ n : ℕ, n > 0 → P n > 0 := by
  sorry

end polynomial_positivity_l3004_300470


namespace morning_milk_calculation_l3004_300439

/-- The number of gallons of milk Aunt May got this morning -/
def morning_milk : ℕ := 365

/-- The number of gallons of milk Aunt May got in the evening -/
def evening_milk : ℕ := 380

/-- The number of gallons of milk Aunt May sold -/
def sold_milk : ℕ := 612

/-- The number of gallons of milk left over from yesterday -/
def leftover_milk : ℕ := 15

/-- The number of gallons of milk remaining -/
def remaining_milk : ℕ := 148

/-- Theorem stating that the morning milk calculation is correct -/
theorem morning_milk_calculation :
  morning_milk + evening_milk + leftover_milk - sold_milk = remaining_milk :=
by sorry

end morning_milk_calculation_l3004_300439


namespace arccos_lt_arcsin_iff_x_in_open_zero_one_l3004_300424

theorem arccos_lt_arcsin_iff_x_in_open_zero_one (x : ℝ) :
  x ∈ Set.Icc (-1) 1 →
  (Real.arccos x < Real.arcsin x ↔ x ∈ Set.Ioo 0 1) :=
by sorry

end arccos_lt_arcsin_iff_x_in_open_zero_one_l3004_300424


namespace a_squared_b_irrational_l3004_300445

def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem a_squared_b_irrational 
  (a b : ℝ) 
  (h_a_rational : is_rational a) 
  (h_b_irrational : ¬ is_rational b) 
  (h_ab_rational : is_rational (a * b)) : 
  ¬ is_rational (a^2 * b) :=
sorry

end a_squared_b_irrational_l3004_300445


namespace greatest_seven_digit_divisible_by_lcm_l3004_300426

def is_seven_digit (n : ℕ) : Prop := n ≥ 1000000 ∧ n ≤ 9999999

def lcm_primes : ℕ := 41 * 43 * 47 * 53

theorem greatest_seven_digit_divisible_by_lcm :
  ∀ n : ℕ, is_seven_digit n → n % lcm_primes = 0 → n ≤ 8833702 := by sorry

end greatest_seven_digit_divisible_by_lcm_l3004_300426


namespace game_cost_calculation_l3004_300413

theorem game_cost_calculation (total_earnings : ℕ) (blade_cost : ℕ) (num_games : ℕ) 
  (h1 : total_earnings = 101)
  (h2 : blade_cost = 47)
  (h3 : num_games = 9)
  (h4 : (total_earnings - blade_cost) % num_games = 0) :
  (total_earnings - blade_cost) / num_games = 6 := by
  sorry

end game_cost_calculation_l3004_300413


namespace board_game_theorem_l3004_300457

/-- Represents the operation of replacing two numbers with their combination -/
def combine (a b : ℚ) : ℚ := a * b + a + b

/-- The set of initial numbers on the board -/
def initial_numbers (n : ℕ) : List ℚ := List.range n |>.map (λ i => 1 / (i + 1))

/-- The invariant product of all numbers on the board increased by 1 -/
def product_plus_one (numbers : List ℚ) : ℚ := numbers.foldl (λ acc x => acc * (x + 1)) 1

/-- The final number after n-1 operations -/
def final_number (n : ℕ) : ℚ := n

theorem board_game_theorem (n : ℕ) (h : n > 0) :
  ∃ (operations : List (ℕ × ℕ)),
    operations.length = n - 1 ∧
    final_number n = product_plus_one (initial_numbers n) - 1 := by
  sorry

#check board_game_theorem

end board_game_theorem_l3004_300457


namespace x_squared_less_than_abs_x_l3004_300461

theorem x_squared_less_than_abs_x (x : ℝ) :
  x^2 < |x| ↔ (-1 < x ∧ x < 0) ∨ (0 < x ∧ x < 1) :=
by sorry

end x_squared_less_than_abs_x_l3004_300461


namespace diagonal_length_of_regular_hexagon_l3004_300497

/-- A regular hexagon with side length 12 units -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 12)

/-- The length of a diagonal in a regular hexagon -/
def diagonal_length (h : RegularHexagon) : ℝ := 2 * h.side_length

/-- Theorem: The diagonal length of a regular hexagon with side length 12 is 24 -/
theorem diagonal_length_of_regular_hexagon (h : RegularHexagon) :
  diagonal_length h = 24 := by
  sorry

#check diagonal_length_of_regular_hexagon

end diagonal_length_of_regular_hexagon_l3004_300497


namespace minimum_discount_for_profit_margin_l3004_300432

theorem minimum_discount_for_profit_margin 
  (cost_price : ℝ) 
  (marked_price : ℝ) 
  (min_profit_margin : ℝ) 
  (discount : ℝ) :
  cost_price = 800 →
  marked_price = 1200 →
  min_profit_margin = 0.2 →
  discount = 0.08 →
  marked_price * (1 - discount) ≥ cost_price * (1 + min_profit_margin) ∧
  ∀ d : ℝ, d < discount → marked_price * (1 - d) < cost_price * (1 + min_profit_margin) :=
by sorry

end minimum_discount_for_profit_margin_l3004_300432


namespace area_between_curves_l3004_300473

/-- The upper function in the integral -/
def f (x : ℝ) : ℝ := 2 * x - x^2 + 3

/-- The lower function in the integral -/
def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

/-- The theorem stating that the area between the curves is 9 -/
theorem area_between_curves : ∫ x in (0)..(3), (f x - g x) = 9 := by
  sorry

end area_between_curves_l3004_300473


namespace min_tile_A_1011_l3004_300477

/-- Represents a tile type -/
inductive Tile
| A  -- Covers 3 squares: 2 in one row and 1 in the adjacent row
| B  -- Covers 4 squares: 2 in one row and 2 in the adjacent row

/-- Represents a tiling of a square grid -/
def Tiling (n : ℕ) := List (Tile × ℕ × ℕ)  -- List of (tile type, row, column)

/-- Checks if a tiling is valid for an n×n square -/
def isValidTiling (n : ℕ) (t : Tiling n) : Prop := sorry

/-- Counts the number of tiles of type A in a tiling -/
def countTileA (t : Tiling n) : ℕ := sorry

/-- Theorem: The minimum number of tiles A required to tile a 1011×1011 square is 2023 -/
theorem min_tile_A_1011 :
  ∀ t : Tiling 1011, isValidTiling 1011 t → countTileA t ≥ 2023 ∧
  ∃ t' : Tiling 1011, isValidTiling 1011 t' ∧ countTileA t' = 2023 := by
  sorry

#check min_tile_A_1011

end min_tile_A_1011_l3004_300477


namespace factorial_sum_equals_5040_l3004_300469

theorem factorial_sum_equals_5040 : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end factorial_sum_equals_5040_l3004_300469


namespace balloon_difference_l3004_300484

def your_balloons : ℕ := 7
def friend_balloons : ℕ := 5

theorem balloon_difference : your_balloons - friend_balloons = 2 := by
  sorry

end balloon_difference_l3004_300484


namespace vector_magnitude_proof_l3004_300491

def a : ℝ × ℝ × ℝ := (1, 2, 3)
def b : ℝ × ℝ × ℝ := (0, 1, -4)

theorem vector_magnitude_proof : ‖a - 2 • b‖ = Real.sqrt 122 := by
  sorry

end vector_magnitude_proof_l3004_300491


namespace danivan_initial_inventory_l3004_300458

/-- Represents the inventory and sales data for Danivan Drugstore --/
structure DrugstoreData where
  monday_sales : ℕ
  tuesday_sales : ℕ
  daily_sales_wed_to_sun : ℕ
  saturday_delivery : ℕ
  end_of_week_inventory : ℕ

/-- Calculates the initial inventory of hand sanitizer gel bottles --/
def initial_inventory (data : DrugstoreData) : ℕ :=
  data.end_of_week_inventory + 
  data.monday_sales + 
  data.tuesday_sales + 
  (5 * data.daily_sales_wed_to_sun) - 
  data.saturday_delivery

/-- Theorem stating that the initial inventory is 4500 bottles --/
theorem danivan_initial_inventory : 
  initial_inventory {
    monday_sales := 2445,
    tuesday_sales := 900,
    daily_sales_wed_to_sun := 50,
    saturday_delivery := 650,
    end_of_week_inventory := 1555
  } = 4500 := by
  sorry


end danivan_initial_inventory_l3004_300458


namespace last_two_digits_sum_factorials_15_l3004_300427

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials_15 :
  lastTwoDigits (sumFactorials 15) = 13 := by sorry

end last_two_digits_sum_factorials_15_l3004_300427


namespace train_length_problem_l3004_300429

/-- Represents the problem of calculating the length of a train based on James's jogging --/
theorem train_length_problem (james_speed : ℝ) (train_speed : ℝ) (steps_forward : ℕ) (steps_backward : ℕ) :
  james_speed > train_speed →
  steps_forward = 400 →
  steps_backward = 160 →
  let train_length := (steps_forward * james_speed - steps_forward * train_speed + 
                       steps_backward * james_speed + steps_backward * train_speed) / 2
  train_length = 640 / 7 := by
  sorry

end train_length_problem_l3004_300429


namespace odd_function_sum_l3004_300488

-- Define an odd function f on the real numbers
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_sum (f : ℝ → ℝ) (h1 : isOddFunction f) (h2 : f 1 = -2) :
  f (-1) + f 0 = 2 := by
  sorry

end odd_function_sum_l3004_300488


namespace shopping_tax_theorem_l3004_300472

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def total_tax_percentage (clothing_percent : ℝ) (food_percent : ℝ) (other_percent : ℝ)
                         (clothing_tax : ℝ) (food_tax : ℝ) (other_tax : ℝ) : ℝ :=
  clothing_percent * clothing_tax + food_percent * food_tax + other_percent * other_tax

/-- Theorem stating that the total tax percentage is 5.2% given the specific conditions -/
theorem shopping_tax_theorem :
  total_tax_percentage 0.5 0.1 0.4 0.04 0 0.08 = 0.052 := by
  sorry

#eval total_tax_percentage 0.5 0.1 0.4 0.04 0 0.08

end shopping_tax_theorem_l3004_300472


namespace bobby_free_throws_l3004_300442

theorem bobby_free_throws (initial_throws : ℕ) (initial_success_rate : ℚ)
  (additional_throws : ℕ) (new_success_rate : ℚ) :
  initial_throws = 30 →
  initial_success_rate = 3/5 →
  additional_throws = 10 →
  new_success_rate = 16/25 →
  ∃ (last_successful_throws : ℕ),
    last_successful_throws = 8 ∧
    (initial_success_rate * initial_throws + last_successful_throws) / 
    (initial_throws + additional_throws) = new_success_rate :=
by
  sorry

end bobby_free_throws_l3004_300442


namespace percentage_of_sikh_boys_l3004_300480

/-- Proves that the percentage of Sikh boys in a school is 10% -/
theorem percentage_of_sikh_boys (total_boys : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (other_boys : ℕ) :
  total_boys = 850 →
  muslim_percent = 34 / 100 →
  hindu_percent = 28 / 100 →
  other_boys = 238 →
  (total_boys - (muslim_percent * total_boys + hindu_percent * total_boys + other_boys : ℚ)) / total_boys * 100 = 10 := by
sorry


end percentage_of_sikh_boys_l3004_300480


namespace lcm_of_9_16_21_l3004_300478

theorem lcm_of_9_16_21 : Nat.lcm 9 (Nat.lcm 16 21) = 1008 := by
  sorry

end lcm_of_9_16_21_l3004_300478


namespace Φ_is_connected_Φ_single_part_l3004_300452

/-- The set of points (x, y) in R^2 satisfying the given system of inequalities -/
def Φ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               Real.sqrt (x^2 - 3*y^2 + 4*x + 4) ≤ 2*x + 1 ∧
               x^2 + y^2 ≤ 4}

/-- Theorem stating that Φ is a connected set -/
theorem Φ_is_connected : IsConnected Φ := by
  sorry

/-- Corollary stating that Φ consists of a single part -/
theorem Φ_single_part : ∃! (S : Set (ℝ × ℝ)), S = Φ ∧ IsConnected S := by
  sorry

end Φ_is_connected_Φ_single_part_l3004_300452


namespace rectangle_y_coordinate_l3004_300466

/-- Given a rectangle with vertices (-8, 1), (1, 1), (1, y), and (-8, y) in a rectangular coordinate system,
    if the area of the rectangle is 72, then y = 9 -/
theorem rectangle_y_coordinate (y : ℝ) : 
  let vertex1 : ℝ × ℝ := (-8, 1)
  let vertex2 : ℝ × ℝ := (1, 1)
  let vertex3 : ℝ × ℝ := (1, y)
  let vertex4 : ℝ × ℝ := (-8, y)
  let length : ℝ := vertex2.1 - vertex1.1
  let width : ℝ := vertex3.2 - vertex2.2
  let area : ℝ := length * width
  area = 72 → y = 9 := by
  sorry

end rectangle_y_coordinate_l3004_300466


namespace box_surface_area_l3004_300417

/-- Calculates the interior surface area of a box formed by removing square corners from a rectangular sheet -/
def interior_surface_area (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- The interior surface area of the box is 731 square units -/
theorem box_surface_area : 
  interior_surface_area 25 35 6 = 731 := by sorry

end box_surface_area_l3004_300417


namespace locus_of_Q_l3004_300494

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (-1, -1)
def C : ℝ × ℝ := (1, 3)

-- Define a point P on line BC
def P : ℝ → ℝ × ℝ := λ t => ((1 - t) * B.1 + t * C.1, (1 - t) * B.2 + t * C.2)

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Define the locus equation
def locus_eq (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- State the theorem
theorem locus_of_Q (t : ℝ) :
  let p := P t
  let q := vec_add p (vec_add (vec_sub A p) (vec_add (vec_sub B p) (vec_sub C p)))
  locus_eq q.1 q.2 := by
  sorry

end locus_of_Q_l3004_300494


namespace cos_95_cos_25_minus_sin_95_sin_25_l3004_300498

theorem cos_95_cos_25_minus_sin_95_sin_25 :
  Real.cos (95 * π / 180) * Real.cos (25 * π / 180) - 
  Real.sin (95 * π / 180) * Real.sin (25 * π / 180) = -1/2 := by
  sorry

end cos_95_cos_25_minus_sin_95_sin_25_l3004_300498


namespace union_complement_equal_l3004_300401

open Set

def U : Finset ℕ := {0,1,2,4,6,8}
def M : Finset ℕ := {0,4,6}
def N : Finset ℕ := {0,1,6}

theorem union_complement_equal : M ∪ (U \ N) = {0,2,4,6,8} := by sorry

end union_complement_equal_l3004_300401


namespace journey_problem_l3004_300411

theorem journey_problem (total_distance : ℝ) (days : ℕ) (ratio : ℝ) 
  (h1 : total_distance = 378)
  (h2 : days = 6)
  (h3 : ratio = 1/2) :
  let first_day := total_distance * (1 - ratio) / (1 - ratio^days)
  first_day * ratio = 96 := by
  sorry

end journey_problem_l3004_300411


namespace angle_TSB_closest_to_27_l3004_300431

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The art gallery setup -/
structure ArtGallery where
  B : Point -- Bottom of painting
  T : Point -- Top of painting
  S : Point -- Spotlight position

/-- Definition of the art gallery setup based on given conditions -/
def setupGallery : ArtGallery :=
  { B := ⟨0, 1⟩,    -- Bottom of painting (0, 1)
    T := ⟨0, 3⟩,    -- Top of painting (0, 3)
    S := ⟨3, 4⟩ }   -- Spotlight position (3, 4)

/-- Calculate the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Theorem stating that the angle TSB is closest to 27° -/
theorem angle_TSB_closest_to_27 (g : ArtGallery) :
  let angleTSB := angle g.T g.S g.B
  ∀ x ∈ [27, 63, 34, 45, 18], |angleTSB - 27| ≤ |angleTSB - x| :=
by sorry

end angle_TSB_closest_to_27_l3004_300431


namespace perpendicular_lines_a_values_l3004_300404

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a + 1) * x + y - 2 = 0
def l₂ (a x y : ℝ) : Prop := a * x + (2 * a + 2) * y + 1 = 0

-- Define perpendicularity of two lines
def perpendicular (a : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ → l₂ a x₂ y₂ → 
    (a + 1) * a = -(2 * a + 2)

-- State the theorem
theorem perpendicular_lines_a_values (a : ℝ) :
  perpendicular a → a = -1 ∨ a = -2 := by sorry

end perpendicular_lines_a_values_l3004_300404


namespace smallest_multiplier_for_three_digit_product_l3004_300412

theorem smallest_multiplier_for_three_digit_product : 
  (∀ k : ℕ, k < 4 → 27 * k < 100) ∧ (27 * 4 ≥ 100 ∧ 27 * 4 < 1000) := by
  sorry

end smallest_multiplier_for_three_digit_product_l3004_300412


namespace complex_point_location_l3004_300416

theorem complex_point_location (x y : ℝ) 
  (h : (x + y) + (y - 1) * Complex.I = (2 * x + 3 * y) + (2 * y + 1) * Complex.I) : 
  x > 0 ∧ y < 0 := by
  sorry

end complex_point_location_l3004_300416


namespace probability_divisible_by_three_l3004_300435

/-- The set of positive integers from 1 to 2007 -/
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 2007}

/-- The probability that a number in S is divisible by 3 -/
def prob_div_3 : ℚ := 669 / 2007

/-- The probability that a number in S is not divisible by 3 -/
def prob_not_div_3 : ℚ := 1338 / 2007

/-- The probability that b and c satisfy the condition when a is not divisible by 3 -/
def prob_bc_condition : ℚ := 2 / 9

theorem probability_divisible_by_three :
  (prob_div_3 + prob_not_div_3 * prob_bc_condition : ℚ) = 1265 / 2007 := by sorry

end probability_divisible_by_three_l3004_300435


namespace unique_rational_solution_l3004_300447

theorem unique_rational_solution (x y z : ℚ) : 
  x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end unique_rational_solution_l3004_300447


namespace sum_of_solutions_l3004_300437

theorem sum_of_solutions : ∃ (S : Finset (ℕ × ℕ)), 
  (∀ (p : ℕ × ℕ), p ∈ S ↔ (p.1 * p.2 = 6 * (p.1 + p.2) ∧ p.1 > 0 ∧ p.2 > 0)) ∧ 
  (S.sum (λ p => p.1 + p.2) = 290) := by
  sorry

end sum_of_solutions_l3004_300437


namespace quadratic_roots_max_reciprocal_sum_l3004_300496

theorem quadratic_roots_max_reciprocal_sum (t q r₁ r₂ : ℝ) : 
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 1003 → r₁^n + r₂^n = r₁ + r₂) →
  r₁ * r₂ = q →
  r₁ + r₂ = t →
  r₁ ≠ 0 →
  r₂ ≠ 0 →
  (1 / r₁^1004 + 1 / r₂^1004) ≤ 2 :=
by sorry

end quadratic_roots_max_reciprocal_sum_l3004_300496


namespace problem_solution_l3004_300481

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := by
  sorry

end problem_solution_l3004_300481


namespace spade_operation_result_l3004_300425

-- Define the spade operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_operation_result : spade 2 (spade 4 7) = 1 := by
  sorry

end spade_operation_result_l3004_300425


namespace height_difference_l3004_300460

/-- Given the heights of three siblings, prove the height difference between two of them. -/
theorem height_difference (cary_height bill_height jan_height : ℕ) :
  cary_height = 72 →
  bill_height = cary_height / 2 →
  jan_height = 42 →
  jan_height - bill_height = 6 := by
  sorry

end height_difference_l3004_300460


namespace log_sum_simplification_l3004_300433

theorem log_sum_simplification : 
  ∀ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 →
  (1 / (Real.log x / Real.log 12 + 1)) + 
  (1 / (Real.log y / Real.log 20 + 1)) + 
  (1 / (Real.log z / Real.log 8 + 1)) = 1.75 :=
by sorry

end log_sum_simplification_l3004_300433


namespace marble_bag_count_l3004_300414

theorem marble_bag_count :
  ∀ (total blue red white : ℕ),
    blue = 5 →
    red = 9 →
    total = blue + red + white →
    (red + white : ℚ) / total = 5 / 6 →
    total = 30 := by
  sorry

end marble_bag_count_l3004_300414


namespace savings_account_growth_l3004_300453

/-- Represents the total amount in a savings account after a given number of months -/
def total_amount (initial_deposit : ℝ) (monthly_rate : ℝ) (months : ℝ) : ℝ :=
  initial_deposit * (1 + monthly_rate * months)

theorem savings_account_growth (x : ℝ) :
  let initial_deposit : ℝ := 100
  let monthly_rate : ℝ := 0.006
  let y : ℝ := total_amount initial_deposit monthly_rate x
  y = 100 * (1 + 0.006 * x) ∧
  total_amount initial_deposit monthly_rate 4 = 102.4 := by
  sorry

#check savings_account_growth

end savings_account_growth_l3004_300453


namespace complex_vector_relation_l3004_300403

theorem complex_vector_relation (z₁ z₂ z₃ : ℂ) (x y : ℝ)
  (h₁ : z₁ = -1 + 2 * Complex.I)
  (h₂ : z₂ = 1 - Complex.I)
  (h₃ : z₃ = 3 - 2 * Complex.I)
  (h₄ : z₃ = x • z₁ + y • z₂) :
  x + y = 5 := by
  sorry

end complex_vector_relation_l3004_300403


namespace remainder_21_pow_2051_mod_29_l3004_300483

theorem remainder_21_pow_2051_mod_29 : 21^2051 % 29 = 15 := by
  sorry

end remainder_21_pow_2051_mod_29_l3004_300483


namespace arithmetic_expression_evaluation_l3004_300475

theorem arithmetic_expression_evaluation : (8 * 6) - (4 / 2) = 46 := by
  sorry

end arithmetic_expression_evaluation_l3004_300475


namespace decreasing_product_function_properties_l3004_300419

/-- A decreasing function f defined on (0, +∞) satisfying f(x) + f(y) = f(xy) -/
def DecreasingProductFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 0 ∧ y > 0 → f x + f y = f (x * y)) ∧
  (∀ x y, x > 0 ∧ y > 0 ∧ x < y → f x > f y)

theorem decreasing_product_function_properties
  (f : ℝ → ℝ)
  (h : DecreasingProductFunction f)
  (h_f4 : f 4 = -4) :
  (∀ x y, x > 0 ∧ y > 0 → f x - f y = f (x / y)) ∧
  (Set.Ioo 12 16 : Set ℝ) = {x | f x - f (1 / (x - 12)) ≥ -12} := by
  sorry

end decreasing_product_function_properties_l3004_300419


namespace parallelogram_sides_sum_l3004_300402

theorem parallelogram_sides_sum (x y : ℚ) : 
  (5 * x - 2 = 10 * x - 4) → 
  (3 * y + 7 = 6 * y + 13) → 
  x + y = -8/5 := by sorry

end parallelogram_sides_sum_l3004_300402


namespace lynn_in_fourth_car_l3004_300463

-- Define the set of people
inductive Person : Type
| Trent : Person
| Jamie : Person
| Eden : Person
| Lynn : Person
| Mira : Person
| Cory : Person

-- Define the seating arrangement
def SeatingArrangement := Fin 6 → Person

-- Define the conditions of the seating arrangement
def ValidArrangement (s : SeatingArrangement) : Prop :=
  -- Trent is in the lead car
  s 0 = Person.Trent ∧
  -- Eden is directly behind Jamie
  (∃ i : Fin 5, s i = Person.Jamie ∧ s (i + 1) = Person.Eden) ∧
  -- Lynn sits ahead of Mira
  (∃ i j : Fin 6, i < j ∧ s i = Person.Lynn ∧ s j = Person.Mira) ∧
  -- Mira is not in the last car
  s 5 ≠ Person.Mira ∧
  -- At least two people sit between Cory and Lynn
  (∃ i j : Fin 6, |i - j| > 2 ∧ s i = Person.Cory ∧ s j = Person.Lynn)

-- The theorem to prove
theorem lynn_in_fourth_car (s : SeatingArrangement) :
  ValidArrangement s → s 3 = Person.Lynn :=
by sorry

end lynn_in_fourth_car_l3004_300463


namespace modulus_of_x_minus_yi_l3004_300436

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def complex_equation (x y : ℝ) : Prop :=
  (x + Real.sqrt 2 * i) / i = y + i

-- Theorem statement
theorem modulus_of_x_minus_yi (x y : ℝ) 
  (h : complex_equation x y) : Complex.abs (x - y * i) = Real.sqrt 3 := by
  sorry

end modulus_of_x_minus_yi_l3004_300436


namespace cost_of_treat_l3004_300406

/-- The cost of dog treats given daily treats, days, and total cost -/
def treat_cost (treats_per_day : ℕ) (days : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (treats_per_day * days)

/-- Theorem: The cost of each treat is $0.10 given the problem conditions -/
theorem cost_of_treat :
  let treats_per_day : ℕ := 2
  let days : ℕ := 30
  let total_cost : ℚ := 6
  treat_cost treats_per_day days total_cost = 1/10 := by
  sorry

end cost_of_treat_l3004_300406


namespace two_digit_number_sum_l3004_300448

theorem two_digit_number_sum (x : ℕ) : 
  x < 10 →                             -- units digit is less than 10
  (11 * x + 30) % (2 * x + 3) = 3 →    -- remainder is 3
  (11 * x + 30) / (2 * x + 3) = 7 →    -- quotient is 7
  2 * x + 3 = 7 :=                     -- sum of digits is 7
by sorry

end two_digit_number_sum_l3004_300448


namespace arithmetic_sequence_ratio_l3004_300487

/-- Given two arithmetic sequences and their sum ratios, prove a specific ratio of their terms -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) 
  (S T : ℕ → ℚ) 
  (h : ∀ n : ℕ, S n / T n = (2 * n - 3 : ℚ) / (4 * n - 1 : ℚ)) :
  (a 3 + a 15) / (2 * (b 3 + b 9)) + a 3 / (b 2 + b 10) = 19 / 43 := by
  sorry

end arithmetic_sequence_ratio_l3004_300487


namespace room_length_proof_l3004_300482

theorem room_length_proof (x : ℝ) 
  (room_width : ℝ) (room_height : ℝ)
  (door_width : ℝ) (door_height : ℝ)
  (large_window_width : ℝ) (large_window_height : ℝ)
  (small_window_width : ℝ) (small_window_height : ℝ)
  (paint_cost_per_sqm : ℝ) (total_paint_cost : ℝ)
  (h1 : room_width = 7)
  (h2 : room_height = 5)
  (h3 : door_width = 1)
  (h4 : door_height = 3)
  (h5 : large_window_width = 2)
  (h6 : large_window_height = 1.5)
  (h7 : small_window_width = 1)
  (h8 : small_window_height = 1.5)
  (h9 : paint_cost_per_sqm = 3)
  (h10 : total_paint_cost = 474)
  (h11 : total_paint_cost = paint_cost_per_sqm * 
    (2 * (x * room_height + room_width * room_height) - 
    2 * (door_width * door_height) - 
    (large_window_width * large_window_height) - 
    2 * (small_window_width * small_window_height))) :
  x = 10 := by
  sorry

end room_length_proof_l3004_300482


namespace complex_sum_problem_l3004_300467

-- Define complex numbers
variable (a b c d e f : ℝ)

-- Define the theorem
theorem complex_sum_problem :
  b = 4 →
  e = -2*a - c →
  (a + b*Complex.I) + (c + d*Complex.I) + (e + f*Complex.I) = 5*Complex.I →
  d + 2*f = 1 := by
  sorry

end complex_sum_problem_l3004_300467


namespace largest_integer_in_interval_l3004_300444

theorem largest_integer_in_interval : ∃ x : ℤ, 
  (1 / 4 : ℚ) < (x : ℚ) / 9 ∧ 
  (x : ℚ) / 9 < (7 / 9 : ℚ) ∧ 
  ∀ y : ℤ, ((1 / 4 : ℚ) < (y : ℚ) / 9 ∧ (y : ℚ) / 9 < (7 / 9 : ℚ)) → y ≤ x :=
by sorry

end largest_integer_in_interval_l3004_300444


namespace magic_shop_cost_correct_l3004_300471

/-- Calculates the total cost for Tom and his friend at the magic shop --/
def magic_shop_cost (trick_deck_price : ℚ) (gimmick_coin_price : ℚ) 
  (trick_deck_count : ℕ) (gimmick_coin_count : ℕ) 
  (trick_deck_discount : ℚ) (gimmick_coin_discount : ℚ) 
  (sales_tax : ℚ) : ℚ :=
  let total_trick_decks := 2 * trick_deck_count * trick_deck_price
  let total_gimmick_coins := 2 * gimmick_coin_count * gimmick_coin_price
  let discounted_trick_decks := 
    if trick_deck_count > 2 then total_trick_decks * (1 - trick_deck_discount) 
    else total_trick_decks
  let discounted_gimmick_coins := 
    if gimmick_coin_count > 3 then total_gimmick_coins * (1 - gimmick_coin_discount) 
    else total_gimmick_coins
  let total_after_discounts := discounted_trick_decks + discounted_gimmick_coins
  let total_with_tax := total_after_discounts * (1 + sales_tax)
  total_with_tax

theorem magic_shop_cost_correct : 
  magic_shop_cost 8 12 3 4 (1/10) (1/20) (7/100) = 14381/100 := by
  sorry

end magic_shop_cost_correct_l3004_300471


namespace smallest_whole_number_satisfying_inequality_two_satisfies_inequality_two_is_smallest_l3004_300421

theorem smallest_whole_number_satisfying_inequality :
  ∀ x : ℤ, (3 * x + 4 > 11 - 2 * x) → x ≥ 2 :=
by sorry

theorem two_satisfies_inequality :
  3 * 2 + 4 > 11 - 2 * 2 :=
by sorry

theorem two_is_smallest :
  ∀ x : ℤ, x < 2 → (3 * x + 4 ≤ 11 - 2 * x) :=
by sorry

end smallest_whole_number_satisfying_inequality_two_satisfies_inequality_two_is_smallest_l3004_300421


namespace power_function_decreasing_m_l3004_300451

/-- A function f: ℝ → ℝ is a power function if it has the form f(x) = ax^b for some constants a and b, where a ≠ 0 -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, x > 0 → f x = a * x ^ b

/-- A function f: ℝ → ℝ is decreasing on (0, +∞) if for any x₁, x₂ ∈ (0, +∞) with x₁ < x₂, we have f(x₁) > f(x₂) -/
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ > f x₂

/-- The main theorem -/
theorem power_function_decreasing_m (m : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (m^2 - m - 1) * x^m
  IsPowerFunction f ∧ IsDecreasingOn f → m = -1 := by
  sorry

end power_function_decreasing_m_l3004_300451


namespace parabola_intersection_l3004_300492

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 2*x - 3

-- Define line l1
def line_l1 (x m : ℝ) : ℝ := -x + m

-- Define the axis of symmetry of the parabola
def axis_of_symmetry : ℝ := -1

-- Define the property that l2 is symmetric with respect to the axis of symmetry
def l2_symmetric (B D : ℝ × ℝ) : Prop :=
  B.1 + D.1 = 2 * axis_of_symmetry

-- Define the condition that A and D are above x-axis, B and C are below
def points_position (A B C D : ℝ × ℝ) : Prop :=
  A.2 > 0 ∧ D.2 > 0 ∧ B.2 < 0 ∧ C.2 < 0

-- Define the condition AC · BD = 26
def product_condition (A B C D : ℝ × ℝ) : Prop :=
  ((A.1 - C.1)^2 + (A.2 - C.2)^2) * ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 26

-- Theorem statement
theorem parabola_intersection (m : ℝ) 
  (A B C D : ℝ × ℝ) 
  (h1 : ∀ x, parabola x = line_l1 x m → (x = A.1 ∨ x = C.1))
  (h2 : l2_symmetric B D)
  (h3 : points_position A B C D)
  (h4 : product_condition A B C D) :
  m = -2 := by sorry

end parabola_intersection_l3004_300492


namespace rounding_down_2A3_l3004_300490

def round_down_to_nearest_ten (n : ℕ) : ℕ :=
  (n / 10) * 10

theorem rounding_down_2A3 (A : ℕ) (h1 : A < 10) :
  (round_down_to_nearest_ten (200 + 10 * A + 3) = 280) → A = 8 := by
  sorry

end rounding_down_2A3_l3004_300490


namespace original_number_proof_l3004_300428

theorem original_number_proof (x : ℕ) : 
  (x + 4) % 23 = 0 → x > 0 → x = 19 := by
sorry

end original_number_proof_l3004_300428


namespace milk_production_l3004_300474

/-- Given that x cows produce y gallons of milk in z days, 
    calculate the amount of milk w cows produce in v days with 10% daily waste. -/
theorem milk_production (x y z w v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (hv : v > 0) :
  let daily_waste : ℝ := 0.1
  let milk_per_cow_per_day : ℝ := y / (z * x)
  let effective_milk_per_cow_per_day : ℝ := milk_per_cow_per_day * (1 - daily_waste)
  effective_milk_per_cow_per_day * w * v = 0.9 * (w * y * v) / (z * x) :=
by sorry

end milk_production_l3004_300474


namespace binomial_18_10_l3004_300409

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 45760 := by
  sorry

end binomial_18_10_l3004_300409


namespace limes_picked_equals_total_l3004_300443

/-- The number of limes picked by Alyssa -/
def alyssas_limes : ℕ := 25

/-- The number of limes picked by Mike -/
def mikes_limes : ℕ := 32

/-- The total number of limes picked -/
def total_limes : ℕ := 57

/-- Theorem: The sum of limes picked by Alyssa and Mike equals the total number of limes picked -/
theorem limes_picked_equals_total : alyssas_limes + mikes_limes = total_limes := by
  sorry

end limes_picked_equals_total_l3004_300443


namespace average_weight_of_class_l3004_300486

theorem average_weight_of_class (group1_count : Nat) (group1_avg : Real) 
  (group2_count : Nat) (group2_avg : Real) :
  group1_count = 22 →
  group2_count = 8 →
  group1_avg = 50.25 →
  group2_avg = 45.15 →
  let total_count := group1_count + group2_count
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  (total_weight / total_count) = 48.89 := by
  sorry

end average_weight_of_class_l3004_300486


namespace fourth_number_in_sequence_l3004_300476

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = a (n - 1) + a (n - 2)

theorem fourth_number_in_sequence
  (a : ℕ → ℕ)
  (h_seq : fibonacci_like_sequence a)
  (h_7 : a 7 = 42)
  (h_9 : a 9 = 110) :
  a 4 = 10 := by
sorry

end fourth_number_in_sequence_l3004_300476


namespace circle_tangent_to_line_l3004_300485

theorem circle_tangent_to_line (m : ℝ) (h : m > 0) :
  ∃ (x y : ℝ), x^2 + y^2 = 4*m ∧ x + y = 2*Real.sqrt m ∧
  ∀ (x' y' : ℝ), x'^2 + y'^2 = 4*m → x' + y' = 2*Real.sqrt m →
  (x' - x)^2 + (y' - y)^2 = 0 ∨ (x' - x)^2 + (y' - y)^2 > 0 :=
by sorry

end circle_tangent_to_line_l3004_300485


namespace mars_mission_cost_share_l3004_300454

-- Define the given conditions
def cost_in_euros : ℝ := 25e9
def number_of_people : ℝ := 300e6
def exchange_rate : ℝ := 1.2

-- Define the theorem to prove
theorem mars_mission_cost_share :
  (cost_in_euros * exchange_rate) / number_of_people = 100 := by
  sorry

end mars_mission_cost_share_l3004_300454


namespace isosceles_triangle_base_angle_with_150_exterior_l3004_300464

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base_angle₁ : ℝ
  base_angle₂ : ℝ
  vertex_angle : ℝ
  is_isosceles : base_angle₁ = base_angle₂
  angle_sum : base_angle₁ + base_angle₂ + vertex_angle = 180

-- Define the exterior angle
def exterior_angle (t : IsoscelesTriangle) : ℝ := 180 - t.vertex_angle

-- Theorem statement
theorem isosceles_triangle_base_angle_with_150_exterior
  (t : IsoscelesTriangle)
  (h : exterior_angle t = 150) :
  t.base_angle₁ = 30 ∨ t.base_angle₁ = 75 :=
by sorry

end isosceles_triangle_base_angle_with_150_exterior_l3004_300464


namespace total_bills_is_126_l3004_300493

/-- Represents the number of bills and their total value -/
structure CashierMoney where
  five_dollar_bills : ℕ
  ten_dollar_bills : ℕ
  total_value : ℕ

/-- Theorem stating that given the conditions, the total number of bills is 126 -/
theorem total_bills_is_126 (money : CashierMoney) 
  (h1 : money.five_dollar_bills = 84)
  (h2 : money.total_value = 840)
  (h3 : money.total_value = 5 * money.five_dollar_bills + 10 * money.ten_dollar_bills) :
  money.five_dollar_bills + money.ten_dollar_bills = 126 := by
  sorry


end total_bills_is_126_l3004_300493


namespace ellipse_parameters_and_eccentricity_l3004_300495

/-- Given an ellipse and a line passing through its vertex and focus, prove the ellipse's parameters and eccentricity. -/
theorem ellipse_parameters_and_eccentricity 
  (a b : ℝ) 
  (h_pos : a > b ∧ b > 0) 
  (h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x - 2*y + 2 = 0 → (x = 0 ∧ y = 1) ∨ (x = -2 ∧ y = 0))) :
  a^2 = 5 ∧ b^2 = 1 ∧ (a^2 - b^2) / a^2 = 4 / 5 := by
  sorry

end ellipse_parameters_and_eccentricity_l3004_300495


namespace b_sixth_congruence_l3004_300479

theorem b_sixth_congruence (n : ℕ+) (b : ℤ) (h : b^3 ≡ 1 [ZMOD n]) :
  b^6 ≡ 1 [ZMOD n] := by
  sorry

end b_sixth_congruence_l3004_300479


namespace wall_area_l3004_300400

/-- Represents the types of tiles used on the wall -/
inductive TileType
  | Small
  | Regular
  | Jumbo

/-- Represents the properties of a tile -/
structure Tile where
  type : TileType
  length : ℝ
  width : ℝ

/-- Represents the wall covered with tiles -/
structure Wall where
  smallTiles : Tile
  regularTiles : Tile
  jumboTiles : Tile
  smallTileProportion : ℝ
  regularTileProportion : ℝ
  jumboTileProportion : ℝ
  regularTileArea : ℝ

/-- Theorem stating that the area of the wall is 300 square feet -/
theorem wall_area (w : Wall) : ℝ :=
  by
  have small_ratio : w.smallTiles.length = 2 * w.smallTiles.width := sorry
  have regular_ratio : w.regularTiles.length = 3 * w.regularTiles.width := sorry
  have jumbo_ratio : w.jumboTiles.length = 3 * w.jumboTiles.width := sorry
  have jumbo_length : w.jumboTiles.length = 3 * w.regularTiles.length := sorry
  have tile_proportions : w.smallTileProportion + w.regularTileProportion + w.jumboTileProportion = 1 := sorry
  have no_overlap : w.smallTileProportion * (300 : ℝ) + w.regularTileProportion * (300 : ℝ) + w.jumboTileProportion * (300 : ℝ) = 300 := sorry
  have regular_area : w.regularTileArea = 90 := sorry
  sorry

#check wall_area

end wall_area_l3004_300400
