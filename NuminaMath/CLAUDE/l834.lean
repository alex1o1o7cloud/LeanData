import Mathlib

namespace series_sum_l834_83432

theorem series_sum : ∑' n, (3 * n - 1 : ℝ) / 2^n = 5 := by sorry

end series_sum_l834_83432


namespace positive_real_power_difference_integer_l834_83496

theorem positive_real_power_difference_integer (x : ℝ) (h1 : x > 0) 
  (h2 : ∃ (a b : ℤ), x^2012 - x^2001 = a ∧ x^2001 - x^1990 = b) : 
  ∃ (n : ℤ), x = n :=
sorry

end positive_real_power_difference_integer_l834_83496


namespace necessary_but_not_sufficient_l834_83435

theorem necessary_but_not_sufficient
  (A B C : Set α)
  (hAnonempty : A.Nonempty)
  (hBnonempty : B.Nonempty)
  (hCnonempty : C.Nonempty)
  (hUnion : A ∪ B = C)
  (hNotSubset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ x, x ∈ C ∧ x ∉ A) := by
  sorry

end necessary_but_not_sufficient_l834_83435


namespace unicorn_rope_length_l834_83411

theorem unicorn_rope_length (rope_length : ℝ) (tower_radius : ℝ) (rope_end_distance : ℝ) 
  (h1 : rope_length = 24)
  (h2 : tower_radius = 10)
  (h3 : rope_end_distance = 6) :
  rope_length - 2 * Real.sqrt (rope_length^2 - tower_radius^2) = 24 - 2 * Real.sqrt 119 :=
by sorry

end unicorn_rope_length_l834_83411


namespace consecutive_integers_product_990_l834_83468

theorem consecutive_integers_product_990 (a b c : ℤ) : 
  b = a + 1 ∧ c = b + 1 ∧ a * b * c = 990 → a + b + c = 30 := by
  sorry

end consecutive_integers_product_990_l834_83468


namespace intersection_M_N_l834_83466

def M : Set ℝ := {x | -4 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - x - 6 < 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end intersection_M_N_l834_83466


namespace apple_cost_calculation_l834_83420

/-- The total cost of apples given weight, price per kg, and packaging fee -/
def total_cost (weight : ℝ) (price_per_kg : ℝ) (packaging_fee : ℝ) : ℝ :=
  weight * (price_per_kg + packaging_fee)

/-- Theorem stating that the total cost of 2.5 kg of apples is 38.875 -/
theorem apple_cost_calculation :
  total_cost 2.5 15.3 0.25 = 38.875 := by
  sorry

end apple_cost_calculation_l834_83420


namespace circle_m_range_l834_83457

-- Define the equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + x + y - m = 0

-- Define what it means for the equation to represent a circle
def represents_circle (m : ℝ) : Prop :=
  ∃ (center_x center_y radius : ℝ), 
    ∀ (x y : ℝ), circle_equation x y m ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2

-- Theorem statement
theorem circle_m_range (m : ℝ) :
  represents_circle m → m > -1/2 := by sorry

end circle_m_range_l834_83457


namespace cos_is_even_and_has_zero_point_l834_83453

-- Define what it means for a function to be even
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define what it means for a function to have a zero point
def HasZeroPoint (f : ℝ → ℝ) : Prop :=
  ∃ x, f x = 0

theorem cos_is_even_and_has_zero_point :
  IsEven Real.cos ∧ HasZeroPoint Real.cos := by sorry

end cos_is_even_and_has_zero_point_l834_83453


namespace min_employees_needed_l834_83484

/-- The minimum number of employees needed for pollution monitoring -/
theorem min_employees_needed (water air soil water_air air_soil water_soil all_three : ℕ)
  (h1 : water = 120)
  (h2 : air = 150)
  (h3 : soil = 100)
  (h4 : water_air = 50)
  (h5 : air_soil = 30)
  (h6 : water_soil = 20)
  (h7 : all_three = 10) :
  water + air + soil - water_air - air_soil - water_soil + all_three = 280 := by
  sorry

end min_employees_needed_l834_83484


namespace triangle_fence_problem_l834_83451

theorem triangle_fence_problem (a b c : ℕ) : 
  a ≤ b → b ≤ c → 
  a + b + c = 2022 → 
  c - b = 1 → 
  b - a = 2 → 
  b = 674 := by
sorry

end triangle_fence_problem_l834_83451


namespace product_of_digits_7891_base7_is_zero_l834_83403

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 7 representation of 7891 is 0 -/
theorem product_of_digits_7891_base7_is_zero :
  productOfList (toBase7 7891) = 0 := by
  sorry

end product_of_digits_7891_base7_is_zero_l834_83403


namespace train_overtake_time_specific_overtake_time_l834_83477

/-- The time it takes for a train to overtake a motorbike -/
theorem train_overtake_time (train_speed : ℝ) (motorbike_speed : ℝ) (train_length : ℝ) : ℝ :=
  let relative_speed_kmph := train_speed - motorbike_speed
  let relative_speed_mps := relative_speed_kmph * 1000 / 3600
  train_length / relative_speed_mps

/-- Proof of the specific overtake time given the problem conditions -/
theorem specific_overtake_time : 
  train_overtake_time 100 64 800.064 = 80.0064 := by
  sorry

end train_overtake_time_specific_overtake_time_l834_83477


namespace equal_ratios_sum_l834_83473

theorem equal_ratios_sum (M N : ℚ) : 
  (5 : ℚ) / 7 = M / 63 ∧ (5 : ℚ) / 7 = 70 / N → M + N = 143 := by
  sorry

end equal_ratios_sum_l834_83473


namespace brianna_marbles_l834_83430

/-- The number of marbles Brianna lost through the hole in the bag. -/
def L : ℕ := sorry

/-- The total number of marbles Brianna started with. -/
def total : ℕ := 24

/-- The number of marbles Brianna had remaining. -/
def remaining : ℕ := 10

theorem brianna_marbles : 
  L + 2 * L + L / 2 = total - remaining ∧ L = 4 := by sorry

end brianna_marbles_l834_83430


namespace min_value_expression_l834_83472

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 3/2) :
  2 * x^2 + 4 * x * y + 9 * y^2 + 10 * y * z + 3 * z^2 ≥ 27 / 2^(4/9) * Real.rpow 90 (1/9) :=
by sorry

end min_value_expression_l834_83472


namespace crayons_in_box_l834_83416

def crayons_problem (given_away lost : ℕ) (difference : ℤ) : Prop :=
  given_away = 90 ∧
  lost = 412 ∧
  difference = lost - given_away ∧
  difference = 322

theorem crayons_in_box (given_away lost : ℕ) (difference : ℤ) 
  (h : crayons_problem given_away lost difference) : 
  given_away + lost = 502 := by
  sorry

end crayons_in_box_l834_83416


namespace alternative_configuration_beats_malfatti_l834_83404

/-- Given an equilateral triangle with side length 1, the total area of three circles
    in an alternative configuration is greater than the total area of Malfatti circles. -/
theorem alternative_configuration_beats_malfatti :
  let malfatti_area : ℝ := 3 * Real.pi * (2 - Real.sqrt 3) / 8
  let alternative_area : ℝ := 11 * Real.pi / 108
  alternative_area > malfatti_area :=
by sorry

end alternative_configuration_beats_malfatti_l834_83404


namespace horner_method_v4_l834_83476

def f (x : ℝ) : ℝ := x^7 - 2*x^6 + 3*x^3 - 4*x^2 + 1

def horner_v4 (x : ℝ) : ℝ := (((x - 2) * x + 0) * x + 0) * x + 3

theorem horner_method_v4 :
  horner_v4 2 = 3 :=
by sorry

end horner_method_v4_l834_83476


namespace smallest_absolute_value_l834_83421

theorem smallest_absolute_value : ∃ x : ℝ, ∀ y : ℝ, abs x ≤ abs y :=
  sorry

end smallest_absolute_value_l834_83421


namespace division_problem_l834_83463

theorem division_problem : (120 : ℝ) / (5 / 2.5) = 60 := by sorry

end division_problem_l834_83463


namespace partnership_profit_l834_83465

/-- Given the investment ratio and B's share of profit, calculate the total profit --/
theorem partnership_profit (a b c : ℕ) (b_share : ℕ) (h1 : a = 6) (h2 : b = 2) (h3 : c = 3) (h4 : b_share = 800) :
  (b_share / b) * (a + b + c) = 4400 := by
  sorry

end partnership_profit_l834_83465


namespace arithmetic_sequence_sum_l834_83450

/-- Given an arithmetic sequence with first term 3 and common difference 12,
    prove that the sum of the first 30 terms is 5310. -/
theorem arithmetic_sequence_sum : 
  let a : ℕ → ℤ := fun n => 3 + (n - 1) * 12
  let S : ℕ → ℤ := fun n => n * (a 1 + a n) / 2
  S 30 = 5310 := by sorry

end arithmetic_sequence_sum_l834_83450


namespace inverse_proportion_problem_l834_83499

/-- Two variables are inversely proportional if their product is constant -/
def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) :
  inversely_proportional x y →
  x + y = 40 →
  x - y = 8 →
  x = 7 →
  y = 54 + 6/7 := by
sorry

end inverse_proportion_problem_l834_83499


namespace area_of_rectangle_l834_83433

/-- The area of rectangle ABCD given the described configuration of squares and triangle -/
theorem area_of_rectangle (
  shaded_square_area : ℝ) 
  (h1 : shaded_square_area = 4) 
  (h2 : ∃ (side : ℝ), side^2 = shaded_square_area) 
  (h3 : ∃ (triangle_height : ℝ), triangle_height = Real.sqrt shaded_square_area) : 
  shaded_square_area + shaded_square_area + (2 * Real.sqrt shaded_square_area * Real.sqrt shaded_square_area / 2) = 12 := by
  sorry

end area_of_rectangle_l834_83433


namespace y_divisibility_l834_83449

def y : ℕ := 48 + 72 + 144 + 216 + 432 + 648 + 2592

theorem y_divisibility :
  (∃ k : ℕ, y = 3 * k) ∧
  (∃ k : ℕ, y = 6 * k) ∧
  ¬(∀ k : ℕ, y = 9 * k) ∧
  ¬(∃ k : ℕ, y = 18 * k) :=
by sorry

end y_divisibility_l834_83449


namespace hyperbola_dot_product_range_l834_83437

/-- The hyperbola with center at origin and left focus at (-2,0) -/
structure Hyperbola where
  a : ℝ
  h_pos : a > 0

/-- A point on the right branch of the hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 = 1
  h_right_branch : x ≥ h.a

/-- The theorem stating the range of the dot product -/
theorem hyperbola_dot_product_range (h : Hyperbola) (p : HyperbolaPoint h) :
  p.x * (p.x + 2) + p.y * p.y ≥ 3 + 2 * Real.sqrt 3 := by sorry

end hyperbola_dot_product_range_l834_83437


namespace cube_squared_equals_sixth_power_l834_83498

theorem cube_squared_equals_sixth_power (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end cube_squared_equals_sixth_power_l834_83498


namespace circle_radius_zero_l834_83494

theorem circle_radius_zero (x y : ℝ) : 
  4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0 → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 0 :=
by sorry

end circle_radius_zero_l834_83494


namespace max_distinct_sum_100_l834_83464

/-- The sum of the first n positive integers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- A function that checks if a number is the maximum number of distinct positive integers that sum to 100 -/
def is_max_distinct_sum (k : ℕ) : Prop :=
  triangular_sum k ≤ 100 ∧ 
  triangular_sum (k + 1) > 100

theorem max_distinct_sum_100 : is_max_distinct_sum 13 := by
  sorry

#check max_distinct_sum_100

end max_distinct_sum_100_l834_83464


namespace min_value_expression_l834_83426

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z)) + 
   1 / ((1 + x) * (1 + y) * (1 + z)) + 
   1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) ≥ 3 ∧
  (1 / ((1 - 0) * (1 - 0) * (1 - 0)) + 
   1 / ((1 + 0) * (1 + 0) * (1 + 0)) + 
   1 / ((1 - 0^2) * (1 - 0^2) * (1 - 0^2))) = 3 :=
by sorry

end min_value_expression_l834_83426


namespace middle_number_8th_row_l834_83438

/-- Represents a number in the array -/
def ArrayNumber (row : ℕ) (position : ℕ) : ℕ := sorry

/-- The number of elements in the nth row -/
def RowLength (n : ℕ) : ℕ := 2 * n - 1

/-- The last number in the nth row -/
def LastNumber (n : ℕ) : ℕ := n ^ 2

/-- The middle position in a row -/
def MiddlePosition (n : ℕ) : ℕ := n

theorem middle_number_8th_row :
  ∀ (row position : ℕ),
  (∀ n : ℕ, LastNumber n = ArrayNumber n (RowLength n)) →
  (∀ n : ℕ, RowLength n = 2 * n - 1) →
  ArrayNumber 8 (MiddlePosition 8) = 57 := by sorry

end middle_number_8th_row_l834_83438


namespace negation_of_forall_proposition_l834_83422

theorem negation_of_forall_proposition :
  (¬ ∀ x : ℝ, x > 2 → x^2 + 2 > 6) ↔ (∃ x : ℝ, x > 2 ∧ x^2 + 2 ≤ 6) := by
  sorry

end negation_of_forall_proposition_l834_83422


namespace tree_planting_problem_l834_83409

theorem tree_planting_problem (a o c m : ℕ) 
  (ha : a = 47)
  (ho : o = 27)
  (hm : m = a * o)
  (hc : c = a - 15) :
  a = 47 ∧ o = 27 ∧ c = 32 ∧ m = 1269 := by
  sorry

end tree_planting_problem_l834_83409


namespace ab_equals_zero_l834_83447

theorem ab_equals_zero (a b : ℤ) (h : |a - b| + |a * b| = 2) : a * b = 0 := by
  sorry

end ab_equals_zero_l834_83447


namespace tan_value_from_double_angle_formula_l834_83497

theorem tan_value_from_double_angle_formula (θ : Real) 
  (h1 : 0 < θ ∧ θ < π) 
  (h2 : Real.sin (2 * θ) = 2 - 2 * Real.cos (2 * θ)) : 
  Real.tan θ = 1/2 := by
  sorry

end tan_value_from_double_angle_formula_l834_83497


namespace intersection_equals_N_implies_t_range_l834_83455

-- Define the sets M and N
def M : Set ℝ := {x | -4 < x ∧ x < 3}
def N (t : ℝ) : Set ℝ := {x | t + 2 < x ∧ x < 2*t - 1}

-- State the theorem
theorem intersection_equals_N_implies_t_range (t : ℝ) : 
  M ∩ N t = N t → t ≤ 3 :=
by sorry

end intersection_equals_N_implies_t_range_l834_83455


namespace chameleon_color_change_l834_83405

/-- The number of chameleons that changed color in the grove --/
def chameleons_changed : ℕ := 80

/-- The total number of chameleons in the grove --/
def total_chameleons : ℕ := 140

/-- The number of blue chameleons after the color change --/
def blue_after : ℕ → ℕ
| n => n

/-- The number of blue chameleons before the color change --/
def blue_before : ℕ → ℕ
| n => 5 * n

/-- The number of red chameleons before the color change --/
def red_before : ℕ → ℕ
| n => total_chameleons - blue_before n

/-- The number of red chameleons after the color change --/
def red_after : ℕ → ℕ
| n => 3 * (red_before n)

theorem chameleon_color_change :
  ∃ n : ℕ, 
    blue_after n + red_after n = total_chameleons ∧ 
    chameleons_changed = blue_before n - blue_after n :=
  sorry

end chameleon_color_change_l834_83405


namespace angle_between_vectors_is_acute_l834_83489

theorem angle_between_vectors_is_acute (A B C : ℝ) (p q : ℝ × ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  p = (Real.cos A, Real.sin A) →
  q = (-Real.cos B, Real.sin B) →
  ∃ α, 0 < α ∧ α < π/2 ∧ Real.cos α = p.1 * q.1 + p.2 * q.2 := by
  sorry

end angle_between_vectors_is_acute_l834_83489


namespace tree_planting_seedlings_l834_83458

theorem tree_planting_seedlings : 
  ∃ (x : ℕ), 
    (∃ (n : ℕ), x - 6 = 5 * n) ∧ 
    (∃ (m : ℕ), x + 9 = 6 * m) ∧ 
    x = 81 := by
  sorry

end tree_planting_seedlings_l834_83458


namespace net_loss_calculation_l834_83479

def initial_value : ℝ := 12000
def loss_percentage : ℝ := 0.1
def gain_percentage : ℝ := 0.15

def first_sale_price : ℝ := initial_value * (1 - loss_percentage)
def second_sale_price : ℝ := first_sale_price * (1 + gain_percentage)

theorem net_loss_calculation :
  second_sale_price - initial_value = 420 := by sorry

end net_loss_calculation_l834_83479


namespace class_ratio_problem_l834_83482

theorem class_ratio_problem (total : ℕ) (boys : ℕ) (h_total : total > 0) (h_boys : boys ≤ total) :
  let p_boy := boys / total
  let p_girl := (total - boys) / total
  (p_boy = (2 : ℚ) / 3 * p_girl) → (boys : ℚ) / total = 2 / 5 := by
  sorry

end class_ratio_problem_l834_83482


namespace min_colors_tessellation_l834_83415

/-- Represents a tile in the tessellation -/
inductive Tile
| Triangle
| Trapezoid

/-- Represents a color used in the tessellation -/
inductive Color
| Red
| Green
| Blue

/-- Represents the tessellation as a function from coordinates to tiles -/
def Tessellation := ℕ → ℕ → Tile

/-- A valid tessellation alternates between rows of triangles and trapezoids -/
def isValidTessellation (t : Tessellation) : Prop :=
  ∀ i j, t i j = if i % 2 = 0 then Tile.Triangle else Tile.Trapezoid

/-- A coloring of the tessellation -/
def Coloring := ℕ → ℕ → Color

/-- Checks if two tiles are adjacent -/
def isAdjacent (i1 j1 i2 j2 : ℕ) : Prop :=
  (i1 = i2 ∧ j1 + 1 = j2) ∨ 
  (i1 + 1 = i2 ∧ j1 = j2) ∨ 
  (i1 + 1 = i2 ∧ j1 + 1 = j2)

/-- A valid coloring ensures no adjacent tiles have the same color -/
def isValidColoring (t : Tessellation) (c : Coloring) : Prop :=
  ∀ i1 j1 i2 j2, isAdjacent i1 j1 i2 j2 → c i1 j1 ≠ c i2 j2

/-- The main theorem: 3 colors are sufficient and necessary -/
theorem min_colors_tessellation (t : Tessellation) (h : isValidTessellation t) :
  (∃ c : Coloring, isValidColoring t c) ∧ 
  (∀ c : Coloring, isValidColoring t c → ∃ (x y z : Color), 
    (∀ i j, c i j = x ∨ c i j = y ∨ c i j = z)) :=
sorry

end min_colors_tessellation_l834_83415


namespace rectangle_area_l834_83408

/-- The area of a rectangle with length twice its width and perimeter equal to a triangle with sides 7, 10, and 11 is 392/9 -/
theorem rectangle_area (w : ℝ) (h : 2 * (2 * w + w) = 7 + 10 + 11) : w * (2 * w) = 392 / 9 := by
  sorry

end rectangle_area_l834_83408


namespace probability_divisible_by_10_and_5_l834_83440

def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def DivisibleBy10 (n : ℕ) : Prop := n % 10 = 0

def DivisibleBy5 (n : ℕ) : Prop := n % 5 = 0

def CountTwoDigitNumbers : ℕ := 90

def CountTwoDigitDivisibleBy10 : ℕ := 9

theorem probability_divisible_by_10_and_5 :
  (CountTwoDigitDivisibleBy10 : ℚ) / CountTwoDigitNumbers = 1 / 10 := by sorry

end probability_divisible_by_10_and_5_l834_83440


namespace least_prime_factor_of_5_cubed_minus_5_squared_l834_83402

theorem least_prime_factor_of_5_cubed_minus_5_squared : 
  (Nat.minFac (5^3 - 5^2) = 2) := by sorry

end least_prime_factor_of_5_cubed_minus_5_squared_l834_83402


namespace total_profit_is_3872_l834_83492

/-- Represents the investment and duration for each person -/
structure Investment where
  amount : ℕ
  duration : ℕ

/-- Calculates the total profit given the investments and profit difference -/
def calculateTotalProfit (suresh rohan sudhir : Investment) (profitDifference : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the total profit is 3872 given the problem conditions -/
theorem total_profit_is_3872 :
  let suresh : Investment := ⟨18000, 12⟩
  let rohan : Investment := ⟨12000, 9⟩
  let sudhir : Investment := ⟨9000, 8⟩
  let profitDifference : ℕ := 352
  calculateTotalProfit suresh rohan sudhir profitDifference = 3872 :=
by sorry

end total_profit_is_3872_l834_83492


namespace trajectory_is_line_segment_l834_83425

/-- The trajectory of a point P(x,y) satisfying |PF₁| + |PF₂| = 10, where F₁(-5,0) and F₂(5,0) are fixed points, is a line segment. -/
theorem trajectory_is_line_segment :
  ∀ (x y : ℝ),
  let P : ℝ × ℝ := (x, y)
  let F₁ : ℝ × ℝ := (-5, 0)
  let F₂ : ℝ × ℝ := (5, 0)
  let dist (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  dist P F₁ + dist P F₂ = 10 →
  ∃ (A B : ℝ × ℝ), P ∈ Set.Icc A B :=
by sorry

end trajectory_is_line_segment_l834_83425


namespace olivia_chocolate_sales_l834_83443

def chocolate_problem (cost_per_bar total_bars unsold_bars : ℕ) : Prop :=
  let sold_bars := total_bars - unsold_bars
  let money_made := sold_bars * cost_per_bar
  money_made = 9

theorem olivia_chocolate_sales : chocolate_problem 3 7 4 := by
  sorry

end olivia_chocolate_sales_l834_83443


namespace sum_of_extrema_equals_two_l834_83431

-- Define the function f(x) = x ln |x| + 1
noncomputable def f (x : ℝ) : ℝ := x * Real.log (abs x) + 1

-- Theorem statement
theorem sum_of_extrema_equals_two :
  ∃ (max_val min_val : ℝ),
    (∀ x, f x ≤ max_val) ∧
    (∃ x, f x = max_val) ∧
    (∀ x, f x ≥ min_val) ∧
    (∃ x, f x = min_val) ∧
    max_val + min_val = 2 := by
  sorry

end sum_of_extrema_equals_two_l834_83431


namespace larry_cards_l834_83434

theorem larry_cards (initial_cards final_cards taken_cards : ℕ) : 
  final_cards = initial_cards - taken_cards → 
  taken_cards = 9 → 
  final_cards = 58 → 
  initial_cards = 67 := by
sorry

end larry_cards_l834_83434


namespace jessica_seashells_l834_83462

theorem jessica_seashells (joan_shells jessica_shells total_shells : ℕ) 
  (h1 : joan_shells = 6)
  (h2 : total_shells = 14)
  (h3 : joan_shells + jessica_shells = total_shells) :
  jessica_shells = 8 := by
  sorry

end jessica_seashells_l834_83462


namespace horse_speed_around_square_field_l834_83461

/-- Given a square field with area 625 km^2 and a horse that runs around it in 4 hours,
    prove that the speed of the horse is 25 km/hour. -/
theorem horse_speed_around_square_field (area : ℝ) (time : ℝ) (horse_speed : ℝ) : 
  area = 625 → time = 4 → horse_speed = (4 * Real.sqrt area) / time → horse_speed = 25 := by
  sorry

end horse_speed_around_square_field_l834_83461


namespace largest_prime_factor_of_6889_l834_83406

theorem largest_prime_factor_of_6889 : ∃ p : ℕ, p.Prime ∧ p ∣ 6889 ∧ ∀ q : ℕ, q.Prime → q ∣ 6889 → q ≤ p :=
by sorry

end largest_prime_factor_of_6889_l834_83406


namespace terrys_spending_ratio_l834_83485

/-- Terry's spending problem -/
theorem terrys_spending_ratio :
  ∀ (monday tuesday wednesday total : ℚ),
    monday = 6 →
    tuesday = 2 * monday →
    total = monday + tuesday + wednesday →
    total = 54 →
    wednesday = 2 * (monday + tuesday) :=
by sorry

end terrys_spending_ratio_l834_83485


namespace investment_loss_l834_83444

/-- Given two investors with capitals in ratio 1:9 and proportional loss distribution,
    if one investor's loss is 603, then the total loss is 670. -/
theorem investment_loss (capital_ratio : ℚ) (investor1_loss : ℚ) (total_loss : ℚ) :
  capital_ratio = 1 / 9 →
  investor1_loss = 603 →
  total_loss = investor1_loss / (capital_ratio / (capital_ratio + 1)) →
  total_loss = 670 := by
  sorry

end investment_loss_l834_83444


namespace A_intersect_B_equals_three_l834_83417

def A : Set ℝ := {0, 1, 3}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

theorem A_intersect_B_equals_three : A ∩ B = {3} := by sorry

end A_intersect_B_equals_three_l834_83417


namespace expression_simplification_l834_83436

theorem expression_simplification (a : ℝ) : 
  a^3 * a^5 + (a^2)^4 + (-2*a^4)^2 - 10*a^10 / (5*a^2) = 4*a^8 := by
  sorry

end expression_simplification_l834_83436


namespace power_product_equality_l834_83428

theorem power_product_equality (a b : ℝ) : (2 * a^2 * b)^3 = 8 * a^6 * b^3 := by
  sorry

end power_product_equality_l834_83428


namespace smallest_quadratic_coefficient_l834_83491

theorem smallest_quadratic_coefficient (a : ℕ) : 
  (∃ (b c : ℤ) (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    0 < x₁ ∧ x₁ < 1 ∧ 
    0 < x₂ ∧ x₂ < 1 ∧ 
    (a : ℝ) * x₁^2 + (b : ℝ) * x₁ + (c : ℝ) = 0 ∧ 
    (a : ℝ) * x₂^2 + (b : ℝ) * x₂ + (c : ℝ) = 0) →
  a ≥ 5 :=
sorry

end smallest_quadratic_coefficient_l834_83491


namespace add_like_terms_l834_83481

theorem add_like_terms (a : ℝ) : 3 * a + 2 * a = 5 * a := by
  sorry

end add_like_terms_l834_83481


namespace eighth_power_sum_l834_83486

theorem eighth_power_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) : 
  a^8 + b^8 = 47 := by
  sorry

end eighth_power_sum_l834_83486


namespace cubic_roots_same_abs_value_iff_l834_83471

-- Define the polynomial type
def CubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

-- Define the property that all roots have the same absolute value
def AllRootsSameAbsValue (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ z : ℂ, f z.re = 0 → Complex.abs z = k

-- Theorem statement
theorem cubic_roots_same_abs_value_iff (a b c : ℝ) :
  AllRootsSameAbsValue (CubicPolynomial a b c) → (a = 0 ↔ b = 0) := by
  sorry

end cubic_roots_same_abs_value_iff_l834_83471


namespace triangle_theorem_l834_83441

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) (h : 4 * t.a^2 = t.b * t.c * Real.cos t.A + t.a * t.c * Real.cos t.B) :
  (t.a / t.c = 1 / 2) ∧
  (t.a = 1 → Real.cos t.B = 3 / 4 → ∃ D : ℝ × ℝ, 
    (D.1 = (t.a + t.c) / 2 ∧ D.2 = 0) → 
    Real.sqrt ((D.1 - t.a)^2 + D.2^2) = Real.sqrt 2) := by
  sorry

end triangle_theorem_l834_83441


namespace good_bulbs_count_l834_83488

def total_bulbs : ℕ := 10
def num_lamps : ℕ := 3
def prob_lighted : ℚ := 29/30

def num_good_bulbs : ℕ := 6

theorem good_bulbs_count :
  (1 : ℚ) - (Nat.choose (total_bulbs - num_good_bulbs) num_lamps : ℚ) / (Nat.choose total_bulbs num_lamps) = prob_lighted :=
sorry

end good_bulbs_count_l834_83488


namespace simplify_complex_fraction_l834_83410

theorem simplify_complex_fraction : 
  (1 / ((1 / (Real.sqrt 3 + 1)) + (2 / (Real.sqrt 5 - 2)))) = Real.sqrt 3 - 2 * Real.sqrt 5 - 3 := by
  sorry

end simplify_complex_fraction_l834_83410


namespace number_equation_proof_l834_83407

theorem number_equation_proof (x : ℤ) : 
  x - (28 - (37 - (15 - 15))) = 54 → x = 45 := by
  sorry

end number_equation_proof_l834_83407


namespace g_expression_l834_83445

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the property of g being a linear function
def is_linear (g : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, g x = a * x + b

-- State the theorem
theorem g_expression (g : ℝ → ℝ) (h_linear : is_linear g) 
    (h_comp : ∀ x, f (g x) = 4 * x^2) :
  (∀ x, g x = 2 * x + 1) ∨ (∀ x, g x = -2 * x + 1) := by sorry

end g_expression_l834_83445


namespace linear_equation_transformation_l834_83469

theorem linear_equation_transformation (x y : ℝ) :
  (3 * x + 4 * y = 5) ↔ (x = (5 - 4 * y) / 3) :=
by sorry

end linear_equation_transformation_l834_83469


namespace teachers_distribution_arrangements_l834_83459

/-- The number of ways to distribute teachers between two classes -/
def distribute_teachers (total_teachers : ℕ) (max_per_class : ℕ) : ℕ :=
  let equal_distribution := 1
  let unequal_distribution := 2 * (Nat.choose total_teachers max_per_class)
  equal_distribution + unequal_distribution

/-- Theorem stating that distributing 6 teachers with a maximum of 4 per class results in 31 arrangements -/
theorem teachers_distribution_arrangements :
  distribute_teachers 6 4 = 31 := by
  sorry

end teachers_distribution_arrangements_l834_83459


namespace sum_divisible_by_ten_l834_83495

theorem sum_divisible_by_ten (n : ℕ) : 
  10 ∣ (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) ↔ n % 5 = 1 := by
sorry

end sum_divisible_by_ten_l834_83495


namespace exponentiation_distributive_multiplication_multiplication_not_distributive_exponentiation_l834_83400

theorem exponentiation_distributive_multiplication (a b c : ℝ) :
  (a * b) ^ c = a ^ c * b ^ c :=
sorry

theorem multiplication_not_distributive_exponentiation :
  ∃ a b c : ℝ, (a ^ b) * c ≠ (a * c) ^ (b * c) :=
sorry

end exponentiation_distributive_multiplication_multiplication_not_distributive_exponentiation_l834_83400


namespace regular_ngon_parallel_pairs_l834_83439

/-- Represents a regular n-gon with a connected path visiting each vertex exactly once -/
structure RegularNGonPath (n : ℕ) where
  path : List ℕ
  is_valid : path.length = n ∧ path.toFinset.card = n

/-- Two edges (i, j) and (p, q) are parallel in a regular n-gon if and only if i + j ≡ p + q (mod n) -/
def parallel_edges (n : ℕ) (i j p q : ℕ) : Prop :=
  (i + j) % n = (p + q) % n

/-- Counts the number of parallel pairs in a path -/
def count_parallel_pairs (n : ℕ) (path : RegularNGonPath n) : ℕ :=
  sorry

theorem regular_ngon_parallel_pairs (n : ℕ) (path : RegularNGonPath n) :
  (Even n → count_parallel_pairs n path > 0) ∧
  (Odd n → count_parallel_pairs n path ≠ 1) :=
sorry

end regular_ngon_parallel_pairs_l834_83439


namespace geometric_sequence_304th_term_l834_83401

/-- Given a geometric sequence with first term 8 and second term -8, the 304th term is -8 -/
theorem geometric_sequence_304th_term :
  ∀ (a : ℕ → ℝ), 
    a 1 = 8 →  -- First term is 8
    a 2 = -8 →  -- Second term is -8
    (∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence property
    a 304 = -8 := by
  sorry

end geometric_sequence_304th_term_l834_83401


namespace min_questions_for_phone_number_l834_83480

theorem min_questions_for_phone_number (n : ℕ) (h : n = 100000) :
  ∃ k : ℕ, k = 17 ∧ 2^k ≥ n ∧ ∀ m : ℕ, m < k → 2^m < n :=
by sorry

end min_questions_for_phone_number_l834_83480


namespace lcm_problem_l834_83446

theorem lcm_problem (a b : ℕ+) (h : Nat.gcd a b = 9) (p : a * b = 1800) :
  Nat.lcm a b = 200 := by
  sorry

end lcm_problem_l834_83446


namespace max_lessons_with_clothing_constraints_l834_83475

/-- The maximum number of lessons an instructor can conduct given specific clothing constraints -/
theorem max_lessons_with_clothing_constraints :
  ∀ (x y z : ℕ),
  (x > 0) → (y > 0) → (z > 0) →
  (3 * y * z = 18) →
  (3 * x * z = 63) →
  (3 * x * y = 42) →
  (3 * x * y * z = 126) := by
sorry

end max_lessons_with_clothing_constraints_l834_83475


namespace functional_equation_solution_l834_83423

-- Define the function type
def FunctionType := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : FunctionType) 
  (h1 : ∀ a b : ℝ, f (a + b) + f (a - b) = 3 * f a + f b) 
  (h2 : f 1 = 1) : 
  ∀ x : ℝ, f x = if x = 1 then 1 else 0 := by
  sorry


end functional_equation_solution_l834_83423


namespace num_colorings_is_162_l834_83429

/-- Represents the three colors available for coloring --/
inductive Color
| Red
| White
| Blue

/-- Represents a coloring of a single triangle --/
structure TriangleColoring :=
  (a b c : Color)
  (different_colors : a ≠ b ∧ b ≠ c ∧ a ≠ c)

/-- Represents a coloring of the entire figure (four triangles) --/
structure FigureColoring :=
  (t1 t2 t3 t4 : TriangleColoring)
  (connected_different : t1.c = t2.a ∧ t2.c = t3.a ∧ t3.c = t4.a)

/-- The number of valid colorings for the figure --/
def num_colorings : ℕ := sorry

/-- Theorem stating that the number of valid colorings is 162 --/
theorem num_colorings_is_162 : num_colorings = 162 := by sorry

end num_colorings_is_162_l834_83429


namespace cos_seventh_power_decomposition_l834_83452

theorem cos_seventh_power_decomposition :
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ),
    (∀ θ : ℝ, (Real.cos θ)^7 = b₁ * Real.cos θ + b₂ * Real.cos (2*θ) + b₃ * Real.cos (3*θ) + 
                               b₄ * Real.cos (4*θ) + b₅ * Real.cos (5*θ) + b₆ * Real.cos (6*θ) + 
                               b₇ * Real.cos (7*θ)) ∧
    (b₁ = 35/64 ∧ b₂ = 0 ∧ b₃ = 21/64 ∧ b₄ = 0 ∧ b₅ = 7/64 ∧ b₆ = 0 ∧ b₇ = 1/64) ∧
    (b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 + b₇^2 = 1785/4096) := by
  sorry

end cos_seventh_power_decomposition_l834_83452


namespace reciprocal_square_inequality_l834_83414

theorem reciprocal_square_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≤ y) : 
  1 / y^2 ≤ 1 / x^2 := by
sorry

end reciprocal_square_inequality_l834_83414


namespace complex_absolute_value_l834_83442

/-- Given that ω = 10 + 3i, prove that |ω² + 10ω + 104| = 212 -/
theorem complex_absolute_value (ω : ℂ) (h : ω = 10 + 3*I) :
  Complex.abs (ω^2 + 10*ω + 104) = 212 := by
  sorry

end complex_absolute_value_l834_83442


namespace binary_ternary_equality_l834_83419

theorem binary_ternary_equality (a b : ℕ) : 
  a ∈ ({0, 1, 2} : Set ℕ) → 
  b ∈ ({0, 1} : Set ℕ) → 
  (8 + 2 * b + 1 = 9 * a + 2) → 
  (a = 1 ∧ b = 1) :=
by sorry

end binary_ternary_equality_l834_83419


namespace exists_set_without_triangle_l834_83470

/-- A set of 10 segment lengths --/
def SegmentSet : Type := Fin 10 → ℝ

/-- Predicate to check if three segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Theorem stating that there exists a set of 10 segments where no three can form a triangle --/
theorem exists_set_without_triangle : 
  ∃ (s : SegmentSet), ∀ (i j k : Fin 10), i ≠ j → j ≠ k → i ≠ k → 
    ¬(can_form_triangle (s i) (s j) (s k)) := by
  sorry

end exists_set_without_triangle_l834_83470


namespace geometric_series_sum_l834_83493

/-- The sum of a geometric series with 6 terms, first term a, and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) : ℚ :=
  a * (1 - r^6) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum :
  let a : ℚ := 1/5
  let r : ℚ := -1/2
  geometric_sum a r = 21/160 := by
sorry

end geometric_series_sum_l834_83493


namespace cubic_equation_solution_l834_83460

theorem cubic_equation_solution (p q : ℝ) : 
  (3 * p^2 - 5 * p - 8 = 0) → 
  (3 * q^2 - 5 * q - 8 = 0) → 
  p ≠ q →
  (5 * p^3 - 5 * q^3) / (p - q) = 245 / 9 := by
  sorry

end cubic_equation_solution_l834_83460


namespace roof_ratio_l834_83418

/-- Proves that a rectangular roof with given area and length-width difference has a specific length-to-width ratio -/
theorem roof_ratio (length width : ℝ) 
  (area_eq : length * width = 675)
  (diff_eq : length - width = 30) :
  length / width = 3 := by
sorry

end roof_ratio_l834_83418


namespace tetrahedron_vector_equality_l834_83467

-- Define the tetrahedron O-ABC
variable (O A B C : EuclideanSpace ℝ (Fin 3))

-- Define vectors a, b, c
variable (a b c : EuclideanSpace ℝ (Fin 3))

-- Define points M and N
variable (M N : EuclideanSpace ℝ (Fin 3))

-- State the theorem
theorem tetrahedron_vector_equality 
  (h1 : A - O = a) 
  (h2 : B - O = b) 
  (h3 : C - O = c) 
  (h4 : M - O = (2/3) • (A - O)) 
  (h5 : N - O = (1/2) • (B - O) + (1/2) • (C - O)) :
  M - N = (1/2) • b + (1/2) • c - (2/3) • a := by sorry

end tetrahedron_vector_equality_l834_83467


namespace handshake_theorem_l834_83413

theorem handshake_theorem (n : ℕ) (total_handshakes : ℕ) :
  n = 10 →
  total_handshakes = 45 →
  total_handshakes = n * (n - 1) / 2 →
  ∀ boy : Fin n, (n - 1 : ℕ) = total_handshakes / n :=
by sorry

end handshake_theorem_l834_83413


namespace worker_travel_time_l834_83474

/-- If a worker walking at 5/6 of her normal speed arrives 12 minutes later than usual, 
    then her usual time to reach the office is 60 minutes. -/
theorem worker_travel_time (S : ℝ) (T : ℝ) (h1 : S > 0) (h2 : T > 0) : 
  S * T = (5/6 * S) * (T + 12) → T = 60 := by
  sorry

end worker_travel_time_l834_83474


namespace eleven_by_eleven_grid_segment_length_l834_83448

/-- Represents a grid of lattice points -/
structure LatticeGrid where
  rows : ℕ
  columns : ℕ

/-- Calculates the total length of segments in a lattice grid -/
def totalSegmentLength (grid : LatticeGrid) : ℕ :=
  (grid.rows - 1) * grid.columns + (grid.columns - 1) * grid.rows

/-- Theorem: The total length of segments in an 11x11 lattice grid is 220 -/
theorem eleven_by_eleven_grid_segment_length :
  totalSegmentLength ⟨11, 11⟩ = 220 := by
  sorry

#eval totalSegmentLength ⟨11, 11⟩

end eleven_by_eleven_grid_segment_length_l834_83448


namespace pencil_count_l834_83490

theorem pencil_count (initial : ℕ) (nancy_added : ℕ) (steven_added : ℕ)
  (h1 : initial = 138)
  (h2 : nancy_added = 256)
  (h3 : steven_added = 97) :
  initial + nancy_added + steven_added = 491 :=
by sorry

end pencil_count_l834_83490


namespace elder_age_is_30_l834_83478

/-- The age difference between two persons -/
def age_difference : ℕ := 16

/-- The number of years ago when the elder was 3 times as old as the younger -/
def years_ago : ℕ := 6

/-- The present age of the younger person -/
def younger_age : ℕ := 14

/-- The present age of the elder person -/
def elder_age : ℕ := younger_age + age_difference

theorem elder_age_is_30 :
  (elder_age - years_ago = 3 * (younger_age - years_ago)) →
  elder_age = 30 :=
by sorry

end elder_age_is_30_l834_83478


namespace rectangle_p_value_l834_83483

/-- Rectangle PQRS with given vertices and area -/
structure Rectangle where
  P : ℝ × ℝ
  S : ℝ × ℝ
  Q : ℝ × ℝ
  area : ℝ

/-- The theorem stating that if a rectangle PQRS has the given properties, then p = 15 -/
theorem rectangle_p_value (rect : Rectangle)
  (h1 : rect.P = (2, 3))
  (h2 : rect.S = (12, 3))
  (h3 : rect.Q.2 = 15)
  (h4 : rect.area = 120) :
  rect.Q.1 = 15 := by
  sorry

end rectangle_p_value_l834_83483


namespace shadow_problem_l834_83487

/-- Given a cube with edge length 2 cm and a point light source x cm above an upper vertex,
    if the shadow area (excluding the area beneath the cube) is 192 sq cm,
    then the greatest integer not exceeding 1000x is 12000. -/
theorem shadow_problem (x : ℝ) : 
  let cube_edge : ℝ := 2
  let shadow_area : ℝ := 192
  let total_shadow_area : ℝ := shadow_area + cube_edge^2
  let shadow_side : ℝ := Real.sqrt total_shadow_area
  x = cube_edge * (shadow_side - cube_edge) / cube_edge →
  Int.floor (1000 * x) = 12000 := by
sorry

end shadow_problem_l834_83487


namespace company_match_percentage_l834_83454

/-- Proves that the company's 401K match percentage is 6% given the problem conditions --/
theorem company_match_percentage (
  paychecks_per_year : ℕ)
  (contribution_per_paycheck : ℚ)
  (total_contribution : ℚ)
  (h1 : paychecks_per_year = 26)
  (h2 : contribution_per_paycheck = 100)
  (h3 : total_contribution = 2756) :
  (total_contribution - (paychecks_per_year : ℚ) * contribution_per_paycheck) /
  ((paychecks_per_year : ℚ) * contribution_per_paycheck) * 100 = 6 :=
by sorry

end company_match_percentage_l834_83454


namespace min_value_fraction_l834_83456

theorem min_value_fraction (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_one : x + y + z + w = 1) :
  (x + y) / (x * y * z * w) ≥ 108 ∧ 
  ∃ x y z w, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧ 
    x + y + z + w = 1 ∧ (x + y) / (x * y * z * w) = 108 := by
  sorry

end min_value_fraction_l834_83456


namespace function_bound_l834_83412

theorem function_bound (x : ℝ) : 
  1/2 ≤ (x^2 + x + 1) / (x^2 + 1) ∧ (x^2 + x + 1) / (x^2 + 1) ≤ 3/2 := by
  sorry

end function_bound_l834_83412


namespace arithmetic_sequence_sum_l834_83427

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum1 : a 1 + a 2 + a 3 = -24) (h_sum2 : a 18 + a 19 + a 20 = 78) : 
  a 1 + a 20 = 18 := by
sorry

end arithmetic_sequence_sum_l834_83427


namespace fundraiser_group_composition_l834_83424

theorem fundraiser_group_composition (initial_total : ℕ) : 
  let initial_girls : ℕ := (initial_total * 3) / 10
  let final_total : ℕ := initial_total
  let final_girls : ℕ := initial_girls - 3
  (initial_girls : ℚ) / initial_total = 3 / 10 →
  (final_girls : ℚ) / final_total = 1 / 4 →
  initial_girls = 18 :=
by
  sorry

#check fundraiser_group_composition

end fundraiser_group_composition_l834_83424
