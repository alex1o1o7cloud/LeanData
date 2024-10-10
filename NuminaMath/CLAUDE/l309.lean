import Mathlib

namespace units_digit_of_27_times_46_l309_30956

theorem units_digit_of_27_times_46 : (27 * 46) % 10 = 2 := by
  sorry

end units_digit_of_27_times_46_l309_30956


namespace fifteenth_term_ratio_l309_30992

/-- Definition of an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℚ
  diff : ℚ

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.first + (n - 1) * seq.diff) / 2

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.first + (n - 1) * seq.diff

theorem fifteenth_term_ratio 
  (seq1 seq2 : ArithmeticSequence) 
  (h : ∀ n : ℕ, sum_n seq1 n / sum_n seq2 n = (5 * n + 3) / (3 * n + 35)) : 
  nth_term seq1 15 / nth_term seq2 15 = 59 / 57 := by
  sorry

end fifteenth_term_ratio_l309_30992


namespace jacob_future_age_l309_30919

-- Define Jacob's current age
def jacob_age : ℕ := sorry

-- Define Michael's current age
def michael_age : ℕ := sorry

-- Define the number of years from now
variable (X : ℕ)

-- Jacob is 14 years younger than Michael
axiom age_difference : jacob_age = michael_age - 14

-- In 9 years, Michael will be twice as old as Jacob
axiom future_age_relation : michael_age + 9 = 2 * (jacob_age + 9)

-- Theorem: Jacob's age in X years from now is 5 + X
theorem jacob_future_age : jacob_age + X = 5 + X := by sorry

end jacob_future_age_l309_30919


namespace equation_solution_l309_30918

theorem equation_solution : ∃ y : ℝ, y^4 - 20*y + 1 = 22 ∧ y = -1 := by
  sorry

end equation_solution_l309_30918


namespace max_c_trees_l309_30964

/-- Represents the types of scenic trees -/
inductive TreeType
| A
| B
| C

/-- The price of a tree given its type -/
def price (t : TreeType) : ℕ :=
  match t with
  | TreeType.A => 200
  | TreeType.B => 200
  | TreeType.C => 300

/-- The total budget for purchasing trees -/
def total_budget : ℕ := 220120

/-- The total number of trees to be purchased -/
def total_trees : ℕ := 1000

/-- Theorem stating the maximum number of C-type trees that can be purchased -/
theorem max_c_trees :
  (∃ (a b c : ℕ), a + b + c = total_trees ∧
                   a * price TreeType.A + b * price TreeType.B + c * price TreeType.C ≤ total_budget) →
  (∀ (a b c : ℕ), a + b + c = total_trees →
                   a * price TreeType.A + b * price TreeType.B + c * price TreeType.C ≤ total_budget →
                   c ≤ 201) ∧
  (∃ (a b : ℕ), a + b + 201 = total_trees ∧
                 a * price TreeType.A + b * price TreeType.B + 201 * price TreeType.C ≤ total_budget) :=
by sorry


end max_c_trees_l309_30964


namespace store_sales_increase_l309_30966

/-- Represents a store's sales performance --/
structure StoreSales where
  original_price : ℝ
  original_quantity : ℝ
  discount_rate : ℝ
  quantity_increase_rate : ℝ

/-- Calculates the percentage change in gross income --/
def gross_income_change (s : StoreSales) : ℝ :=
  ((1 + s.quantity_increase_rate) * (1 - s.discount_rate) - 1) * 100

/-- Theorem: If a store applies a 10% discount and experiences a 15% increase in sales quantity,
    then the gross income increases by 3.5% --/
theorem store_sales_increase (s : StoreSales) 
  (h1 : s.discount_rate = 0.1)
  (h2 : s.quantity_increase_rate = 0.15) :
  gross_income_change s = 3.5 := by
  sorry


end store_sales_increase_l309_30966


namespace area_triangle_XMY_l309_30963

/-- Triangle XMY with given dimensions --/
structure TriangleXMY where
  YM : ℝ
  MX : ℝ
  YZ : ℝ

/-- The area of triangle XMY is 3 square miles --/
theorem area_triangle_XMY (t : TriangleXMY) (h1 : t.YM = 2) (h2 : t.MX = 3) (h3 : t.YZ = 5) :
  (1 / 2) * t.YM * t.MX = 3 := by
  sorry


end area_triangle_XMY_l309_30963


namespace ceiling_product_equation_solution_l309_30940

theorem ceiling_product_equation_solution :
  ∃! (x : ℝ), ⌈x⌉ * x = 225 :=
by
  -- The proof goes here
  sorry

end ceiling_product_equation_solution_l309_30940


namespace reciprocal_proof_l309_30977

theorem reciprocal_proof (a b : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_diff : a ≠ b) 
  (h_eq : 1 / (1 + a) + 1 / (1 + b) = 2 / (1 + Real.sqrt (a * b))) : 
  a * b = 1 := by
sorry

end reciprocal_proof_l309_30977


namespace future_age_comparison_l309_30985

/-- Represents the age difference between Martha and Ellen in years -/
def AgeDifference : ℕ → Prop :=
  fun x => 32 = 2 * (10 + x)

/-- Proves that the number of years into the future when Martha's age is twice Ellen's age is 6 -/
theorem future_age_comparison : ∃ (x : ℕ), AgeDifference x ∧ x = 6 := by
  sorry

end future_age_comparison_l309_30985


namespace sin_45_degrees_l309_30980

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end sin_45_degrees_l309_30980


namespace min_distance_point_to_circle_l309_30959

/-- The minimum distance from a point to a circle -/
theorem min_distance_point_to_circle (x y : ℝ) :
  (x - 1)^2 + y^2 = 4 →  -- Circle equation
  ∃ (d : ℝ), d = Real.sqrt 18 - 2 ∧ 
  ∀ (px py : ℝ), (px + 2)^2 + (py + 3)^2 ≥ d^2 := by
  sorry

end min_distance_point_to_circle_l309_30959


namespace ratio_evaluation_l309_30996

theorem ratio_evaluation : (2^2005 * 3^2003) / 6^2004 = 2/3 := by sorry

end ratio_evaluation_l309_30996


namespace school_track_length_l309_30952

/-- Given that 200 steps correspond to 100 meters and 800 steps were walked along a track,
    the length of the track is 400 meters. -/
theorem school_track_length (steps_per_hundred_meters : ℕ) (track_steps : ℕ) : 
  steps_per_hundred_meters = 200 →
  track_steps = 800 →
  (100 : ℝ) / steps_per_hundred_meters * track_steps = 400 := by
  sorry

end school_track_length_l309_30952


namespace inequality_proof_l309_30924

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (h : a * b + b * c + c * d + d * a = 1) : 
  a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (d + a + b) + d^3 / (a + b + c) ≥ 1/3 := by
  sorry

end inequality_proof_l309_30924


namespace greatest_number_of_bouquets_l309_30906

/-- Represents the number of tulips of each color --/
structure TulipCount where
  white : Nat
  red : Nat
  blue : Nat
  yellow : Nat

/-- Represents the ratio of tulips in each bouquet --/
structure BouquetRatio where
  white : Nat
  red : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the number of bouquets that can be made with given tulips and ratio --/
def calculateBouquets (tulips : TulipCount) (ratio : BouquetRatio) : Nat :=
  min (tulips.white / ratio.white)
      (min (tulips.red / ratio.red)
           (min (tulips.blue / ratio.blue)
                (tulips.yellow / ratio.yellow)))

/-- Calculates the total number of flowers in a bouquet --/
def flowersPerBouquet (ratio : BouquetRatio) : Nat :=
  ratio.white + ratio.red + ratio.blue + ratio.yellow

theorem greatest_number_of_bouquets
  (tulips : TulipCount)
  (ratio : BouquetRatio)
  (h1 : tulips = ⟨21, 91, 37, 67⟩)
  (h2 : ratio = ⟨3, 7, 5, 9⟩)
  (h3 : flowersPerBouquet ratio ≥ 24)
  (h4 : flowersPerBouquet ratio ≤ 50)
  : calculateBouquets tulips ratio = 7 := by
  sorry

end greatest_number_of_bouquets_l309_30906


namespace symmetry_of_shifted_functions_l309_30953

-- Define a function f from reals to reals
variable (f : ℝ → ℝ)

-- Define the property of symmetry about x = -1
def symmetric_about_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + 1) = y ↔ f (-x - 1) = y

-- State the theorem
theorem symmetry_of_shifted_functions :
  symmetric_about_neg_one f := by sorry

end symmetry_of_shifted_functions_l309_30953


namespace sqrt_sum_equals_four_sqrt_three_l309_30944

theorem sqrt_sum_equals_four_sqrt_three :
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end sqrt_sum_equals_four_sqrt_three_l309_30944


namespace arthur_walk_distance_l309_30939

/-- Calculates the total distance walked given the number of blocks and the length of each block -/
def total_distance (blocks_east blocks_north block_length : ℚ) : ℚ :=
  (blocks_east + blocks_north) * block_length

theorem arthur_walk_distance :
  let blocks_east : ℚ := 8
  let blocks_north : ℚ := 15
  let block_length : ℚ := 1/4
  total_distance blocks_east blocks_north block_length = 5.75 := by sorry

end arthur_walk_distance_l309_30939


namespace one_less_than_negative_one_l309_30950

theorem one_less_than_negative_one : -1 - 1 = -2 := by
  sorry

end one_less_than_negative_one_l309_30950


namespace solve_for_t_l309_30922

theorem solve_for_t (s t : ℚ) (eq1 : 8 * s + 6 * t = 160) (eq2 : s = t + 3) : t = 68 / 7 := by
  sorry

end solve_for_t_l309_30922


namespace partnership_profit_l309_30957

/-- Given two partners A and B in a business partnership, this theorem proves
    that the total profit is 7 times B's profit under certain conditions. -/
theorem partnership_profit
  (investment_A investment_B : ℝ)
  (period_A period_B : ℝ)
  (profit_B : ℝ)
  (h1 : investment_A = 3 * investment_B)
  (h2 : period_A = 2 * period_B)
  (h3 : profit_B = investment_B * period_B)
  : investment_A * period_A + investment_B * period_B = 7 * profit_B :=
by sorry

end partnership_profit_l309_30957


namespace shark_percentage_is_25_l309_30984

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

end shark_percentage_is_25_l309_30984


namespace prob_more_ones_than_fives_five_dice_l309_30929

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define a function to calculate the probability
def prob_more_ones_than_fives (n : ℕ) (s : ℕ) : ℚ :=
  190 / (s^n : ℚ)

-- Theorem statement
theorem prob_more_ones_than_fives_five_dice : 
  prob_more_ones_than_fives num_dice num_sides = 190 / 7776 := by
  sorry


end prob_more_ones_than_fives_five_dice_l309_30929


namespace polynomial_expansion_l309_30942

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^3 + 7) * (3 * x + 4) = 15 * x^4 + 20 * x^3 + 21 * x + 28 := by
  sorry

end polynomial_expansion_l309_30942


namespace possible_to_end_with_80_19_specific_sequence_leads_to_80_19_l309_30987

/-- Represents the result of a shot --/
inductive ShotResult
  | Success
  | Miss

/-- Applies the effect of a shot to the current amount --/
def applyShot (amount : ℝ) (result : ShotResult) : ℝ :=
  match result with
  | ShotResult.Success => amount * 1.1
  | ShotResult.Miss => amount * 0.9

/-- Theorem stating that it's possible to end up with 80.19 rubles --/
theorem possible_to_end_with_80_19 : ∃ (shots : List ShotResult), 
  shots.foldl applyShot 100 = 80.19 := by
  sorry

/-- Proof that the specific sequence of shots leads to 80.19 rubles --/
theorem specific_sequence_leads_to_80_19 : 
  [ShotResult.Miss, ShotResult.Miss, ShotResult.Miss, ShotResult.Success].foldl applyShot 100 = 80.19 := by
  sorry

end possible_to_end_with_80_19_specific_sequence_leads_to_80_19_l309_30987


namespace smallest_divisible_by_11_with_remainders_l309_30972

theorem smallest_divisible_by_11_with_remainders :
  ∃! n : ℕ, n > 0 ∧ 
    11 ∣ n ∧
    n % 2 = 1 ∧
    n % 3 = 1 ∧
    n % 4 = 1 ∧
    n % 5 = 1 ∧
    ∀ m : ℕ, m > 0 ∧ 
      11 ∣ m ∧
      m % 2 = 1 ∧
      m % 3 = 1 ∧
      m % 4 = 1 ∧
      m % 5 = 1
    → n ≤ m :=
by
  sorry

end smallest_divisible_by_11_with_remainders_l309_30972


namespace election_votes_l309_30914

theorem election_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) :
  total_votes > 0 →
  winner_percentage = 62 / 100 →
  vote_difference = 300 →
  ⌊total_votes * winner_percentage⌋ - ⌊total_votes * (1 - winner_percentage)⌋ = vote_difference →
  ⌊total_votes * winner_percentage⌋ = 775 :=
by
  sorry

end election_votes_l309_30914


namespace brownie_pieces_count_l309_30943

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

end brownie_pieces_count_l309_30943


namespace gcd_143_144_l309_30932

theorem gcd_143_144 : Nat.gcd 143 144 = 1 := by
  sorry

end gcd_143_144_l309_30932


namespace sum_calculation_l309_30928

theorem sum_calculation : 3 * 501 + 2 * 501 + 4 * 501 + 500 = 5009 := by
  sorry

end sum_calculation_l309_30928


namespace carol_trivia_score_l309_30916

/-- Represents Carol's trivia game scores -/
structure TriviaGame where
  round1 : Int
  round2 : Int
  round3 : Int

/-- Calculates the total score of a trivia game -/
def totalScore (game : TriviaGame) : Int :=
  game.round1 + game.round2 + game.round3

/-- Theorem: Carol's total score at the end of the game is 7 points -/
theorem carol_trivia_score :
  ∃ (game : TriviaGame), game.round1 = 17 ∧ game.round2 = 6 ∧ game.round3 = -16 ∧ totalScore game = 7 := by
  sorry

end carol_trivia_score_l309_30916


namespace percent_value_in_quarters_l309_30910

def num_dimes : ℕ := 75
def num_quarters : ℕ := 30
def value_dime : ℕ := 10
def value_quarter : ℕ := 25

def total_value : ℕ := num_dimes * value_dime + num_quarters * value_quarter
def value_in_quarters : ℕ := num_quarters * value_quarter

theorem percent_value_in_quarters :
  (value_in_quarters : ℚ) / total_value * 100 = 50 := by sorry

end percent_value_in_quarters_l309_30910


namespace bridget_apples_l309_30970

/-- The number of apples Bridget originally bought -/
def original_apples : ℕ := 18

/-- The number of apples Bridget kept for herself in the end -/
def kept_apples : ℕ := 6

/-- The number of apples Bridget gave to Cassie -/
def given_apples : ℕ := 5

/-- The number of additional apples Bridget found in the bag -/
def found_apples : ℕ := 2

theorem bridget_apples : 
  original_apples / 2 - given_apples + found_apples = kept_apples := by
  sorry

#check bridget_apples

end bridget_apples_l309_30970


namespace cylinder_dimensions_l309_30978

/-- A cylinder whose bases' centers coincide with two opposite vertices of a unit cube,
    and whose lateral surface contains the remaining vertices of the cube -/
structure CylinderWithUnitCube where
  -- The height of the cylinder
  height : ℝ
  -- The base radius of the cylinder
  radius : ℝ
  -- The opposite vertices of the unit cube coincide with the centers of the cylinder bases
  opposite_vertices_on_bases : height = Real.sqrt 3
  -- The remaining vertices of the cube are on the lateral surface of the cylinder
  other_vertices_on_surface : radius = Real.sqrt 6 / 3

/-- The height and radius of a cylinder satisfying the given conditions -/
theorem cylinder_dimensions (c : CylinderWithUnitCube) :
  c.height = Real.sqrt 3 ∧ c.radius = Real.sqrt 6 / 3 := by
  sorry


end cylinder_dimensions_l309_30978


namespace inequality_proof_l309_30938

theorem inequality_proof (u v w : ℝ) 
  (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h : u + v + w + Real.sqrt (u * v * w) = 4) :
  Real.sqrt (v * w / u) + Real.sqrt (u * w / v) + Real.sqrt (u * v / w) ≥ u + v + w :=
by sorry

end inequality_proof_l309_30938


namespace candle_lighting_time_l309_30969

/-- The time (in minutes) when the candles are lit before 5 PM -/
def lighting_time : ℝ := 218

/-- The length of time (in minutes) it takes for the first candle to burn out completely -/
def burn_time_1 : ℝ := 240

/-- The length of time (in minutes) it takes for the second candle to burn out completely -/
def burn_time_2 : ℝ := 300

/-- The ratio of the length of the longer stub to the shorter stub at 5 PM -/
def stub_ratio : ℝ := 3

theorem candle_lighting_time :
  (burn_time_2 - lighting_time) / burn_time_2 = stub_ratio * ((burn_time_1 - lighting_time) / burn_time_1) :=
sorry

end candle_lighting_time_l309_30969


namespace circle_inscribed_angles_sum_l309_30974

theorem circle_inscribed_angles_sum (n : ℕ) (x y : ℝ) : 
  n = 18 →
  x = 3 * (360 / n) / 2 →
  y = 5 * (360 / n) / 2 →
  x + y = 80 := by
  sorry

end circle_inscribed_angles_sum_l309_30974


namespace problem_solution_l309_30983

theorem problem_solution (w y z x : ℕ) 
  (hw : w = 50)
  (hz : z = 2 * w + 3)
  (hy : y = z + 5)
  (hx : x = 2 * y + 4) :
  x = 220 := by
sorry

end problem_solution_l309_30983


namespace square_area_side_ratio_l309_30945

theorem square_area_side_ratio (a b : ℝ) (h : b ^ 2 = 16 * a ^ 2) : b = 4 * a := by
  sorry

end square_area_side_ratio_l309_30945


namespace fraction_comparison_l309_30904

theorem fraction_comparison : (1 / (Real.sqrt 5 - 2)) < (1 / (Real.sqrt 6 - Real.sqrt 5)) := by
  sorry

end fraction_comparison_l309_30904


namespace set_a_contains_one_l309_30967

theorem set_a_contains_one (a : ℝ) : 
  1 ∈ ({a + 2, (a + 1)^2, a^2 + 3*a + 3} : Set ℝ) → a = 0 := by
  sorry

end set_a_contains_one_l309_30967


namespace unique_x_with_rational_sums_l309_30988

theorem unique_x_with_rational_sums (x : ℝ) :
  (∃ a : ℚ, x + Real.sqrt 3 = a) →
  (∃ b : ℚ, x^2 + Real.sqrt 3 = b) →
  x = 1/2 - Real.sqrt 3 := by
sorry

end unique_x_with_rational_sums_l309_30988


namespace min_value_of_sum_l309_30909

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

-- Theorem statement
theorem min_value_of_sum (a : ℝ) :
  (∃ x, x = 2 ∧ f_derivative a x = 0) →
  (∀ m n, m ∈ Set.Icc (-1 : ℝ) 1 → n ∈ Set.Icc (-1 : ℝ) 1 →
    f a m + f_derivative a n ≥ -13) ∧
  (∃ m n, m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a m + f_derivative a n = -13) :=
by sorry


end min_value_of_sum_l309_30909


namespace tank_emptying_equivalence_l309_30933

/-- Represents the work capacity of pumps emptying a tank -/
def tank_emptying_work (pumps : ℕ) (hours_per_day : ℝ) (days : ℝ) : ℝ :=
  pumps * hours_per_day * days

theorem tank_emptying_equivalence (d : ℝ) (h : d > 0) :
  let original_work := tank_emptying_work 3 8 2
  let new_work := tank_emptying_work 6 (8 / d) d
  original_work = new_work :=
by sorry

end tank_emptying_equivalence_l309_30933


namespace geometric_sequence_ratio_l309_30917

/-- A geometric sequence with a_3 = 8 * a_6 has S_4 / S_2 = 5/4 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- geometric sequence condition
  (∀ n, S n = (a 1 * (1 - (a 2 / a 1)^n)) / (1 - (a 2 / a 1))) →  -- sum formula
  a 3 = 8 * a 6 →  -- given condition
  S 4 / S 2 = 5 / 4 := by
sorry

end geometric_sequence_ratio_l309_30917


namespace mark_baking_time_l309_30960

/-- The time Mark spends baking bread -/
def baking_time (total_time rising_time kneading_time : ℕ) : ℕ :=
  total_time - (2 * rising_time + kneading_time)

/-- Theorem stating that Mark spends 30 minutes baking bread -/
theorem mark_baking_time :
  baking_time 280 120 10 = 30 := by
  sorry

end mark_baking_time_l309_30960


namespace root_in_interval_implies_m_range_l309_30921

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the theorem
theorem root_in_interval_implies_m_range :
  ∀ m : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ f x + m = 0) → m ∈ Set.Icc (-2) 2 := by
  sorry

end root_in_interval_implies_m_range_l309_30921


namespace union_of_A_and_B_l309_30907

def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {1, 2}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 6} := by
  sorry

end union_of_A_and_B_l309_30907


namespace intersection_with_complement_l309_30991

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem intersection_with_complement : M ∩ (U \ N) = {0, 3} := by
  sorry

end intersection_with_complement_l309_30991


namespace first_term_of_special_arithmetic_sequence_l309_30923

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem first_term_of_special_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n, a (n + 1) > a n) →
  a 1 + a 2 + a 3 = 12 →
  a 1 * a 2 * a 3 = 48 →
  a 1 = 2 := by
sorry

end first_term_of_special_arithmetic_sequence_l309_30923


namespace parabola_equation_c_value_l309_30920

/-- A parabola with vertex at (5, 1) passing through (2, 3) has equation x = ay^2 + by + c where c = 17/4 -/
theorem parabola_equation_c_value (a b c : ℝ) : 
  (∀ y : ℝ, 5 = a * 1^2 + b * 1 + c) →  -- vertex at (5, 1)
  (2 = a * 3^2 + b * 3 + c) →           -- passes through (2, 3)
  (∀ x y : ℝ, x = a * y^2 + b * y + c) →  -- equation of the form x = ay^2 + by + c
  c = 17/4 := by
sorry

end parabola_equation_c_value_l309_30920


namespace baker_flour_remaining_l309_30913

/-- A baker's recipe requires 3 eggs for every 2 cups of flour. -/
def recipe_ratio : ℚ := 3 / 2

/-- The number of eggs needed to use up all remaining flour. -/
def eggs_needed : ℕ := 9

/-- Calculates the number of cups of flour remaining given the recipe ratio and eggs needed. -/
def flour_remaining (ratio : ℚ) (eggs : ℕ) : ℚ := (eggs : ℚ) / ratio

theorem baker_flour_remaining :
  flour_remaining recipe_ratio eggs_needed = 6 := by
  sorry

end baker_flour_remaining_l309_30913


namespace equal_expressions_condition_l309_30912

theorem equal_expressions_condition (a b c : ℝ) :
  a + b + c = (a + b) * (a + c) ↔ a = 1 ∧ b = 1 ∧ c = 1 := by
  sorry

end equal_expressions_condition_l309_30912


namespace point_distance_product_l309_30931

theorem point_distance_product : 
  ∃ (y₁ y₂ : ℝ), 
    (((5 - (-3))^2 + (2 - y₁)^2 : ℝ) = 14^2) ∧
    (((5 - (-3))^2 + (2 - y₂)^2 : ℝ) = 14^2) ∧
    (y₁ * y₂ = -128) :=
by sorry

end point_distance_product_l309_30931


namespace two_numbers_with_110_divisors_and_nine_zeros_sum_l309_30954

/-- A number ends with 9 zeros if it's divisible by 10^9 -/
def ends_with_nine_zeros (n : ℕ) : Prop := n % (10^9) = 0

/-- Count the number of divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The main theorem -/
theorem two_numbers_with_110_divisors_and_nine_zeros_sum :
  ∃ (a b : ℕ), a ≠ b ∧
                ends_with_nine_zeros a ∧
                ends_with_nine_zeros b ∧
                count_divisors a = 110 ∧
                count_divisors b = 110 ∧
                a + b = 7000000000 := by sorry

end two_numbers_with_110_divisors_and_nine_zeros_sum_l309_30954


namespace acute_angles_tangent_sum_l309_30973

theorem acute_angles_tangent_sum (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  (1 + Real.tan α) * (1 + Real.tan β) = 2 →
  α + β = π/4 := by
sorry

end acute_angles_tangent_sum_l309_30973


namespace quadratic_factorization_sum_l309_30982

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x, x^2 + 9*x + 18 = (x + a) * (x + b)) →
  (∀ x, x^2 + 19*x + 90 = (x + b) * (x + c)) →
  a + b + c = 22 := by
sorry

end quadratic_factorization_sum_l309_30982


namespace area_triangle_ABC_l309_30915

/-- Linear function f(x) = ax + b -/
def linear_function (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

/-- The y-intercept of a linear function f(x) = ax + b -/
def y_intercept (f : ℝ → ℝ) : ℝ := f 0

theorem area_triangle_ABC : 
  ∀ (m n : ℝ),
  let f := linear_function (3/2) m
  let g := linear_function (-1/2) n
  (f (-4) = 0) →
  (g (-4) = 0) →
  let B := (0, y_intercept f)
  let C := (0, y_intercept g)
  let A := (-4, 0)
  (1/2 * |A.1| * |B.2 - C.2| = 16) :=
by sorry

end area_triangle_ABC_l309_30915


namespace student_distribution_l309_30986

/-- The number of ways to distribute n distinct students among k distinct universities,
    with each university receiving at least one student. -/
def distribute_students (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n r : ℕ) : ℕ := sorry

theorem student_distribution :
  distribute_students 4 3 = 36 :=
by
  have h1 : distribute_students 4 3 = choose 3 1 * choose 4 2 * 2
  sorry
  sorry

end student_distribution_l309_30986


namespace cubic_factorization_l309_30926

theorem cubic_factorization (a : ℝ) : a^3 - 25*a = a*(a+5)*(a-5) := by sorry

end cubic_factorization_l309_30926


namespace complex_modulus_l309_30949

theorem complex_modulus (z : ℂ) : z + Complex.I = (2 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_modulus_l309_30949


namespace smallest_gcd_with_lcm_condition_l309_30946

theorem smallest_gcd_with_lcm_condition (x y : ℕ) 
  (h : Nat.lcm x y = (x - y)^2) : 
  Nat.gcd x y ≥ 2 :=
sorry

end smallest_gcd_with_lcm_condition_l309_30946


namespace problem_1_problem_2_problem_3_problem_4_l309_30965

-- Problem 1
theorem problem_1 (a : ℝ) : -2 * a^3 * 3 * a^2 = -6 * a^5 := by sorry

-- Problem 2
theorem problem_2 (m : ℝ) (hm : m ≠ 0) : m^4 * (m^2)^3 / m^8 = m^2 := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) : (-2*x - 1) * (2*x - 1) = 1 - 4*x^2 := by sorry

-- Problem 4
theorem problem_4 (x : ℝ) : (-3*x + 2)^2 = 9*x^2 - 12*x + 4 := by sorry

end problem_1_problem_2_problem_3_problem_4_l309_30965


namespace max_ratio_two_digit_integers_l309_30937

/-- Two-digit positive integer -/
def TwoDigitPositiveInteger (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem max_ratio_two_digit_integers (x y : ℕ) :
  TwoDigitPositiveInteger x →
  TwoDigitPositiveInteger y →
  (x + y) / 2 = 55 →
  ∀ a b : ℕ, TwoDigitPositiveInteger a → TwoDigitPositiveInteger b → (a + b) / 2 = 55 →
    (a : ℚ) / b ≤ 79 / 31 :=
by sorry

end max_ratio_two_digit_integers_l309_30937


namespace count_integers_with_repeated_digits_is_156_l309_30955

/-- A function that counts the number of positive three-digit integers less than 700 
    with at least two digits that are the same. -/
def count_integers_with_repeated_digits : ℕ :=
  sorry

/-- The theorem stating that the count of integers with the given properties is 156. -/
theorem count_integers_with_repeated_digits_is_156 : 
  count_integers_with_repeated_digits = 156 := by
  sorry

end count_integers_with_repeated_digits_is_156_l309_30955


namespace second_worker_loading_time_l309_30941

/-- The time it takes for the second worker to load one truck alone, given that:
    1. The first worker can load one truck in 5 hours
    2. Both workers together can load one truck in approximately 2.2222222222222223 hours
-/
theorem second_worker_loading_time :
  let first_worker_time : ℝ := 5
  let combined_time : ℝ := 2.2222222222222223
  let second_worker_time : ℝ := (first_worker_time * combined_time) / (first_worker_time - combined_time)
  ‖second_worker_time - 1.4285714285714286‖ < 0.0001 := by
sorry


end second_worker_loading_time_l309_30941


namespace union_of_M_and_N_l309_30999

def M : Set ℝ := {-1, 1, 2}
def N : Set ℝ := {x : ℝ | x^2 - 5*x + 4 = 0}

theorem union_of_M_and_N : M ∪ N = {-1, 1, 2, 4} := by
  sorry

end union_of_M_and_N_l309_30999


namespace student_number_problem_l309_30900

theorem student_number_problem (x : ℝ) : 2 * x - 148 = 110 → x = 129 := by
  sorry

end student_number_problem_l309_30900


namespace solve_lindas_savings_l309_30951

def lindas_savings_problem (savings : ℚ) : Prop :=
  let furniture_fraction : ℚ := 3 / 5
  let tv_fraction : ℚ := 1 - furniture_fraction
  let tv_cost : ℚ := 400
  tv_fraction * savings = tv_cost ∧ savings = 1000

theorem solve_lindas_savings : ∃ (savings : ℚ), lindas_savings_problem savings :=
sorry

end solve_lindas_savings_l309_30951


namespace f_sum_symmetric_l309_30962

/-- Given a function f(x) = ax^7 - bx^5 + cx^3 + 2, prove that f(5) + f(-5) = 4 -/
theorem f_sum_symmetric (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^7 - b * x^5 + c * x^3 + 2
  f 5 + f (-5) = 4 := by sorry

end f_sum_symmetric_l309_30962


namespace pens_purchased_l309_30998

theorem pens_purchased (total_cost : ℝ) (num_pencils : ℕ) (pencil_price : ℝ) (pen_price : ℝ)
  (h1 : total_cost = 690)
  (h2 : num_pencils = 75)
  (h3 : pencil_price = 2)
  (h4 : pen_price = 18) :
  (total_cost - num_pencils * pencil_price) / pen_price = 30 := by
  sorry

end pens_purchased_l309_30998


namespace number_of_dolls_l309_30961

theorem number_of_dolls (total_toys : ℕ) (action_figure_percentage : ℚ) (number_of_dolls : ℕ) : 
  total_toys = 120 →
  action_figure_percentage = 35 / 100 →
  number_of_dolls = total_toys - (action_figure_percentage * total_toys).floor →
  number_of_dolls = 78 := by
  sorry

end number_of_dolls_l309_30961


namespace two_in_M_l309_30989

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define the complement of M in U
def complementM : Set Nat := {1, 3}

-- Theorem to prove
theorem two_in_M : 2 ∈ (U \ complementM) := by
  sorry

end two_in_M_l309_30989


namespace ravi_jump_multiple_l309_30905

def jump_heights : List ℝ := [23, 27, 28]
def ravi_jump : ℝ := 39

theorem ravi_jump_multiple :
  ravi_jump / (jump_heights.sum / jump_heights.length) = 1.5 := by
  sorry

end ravi_jump_multiple_l309_30905


namespace paper_count_l309_30911

theorem paper_count (initial_math initial_science used_math used_science received_math given_science : ℕ) :
  initial_math = 220 →
  initial_science = 150 →
  used_math = 95 →
  used_science = 68 →
  received_math = 30 →
  given_science = 15 →
  (initial_math - used_math + received_math) + (initial_science - used_science - given_science) = 222 := by
  sorry

end paper_count_l309_30911


namespace prank_combinations_l309_30930

theorem prank_combinations (monday tuesday wednesday thursday friday : ℕ) :
  monday = 3 →
  tuesday = 1 →
  wednesday = 6 →
  thursday = 4 →
  friday = 2 →
  monday * tuesday * wednesday * thursday * friday = 144 := by
  sorry

end prank_combinations_l309_30930


namespace women_work_hours_l309_30981

/-- Given work completed by men and women under specific conditions, prove that women worked 6 hours per day. -/
theorem women_work_hours (men : ℕ) (women : ℕ) (men_days : ℕ) (women_days : ℕ) (men_hours : ℕ) 
  (h_men : men = 15)
  (h_women : women = 21)
  (h_men_days : men_days = 21)
  (h_women_days : women_days = 30)
  (h_men_hours : men_hours = 8)
  (h_work_rate : (3 : ℚ) / women = (2 : ℚ) / men) :
  ∃ women_hours : ℚ, women_hours = 6 ∧ 
    (men * men_days * men_hours : ℚ) = (women * women_days * women_hours) :=
sorry

end women_work_hours_l309_30981


namespace circle_triangle_count_l309_30947

/-- The number of points on the circle's circumference -/
def n : ℕ := 10

/-- The total number of triangles that can be formed from n points -/
def total_triangles (n : ℕ) : ℕ := n.choose 3

/-- The number of triangles with consecutive vertices -/
def consecutive_triangles (n : ℕ) : ℕ := n

/-- The number of valid triangles (no consecutive vertices) -/
def valid_triangles (n : ℕ) : ℕ := total_triangles n - consecutive_triangles n

theorem circle_triangle_count :
  valid_triangles n = 110 :=
sorry

end circle_triangle_count_l309_30947


namespace vector_equality_sufficient_not_necessary_l309_30902

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_equality_sufficient_not_necessary :
  (∀ a b : V, a = b → (‖a‖ = ‖b‖ ∧ parallel a b)) ∧
  (∃ a b : V, ‖a‖ = ‖b‖ ∧ parallel a b ∧ a ≠ b) :=
sorry

end vector_equality_sufficient_not_necessary_l309_30902


namespace bookshelf_selections_l309_30903

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

end bookshelf_selections_l309_30903


namespace excel_manufacturing_company_women_percentage_l309_30948

theorem excel_manufacturing_company_women_percentage
  (total_employees : ℕ)
  (male_percentage : Real)
  (union_percentage : Real)
  (non_union_women_percentage : Real)
  (h1 : male_percentage = 0.46)
  (h2 : union_percentage = 0.60)
  (h3 : non_union_women_percentage = 0.90) :
  non_union_women_percentage = 0.90 := by
sorry

end excel_manufacturing_company_women_percentage_l309_30948


namespace coefficient_x3_is_negative_five_l309_30976

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (x+1)(x-2)^3
def coefficient_x3 : ℤ :=
  (1 * (binomial 3 0)) + (-2 * (binomial 3 1))

-- Theorem statement
theorem coefficient_x3_is_negative_five :
  coefficient_x3 = -5 := by sorry

end coefficient_x3_is_negative_five_l309_30976


namespace algebra_test_female_students_l309_30925

theorem algebra_test_female_students 
  (total_average : ℝ) 
  (male_count : ℕ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (h1 : total_average = 90) 
  (h2 : male_count = 8) 
  (h3 : male_average = 87) 
  (h4 : female_average = 92) : 
  ∃ (female_count : ℕ), 
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧ 
    female_count = 12 := by
  sorry

end algebra_test_female_students_l309_30925


namespace sqrt_54_div_sqrt_9_eq_sqrt_6_l309_30993

theorem sqrt_54_div_sqrt_9_eq_sqrt_6 : Real.sqrt 54 / Real.sqrt 9 = Real.sqrt 6 := by
  sorry

end sqrt_54_div_sqrt_9_eq_sqrt_6_l309_30993


namespace cosine_shift_l309_30901

theorem cosine_shift (x : ℝ) :
  let f (x : ℝ) := 3 * Real.cos (1/2 * x - π/3)
  let period := 4 * π
  let shift := period / 8
  let g (x : ℝ) := f (x + shift)
  g x = 3 * Real.cos (1/2 * x - π/12) := by
  sorry

end cosine_shift_l309_30901


namespace hillary_activities_lcm_l309_30936

theorem hillary_activities_lcm : Nat.lcm (Nat.lcm 6 4) 16 = 48 := by sorry

end hillary_activities_lcm_l309_30936


namespace root_difference_of_equation_l309_30927

theorem root_difference_of_equation : ∃ a b : ℝ,
  (∀ x : ℝ, (6 * x - 18) / (x^2 + 3 * x - 28) = x + 3 ↔ x = a ∨ x = b) ∧
  a > b ∧
  a - b = 2 := by
sorry

end root_difference_of_equation_l309_30927


namespace bottles_per_box_l309_30975

theorem bottles_per_box 
  (num_boxes : ℕ) 
  (bottle_capacity : ℚ) 
  (fill_ratio : ℚ) 
  (total_water : ℚ) :
  num_boxes = 10 →
  bottle_capacity = 12 →
  fill_ratio = 3/4 →
  total_water = 4500 →
  (total_water / (bottle_capacity * fill_ratio)) / num_boxes = 50 :=
by sorry

end bottles_per_box_l309_30975


namespace symmetric_function_trigonometric_identity_l309_30935

theorem symmetric_function_trigonometric_identity (θ : ℝ) :
  (∀ x : ℝ, x^2 + (Real.sin θ - Real.cos θ) * x + Real.sin θ = 
            (-x)^2 + (Real.sin θ - Real.cos θ) * (-x) + Real.sin θ) →
  2 * Real.sin θ * Real.cos θ + Real.cos (2 * θ) = 1 := by
  sorry

end symmetric_function_trigonometric_identity_l309_30935


namespace roger_tray_trips_l309_30994

/-- Calculates the number of trips needed to carry a given number of trays -/
def trips_needed (trays_per_trip : ℕ) (total_trays : ℕ) : ℕ :=
  (total_trays + trays_per_trip - 1) / trays_per_trip

/-- Proves that 3 trips are needed to carry 12 trays when 4 trays can be carried per trip -/
theorem roger_tray_trips : trips_needed 4 12 = 3 := by
  sorry

end roger_tray_trips_l309_30994


namespace replacement_philosophy_in_lines_one_and_three_l309_30997

/-- Represents a line of poetry -/
inductive PoeticLine
| EndlessFalling
| SpringRiver
| NewLeaves
| Waterfall

/-- Checks if a poetic line contains the philosophy of new things replacing old ones -/
def containsReplacementPhilosophy (line : PoeticLine) : Prop :=
  match line with
  | PoeticLine.EndlessFalling => True
  | PoeticLine.SpringRiver => False
  | PoeticLine.NewLeaves => True
  | PoeticLine.Waterfall => False

/-- The theorem stating that only lines ① and ③ contain the replacement philosophy -/
theorem replacement_philosophy_in_lines_one_and_three :
  (∀ line : PoeticLine, containsReplacementPhilosophy line ↔
    (line = PoeticLine.EndlessFalling ∨ line = PoeticLine.NewLeaves)) :=
by sorry

end replacement_philosophy_in_lines_one_and_three_l309_30997


namespace x_value_proof_l309_30934

theorem x_value_proof (x y z : ℝ) (h1 : x = y) (h2 : y = 2 * z) (h3 : x * y * z = 256) : x = 8 := by
  sorry

end x_value_proof_l309_30934


namespace data_transmission_time_l309_30990

/-- Represents the number of blocks of data to be sent -/
def num_blocks : ℕ := 100

/-- Represents the number of chunks in each block -/
def chunks_per_block : ℕ := 450

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 200

/-- Represents the number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem stating that the time to send the data is 0.0625 hours -/
theorem data_transmission_time :
  (num_blocks * chunks_per_block : ℚ) / transmission_rate / seconds_per_hour = 0.0625 := by
  sorry

end data_transmission_time_l309_30990


namespace quadratic_function_range_l309_30968

/-- The function f(x) = x^2 + mx - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

theorem quadratic_function_range (m : ℝ) :
  (∀ x ∈ Set.Icc m (m + 1), f m x < 0) ↔ m ∈ Set.Ioo (- Real.sqrt 2 / 2) 0 :=
sorry

end quadratic_function_range_l309_30968


namespace solve_percentage_problem_l309_30971

def percentage_problem (P : ℝ) (x : ℝ) : Prop :=
  (P / 100) * x = (5 / 100) * 500 - 20 ∧ x = 10

theorem solve_percentage_problem :
  ∃ P : ℝ, percentage_problem P 10 ∧ P = 50 := by
  sorry

end solve_percentage_problem_l309_30971


namespace bowling_ball_weight_proof_l309_30995

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 20

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 30

theorem bowling_ball_weight_proof :
  (9 * bowling_ball_weight = 6 * canoe_weight) ∧
  (4 * canoe_weight = 120) →
  bowling_ball_weight = 20 := by
sorry

end bowling_ball_weight_proof_l309_30995


namespace smallest_positive_multiple_of_45_l309_30958

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l309_30958


namespace pure_imaginary_fraction_l309_30908

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + 2 * Complex.I) / (1 + 2 * Complex.I) = b * Complex.I) → a = -4 := by
  sorry

end pure_imaginary_fraction_l309_30908


namespace sum_first_n_integers_remainder_l309_30979

theorem sum_first_n_integers_remainder (n : ℕ+) :
  let sum := n.val * (n.val + 1) / 2
  sum % n.val = if n.val % 2 = 1 then 0 else n.val / 2 := by
  sorry

end sum_first_n_integers_remainder_l309_30979
