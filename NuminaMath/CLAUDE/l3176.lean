import Mathlib

namespace sum_base6_to_55_l3176_317608

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Sums numbers from 1 to n in base 6 -/
def sumBase6 (n : ℕ) : ℕ := sorry

theorem sum_base6_to_55 : base6ToBase10 (sumBase6 55) = 630 := by sorry

end sum_base6_to_55_l3176_317608


namespace tan_alpha_value_l3176_317660

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 1/5) : Real.tan α = -2/3 := by
  sorry

end tan_alpha_value_l3176_317660


namespace blue_notebook_cost_l3176_317675

/-- The cost of each blue notebook given Mike's purchase details -/
theorem blue_notebook_cost 
  (total_spent : ℕ)
  (total_notebooks : ℕ)
  (red_notebooks : ℕ)
  (red_price : ℕ)
  (green_notebooks : ℕ)
  (green_price : ℕ)
  (h1 : total_spent = 37)
  (h2 : total_notebooks = 12)
  (h3 : red_notebooks = 3)
  (h4 : red_price = 4)
  (h5 : green_notebooks = 2)
  (h6 : green_price = 2)
  (h7 : total_notebooks = red_notebooks + green_notebooks + (total_notebooks - red_notebooks - green_notebooks))
  : (total_spent - red_notebooks * red_price - green_notebooks * green_price) / (total_notebooks - red_notebooks - green_notebooks) = 3 := by
  sorry

end blue_notebook_cost_l3176_317675


namespace rhino_fold_swap_impossible_l3176_317630

/-- Represents the number of folds on a rhinoceros -/
structure FoldCount where
  vertical : ℕ
  horizontal : ℕ

/-- Represents the state of folds on both sides of a rhinoceros -/
structure RhinoState where
  left : FoldCount
  right : FoldCount

def total_folds (state : RhinoState) : ℕ :=
  state.left.vertical + state.left.horizontal + state.right.vertical + state.right.horizontal

/-- Represents a single scratch action -/
inductive ScratchAction
  | left_vertical
  | left_horizontal
  | right_vertical
  | right_horizontal

/-- Applies a scratch action to a RhinoState -/
def apply_scratch (state : RhinoState) (action : ScratchAction) : RhinoState :=
  match action with
  | ScratchAction.left_vertical => 
      { left := { vertical := state.left.vertical - 2, horizontal := state.left.horizontal },
        right := { vertical := state.right.vertical + 1, horizontal := state.right.horizontal + 1 } }
  | ScratchAction.left_horizontal => 
      { left := { vertical := state.left.vertical, horizontal := state.left.horizontal - 2 },
        right := { vertical := state.right.vertical + 1, horizontal := state.right.horizontal + 1 } }
  | ScratchAction.right_vertical => 
      { left := { vertical := state.left.vertical + 1, horizontal := state.left.horizontal + 1 },
        right := { vertical := state.right.vertical - 2, horizontal := state.right.horizontal } }
  | ScratchAction.right_horizontal => 
      { left := { vertical := state.left.vertical + 1, horizontal := state.left.horizontal + 1 },
        right := { vertical := state.right.vertical, horizontal := state.right.horizontal - 2 } }

theorem rhino_fold_swap_impossible (initial : RhinoState) 
    (h_total : total_folds initial = 17) :
    ¬∃ (actions : List ScratchAction), 
      let final := actions.foldl apply_scratch initial
      total_folds final = 17 ∧ 
      final.left.vertical = initial.left.horizontal ∧
      final.left.horizontal = initial.left.vertical ∧
      final.right.vertical = initial.right.horizontal ∧
      final.right.horizontal = initial.right.vertical :=
  sorry

end rhino_fold_swap_impossible_l3176_317630


namespace greatest_number_of_bouquets_l3176_317683

theorem greatest_number_of_bouquets (white_tulips red_tulips : ℕ) 
  (h_white : white_tulips = 21) (h_red : red_tulips = 91) : 
  (∃ (bouquets_count : ℕ) (white_per_bouquet red_per_bouquet : ℕ), 
    bouquets_count * white_per_bouquet = white_tulips ∧ 
    bouquets_count * red_per_bouquet = red_tulips ∧ 
    ∀ (other_count : ℕ) (other_white other_red : ℕ), 
      other_count * other_white = white_tulips → 
      other_count * other_red = red_tulips → 
      other_count ≤ bouquets_count) ∧ 
  (∃ (max_bouquets : ℕ), max_bouquets = 3 ∧ 
    ∀ (bouquets_count : ℕ) (white_per_bouquet red_per_bouquet : ℕ), 
      bouquets_count * white_per_bouquet = white_tulips → 
      bouquets_count * red_per_bouquet = red_tulips → 
      bouquets_count ≤ max_bouquets) := by
sorry

end greatest_number_of_bouquets_l3176_317683


namespace product_inequality_l3176_317645

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c + a * b * c = 4) : 
  (1 + a / b + c * a) * (1 + b / c + a * b) * (1 + c / a + b * c) ≥ 27 := by
  sorry

end product_inequality_l3176_317645


namespace sum_of_solutions_is_five_l3176_317678

theorem sum_of_solutions_is_five : 
  ∃! (s : ℝ), ∀ (x : ℝ), (x + 25 / x = 10) → (s = x) :=
by
  sorry

end sum_of_solutions_is_five_l3176_317678


namespace emergency_kit_problem_l3176_317620

/-- Given the conditions of Veronica's emergency-preparedness kits problem, 
    prove that the number of food cans must be a multiple of 4 and at least 4. -/
theorem emergency_kit_problem (num_water_bottles : Nat) (num_food_cans : Nat) :
  num_water_bottles = 20 →
  num_water_bottles % 4 = 0 →
  num_food_cans % 4 = 0 →
  (num_water_bottles + num_food_cans) % 4 = 0 →
  num_food_cans ≥ 4 :=
by sorry

end emergency_kit_problem_l3176_317620


namespace library_visitors_average_l3176_317655

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (days_in_month : ℕ) (h1 : sunday_visitors = 140) (h2 : other_day_visitors = 80) 
  (h3 : days_in_month = 30) :
  let sundays : ℕ := (days_in_month + 6) / 7
  let other_days : ℕ := days_in_month - sundays
  let total_visitors : ℕ := sundays * sunday_visitors + other_days * other_day_visitors
  (total_visitors : ℚ) / days_in_month = 88 := by
  sorry

end library_visitors_average_l3176_317655


namespace diagonal_length_of_prism_l3176_317691

/-- 
Given a rectangular prism with dimensions x, y, and z,
if the projections of its diagonal on each plane (xy, xz, yz) have length √2,
then the length of the diagonal is √3.
-/
theorem diagonal_length_of_prism (x y z : ℝ) 
  (h1 : x^2 + y^2 = 2)
  (h2 : x^2 + z^2 = 2)
  (h3 : y^2 + z^2 = 2) :
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 3 := by
  sorry


end diagonal_length_of_prism_l3176_317691


namespace johns_class_boys_count_l3176_317632

theorem johns_class_boys_count :
  ∀ (g b : ℕ),
  g + b = 28 →
  g = (3 * b) / 4 →
  b = 16 :=
by
  sorry

end johns_class_boys_count_l3176_317632


namespace sum_of_digit_products_2019_l3176_317693

/-- Product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Sum of products of digits for numbers from 1 to n -/
def sumOfDigitProducts (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of products of digits for integers from 1 to 2019 is 184320 -/
theorem sum_of_digit_products_2019 : sumOfDigitProducts 2019 = 184320 := by sorry

end sum_of_digit_products_2019_l3176_317693


namespace coffee_consumption_l3176_317634

theorem coffee_consumption (x : ℝ) : 
  x > 0 → -- Tom's coffee size is positive
  (2/3 * x + (5/48 * x + 3) = 5/4 * (2/3 * x) - (5/48 * x + 3)) → -- They drink the same amount
  x + 1.25 * x = 36 -- Total coffee consumed is 36 ounces
  := by sorry

end coffee_consumption_l3176_317634


namespace valid_arrangements_count_l3176_317697

/-- Number of ways to arrange n distinct objects in r positions --/
def arrangement (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of boxes --/
def num_boxes : ℕ := 7

/-- The number of balls --/
def num_balls : ℕ := 4

/-- The number of ways to arrange the balls satisfying all conditions --/
def valid_arrangements : ℕ :=
  arrangement num_balls num_balls * arrangement (num_balls + 1) 2 -
  arrangement 2 2 * arrangement 3 3 * arrangement 4 2

theorem valid_arrangements_count :
  valid_arrangements = 336 := by sorry

end valid_arrangements_count_l3176_317697


namespace a_completes_in_15_days_l3176_317603

/-- The number of days it takes for B to complete the work alone -/
def b_days : ℝ := 20

/-- The number of days A and B work together -/
def days_together : ℝ := 6

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.3

/-- The number of days it takes for A to complete the work alone -/
def a_days : ℝ := 15

/-- Proves that given the conditions, A takes 15 days to complete the work alone -/
theorem a_completes_in_15_days :
  days_together * (1 / a_days + 1 / b_days) = 1 - work_left := by
  sorry

end a_completes_in_15_days_l3176_317603


namespace max_winner_number_l3176_317617

/-- Represents a player in the tournament -/
structure Player where
  number : Nat
  deriving Repr

/-- Represents the tournament -/
def Tournament :=
  {players : Finset Player // players.card = 1024 ∧ ∀ p ∈ players, p.number ≤ 1024}

/-- Predicate for whether a player wins against another player -/
def wins (p1 p2 : Player) : Prop :=
  p1.number < p2.number ∧ p2.number - p1.number > 2

/-- The winner of the tournament -/
def tournamentWinner (t : Tournament) : Player :=
  sorry

/-- The theorem stating the maximum qualification number of the winner -/
theorem max_winner_number (t : Tournament) :
  (tournamentWinner t).number ≤ 20 :=
sorry

end max_winner_number_l3176_317617


namespace inequality_solution_set_l3176_317695

theorem inequality_solution_set (a : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 → 
  (∀ x, x^2 - 2*a*x - 3*a^2 < 0 ↔ x₁ < x ∧ x < x₂) → 
  |x₁ - x₂| = 8 → 
  a = 2 := by
sorry

end inequality_solution_set_l3176_317695


namespace early_arrival_l3176_317602

theorem early_arrival (usual_time : ℝ) (rate_increase : ℝ) (early_time : ℝ) : 
  usual_time = 35 →
  rate_increase = 7/6 →
  early_time = usual_time - (usual_time / rate_increase) →
  early_time = 5 := by
sorry

end early_arrival_l3176_317602


namespace inequality_proof_l3176_317676

theorem inequality_proof (a b : ℝ) (ha : a > 1/2) (hb : b > 1/2) :
  a + 2*b - 5*a*b < 1/4 := by
  sorry

end inequality_proof_l3176_317676


namespace correct_subtraction_l3176_317642

theorem correct_subtraction (x : ℤ) (h : x - 32 = 25) : x - 23 = 34 := by
  sorry

end correct_subtraction_l3176_317642


namespace inequality_proof_l3176_317644

theorem inequality_proof (n : ℕ+) : (2*n+1)^(n : ℕ) ≥ (2*n)^(n : ℕ) + (2*n-1)^(n : ℕ) := by
  sorry

end inequality_proof_l3176_317644


namespace sum_of_four_numbers_l3176_317685

theorem sum_of_four_numbers : 1256 + 2561 + 5612 + 6125 = 15554 := by
  sorry

end sum_of_four_numbers_l3176_317685


namespace range_of_a_l3176_317610

/-- Two circles intersect at exactly two points if and only if 
the distance between their centers is greater than the absolute difference 
of their radii and less than the sum of their radii. -/
axiom circle_intersection_condition (r₁ r₂ d : ℝ) : 
  (∃! (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ 
    ((p₁.1 - d)^2 + p₁.2^2 = r₁^2) ∧ (p₁.1^2 + p₁.2^2 = r₂^2) ∧
    ((p₂.1 - d)^2 + p₂.2^2 = r₁^2) ∧ (p₂.1^2 + p₂.2^2 = r₂^2)) ↔ 
  (abs (r₁ - r₂) < d ∧ d < r₁ + r₂)

/-- The main theorem stating the range of a given the intersection condition. -/
theorem range_of_a : 
  ∀ a : ℝ, (∃! (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ 
    ((p₁.1 - a)^2 + (p₁.2 - a)^2 = 4) ∧ (p₁.1^2 + p₁.2^2 = 4) ∧
    ((p₂.1 - a)^2 + (p₂.2 - a)^2 = 4) ∧ (p₂.1^2 + p₂.2^2 = 4)) → 
  (-2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2 ∧ a ≠ 0) :=
sorry

end range_of_a_l3176_317610


namespace intersection_range_l3176_317667

/-- The function f(x) = 3x - x^3 --/
def f (x : ℝ) : ℝ := 3*x - x^3

/-- The line y = m intersects the graph of f at three distinct points --/
def intersects_at_three_points (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m

theorem intersection_range (m : ℝ) :
  intersects_at_three_points m → -2 < m ∧ m < 2 :=
by sorry

end intersection_range_l3176_317667


namespace nth_equation_l3176_317658

theorem nth_equation (n : ℕ) (hn : n > 0) :
  1 / (n + 2 : ℚ) + 2 / (n^2 + 2*n : ℚ) = 1 / n :=
by sorry

end nth_equation_l3176_317658


namespace dance_camp_rabbits_l3176_317656

theorem dance_camp_rabbits :
  ∀ (R S : ℕ),
  R + S = 50 →
  4 * R + 8 * S = 2 * R + 16 * S →
  R = 40 :=
by
  sorry

end dance_camp_rabbits_l3176_317656


namespace square_of_linear_expression_l3176_317654

theorem square_of_linear_expression (p : ℝ) (m : ℝ) : p ≠ 0 →
  (∃ a b : ℝ, ∀ x : ℝ, (9 * x^2 + 21 * x + 4 * m) / 9 = (a * x + b)^2) ∧
  (∃ a b : ℝ, (9 * (p - 1)^2 + 21 * (p - 1) + 4 * m) / 9 = (a * (p - 1) + b)^2) →
  m = 49 / 16 := by
sorry

end square_of_linear_expression_l3176_317654


namespace complex_fraction_simplification_l3176_317639

theorem complex_fraction_simplification :
  (I : ℂ) / (3 + 4 * I) = (4 : ℂ) / 25 + (3 : ℂ) / 25 * I :=
by sorry

end complex_fraction_simplification_l3176_317639


namespace melissa_games_l3176_317604

/-- The number of games Melissa played -/
def num_games : ℕ := 91 / 7

/-- The total points Melissa scored -/
def total_points : ℕ := 91

/-- The points Melissa scored per game -/
def points_per_game : ℕ := 7

theorem melissa_games : num_games = 13 := by
  sorry

end melissa_games_l3176_317604


namespace average_weight_BCDE_l3176_317674

/-- Given the weights of individuals A, B, C, D, and E, prove that the average weight of B, C, D, and E is 97.25 kg. -/
theorem average_weight_BCDE (w_A w_B w_C w_D w_E : ℝ) : 
  w_A = 77 →
  (w_A + w_B + w_C) / 3 = 84 →
  (w_A + w_B + w_C + w_D) / 4 = 80 →
  w_E = w_D + 5 →
  (w_B + w_C + w_D + w_E) / 4 = 97.25 := by
  sorry

end average_weight_BCDE_l3176_317674


namespace constant_function_theorem_l3176_317686

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The property that a function satisfies the given functional equation -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * floor y) = floor (f x) * f y

/-- The theorem stating that functions satisfying the equation are constant functions with values in [1, 2) -/
theorem constant_function_theorem (f : ℝ → ℝ) (h : satisfies_equation f) :
  ∃ c : ℝ, (∀ x : ℝ, f x = c) ∧ 1 ≤ c ∧ c < 2 := by
  sorry

end constant_function_theorem_l3176_317686


namespace max_digit_sum_for_valid_number_l3176_317618

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 2000 ∧ n < 3000 ∧ n % 13 = 0

def digit_sum (n : ℕ) : ℕ :=
  (n / 100 % 10) + (n / 10 % 10) + (n % 10)

theorem max_digit_sum_for_valid_number :
  ∃ (n : ℕ), is_valid_number n ∧
    ∀ (m : ℕ), is_valid_number m → digit_sum m ≤ digit_sum n ∧
    digit_sum n = 26 :=
sorry

end max_digit_sum_for_valid_number_l3176_317618


namespace total_chickens_l3176_317650

def farm_animals (ducks rabbits : ℕ) : Prop :=
  ∃ (hens roosters chickens : ℕ),
    hens = ducks + 20 ∧
    roosters = rabbits - 10 ∧
    chickens = hens + roosters ∧
    chickens = 80

theorem total_chickens : farm_animals 40 30 := by
  sorry

end total_chickens_l3176_317650


namespace problem_solution_l3176_317633

theorem problem_solution : -1^6 + 8 / (-2)^2 - |(-4) * 3| = -9 := by
  sorry

end problem_solution_l3176_317633


namespace cost_per_box_is_three_fifty_l3176_317661

/-- The cost per box of wafer cookies -/
def cost_per_box (num_trays : ℕ) (cookies_per_tray : ℕ) (cookies_per_box : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (((num_trays * cookies_per_tray) + cookies_per_box - 1) / cookies_per_box)

/-- Theorem stating that the cost per box is $3.50 given the problem conditions -/
theorem cost_per_box_is_three_fifty :
  cost_per_box 3 80 60 14 = 7/2 := by
  sorry

end cost_per_box_is_three_fifty_l3176_317661


namespace no_real_solution_for_equation_and_convergence_l3176_317687

theorem no_real_solution_for_equation_and_convergence : 
  ¬∃ y : ℝ, y = 2 / (1 + y) ∧ abs y < 1 := by
  sorry

end no_real_solution_for_equation_and_convergence_l3176_317687


namespace impossible_cross_sections_l3176_317623

-- Define a cube
structure Cube where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define a plane
structure Plane where
  normal_vector : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Define possible shapes of cross-sections
inductive CrossSectionShape
  | ObtuseTriangle
  | RightAngledTrapezoid
  | Rhombus
  | RegularPentagon
  | RegularHexagon

-- Function to determine if a shape is possible
def is_possible_cross_section (cube : Cube) (plane : Plane) (shape : CrossSectionShape) : Prop :=
  match shape with
  | CrossSectionShape.ObtuseTriangle => False
  | CrossSectionShape.RightAngledTrapezoid => False
  | CrossSectionShape.Rhombus => True
  | CrossSectionShape.RegularPentagon => False
  | CrossSectionShape.RegularHexagon => True

-- Theorem statement
theorem impossible_cross_sections (cube : Cube) (plane : Plane) :
  ¬(is_possible_cross_section cube plane CrossSectionShape.ObtuseTriangle) ∧
  ¬(is_possible_cross_section cube plane CrossSectionShape.RightAngledTrapezoid) ∧
  ¬(is_possible_cross_section cube plane CrossSectionShape.RegularPentagon) :=
sorry

end impossible_cross_sections_l3176_317623


namespace fraction_sum_equality_l3176_317659

theorem fraction_sum_equality (p q r : ℝ) 
  (h : p / (30 - p) + q / (75 - q) + r / (45 - r) = 8) :
  6 / (30 - p) + 15 / (75 - q) + 9 / (45 - r) = 11 / 5 := by
  sorry

end fraction_sum_equality_l3176_317659


namespace remainder_sum_l3176_317600

theorem remainder_sum (n : ℤ) : n % 12 = 5 → (n % 3 + n % 4 = 3) := by
  sorry

end remainder_sum_l3176_317600


namespace inequality_solutions_l3176_317669

theorem inequality_solutions :
  (∀ x : ℝ, x^2 + 3*x - 10 ≥ 0 ↔ x ≤ -5 ∨ x ≥ 2) ∧
  (∀ x : ℝ, x^2 - 3*x - 2 ≤ 0 ↔ (3 - Real.sqrt 17) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 17) / 2) :=
by sorry

end inequality_solutions_l3176_317669


namespace base_12_remainder_l3176_317624

-- Define the base-12 number 2625₁₂
def base_12_num : ℕ := 2 * 12^3 + 6 * 12^2 + 2 * 12 + 5

-- Theorem statement
theorem base_12_remainder :
  base_12_num % 10 = 9 := by
  sorry

end base_12_remainder_l3176_317624


namespace triangle_perimeter_from_average_side_length_l3176_317613

/-- The perimeter of a triangle with average side length 12 is 36 -/
theorem triangle_perimeter_from_average_side_length :
  ∀ (a b c : ℝ), 
  (a + b + c) / 3 = 12 →
  a + b + c = 36 := by
sorry

end triangle_perimeter_from_average_side_length_l3176_317613


namespace star_count_l3176_317629

theorem star_count (east : ℕ) (west : ℕ) : 
  east = 120 → 
  west = 6 * east → 
  east + west = 840 := by
sorry

end star_count_l3176_317629


namespace repeating_decimal_sum_l3176_317653

theorem repeating_decimal_sum : 
  (1 : ℚ) / 3 + 7 / 99 + 1 / 111 = 499 / 1189 := by sorry

end repeating_decimal_sum_l3176_317653


namespace log_equality_implies_golden_ratio_l3176_317641

theorem log_equality_implies_golden_ratio (p q : ℝ) 
  (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 8 = Real.log q / Real.log 12 ∧ 
       Real.log p / Real.log 8 = Real.log (p - q) / Real.log 18) : 
  q / p = (Real.sqrt 5 - 1) / 2 := by
  sorry

end log_equality_implies_golden_ratio_l3176_317641


namespace horner_method_example_l3176_317679

def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

theorem horner_method_example : f 2 = -80 := by
  sorry

end horner_method_example_l3176_317679


namespace average_speed_calculation_l3176_317652

/-- Given a distance of 88 miles and a time of 4 hours, prove that the average speed is 22 miles per hour. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 88) (h2 : time = 4) :
  distance / time = 22 := by
  sorry

end average_speed_calculation_l3176_317652


namespace eighty_one_to_negative_two_to_negative_two_equals_three_l3176_317614

theorem eighty_one_to_negative_two_to_negative_two_equals_three :
  (81 : ℝ) ^ (-(2 : ℝ)^(-(2 : ℝ))) = 3 := by
  sorry

end eighty_one_to_negative_two_to_negative_two_equals_three_l3176_317614


namespace red_ball_probability_l3176_317657

theorem red_ball_probability (n : ℕ) (r : ℕ) (k : ℕ) (h1 : n = 10) (h2 : r = 3) (h3 : k = 3) :
  let total_balls := n
  let red_balls := r
  let last_children := k
  let prob_one_red := (last_children.choose 1 : ℚ) * (red_balls / total_balls) * ((total_balls - red_balls) / total_balls) ^ 2
  prob_one_red = 441 / 1000 :=
by sorry

end red_ball_probability_l3176_317657


namespace six_is_simplified_quadratic_radical_l3176_317643

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_simplified_quadratic_radical (n : ℕ) : Prop :=
  n ≠ 0 ∧ ¬ is_perfect_square n ∧ ∀ m : ℕ, m > 1 → is_perfect_square m → ¬ (m ∣ n)

theorem six_is_simplified_quadratic_radical :
  is_simplified_quadratic_radical 6 :=
sorry

end six_is_simplified_quadratic_radical_l3176_317643


namespace race_heartbeats_l3176_317694

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (initial_rate : ℕ) (rate_increase : ℕ) (distance : ℕ) (pace : ℕ) : ℕ :=
  let final_rate := initial_rate + (distance - 1) * rate_increase
  let avg_rate := (initial_rate + final_rate) / 2
  avg_rate * distance * pace

/-- Theorem stating that the total heartbeats for the given conditions is 9750 -/
theorem race_heartbeats :
  total_heartbeats 140 5 10 6 = 9750 := by
  sorry

#eval total_heartbeats 140 5 10 6

end race_heartbeats_l3176_317694


namespace characterization_of_n_l3176_317668

def has_finite_multiples_with_n_divisors (n : ℕ+) : Prop :=
  ∃ (S : Finset ℕ+), ∀ (k : ℕ+), (n ∣ k) → (Nat.card (Nat.divisors k) = n) → k ∈ S

def not_divisible_by_square_of_prime (n : ℕ+) : Prop :=
  ∀ (p : ℕ+), Nat.Prime p → (p * p ∣ n) → False

theorem characterization_of_n (n : ℕ+) :
  has_finite_multiples_with_n_divisors n ↔ not_divisible_by_square_of_prime n ∨ n = 4 := by
  sorry

end characterization_of_n_l3176_317668


namespace min_value_of_f_range_of_a_l3176_317664

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

-- Theorem for the minimum value of f when a = 2
theorem min_value_of_f (x : ℝ) (h : x ≥ 1) :
  f 2 x ≥ 5 :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x ≥ 1, f a x > 0) ↔ a > -3 :=
sorry

end min_value_of_f_range_of_a_l3176_317664


namespace solid_color_non_yellow_purple_percentage_l3176_317628

/-- Represents the distribution of marble types and colors -/
structure MarbleDistribution where
  solid_colored : ℝ
  striped : ℝ
  dotted : ℝ
  swirl_patterned : ℝ
  red_solid : ℝ
  blue_solid : ℝ
  green_solid : ℝ
  yellow_solid : ℝ
  purple_solid : ℝ

/-- The given marble distribution -/
def given_distribution : MarbleDistribution :=
  { solid_colored := 0.70
    striped := 0.10
    dotted := 0.10
    swirl_patterned := 0.10
    red_solid := 0.25
    blue_solid := 0.25
    green_solid := 0.20
    yellow_solid := 0.15
    purple_solid := 0.15 }

/-- Theorem stating that 49% of all marbles are solid-colored and neither yellow nor purple -/
theorem solid_color_non_yellow_purple_percentage
  (d : MarbleDistribution)
  (h1 : d.solid_colored + d.striped + d.dotted + d.swirl_patterned = 1)
  (h2 : d.red_solid + d.blue_solid + d.green_solid + d.yellow_solid + d.purple_solid = 1)
  (h3 : d = given_distribution) :
  d.solid_colored * (d.red_solid + d.blue_solid + d.green_solid) = 0.49 := by
  sorry

end solid_color_non_yellow_purple_percentage_l3176_317628


namespace triple_angle_square_equal_to_circle_l3176_317611

-- Tripling an angle
theorem triple_angle (α : Real) : ∃ β, β = 3 * α := by sorry

-- Constructing a square equal in area to a given circle
theorem square_equal_to_circle (r : Real) : 
  ∃ s, s^2 = π * r^2 := by sorry

end triple_angle_square_equal_to_circle_l3176_317611


namespace inverse_proportion_doubling_l3176_317696

/-- Given two positive real numbers x and y that are inversely proportional,
    if x doubles, then y decreases by 50%. -/
theorem inverse_proportion_doubling (x y x' y' : ℝ) (k : ℝ) (hxy_pos : x > 0 ∧ y > 0) 
    (hk_pos : k > 0) (hxy : x * y = k) (hx'y' : x' * y' = k) (hx_double : x' = 2 * x) : 
    y' = y / 2 := by
  sorry

end inverse_proportion_doubling_l3176_317696


namespace max_value_of_expression_l3176_317680

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26*a*b*c) ≤ 3 :=
by sorry

end max_value_of_expression_l3176_317680


namespace compare_negative_numbers_l3176_317607

theorem compare_negative_numbers : -4 < -2.1 := by
  sorry

end compare_negative_numbers_l3176_317607


namespace sector_area_l3176_317670

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 4) (h2 : θ = π / 4) :
  (1 / 2) * θ * r^2 = 2 * π := by
  sorry

end sector_area_l3176_317670


namespace triangle_geometric_sequence_l3176_317606

/-- In a triangle ABC, if sides a, b, c form a geometric sequence and angle A is 60°,
    then (b * sin B) / c = √3/2 -/
theorem triangle_geometric_sequence (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (b / c = a / b) →  -- Geometric sequence condition
  A = π / 3 →        -- 60° in radians
  A + B + C = π →    -- Sum of angles in a triangle
  a = b * Real.sin A / Real.sin B →  -- Sine rule
  b = c * Real.sin B / Real.sin C →  -- Sine rule
  c = a * Real.sin C / Real.sin A →  -- Sine rule
  (b * Real.sin B) / c = Real.sqrt 3 / 2 := by
sorry


end triangle_geometric_sequence_l3176_317606


namespace deposit_percentage_l3176_317665

/-- Proves that the percentage P of the initial amount used in the deposit calculation is 30% --/
theorem deposit_percentage (initial_amount deposit_amount : ℝ) 
  (h1 : initial_amount = 50000)
  (h2 : deposit_amount = 750)
  (h3 : ∃ P : ℝ, deposit_amount = 0.20 * 0.25 * (P / 100) * initial_amount) :
  ∃ P : ℝ, P = 30 ∧ deposit_amount = 0.20 * 0.25 * (P / 100) * initial_amount :=
by sorry

end deposit_percentage_l3176_317665


namespace assignment_plans_l3176_317688

theorem assignment_plans (n_females : ℕ) (n_males : ℕ) (n_positions : ℕ) 
  (h_females : n_females = 10)
  (h_males : n_males = 40)
  (h_positions : n_positions = 5) :
  (n_females.choose 2) * 3 * 24 * (n_males.choose 3) = 
    Nat.choose n_females 2 * (Nat.factorial 3 / Nat.factorial 2) * 
    (Nat.factorial 4 / Nat.factorial 0) * Nat.choose n_males 3 :=
by sorry

end assignment_plans_l3176_317688


namespace T_is_three_rays_l3176_317640

/-- The set T of points in the coordinate plane -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               ((4 = x + 3 ∧ y - 5 ≤ 4) ∨
                (4 = y - 5 ∧ x + 3 ≤ 4) ∨
                (x + 3 = y - 5 ∧ 4 ≤ x + 3))}

/-- Definition of a ray starting from a point -/
def Ray (start : ℝ × ℝ) (dir : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (start.1 + t * dir.1, start.2 + t * dir.2)}

/-- The three rays that should compose T -/
def ThreeRays : Set (ℝ × ℝ) :=
  Ray (1, 9) (0, -1) ∪ Ray (1, 9) (-1, 0) ∪ Ray (1, 9) (1, 1)

theorem T_is_three_rays : T = ThreeRays := by sorry

end T_is_three_rays_l3176_317640


namespace min_value_quadratic_sum_l3176_317648

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (m : ℝ), m = 6/11 ∧ ∀ (a b c : ℝ), a + b + c = 1 → 2*a^2 + b^2 + 3*c^2 ≥ m :=
sorry

end min_value_quadratic_sum_l3176_317648


namespace sum_of_coefficients_l3176_317638

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
sorry

end sum_of_coefficients_l3176_317638


namespace parabola_no_y_intercepts_l3176_317698

/-- The parabola defined by x = 3y^2 - 5y + 12 has no y-intercepts -/
theorem parabola_no_y_intercepts :
  ∀ y : ℝ, 3 * y^2 - 5 * y + 12 ≠ 0 :=
by sorry

end parabola_no_y_intercepts_l3176_317698


namespace triangle_angle_measure_l3176_317673

theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_eq : a^2 + b^2 + Real.sqrt 2 * a * b = c^2) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  C = 3 * π / 4 := by
sorry

end triangle_angle_measure_l3176_317673


namespace regular_pentagon_perimeter_l3176_317615

/-- The sum of sides of a regular pentagon with side length 15 cm is 75 cm. -/
theorem regular_pentagon_perimeter (side_length : ℝ) (n_sides : ℕ) : 
  side_length = 15 → n_sides = 5 → side_length * n_sides = 75 := by
  sorry

end regular_pentagon_perimeter_l3176_317615


namespace car_average_mpg_l3176_317690

def initial_odometer : ℕ := 57300
def final_odometer : ℕ := 58300
def initial_gas : ℕ := 8
def second_gas : ℕ := 14
def final_gas : ℕ := 22

def total_distance : ℕ := final_odometer - initial_odometer
def total_gas : ℕ := initial_gas + second_gas + final_gas

def average_mpg : ℚ := total_distance / total_gas

theorem car_average_mpg :
  (round (average_mpg * 10) / 10 : ℚ) = 227/10 := by sorry

end car_average_mpg_l3176_317690


namespace square_plus_reciprocal_square_l3176_317612

theorem square_plus_reciprocal_square (n : ℝ) (h : n + 1/n = 10) :
  n^2 + 1/n^2 + 6 = 104 := by sorry

end square_plus_reciprocal_square_l3176_317612


namespace complex_number_in_second_quadrant_l3176_317671

theorem complex_number_in_second_quadrant : 
  let z : ℂ := 2 * I / (1 - I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_second_quadrant_l3176_317671


namespace carrot_count_l3176_317682

theorem carrot_count (initial_carrots thrown_out_carrots picked_next_day : ℕ) :
  initial_carrots = 48 →
  thrown_out_carrots = 11 →
  picked_next_day = 15 →
  initial_carrots - thrown_out_carrots + picked_next_day = 52 :=
by
  sorry

end carrot_count_l3176_317682


namespace inscribed_sphere_volume_l3176_317635

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 24) :
  let r := d / 4
  (4 / 3) * π * r^3 = 2304 * π := by sorry

end inscribed_sphere_volume_l3176_317635


namespace average_and_difference_l3176_317622

theorem average_and_difference (x : ℝ) : 
  (40 + x + 15) / 3 = 35 → |x - 40| = 10 := by
  sorry

end average_and_difference_l3176_317622


namespace mike_has_one_unbroken_seashell_l3176_317663

/-- Represents the number of unbroken seashells Mike has left after his beach trip and giving away one shell. -/
def unbroken_seashells_left : ℕ :=
  let total_seashells := 6
  let cone_shells := 3
  let conch_shells := 3
  let broken_cone_shells := 2
  let broken_conch_shells := 2
  let unbroken_cone_shells := cone_shells - broken_cone_shells
  let unbroken_conch_shells := conch_shells - broken_conch_shells
  let given_away_shells := 1
  unbroken_cone_shells + (unbroken_conch_shells - given_away_shells)

/-- Theorem stating that Mike has 1 unbroken seashell left. -/
theorem mike_has_one_unbroken_seashell : unbroken_seashells_left = 1 := by
  sorry

end mike_has_one_unbroken_seashell_l3176_317663


namespace conference_handshakes_theorem_l3176_317666

/-- The number of handshakes in a conference with special conditions -/
def conference_handshakes (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) - (k.choose 2)

/-- Theorem: In a conference of 30 people, where 3 specific people don't shake hands with each other,
    the total number of handshakes is 432 -/
theorem conference_handshakes_theorem :
  conference_handshakes 30 3 = 432 := by
  sorry

end conference_handshakes_theorem_l3176_317666


namespace incorrect_survey_method_statement_l3176_317627

-- Define survey methods
inductive SurveyMethod
| Sampling
| Comprehensive

-- Define scenarios
inductive Scenario
| StudentInterests
| ParentWorkConditions
| PopulationCensus
| LakeWaterQuality

-- Define function to determine appropriate survey method
def appropriateSurveyMethod (scenario : Scenario) : SurveyMethod :=
  match scenario with
  | Scenario.StudentInterests => SurveyMethod.Sampling
  | Scenario.ParentWorkConditions => SurveyMethod.Comprehensive
  | Scenario.PopulationCensus => SurveyMethod.Comprehensive
  | Scenario.LakeWaterQuality => SurveyMethod.Sampling

-- Theorem to prove
theorem incorrect_survey_method_statement :
  appropriateSurveyMethod Scenario.ParentWorkConditions ≠ SurveyMethod.Sampling :=
by sorry

end incorrect_survey_method_statement_l3176_317627


namespace simplify_fraction_l3176_317646

theorem simplify_fraction : (54 : ℚ) / 486 = 1 / 9 := by
  sorry

end simplify_fraction_l3176_317646


namespace choir_selection_l3176_317662

theorem choir_selection (boys girls : ℕ) (h1 : boys = 3) (h2 : girls = 5) :
  let total := boys + girls
  (Nat.choose boys 2 * Nat.choose girls 2 = 30) ∧
  (Nat.choose total 4 - Nat.choose girls 4 = 65) :=
by sorry

end choir_selection_l3176_317662


namespace f_is_convex_l3176_317631

/-- The function f(x) = x^4 - 2x^3 + 36x^2 - x + 7 -/
def f (x : ℝ) : ℝ := x^4 - 2*x^3 + 36*x^2 - x + 7

/-- The second derivative of f(x) -/
def f'' (x : ℝ) : ℝ := 12*x^2 - 12*x + 72

theorem f_is_convex : ConvexOn ℝ Set.univ f := by
  sorry

end f_is_convex_l3176_317631


namespace simple_interest_problem_l3176_317621

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * time

theorem simple_interest_problem (P : ℝ) : 
  simple_interest P 0.08 3 = 
  (1/2) * compound_interest 4000 0.1 2 → P = 1750 := by
  sorry

end simple_interest_problem_l3176_317621


namespace proportional_expression_l3176_317672

/-- Given that y is directly proportional to x-2 and y = -4 when x = 3,
    prove that the analytical expression of y with respect to x is y = -4x + 8 -/
theorem proportional_expression (x y : ℝ) :
  (∃ k : ℝ, ∀ x, y = k * (x - 2)) →  -- y is directly proportional to x-2
  (3 : ℝ) = x → (-4 : ℝ) = y →       -- when x = 3, y = -4
  y = -4 * x + 8 :=                   -- the analytical expression
by sorry

end proportional_expression_l3176_317672


namespace fractional_equation_m_range_l3176_317684

theorem fractional_equation_m_range :
  ∀ m x : ℝ,
  (m / (1 - x) - 2 / (x - 1) = 1) →
  (x ≥ 0) →
  (x ≠ 1) →
  (m ≤ -1 ∧ m ≠ -2) :=
by sorry

end fractional_equation_m_range_l3176_317684


namespace proposition_conjunction_false_perpendicular_lines_condition_converse_equivalence_l3176_317689

-- Statement 1
theorem proposition_conjunction_false :
  ¬(∃ x : ℝ, Real.tan x = 1 ∧ ¬(∀ x : ℝ, x^2 + 1 > 0)) := by sorry

-- Statement 2
theorem perpendicular_lines_condition :
  ∃ a b : ℝ, (∀ x y : ℝ, a * x + 3 * y - 1 = 0 ↔ x + b * y + 1 = 0) ∧
             (a * 1 + b * 3 = 0) ∧
             (a / b ≠ -3) := by sorry

-- Statement 3
theorem converse_equivalence :
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) := by sorry

end proposition_conjunction_false_perpendicular_lines_condition_converse_equivalence_l3176_317689


namespace money_distribution_l3176_317626

theorem money_distribution (total : ℝ) (share_d : ℝ) :
  let proportion_sum := 5 + 2 + 4 + 3
  let proportion_d := 3
  let proportion_c := 4
  share_d = 1500 →
  share_d = (proportion_d / proportion_sum) * total →
  let share_c := (proportion_c / proportion_sum) * total
  share_c - share_d = 500 :=
by sorry

end money_distribution_l3176_317626


namespace max_sphere_in_intersecting_cones_l3176_317619

/-- 
Given two congruent right circular cones with base radius 5 and height 12,
whose axes of symmetry intersect at right angles at a point 4 units from
the base of each cone, prove that the maximum possible value of r^2 for a
sphere lying within both cones is 625/169.
-/
theorem max_sphere_in_intersecting_cones (r : ℝ) : 
  let base_radius : ℝ := 5
  let cone_height : ℝ := 12
  let intersection_distance : ℝ := 4
  let slant_height : ℝ := Real.sqrt (cone_height^2 + base_radius^2)
  let max_r_squared : ℝ := (base_radius * (slant_height - intersection_distance) / slant_height)^2
  max_r_squared = 625 / 169 :=
by sorry

end max_sphere_in_intersecting_cones_l3176_317619


namespace sum_of_ninth_powers_of_roots_l3176_317699

theorem sum_of_ninth_powers_of_roots (u v w : ℂ) : 
  (u^3 - 3*u - 1 = 0) → 
  (v^3 - 3*v - 1 = 0) → 
  (w^3 - 3*w - 1 = 0) → 
  u^9 + v^9 + w^9 = 246 := by sorry

end sum_of_ninth_powers_of_roots_l3176_317699


namespace candy_bars_problem_l3176_317692

theorem candy_bars_problem (fred : ℕ) (bob : ℕ) (jacqueline : ℕ) : 
  fred = 12 →
  bob = fred + 6 →
  jacqueline = 10 * (fred + bob) →
  (40 : ℝ) / 100 * jacqueline = 120 := by
  sorry

end candy_bars_problem_l3176_317692


namespace difference_sum_rational_product_irrational_l3176_317625

theorem difference_sum_rational_product_irrational : 
  let a : ℝ := 8
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 3 - 1
  let d : ℝ := 3 * Real.sqrt 3
  (a + b) - (c * d) = 3 * Real.sqrt 3 := by sorry

end difference_sum_rational_product_irrational_l3176_317625


namespace line_through_two_points_l3176_317601

/-- A line in the rectangular coordinate system -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the rectangular coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (l : Line) (p : Point) : Prop :=
  p.x = l.slope * p.y + l.intercept

theorem line_through_two_points 
  (l : Line)
  (p1 p2 : Point)
  (h1 : pointOnLine l p1)
  (h2 : pointOnLine l p2)
  (h3 : l.slope = 8)
  (h4 : l.intercept = 5)
  (h5 : p2.x = p1.x + 2)
  (h6 : p2.y = p1.y + p)
  : p = 1/4 := by
  sorry

#check line_through_two_points

end line_through_two_points_l3176_317601


namespace lunch_cost_proof_l3176_317681

/-- The cost of Mike's additional items -/
def mike_additional : ℝ := 11.75

/-- The cost of John's additional items -/
def john_additional : ℝ := 5.25

/-- The ratio of Mike's bill to John's bill -/
def bill_ratio : ℝ := 1.5

/-- The combined total cost of Mike and John's lunch -/
def total_cost : ℝ := 58.75

theorem lunch_cost_proof :
  ∃ (taco_grande_price : ℝ),
    taco_grande_price > 0 ∧
    (taco_grande_price + mike_additional) = bill_ratio * (taco_grande_price + john_additional) ∧
    (taco_grande_price + mike_additional) + (taco_grande_price + john_additional) = total_cost := by
  sorry

end lunch_cost_proof_l3176_317681


namespace triangle_side_sum_range_l3176_317616

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (law_of_sines : a / Real.sin A = b / Real.sin B)
  (law_of_cosines : Real.cos A = (b^2 + c^2 - a^2) / (2*b*c))

-- State the theorem
theorem triangle_side_sum_range (t : Triangle) 
  (h1 : Real.cos t.A / t.a + Real.cos t.C / t.c = Real.sin t.B * Real.sin t.C / (3 * Real.sin t.A))
  (h2 : Real.sqrt 3 * Real.sin t.C + Real.cos t.C = 2) :
  6 < t.a + t.b ∧ t.a + t.b ≤ 4 * Real.sqrt 3 :=
by sorry

end triangle_side_sum_range_l3176_317616


namespace water_added_third_hour_is_one_l3176_317651

/-- Calculates the amount of water added in the third hour -/
def water_added_third_hour (initial_water : ℝ) (loss_rate : ℝ) (fourth_hour_addition : ℝ) (final_water : ℝ) : ℝ :=
  final_water - (initial_water - 3 * loss_rate + fourth_hour_addition)

theorem water_added_third_hour_is_one :
  let initial_water : ℝ := 40
  let loss_rate : ℝ := 2
  let fourth_hour_addition : ℝ := 3
  let final_water : ℝ := 36
  water_added_third_hour initial_water loss_rate fourth_hour_addition final_water = 1 := by
  sorry

#eval water_added_third_hour 40 2 3 36

end water_added_third_hour_is_one_l3176_317651


namespace josh_doug_money_ratio_l3176_317637

/-- Proves that the ratio of Josh's money to Doug's money is 3:4 given the problem conditions -/
theorem josh_doug_money_ratio :
  ∀ (josh doug brad : ℕ),
  josh + doug + brad = 68 →
  josh = 2 * brad →
  doug = 32 →
  (josh : ℚ) / doug = 3 / 4 := by
sorry

end josh_doug_money_ratio_l3176_317637


namespace smallest_perimeter_after_folding_l3176_317647

theorem smallest_perimeter_after_folding (l w : ℝ) (hl : l = 17 / 2) (hw : w = 11) : 
  let original_perimeter := 2 * l + 2 * w
  let folded_perimeter1 := 2 * l + 2 * (w / 4)
  let folded_perimeter2 := 2 * (l / 2) + 2 * (w / 2)
  min folded_perimeter1 folded_perimeter2 = 39 / 2 := by
sorry

end smallest_perimeter_after_folding_l3176_317647


namespace hannah_running_difference_l3176_317609

/-- Hannah's running distances for different days of the week -/
structure RunningDistances where
  monday : ℕ     -- Distance in kilometers
  wednesday : ℕ  -- Distance in meters
  friday : ℕ     -- Distance in meters

/-- Calculates the difference in meters between Monday's run and the combined Wednesday and Friday runs -/
def run_difference (distances : RunningDistances) : ℕ :=
  distances.monday * 1000 - (distances.wednesday + distances.friday)

/-- Theorem stating the difference in Hannah's running distances -/
theorem hannah_running_difference : 
  let distances : RunningDistances := { monday := 9, wednesday := 4816, friday := 2095 }
  run_difference distances = 2089 := by
  sorry

end hannah_running_difference_l3176_317609


namespace sum_of_a_and_b_l3176_317636

theorem sum_of_a_and_b (a b : ℚ) : 5 - Real.sqrt 3 * a = 2 * b + Real.sqrt 3 - a → a + b = 1 := by
  sorry

end sum_of_a_and_b_l3176_317636


namespace parent_chaperones_count_l3176_317677

/-- The number of parent chaperones on a school field trip -/
def num_parent_chaperones (total_students : ℕ) (num_teachers : ℕ) (students_left : ℕ) (chaperones_left : ℕ) (remaining_individuals : ℕ) : ℕ :=
  (remaining_individuals + students_left + chaperones_left) - (total_students + num_teachers)

theorem parent_chaperones_count :
  num_parent_chaperones 20 2 10 2 15 = 5 := by
  sorry

end parent_chaperones_count_l3176_317677


namespace rectangle_division_l3176_317649

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Represents the large rectangle ABCD -/
def large_rectangle : Rectangle := { a := 18, b := 16 }

/-- Represents a small rectangle within ABCD -/
structure SmallRectangle where
  x : ℝ
  y : ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.a + r.b)

/-- The perimeter of a small rectangle -/
def small_perimeter (r : SmallRectangle) : ℝ := 2 * (r.x + r.y)

/-- The theorem to be proved -/
theorem rectangle_division (small1 small2 small3 small4 : SmallRectangle) :
  large_rectangle.a = 18 ∧ large_rectangle.b = 16 ∧
  small_perimeter small1 = small_perimeter small2 ∧
  small_perimeter small2 = small_perimeter small3 ∧
  small_perimeter small3 = small_perimeter small4 ∧
  small1.x + small2.x + small3.x = large_rectangle.a ∧
  small1.y + small2.y + small3.y + small4.y = large_rectangle.b →
  (small1.x = 2 ∧ small1.y = 18 ∧
   small2.x = 6 ∧ small2.y = 14 ∧
   small3.x = 6 ∧ small3.y = 14 ∧
   small4.x = 6 ∧ small4.y = 14) :=
by
  sorry

end rectangle_division_l3176_317649


namespace sin_two_theta_l3176_317605

theorem sin_two_theta (θ : ℝ) (h : Real.sin (π/4 + θ) = 1/3) : Real.sin (2*θ) = -7/9 := by
  sorry

end sin_two_theta_l3176_317605
