import Mathlib

namespace total_volume_of_cubes_l2116_211632

def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

def carl_cubes : ℕ := 6
def carl_side_length : ℝ := 1

def kate_cubes : ℕ := 4
def kate_side_length : ℝ := 3

theorem total_volume_of_cubes :
  (carl_cubes : ℝ) * cube_volume carl_side_length +
  (kate_cubes : ℝ) * cube_volume kate_side_length = 114 := by
  sorry

end total_volume_of_cubes_l2116_211632


namespace otimes_one_eq_two_implies_k_eq_one_l2116_211675

def otimes (a b : ℝ) : ℝ := a * b + a + b^2

theorem otimes_one_eq_two_implies_k_eq_one (k : ℝ) (h1 : k > 0) (h2 : otimes 1 k = 2) : k = 1 := by
  sorry

end otimes_one_eq_two_implies_k_eq_one_l2116_211675


namespace product_simplification_l2116_211668

theorem product_simplification (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^8 + 21 * x^4 * y^3 + 49 * y^6) = 27 * x^12 - 343 * y^9 := by
  sorry

end product_simplification_l2116_211668


namespace complex_magnitude_squared_l2116_211661

theorem complex_magnitude_squared (z : ℂ) (h : z + Complex.abs z = -3 + 4*I) : Complex.abs z ^ 2 = 625 / 36 := by
  sorry

end complex_magnitude_squared_l2116_211661


namespace existence_of_special_number_l2116_211682

theorem existence_of_special_number (P : Finset Nat) (h_prime : ∀ p ∈ P, Prime p) :
  ∃ x : Nat,
    (∀ p ∈ P, ∃ a b : Nat, x = a^p + b^p) ∧
    (∀ p : Nat, Prime p → p ∉ P → ¬∃ a b : Nat, x = a^p + b^p) := by
  sorry

end existence_of_special_number_l2116_211682


namespace arithmetic_sequence_theorem_l2116_211655

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

-- Define the solution set of the quadratic inequality
def solution_set (a : ℕ → ℝ) : Set ℝ :=
  {x : ℝ | x^2 - a 3 * x + a 4 ≤ 0}

-- Theorem statement
theorem arithmetic_sequence_theorem (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  solution_set a = {x : ℝ | a 1 ≤ x ∧ x ≤ a 2} →
  ∀ n : ℕ, a n = 2 * n :=
by sorry

end arithmetic_sequence_theorem_l2116_211655


namespace exist_four_cells_l2116_211613

/-- Represents a cell in the grid -/
structure Cell :=
  (x : Fin 17)
  (y : Fin 17)
  (value : Fin 70)

/-- The type of the grid -/
def Grid := Fin 17 → Fin 17 → Fin 70

/-- Predicate to check if all numbers from 1 to 70 appear exactly once in the grid -/
def valid_grid (g : Grid) : Prop :=
  ∀ n : Fin 70, ∃! (x y : Fin 17), g x y = n

/-- Distance between two cells -/
def distance (a b : Cell) : ℕ :=
  (a.x - b.x) ^ 2 + (a.y - b.y) ^ 2

/-- Sum of values in two cells -/
def sum_values (a b : Cell) : ℕ :=
  a.value.val + b.value.val

/-- Main theorem -/
theorem exist_four_cells (g : Grid) (h : valid_grid g) :
  ∃ (a b c d : Cell),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    distance a b = distance c d ∧
    distance a d = distance b c ∧
    sum_values a c = sum_values b d :=
  sorry

end exist_four_cells_l2116_211613


namespace wage_percentage_difference_l2116_211634

/-- Proves that the percentage difference between chef's and dishwasher's hourly wage is 20% -/
theorem wage_percentage_difference
  (manager_wage : ℝ)
  (chef_wage_difference : ℝ)
  (h_manager_wage : manager_wage = 6.50)
  (h_chef_wage_difference : chef_wage_difference = 2.60)
  (h_dishwasher_wage : dishwasher_wage = manager_wage / 2)
  (h_chef_wage : chef_wage = manager_wage - chef_wage_difference) :
  (chef_wage - dishwasher_wage) / dishwasher_wage * 100 = 20 :=
by sorry


end wage_percentage_difference_l2116_211634


namespace exists_initial_order_l2116_211666

/-- Represents a playing card suit -/
inductive Suit
| Diamonds
| Hearts
| Spades
| Clubs

/-- Represents a playing card rank -/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents a playing card -/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- The number of letters in the name of a card's rank -/
def letterCount : Rank → Nat
| Rank.Ace => 3
| Rank.Two => 3
| Rank.Three => 5
| Rank.Four => 4
| Rank.Five => 4
| Rank.Six => 3
| Rank.Seven => 5
| Rank.Eight => 5
| Rank.Nine => 4
| Rank.Ten => 3
| Rank.Jack => 4
| Rank.Queen => 5
| Rank.King => 4

/-- Applies the card moving rule to a deck -/
def applyRule (deck : List Card) : List Card :=
  sorry

/-- The final order of cards after applying the rule -/
def finalOrder : List Card :=
  sorry

/-- Theorem: There exists an initial deck ordering that results in the specified final order -/
theorem exists_initial_order :
  ∃ (initialDeck : List Card),
    initialDeck.length = 52 ∧
    (∀ s : Suit, ∀ r : Rank, ∃ c ∈ initialDeck, c.suit = s ∧ c.rank = r) ∧
    applyRule initialDeck = finalOrder :=
  sorry

end exists_initial_order_l2116_211666


namespace new_boarders_count_l2116_211617

theorem new_boarders_count (initial_boarders : ℕ) (initial_ratio_boarders initial_ratio_day_scholars : ℕ) 
  (final_ratio_boarders final_ratio_day_scholars : ℕ) :
  initial_boarders = 560 →
  initial_ratio_boarders = 7 →
  initial_ratio_day_scholars = 16 →
  final_ratio_boarders = 1 →
  final_ratio_day_scholars = 2 →
  ∃ (new_boarders : ℕ),
    (initial_boarders + new_boarders) * final_ratio_day_scholars = 
    (initial_boarders * initial_ratio_day_scholars / initial_ratio_boarders) * final_ratio_boarders ∧
    new_boarders = 80 :=
by sorry

end new_boarders_count_l2116_211617


namespace special_function_is_zero_l2116_211663

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≥ 0) ∧
  (∀ x, DifferentiableAt ℝ f x) ∧
  (∀ x, deriv f x ≥ 0) ∧
  (∀ n : ℤ, f n = 0)

/-- Theorem stating that any function satisfying the conditions must be identically zero -/
theorem special_function_is_zero (f : ℝ → ℝ) (hf : SpecialFunction f) : 
  ∀ x, f x = 0 := by sorry

end special_function_is_zero_l2116_211663


namespace tangent_line_equations_l2116_211671

-- Define the function
def f (x : ℝ) : ℝ := -x^3 - 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -3 * x^2

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the tangent line equation
def tangent_line (m : ℝ) (x : ℝ) : ℝ := 
  -m^3 - 1 + (f' m) * (x - m)

-- Theorem statement
theorem tangent_line_equations :
  ∃ (m₁ m₂ : ℝ), 
    m₁ ≠ m₂ ∧
    tangent_line m₁ P.1 = P.2 ∧
    tangent_line m₂ P.1 = P.2 ∧
    (∀ (x : ℝ), tangent_line m₁ x = -3*x - 1) ∧
    (∀ (x : ℝ), tangent_line m₂ x = -(3*x + 5) / 4) :=
sorry

end tangent_line_equations_l2116_211671


namespace parabola_focus_theorem_l2116_211664

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line type -/
structure Line

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle type -/
structure Circle where
  center : Point
  radius : ℝ

def intersects (l : Line) (para : Parabola) (A B : Point) : Prop :=
  sorry

def focus_on_line (para : Parabola) (l : Line) : Prop :=
  sorry

def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.x)^2 + (y - c.center.y)^2 = c.radius^2

def is_diameter (A B : Point) (c : Circle) : Prop :=
  sorry

theorem parabola_focus_theorem (para : Parabola) (l : Line) (A B : Point) (c : Circle) :
  intersects l para A B →
  focus_on_line para l →
  is_diameter A B c →
  circle_equation c 3 2 →
  c.radius = 4 →
  para.p = 2 :=
sorry

end parabola_focus_theorem_l2116_211664


namespace identical_solutions_l2116_211611

/-- Two equations have identical solutions when k = 0 -/
theorem identical_solutions (x y k : ℝ) : 
  (y = x^2 ∧ y = 3*x^2 + k) ↔ k = 0 :=
by sorry

end identical_solutions_l2116_211611


namespace y_derivative_l2116_211696

noncomputable def y (x : ℝ) : ℝ := Real.cos (Real.log 2) - (1/3) * (Real.cos (3*x))^2 / Real.sin (6*x)

theorem y_derivative (x : ℝ) (h : Real.sin (6*x) ≠ 0) :
  deriv y x = 1 / (2 * (Real.sin (3*x))^2) :=
by sorry

end y_derivative_l2116_211696


namespace perimeter_folded_square_l2116_211630

/-- Given a square ABCD with side length 2, where A is folded to meet BC at A' such that A'C = 1/2,
    the perimeter of triangle A'BD is (3 + √17)/2 + 2√2. -/
theorem perimeter_folded_square (A B C D A' : ℝ × ℝ) : 
  (∀ (X Y : ℝ × ℝ), ‖X - Y‖ = 2 → (X = A ∧ Y = B) ∨ (X = B ∧ Y = C) ∨ (X = C ∧ Y = D) ∨ (X = D ∧ Y = A)) →
  A'.1 = B.1 + 3/2 →
  A'.2 = B.2 →
  C.1 = B.1 + 2 →
  C.2 = B.2 →
  ‖A' - C‖ = 1/2 →
  ‖A' - B‖ + ‖B - D‖ + ‖D - A'‖ = (3 + Real.sqrt 17) / 2 + 2 * Real.sqrt 2 :=
by sorry

end perimeter_folded_square_l2116_211630


namespace f_6_equals_21_l2116_211621

def f (x : ℝ) : ℝ := (x - 1)^2 - 4

theorem f_6_equals_21 : f 6 = 21 := by
  sorry

end f_6_equals_21_l2116_211621


namespace average_of_subset_l2116_211687

theorem average_of_subset (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 63 →
  (list.get! 2 + list.get! 3) / 2 = 60 →
  (list.get! 0 + list.get! 1 + list.get! 4 + list.get! 5 + list.get! 6) / 5 = 64.2 := by
sorry

end average_of_subset_l2116_211687


namespace imaginary_part_z2_l2116_211640

theorem imaginary_part_z2 (z₁ z₂ : ℂ) : 
  z₁ = 2 - 3*I → z₁ * z₂ = 1 + 2*I → z₂.im = 7/13 := by
  sorry

end imaginary_part_z2_l2116_211640


namespace no_safe_numbers_l2116_211638

def is_p_safe (n p : ℕ+) : Prop :=
  ∀ k : ℤ, |n.val - k * p.val| > 3

theorem no_safe_numbers : 
  ¬ ∃ n : ℕ+, n.val ≤ 15000 ∧ 
    is_p_safe n 5 ∧ 
    is_p_safe n 7 ∧ 
    is_p_safe n 11 :=
sorry

end no_safe_numbers_l2116_211638


namespace sum_congruence_mod_nine_l2116_211633

theorem sum_congruence_mod_nine :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end sum_congruence_mod_nine_l2116_211633


namespace diagonals_in_30_sided_polygon_l2116_211644

theorem diagonals_in_30_sided_polygon : ∀ (n : ℕ), n = 30 → (n * (n - 3)) / 2 = 405 := by
  sorry

end diagonals_in_30_sided_polygon_l2116_211644


namespace sqrt_33_between_5_and_6_l2116_211650

theorem sqrt_33_between_5_and_6 : 5 < Real.sqrt 33 ∧ Real.sqrt 33 < 6 := by
  sorry

end sqrt_33_between_5_and_6_l2116_211650


namespace mailing_weight_calculation_l2116_211654

/-- The total weight of a mailing with multiple envelopes and additional materials -/
def total_mailing_weight (envelope_weight : ℝ) (num_envelopes : ℕ) (additional_weight : ℝ) : ℝ :=
  (envelope_weight + additional_weight) * num_envelopes

/-- Theorem stating that the total weight of the mailing is 9240 grams -/
theorem mailing_weight_calculation :
  total_mailing_weight 8.5 880 2 = 9240 := by
  sorry

end mailing_weight_calculation_l2116_211654


namespace eqLength_is_53_l2116_211649

/-- Represents a trapezoid with a circle inscribed in it. -/
structure InscribedTrapezoid where
  /-- Length of side EF -/
  ef : ℝ
  /-- Length of side FG -/
  fg : ℝ
  /-- Length of side GH -/
  gh : ℝ
  /-- Length of side HE -/
  he : ℝ
  /-- EF is parallel to GH -/
  parallel : ef > gh

/-- The length of EQ in the inscribed trapezoid. -/
def eqLength (t : InscribedTrapezoid) : ℝ := sorry

/-- Theorem stating that for the given trapezoid, EQ = 53 -/
theorem eqLength_is_53 (t : InscribedTrapezoid) 
  (h1 : t.ef = 84) 
  (h2 : t.fg = 58) 
  (h3 : t.gh = 27) 
  (h4 : t.he = 64) : 
  eqLength t = 53 := by sorry

end eqLength_is_53_l2116_211649


namespace betty_has_winning_strategy_l2116_211651

/-- Represents the state of a bowl -/
structure BowlState :=
  (redBalls : Nat)
  (blueBalls : Nat)

/-- Represents the state of the game -/
structure GameState :=
  (blueBowl : BowlState)
  (redBowl : BowlState)

/-- Enum for the possible moves in the game -/
inductive Move
  | TakeRedFromBlue
  | TakeBlueFromRed
  | ThrowAway

/-- Enum for the players -/
inductive Player
  | Albert
  | Betty

/-- Function to check if a game state is winning for the current player -/
def isWinningState (state : GameState) : Bool :=
  state.blueBowl.redBalls = 0 || state.redBowl.blueBalls = 0

/-- Function to get the next player -/
def nextPlayer (player : Player) : Player :=
  match player with
  | Player.Albert => Player.Betty
  | Player.Betty => Player.Albert

/-- Function to apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeRedFromBlue => 
      { blueBowl := { redBalls := state.blueBowl.redBalls - 2, blueBalls := state.blueBowl.blueBalls },
        redBowl := { redBalls := state.redBowl.redBalls + 2, blueBalls := state.redBowl.blueBalls } }
  | Move.TakeBlueFromRed => 
      { blueBowl := { redBalls := state.blueBowl.redBalls, blueBalls := state.blueBowl.blueBalls + 2 },
        redBowl := { redBalls := state.redBowl.redBalls, blueBalls := state.redBowl.blueBalls - 2 } }
  | Move.ThrowAway => 
      { blueBowl := { redBalls := state.blueBowl.redBalls - 1, blueBalls := state.blueBowl.blueBalls - 1 },
        redBowl := state.redBowl }

/-- The initial state of the game -/
def initialState : GameState :=
  { blueBowl := { redBalls := 100, blueBalls := 0 },
    redBowl := { redBalls := 0, blueBalls := 100 } }

/-- Theorem stating that Betty has a winning strategy -/
theorem betty_has_winning_strategy :
  ∃ (strategy : Player → GameState → Move),
    ∀ (game : Nat → GameState),
      game 0 = initialState →
      (∀ n, game (n + 1) = applyMove (game n) (strategy (if n % 2 = 0 then Player.Albert else Player.Betty) (game n))) →
      ∃ n, isWinningState (game n) ∧ n % 2 = 1 :=
sorry


end betty_has_winning_strategy_l2116_211651


namespace sum_of_solutions_squared_equation_l2116_211685

theorem sum_of_solutions_squared_equation (x : ℝ) :
  (∃ a b : ℝ, (a - 8)^2 = 49 ∧ (b - 8)^2 = 49 ∧ a ≠ b) →
  (∃ s : ℝ, s = a + b ∧ s = 16) :=
by sorry

end sum_of_solutions_squared_equation_l2116_211685


namespace june_decrease_percentage_l2116_211648

-- Define the price changes for each month
def january_change : ℝ := 0.15
def february_change : ℝ := -0.10
def march_change : ℝ := 0.20
def april_change : ℝ := -0.30
def may_change : ℝ := 0.10

-- Function to calculate the price after a change
def apply_change (price : ℝ) (change : ℝ) : ℝ :=
  price * (1 + change)

-- Theorem stating the required decrease in June
theorem june_decrease_percentage (initial_price : ℝ) (initial_price_pos : initial_price > 0) :
  let final_price := apply_change (apply_change (apply_change (apply_change (apply_change initial_price january_change) february_change) march_change) april_change) may_change
  ∃ (june_decrease : ℝ), 
    (apply_change final_price june_decrease = initial_price) ∧ 
    (abs (june_decrease + 0.0456) < 0.0001) := by
  sorry

end june_decrease_percentage_l2116_211648


namespace average_monthly_balance_l2116_211642

def january_balance : ℝ := 120
def february_balance : ℝ := 250
def march_balance : ℝ := 200
def april_balance : ℝ := 200
def may_balance : ℝ := 180
def num_months : ℝ := 5

theorem average_monthly_balance :
  (january_balance + february_balance + march_balance + april_balance + may_balance) / num_months = 190 := by
  sorry

end average_monthly_balance_l2116_211642


namespace exists_acute_triangle_l2116_211673

/-- A set of 5 positive real numbers representing segment lengths -/
def SegmentSet : Type := Fin 5 → ℝ

/-- Predicate to check if three numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate to check if a triangle is acute-angled -/
def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

/-- Main theorem: Given 5 segments where any three can form a triangle,
    there exists at least one acute-angled triangle -/
theorem exists_acute_triangle (s : SegmentSet) 
  (h_positive : ∀ i, s i > 0)
  (h_triangle : ∀ i j k, i ≠ j → j ≠ k → k ≠ i → can_form_triangle (s i) (s j) (s k)) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ is_acute_triangle (s i) (s j) (s k) := by
  sorry


end exists_acute_triangle_l2116_211673


namespace original_expenditure_problem_l2116_211619

/-- The original expenditure problem -/
theorem original_expenditure_problem :
  ∀ (E A : ℕ),
  -- Initial conditions
  E = 35 * A →
  -- After first admission
  E + 84 = 42 * (A - 1) →
  -- After second change
  E + 124 = 37 * (A + 1) →
  -- Conclusion
  E = 630 := by
  sorry

end original_expenditure_problem_l2116_211619


namespace largest_mersenne_prime_under_500_l2116_211615

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

def is_mersenne_prime (m : Nat) : Prop :=
  ∃ n : Nat, is_prime n ∧ m = 2^n - 1 ∧ is_prime m

theorem largest_mersenne_prime_under_500 :
  ∀ m : Nat, is_mersenne_prime m → m < 500 → m ≤ 127 :=
by sorry

end largest_mersenne_prime_under_500_l2116_211615


namespace paper_clips_count_l2116_211657

/-- The number of paper clips in a storage unit -/
def total_paper_clips (c b p : ℕ) : ℕ :=
  300 * c + 550 * b + 1200 * p

/-- Theorem stating that the total number of paper clips in 3 cartons, 4 boxes, and 2 bags is 5500 -/
theorem paper_clips_count : total_paper_clips 3 4 2 = 5500 := by
  sorry

end paper_clips_count_l2116_211657


namespace max_value_sum_and_reciprocal_l2116_211622

theorem max_value_sum_and_reciprocal (nums : Finset ℝ) (x : ℝ) :
  (Finset.card nums = 11) →
  (∀ y ∈ nums, y > 0) →
  (x ∈ nums) →
  (Finset.sum nums id = 102) →
  (Finset.sum nums (λ y => 1 / y) = 102) →
  (x + 1 / x ≤ 10304 / 102) :=
by sorry

end max_value_sum_and_reciprocal_l2116_211622


namespace hexagon_perimeter_sum_l2116_211660

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  perimeter : ℝ

/-- Represents a hexagon formed by two equilateral triangles -/
def hexagon_from_triangles (t1 t2 : EquilateralTriangle) : ℝ := t1.perimeter + t2.perimeter

theorem hexagon_perimeter_sum (t1 t2 : EquilateralTriangle) 
  (h1 : t1.perimeter = 12) (h2 : t2.perimeter = 15) : 
  hexagon_from_triangles t1 t2 = 27 := by
  sorry

#check hexagon_perimeter_sum

end hexagon_perimeter_sum_l2116_211660


namespace dvd_pack_cost_l2116_211691

/-- If 10 packs of DVDs cost 110 dollars, then the cost of one pack is 11 dollars -/
theorem dvd_pack_cost (total_cost : ℝ) (num_packs : ℕ) (h1 : total_cost = 110) (h2 : num_packs = 10) :
  total_cost / num_packs = 11 := by
  sorry

end dvd_pack_cost_l2116_211691


namespace intersection_point_a_l2116_211690

-- Define the function f
def f (b : ℤ) (x : ℝ) : ℝ := 2 * x + b

-- Define the theorem
theorem intersection_point_a (b : ℤ) (a : ℤ) :
  (∃ (x : ℝ), f b x = a ∧ f b (-4) = a) →  -- f and f^(-1) intersect at (-4, a)
  a = -4 :=
by
  sorry

end intersection_point_a_l2116_211690


namespace parabola_directrix_l2116_211680

/-- The equation of a parabola -/
def parabola_eq (x y : ℝ) : Prop := y^2 = -12*x

/-- The equation of the directrix -/
def directrix_eq (x : ℝ) : Prop := x = 3

/-- Theorem: The directrix of the parabola y^2 = -12x is x = 3 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_eq x y → directrix_eq x :=
by sorry

end parabola_directrix_l2116_211680


namespace red_peaches_count_l2116_211683

/-- Given a basket of peaches, calculate the number of red peaches. -/
def red_peaches (total_peaches green_peaches : ℕ) : ℕ :=
  total_peaches - green_peaches

/-- Theorem: The number of red peaches in the basket is 4. -/
theorem red_peaches_count : red_peaches 7 3 = 4 := by
  sorry

end red_peaches_count_l2116_211683


namespace continuity_at_negative_one_l2116_211645

noncomputable def f (x : ℝ) : ℝ := (x^3 - x^2 + x + 1) / (x^2 - 1)

theorem continuity_at_negative_one :
  Filter.Tendsto f (nhds (-1)) (nhds (-2)) :=
sorry

end continuity_at_negative_one_l2116_211645


namespace circumcenter_rational_coords_l2116_211603

/-- If the coordinates of the vertices of a triangle are rational, 
    then the coordinates of the center of its circumscribed circle are also rational. -/
theorem circumcenter_rational_coords 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℚ) : ∃ x y : ℚ, 
  (x - a₁)^2 + (y - b₁)^2 = (x - a₂)^2 + (y - b₂)^2 ∧
  (x - a₁)^2 + (y - b₁)^2 = (x - a₃)^2 + (y - b₃)^2 := by
  sorry

end circumcenter_rational_coords_l2116_211603


namespace original_car_price_l2116_211624

theorem original_car_price (used_price : ℝ) (percentage : ℝ) (original_price : ℝ) : 
  used_price = 15000 →
  percentage = 0.40 →
  used_price = percentage * original_price →
  original_price = 37500 := by
sorry

end original_car_price_l2116_211624


namespace discount_sales_income_increase_l2116_211656

/-- Proves that a 10% discount with 30% increase in sales leads to 17% increase in gross income -/
theorem discount_sales_income_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (discount_rate : ℝ) 
  (sales_increase_rate : ℝ) 
  (h1 : discount_rate = 0.1) 
  (h2 : sales_increase_rate = 0.3) : 
  (((1 + sales_increase_rate) * (1 - discount_rate) - 1) * 100 : ℝ) = 17 := by
  sorry

end discount_sales_income_increase_l2116_211656


namespace complex_multiplication_l2116_211662

theorem complex_multiplication (z : ℂ) : 
  (z.re = -1 ∧ z.im = 1) → z * (1 + Complex.I) = -2 := by
  sorry

end complex_multiplication_l2116_211662


namespace senior_sports_solution_l2116_211602

def senior_sports_problem (total_seniors : ℕ) 
  (football : Finset ℕ) (baseball : Finset ℕ) (lacrosse : Finset ℕ) : Prop :=
  (total_seniors = 85) ∧
  (football.card = 74) ∧
  (baseball.card = 26) ∧
  ((football ∩ lacrosse).card = 17) ∧
  ((baseball ∩ football).card = 18) ∧
  ((baseball ∩ lacrosse).card = 13) ∧
  (lacrosse.card = 2 * (football ∩ baseball ∩ lacrosse).card) ∧
  (∀ s, s ∈ football ∪ baseball ∪ lacrosse) ∧
  ((football ∪ baseball ∪ lacrosse).card = total_seniors)

theorem senior_sports_solution 
  {total_seniors : ℕ} {football baseball lacrosse : Finset ℕ} 
  (h : senior_sports_problem total_seniors football baseball lacrosse) :
  (football ∩ baseball ∩ lacrosse).card = 11 := by
  sorry

end senior_sports_solution_l2116_211602


namespace volume_equivalence_l2116_211686

/-- A parallelepiped with congruent rhombic faces and a special vertex -/
structure RhombicParallelepiped where
  -- Side length of the rhombic face
  a : ℝ
  -- Angle between edges at the special vertex
  α : ℝ
  -- Diagonals of the rhombic face
  e : ℝ
  f : ℝ
  -- Conditions
  a_pos : 0 < a
  α_pos : 0 < α
  α_not_right : α ≠ π / 2
  α_less_120 : α < 2 * π / 3
  e_pos : 0 < e
  f_pos : 0 < f
  diag_relation : a = (1 / 2) * Real.sqrt (e^2 + f^2)
  angle_relation : Real.tan (α / 2) = f / e

/-- The volume of a rhombic parallelepiped -/
noncomputable def volume (p : RhombicParallelepiped) : ℝ :=
  p.a^3 * Real.sin p.α * Real.sqrt (Real.sin p.α^2 - Real.cos p.α^2 * Real.tan (p.α / 2)^2)

/-- The volume of a rhombic parallelepiped in terms of diagonals -/
noncomputable def volume_diag (p : RhombicParallelepiped) : ℝ :=
  (p.f / (8 * p.e)) * (p.e^2 + p.f^2) * Real.sqrt (3 * p.e^2 - p.f^2)

/-- The main theorem: equivalence of volume formulas -/
theorem volume_equivalence (p : RhombicParallelepiped) : volume p = volume_diag p := by
  sorry

end volume_equivalence_l2116_211686


namespace sum_of_zeros_is_eight_l2116_211699

/-- A function that is symmetric about x = 2 and has exactly four distinct zeros -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (2 + x) = f (2 - x)) ∧
  (∃! (z₁ z₂ z₃ z₄ : ℝ), z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    f z₁ = 0 ∧ f z₂ = 0 ∧ f z₃ = 0 ∧ f z₄ = 0)

/-- The theorem stating that the sum of zeros for a symmetric function with four distinct zeros is 8 -/
theorem sum_of_zeros_is_eight (f : ℝ → ℝ) (h : SymmetricFunction f) :
  ∃ z₁ z₂ z₃ z₄ : ℝ, z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    f z₁ = 0 ∧ f z₂ = 0 ∧ f z₃ = 0 ∧ f z₄ = 0 ∧
    z₁ + z₂ + z₃ + z₄ = 8 :=
by sorry

end sum_of_zeros_is_eight_l2116_211699


namespace expected_population_after_increase_l2116_211688

def current_population : ℝ := 1.75
def percentage_increase : ℝ := 325

theorem expected_population_after_increase :
  let increase_factor := 1 + percentage_increase / 100
  let expected_population := current_population * increase_factor
  expected_population = 7.4375 := by sorry

end expected_population_after_increase_l2116_211688


namespace inequality_proof_l2116_211641

theorem inequality_proof (a b c : ℝ) 
  (ha : 1 - a^2 ≥ 0) (hb : 1 - b^2 ≥ 0) (hc : 1 - c^2 ≥ 0) : 
  Real.sqrt (1 - a^2) + Real.sqrt (1 - b^2) + Real.sqrt (1 - c^2) ≤ 
  Real.sqrt (9 - (a + b + c)^2) := by
  sorry

end inequality_proof_l2116_211641


namespace room_space_is_400_l2116_211623

/-- The total space of a room with bookshelves and reserved space for desk and walking -/
def room_space (num_shelves : ℕ) (shelf_space desk_space : ℝ) : ℝ :=
  num_shelves * shelf_space + desk_space

/-- Theorem: The room space is 400 square feet -/
theorem room_space_is_400 :
  room_space 3 80 160 = 400 :=
by sorry

end room_space_is_400_l2116_211623


namespace total_jump_sequences_l2116_211628

-- Define a regular hexagon
structure RegularHexagon :=
  (vertices : Fin 6 → Point)

-- Define a frog's jump
inductive Jump
| clockwise
| counterclockwise

-- Define a sequence of jumps
def JumpSequence := List Jump

-- Define the result of a jump sequence
inductive JumpResult
| reachedD
| notReachedD

-- Function to determine the result of a jump sequence
def jumpSequenceResult (h : RegularHexagon) (js : JumpSequence) : JumpResult :=
  sorry

-- Function to count valid jump sequences
def countValidJumpSequences (h : RegularHexagon) : Nat :=
  sorry

-- The main theorem
theorem total_jump_sequences (h : RegularHexagon) :
  countValidJumpSequences h = 26 :=
sorry

end total_jump_sequences_l2116_211628


namespace intersection_condition_l2116_211695

theorem intersection_condition (a : ℝ) : 
  (∃ x y : ℝ, ax + 2*y = 3 ∧ x + (a-1)*y = 1) → a ≠ 2 ∧ 
  ¬(a ≠ 2 → ∃ x y : ℝ, ax + 2*y = 3 ∧ x + (a-1)*y = 1) :=
by sorry

end intersection_condition_l2116_211695


namespace tangent_line_equation_l2116_211672

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 + x - 8

-- Define the point of tangency
def point : ℝ × ℝ := (1, -6)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (HasDerivAt f (m * x - y + b) point.1) ∧
    (f point.1 = point.2) ∧
    (m = 4 ∧ b = -10) := by
  sorry

end tangent_line_equation_l2116_211672


namespace circle_and_line_equations_l2116_211670

-- Define the circles and points
def circle_O (x y : ℝ) := x^2 + y^2 = 16
def circle_C (x y : ℝ) := (x + 1)^2 + (y + 1)^2 = 2
def point_P : ℝ × ℝ := (-4, 0)

-- Define the line l
def line_l (x y : ℝ) := (3 * x - y = 0) ∨ (3 * x - y + 4 = 0)

-- Define the conditions
def condition_P_on_O := circle_O point_P.1 point_P.2

-- Theorem statement
theorem circle_and_line_equations :
  ∃ (A B M N : ℝ × ℝ),
    -- Line l intersects circle O at A and B
    circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    -- Line l intersects circle C at M and N
    circle_C M.1 M.2 ∧ circle_C N.1 N.2 ∧
    line_l M.1 M.2 ∧ line_l N.1 N.2 ∧
    -- M is the midpoint of AB
    M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2 ∧
    -- |PM| = |PN|
    (M.1 - point_P.1)^2 + (M.2 - point_P.2)^2 =
    (N.1 - point_P.1)^2 + (N.2 - point_P.2)^2 ∧
    -- Point P is on circle O
    condition_P_on_O :=
  by sorry

end circle_and_line_equations_l2116_211670


namespace total_marbles_l2116_211610

/-- Given a bag of marbles with only red, blue, and yellow marbles, where the ratio of
    red:blue:yellow is 2:3:4, and there are 24 blue marbles, prove that the total number
    of marbles is 72. -/
theorem total_marbles (red blue yellow total : ℕ) : 
  red + blue + yellow = total →
  red = 2 * n ∧ blue = 3 * n ∧ yellow = 4 * n →
  blue = 24 →
  total = 72 := by
  sorry

end total_marbles_l2116_211610


namespace first_day_is_saturday_l2116_211647

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a month -/
structure MonthDay where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Function to get the day of the week for a given day number -/
def getDayOfWeek (dayNumber : Nat) : DayOfWeek := sorry

/-- Theorem stating that if the 25th is a Tuesday, the 1st is a Saturday -/
theorem first_day_is_saturday 
  (h : getDayOfWeek 25 = DayOfWeek.Tuesday) : 
  getDayOfWeek 1 = DayOfWeek.Saturday := by
  sorry

end first_day_is_saturday_l2116_211647


namespace range_of_m_l2116_211652

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3)
  (hineq : ∀ m : ℝ, (4 / (x + 1)) + (16 / y) > m^2 - 3*m + 11) :
  ∀ m : ℝ, (1 < m ∧ m < 2) ↔ (4 / (x + 1)) + (16 / y) > m^2 - 3*m + 11 :=
by sorry

end range_of_m_l2116_211652


namespace undefined_condition_l2116_211667

theorem undefined_condition (y : ℝ) : 
  ¬(∃ x : ℝ, x = (3 * y^3 + 5) / (y^2 - 18*y + 81)) ↔ y = 9 := by
  sorry

end undefined_condition_l2116_211667


namespace composite_and_three_factors_l2116_211681

theorem composite_and_three_factors (n : ℕ) (h : n > 10) :
  let N := n^4 - 90*n^2 - 91*n - 90
  (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N = a * b) ∧
  (∃ (x y z : ℕ), x > 1 ∧ y > 1 ∧ z > 1 ∧ N = x * y * z) :=
by sorry

end composite_and_three_factors_l2116_211681


namespace infinite_solutions_cube_equation_l2116_211689

theorem infinite_solutions_cube_equation :
  ∃ f : ℕ → ℕ × ℕ × ℕ × ℕ, ∀ m : ℕ,
    let (n, x, y, z) := f m
    n > m ∧ (x + y + z)^3 = n^2 * x * y * z :=
by sorry

end infinite_solutions_cube_equation_l2116_211689


namespace fraction_of_108_l2116_211605

theorem fraction_of_108 : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 108 = 3 := by
  sorry

end fraction_of_108_l2116_211605


namespace hundredth_digit_of_seven_twentysixths_l2116_211676

/-- The fraction we're working with -/
def f : ℚ := 7/26

/-- The length of the repeating sequence in the decimal representation of f -/
def repeat_length : ℕ := 9

/-- The repeating sequence in the decimal representation of f -/
def repeat_sequence : List ℕ := [2, 6, 9, 2, 3, 0, 7, 6, 9]

/-- The position we're interested in -/
def target_position : ℕ := 100

theorem hundredth_digit_of_seven_twentysixths (h1 : f = 7/26)
  (h2 : repeat_length = 9)
  (h3 : repeat_sequence = [2, 6, 9, 2, 3, 0, 7, 6, 9])
  (h4 : target_position = 100) :
  repeat_sequence[(target_position - 1) % repeat_length] = 2 := by
  sorry

end hundredth_digit_of_seven_twentysixths_l2116_211676


namespace quadratic_function_theorem_l2116_211631

/-- A quadratic function passing through given points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  point_1 : a * (-3)^2 + b * (-3) + c = 0
  point_2 : a * (-2)^2 + b * (-2) + c = -3
  point_3 : a * (-1)^2 + b * (-1) + c = -4
  point_4 : c = -3

/-- Statements about the quadratic function -/
def statements (f : QuadraticFunction) : Fin 4 → Prop
  | 0 => f.a * f.c < 0
  | 1 => ∀ x > 1, ∀ y > x, f.a * y^2 + f.b * y + f.c > f.a * x^2 + f.b * x + f.c
  | 2 => f.a * (-4)^2 + (f.b - 4) * (-4) + f.c = 0
  | 3 => ∀ x, -1 < x → x < 0 → f.a * x^2 + (f.b - 1) * x + f.c + 3 > 0

/-- The main theorem -/
theorem quadratic_function_theorem (f : QuadraticFunction) :
  ∃ (S : Finset (Fin 4)), S.card = 2 ∧ (∀ i, i ∈ S ↔ statements f i) :=
sorry

end quadratic_function_theorem_l2116_211631


namespace instantaneous_velocity_at_one_l2116_211612

/-- The motion equation of an object -/
def s (t : ℝ) : ℝ := 7 * t^2 - 13 * t + 8

/-- The instantaneous velocity (derivative of s with respect to t) -/
def v (t : ℝ) : ℝ := 14 * t - 13

/-- Theorem: If the instantaneous velocity at t₀ is 1, then t₀ = 1 -/
theorem instantaneous_velocity_at_one (t₀ : ℝ) : v t₀ = 1 → t₀ = 1 := by
  sorry

end instantaneous_velocity_at_one_l2116_211612


namespace circle_area_relation_l2116_211643

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circle_area_relation :
  ∀ (r_A r_B : ℝ),
  circle_area r_A = 9 →
  r_A = r_B / 2 →
  circle_area r_B = 36 := by
sorry

end circle_area_relation_l2116_211643


namespace binomial_expansion_coefficient_l2116_211698

theorem binomial_expansion_coefficient (x : ℝ) : 
  ∃ (c : ℕ), c = 45 ∧ 
  (∃ (terms : ℕ → ℝ), 
    (∀ r, terms r = (Nat.choose 10 r) * (-1)^r * x^(5 - 3*r/2)) ∧
    (∃ r, 5 - 3*r/2 = 2 ∧ terms r = c * x^2)) := by
  sorry

end binomial_expansion_coefficient_l2116_211698


namespace negation_of_universal_proposition_l2116_211625

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l2116_211625


namespace mean_variance_preserved_l2116_211627

def initial_set : List Int := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
def new_set : List Int := [-5, -5, -3, -2, -1, 0, 1, 1, 2, 3, 4, 5]

def mean (s : List Int) : ℚ :=
  (s.sum : ℚ) / s.length

def variance (s : List Int) : ℚ :=
  let m := mean s
  (s.map (fun x => ((x : ℚ) - m) ^ 2)).sum / s.length

theorem mean_variance_preserved :
  mean initial_set = mean new_set ∧
  variance initial_set = variance new_set := by
  sorry

#eval mean initial_set
#eval mean new_set
#eval variance initial_set
#eval variance new_set

end mean_variance_preserved_l2116_211627


namespace livestock_theorem_l2116_211626

/-- Represents the value of livestock in taels of silver -/
structure LivestockValue where
  cow : ℕ
  sheep : ℕ
  total : ℕ

/-- Represents a purchase of livestock -/
structure Purchase where
  cows : ℕ
  sheep : ℕ

/-- The main theorem about livestock values and purchases -/
theorem livestock_theorem 
  (eq1 : LivestockValue)
  (eq2 : LivestockValue)
  (h1 : eq1.cow = 5 ∧ eq1.sheep = 2 ∧ eq1.total = 19)
  (h2 : eq2.cow = 2 ∧ eq2.sheep = 5 ∧ eq2.total = 16) :
  (∃ (cow_value sheep_value : ℕ),
    cow_value = 3 ∧ sheep_value = 2 ∧
    eq1.cow * cow_value + eq1.sheep * sheep_value = eq1.total ∧
    eq2.cow * cow_value + eq2.sheep * sheep_value = eq2.total) ∧
  (∃ (purchases : List Purchase),
    purchases.length = 3 ∧
    purchases.all (λ p => p.cows > 0 ∧ p.sheep > 0 ∧ p.cows * 3 + p.sheep * 2 = 20) ∧
    ∀ p : Purchase, p.cows > 0 → p.sheep > 0 → p.cows * 3 + p.sheep * 2 = 20 → p ∈ purchases) :=
by sorry

end livestock_theorem_l2116_211626


namespace G_equals_3F_l2116_211679

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := Real.log ((1 + (3 * x + x^3) / (1 + 3 * x^2)) / (1 - (3 * x + x^3) / (1 + 3 * x^2)))

theorem G_equals_3F : ∀ x : ℝ, x ≠ 1 → x ≠ -1 → G x = 3 * F x := by sorry

end G_equals_3F_l2116_211679


namespace coefficient_x_squared_in_expansion_l2116_211658

theorem coefficient_x_squared_in_expansion (x : ℝ) :
  ∃ c, (x + 2)^4 = c * x^2 + (x + 2)^4 - c * x^2 ∧ c = 24 := by
  sorry

end coefficient_x_squared_in_expansion_l2116_211658


namespace greatest_multiple_of_5_and_7_under_1000_l2116_211678

theorem greatest_multiple_of_5_and_7_under_1000 : ∃ n : ℕ, n = 980 ∧ 
  (∀ m : ℕ, m < 1000 ∧ 5 ∣ m ∧ 7 ∣ m → m ≤ n) := by
  sorry

end greatest_multiple_of_5_and_7_under_1000_l2116_211678


namespace smallest_integer_S_n_l2116_211601

def K' : ℚ := 137 / 60

def S (n : ℕ) : ℚ := n * 5^(n-1) * K' + 1

theorem smallest_integer_S_n : 
  (∀ m : ℕ, m > 0 → m < 12 → ¬(S m).isInt) ∧ (S 12).isInt :=
sorry

end smallest_integer_S_n_l2116_211601


namespace max_perimeter_l2116_211637

-- Define the triangle sides
def side1 : ℝ := 7
def side2 : ℝ := 8

-- Define the third side as a natural number
def x : ℕ → ℝ := λ n => n

-- Define the triangle inequality
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- The theorem to prove
theorem max_perimeter :
  ∃ n : ℕ, (is_valid_triangle side1 side2 (x n)) ∧
  (∀ m : ℕ, is_valid_triangle side1 side2 (x m) →
    perimeter side1 side2 (x n) ≥ perimeter side1 side2 (x m)) ∧
  perimeter side1 side2 (x n) = 29 :=
sorry

end max_perimeter_l2116_211637


namespace homework_time_difference_l2116_211669

/-- Given the conditions of the homework problem, prove that Greg has 6 hours less than Jacob. -/
theorem homework_time_difference :
  ∀ (greg_hours patrick_hours jacob_hours : ℕ),
  patrick_hours = 2 * greg_hours - 4 →
  jacob_hours = 18 →
  patrick_hours + greg_hours + jacob_hours = 50 →
  jacob_hours - greg_hours = 6 := by
sorry

end homework_time_difference_l2116_211669


namespace binomial_coefficient_26_6_l2116_211677

theorem binomial_coefficient_26_6 
  (h1 : Nat.choose 24 5 = 42504)
  (h2 : Nat.choose 25 5 = 53130)
  (h3 : Nat.choose 25 6 = 177100) :
  Nat.choose 26 6 = 230230 := by
sorry

end binomial_coefficient_26_6_l2116_211677


namespace trigonometric_equality_l2116_211600

theorem trigonometric_equality : 
  Real.sqrt (1 - 2 * Real.cos (π / 2 + 3) * Real.sin (π / 2 - 3)) = Real.sin 3 + Real.cos 3 := by
  sorry

end trigonometric_equality_l2116_211600


namespace expression_value_l2116_211639

theorem expression_value : (19 + 43 / 151) * 151 = 2910 := by
  sorry

end expression_value_l2116_211639


namespace inequality_implication_l2116_211665

theorem inequality_implication (a b c : ℝ) : a > b → a * c^2 ≥ b * c^2 := by
  sorry

end inequality_implication_l2116_211665


namespace factorization_left_to_right_l2116_211629

theorem factorization_left_to_right : 
  ∀ x : ℝ, x^2 - 1 = (x + 1) * (x - 1) :=
by sorry

end factorization_left_to_right_l2116_211629


namespace flower_bee_difference_l2116_211636

def number_of_flowers : ℕ := 5
def number_of_bees : ℕ := 3

theorem flower_bee_difference : 
  number_of_flowers - number_of_bees = 2 := by sorry

end flower_bee_difference_l2116_211636


namespace shortest_side_of_special_triangle_l2116_211694

theorem shortest_side_of_special_triangle : 
  ∀ (a b c : ℕ), 
    a = 18 →
    a + b + c = 42 →
    (∃ A : ℕ, A^2 = (21 * (21 - a) * (21 - b) * (21 - c))) →
    a ≤ b ∧ a ≤ c →
    b + c > a →
    a + c > b →
    a + b > c →
    b ≥ 5 ∧ c ≥ 5 :=
by sorry

#check shortest_side_of_special_triangle

end shortest_side_of_special_triangle_l2116_211694


namespace divisor_property_l2116_211653

theorem divisor_property (k : ℕ) : 18^k ∣ 624938 → 6^k - k^6 = 1 := by
  sorry

end divisor_property_l2116_211653


namespace line_through_points_sum_of_slope_and_intercept_l2116_211616

/-- Given a line passing through points (1,2) and (4,11) with equation y = mx + b, prove that m + b = 2 -/
theorem line_through_points_sum_of_slope_and_intercept :
  ∀ (m b : ℝ),
  (2 : ℝ) = m * 1 + b →  -- Point (1,2) satisfies the equation
  (11 : ℝ) = m * 4 + b →  -- Point (4,11) satisfies the equation
  m + b = 2 := by
  sorry

end line_through_points_sum_of_slope_and_intercept_l2116_211616


namespace sum_of_constants_l2116_211604

/-- Given a function y = a + b/x, prove that a + b = 11 under specific conditions -/
theorem sum_of_constants (a b : ℝ) : 
  (2 = a + b/(-2)) → 
  (6 = a + b/(-6)) → 
  (10 = a + b/(-3)) → 
  a + b = 11 := by
sorry

end sum_of_constants_l2116_211604


namespace triangle_sides_l2116_211697

-- Define the rhombus side length
def rhombus_side : ℝ := 6

-- Define the triangle
structure Triangle where
  a : ℝ  -- shortest side
  b : ℝ  -- middle side
  c : ℝ  -- hypotenuse

-- Define the properties of the triangle and rhombus
def triangle_with_inscribed_rhombus (t : Triangle) : Prop :=
  -- The triangle is right-angled with a 60° angle
  t.a^2 + t.b^2 = t.c^2 ∧
  t.a / t.c = 1 / 2 ∧
  -- The rhombus is inscribed in the triangle
  -- (We don't need to explicitly state this as it's implied by the problem setup)
  -- The rhombus side length is 6
  rhombus_side = 6

-- The theorem to prove
theorem triangle_sides (t : Triangle) 
  (h : triangle_with_inscribed_rhombus t) : 
  t.a = 9 ∧ t.b = 9 * Real.sqrt 3 ∧ t.c = 18 := by
  sorry

end triangle_sides_l2116_211697


namespace marla_errand_time_l2116_211674

/-- Calculates the total time Marla spends on her errand to her son's school -/
def total_errand_time (one_way_drive_time parent_teacher_time : ℕ) : ℕ :=
  2 * one_way_drive_time + parent_teacher_time

/-- Proves that Marla spends 110 minutes on her errand -/
theorem marla_errand_time :
  total_errand_time 20 70 = 110 := by
  sorry

end marla_errand_time_l2116_211674


namespace fraction_addition_l2116_211606

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l2116_211606


namespace probability_of_blue_ball_l2116_211659

theorem probability_of_blue_ball (p_red p_yellow p_blue : ℝ) : 
  p_red = 0.48 → p_yellow = 0.35 → p_red + p_yellow + p_blue = 1 → p_blue = 0.17 := by
  sorry

end probability_of_blue_ball_l2116_211659


namespace ram_ravi_selection_probability_l2116_211609

theorem ram_ravi_selection_probability 
  (p_ram : ℝ) 
  (p_both : ℝ) 
  (h1 : p_ram = 1/7)
  (h2 : p_both = 0.02857142857142857) :
  ∃ (p_ravi : ℝ), p_ravi = 0.2 ∧ p_both = p_ram * p_ravi :=
by sorry

end ram_ravi_selection_probability_l2116_211609


namespace average_not_equal_given_l2116_211693

def numbers : List ℝ := [1200, 1300, 1400, 1520, 1530, 1200]

def given_average : ℝ := 1380

theorem average_not_equal_given : (numbers.sum / numbers.length) ≠ given_average := by
  sorry

end average_not_equal_given_l2116_211693


namespace distance_is_60km_l2116_211614

/-- A ship's journey relative to a lighthouse -/
structure ShipJourney where
  speed : ℝ
  time : ℝ
  initial_angle : ℝ
  final_angle : ℝ

/-- Calculate the distance between the ship and the lighthouse at the end of the journey -/
def distance_to_lighthouse (journey : ShipJourney) : ℝ :=
  sorry

/-- Theorem stating that for the given journey parameters, the distance to the lighthouse is 60 km -/
theorem distance_is_60km (journey : ShipJourney) 
  (h1 : journey.speed = 15)
  (h2 : journey.time = 4)
  (h3 : journey.initial_angle = π / 3)
  (h4 : journey.final_angle = π / 6) :
  distance_to_lighthouse journey = 60 := by
  sorry

end distance_is_60km_l2116_211614


namespace second_group_count_l2116_211618

theorem second_group_count (total : ℕ) (avg : ℚ) (sum_three : ℕ) (avg_others : ℚ) :
  total = 5 ∧ 
  avg = 20 ∧ 
  sum_three = 48 ∧ 
  avg_others = 26 →
  (total - 3 : ℕ) = 2 :=
by sorry

end second_group_count_l2116_211618


namespace plate_tower_problem_l2116_211692

theorem plate_tower_problem (initial_plates : ℕ) (first_addition : ℕ) (common_difference : ℕ) (total_plates : ℕ) :
  initial_plates = 27 →
  first_addition = 12 →
  common_difference = 3 →
  total_plates = 123 →
  ∃ (n : ℕ) (last_addition : ℕ),
    n = 4 ∧
    last_addition = 21 ∧
    total_plates = initial_plates + n * (2 * first_addition + (n - 1) * common_difference) / 2 :=
by sorry

end plate_tower_problem_l2116_211692


namespace john_video_release_l2116_211684

/-- The number of videos John releases per day -/
def videos_per_day : ℕ := 3

/-- The length of a short video in minutes -/
def short_video_length : ℕ := 2

/-- The number of short videos released per day -/
def short_videos_per_day : ℕ := 2

/-- The factor by which the long video is longer than the short videos -/
def long_video_factor : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Calculates the total minutes of video John releases per week -/
def total_minutes_per_week : ℕ :=
  days_per_week * (
    short_videos_per_day * short_video_length +
    (videos_per_day - short_videos_per_day) * (long_video_factor * short_video_length)
  )

theorem john_video_release :
  total_minutes_per_week = 112 := by
  sorry

end john_video_release_l2116_211684


namespace cell_growth_problem_l2116_211608

/-- Calculates the number of cells after a given number of days, 
    given an initial population and growth rate every two days. -/
def cell_population (initial_cells : ℕ) (growth_rate : ℕ) (days : ℕ) : ℕ :=
  initial_cells * growth_rate ^ (days / 2)

/-- Theorem stating that given the specific conditions of the problem,
    the cell population after 10 days is 1215. -/
theorem cell_growth_problem : cell_population 5 3 10 = 1215 := by
  sorry


end cell_growth_problem_l2116_211608


namespace train_length_l2116_211607

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 180 → time = 9 → speed * time * (1000 / 3600) = 450 := by
  sorry

end train_length_l2116_211607


namespace school_students_count_l2116_211635

/-- The number of students who play football -/
def football : ℕ := 325

/-- The number of students who play cricket -/
def cricket : ℕ := 175

/-- The number of students who play neither football nor cricket -/
def neither : ℕ := 50

/-- The number of students who play both football and cricket -/
def both : ℕ := 140

/-- The total number of students in the school -/
def total_students : ℕ := football + cricket - both + neither

theorem school_students_count :
  total_students = 410 := by sorry

end school_students_count_l2116_211635


namespace binary_253_ones_minus_zeros_l2116_211620

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Count the number of true values in a list of booleans -/
def countOnes (l : List Bool) : ℕ :=
  l.filter id |>.length

/-- Count the number of false values in a list of booleans -/
def countZeros (l : List Bool) : ℕ :=
  l.filter not |>.length

theorem binary_253_ones_minus_zeros :
  let binary := toBinary 253
  let y := countOnes binary
  let x := countZeros binary
  y - x = 6 := by sorry

end binary_253_ones_minus_zeros_l2116_211620


namespace polynomial_factorization_l2116_211646

theorem polynomial_factorization (x : ℝ) :
  4 * (x + 5) * (x + 6) * (x + 10) * (x + 12) - 3 * x^2 =
  (2 * x^2 + 35 * x + 120) * (x + 8) * (2 * x + 15) := by
  sorry

end polynomial_factorization_l2116_211646
