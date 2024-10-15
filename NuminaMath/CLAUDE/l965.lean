import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l965_96557

theorem quadratic_inequality_solution_set :
  {x : ℝ | x * (x - 3) < 0} = {x : ℝ | 0 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l965_96557


namespace NUMINAMATH_CALUDE_rowing_speeds_calculation_l965_96594

/-- Represents the rowing speeds and wind effects for a man rowing a boat -/
structure RowingScenario where
  withStreamSpeed : ℝ
  againstStreamSpeed : ℝ
  windSpeedDownstream : ℝ
  windReductionAgainstStream : ℝ
  windIncreaseWithStream : ℝ

/-- Calculates the effective rowing speeds given a RowingScenario -/
def effectiveRowingSpeeds (scenario : RowingScenario) : ℝ × ℝ :=
  let effectiveAgainstStream := scenario.againstStreamSpeed * (1 - scenario.windReductionAgainstStream)
  let effectiveWithStream := scenario.withStreamSpeed * (1 + scenario.windIncreaseWithStream)
  (effectiveWithStream, effectiveAgainstStream)

/-- Theorem stating the effective rowing speeds for the given scenario -/
theorem rowing_speeds_calculation (scenario : RowingScenario) 
    (h1 : scenario.withStreamSpeed = 8)
    (h2 : scenario.againstStreamSpeed = 4)
    (h3 : scenario.windSpeedDownstream = 2)
    (h4 : scenario.windReductionAgainstStream = 0.2)
    (h5 : scenario.windIncreaseWithStream = 0.1) :
    effectiveRowingSpeeds scenario = (8.8, 3.2) := by
  sorry

#eval effectiveRowingSpeeds { 
  withStreamSpeed := 8, 
  againstStreamSpeed := 4, 
  windSpeedDownstream := 2, 
  windReductionAgainstStream := 0.2, 
  windIncreaseWithStream := 0.1 
}

end NUMINAMATH_CALUDE_rowing_speeds_calculation_l965_96594


namespace NUMINAMATH_CALUDE_unique_valid_number_l965_96572

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ 
  (n / 100 % 10 > 6) ∧ (n / 10 % 10 > 6) ∧ (n % 10 > 6) ∧
  n % 12 = 0

theorem unique_valid_number : ∃! n, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l965_96572


namespace NUMINAMATH_CALUDE_product_of_fractions_l965_96558

theorem product_of_fractions : (2 : ℚ) / 5 * (3 : ℚ) / 4 = (3 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l965_96558


namespace NUMINAMATH_CALUDE_animal_costs_l965_96566

theorem animal_costs (dog_cost cow_cost horse_cost : ℚ) : 
  cow_cost = 4 * dog_cost →
  horse_cost = 4 * cow_cost →
  dog_cost + 2 * cow_cost + horse_cost = 200 →
  dog_cost = 8 ∧ cow_cost = 32 ∧ horse_cost = 128 := by
sorry

end NUMINAMATH_CALUDE_animal_costs_l965_96566


namespace NUMINAMATH_CALUDE_trigonometric_identity_l965_96596

theorem trigonometric_identity (α : Real) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * (Real.cos (π / 6 + α / 2))^2 - 1 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l965_96596


namespace NUMINAMATH_CALUDE_decreasing_number_a312_max_decreasing_number_divisible_by_9_l965_96551

/-- A four-digit natural number with all digits different and not equal to 0 -/
structure DecreasingNumber :=
  (a b c d : ℕ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (c_pos : c > 0)
  (d_pos : d > 0)
  (all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (decreasing_property : 10 * a + b - (10 * b + c) = 10 * c + d)

theorem decreasing_number_a312 :
  ∃ (n : DecreasingNumber), n.a = 4 ∧ n.b = 3 ∧ n.c = 1 ∧ n.d = 2 :=
sorry

theorem max_decreasing_number_divisible_by_9 :
  ∃ (n : DecreasingNumber),
    (100 * n.a + 10 * n.b + n.c + 100 * n.b + 10 * n.c + n.d) % 9 = 0 ∧
    ∀ (m : DecreasingNumber),
      (100 * m.a + 10 * m.b + m.c + 100 * m.b + 10 * m.c + m.d) % 9 = 0 →
      1000 * n.a + 100 * n.b + 10 * n.c + n.d ≥ 1000 * m.a + 100 * m.b + 10 * m.c + m.d ∧
    n.a = 8 ∧ n.b = 1 ∧ n.c = 6 ∧ n.d = 5 :=
sorry

end NUMINAMATH_CALUDE_decreasing_number_a312_max_decreasing_number_divisible_by_9_l965_96551


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_exists_l965_96553

theorem geometric_arithmetic_sequence_exists : ∃ (a b : ℝ),
  1 < a ∧ a < b ∧ b < 16 ∧
  (∃ (r : ℝ), a = 1 * r ∧ b = 1 * r^2) ∧
  (∃ (d : ℝ), b = a + d ∧ 16 = b + d) ∧
  a + b = 12.64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_exists_l965_96553


namespace NUMINAMATH_CALUDE_function_determination_l965_96523

/-- Given two functions f and g with specific forms and conditions, prove they have specific expressions. -/
theorem function_determination (a b c : ℝ) : 
  let f := fun (x : ℝ) ↦ 2 * x^3 + a * x
  let g := fun (x : ℝ) ↦ b * x^2 + c
  (f 2 = 0) → 
  (g 2 = 0) → 
  (deriv f 2 = deriv g 2) →
  (f = fun (x : ℝ) ↦ 2 * x^3 - 8 * x) ∧ 
  (g = fun (x : ℝ) ↦ 4 * x^2 - 16) :=
by sorry

end NUMINAMATH_CALUDE_function_determination_l965_96523


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_4_and_5_l965_96564

theorem largest_three_digit_multiple_of_4_and_5 : 
  ∀ n : ℕ, n ≤ 999 ∧ n ≥ 100 ∧ 4 ∣ n ∧ 5 ∣ n → n ≤ 980 :=
by
  sorry

#check largest_three_digit_multiple_of_4_and_5

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_4_and_5_l965_96564


namespace NUMINAMATH_CALUDE_lottery_profit_l965_96597

-- Define the card colors
inductive Color
| Black
| Red

-- Define the card values
inductive Value
| One
| Two
| Three
| Four

-- Define a card as a pair of color and value
structure Card where
  color : Color
  value : Value

-- Define the set of cards
def cards : Finset Card := sorry

-- Define the categories
inductive Category
| A  -- Flush
| B  -- Same color
| C  -- Straight
| D  -- Pair
| E  -- Others

-- Function to determine the category of a pair of cards
def categorize : Card → Card → Category := sorry

-- Function to calculate the probability of a category
def probability (c : Category) : Rat := sorry

-- Define the prize values
def prizeValue : Category → Nat
| Category.D => 9  -- First prize
| Category.B => 3  -- Second prize
| _ => 1           -- Third prize

-- Number of participants
def participants : Nat := 300

-- Theorem to prove
theorem lottery_profit :
  (∀ c : Category, c ≠ Category.D → probability Category.D ≤ probability c) ∧
  (∀ c : Category, c ≠ Category.B → probability c ≤ probability Category.B) ∧
  (participants * 3 - (participants * probability Category.D * prizeValue Category.D +
                       participants * probability Category.B * prizeValue Category.B +
                       participants * (1 - probability Category.D - probability Category.B) * 1) = 120) := by
  sorry

end NUMINAMATH_CALUDE_lottery_profit_l965_96597


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l965_96583

theorem quadrilateral_diagonal_length 
  (area : ℝ) 
  (offset1 : ℝ) 
  (offset2 : ℝ) 
  (h1 : area = 300) 
  (h2 : offset1 = 9) 
  (h3 : offset2 = 6) : 
  area = (1/2) * (offset1 + offset2) * 40 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l965_96583


namespace NUMINAMATH_CALUDE_truck_travel_distance_truck_specific_distance_l965_96561

/-- Represents the distance a truck can travel given an amount of gas -/
def distance_traveled (miles_per_gallon : ℝ) (gallons : ℝ) : ℝ :=
  miles_per_gallon * gallons

theorem truck_travel_distance 
  (initial_distance : ℝ) 
  (initial_gas : ℝ) 
  (new_gas : ℝ) : 
  initial_distance > 0 → 
  initial_gas > 0 → 
  new_gas > 0 → 
  distance_traveled (initial_distance / initial_gas) new_gas = 
    (initial_distance / initial_gas) * new_gas := by
  sorry

/-- Proves that a truck traveling 240 miles on 10 gallons of gas can travel 360 miles on 15 gallons of gas -/
theorem truck_specific_distance : 
  distance_traveled (240 / 10) 15 = 360 := by
  sorry

end NUMINAMATH_CALUDE_truck_travel_distance_truck_specific_distance_l965_96561


namespace NUMINAMATH_CALUDE_symmetric_line_proof_l965_96576

/-- Given two lines in a 2D plane, this function returns the equation of the line symmetric to the first line with respect to the second line. -/
def symmetricLine (l1 l2 : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  sorry

/-- The line x - y - 2 = 0 -/
def line1 : ℝ → ℝ → Prop :=
  λ x y ↦ x - y - 2 = 0

/-- The line x - 2y + 2 = 0 -/
def line2 : ℝ → ℝ → Prop :=
  λ x y ↦ x - 2*y + 2 = 0

/-- The line x - 7y + 22 = 0 -/
def resultLine : ℝ → ℝ → Prop :=
  λ x y ↦ x - 7*y + 22 = 0

theorem symmetric_line_proof : 
  symmetricLine line1 line2 = resultLine := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_proof_l965_96576


namespace NUMINAMATH_CALUDE_parabola_theorem_l965_96555

/-- Represents a parabola in the form y = x^2 + bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- The parabola passes through the point (2, 0) -/
def passes_through_A (p : Parabola) : Prop :=
  4 + 2 * p.b + p.c = 0

/-- The parabola passes through the point (0, 6) -/
def passes_through_B (p : Parabola) : Prop :=
  p.c = 6

/-- The parabola equation is y = x^2 - 5x + 6 -/
def is_correct_equation (p : Parabola) : Prop :=
  p.b = -5 ∧ p.c = 6

/-- The y-coordinate of the point (4, 0) on the parabola -/
def y_at_x_4 (p : Parabola) : ℝ :=
  16 - 5 * 4 + p.c

/-- The downward shift required for the parabola to pass through (4, 0) -/
def downward_shift (p : Parabola) : ℝ :=
  y_at_x_4 p

theorem parabola_theorem (p : Parabola) 
  (h1 : passes_through_A p) (h2 : passes_through_B p) : 
  is_correct_equation p ∧ downward_shift p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_theorem_l965_96555


namespace NUMINAMATH_CALUDE_initial_seashells_count_l965_96552

/-- The number of seashells Jason found initially -/
def initial_seashells : ℕ := sorry

/-- The number of starfish Jason found -/
def starfish : ℕ := 48

/-- The number of seashells Jason gave to Tim -/
def seashells_given_away : ℕ := 13

/-- The number of seashells Jason has now -/
def current_seashells : ℕ := 36

/-- Theorem stating that the initial number of seashells is equal to the current number plus the number given away -/
theorem initial_seashells_count : initial_seashells = current_seashells + seashells_given_away := by
  sorry

end NUMINAMATH_CALUDE_initial_seashells_count_l965_96552


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l965_96579

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_condition : 2*a + 2*b + 2*c = 3) :
  (1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a)) ≥ 2 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ 
    2*a₀ + 2*b₀ + 2*c₀ = 3 ∧
    1 / (2*a₀ + b₀) + 1 / (2*b₀ + c₀) + 1 / (2*c₀ + a₀) = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l965_96579


namespace NUMINAMATH_CALUDE_four_integer_sum_problem_l965_96506

theorem four_integer_sum_problem (a b c d : ℕ+) 
  (h_order : a < b ∧ b < c ∧ c < d)
  (h_sums_different : a + b ≠ a + c ∧ a + b ≠ a + d ∧ a + b ≠ b + c ∧ 
                      a + b ≠ b + d ∧ a + b ≠ c + d ∧ a + c ≠ a + d ∧ 
                      a + c ≠ b + c ∧ a + c ≠ b + d ∧ a + c ≠ c + d ∧ 
                      a + d ≠ b + c ∧ a + d ≠ b + d ∧ a + d ≠ c + d ∧ 
                      b + c ≠ b + d ∧ b + c ≠ c + d ∧ b + d ≠ c + d)
  (h_smallest_sums : min (a + b) (min (a + c) (min (a + d) (min (b + c) (min (b + d) (c + d))))) = 6 ∧
                     min (a + c) (min (a + d) (min (b + c) (min (b + d) (c + d)))) = 8 ∧
                     min (a + d) (min (b + c) (min (b + d) (c + d))) = 12 ∧
                     min (b + c) (min (b + d) (c + d)) = 21) : 
  d = 20 := by
sorry

end NUMINAMATH_CALUDE_four_integer_sum_problem_l965_96506


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l965_96527

-- System 1
theorem system_one_solution :
  ∃ (x y : ℚ), 3 * x + 2 * y = 8 ∧ y = 2 * x - 3 ∧ x = 2 ∧ y = 1 := by sorry

-- System 2
theorem system_two_solution :
  ∃ (x y : ℚ), 2 * x + 3 * y = 6 ∧ 3 * x - 2 * y = -2 ∧ x = 6/13 ∧ y = 22/13 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l965_96527


namespace NUMINAMATH_CALUDE_mao_saying_moral_l965_96530

/-- Represents the moral of a saying -/
inductive Moral
| KnowledgeDrivesPractice
| KnowledgeGuidesPractice
| PracticeSourceOfKnowledge
| PracticeSocialHistorical

/-- Represents a philosophical saying -/
structure Saying :=
(content : String)
(moral : Moral)

/-- Mao Zedong's saying about tasting a pear -/
def maoSaying : Saying :=
{ content := "If you want to know the taste of a pear, you must change the pear and taste it yourself",
  moral := Moral.PracticeSourceOfKnowledge }

/-- Theorem stating that the moral of Mao's saying is "Practice is the source of knowledge" -/
theorem mao_saying_moral :
  maoSaying.moral = Moral.PracticeSourceOfKnowledge :=
sorry

end NUMINAMATH_CALUDE_mao_saying_moral_l965_96530


namespace NUMINAMATH_CALUDE_painted_cells_theorem_l965_96509

theorem painted_cells_theorem (k l : ℕ) : 
  k * l = 74 → 
  (((2 * k + 1) * (2 * l + 1) - 74 = 373) ∨ 
   ((2 * k + 1) * (2 * l + 1) - 74 = 301)) := by
  sorry

end NUMINAMATH_CALUDE_painted_cells_theorem_l965_96509


namespace NUMINAMATH_CALUDE_shelves_needed_l965_96587

theorem shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) 
  (h1 : total_books = 46)
  (h2 : books_taken = 10)
  (h3 : books_per_shelf = 4)
  (h4 : books_per_shelf > 0) :
  (total_books - books_taken) / books_per_shelf = 9 :=
by sorry

end NUMINAMATH_CALUDE_shelves_needed_l965_96587


namespace NUMINAMATH_CALUDE_probability_of_selecting_specific_car_type_l965_96567

theorem probability_of_selecting_specific_car_type 
  (total_car_types : ℕ) 
  (cars_selected : ℕ) 
  (h1 : total_car_types = 5) 
  (h2 : cars_selected = 2) :
  (cars_selected : ℚ) / (total_car_types.choose cars_selected) = 2/5 := by
sorry

end NUMINAMATH_CALUDE_probability_of_selecting_specific_car_type_l965_96567


namespace NUMINAMATH_CALUDE_silverware_probability_l965_96503

theorem silverware_probability (forks spoons knives : ℕ) 
  (h1 : forks = 4) (h2 : spoons = 8) (h3 : knives = 6) : 
  let total := forks + spoons + knives
  let ways_to_choose_3 := Nat.choose total 3
  let ways_to_choose_2_spoons := Nat.choose spoons 2
  let ways_to_choose_1_knife := Nat.choose knives 1
  let favorable_outcomes := ways_to_choose_2_spoons * ways_to_choose_1_knife
  (favorable_outcomes : ℚ) / ways_to_choose_3 = 7 / 34 := by
sorry

end NUMINAMATH_CALUDE_silverware_probability_l965_96503


namespace NUMINAMATH_CALUDE_probability_red_ball_two_fifths_l965_96577

/-- Represents a bag of colored balls -/
structure BallBag where
  red : ℕ
  black : ℕ

/-- Calculates the probability of drawing a red ball from the bag -/
def probabilityRedBall (bag : BallBag) : ℚ :=
  bag.red / (bag.red + bag.black)

/-- Theorem: The probability of drawing a red ball from a bag with 2 red balls and 3 black balls is 2/5 -/
theorem probability_red_ball_two_fifths :
  let bag : BallBag := { red := 2, black := 3 }
  probabilityRedBall bag = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_two_fifths_l965_96577


namespace NUMINAMATH_CALUDE_wrong_to_right_ratio_l965_96500

theorem wrong_to_right_ratio (total : ℕ) (correct : ℕ) 
  (h1 : total = 54) (h2 : correct = 18) :
  (total - correct) / correct = 2 := by
  sorry

end NUMINAMATH_CALUDE_wrong_to_right_ratio_l965_96500


namespace NUMINAMATH_CALUDE_chichikov_game_l965_96590

theorem chichikov_game (total_nuts : ℕ) (box1 box2 : ℕ) : total_nuts = 222 → box1 + box2 = total_nuts →
  ∃ N : ℕ, 1 ≤ N ∧ N ≤ 222 ∧
  (∀ move : ℕ, move < 37 →
    ¬(∃ new_box1 new_box2 new_box3 : ℕ,
      new_box1 + new_box2 + new_box3 = total_nuts ∧
      (new_box1 = N ∨ new_box2 = N ∨ new_box3 = N ∨ new_box1 + new_box2 = N ∨ new_box1 + new_box3 = N ∨ new_box2 + new_box3 = N) ∧
      new_box1 + new_box2 + move = box1 + box2)) ∧
  (∀ N : ℕ, 1 ≤ N ∧ N ≤ 222 →
    ∃ new_box1 new_box2 new_box3 : ℕ,
      new_box1 + new_box2 + new_box3 = total_nuts ∧
      (new_box1 = N ∨ new_box2 = N ∨ new_box3 = N ∨ new_box1 + new_box2 = N ∨ new_box1 + new_box3 = N ∨ new_box2 + new_box3 = N) ∧
      new_box1 + new_box2 + 37 ≥ box1 + box2) :=
by
  sorry

end NUMINAMATH_CALUDE_chichikov_game_l965_96590


namespace NUMINAMATH_CALUDE_outfits_count_l965_96517

/-- The number of outfits with different colored shirts and hats -/
def num_outfits (blue_shirts yellow_shirts pants blue_hats yellow_hats : ℕ) : ℕ :=
  blue_shirts * pants * yellow_hats + yellow_shirts * pants * blue_hats

/-- Theorem: The number of outfits is 756 given the specified numbers of clothing items -/
theorem outfits_count :
  num_outfits 6 6 7 9 9 = 756 :=
by sorry

end NUMINAMATH_CALUDE_outfits_count_l965_96517


namespace NUMINAMATH_CALUDE_lesser_fraction_l965_96570

theorem lesser_fraction (x y : ℝ) (h_sum : x + y = 10/11) (h_prod : x * y = 1/8) :
  min x y = (80 - 2 * Real.sqrt 632) / 176 := by sorry

end NUMINAMATH_CALUDE_lesser_fraction_l965_96570


namespace NUMINAMATH_CALUDE_third_pedal_similar_l965_96502

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents a point in a 2D plane -/
def Point := ℝ × ℝ

/-- Generates the pedal triangle of a point P with respect to a given triangle -/
def pedalTriangle (P : Point) (T : Triangle) : Triangle :=
  sorry

/-- Checks if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop :=
  sorry

/-- Theorem stating that the third pedal triangle is similar to the original triangle -/
theorem third_pedal_similar (P : Point) (H₀ : Triangle) :
  let H₁ := pedalTriangle P H₀
  let H₂ := pedalTriangle P H₁
  let H₃ := pedalTriangle P H₂
  areSimilar H₃ H₀ :=
by
  sorry

end NUMINAMATH_CALUDE_third_pedal_similar_l965_96502


namespace NUMINAMATH_CALUDE_prob_same_color_is_17_35_l965_96508

/-- A box containing chess pieces -/
structure ChessBox where
  total_pieces : ℕ
  black_pieces : ℕ
  white_pieces : ℕ
  prob_two_black : ℚ
  prob_two_white : ℚ

/-- The probability of drawing two pieces of the same color -/
def prob_same_color (box : ChessBox) : ℚ :=
  box.prob_two_black + box.prob_two_white

/-- Theorem stating the probability of drawing two pieces of the same color -/
theorem prob_same_color_is_17_35 (box : ChessBox)
  (h1 : box.total_pieces = 15)
  (h2 : box.black_pieces = 6)
  (h3 : box.white_pieces = 9)
  (h4 : box.prob_two_black = 1 / 7)
  (h5 : box.prob_two_white = 12 / 35) :
  prob_same_color box = 17 / 35 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_17_35_l965_96508


namespace NUMINAMATH_CALUDE_fraction_problem_l965_96575

theorem fraction_problem (x : ℝ) (f : ℝ) (h1 : x = 140) (h2 : 0.65 * x = f * x - 21) : f = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l965_96575


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l965_96518

/-- An arithmetic sequence with given second and third terms -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ a 2 = 2 ∧ a 3 = 4

/-- The 10th term of the arithmetic sequence is 18 -/
theorem arithmetic_sequence_10th_term (a : ℕ → ℝ) (h : arithmeticSequence a) : 
  a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l965_96518


namespace NUMINAMATH_CALUDE_smallest_c_for_g_range_contains_one_l965_96519

/-- The function g(x) defined as x^2 - 2x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + c

/-- Theorem stating that 2 is the smallest value of c such that 1 is in the range of g(x) -/
theorem smallest_c_for_g_range_contains_one :
  ∀ c : ℝ, (∃ x : ℝ, g c x = 1) ↔ c ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_g_range_contains_one_l965_96519


namespace NUMINAMATH_CALUDE_min_distance_to_line_l965_96588

/-- Given that 5x + 12y = 60, the minimum value of √(x² + y²) is 60/13 -/
theorem min_distance_to_line (x y : ℝ) (h : 5 * x + 12 * y = 60) :
  ∃ (min_val : ℝ), min_val = 60 / 13 ∧ 
  ∀ (x' y' : ℝ), 5 * x' + 12 * y' = 60 → Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_min_distance_to_line_l965_96588


namespace NUMINAMATH_CALUDE_point_difference_l965_96569

def zachScore : ℕ := 42
def benScore : ℕ := 21

theorem point_difference : zachScore - benScore = 21 := by
  sorry

end NUMINAMATH_CALUDE_point_difference_l965_96569


namespace NUMINAMATH_CALUDE_ladybug_dots_average_l965_96528

/-- The number of ladybugs caught on Monday -/
def monday_ladybugs : ℕ := 8

/-- The number of ladybugs caught on Tuesday -/
def tuesday_ladybugs : ℕ := 5

/-- The total number of dots on all ladybugs -/
def total_dots : ℕ := 78

/-- The average number of dots per ladybug -/
def average_dots : ℚ := total_dots / (monday_ladybugs + tuesday_ladybugs)

theorem ladybug_dots_average :
  average_dots = 6 := by sorry

end NUMINAMATH_CALUDE_ladybug_dots_average_l965_96528


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l965_96580

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem plywood_cut_perimeter_difference :
  let plywood := Rectangle.mk 12 6
  let area := plywood.length * plywood.width
  ∀ (piece : Rectangle),
    (6 * piece.length * piece.width = area) →
    (∃ (max_piece min_piece : Rectangle),
      (6 * max_piece.length * max_piece.width = area) ∧
      (6 * min_piece.length * min_piece.width = area) ∧
      (∀ (r : Rectangle), (6 * r.length * r.width = area) →
        perimeter r ≤ perimeter max_piece ∧
        perimeter r ≥ perimeter min_piece)) →
    (perimeter max_piece - perimeter min_piece = 14) := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l965_96580


namespace NUMINAMATH_CALUDE_product_expansion_l965_96593

theorem product_expansion (y : ℝ) : 4 * (y - 3) * (y + 2) = 4 * y^2 - 4 * y - 24 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l965_96593


namespace NUMINAMATH_CALUDE_chocolate_cost_l965_96531

theorem chocolate_cost (box_size : ℕ) (box_cost : ℚ) (total_candies : ℕ) : 
  box_size = 30 → 
  box_cost = 9 → 
  total_candies = 450 → 
  (total_candies / box_size : ℚ) * box_cost = 135 := by
sorry

end NUMINAMATH_CALUDE_chocolate_cost_l965_96531


namespace NUMINAMATH_CALUDE_dedekind_cut_property_l965_96589

-- Define a Dedekind cut
def DedekindCut (M N : Set ℚ) : Prop :=
  (M ∪ N = Set.univ) ∧ 
  (M ∩ N = ∅) ∧ 
  (∀ x ∈ M, ∀ y ∈ N, x < y) ∧
  M.Nonempty ∧ 
  N.Nonempty

-- Theorem stating the impossibility of M having a largest element and N having a smallest element
theorem dedekind_cut_property (M N : Set ℚ) (h : DedekindCut M N) :
  ¬(∃ (m : ℚ), m ∈ M ∧ ∀ x ∈ M, x ≤ m) ∨ ¬(∃ (n : ℚ), n ∈ N ∧ ∀ y ∈ N, n ≤ y) :=
sorry

end NUMINAMATH_CALUDE_dedekind_cut_property_l965_96589


namespace NUMINAMATH_CALUDE_simplify_expression_l965_96585

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : x + y + z = 3) :
  (1 / (y^2 + z^2 - x^2)) + (1 / (x^2 + z^2 - y^2)) + (1 / (x^2 + y^2 - z^2)) =
  3 / (-9 + 6*y + 6*z - 2*y*z) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l965_96585


namespace NUMINAMATH_CALUDE_absolute_value_equality_l965_96586

theorem absolute_value_equality (y : ℝ) : 
  |y - 3| = |y - 5| → y = 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l965_96586


namespace NUMINAMATH_CALUDE_storks_joining_fence_l965_96550

theorem storks_joining_fence (initial_birds initial_storks : ℕ) 
  (h1 : initial_birds = 6)
  (h2 : initial_storks = 3)
  (joined_storks : ℕ)
  (h3 : initial_birds = initial_storks + joined_storks + 1) :
  joined_storks = 2 := by
sorry

end NUMINAMATH_CALUDE_storks_joining_fence_l965_96550


namespace NUMINAMATH_CALUDE_nelly_paid_correct_amount_l965_96514

/-- Nelly's payment for a painting at an auction -/
def nellys_payment (joe_bid sarah_bid : ℕ) : ℕ :=
  max
    (3 * joe_bid + 2000)
    (4 * sarah_bid + 1500)

/-- Theorem stating the correct amount Nelly paid for the painting -/
theorem nelly_paid_correct_amount :
  nellys_payment 160000 50000 = 482000 := by
  sorry

end NUMINAMATH_CALUDE_nelly_paid_correct_amount_l965_96514


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l965_96507

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0 ∧ 
   ∀ y : ℝ, k * y^2 + 2 * y + 1 = 0 → y = x) ↔ 
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l965_96507


namespace NUMINAMATH_CALUDE_triangle_problem_l965_96535

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the given conditions
def given_conditions (t : Triangle) : Prop :=
  (1 + Real.sin t.B + Real.cos t.B) * (Real.cos (t.B / 2) - Real.sin (t.B / 2)) = 
    7 / 12 * Real.sqrt (2 + 2 * Real.cos t.B) ∧
  t.c / t.a = 2 / 3

-- Define point D on side AC such that BD = AC
def point_D (t : Triangle) (D : ℝ) : Prop :=
  0 < D ∧ D < t.c ∧ Real.sqrt (t.a^2 + D^2 - 2 * t.a * D * Real.cos t.A) = t.c

-- State the theorem
theorem triangle_problem (t : Triangle) (D : ℝ) :
  given_conditions t → point_D t D →
  Real.cos t.B = 7 / 12 ∧ D / (t.c - D) = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l965_96535


namespace NUMINAMATH_CALUDE_number_properties_l965_96504

theorem number_properties :
  (∃! x : ℤ, ¬(x > 0) ∧ ¬(x < 0) ∧ x = 0) ∧
  (∃ x : ℤ, x < 0 ∧ ∀ y : ℤ, y < 0 → y ≤ x ∧ x = -1) ∧
  (∃ x : ℤ, x > 0 ∧ ∀ y : ℤ, y > 0 → x ≤ y ∧ x = 1) ∧
  (∃! x : ℤ, ∀ y : ℤ, |x| ≤ |y| ∧ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_number_properties_l965_96504


namespace NUMINAMATH_CALUDE_point_on_axes_l965_96582

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The coordinate axes, represented as a set of points -/
def CoordinateAxes : Set Point :=
  {p : Point | p.x = 0 ∨ p.y = 0}

/-- Theorem: If xy = 0, then the point is on the coordinate axes -/
theorem point_on_axes (p : Point) (h : p.x * p.y = 0) : p ∈ CoordinateAxes := by
  sorry

end NUMINAMATH_CALUDE_point_on_axes_l965_96582


namespace NUMINAMATH_CALUDE_indeterminate_value_l965_96547

theorem indeterminate_value (a b c d : ℝ) : 
  (b - d)^2 = 4 → 
  ¬∃!x, x = a + b - c - d :=
by sorry

end NUMINAMATH_CALUDE_indeterminate_value_l965_96547


namespace NUMINAMATH_CALUDE_profit_comparison_l965_96537

/-- The profit function for Product A before upgrade -/
def profit_A_before (raw_material : ℝ) : ℝ := 120000 * raw_material

/-- The profit function for Product A after upgrade -/
def profit_A_after (x : ℝ) : ℝ := 12 * (500 - x) * (1 + 0.005 * x)

/-- The profit function for Product B -/
def profit_B (x a : ℝ) : ℝ := 12 * (a - 0.013 * x) * x

theorem profit_comparison (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, 0 < x ∧ x ≤ 300 ∧ profit_A_after x ≥ profit_A_before 500) ∧
  (∀ x : ℝ, 0 < x → x ≤ 300 → profit_B x a ≤ profit_A_after x) →
  a ≤ 5.5 :=
sorry

end NUMINAMATH_CALUDE_profit_comparison_l965_96537


namespace NUMINAMATH_CALUDE_triangle_side_values_l965_96591

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_side_values :
  ∀ x : ℕ+, 
    (triangle_exists 5 (x.val ^ 2) 12) ↔ (x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l965_96591


namespace NUMINAMATH_CALUDE_flooring_rate_calculation_l965_96532

/-- Given a rectangular room with length 5.5 meters and width 3.75 meters,
    and a total flooring cost of 20625 rupees, the rate per square meter is 1000 rupees. -/
theorem flooring_rate_calculation (length : ℝ) (width : ℝ) (total_cost : ℝ) :
  length = 5.5 →
  width = 3.75 →
  total_cost = 20625 →
  total_cost / (length * width) = 1000 := by
  sorry

#check flooring_rate_calculation

end NUMINAMATH_CALUDE_flooring_rate_calculation_l965_96532


namespace NUMINAMATH_CALUDE_function_floor_property_l965_96520

theorem function_floor_property (f : ℝ → ℝ) :
  (∃ g : ℝ → ℝ, ∀ x y : ℝ, f x + f y = ⌊g (x + y)⌋) →
  ∃ n : ℤ, ∀ x : ℝ, f x = n / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_floor_property_l965_96520


namespace NUMINAMATH_CALUDE_expand_and_simplify_l965_96546

theorem expand_and_simplify (x : ℝ) : (x^2 + 4) * (x - 5) = x^3 - 5*x^2 + 4*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l965_96546


namespace NUMINAMATH_CALUDE_five_sqrt_two_gt_three_sqrt_three_l965_96599

theorem five_sqrt_two_gt_three_sqrt_three : 5 * Real.sqrt 2 > 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_five_sqrt_two_gt_three_sqrt_three_l965_96599


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l965_96573

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the bridge length calculation -/
theorem bridge_length_proof :
  bridge_length 80 45 30 = 295 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l965_96573


namespace NUMINAMATH_CALUDE_simplify_expression_l965_96525

theorem simplify_expression (p : ℝ) :
  ((6*p + 2) - 3*p*3)*4 + (5 - 2/4)*(8*p - 12) = 24*p - 46 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l965_96525


namespace NUMINAMATH_CALUDE_fishing_ratio_l965_96578

/-- Given the conditions of the fishing problem, prove that the ratio of trout to bass is 1:4 -/
theorem fishing_ratio : 
  ∀ (trout bass bluegill : ℕ),
  bass = 32 →
  bluegill = 2 * bass →
  trout + bass + bluegill = 104 →
  trout.gcd bass = 8 →
  (trout / 8 : ℚ) = 1 ∧ (bass / 8 : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fishing_ratio_l965_96578


namespace NUMINAMATH_CALUDE_bicycle_wheel_revolutions_l965_96598

/-- Calculates the number of revolutions of the back wheel given the diameters of both wheels and the number of revolutions of the front wheel. -/
theorem bicycle_wheel_revolutions 
  (front_diameter : ℝ) 
  (back_diameter : ℝ) 
  (front_revolutions : ℝ) : 
  front_diameter = 28 →
  back_diameter = 20 →
  front_revolutions = 50 →
  (back_diameter / front_diameter) * front_revolutions = 70 := by
sorry

end NUMINAMATH_CALUDE_bicycle_wheel_revolutions_l965_96598


namespace NUMINAMATH_CALUDE_min_value_2x_l965_96542

theorem min_value_2x (x y z : ℕ+) (h1 : 2 * x = 6 * z) (h2 : x + y + z = 26) : 2 * x = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_2x_l965_96542


namespace NUMINAMATH_CALUDE_towel_price_calculation_l965_96521

theorem towel_price_calculation (price1 price2 avg_price : ℕ) 
  (h1 : price1 = 100)
  (h2 : price2 = 150)
  (h3 : avg_price = 145) : 
  ∃ (unknown_price : ℕ), 
    (3 * price1 + 5 * price2 + 2 * unknown_price) / 10 = avg_price ∧ 
    unknown_price = 200 := by
sorry

end NUMINAMATH_CALUDE_towel_price_calculation_l965_96521


namespace NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l965_96562

def geometric_sequence (a₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₀ * r ^ n

theorem sum_of_fourth_and_fifth_terms :
  ∀ (a₀ r : ℝ),
    geometric_sequence a₀ r 0 = 4096 →
    geometric_sequence a₀ r 1 = 1024 →
    geometric_sequence a₀ r 2 = 256 →
    geometric_sequence a₀ r 5 = 4 →
    geometric_sequence a₀ r 6 = 1 →
    geometric_sequence a₀ r 3 + geometric_sequence a₀ r 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l965_96562


namespace NUMINAMATH_CALUDE_decagon_diagonals_l965_96524

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem decagon_diagonals : 
  num_diagonals 4 = 2 ∧ num_diagonals 5 = 5 → num_diagonals 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l965_96524


namespace NUMINAMATH_CALUDE_brennan_pepper_usage_l965_96559

/-- The amount of pepper Brennan used for scrambled eggs -/
def pepper_used (initial : ℝ) (remaining : ℝ) : ℝ := initial - remaining

/-- Theorem: Given Brennan's initial and remaining pepper amounts, prove he used 0.16 grams for scrambled eggs -/
theorem brennan_pepper_usage :
  let initial : ℝ := 0.25
  let remaining : ℝ := 0.09
  pepper_used initial remaining = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_brennan_pepper_usage_l965_96559


namespace NUMINAMATH_CALUDE_sqrt_product_plus_ten_l965_96515

theorem sqrt_product_plus_ten : Real.sqrt 18 * Real.sqrt 32 + 10 = 34 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_ten_l965_96515


namespace NUMINAMATH_CALUDE_tomorrow_is_saturday_l965_96526

-- Define the days of the week
inductive Day : Type
  | Sunday : Day
  | Monday : Day
  | Tuesday : Day
  | Wednesday : Day
  | Thursday : Day
  | Friday : Day
  | Saturday : Day

-- Define a function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

-- Define a function to add days
def addDays (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ k => nextDay (addDays d k)

-- Theorem statement
theorem tomorrow_is_saturday 
  (h : addDays Day.Wednesday 5 = Day.Monday) : 
  nextDay Day.Friday = Day.Saturday :=
by sorry

end NUMINAMATH_CALUDE_tomorrow_is_saturday_l965_96526


namespace NUMINAMATH_CALUDE_rectangle_burn_time_l965_96544

/-- Represents a rectangle made of wooden toothpicks -/
structure ToothpickRectangle where
  rows : Nat
  cols : Nat
  burnTime : Nat  -- Time for one toothpick to burn in seconds

/-- Calculates the time for the entire structure to burn -/
def burnTime (rect : ToothpickRectangle) : Nat :=
  let maxPath := rect.rows + rect.cols - 2  -- Longest path from corner to middle
  (maxPath * rect.burnTime) + (rect.burnTime / 2)

theorem rectangle_burn_time :
  let rect := ToothpickRectangle.mk 3 5 10
  burnTime rect = 65 := by
  sorry

#eval burnTime (ToothpickRectangle.mk 3 5 10)

end NUMINAMATH_CALUDE_rectangle_burn_time_l965_96544


namespace NUMINAMATH_CALUDE_monkey_banana_distribution_l965_96536

/-- Represents the number of bananas each monkey receives when dividing the total equally -/
def bananas_per_monkey (num_monkeys : ℕ) (num_piles_type1 num_piles_type2 : ℕ) 
  (hands_per_pile_type1 hands_per_pile_type2 : ℕ) 
  (bananas_per_hand_type1 bananas_per_hand_type2 : ℕ) : ℕ :=
  let total_bananas := 
    num_piles_type1 * hands_per_pile_type1 * bananas_per_hand_type1 +
    num_piles_type2 * hands_per_pile_type2 * bananas_per_hand_type2
  total_bananas / num_monkeys

/-- Theorem stating that given the problem conditions, each monkey receives 99 bananas -/
theorem monkey_banana_distribution :
  bananas_per_monkey 12 6 4 9 12 14 9 = 99 := by
  sorry

end NUMINAMATH_CALUDE_monkey_banana_distribution_l965_96536


namespace NUMINAMATH_CALUDE_three_digit_number_divisibility_l965_96522

theorem three_digit_number_divisibility : ∃! x : ℕ, 
  100 ≤ x ∧ x ≤ 999 ∧ 
  (x - 6) % 7 = 0 ∧ 
  (x - 7) % 8 = 0 ∧ 
  (x - 8) % 9 = 0 ∧ 
  x = 503 := by sorry

end NUMINAMATH_CALUDE_three_digit_number_divisibility_l965_96522


namespace NUMINAMATH_CALUDE_mango_price_reduction_mango_price_reduction_result_l965_96541

/-- Calculates the percentage reduction in mango prices --/
theorem mango_price_reduction (original_cost : ℝ) (original_quantity : ℕ) 
  (reduced_cost : ℝ) (original_purchase : ℕ) (additional_mangoes : ℕ) : ℝ :=
  let original_price_per_mango := original_cost / original_quantity
  let original_purchase_quantity := reduced_cost / original_price_per_mango
  let new_purchase_quantity := original_purchase_quantity + additional_mangoes
  let new_price_per_mango := reduced_cost / new_purchase_quantity
  let price_reduction_percentage := (original_price_per_mango - new_price_per_mango) / original_price_per_mango * 100
  price_reduction_percentage

/-- The percentage reduction in mango prices is approximately 9.91% --/
theorem mango_price_reduction_result : 
  abs (mango_price_reduction 450 135 360 108 12 - 9.91) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_mango_price_reduction_mango_price_reduction_result_l965_96541


namespace NUMINAMATH_CALUDE_expression_simplification_l965_96501

theorem expression_simplification (x : ℝ) (hx : x ≠ 0) :
  (x - 2)^2 - x*(x - 1) + (x^3 - 4*x^2) / x^2 = -2*x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l965_96501


namespace NUMINAMATH_CALUDE_factor_polynomial_l965_96516

theorem factor_polynomial (x : ℝ) : 72 * x^7 - 250 * x^13 = 2 * x^7 * (2^2 * 3^2 - 5^3 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l965_96516


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_pi_third_l965_96581

theorem cos_2alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/3) = 2/3) : 
  Real.cos (2*α + π/3) = -1/9 := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_pi_third_l965_96581


namespace NUMINAMATH_CALUDE_unspent_portion_after_transfer_l965_96534

/-- Represents a credit card with a spending limit and balance. -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Calculates the unspent portion of a credit card's limit after a balance transfer. -/
def unspentPortionAfterTransfer (gold : CreditCard) (platinum : CreditCard) : ℝ :=
  platinum.limit - (platinum.balance + gold.balance)

/-- Theorem stating the unspent portion of the platinum card's limit after transferring the gold card's balance. -/
theorem unspent_portion_after_transfer 
  (gold : CreditCard) 
  (platinum : CreditCard) 
  (h1 : gold.limit > 0)
  (h2 : platinum.limit = 2 * gold.limit)
  (h3 : gold.balance = (1/3) * gold.limit)
  (h4 : platinum.balance = (1/4) * platinum.limit) :
  unspentPortionAfterTransfer gold platinum = (7/6) * gold.limit :=
by
  sorry

#check unspent_portion_after_transfer

end NUMINAMATH_CALUDE_unspent_portion_after_transfer_l965_96534


namespace NUMINAMATH_CALUDE_not_first_class_probability_l965_96548

theorem not_first_class_probability (A : Set α) (P : Set α → ℝ) 
  (h1 : P A = 0.65) : P (Aᶜ) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_not_first_class_probability_l965_96548


namespace NUMINAMATH_CALUDE_volunteer_schedule_l965_96563

theorem volunteer_schedule (ella fiona george harry : ℕ) 
  (h_ella : ella = 5)
  (h_fiona : fiona = 6)
  (h_george : george = 8)
  (h_harry : harry = 9) :
  Nat.lcm (Nat.lcm (Nat.lcm ella fiona) george) harry = 360 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_schedule_l965_96563


namespace NUMINAMATH_CALUDE_original_sheet_area_l965_96513

/-- Represents the dimensions and properties of a cardboard box created from a rectangular sheet. -/
structure CardboardBox where
  base_length : ℝ
  base_width : ℝ
  volume : ℝ

/-- Theorem stating that given the specified conditions, the original sheet area is 110 cm². -/
theorem original_sheet_area
  (box : CardboardBox)
  (base_length_eq : box.base_length = 5)
  (base_width_eq : box.base_width = 4)
  (volume_eq : box.volume = 60)
  : ℝ :=
by
  -- The proof goes here
  sorry

#check original_sheet_area

end NUMINAMATH_CALUDE_original_sheet_area_l965_96513


namespace NUMINAMATH_CALUDE_james_pizza_toppings_cost_l965_96568

/-- Calculates the cost of pizza toppings eaten by James -/
theorem james_pizza_toppings_cost :
  let num_pizzas : ℕ := 2
  let slices_per_pizza : ℕ := 6
  let topping_costs : List ℚ := [3/2, 2, 5/4]
  let james_portion : ℚ := 2/3

  let total_slices : ℕ := num_pizzas * slices_per_pizza
  let total_topping_cost : ℚ := (num_pizzas : ℚ) * (topping_costs.sum)
  let james_topping_cost : ℚ := james_portion * total_topping_cost

  james_topping_cost = 633/100 :=
by
  sorry

end NUMINAMATH_CALUDE_james_pizza_toppings_cost_l965_96568


namespace NUMINAMATH_CALUDE_average_allowance_proof_l965_96560

theorem average_allowance_proof (total_students : ℕ) (total_amount : ℚ) 
  (h1 : total_students = 60)
  (h2 : total_amount = 320)
  (h3 : (2 : ℚ) / 3 * total_students + (1 : ℚ) / 3 * total_students = total_students)
  (h4 : (1 : ℚ) / 3 * total_students * 4 + (2 : ℚ) / 3 * total_students * x = total_amount) :
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_allowance_proof_l965_96560


namespace NUMINAMATH_CALUDE_probability_of_ravi_selection_l965_96595

theorem probability_of_ravi_selection 
  (p_ram : ℝ) 
  (p_both : ℝ) 
  (h1 : p_ram = 6/7)
  (h2 : p_both = 0.17142857142857143) :
  p_both / p_ram = 0.2 :=
sorry

end NUMINAMATH_CALUDE_probability_of_ravi_selection_l965_96595


namespace NUMINAMATH_CALUDE_production_increase_l965_96543

theorem production_increase (original_hours original_output : ℝ) 
  (h_positive_hours : original_hours > 0)
  (h_positive_output : original_output > 0) :
  let new_hours := 0.9 * original_hours
  let new_rate := 2 * (original_output / original_hours)
  let new_output := new_hours * new_rate
  (new_output - original_output) / original_output = 0.8 := by
sorry

end NUMINAMATH_CALUDE_production_increase_l965_96543


namespace NUMINAMATH_CALUDE_function_forms_theorem_l965_96529

/-- The set of all non-negative integers -/
def S : Set ℕ := Set.univ

/-- The condition that must be satisfied by f, g, and h -/
def satisfies_condition (f g h : ℕ → ℕ) : Prop :=
  ∀ m n, f (m + n) = g m + h n + 2 * m * n

/-- The theorem stating the only possible forms of f, g, and h -/
theorem function_forms_theorem (f g h : ℕ → ℕ) 
  (h1 : satisfies_condition f g h) (h2 : g 1 = 1) (h3 : h 1 = 1) :
  ∃ a : ℕ, a ≤ 4 ∧ 
    (∀ n, f n = n^2 - a*n + 2*a) ∧
    (∀ n, g n = n^2 - a*n + a) ∧
    (∀ n, h n = n^2 - a*n + a) :=
sorry


end NUMINAMATH_CALUDE_function_forms_theorem_l965_96529


namespace NUMINAMATH_CALUDE_total_pictures_correct_l965_96538

/-- The number of pictures Nancy uploaded to Facebook -/
def total_pictures : ℕ := 51

/-- The number of pictures in the first album -/
def first_album_pictures : ℕ := 11

/-- The number of additional albums -/
def additional_albums : ℕ := 8

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 5

/-- Theorem stating that the total number of pictures is correct -/
theorem total_pictures_correct : 
  total_pictures = first_album_pictures + additional_albums * pictures_per_additional_album :=
by sorry

end NUMINAMATH_CALUDE_total_pictures_correct_l965_96538


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l965_96556

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x + 1 > 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l965_96556


namespace NUMINAMATH_CALUDE_equation_solution_l965_96505

theorem equation_solution (a : ℤ) : 
  (∃ x : ℕ+, (x - 4) / 6 - (a * x - 1) / 3 = 1 / 3) ↔ a = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l965_96505


namespace NUMINAMATH_CALUDE_six_friends_assignment_l965_96549

/-- The number of ways to assign friends to rooms -/
def assignment_ways (n : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.choose n 3 * 1 * Nat.factorial 3

/-- Theorem stating the number of ways to assign 6 friends to 6 rooms -/
theorem six_friends_assignment :
  assignment_ways 6 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_six_friends_assignment_l965_96549


namespace NUMINAMATH_CALUDE_lemonade_sales_l965_96540

theorem lemonade_sales (last_week : ℝ) (this_week : ℝ) (total : ℝ) : 
  this_week = 1.3 * last_week →
  total = last_week + this_week →
  total = 46 →
  last_week = 20 := by
sorry

end NUMINAMATH_CALUDE_lemonade_sales_l965_96540


namespace NUMINAMATH_CALUDE_solution_product_l965_96512

theorem solution_product (p q : ℝ) : 
  p ≠ q ∧ 
  (p - 7) * (3 * p + 11) = p^2 - 20 * p + 63 ∧ 
  (q - 7) * (3 * q + 11) = q^2 - 20 * q + 63 →
  (p + 2) * (q + 2) = -72 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l965_96512


namespace NUMINAMATH_CALUDE_zoo_animal_count_l965_96554

/-- Calculates the total number of animals in a zoo with specific enclosure arrangements. -/
def total_animals_in_zoo : ℕ :=
  let tiger_enclosures : ℕ := 4
  let zebra_enclosures : ℕ := tiger_enclosures * 2
  let elephant_enclosures : ℕ := zebra_enclosures + 1
  let giraffe_enclosures : ℕ := elephant_enclosures * 3
  let rhino_enclosures : ℕ := 4

  let tigers : ℕ := tiger_enclosures * 4
  let zebras : ℕ := zebra_enclosures * 10
  let elephants : ℕ := elephant_enclosures * 3
  let giraffes : ℕ := giraffe_enclosures * 2
  let rhinos : ℕ := rhino_enclosures * 1

  tigers + zebras + elephants + giraffes + rhinos

/-- Theorem stating that the total number of animals in the zoo is 181. -/
theorem zoo_animal_count : total_animals_in_zoo = 181 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_count_l965_96554


namespace NUMINAMATH_CALUDE_simplify_fraction_l965_96533

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l965_96533


namespace NUMINAMATH_CALUDE_sequence_sum_and_kth_term_l965_96571

theorem sequence_sum_and_kth_term 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (k : ℕ) 
  (h1 : ∀ n, S n = n^2 - 8*n) 
  (h2 : a k = 5) : 
  k = 7 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_and_kth_term_l965_96571


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l965_96545

/-- Definition of a repeating decimal with a single digit repeating -/
def repeating_decimal (d : ℕ) : ℚ := (d : ℚ) / 9

/-- The problem statement -/
theorem repeating_decimal_sum : 
  repeating_decimal 6 + repeating_decimal 2 - repeating_decimal 4 = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l965_96545


namespace NUMINAMATH_CALUDE_cos_135_degrees_l965_96511

theorem cos_135_degrees : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l965_96511


namespace NUMINAMATH_CALUDE_min_sum_positive_integers_l965_96574

theorem min_sum_positive_integers (a b x y z : ℕ+) 
  (h : (3 : ℕ) * a.val = (7 : ℕ) * b.val ∧ 
       (7 : ℕ) * b.val = (5 : ℕ) * x.val ∧ 
       (5 : ℕ) * x.val = (4 : ℕ) * y.val ∧ 
       (4 : ℕ) * y.val = (6 : ℕ) * z.val) : 
  a.val + b.val + x.val + y.val + z.val ≥ 459 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_positive_integers_l965_96574


namespace NUMINAMATH_CALUDE_function_properties_l965_96565

-- Define the function f(x) = -x^2 + mx - m
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x - m

theorem function_properties (m : ℝ) :
  -- 1. If the maximum value of f(x) is 0, then m = 0 or m = 4
  (∃ (max : ℝ), (∀ (x : ℝ), f m x ≤ max) ∧ (max = 0)) →
  (m = 0 ∨ m = 4) ∧

  -- 2. If f(x) is monotonically decreasing on [-1, 0], then m ≤ -2
  (∀ (x y : ℝ), -1 ≤ x ∧ x < y ∧ y ≤ 0 → f m x > f m y) →
  (m ≤ -2) ∧

  -- 3. The range of f(x) on [2, 3] is exactly [2, 3] if and only if m = 6
  (∀ (y : ℝ), 2 ≤ y ∧ y ≤ 3 ↔ ∃ (x : ℝ), 2 ≤ x ∧ x ≤ 3 ∧ f m x = y) ↔
  (m = 6) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l965_96565


namespace NUMINAMATH_CALUDE_power_division_multiplication_l965_96510

theorem power_division_multiplication : (8^3 / 8^2) * 3^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_power_division_multiplication_l965_96510


namespace NUMINAMATH_CALUDE_four_three_three_cuboid_two_face_count_l965_96539

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with exactly two painted faces in a cuboid -/
def count_two_face_cubes (c : Cuboid) : ℕ :=
  2 * (c.length - 2) + 2 * (c.width - 2) + 2 * (c.height - 2)

/-- Theorem: A 4x3x3 cuboid has 16 cubes with exactly two painted faces -/
theorem four_three_three_cuboid_two_face_count :
  count_two_face_cubes ⟨4, 3, 3⟩ = 16 := by
  sorry

end NUMINAMATH_CALUDE_four_three_three_cuboid_two_face_count_l965_96539


namespace NUMINAMATH_CALUDE_orange_harvest_sacks_l965_96592

/-- Proves that harvesting 38 sacks per day for 49 days results in 1862 sacks total. -/
theorem orange_harvest_sacks (daily_harvest : ℕ) (days : ℕ) (total_sacks : ℕ) 
  (h1 : daily_harvest = 38)
  (h2 : days = 49)
  (h3 : total_sacks = 1862) :
  daily_harvest * days = total_sacks :=
by sorry

end NUMINAMATH_CALUDE_orange_harvest_sacks_l965_96592


namespace NUMINAMATH_CALUDE_stating_paint_usage_calculation_l965_96584

/-- 
Given an initial amount of paint and usage fractions for two weeks,
calculate the total amount of paint used.
-/
def paint_used (initial_paint : ℝ) (week1_fraction : ℝ) (week2_fraction : ℝ) : ℝ :=
  let week1_usage := initial_paint * week1_fraction
  let remaining_paint := initial_paint - week1_usage
  let week2_usage := remaining_paint * week2_fraction
  week1_usage + week2_usage

/-- 
Theorem stating that given 360 gallons of initial paint, 
using 1/4 of all paint in the first week and 1/6 of the remaining paint 
in the second week results in a total usage of 135 gallons of paint.
-/
theorem paint_usage_calculation :
  paint_used 360 (1/4) (1/6) = 135 := by
  sorry


end NUMINAMATH_CALUDE_stating_paint_usage_calculation_l965_96584
