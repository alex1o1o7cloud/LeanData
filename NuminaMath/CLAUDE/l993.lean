import Mathlib

namespace NUMINAMATH_CALUDE_position_change_l993_99363

/-- The position of a person from the back in a line of descending height order -/
def position_from_back_descending (total : ℕ) (position : ℕ) : Prop :=
  position > 0 ∧ position ≤ total

/-- The position of a person from the back in a line of ascending height order -/
def position_from_back_ascending (total : ℕ) (position : ℕ) : Prop :=
  position > 0 ∧ position ≤ total

/-- Theorem stating the relationship between a person's position in descending and ascending order lines -/
theorem position_change 
  (total : ℕ) 
  (position_desc : ℕ) 
  (position_asc : ℕ) 
  (h1 : total = 22)
  (h2 : position_desc = 13)
  (h3 : position_from_back_descending total position_desc)
  (h4 : position_from_back_ascending total position_asc)
  : position_asc = 10 := by
  sorry

#check position_change

end NUMINAMATH_CALUDE_position_change_l993_99363


namespace NUMINAMATH_CALUDE_max_m_value_max_m_is_maximum_l993_99302

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem max_m_value (m : ℝ) : 
  (∃ t : ℝ, ∀ x ∈ Set.Icc 2 m, f (x + t) ≤ 2 * x) → 
  m ≤ 8 ∧ ∃ t : ℝ, ∀ x ∈ Set.Icc 2 8, f (x + t) ≤ 2 * x :=
sorry

-- Define the maximum value of m
def max_m : ℝ := 8

-- Prove that max_m is indeed the maximum value
theorem max_m_is_maximum :
  (∃ t : ℝ, ∀ x ∈ Set.Icc 2 max_m, f (x + t) ≤ 2 * x) ∧
  ∀ m > max_m, ¬(∃ t : ℝ, ∀ x ∈ Set.Icc 2 m, f (x + t) ≤ 2 * x) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_max_m_is_maximum_l993_99302


namespace NUMINAMATH_CALUDE_probability_in_standard_deck_l993_99392

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (face_cards : Nat)
  (hearts : Nat)
  (face_hearts : Nat)

/-- Calculates the probability of drawing a face card, then any heart, then a face card -/
def probability_face_heart_face (d : Deck) : Rat :=
  let first_draw := d.face_cards / d.total_cards
  let second_draw := (d.hearts - d.face_hearts) / (d.total_cards - 1)
  let third_draw := (d.face_cards - 1) / (d.total_cards - 2)
  first_draw * second_draw * third_draw

/-- Standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52
  , face_cards := 12
  , hearts := 13
  , face_hearts := 3 }

theorem probability_in_standard_deck :
  probability_face_heart_face standard_deck = 1320 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_standard_deck_l993_99392


namespace NUMINAMATH_CALUDE_shortest_translation_distance_line_to_circle_l993_99333

/-- The shortest distance to translate a line to become tangent to a circle -/
theorem shortest_translation_distance_line_to_circle 
  (line : ℝ → ℝ → Prop) 
  (circle : ℝ → ℝ → Prop) 
  (h_line : ∀ x y, line x y ↔ x - y + 1 = 0)
  (h_circle : ∀ x y, circle x y ↔ (x - 2)^2 + (y - 1)^2 = 1) :
  ∃ d : ℝ, d = Real.sqrt 2 - 1 ∧ 
    (∀ d' : ℝ, d' ≥ 0 → 
      (∃ c : ℝ, ∀ x y, (x - y + c = 0 → circle x y) → d' ≥ d)) :=
sorry

end NUMINAMATH_CALUDE_shortest_translation_distance_line_to_circle_l993_99333


namespace NUMINAMATH_CALUDE_alissa_presents_l993_99346

/-- Given that Ethan has 31 presents and Alissa has 22 more presents than Ethan,
    prove that Alissa has 53 presents. -/
theorem alissa_presents (ethan_presents : ℕ) (alissa_extra : ℕ) :
  ethan_presents = 31 → alissa_extra = 22 → ethan_presents + alissa_extra = 53 :=
by sorry

end NUMINAMATH_CALUDE_alissa_presents_l993_99346


namespace NUMINAMATH_CALUDE_even_composition_l993_99394

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem even_composition (g : ℝ → ℝ) (h : IsEven g) : IsEven (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_even_composition_l993_99394


namespace NUMINAMATH_CALUDE_number_equation_solution_l993_99347

theorem number_equation_solution :
  ∃ N : ℝ, N - (1002 / 20.04) = 2450 ∧ N = 2500 := by
sorry

end NUMINAMATH_CALUDE_number_equation_solution_l993_99347


namespace NUMINAMATH_CALUDE_four_coin_stacking_methods_l993_99300

/-- Represents a coin with two sides -/
inductive Coin
| Head
| Tail

/-- Represents a stack of coins -/
def CoinStack := List Coin

/-- Checks if a given coin stack is valid (no adjacent heads) -/
def is_valid_stack (stack : CoinStack) : Bool :=
  match stack with
  | [] => true
  | [_] => true
  | Coin.Head :: Coin.Head :: _ => false
  | _ :: rest => is_valid_stack rest

/-- Generates all possible coin stacks of a given length -/
def generate_stacks (n : Nat) : List CoinStack :=
  if n = 0 then [[]]
  else
    let prev_stacks := generate_stacks (n - 1)
    prev_stacks.bind (fun stack => [Coin.Head :: stack, Coin.Tail :: stack])

/-- Counts the number of valid coin stacks of a given length -/
def count_valid_stacks (n : Nat) : Nat :=
  (generate_stacks n).filter is_valid_stack |>.length

/-- The main theorem to be proved -/
theorem four_coin_stacking_methods :
  count_valid_stacks 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_four_coin_stacking_methods_l993_99300


namespace NUMINAMATH_CALUDE_cube_sum_of_sqrt_equals_24_l993_99395

theorem cube_sum_of_sqrt_equals_24 :
  (Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3))^3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_of_sqrt_equals_24_l993_99395


namespace NUMINAMATH_CALUDE_ellipse_set_is_ellipse_l993_99398

/-- Two fixed points in a plane -/
structure FixedPoints (α : Type*) [NormedAddCommGroup α] :=
  (A B : α)

/-- The set of points P such that PA + PB = 2AB -/
def EllipseSet (α : Type*) [NormedAddCommGroup α] (points : FixedPoints α) : Set α :=
  {P : α | ‖P - points.A‖ + ‖P - points.B‖ = 2 * ‖points.A - points.B‖}

/-- Definition of an ellipse with given foci and major axis -/
def Ellipse (α : Type*) [NormedAddCommGroup α] (F₁ F₂ : α) (major_axis : ℝ) : Set α :=
  {P : α | ‖P - F₁‖ + ‖P - F₂‖ = major_axis}

/-- Theorem stating that the set of points P such that PA + PB = 2AB 
    forms an ellipse with A and B as foci and major axis 2AB -/
theorem ellipse_set_is_ellipse (α : Type*) [NormedAddCommGroup α] (points : FixedPoints α) :
  EllipseSet α points = Ellipse α points.A points.B (2 * ‖points.A - points.B‖) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_set_is_ellipse_l993_99398


namespace NUMINAMATH_CALUDE_original_average_l993_99380

theorem original_average (n : ℕ) (a : ℝ) (b : ℝ) (c : ℝ) :
  n > 0 →
  n = 15 →
  b = 13 →
  c = 53 →
  (a + b = c) →
  a = 40 := by
sorry

end NUMINAMATH_CALUDE_original_average_l993_99380


namespace NUMINAMATH_CALUDE_roof_area_l993_99370

theorem roof_area (width length : ℝ) : 
  width > 0 →
  length > 0 →
  length = 4 * width →
  length - width = 48 →
  width * length = 1024 := by
  sorry

end NUMINAMATH_CALUDE_roof_area_l993_99370


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_hundred_equals_one_fourth_more_than_sixty_l993_99319

theorem twenty_five_percent_less_than_hundred_equals_one_fourth_more_than_sixty : 
  (100 : ℝ) * (1 - 0.25) = 60 * (1 + 0.25) := by sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_hundred_equals_one_fourth_more_than_sixty_l993_99319


namespace NUMINAMATH_CALUDE_safe_combination_l993_99325

/-- Represents a digit in base 10 -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def distinct (n i m a : Digit) : Prop :=
  n ≠ i ∧ n ≠ m ∧ n ≠ a ∧ i ≠ m ∧ i ≠ a ∧ m ≠ a

/-- Converts a three-digit number in base 10 to its decimal value -/
def toDecimal (n i m : Digit) : Nat :=
  100 * n.val + 10 * i.val + m.val

/-- Checks if the equation NIM + AM + MIA = MINA holds in base 10 -/
def equationHolds (n i m a : Digit) : Prop :=
  (100 * n.val + 10 * i.val + m.val) +
  (10 * a.val + m.val) +
  (100 * m.val + 10 * i.val + a.val) =
  (1000 * m.val + 100 * i.val + 10 * n.val + a.val)

theorem safe_combination :
  ∃! (n i m a : Digit), distinct n i m a ∧
  equationHolds n i m a ∧
  toDecimal n i m = 845 := by
sorry

end NUMINAMATH_CALUDE_safe_combination_l993_99325


namespace NUMINAMATH_CALUDE_eulers_formula_applications_l993_99375

open Complex

theorem eulers_formula_applications :
  let e_2pi_3i : ℂ := Complex.exp ((2 * Real.pi / 3) * I)
  let e_pi_2i : ℂ := Complex.exp ((Real.pi / 2) * I)
  let e_pi_i : ℂ := Complex.exp (Real.pi * I)
  (e_2pi_3i.re < 0 ∧ e_2pi_3i.im > 0) ∧
  (e_pi_2i = I) ∧
  (abs (e_pi_i / (Real.sqrt 3 + I)) = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_eulers_formula_applications_l993_99375


namespace NUMINAMATH_CALUDE_stratified_sampling_third_year_l993_99357

theorem stratified_sampling_third_year 
  (total_students : ℕ) 
  (third_year_students : ℕ) 
  (total_sample : ℕ) 
  (h1 : total_students = 1200)
  (h2 : third_year_students = 300)
  (h3 : total_sample = 100) :
  (third_year_students : ℚ) / total_students * total_sample = 25 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_year_l993_99357


namespace NUMINAMATH_CALUDE_complex_linear_combination_l993_99388

theorem complex_linear_combination (a b : ℂ) (h1 : a = 3 + 2*I) (h2 : b = 2 - 3*I) :
  2*a + 3*b = 12 - 5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_linear_combination_l993_99388


namespace NUMINAMATH_CALUDE_drilled_cube_surface_area_l993_99317

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a drilled tunnel -/
structure DrilledCube where
  edgeLength : ℝ
  tunnelStartDistance : ℝ

/-- Calculates the surface area of a drilled cube -/
noncomputable def surfaceArea (cube : DrilledCube) : ℝ :=
  sorry

theorem drilled_cube_surface_area :
  let cube : DrilledCube := { edgeLength := 10, tunnelStartDistance := 3 }
  surfaceArea cube = 582 + 42 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_drilled_cube_surface_area_l993_99317


namespace NUMINAMATH_CALUDE_inequality_proof_l993_99373

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l993_99373


namespace NUMINAMATH_CALUDE_fish_count_ratio_l993_99308

/-- The ratio of fish counted on day 2 to fish counted on day 1 -/
theorem fish_count_ratio : 
  ∀ (fish_day1 fish_day2 sharks_total : ℕ) 
    (shark_percentage : ℚ),
  fish_day1 = 15 →
  sharks_total = 15 →
  shark_percentage = 1/4 →
  (↑fish_day1 * shark_percentage).floor + 
    (↑fish_day2 * shark_percentage).floor = sharks_total →
  fish_day2 / fish_day1 = 16/5 := by
sorry

end NUMINAMATH_CALUDE_fish_count_ratio_l993_99308


namespace NUMINAMATH_CALUDE_digit_101_of_7_12_l993_99355

/-- The decimal representation of 7/12 has a repeating sequence of 4 digits. -/
def decimal_7_12_period : ℕ := 4

/-- The first digit of the repeating sequence in the decimal representation of 7/12. -/
def first_digit_7_12 : ℕ := 5

/-- The 101st digit after the decimal point in the decimal representation of 7/12 is 5. -/
theorem digit_101_of_7_12 : 
  (101 % decimal_7_12_period = 1) → 
  (Nat.digitChar (first_digit_7_12) = '5') := by
sorry

end NUMINAMATH_CALUDE_digit_101_of_7_12_l993_99355


namespace NUMINAMATH_CALUDE_physics_marks_l993_99322

theorem physics_marks (P C M : ℝ) 
  (total_avg : (P + C + M) / 3 = 70)
  (physics_math_avg : (P + M) / 2 = 90)
  (physics_chem_avg : (P + C) / 2 = 70) :
  P = 110 := by
sorry

end NUMINAMATH_CALUDE_physics_marks_l993_99322


namespace NUMINAMATH_CALUDE_age_when_dog_born_is_15_l993_99323

/-- The age of the person when their dog was born -/
def age_when_dog_born (current_age : ℕ) (dog_future_age : ℕ) (years_until_future : ℕ) : ℕ :=
  current_age - (dog_future_age - years_until_future)

/-- Theorem stating the age when the dog was born -/
theorem age_when_dog_born_is_15 :
  age_when_dog_born 17 4 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_age_when_dog_born_is_15_l993_99323


namespace NUMINAMATH_CALUDE_exterior_angle_is_60_l993_99331

/-- An isosceles triangle with one angle opposite an equal side being 30 degrees -/
structure IsoscelesTriangle30 where
  /-- The measure of one of the angles opposite an equal side -/
  angle_opposite_equal_side : ℝ
  /-- The measure of the largest angle in the triangle -/
  largest_angle : ℝ
  /-- The fact that the triangle is isosceles with one angle being 30 degrees -/
  is_isosceles_30 : angle_opposite_equal_side = 30

/-- The measure of the exterior angle adjacent to the largest angle in the triangle -/
def exterior_angle (t : IsoscelesTriangle30) : ℝ := 180 - t.largest_angle

/-- Theorem: The measure of the exterior angle adjacent to the largest angle is 60 degrees -/
theorem exterior_angle_is_60 (t : IsoscelesTriangle30) : exterior_angle t = 60 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_is_60_l993_99331


namespace NUMINAMATH_CALUDE_max_profit_at_0_032_l993_99340

-- Define the bank's profit function
def bankProfit (k : ℝ) (x : ℝ) : ℝ := 0.048 * k * x^2 - k * x^3

-- State the theorem
theorem max_profit_at_0_032 (k : ℝ) (h_k : k > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 0.048 ∧
  ∀ (y : ℝ), y ∈ Set.Ioo 0 0.048 → bankProfit k x ≥ bankProfit k y :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_0_032_l993_99340


namespace NUMINAMATH_CALUDE_square_distance_equivalence_l993_99336

theorem square_distance_equivalence :
  ∀ (s : Real), s = 1 →
  (5 : Real) / Real.sqrt 2 = (5 : Real) / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_square_distance_equivalence_l993_99336


namespace NUMINAMATH_CALUDE_cycle_price_proof_l993_99386

/-- Represents the original price of a cycle -/
def original_price : ℝ := 800

/-- Represents the selling price of the cycle -/
def selling_price : ℝ := 680

/-- Represents the loss percentage -/
def loss_percentage : ℝ := 15

theorem cycle_price_proof :
  selling_price = original_price * (1 - loss_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_cycle_price_proof_l993_99386


namespace NUMINAMATH_CALUDE_nancy_pots_proof_l993_99339

/-- Represents the number of clay pots Nancy created on Monday -/
def monday_pots : ℕ := sorry

/-- The total number of pots Nancy created over three days -/
def total_pots : ℕ := 50

/-- The number of pots Nancy created on Wednesday -/
def wednesday_pots : ℕ := 14

theorem nancy_pots_proof :
  monday_pots = 12 ∧
  monday_pots + 2 * monday_pots + wednesday_pots = total_pots :=
sorry

end NUMINAMATH_CALUDE_nancy_pots_proof_l993_99339


namespace NUMINAMATH_CALUDE_positive_cubic_interval_l993_99304

theorem positive_cubic_interval (x : ℝ) :
  (x + 1) * (x - 1) * (x + 3) > 0 ↔ x ∈ Set.Ioo (-3 : ℝ) (-1 : ℝ) ∪ Set.Ioi (1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_positive_cubic_interval_l993_99304


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l993_99311

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) : 
  square_perimeter = 40 →
  triangle_height = 60 →
  (square_perimeter / 4)^2 = (1/2) * triangle_height * x →
  x = 10/3 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l993_99311


namespace NUMINAMATH_CALUDE_initial_maple_trees_l993_99316

theorem initial_maple_trees (cut_trees : ℝ) (remaining_trees : ℕ) 
  (h1 : cut_trees = 2.0)
  (h2 : remaining_trees = 7) :
  cut_trees + remaining_trees = 9.0 := by
  sorry

end NUMINAMATH_CALUDE_initial_maple_trees_l993_99316


namespace NUMINAMATH_CALUDE_coloring_books_remaining_l993_99365

theorem coloring_books_remaining (initial : Real) (first_giveaway : Real) (second_giveaway : Real) :
  initial = 48.0 →
  first_giveaway = 34.0 →
  second_giveaway = 3.0 →
  initial - first_giveaway - second_giveaway = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_remaining_l993_99365


namespace NUMINAMATH_CALUDE_min_a_for_increasing_f_l993_99310

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

-- State the theorem
theorem min_a_for_increasing_f :
  (∀ a : ℝ, ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x < y → f a x < f a y) →
  (∃ a_min : ℝ, a_min = -3 ∧ 
    (∀ a : ℝ, (∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x < y → f a x < f a y) → a ≥ a_min)) :=
sorry

end NUMINAMATH_CALUDE_min_a_for_increasing_f_l993_99310


namespace NUMINAMATH_CALUDE_round_trip_distance_l993_99378

/-- Proves that the total distance of a round trip is 2 miles given the specified conditions -/
theorem round_trip_distance
  (outbound_time : ℝ) (return_time : ℝ) (average_speed : ℝ)
  (h1 : outbound_time = 10) -- outbound time in minutes
  (h2 : return_time = 20) -- return time in minutes
  (h3 : average_speed = 4) -- average speed in miles per hour
  : (outbound_time + return_time) / 60 * average_speed = 2 := by
  sorry

#check round_trip_distance

end NUMINAMATH_CALUDE_round_trip_distance_l993_99378


namespace NUMINAMATH_CALUDE_fraction_equality_l993_99351

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 3)
  (h3 : p / q = 1 / 5) :
  m / q = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l993_99351


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l993_99367

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 - 2*x₁ - 8 = 0) ∧ 
  (x₂^2 - 2*x₂ - 8 = 0) ∧ 
  x₁ = 4 ∧ 
  x₂ = -2 :=
by
  sorry

#check quadratic_equation_solution

end NUMINAMATH_CALUDE_quadratic_equation_solution_l993_99367


namespace NUMINAMATH_CALUDE_peach_price_to_friends_peach_price_proof_l993_99321

/-- The price of peaches sold to friends, given the following conditions:
  * Lilia has 15 peaches
  * She sold 10 peaches to friends
  * She sold 4 peaches to relatives for $1.25 each
  * She kept 1 peach for herself
  * She earned $25 in total from selling 14 peaches
-/
theorem peach_price_to_friends : ℝ :=
  let total_peaches : ℕ := 15
  let peaches_to_friends : ℕ := 10
  let peaches_to_relatives : ℕ := 4
  let peaches_kept : ℕ := 1
  let price_to_relatives : ℝ := 1.25
  let total_earned : ℝ := 25
  let price_to_friends : ℝ := (total_earned - peaches_to_relatives * price_to_relatives) / peaches_to_friends
  2

theorem peach_price_proof (total_peaches : ℕ) (peaches_to_friends : ℕ) (peaches_to_relatives : ℕ) 
    (peaches_kept : ℕ) (price_to_relatives : ℝ) (total_earned : ℝ) :
    total_peaches = peaches_to_friends + peaches_to_relatives + peaches_kept →
    total_earned = peaches_to_friends * peach_price_to_friends + peaches_to_relatives * price_to_relatives →
    peach_price_to_friends = 2 :=
by sorry

end NUMINAMATH_CALUDE_peach_price_to_friends_peach_price_proof_l993_99321


namespace NUMINAMATH_CALUDE_inequality_proof_l993_99341

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a * (c - 1) < b * (c - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l993_99341


namespace NUMINAMATH_CALUDE_max_gcd_17n_plus_4_10n_plus_3_l993_99379

theorem max_gcd_17n_plus_4_10n_plus_3 :
  ∃ (k : ℕ), k > 0 ∧
  Nat.gcd (17 * k + 4) (10 * k + 3) = 11 ∧
  ∀ (n : ℕ), n > 0 → Nat.gcd (17 * n + 4) (10 * n + 3) ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_17n_plus_4_10n_plus_3_l993_99379


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l993_99387

theorem sum_of_three_consecutive_integers (a b c : ℤ) : 
  (a + 1 = b) → (b + 1 = c) → (c = 12) → (a + b + c = 33) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l993_99387


namespace NUMINAMATH_CALUDE_border_area_is_198_l993_99328

-- Define the photograph dimensions
def photo_height : ℕ := 12
def photo_width : ℕ := 15

-- Define the frame border width
def border_width : ℕ := 3

-- Define the function to calculate the area of the border
def border_area (h w b : ℕ) : ℕ :=
  (h + 2*b) * (w + 2*b) - h * w

-- Theorem statement
theorem border_area_is_198 :
  border_area photo_height photo_width border_width = 198 := by
  sorry

end NUMINAMATH_CALUDE_border_area_is_198_l993_99328


namespace NUMINAMATH_CALUDE_maria_furniture_assembly_time_l993_99350

def total_time (num_chairs num_tables time_per_piece : ℕ) : ℕ :=
  (num_chairs + num_tables) * time_per_piece

theorem maria_furniture_assembly_time :
  total_time 2 2 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_maria_furniture_assembly_time_l993_99350


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_8_l993_99335

def is_greatest_factorial_under_1000 (n : ℕ) : Prop :=
  n.factorial < 1000 ∧ ∀ m : ℕ, m > n → m.factorial ≥ 1000

theorem sum_of_x_and_y_is_8 :
  ∀ x y : ℕ,
    x > 0 →
    y > 1 →
    is_greatest_factorial_under_1000 x →
    x + y = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_8_l993_99335


namespace NUMINAMATH_CALUDE_abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two_l993_99343

theorem abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two :
  (∀ x : ℝ, x < -2 → |x| > 2) ∧
  ¬(∀ x : ℝ, |x| > 2 → x < -2) :=
by sorry

end NUMINAMATH_CALUDE_abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two_l993_99343


namespace NUMINAMATH_CALUDE_car_speed_problem_l993_99383

/-- Given a car traveling for 2 hours with an average speed of 55 km/h,
    if its speed in the second hour is 60 km/h,
    then its speed in the first hour must be 50 km/h. -/
theorem car_speed_problem (x : ℝ) : 
  (x + 60) / 2 = 55 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l993_99383


namespace NUMINAMATH_CALUDE_apples_bought_l993_99382

theorem apples_bought (apples pears : ℕ) : 
  pears = (3 * apples) / 5 →
  apples + pears = 240 →
  apples = 150 := by
sorry

end NUMINAMATH_CALUDE_apples_bought_l993_99382


namespace NUMINAMATH_CALUDE_selene_and_tanya_spending_l993_99372

/-- Represents the prices of items in the school canteen -/
structure CanteenPrices where
  sandwich : ℕ
  hamburger : ℕ
  hotdog : ℕ
  fruitJuice : ℕ

/-- Represents Selene's purchase -/
structure SelenePurchase where
  sandwiches : ℕ
  fruitJuice : ℕ

/-- Represents Tanya's purchase -/
structure TanyaPurchase where
  hamburgers : ℕ
  fruitJuice : ℕ

/-- Calculates the total spending of Selene and Tanya -/
def totalSpending (prices : CanteenPrices) (selene : SelenePurchase) (tanya : TanyaPurchase) : ℕ :=
  prices.sandwich * selene.sandwiches + prices.fruitJuice * selene.fruitJuice +
  prices.hamburger * tanya.hamburgers + prices.fruitJuice * tanya.fruitJuice

/-- Theorem stating that Selene and Tanya spend $16 in total -/
theorem selene_and_tanya_spending :
  ∀ (prices : CanteenPrices) (selene : SelenePurchase) (tanya : TanyaPurchase),
    prices.sandwich = 2 →
    prices.hamburger = 2 →
    prices.hotdog = 1 →
    prices.fruitJuice = 2 →
    selene.sandwiches = 3 →
    selene.fruitJuice = 1 →
    tanya.hamburgers = 2 →
    tanya.fruitJuice = 2 →
    totalSpending prices selene tanya = 16 := by
  sorry

end NUMINAMATH_CALUDE_selene_and_tanya_spending_l993_99372


namespace NUMINAMATH_CALUDE_possible_values_of_a_l993_99326

theorem possible_values_of_a (x y a : ℝ) 
  (eq1 : x + y = a) 
  (eq2 : x^3 + y^3 = a) 
  (eq3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l993_99326


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l993_99362

theorem rectangular_solid_volume (a b c : ℝ) 
  (side_area : a * b = 20)
  (front_area : b * c = 15)
  (bottom_area : a * c = 12)
  (dimension_relation : a = 2 * b) : 
  a * b * c = 12 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l993_99362


namespace NUMINAMATH_CALUDE_vector_projection_l993_99352

theorem vector_projection (a b : ℝ × ℝ) 
  (h1 : (a.1 + b.1, a.2 + b.2) • (2*a.1 - b.1, 2*a.2 - b.2) = -12)
  (h2 : Real.sqrt (a.1^2 + a.2^2) = 2)
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 4) :
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2) = -2 := by
sorry

end NUMINAMATH_CALUDE_vector_projection_l993_99352


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l993_99345

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = a n * q) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l993_99345


namespace NUMINAMATH_CALUDE_max_product_of_prime_factors_l993_99399

def primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

theorem max_product_of_prime_factors :
  ∃ (a b c d e f g : Nat),
    a ∈ primes ∧ b ∈ primes ∧ c ∈ primes ∧ d ∈ primes ∧
    e ∈ primes ∧ f ∈ primes ∧ g ∈ primes ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
    e ≠ f ∧ e ≠ g ∧
    f ≠ g ∧
    (a + b + c + d) * (e + f + g) = 841 ∧
    ∀ (x y z w u v t : Nat),
      x ∈ primes → y ∈ primes → z ∈ primes → w ∈ primes →
      u ∈ primes → v ∈ primes → t ∈ primes →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ u ∧ x ≠ v ∧ x ≠ t ∧
      y ≠ z ∧ y ≠ w ∧ y ≠ u ∧ y ≠ v ∧ y ≠ t ∧
      z ≠ w ∧ z ≠ u ∧ z ≠ v ∧ z ≠ t ∧
      w ≠ u ∧ w ≠ v ∧ w ≠ t ∧
      u ≠ v ∧ u ≠ t ∧
      v ≠ t →
      (x + y + z + w) * (u + v + t) ≤ 841 :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_prime_factors_l993_99399


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l993_99329

theorem diophantine_equation_solutions :
  let S : Set (ℤ × ℤ) := {(4, -1), (-26, -9), (-16, -9), (-6, -1), (50, 15), (-72, -25)}
  ∀ (x y : ℤ), (x^2 - 5*x*y + 6*y^2 - 3*x + 5*y - 25 = 0) ↔ (x, y) ∈ S :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l993_99329


namespace NUMINAMATH_CALUDE_angle_sum_l993_99332

theorem angle_sum (θ φ : Real) (h1 : 4 * (Real.cos θ)^2 + 3 * (Real.cos φ)^2 = 1)
  (h2 : 4 * Real.cos (2 * θ) + 3 * Real.sin (2 * φ) = 0)
  (h3 : 0 < θ ∧ θ < Real.pi / 2) (h4 : 0 < φ ∧ φ < Real.pi / 2) :
  θ + 3 * φ = Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_angle_sum_l993_99332


namespace NUMINAMATH_CALUDE_x_value_proof_l993_99393

theorem x_value_proof : 
  ∀ x : ℝ, x = 88 * (1 + 25 / 100) → x = 110 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l993_99393


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l993_99385

/-- The discriminant of a quadratic equation ax² + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_discriminant :
  let a : ℝ := 5
  let b : ℝ := 2
  let c : ℝ := -8
  discriminant a b c = 164 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l993_99385


namespace NUMINAMATH_CALUDE_escalator_solution_l993_99381

/-- Represents the escalator problem with given conditions -/
structure EscalatorProblem where
  total_steps : ℕ
  escalator_speed : ℚ
  walking_speed : ℚ
  first_condition : 26 + 30 * escalator_speed = total_steps
  second_condition : 34 + 18 * escalator_speed = total_steps

/-- The solution to the escalator problem -/
theorem escalator_solution (problem : EscalatorProblem) : problem.total_steps = 46 := by
  sorry

#check escalator_solution

end NUMINAMATH_CALUDE_escalator_solution_l993_99381


namespace NUMINAMATH_CALUDE_jelly_bean_percentage_l993_99320

theorem jelly_bean_percentage : 
  ∀ (total : ℕ) (removed : ℕ),
  total > 0 →
  removed ≤ (54 * total) / 100 →
  removed ≤ (30 * total) / 100 →
  (16 * total) / ((total - 2 * removed) * 100) = 1/5 →
  ((54 * total) / 100 - removed) / (total - 2 * removed) = 11/20 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_percentage_l993_99320


namespace NUMINAMATH_CALUDE_scientific_notation_393000_l993_99315

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_393000 :
  toScientificNotation 393000 = ScientificNotation.mk 3.93 5 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_393000_l993_99315


namespace NUMINAMATH_CALUDE_fraction_simplification_l993_99359

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hyz : y^3 - 1/x ≠ 0) : 
  (x^3 - 1/y) / (y^3 - 1/x) = x / y := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l993_99359


namespace NUMINAMATH_CALUDE_system_solutions_l993_99337

theorem system_solutions (a : ℤ) :
  let eq1 := fun (x y z : ℤ) => 5 * x + (a + 2) * y + (a + 2) * z = a
  let eq2 := fun (x y z : ℤ) => (2 * a + 4) * x + (a^2 + 3) * y + (2 * a + 2) * z = 3 * a - 1
  let eq3 := fun (x y z : ℤ) => (2 * a + 4) * x + (2 * a + 2) * y + (a^2 + 3) * z = a + 1
  (∀ x y z : ℤ, eq1 x y z ∧ eq2 x y z ∧ eq3 x y z ↔
    (a = 1 ∧ ∃ n : ℤ, x = -1 ∧ y = n ∧ z = 2 - n) ∨
    (a = -1 ∧ x = 0 ∧ y = -1 ∧ z = 0) ∨
    (a = 0 ∧ x = 0 ∧ y = -1 ∧ z = 1) ∨
    (a = 2 ∧ x = -6 ∧ y = 5 ∧ z = 3)) ∧
  (a = 3 → ¬∃ x y z : ℤ, eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) ∧
  (a ≠ 1 ∧ a ≠ -1 ∧ a ≠ 0 ∧ a ≠ 2 ∧ a ≠ 3 →
    ¬∃ x y z : ℤ, eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l993_99337


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_l993_99327

theorem min_bottles_to_fill (large_capacity : ℕ) (small_capacity1 small_capacity2 : ℕ) :
  large_capacity = 720 ∧ small_capacity1 = 40 ∧ small_capacity2 = 45 →
  ∃ (x y : ℕ), x * small_capacity1 + y * small_capacity2 = large_capacity ∧
                x + y = 16 ∧
                ∀ (a b : ℕ), a * small_capacity1 + b * small_capacity2 = large_capacity →
                              x + y ≤ a + b :=
by sorry

end NUMINAMATH_CALUDE_min_bottles_to_fill_l993_99327


namespace NUMINAMATH_CALUDE_principal_calculation_l993_99397

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (interest : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that given the specific conditions, the principal is 10040.625 -/
theorem principal_calculation :
  let interest : ℚ := 4016.25
  let rate : ℚ := 8
  let time : ℚ := 5
  calculate_principal interest rate time = 10040.625 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l993_99397


namespace NUMINAMATH_CALUDE_min_sum_of_product_2450_l993_99376

theorem min_sum_of_product_2450 (a b c : ℕ+) (h : a * b * c = 2450) :
  (∀ x y z : ℕ+, x * y * z = 2450 → a + b + c ≤ x + y + z) ∧ a + b + c = 82 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2450_l993_99376


namespace NUMINAMATH_CALUDE_tank_volume_l993_99369

/-- Given a cube-shaped tank constructed from metal sheets, calculate its volume in liters -/
theorem tank_volume (sheet_length : ℝ) (sheet_width : ℝ) (num_sheets : ℕ) : 
  sheet_length = 2 →
  sheet_width = 3 →
  num_sheets = 100 →
  (((num_sheets * sheet_length * sheet_width / 6) ^ (1/2 : ℝ)) ^ 3) * 1000 = 1000000 := by
  sorry

#check tank_volume

end NUMINAMATH_CALUDE_tank_volume_l993_99369


namespace NUMINAMATH_CALUDE_hyperbola_property_l993_99334

/-- The hyperbola with equation x^2 - y^2 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p | p.1^2 - p.2^2 = 1}

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- The length of the semi-major axis of the hyperbola -/
def a : ℝ := sorry

theorem hyperbola_property (P : ℝ × ℝ) (h₁ : P ∈ Hyperbola)
    (h₂ : (P.1 - F₁.1, P.2 - F₁.2) • (P.1 - F₂.1, P.2 - F₂.2) = 0) :
    ‖(P.1 - F₁.1, P.2 - F₁.2) + (P.1 - F₂.1, P.2 - F₂.2)‖ = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_property_l993_99334


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l993_99361

/-- Theorem: For a rectangular field with length twice its width and perimeter 600 meters,
    the width is 100 meters and the length is 200 meters. -/
theorem rectangular_field_dimensions :
  ∀ (width length : ℝ),
  length = 2 * width →
  2 * (length + width) = 600 →
  width = 100 ∧ length = 200 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l993_99361


namespace NUMINAMATH_CALUDE_seventh_root_product_l993_99338

theorem seventh_root_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_product_l993_99338


namespace NUMINAMATH_CALUDE_second_bag_count_l993_99348

/-- The number of bags of M&M's -/
def num_bags : ℕ := 5

/-- The number of brown M&M's in the first bag -/
def first_bag : ℕ := 9

/-- The number of brown M&M's in the third bag -/
def third_bag : ℕ := 8

/-- The number of brown M&M's in the fourth bag -/
def fourth_bag : ℕ := 8

/-- The number of brown M&M's in the fifth bag -/
def fifth_bag : ℕ := 3

/-- The average number of brown M&M's per bag -/
def average : ℕ := 8

/-- Theorem: Given the conditions, the second bag must contain 12 brown M&M's -/
theorem second_bag_count : 
  ∃ (second_bag : ℕ), 
    (first_bag + second_bag + third_bag + fourth_bag + fifth_bag) / num_bags = average ∧
    second_bag = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_bag_count_l993_99348


namespace NUMINAMATH_CALUDE_event_ticket_revenue_l993_99384

theorem event_ticket_revenue :
  ∀ (full_price half_price : ℕ),
  full_price + half_price = 180 →
  full_price * 20 + half_price * 10 = 2750 →
  full_price * 20 = 1900 :=
by
  sorry

end NUMINAMATH_CALUDE_event_ticket_revenue_l993_99384


namespace NUMINAMATH_CALUDE_asymptote_of_hyperbola_l993_99356

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - y^2 / 16 = 1

/-- The equation of an asymptote -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = (4/5) * x

/-- Theorem: The given equation is an asymptote of the hyperbola -/
theorem asymptote_of_hyperbola :
  ∀ x y : ℝ, asymptote_equation x y → (∃ ε > 0, ∀ δ > ε, 
    ∃ x' y' : ℝ, hyperbola_equation x' y' ∧ 
    ((x' - x)^2 + (y' - y)^2 < δ^2)) :=
sorry

end NUMINAMATH_CALUDE_asymptote_of_hyperbola_l993_99356


namespace NUMINAMATH_CALUDE_framing_for_enlarged_picture_l993_99303

/-- Calculates the minimum number of linear feet of framing needed for an enlarged picture with border -/
def minimum_framing_feet (original_width original_height enlargement_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (final_width + final_height)
  (perimeter_inches + 11) / 12  -- Dividing by 12 and rounding up

/-- Theorem stating that the minimum framing needed for the given picture is 10 feet -/
theorem framing_for_enlarged_picture :
  minimum_framing_feet 5 7 4 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_framing_for_enlarged_picture_l993_99303


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l993_99368

theorem condition_necessary_not_sufficient :
  (∀ a b : ℝ, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  (∃ a b : ℝ, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l993_99368


namespace NUMINAMATH_CALUDE_sum_of_ratios_geq_two_l993_99364

theorem sum_of_ratios_geq_two (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_geq_two_l993_99364


namespace NUMINAMATH_CALUDE_students_playing_neither_l993_99360

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) 
  (h1 : total = 40)
  (h2 : football = 26)
  (h3 : tennis = 20)
  (h4 : both = 17) :
  total - (football + tennis - both) = 11 := by
sorry

end NUMINAMATH_CALUDE_students_playing_neither_l993_99360


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l993_99312

theorem ratio_equation_solution (a b : ℚ) 
  (h1 : b / a = 4)
  (h2 : b = 15 - 6 * a) : 
  a = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l993_99312


namespace NUMINAMATH_CALUDE_wolves_out_hunting_l993_99374

def wolves_in_pack : ℕ := 16
def meat_per_wolf_per_day : ℕ := 8
def days_between_hunts : ℕ := 5
def meat_per_deer : ℕ := 200
def deer_per_hunting_wolf : ℕ := 1

def total_meat_needed : ℕ := wolves_in_pack * meat_per_wolf_per_day * days_between_hunts

def deer_needed : ℕ := (total_meat_needed + meat_per_deer - 1) / meat_per_deer

theorem wolves_out_hunting (hunting_wolves : ℕ) : 
  hunting_wolves * deer_per_hunting_wolf = deer_needed → hunting_wolves = 4 := by
  sorry

end NUMINAMATH_CALUDE_wolves_out_hunting_l993_99374


namespace NUMINAMATH_CALUDE_unique_zero_quadratic_l993_99391

/-- Given a quadratic function f(x) = 3x^2 + 2x - a with a unique zero in (-1, 1),
    prove that a ∈ (1, 5) ∪ {-1/3} -/
theorem unique_zero_quadratic (a : ℝ) :
  (∃! x : ℝ, x ∈ (Set.Ioo (-1) 1) ∧ 3 * x^2 + 2 * x - a = 0) →
  (a ∈ Set.Ioo 1 5 ∨ a = -1/3) :=
sorry

end NUMINAMATH_CALUDE_unique_zero_quadratic_l993_99391


namespace NUMINAMATH_CALUDE_necessary_condition_example_l993_99314

theorem necessary_condition_example : 
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧ 
  (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_example_l993_99314


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l993_99306

/-- Given points A(-3,m) and B(-2,n) lying on the hyperbolic function y = (k-1)/x, 
    with m > n, the range of k is k > 1 -/
theorem hyperbola_k_range (k m n : ℝ) : 
  (m = (k - 1) / (-3)) → 
  (n = (k - 1) / (-2)) → 
  (m > n) → 
  (k > 1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l993_99306


namespace NUMINAMATH_CALUDE_matches_played_before_increase_l993_99358

def cricket_matches (current_average : ℚ) (next_match_runs : ℕ) (new_average : ℚ) : Prop :=
  ∃ m : ℕ,
    (current_average * m + next_match_runs) / (m + 1) = new_average ∧
    m > 0

theorem matches_played_before_increase (current_average : ℚ) (next_match_runs : ℕ) (new_average : ℚ) :
  cricket_matches current_average next_match_runs new_average →
  current_average = 51 →
  next_match_runs = 78 →
  new_average = 54 →
  ∃ m : ℕ, m = 8 ∧ cricket_matches current_average next_match_runs new_average :=
by
  sorry

#check matches_played_before_increase

end NUMINAMATH_CALUDE_matches_played_before_increase_l993_99358


namespace NUMINAMATH_CALUDE_third_member_reels_six_l993_99324

/-- Represents a fishing competition with three team members -/
structure FishingCompetition where
  days : ℕ
  fish_per_day_1 : ℕ
  fish_per_day_2 : ℕ
  total_fish : ℕ

/-- Calculates the number of fish the third member reels per day -/
def third_member_fish_per_day (comp : FishingCompetition) : ℕ :=
  (comp.total_fish - (comp.fish_per_day_1 + comp.fish_per_day_2) * comp.days) / comp.days

/-- Theorem stating that in the given conditions, the third member reels 6 fish per day -/
theorem third_member_reels_six (comp : FishingCompetition) 
  (h1 : comp.days = 5)
  (h2 : comp.fish_per_day_1 = 4)
  (h3 : comp.fish_per_day_2 = 8)
  (h4 : comp.total_fish = 90) : 
  third_member_fish_per_day comp = 6 := by
  sorry

#eval third_member_fish_per_day ⟨5, 4, 8, 90⟩

end NUMINAMATH_CALUDE_third_member_reels_six_l993_99324


namespace NUMINAMATH_CALUDE_square_of_complex_l993_99301

theorem square_of_complex (z : ℂ) : z = 2 - 3*I → z^2 = -5 - 12*I := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_l993_99301


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l993_99377

theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : ∃ r : ℝ, b = a * r ∧ c = b * r) : 
  (a = 25 ∧ c = 1/4) → b = 5/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l993_99377


namespace NUMINAMATH_CALUDE_smallest_square_partition_l993_99354

theorem smallest_square_partition : ∃ (n : ℕ),
  n > 0 ∧
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a + b = 10 ∧ a ≥ 8 ∧ n^2 = a * 1^2 + b * 2^2) ∧
  (∀ (m : ℕ), m < n →
    ¬(∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c + d = 10 ∧ c ≥ 8 ∧ m^2 = c * 1^2 + d * 2^2)) ∧
  n = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_partition_l993_99354


namespace NUMINAMATH_CALUDE_complex_equation_sum_l993_99330

theorem complex_equation_sum (a b : ℝ) :
  (a + 2 * Complex.I) * Complex.I = b + Complex.I →
  ∃ (result : ℝ), a + b = result :=
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l993_99330


namespace NUMINAMATH_CALUDE_parallel_chords_central_angles_l993_99309

/-- Given a circle with parallel chords of lengths 5, 12, and 13 determining
    central angles α, β, and α + β radians respectively, where α + β < π,
    prove that α + β = π/2 -/
theorem parallel_chords_central_angles
  (α β : Real)
  (h1 : 0 < α) (h2 : 0 < β)
  (h3 : α + β < π)
  (h4 : 2 * Real.sin (α / 2) = 5 / (2 * R))
  (h5 : 2 * Real.sin (β / 2) = 12 / (2 * R))
  (h6 : 2 * Real.sin ((α + β) / 2) = 13 / (2 * R))
  (R : Real) (h7 : R > 0) :
  α + β = π / 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_chords_central_angles_l993_99309


namespace NUMINAMATH_CALUDE_older_brother_stamps_l993_99313

theorem older_brother_stamps (total : ℕ) (younger : ℕ) (older : ℕ) : 
  total = 25 →
  older = 2 * younger + 1 →
  total = older + younger →
  older = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_older_brother_stamps_l993_99313


namespace NUMINAMATH_CALUDE_cakes_left_with_brenda_l993_99371

def cakes_per_day : ℕ := 20
def days_baking : ℕ := 9
def fraction_sold : ℚ := 1/2

theorem cakes_left_with_brenda : 
  (cakes_per_day * days_baking) * (1 - fraction_sold) = 90 := by
  sorry

end NUMINAMATH_CALUDE_cakes_left_with_brenda_l993_99371


namespace NUMINAMATH_CALUDE_tangency_points_coordinates_l993_99342

/-- The coordinates of points of tangency to the discriminant parabola -/
theorem tangency_points_coordinates (p q : ℝ) :
  let parabola := {(x, y) : ℝ × ℝ | x^2 - 4*y = 0}
  let tangent_point := (p, q)
  ∃ (p₀ q₀ : ℝ), (p₀, q₀) ∈ parabola ∧
    (p₀ = p + Real.sqrt (p^2 - 4*q) ∨ p₀ = p - Real.sqrt (p^2 - 4*q)) ∧
    q₀ = (p^2 - 2*q + p * Real.sqrt (p^2 - 4*q)) / 2 ∨
    q₀ = (p^2 - 2*q - p * Real.sqrt (p^2 - 4*q)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_tangency_points_coordinates_l993_99342


namespace NUMINAMATH_CALUDE_log_sum_exists_base_l993_99353

theorem log_sum_exists_base : ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, x > 0 → Real.log x / Real.log 2 + Real.log x / Real.log 3 = Real.log x / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_log_sum_exists_base_l993_99353


namespace NUMINAMATH_CALUDE_simplification_proof_l993_99349

theorem simplification_proof (x a : ℝ) :
  (3 * x^2 - 1 - 2*x - 5 + 3*x - x^2 = 2 * x^2 + x - 6) ∧
  (4 * (2 * a^2 - 1 + 2*a) - 3 * (a - 1 + a^2) = 5 * a^2 + 5*a - 1) :=
by sorry

end NUMINAMATH_CALUDE_simplification_proof_l993_99349


namespace NUMINAMATH_CALUDE_mosaic_tiles_l993_99366

/-- Calculates the number of square tiles needed to cover a rectangular area -/
def tilesNeeded (height_feet width_feet tile_side_inches : ℕ) : ℕ :=
  (height_feet * 12 * width_feet * 12) / (tile_side_inches * tile_side_inches)

/-- Theorem stating the number of 1-inch square tiles needed for a 10ft by 15ft mosaic -/
theorem mosaic_tiles : tilesNeeded 10 15 1 = 21600 := by
  sorry

end NUMINAMATH_CALUDE_mosaic_tiles_l993_99366


namespace NUMINAMATH_CALUDE_special_triangle_longest_altitudes_sum_l993_99344

/-- A triangle with sides 8, 15, and 17 -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 8
  hb : b = 15
  hc : c = 17

/-- The sum of the lengths of the two longest altitudes in the special triangle -/
def longestAltitudesSum (t : SpecialTriangle) : ℝ := 23

/-- Theorem stating that the sum of the lengths of the two longest altitudes
    in the special triangle is 23 -/
theorem special_triangle_longest_altitudes_sum (t : SpecialTriangle) :
  longestAltitudesSum t = 23 := by sorry

end NUMINAMATH_CALUDE_special_triangle_longest_altitudes_sum_l993_99344


namespace NUMINAMATH_CALUDE_sams_dimes_given_to_dad_l993_99307

theorem sams_dimes_given_to_dad (initial_dimes : ℕ) (remaining_dimes : ℕ) 
  (h1 : initial_dimes = 9) 
  (h2 : remaining_dimes = 2) : 
  initial_dimes - remaining_dimes = 7 := by
  sorry

end NUMINAMATH_CALUDE_sams_dimes_given_to_dad_l993_99307


namespace NUMINAMATH_CALUDE_min_value_of_expression_l993_99390

theorem min_value_of_expression (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  x*y/z + y*z/x + z*x/y ≥ Real.sqrt 3 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a^2 + b^2 + c^2 = 1 ∧ 
    a*b/c + b*c/a + c*a/b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l993_99390


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l993_99389

theorem rectangular_box_volume (a b c : ℝ) 
  (h1 : a * b = 15) 
  (h2 : b * c = 10) 
  (h3 : c * a = 6) : 
  a * b * c = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l993_99389


namespace NUMINAMATH_CALUDE_ceiling_squared_fraction_l993_99305

theorem ceiling_squared_fraction : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_squared_fraction_l993_99305


namespace NUMINAMATH_CALUDE_last_five_shots_made_l993_99396

/-- Represents the number of shots made in a series of basketball attempts -/
structure ShotsMade where
  total : ℕ
  made : ℕ

/-- Calculates the shooting percentage -/
def shootingPercentage (s : ShotsMade) : ℚ :=
  s.made / s.total

theorem last_five_shots_made
  (initial : ShotsMade)
  (second : ShotsMade)
  (final : ShotsMade)
  (h1 : initial.total = 30)
  (h2 : shootingPercentage initial = 2/5)
  (h3 : second.total = initial.total + 10)
  (h4 : shootingPercentage second = 9/20)
  (h5 : final.total = second.total + 5)
  (h6 : shootingPercentage final = 23/50)
  : final.made - second.made = 2 := by
  sorry

end NUMINAMATH_CALUDE_last_five_shots_made_l993_99396


namespace NUMINAMATH_CALUDE_train_passing_time_l993_99318

theorem train_passing_time 
  (L : ℝ) 
  (v₁ v₂ : ℝ) 
  (h₁ : L > 0) 
  (h₂ : v₁ > 0) 
  (h₃ : v₂ > 0) : 
  (2 * L) / ((v₁ + v₂) * (1000 / 3600)) = 
  (2 * L) / ((v₁ + v₂) * (1000 / 3600)) :=
by sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l993_99318
