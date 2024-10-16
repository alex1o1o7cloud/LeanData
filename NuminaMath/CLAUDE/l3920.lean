import Mathlib

namespace NUMINAMATH_CALUDE_point_on_line_max_product_l3920_392017

/-- Given points A(a,b) and B(4,c) lie on the line y = kx + 3, where k is a constant and k ≠ 0,
    and the maximum value of ab is 9, then c = 2. -/
theorem point_on_line_max_product (k a b c : ℝ) : 
  k ≠ 0 → 
  b = k * a + 3 → 
  c = k * 4 + 3 → 
  (∀ x y, x * y ≤ 9 → a * b ≥ x * y) → 
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_max_product_l3920_392017


namespace NUMINAMATH_CALUDE_product_of_positive_real_part_roots_l3920_392030

theorem product_of_positive_real_part_roots : ∃ (roots : Finset ℂ),
  (∀ z ∈ roots, z^6 = -64) ∧
  (∀ z ∈ roots, (z.re : ℝ) > 0) ∧
  (roots.prod id = 4) := by
sorry

end NUMINAMATH_CALUDE_product_of_positive_real_part_roots_l3920_392030


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3920_392029

theorem sum_of_two_numbers (x y : ℝ) : 
  (0.45 * x = 2700) → (y = 2 * x) → (x + y = 18000) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3920_392029


namespace NUMINAMATH_CALUDE_abs_difference_opposite_signs_l3920_392009

theorem abs_difference_opposite_signs (a b : ℝ) 
  (ha : |a| = 4) 
  (hb : |b| = 2) 
  (hab : a * b < 0) : 
  |a - b| = 6 := by
sorry

end NUMINAMATH_CALUDE_abs_difference_opposite_signs_l3920_392009


namespace NUMINAMATH_CALUDE_multiple_of_smaller_number_l3920_392091

theorem multiple_of_smaller_number 
  (L S m : ℝ) 
  (h1 : L = 33) 
  (h2 : L + S = 51) 
  (h3 : L = m * S - 3) : 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_smaller_number_l3920_392091


namespace NUMINAMATH_CALUDE_a_4k_plus_2_div_3_l3920_392038

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => a (n + 2) + a (n + 1)

theorem a_4k_plus_2_div_3 (k : ℕ) : ∃ m : ℕ, a (4 * k + 2) = 3 * m := by
  sorry

end NUMINAMATH_CALUDE_a_4k_plus_2_div_3_l3920_392038


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3920_392093

theorem solution_set_of_inequality (x : ℝ) :
  {x | x^2 - 2*x + 1 ≤ 0} = {1} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3920_392093


namespace NUMINAMATH_CALUDE_eagle_eye_camera_is_analogical_reasoning_l3920_392073

-- Define the different types of reasoning
inductive ReasoningType
  | Inductive
  | Analogical
  | Deductive

-- Define a structure for a reasoning process
structure ReasoningProcess where
  description : String
  type : ReasoningType

-- Define the four options
def optionA : ReasoningProcess :=
  { description := "People derive that the probability of getting heads when flipping a coin is 1/2 through numerous experiments",
    type := ReasoningType.Inductive }

def optionB : ReasoningProcess :=
  { description := "Scientists invent the eagle eye camera by studying the eyes of eagles",
    type := ReasoningType.Analogical }

def optionC : ReasoningProcess :=
  { description := "Determine the acidity or alkalinity of a solution by testing its pH value",
    type := ReasoningType.Deductive }

def optionD : ReasoningProcess :=
  { description := "Determine whether a function is periodic based on the definition of a periodic function in mathematics",
    type := ReasoningType.Deductive }

-- Theorem to prove
theorem eagle_eye_camera_is_analogical_reasoning :
  optionB.type = ReasoningType.Analogical :=
by sorry

end NUMINAMATH_CALUDE_eagle_eye_camera_is_analogical_reasoning_l3920_392073


namespace NUMINAMATH_CALUDE_lineGraphMostSuitable_l3920_392054

/-- Represents different types of graphs --/
inductive GraphType
  | LineGraph
  | PieChart
  | BarGraph
  | Histogram

/-- Represents the properties of data to be visualized --/
structure DataProperties where
  timeDependent : Bool
  continuous : Bool
  showsTrends : Bool

/-- Determines if a graph type is suitable for given data properties --/
def isSuitable (g : GraphType) (d : DataProperties) : Prop :=
  match g with
  | GraphType.LineGraph => d.timeDependent ∧ d.continuous ∧ d.showsTrends
  | GraphType.PieChart => ¬d.timeDependent
  | GraphType.BarGraph => d.timeDependent
  | GraphType.Histogram => ¬d.timeDependent

/-- The properties of temperature data over a week --/
def temperatureDataProperties : DataProperties :=
  { timeDependent := true
    continuous := true
    showsTrends := true }

/-- Theorem stating that a line graph is the most suitable for temperature data --/
theorem lineGraphMostSuitable :
    ∀ g : GraphType, isSuitable g temperatureDataProperties → g = GraphType.LineGraph :=
  sorry


end NUMINAMATH_CALUDE_lineGraphMostSuitable_l3920_392054


namespace NUMINAMATH_CALUDE_line_bisecting_circle_min_value_l3920_392067

/-- Given a line that bisects a circle, prove the minimum value of a certain expression -/
theorem line_bisecting_circle_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, 2*a*x + b*y - 2 = 0 → x^2 + y^2 - 2*x - 4*y - 6 = 0) →
  (∃ x₀ y₀ : ℝ, 2*a*x₀ + b*y₀ - 2 = 0 ∧ x₀ = 1 ∧ y₀ = 2) →
  (∀ k : ℝ, 2/a + 1/b ≥ k) →
  k = 3 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_bisecting_circle_min_value_l3920_392067


namespace NUMINAMATH_CALUDE_power_of_product_l3920_392008

theorem power_of_product (a : ℝ) : (-4 * a^3)^2 = 16 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3920_392008


namespace NUMINAMATH_CALUDE_camphor_ball_shrinkage_l3920_392070

/-- The time it takes for a camphor ball to shrink to a specific volume -/
theorem camphor_ball_shrinkage (a k : ℝ) (h1 : a > 0) (h2 : k > 0) : 
  let V : ℝ → ℝ := λ t => a * Real.exp (-k * t)
  (V 50 = 4/9 * a) → (V 75 = 8/27 * a) := by
  sorry

end NUMINAMATH_CALUDE_camphor_ball_shrinkage_l3920_392070


namespace NUMINAMATH_CALUDE_number_line_positions_l3920_392016

theorem number_line_positions (x : ℝ) : 
  (x > 0 → (0 = -4*x + 1/4 * (12*x - (-4*x)) ∧ x = 0 + 1/4 * (4*x - 0))) ∧
  (x < 0 → (0 = 12*x + 3/4 * (-4*x - 12*x) ∧ x = 4*x + 3/4 * (0 - 4*x))) :=
by sorry

end NUMINAMATH_CALUDE_number_line_positions_l3920_392016


namespace NUMINAMATH_CALUDE_martha_cards_l3920_392001

/-- The number of cards Martha has at the end of the process -/
def final_cards (initial : ℕ) (multiplier : ℕ) (given_away : ℕ) : ℕ :=
  initial + multiplier * initial - given_away

/-- Theorem stating that Martha ends up with 1479 cards -/
theorem martha_cards : final_cards 423 3 213 = 1479 := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_l3920_392001


namespace NUMINAMATH_CALUDE_top_square_is_five_l3920_392094

/-- Represents a square on the grid --/
structure Square :=
  (number : Nat)
  (row : Nat)
  (col : Nat)

/-- Represents the grid of squares --/
def Grid := List Square

/-- Creates the initial 5x5 grid --/
def initialGrid : Grid :=
  sorry

/-- Performs the first diagonal fold --/
def foldDiagonal (g : Grid) : Grid :=
  sorry

/-- Performs the second fold (bottom half up) --/
def foldBottomUp (g : Grid) : Grid :=
  sorry

/-- Performs the third fold (left half behind) --/
def foldLeftBehind (g : Grid) : Grid :=
  sorry

/-- Returns the top square after all folds --/
def topSquareAfterFolds (g : Grid) : Square :=
  sorry

theorem top_square_is_five :
  let finalGrid := foldLeftBehind (foldBottomUp (foldDiagonal initialGrid))
  (topSquareAfterFolds finalGrid).number = 5 := by
  sorry

end NUMINAMATH_CALUDE_top_square_is_five_l3920_392094


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l3920_392036

theorem fourth_number_in_sequence (a : Fin 6 → ℝ) 
  (h1 : (a 0 + a 1 + a 2 + a 3 + a 4 + a 5) / 6 = 27)
  (h2 : (a 0 + a 1 + a 2 + a 3) / 4 = 23)
  (h3 : (a 3 + a 4 + a 5) / 3 = 34) :
  a 3 = 32 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l3920_392036


namespace NUMINAMATH_CALUDE_original_price_after_discount_l3920_392076

theorem original_price_after_discount (a : ℝ) (h : a > 0) : 
  (4/5 : ℝ) * ((5/4 : ℝ) * a) = a := by sorry

end NUMINAMATH_CALUDE_original_price_after_discount_l3920_392076


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3920_392031

/-- Triangle ABC with side lengths AB = 2, CA = 3, and BC = 4 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB_length : dist A B = 2)
  (CA_length : dist C A = 3)
  (BC_length : dist B C = 4)

/-- A circle tangent to two lines of a triangle -/
structure TangentCircle (T : Triangle) :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (on_BC : center.1 = T.B.1 + (T.C.1 - T.B.1) * (center.2 - T.B.2) / (T.C.2 - T.B.2))
  (tangent_AB : dist center T.A = radius)
  (tangent_AC : dist center T.A = radius)

/-- The radius of the tangent circle is 15/8 -/
theorem tangent_circle_radius (T : Triangle) (C : TangentCircle T) : C.radius = 15/8 :=
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3920_392031


namespace NUMINAMATH_CALUDE_number_puzzle_l3920_392079

theorem number_puzzle : ∃! x : ℝ, (x / 5 + 4 = x / 4 - 4) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3920_392079


namespace NUMINAMATH_CALUDE_cos_difference_formula_l3920_392044

theorem cos_difference_formula (a b : ℝ) 
  (h1 : Real.sin a + Real.sin b = 1)
  (h2 : Real.cos a + Real.cos b = 3/2) : 
  Real.cos (a - b) = 5/8 := by sorry

end NUMINAMATH_CALUDE_cos_difference_formula_l3920_392044


namespace NUMINAMATH_CALUDE_apple_pricing_l3920_392004

/-- The cost of apples per kilogram for the first 30 kgs -/
def l : ℝ := 0.362

/-- The cost of apples per kilogram for each additional kg after 30 kgs -/
def m : ℝ := 0.27

/-- The price of 33 kilograms of apples -/
def price_33kg : ℝ := 11.67

/-- The price of 36 kilograms of apples -/
def price_36kg : ℝ := 12.48

/-- The cost of the first 10 kgs of apples -/
def cost_10kg : ℝ := 3.62

theorem apple_pricing :
  (10 * l = cost_10kg) ∧
  (30 * l + 3 * m = price_33kg) ∧
  (30 * l + 6 * m = price_36kg) →
  m = 0.27 := by
  sorry

end NUMINAMATH_CALUDE_apple_pricing_l3920_392004


namespace NUMINAMATH_CALUDE_locus_of_N_l3920_392095

/-- The locus of point N in an equilateral triangle with a moving point on the unit circle -/
theorem locus_of_N (M N : ℂ) (t : ℝ) : 
  (∀ t, M = Complex.exp (Complex.I * t)) →  -- M is on the unit circle
  (N - 3 = Complex.exp (Complex.I * (5 * Real.pi / 3)) * (M - 3)) →  -- N forms equilateral triangle with A(3,0) and M
  (Complex.abs (N - (3/2 + Complex.I * (3 * Real.sqrt 3 / 2))) = 1) :=  -- Locus of N is a circle
by sorry

end NUMINAMATH_CALUDE_locus_of_N_l3920_392095


namespace NUMINAMATH_CALUDE_equal_roots_condition_l3920_392045

/-- A quadratic equation of the form x(x+1) + ax = 0 has two equal real roots if and only if a = -1 -/
theorem equal_roots_condition (a : ℝ) : 
  (∃ x : ℝ, x * (x + 1) + a * x = 0 ∧ 
   ∀ y : ℝ, y * (y + 1) + a * y = 0 → y = x) ↔ 
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l3920_392045


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l3920_392002

theorem intersection_point_of_lines :
  ∃! p : ℝ × ℝ, 
    2 * p.1 + p.2 - 7 = 0 ∧
    p.1 + 2 * p.2 - 5 = 0 ∧
    p = (3, 1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l3920_392002


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l3920_392099

theorem greatest_three_digit_number : ∃ n : ℕ, 
  n = 978 ∧ 
  n < 1000 ∧ 
  n > 99 ∧
  ∃ k : ℕ, n = 8 * k + 2 ∧
  ∃ m : ℕ, n = 7 * m + 4 ∧
  ∀ x : ℕ, x < 1000 ∧ x > 99 ∧ (∃ a : ℕ, x = 8 * a + 2) ∧ (∃ b : ℕ, x = 7 * b + 4) → x ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l3920_392099


namespace NUMINAMATH_CALUDE_permutations_minus_combinations_l3920_392032

/-- The number of r-permutations from n elements -/
def permutations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of r-combinations from n elements -/
def combinations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem permutations_minus_combinations : permutations 7 3 - combinations 6 4 = 195 := by
  sorry

end NUMINAMATH_CALUDE_permutations_minus_combinations_l3920_392032


namespace NUMINAMATH_CALUDE_final_segment_speed_final_segment_speed_is_90_l3920_392081

/-- Calculates the average speed for the final segment of a journey given specific conditions. -/
theorem final_segment_speed (total_distance : ℝ) (total_time : ℝ) (first_hour_speed : ℝ) 
  (stop_time : ℝ) (second_segment_speed : ℝ) (second_segment_time : ℝ) : ℝ :=
  let net_driving_time := total_time - stop_time / 60
  let first_segment_distance := first_hour_speed * 1
  let second_segment_distance := second_segment_speed * second_segment_time
  let remaining_distance := total_distance - (first_segment_distance + second_segment_distance)
  let remaining_time := net_driving_time - (1 + second_segment_time)
  remaining_distance / remaining_time

/-- Proves that the average speed for the final segment is 90 mph under given conditions. -/
theorem final_segment_speed_is_90 : 
  final_segment_speed 150 3 45 30 50 0.75 = 90 := by
  sorry

end NUMINAMATH_CALUDE_final_segment_speed_final_segment_speed_is_90_l3920_392081


namespace NUMINAMATH_CALUDE_store_refusal_illegal_l3920_392062

/-- Represents a banknote --/
structure Banknote where
  issued_by_bank_of_russia : Bool
  has_tears : Bool

/-- Represents the store's action --/
inductive StoreAction
  | Accept
  | Refuse

/-- Determines if a banknote is legal tender --/
def is_legal_tender (note : Banknote) : Prop :=
  note.issued_by_bank_of_russia ∧ (note.has_tears ∨ ¬note.has_tears)

/-- Determines if the store's action is legal --/
def is_legal_action (note : Banknote) (action : StoreAction) : Prop :=
  is_legal_tender note → action = StoreAction.Accept

/-- The main theorem --/
theorem store_refusal_illegal 
  (lydia_note : Banknote)
  (h1 : lydia_note.has_tears)
  (h2 : lydia_note.issued_by_bank_of_russia)
  (h3 : ∀ (note : Banknote), note.has_tears → is_legal_tender note)
  (store_action : StoreAction)
  (h4 : store_action = StoreAction.Refuse) :
  ¬(is_legal_action lydia_note store_action) :=
by sorry

end NUMINAMATH_CALUDE_store_refusal_illegal_l3920_392062


namespace NUMINAMATH_CALUDE_sequence_sum_l3920_392064

theorem sequence_sum (n : ℕ) (x : ℕ → ℚ) (h1 : x 1 = 1) 
  (h2 : ∀ k ∈ Finset.range (n - 1), x (k + 1) = x k + 1/2) : 
  Finset.sum (Finset.range n) (λ i => x (i + 1)) = (n^2 + 3*n) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3920_392064


namespace NUMINAMATH_CALUDE_card_combinations_l3920_392072

def deck_size : ℕ := 60
def hand_size : ℕ := 15

theorem card_combinations :
  Nat.choose deck_size hand_size = 660665664066 := by
  sorry

end NUMINAMATH_CALUDE_card_combinations_l3920_392072


namespace NUMINAMATH_CALUDE_system_has_no_solution_l3920_392027

theorem system_has_no_solution :
  ¬ (∃ (x₁ x₂ x₃ x₄ : ℝ),
    (5 * x₁ + 12 * x₂ + 19 * x₃ + 25 * x₄ = 25) ∧
    (10 * x₁ + 22 * x₂ + 16 * x₃ + 39 * x₄ = 25) ∧
    (5 * x₁ + 12 * x₂ + 9 * x₃ + 25 * x₄ = 30) ∧
    (20 * x₁ + 46 * x₂ + 34 * x₃ + 89 * x₄ = 70)) :=
by
  sorry


end NUMINAMATH_CALUDE_system_has_no_solution_l3920_392027


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l3920_392003

theorem no_solution_for_equation : ¬∃ (x : ℝ), 1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l3920_392003


namespace NUMINAMATH_CALUDE_children_creativity_center_contradiction_l3920_392055

theorem children_creativity_center_contradiction (N : ℕ) (d : Fin N → ℕ) : 
  N = 32 ∧ 
  (∀ i, d i = 6) ∧ 
  (∀ i j, i ≠ j → d i + d j = 13) → 
  False :=
sorry

end NUMINAMATH_CALUDE_children_creativity_center_contradiction_l3920_392055


namespace NUMINAMATH_CALUDE_max_boat_shipments_l3920_392046

theorem max_boat_shipments (B : ℕ) (h1 : B ≥ 120) (h2 : B % 24 = 0) :
  ∃ S : ℕ, S ≠ 24 ∧ B % S = 0 ∧ ∀ T : ℕ, T ≠ 24 → B % T = 0 → T ≤ S :=
by
  sorry

end NUMINAMATH_CALUDE_max_boat_shipments_l3920_392046


namespace NUMINAMATH_CALUDE_ninth_term_value_l3920_392012

/-- A geometric sequence with a₁ = 2 and a₅ = 18 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ a 5 = 18 ∧ ∀ n m : ℕ, a (n + m) = a n * a m

theorem ninth_term_value (a : ℕ → ℝ) (h : geometric_sequence a) : a 9 = 162 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_value_l3920_392012


namespace NUMINAMATH_CALUDE_associativity_of_mul_l3920_392082

-- Define the set S and its binary operation
variable {S : Type}
variable (add : S → S → S)

-- Define the properties of the set S
variable (h1 : ∀ (a b c : S), add (add a c) (add b c) = add a b)
variable (h2 : ∃ (e : S), (∀ (a : S), add a e = a ∧ add a a = e))

-- Define the * operation
def mul (add : S → S → S) (e : S) (a b : S) : S := add a (add e b)

-- State the theorem
theorem associativity_of_mul 
  (add : S → S → S) 
  (h1 : ∀ (a b c : S), add (add a c) (add b c) = add a b)
  (h2 : ∃ (e : S), (∀ (a : S), add a e = a ∧ add a a = e)) :
  ∀ (a b c : S), mul add (Classical.choose h2) (mul add (Classical.choose h2) a b) c = 
                 mul add (Classical.choose h2) a (mul add (Classical.choose h2) b c) :=
by
  sorry


end NUMINAMATH_CALUDE_associativity_of_mul_l3920_392082


namespace NUMINAMATH_CALUDE_base6_divisibility_l3920_392011

/-- Converts a base 6 number to decimal --/
def base6ToDecimal (a b c d : ℕ) : ℕ :=
  a * 6^3 + b * 6^2 + c * 6 + d

/-- Checks if a number is divisible by 19 --/
def isDivisibleBy19 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 19 * k

theorem base6_divisibility :
  isDivisibleBy19 (base6ToDecimal 4 5 5 2) ∧
  ∀ x : ℕ, x < 5 → ¬isDivisibleBy19 (base6ToDecimal 4 5 x 2) :=
by sorry

end NUMINAMATH_CALUDE_base6_divisibility_l3920_392011


namespace NUMINAMATH_CALUDE_train_length_l3920_392033

/-- The length of a train given its speed and time to cross a bridge -/
theorem train_length (speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  speed = 10 →
  bridge_length = 250 →
  crossing_time = 34.997200223982084 →
  ∃ train_length : ℝ, 
    train_length + bridge_length = speed * crossing_time ∧ 
    abs (train_length - 99.97) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3920_392033


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3920_392090

/-- Given a hyperbola with asymptote equations x ± 2y = 0 and focal length 10,
    prove that its equation is either x²/20 - y²/5 = 1 or y²/5 - x²/20 = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ k : ℝ, k * x + 2 * y = 0 ∧ k * x - 2 * y = 0) →
  (∃ c : ℝ, c^2 = 100) →
  (x^2 / 20 - y^2 / 5 = 1) ∨ (y^2 / 5 - x^2 / 20 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3920_392090


namespace NUMINAMATH_CALUDE_quadratic_form_value_l3920_392060

theorem quadratic_form_value (x y : ℝ) 
  (eq1 : 4 * x + y = 12) 
  (eq2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_value_l3920_392060


namespace NUMINAMATH_CALUDE_teaching_years_difference_l3920_392098

/-- Represents the teaching years of Virginia, Adrienne, and Dennis -/
structure TeachingYears where
  virginia : ℕ
  adrienne : ℕ
  dennis : ℕ

/-- The conditions of the problem -/
def problem_conditions (years : TeachingYears) : Prop :=
  years.virginia + years.adrienne + years.dennis = 75 ∧
  years.virginia = years.adrienne + 9 ∧
  years.dennis = 34

/-- The theorem to be proved -/
theorem teaching_years_difference (years : TeachingYears) 
  (h : problem_conditions years) : 
  years.dennis - years.virginia = 9 := by
  sorry


end NUMINAMATH_CALUDE_teaching_years_difference_l3920_392098


namespace NUMINAMATH_CALUDE_no_solution_composite_l3920_392037

/-- Two polynomials P and Q that satisfy the given conditions -/
class SpecialPolynomials (P Q : ℝ → ℝ) : Prop where
  commutativity : ∀ x : ℝ, P (Q x) = Q (P x)
  no_solution : ∀ x : ℝ, P x ≠ Q x

/-- Theorem stating that if P and Q satisfy the special conditions,
    then P(P(x)) = Q(Q(x)) has no solutions -/
theorem no_solution_composite 
  (P Q : ℝ → ℝ) [SpecialPolynomials P Q] :
  ∀ x : ℝ, P (P x) ≠ Q (Q x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_composite_l3920_392037


namespace NUMINAMATH_CALUDE_bus_fare_difference_sam_alex_l3920_392047

/-- The cost difference between two people's bus fares for a given number of trips -/
def busFareDifference (alexFare samFare : ℚ) (numTrips : ℕ) : ℚ :=
  numTrips * (samFare - alexFare)

/-- Theorem stating the cost difference between Sam and Alex's bus fares for 20 trips -/
theorem bus_fare_difference_sam_alex :
  busFareDifference (25/10) 3 20 = 15 := by sorry

end NUMINAMATH_CALUDE_bus_fare_difference_sam_alex_l3920_392047


namespace NUMINAMATH_CALUDE_hit_probability_theorem_l3920_392077

/-- The probability of hitting a target with one shot -/
def hit_probability : ℚ := 1 / 2

/-- The number of shots taken -/
def total_shots : ℕ := 6

/-- The number of hits required -/
def required_hits : ℕ := 3

/-- The number of consecutive hits required -/
def consecutive_hits : ℕ := 2

/-- The probability of hitting the target 3 times with exactly 2 consecutive hits out of 6 shots -/
def target_probability : ℚ := (Nat.choose 4 2 : ℚ) * (hit_probability ^ total_shots)

theorem hit_probability_theorem : 
  target_probability = (Nat.choose 4 2 : ℚ) * ((1 : ℚ) / 2) ^ 6 := by sorry

end NUMINAMATH_CALUDE_hit_probability_theorem_l3920_392077


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3920_392015

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (2 + Complex.I) / Complex.I
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3920_392015


namespace NUMINAMATH_CALUDE_digit_interchange_effect_l3920_392049

theorem digit_interchange_effect (n : ℕ) (p q : ℕ) 
  (h1 : n = 9)
  (h2 : p < 10 ∧ q < 10)
  (h3 : p ≠ q)
  (original_sum : ℕ) 
  (new_sum : ℕ)
  (h4 : new_sum = original_sum - n)
  (h5 : new_sum = original_sum - (10*p + q - (10*q + p))) :
  p - q = 1 ∨ q - p = 1 :=
sorry

end NUMINAMATH_CALUDE_digit_interchange_effect_l3920_392049


namespace NUMINAMATH_CALUDE_no_prime_base_n_representation_l3920_392025

def base_n_representation (n : ℕ) : ℕ := n^4 + n^2 + 1

theorem no_prime_base_n_representation :
  ∀ n : ℕ, n ≥ 2 → ¬(Nat.Prime (base_n_representation n)) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_base_n_representation_l3920_392025


namespace NUMINAMATH_CALUDE_points_three_units_from_negative_one_l3920_392041

theorem points_three_units_from_negative_one : 
  {x : ℝ | |x - (-1)| = 3} = {2, -4} := by sorry

end NUMINAMATH_CALUDE_points_three_units_from_negative_one_l3920_392041


namespace NUMINAMATH_CALUDE_waddle_hop_difference_l3920_392013

/-- The number of hops Winston takes between consecutive markers -/
def winston_hops : ℕ := 88

/-- The number of waddles Petra takes between consecutive markers -/
def petra_waddles : ℕ := 24

/-- The total number of markers -/
def total_markers : ℕ := 81

/-- The total distance in feet between the first and last marker -/
def total_distance : ℕ := 10560

/-- The length of Petra's waddle in feet -/
def petra_waddle_length : ℚ := total_distance / (petra_waddles * (total_markers - 1))

/-- The length of Winston's hop in feet -/
def winston_hop_length : ℚ := total_distance / (winston_hops * (total_markers - 1))

/-- The difference between Petra's waddle length and Winston's hop length -/
def length_difference : ℚ := petra_waddle_length - winston_hop_length

theorem waddle_hop_difference : length_difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_waddle_hop_difference_l3920_392013


namespace NUMINAMATH_CALUDE_distance_difference_l3920_392084

theorem distance_difference (john_distance nina_distance : ℝ) 
  (h1 : john_distance = 0.7)
  (h2 : nina_distance = 0.4) :
  john_distance - nina_distance = 0.3 := by
sorry

end NUMINAMATH_CALUDE_distance_difference_l3920_392084


namespace NUMINAMATH_CALUDE_quaternary_201_is_33_l3920_392006

def quaternary_to_decimal (q : ℕ) : ℕ :=
  (q / 100) * 4^2 + ((q / 10) % 10) * 4^1 + (q % 10) * 4^0

theorem quaternary_201_is_33 : quaternary_to_decimal 201 = 33 := by
  sorry

end NUMINAMATH_CALUDE_quaternary_201_is_33_l3920_392006


namespace NUMINAMATH_CALUDE_color_tv_price_l3920_392007

theorem color_tv_price : ∃ (x : ℝ), x > 0 ∧ (1.4 * x * 0.8) - x = 144 ∧ x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_color_tv_price_l3920_392007


namespace NUMINAMATH_CALUDE_ring_toss_earnings_l3920_392061

theorem ring_toss_earnings (daily_earnings : ℕ) (days : ℕ) (total_earnings : ℕ) : 
  daily_earnings = 144 → days = 22 → total_earnings = daily_earnings * days → total_earnings = 3168 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_earnings_l3920_392061


namespace NUMINAMATH_CALUDE_budget_calculation_l3920_392080

theorem budget_calculation (initial_budget : ℕ) 
  (shirt_cost pants_cost coat_cost socks_cost belt_cost shoes_cost : ℕ) :
  initial_budget = 200 →
  shirt_cost = 30 →
  pants_cost = 46 →
  coat_cost = 38 →
  socks_cost = 11 →
  belt_cost = 18 →
  shoes_cost = 41 →
  initial_budget - (shirt_cost + pants_cost + coat_cost + socks_cost + belt_cost + shoes_cost) = 16 := by
  sorry

end NUMINAMATH_CALUDE_budget_calculation_l3920_392080


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3920_392065

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1)) / (a * b * c) ≥ 216 :=
by
  sorry

theorem min_value_achieved (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    ((x^2 + 4*x + 1) * (y^2 + 4*y + 1) * (z^2 + 4*z + 1)) / (x * y * z) = 216 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3920_392065


namespace NUMINAMATH_CALUDE_cosine_sine_inequality_l3920_392021

theorem cosine_sine_inequality (x : ℝ) : 
  (Real.cos (x / 2) + Real.sin (x / 2) ≤ (Real.sin x - 3) / Real.sqrt 2) ↔ 
  ∃ k : ℤ, x = -3 * Real.pi / 2 + 4 * Real.pi * ↑k :=
sorry

end NUMINAMATH_CALUDE_cosine_sine_inequality_l3920_392021


namespace NUMINAMATH_CALUDE_min_value_theorem_l3920_392014

theorem min_value_theorem (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : x + 2*y = 1) :
  (2/x + 3/y) ≥ 8 + 4*Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3920_392014


namespace NUMINAMATH_CALUDE_sector_area_l3920_392075

/-- Given a circular sector with central angle 2 radians and arc length 4, its area is 4. -/
theorem sector_area (θ : ℝ) (l : ℝ) (r : ℝ) (h1 : θ = 2) (h2 : l = 4) (h3 : l = r * θ) :
  (1 / 2) * r^2 * θ = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3920_392075


namespace NUMINAMATH_CALUDE_arrangement_count_l3920_392048

theorem arrangement_count (volunteers : ℕ) (elderly : ℕ) : 
  volunteers = 4 ∧ elderly = 1 → (volunteers.factorial : ℕ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l3920_392048


namespace NUMINAMATH_CALUDE_initial_car_cost_is_24000_l3920_392042

/-- Represents the financial details of John's Uber driving and car ownership -/
structure UberFinances where
  earnings : ℕ  -- Earnings from Uber before depreciation
  tradeIn : ℕ   -- Trade-in value of the car
  profit : ℕ    -- Profit from driving Uber

/-- Calculates the initial cost of the car based on Uber finances -/
def initialCarCost (f : UberFinances) : ℕ :=
  f.profit + f.tradeIn

/-- Theorem stating that the initial car cost is $24,000 given John's financial details -/
theorem initial_car_cost_is_24000 (f : UberFinances) 
  (h1 : f.earnings = 30000)
  (h2 : f.tradeIn = 6000)
  (h3 : f.profit = 18000) : 
  initialCarCost f = 24000 := by
  sorry

#eval initialCarCost ⟨30000, 6000, 18000⟩

end NUMINAMATH_CALUDE_initial_car_cost_is_24000_l3920_392042


namespace NUMINAMATH_CALUDE_tan_330_degrees_l3920_392053

theorem tan_330_degrees : Real.tan (330 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_330_degrees_l3920_392053


namespace NUMINAMATH_CALUDE_sum_of_y_coordinates_l3920_392020

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Predicate to check if two points are opposite vertices of a rectangle --/
def are_opposite_vertices (r : Rectangle) (p1 p2 : ℝ × ℝ) : Prop :=
  (r.v1 = p1 ∧ r.v3 = p2) ∨ (r.v1 = p2 ∧ r.v3 = p1) ∨
  (r.v2 = p1 ∧ r.v4 = p2) ∨ (r.v2 = p2 ∧ r.v4 = p1)

/-- Theorem: Sum of y-coordinates of two vertices given the other two --/
theorem sum_of_y_coordinates (r : Rectangle) :
  are_opposite_vertices r (4, 20) (12, -6) →
  (r.v1.2 + r.v2.2 + r.v3.2 + r.v4.2) = 14 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_y_coordinates_l3920_392020


namespace NUMINAMATH_CALUDE_average_marks_l3920_392026

theorem average_marks (n : ℕ) (avg_five : ℝ) (sixth_mark : ℝ) (h1 : n = 6) (h2 : avg_five = 74) (h3 : sixth_mark = 86) :
  ((n - 1) * avg_five + sixth_mark) / n = 76 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_l3920_392026


namespace NUMINAMATH_CALUDE_middle_number_between_52_and_certain_number_l3920_392059

theorem middle_number_between_52_and_certain_number 
  (certain_number : ℕ) 
  (h1 : certain_number > 52) 
  (h2 : ∃ (n : ℕ), n ≥ 52 ∧ n < certain_number ∧ certain_number - 52 - 1 = 15) :
  (52 + certain_number) / 2 = 60 :=
sorry

end NUMINAMATH_CALUDE_middle_number_between_52_and_certain_number_l3920_392059


namespace NUMINAMATH_CALUDE_add_neg_two_three_l3920_392023

theorem add_neg_two_three : -2 + 3 = 1 := by sorry

end NUMINAMATH_CALUDE_add_neg_two_three_l3920_392023


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3920_392069

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 7 players, where each player plays twice with every other player, the total number of games played is 84 -/
theorem chess_tournament_games :
  tournament_games 7 * 2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3920_392069


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l3920_392039

/-- Given an initial angle of 30 degrees rotated 450 degrees clockwise,
    the resulting new acute angle measures 60 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 30 →
  rotation = 450 →
  let effective_rotation := rotation % 360
  let new_angle := (initial_angle - effective_rotation) % 360
  let acute_angle := min new_angle (360 - new_angle)
  acute_angle = 60 := by
  sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l3920_392039


namespace NUMINAMATH_CALUDE_work_completion_time_l3920_392092

/-- Given that 72 men can complete a piece of work in 18 days, and the number of men and days are inversely proportional, prove that 144 men will complete the same work in 9 days. -/
theorem work_completion_time 
  (men : ℕ → ℝ)
  (days : ℕ → ℝ)
  (h1 : men 1 = 72)
  (h2 : days 1 = 18)
  (h3 : ∀ k : ℕ, k > 0 → men k * days k = men 1 * days 1) :
  men 2 = 144 ∧ days 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3920_392092


namespace NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l3920_392005

-- Part 1: System of Equations
theorem solve_system_equations (x y : ℝ) :
  (3 * (y - 2) = x - 1) ∧ (2 * (x - 1) = 5 * y - 8) →
  x = 7 ∧ y = 4 := by sorry

-- Part 2: System of Inequalities
theorem solve_system_inequalities (x : ℝ) :
  (3 * x ≤ 2 * x + 3) ∧ ((x + 1) / 6 - 1 < (2 * x + 2) / 3) ↔
  -3 < x ∧ x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l3920_392005


namespace NUMINAMATH_CALUDE_nested_function_evaluation_l3920_392024

def a (k : ℕ) : ℕ := (k + 1)^2

theorem nested_function_evaluation :
  let k : ℕ := 1
  a (a (a (a k))) = 458329 := by
  sorry

end NUMINAMATH_CALUDE_nested_function_evaluation_l3920_392024


namespace NUMINAMATH_CALUDE_floor_length_is_sqrt_150_l3920_392068

/-- Represents a rectangular floor with specific properties -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  total_paint_cost : ℝ
  paint_rate_per_sqm : ℝ

/-- The length is 200% more than the breadth -/
def length_breadth_relation (floor : RectangularFloor) : Prop :=
  floor.length = 3 * floor.breadth

/-- The total paint cost divided by the rate per sqm gives the area -/
def area_from_paint_cost (floor : RectangularFloor) : Prop :=
  floor.total_paint_cost / floor.paint_rate_per_sqm = floor.length * floor.breadth

/-- Theorem stating the length of the floor -/
theorem floor_length_is_sqrt_150 (floor : RectangularFloor) 
  (h1 : length_breadth_relation floor)
  (h2 : area_from_paint_cost floor)
  (h3 : floor.total_paint_cost = 100)
  (h4 : floor.paint_rate_per_sqm = 2) : 
  floor.length = Real.sqrt 150 := by
  sorry

end NUMINAMATH_CALUDE_floor_length_is_sqrt_150_l3920_392068


namespace NUMINAMATH_CALUDE_average_weight_increase_l3920_392083

/-- Proves that replacing a 70 kg person with a 110 kg person in a group of 10 increases the average weight by 4 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 10 * initial_average
  let new_total := initial_total - 70 + 110
  let new_average := new_total / 10
  new_average - initial_average = 4 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3920_392083


namespace NUMINAMATH_CALUDE_johnnys_hourly_wage_l3920_392088

/-- Johnny's hourly wage calculation --/
theorem johnnys_hourly_wage :
  let total_earned : ℚ := 11.75
  let hours_worked : ℕ := 5
  let hourly_wage : ℚ := total_earned / hours_worked
  hourly_wage = 2.35 := by
  sorry

end NUMINAMATH_CALUDE_johnnys_hourly_wage_l3920_392088


namespace NUMINAMATH_CALUDE_tan_alpha_values_l3920_392052

theorem tan_alpha_values (α : ℝ) (h : Real.sin (2 * α) = -Real.sin α) : 
  Real.tan α = 0 ∨ Real.tan α = Real.sqrt 3 ∨ Real.tan α = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_alpha_values_l3920_392052


namespace NUMINAMATH_CALUDE_some_number_value_l3920_392089

theorem some_number_value (some_number : ℝ) : 
  (3.242 * 10) / some_number = 0.032420000000000004 → some_number = 1000 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l3920_392089


namespace NUMINAMATH_CALUDE_amp_six_three_l3920_392097

/-- The & operation defined on two real numbers -/
def amp (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- Theorem stating that 6 & 3 = 27 -/
theorem amp_six_three : amp 6 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_amp_six_three_l3920_392097


namespace NUMINAMATH_CALUDE_g_of_2_l3920_392086

/-- Given functions f and g, prove the value of g(2) -/
theorem g_of_2 (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 2 * x^2 + 4 * x - 6)
  (hg : ∀ x, g (f x) = 3 * x^3 + 2 * x - 5) :
  g 2 = 3 * (-1 + Real.sqrt 5)^3 + 2 * (-1 + Real.sqrt 5) - 5 := by
sorry

end NUMINAMATH_CALUDE_g_of_2_l3920_392086


namespace NUMINAMATH_CALUDE_problem_solution_l3920_392019

theorem problem_solution (p q : ℝ) 
  (h1 : 3 / p = 6)
  (h2 : 3 / q = 18)
  (h3 : abs (p - q - 0.33333333333333337) < 1e-14) :
  p = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3920_392019


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3920_392051

theorem unique_triple_solution :
  ∃! (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x^2 + y^2 + z^2 = 3 ∧
  (x + y + z) * (x^2 + y^2 + z^2) = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3920_392051


namespace NUMINAMATH_CALUDE_ship_arrangement_count_l3920_392035

/-- The number of ways to select and arrange ships for tasks -/
def arrange_ships (destroyers frigates selected : ℕ) (tasks : ℕ) : ℕ :=
  (Nat.choose (destroyers + frigates) selected - Nat.choose frigates selected) * Nat.factorial tasks

/-- Theorem stating the correct number of arrangements -/
theorem ship_arrangement_count :
  arrange_ships 2 6 3 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_ship_arrangement_count_l3920_392035


namespace NUMINAMATH_CALUDE_miss_spelling_paper_sheets_l3920_392071

theorem miss_spelling_paper_sheets : ∃ (total_sheets : ℕ) (num_pupils : ℕ),
  total_sheets = 3 * num_pupils + 31 ∧
  total_sheets = 4 * num_pupils + 8 ∧
  total_sheets = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_miss_spelling_paper_sheets_l3920_392071


namespace NUMINAMATH_CALUDE_max_sum_cyclic_fraction_l3920_392085

open Real BigOperators

/-- The maximum value of the sum for positive real numbers with sum 1 -/
theorem max_sum_cyclic_fraction (n : ℕ) (a : ℕ → ℝ) 
  (hn : n ≥ 4)
  (ha_pos : ∀ k, a k > 0)
  (ha_sum : ∑ k in Finset.range n, a k = 1) :
  (∑ k in Finset.range n, (a k)^2 / (a k + a ((k + 1) % n) + a ((k + 2) % n))) ≤ 1/3 :=
sorry


end NUMINAMATH_CALUDE_max_sum_cyclic_fraction_l3920_392085


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l3920_392040

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l3920_392040


namespace NUMINAMATH_CALUDE_grant_total_earnings_l3920_392010

/-- The total amount Grant made from selling his baseball gear -/
def total_amount : ℝ :=
  let baseball_cards := 25
  let baseball_bat := 10
  let baseball_glove_original := 30
  let baseball_glove_discount := 0.20
  let baseball_glove := baseball_glove_original * (1 - baseball_glove_discount)
  let baseball_cleats := 10
  let num_cleats := 2
  baseball_cards + baseball_bat + baseball_glove + (baseball_cleats * num_cleats)

/-- Theorem stating that the total amount Grant made is $79 -/
theorem grant_total_earnings : total_amount = 79 := by
  sorry

end NUMINAMATH_CALUDE_grant_total_earnings_l3920_392010


namespace NUMINAMATH_CALUDE_mixed_fruit_juice_cost_l3920_392096

/-- The cost per litre of the superfruit juice cocktail -/
def cocktail_cost_per_litre : ℝ := 1399.45

/-- The cost per litre of açaí berry juice -/
def acai_cost_per_litre : ℝ := 3104.35

/-- The volume of mixed fruit juice used -/
def mixed_fruit_volume : ℝ := 32

/-- The volume of açaí berry juice used -/
def acai_volume : ℝ := 21.333333333333332

/-- The cost per litre of mixed fruit juice -/
def mixed_fruit_cost_per_litre : ℝ := 262.8125

theorem mixed_fruit_juice_cost : 
  cocktail_cost_per_litre * (mixed_fruit_volume + acai_volume) = 
  mixed_fruit_cost_per_litre * mixed_fruit_volume + acai_cost_per_litre * acai_volume := by
  sorry

end NUMINAMATH_CALUDE_mixed_fruit_juice_cost_l3920_392096


namespace NUMINAMATH_CALUDE_average_monthly_sales_l3920_392022

def january_sales : ℝ := 150
def february_sales : ℝ := 90
def march_sales : ℝ := 60
def april_sales : ℝ := 140
def may_sales_before_discount : ℝ := 100
def discount_rate : ℝ := 0.2

def may_sales : ℝ := may_sales_before_discount * (1 - discount_rate)

def total_sales : ℝ := january_sales + february_sales + march_sales + april_sales + may_sales

def number_of_months : ℕ := 5

theorem average_monthly_sales :
  total_sales / number_of_months = 104 := by sorry

end NUMINAMATH_CALUDE_average_monthly_sales_l3920_392022


namespace NUMINAMATH_CALUDE_quilt_patch_cost_l3920_392043

/-- The total cost of patches for a quilt with given dimensions and pricing structure -/
theorem quilt_patch_cost (quilt_length quilt_width patch_area : ℕ)
  (first_batch_size first_batch_price : ℕ) : 
  quilt_length = 16 →
  quilt_width = 20 →
  patch_area = 4 →
  first_batch_size = 10 →
  first_batch_price = 10 →
  (quilt_length * quilt_width) % patch_area = 0 →
  (first_batch_size * first_batch_price) + 
  ((quilt_length * quilt_width / patch_area - first_batch_size) * (first_batch_price / 2)) = 450 :=
by sorry

end NUMINAMATH_CALUDE_quilt_patch_cost_l3920_392043


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3920_392034

theorem quadratic_root_property (a : ℝ) : 
  a^2 + 2*a - 3 = 0 → 2*a^2 + 4*a = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3920_392034


namespace NUMINAMATH_CALUDE_prob_non_red_is_half_l3920_392028

-- Define the die
def total_faces : ℕ := 10
def red_faces : ℕ := 5
def yellow_faces : ℕ := 3
def blue_faces : ℕ := 1
def green_faces : ℕ := 1

-- Define the probability of rolling a non-red face
def prob_non_red : ℚ := (yellow_faces + blue_faces + green_faces : ℚ) / total_faces

-- Theorem statement
theorem prob_non_red_is_half : prob_non_red = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_non_red_is_half_l3920_392028


namespace NUMINAMATH_CALUDE_fabric_cost_per_yard_l3920_392018

theorem fabric_cost_per_yard 
  (total_spent : ℝ) 
  (total_yards : ℝ) 
  (h1 : total_spent = 120) 
  (h2 : total_yards = 16) : 
  total_spent / total_yards = 7.50 := by
sorry

end NUMINAMATH_CALUDE_fabric_cost_per_yard_l3920_392018


namespace NUMINAMATH_CALUDE_square_2004_content_l3920_392056

/-- Represents the content of a square in the sequence -/
inductive SquareContent
  | A
  | AB
  | ABCD
  | Number (n : ℕ)

/-- Returns the letter content of the nth square -/
def letterContent (n : ℕ) : SquareContent :=
  match n % 3 with
  | 0 => SquareContent.ABCD
  | 1 => SquareContent.A
  | 2 => SquareContent.AB
  | _ => SquareContent.A  -- This case is mathematically impossible, but needed for completeness

/-- Returns the number content of the nth square -/
def numberContent (n : ℕ) : SquareContent :=
  SquareContent.Number n

/-- Combines letter and number content for the nth square -/
def squareContent (n : ℕ) : (SquareContent × SquareContent) :=
  (letterContent n, numberContent n)

/-- The main theorem to prove -/
theorem square_2004_content :
  squareContent 2004 = (SquareContent.ABCD, SquareContent.Number 2004) := by
  sorry


end NUMINAMATH_CALUDE_square_2004_content_l3920_392056


namespace NUMINAMATH_CALUDE_income_comparison_percentage_difference_l3920_392058

theorem income_comparison (juan tim person : ℝ) 
  (h1 : tim = 0.5 * juan)
  (h2 : person = 0.8 * juan) : 
  person = 1.6 * tim := by
sorry

theorem percentage_difference (juan tim person : ℝ) 
  (h1 : tim = 0.5 * juan)
  (h2 : person = 0.8 * juan) : 
  (person - tim) / tim * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_percentage_difference_l3920_392058


namespace NUMINAMATH_CALUDE_luke_total_score_l3920_392000

def total_points (points_per_round : ℕ) (num_rounds : ℕ) : ℕ :=
  points_per_round * num_rounds

theorem luke_total_score :
  let points_per_round : ℕ := 42
  let num_rounds : ℕ := 2
  total_points points_per_round num_rounds = 84 := by sorry

end NUMINAMATH_CALUDE_luke_total_score_l3920_392000


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l3920_392074

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_alcohol : ℝ) :
  initial_volume = 6 →
  initial_percentage = 0.3 →
  added_alcohol = 2.4 →
  let final_volume := initial_volume + added_alcohol
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  let final_percentage := final_alcohol / final_volume
  final_percentage = 0.5 := by sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l3920_392074


namespace NUMINAMATH_CALUDE_pie_chart_probability_l3920_392063

theorem pie_chart_probability (W X Y Z : ℝ) : 
  W = 1/4 → X = 1/3 → Z = 1/6 → W + X + Y + Z = 1 → Y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_probability_l3920_392063


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3920_392078

theorem right_triangle_segment_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  r * s = a^2 →      -- Geometric mean theorem
  r + s = c →        -- r and s are segments of c
  a / b = 2 / 5 →    -- Given ratio of a to b
  r / s = 4 / 25 :=  -- Conclusion: ratio of r to s
by sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3920_392078


namespace NUMINAMATH_CALUDE_remainder_problem_l3920_392066

theorem remainder_problem (n : ℤ) (h : n ≡ 3 [ZMOD 4]) : 7 * n ≡ 1 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3920_392066


namespace NUMINAMATH_CALUDE_sarahs_reading_capacity_l3920_392050

/-- Sarah's reading problem -/
theorem sarahs_reading_capacity 
  (pages_per_hour : ℕ) 
  (pages_per_book : ℕ) 
  (available_hours : ℕ) 
  (h1 : pages_per_hour = 120) 
  (h2 : pages_per_book = 360) 
  (h3 : available_hours = 8) :
  (available_hours * pages_per_hour) / pages_per_book = 2 :=
sorry

end NUMINAMATH_CALUDE_sarahs_reading_capacity_l3920_392050


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l3920_392057

/-- Proves that the percentage of invalid votes is 20% in an election with given conditions -/
theorem invalid_votes_percentage (total_votes : ℕ) (valid_votes : ℕ) (winner_percentage : ℚ) (loser_votes : ℕ) 
  (h1 : total_votes = 5500)
  (h2 : winner_percentage = 55 / 100)
  (h3 : loser_votes = 1980)
  (h4 : valid_votes * winner_percentage = valid_votes - loser_votes) :
  (total_votes - valid_votes) / total_votes = 1 / 5 := by
  sorry

#eval (5500 : ℚ) * (1 / 5) -- Should evaluate to 1100, which is the number of invalid votes

end NUMINAMATH_CALUDE_invalid_votes_percentage_l3920_392057


namespace NUMINAMATH_CALUDE_complex_modulus_inequality_l3920_392087

open Complex

theorem complex_modulus_inequality (z : ℂ) (h : abs z = 1) :
  abs ((z + 1) + Complex.I * (7 - z)) ≠ 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_inequality_l3920_392087
