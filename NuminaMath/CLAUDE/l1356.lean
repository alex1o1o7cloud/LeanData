import Mathlib

namespace solution_range_l1356_135652

-- Define the system of inequalities
def has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, a * x > -1 ∧ x + a > 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a < -1 ∨ a ≥ 0

-- Theorem statement
theorem solution_range :
  ∀ a : ℝ, has_solution a ↔ a_range a := by sorry

end solution_range_l1356_135652


namespace inverse_function_problem_l1356_135654

theorem inverse_function_problem (g : ℝ → ℝ) (g_inv : ℝ → ℝ) 
  (h1 : Function.LeftInverse g_inv g) 
  (h2 : Function.RightInverse g_inv g)
  (h3 : g 4 = 6)
  (h4 : g 6 = 2)
  (h5 : g 3 = 7) :
  g_inv (g_inv 7 + g_inv 6) = 3 := by
  sorry

end inverse_function_problem_l1356_135654


namespace hoseok_calculation_l1356_135635

theorem hoseok_calculation : ∃ x : ℤ, (x - 7 = 9) ∧ (3 * x = 48) := by
  sorry

end hoseok_calculation_l1356_135635


namespace subset_implies_a_equals_three_l1356_135630

theorem subset_implies_a_equals_three (A B : Set ℕ) (a : ℕ) 
  (h1 : A = {1, 3})
  (h2 : B = {1, 2, a})
  (h3 : A ⊆ B) : 
  a = 3 := by
  sorry

end subset_implies_a_equals_three_l1356_135630


namespace cars_sold_first_three_days_l1356_135600

/-- Proves that the number of cars sold each day for the first three days is 5 --/
theorem cars_sold_first_three_days :
  let total_quota : ℕ := 50
  let cars_sold_next_four_days : ℕ := 3 * 4
  let remaining_cars_to_sell : ℕ := 23
  let cars_per_day_first_three_days : ℕ := (total_quota - cars_sold_next_four_days - remaining_cars_to_sell) / 3
  cars_per_day_first_three_days = 5 := by
  sorry

#eval (50 - 3 * 4 - 23) / 3

end cars_sold_first_three_days_l1356_135600


namespace average_weight_increase_l1356_135660

/-- Proves that replacing a person weighing 68 kg with a person weighing 95.5 kg
    in a group of 5 people increases the average weight by 5.5 kg -/
theorem average_weight_increase (initial_average : ℝ) :
  let initial_total := 5 * initial_average
  let new_total := initial_total - 68 + 95.5
  let new_average := new_total / 5
  new_average - initial_average = 5.5 := by
sorry

end average_weight_increase_l1356_135660


namespace parallelogram_split_slope_l1356_135697

/-- A parallelogram in a 2D plane --/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- A line in a 2D plane represented by its slope --/
structure Line where
  slope : ℝ

/-- Predicate to check if a line passes through the origin and splits a parallelogram into two congruent parts --/
def splits_congruently (p : Parallelogram) (l : Line) : Prop :=
  sorry

/-- The main theorem --/
theorem parallelogram_split_slope :
  let p := Parallelogram.mk (8, 50) (8, 120) (30, 160) (30, 90)
  let l := Line.mk (265 / 38)
  splits_congruently p l := by sorry

end parallelogram_split_slope_l1356_135697


namespace danielles_apartment_rooms_l1356_135627

theorem danielles_apartment_rooms (heidi_rooms danielle_rooms grant_rooms : ℕ) : 
  heidi_rooms = 3 * danielle_rooms →
  grant_rooms = heidi_rooms / 9 →
  grant_rooms = 2 →
  danielle_rooms = 6 := by
sorry

end danielles_apartment_rooms_l1356_135627


namespace wholesale_price_calculation_l1356_135672

/-- The wholesale price of a pair of pants -/
def wholesale_price : ℝ := 20

/-- The retail price of a pair of pants -/
def retail_price : ℝ := 36

/-- The markup percentage as a decimal -/
def markup : ℝ := 0.8

theorem wholesale_price_calculation :
  wholesale_price = retail_price / (1 + markup) :=
by sorry

end wholesale_price_calculation_l1356_135672


namespace vending_machine_failure_rate_l1356_135631

/-- Calculates the failure rate of a vending machine. -/
theorem vending_machine_failure_rate 
  (total_users : ℕ) 
  (snacks_dropped : ℕ) 
  (extra_snack_rate : ℚ) : 
  total_users = 30 → 
  snacks_dropped = 28 → 
  extra_snack_rate = 1/10 → 
  (total_users : ℚ) - snacks_dropped = 
    total_users * (1 - extra_snack_rate) * (1/6 : ℚ) := by
  sorry

end vending_machine_failure_rate_l1356_135631


namespace multiply_by_15_subtract_1_l1356_135655

theorem multiply_by_15_subtract_1 (x : ℝ) : 15 * x = 45 → x - 1 = 2 := by
  sorry

end multiply_by_15_subtract_1_l1356_135655


namespace probability_green_second_is_three_fifths_l1356_135658

/-- Represents the contents of a bag of marbles -/
structure BagContents where
  white : ℕ := 0
  black : ℕ := 0
  red : ℕ := 0
  green : ℕ := 0

/-- Calculate the probability of drawing a green marble as the second marble -/
def probabilityGreenSecond (bagX bagY bagZ : BagContents) : ℚ :=
  let probWhiteX := bagX.white / (bagX.white + bagX.black)
  let probBlackX := bagX.black / (bagX.white + bagX.black)
  let probGreenY := bagY.green / (bagY.red + bagY.green)
  let probGreenZ := bagZ.green / (bagZ.red + bagZ.green)
  probWhiteX * probGreenY + probBlackX * probGreenZ

/-- The main theorem to prove -/
theorem probability_green_second_is_three_fifths :
  let bagX := BagContents.mk 5 5 0 0
  let bagY := BagContents.mk 0 0 7 8
  let bagZ := BagContents.mk 0 0 3 6
  probabilityGreenSecond bagX bagY bagZ = 3/5 := by
  sorry


end probability_green_second_is_three_fifths_l1356_135658


namespace r_fourth_plus_inverse_r_fourth_l1356_135656

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^4 + 1/r^4 = 7 := by
sorry

end r_fourth_plus_inverse_r_fourth_l1356_135656


namespace complex_number_equality_l1356_135669

theorem complex_number_equality (z : ℂ) : (z - 2) * Complex.I = 1 + Complex.I → z = 3 - Complex.I := by
  sorry

end complex_number_equality_l1356_135669


namespace y_value_when_x_is_8_l1356_135673

theorem y_value_when_x_is_8 (k : ℝ) :
  (∀ x, (x : ℝ) > 0 → k * x^(1/3) = 3 * Real.sqrt 2 → x = 64) →
  k * 8^(1/3) = (3 * Real.sqrt 2) / 2 := by
  sorry

end y_value_when_x_is_8_l1356_135673


namespace distance_focus_to_asymptote_l1356_135683

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (5, 0)

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := 3 * y = 4 * x

-- Theorem statement
theorem distance_focus_to_asymptote :
  let F := right_focus
  ∃ (d : ℝ), d = 4 ∧
  ∀ (x y : ℝ), asymptote x y →
    (F.1 - x)^2 + (F.2 - y)^2 = d^2 :=
sorry

end distance_focus_to_asymptote_l1356_135683


namespace sum_of_two_numbers_l1356_135671

theorem sum_of_two_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : y = 18.5) : x + y = 46 := by
  sorry

end sum_of_two_numbers_l1356_135671


namespace zeros_in_Q_l1356_135605

def R (k : ℕ+) : ℕ := (10^k.val - 1) / 9

def Q : ℕ := R 30 / R 6

def count_zeros (n : ℕ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 30 := by sorry

end zeros_in_Q_l1356_135605


namespace sum_and_count_theorem_l1356_135602

def sum_of_range (a b : ℕ) : ℕ := 
  ((b - a + 1) * (a + b)) / 2

def count_even_in_range (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_theorem : 
  let x := sum_of_range 20 30
  let y := count_even_in_range 20 30
  x + y = 281 := by sorry

end sum_and_count_theorem_l1356_135602


namespace marbles_remainder_l1356_135622

theorem marbles_remainder (r p : ℕ) 
  (h1 : r % 8 = 5) 
  (h2 : p % 8 = 7) 
  (h3 : (r + p) % 10 = 0) :
  (r + p) % 8 = 4 := by sorry

end marbles_remainder_l1356_135622


namespace problem_solution_l1356_135616

theorem problem_solution :
  (999 * (-13) = -12987) ∧
  (999 * 118 * (4/5) + 333 * (-3/5) - 999 * 18 * (3/5) = 99900) ∧
  (6 / (-1/2 + 1/3) = -36) := by
sorry

end problem_solution_l1356_135616


namespace standard_deviation_proof_l1356_135650

/-- The average age of job applicants -/
def average_age : ℝ := 31

/-- The number of different ages in the acceptable range -/
def different_ages : ℕ := 19

/-- The standard deviation of applicants' ages -/
def standard_deviation : ℝ := 9

/-- Theorem stating that the standard deviation is correct given the problem conditions -/
theorem standard_deviation_proof : 
  (average_age + standard_deviation) - (average_age - standard_deviation) = different_ages - 1 := by
  sorry

end standard_deviation_proof_l1356_135650


namespace expression_value_l1356_135670

theorem expression_value (x y : ℝ) (h1 : x ≠ y) 
  (h2 : 1 / (1 + x^2) + 1 / (1 + y^2) = 2 / (1 + x*y)) : 
  1 / (1 + x^2) + 1 / (1 + y^2) + 2 / (1 + x*y) = 2 := by
  sorry

end expression_value_l1356_135670


namespace total_games_calculation_l1356_135636

/-- The number of baseball games played at night -/
def night_games : ℕ := 128

/-- The number of games Joan attended -/
def attended_games : ℕ := 395

/-- The number of games Joan missed -/
def missed_games : ℕ := 469

/-- The total number of baseball games played this year -/
def total_games : ℕ := attended_games + missed_games

theorem total_games_calculation : 
  total_games = attended_games + missed_games := by sorry

end total_games_calculation_l1356_135636


namespace correct_reasoning_directions_l1356_135665

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
| SpecificToGeneral
| GeneralToSpecific
| SpecificToSpecific

-- Define a function that describes the direction of each reasoning type
def reasoningDirection (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.SpecificToGeneral
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating that the reasoning directions are correct
theorem correct_reasoning_directions :
  (reasoningDirection ReasoningType.Inductive = ReasoningDirection.SpecificToGeneral) ∧
  (reasoningDirection ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (reasoningDirection ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
by sorry

end correct_reasoning_directions_l1356_135665


namespace sqrt_equation_solution_l1356_135678

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (x - 4) = 10 → x = 104 :=
by
  sorry

end sqrt_equation_solution_l1356_135678


namespace stock_price_return_l1356_135677

theorem stock_price_return (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * 1.3
  let decrease_rate := 1 - 1 / 1.3
  increased_price * (1 - decrease_rate) = original_price :=
by
  sorry

#eval (1 - 1 / 1.3) * 100 -- This will output approximately 23.08

end stock_price_return_l1356_135677


namespace prob_same_type_three_pairs_l1356_135661

/-- Represents a collection of paired items -/
structure PairedCollection :=
  (num_pairs : ℕ)
  (items_per_pair : ℕ)
  (total_items : ℕ)
  (h_total : total_items = num_pairs * items_per_pair)

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the probability of selecting two items of the same type -/
def prob_same_type (collection : PairedCollection) : ℚ :=
  (collection.num_pairs : ℚ) / (choose collection.total_items 2)

/-- The main theorem to be proved -/
theorem prob_same_type_three_pairs :
  let shoe_collection : PairedCollection :=
    { num_pairs := 3
    , items_per_pair := 2
    , total_items := 6
    , h_total := rfl }
  prob_same_type shoe_collection = 1 / 5 := by
  sorry

end prob_same_type_three_pairs_l1356_135661


namespace sine_translation_stretch_l1356_135601

/-- The transformation of the sine function -/
theorem sine_translation_stretch (x : ℝ) :
  let f := λ x : ℝ => Real.sin x
  let g := λ x : ℝ => Real.sin (x / 2 - π / 8)
  g x = (f ∘ (λ y => y - π / 8) ∘ (λ y => y / 2)) x :=
by sorry

end sine_translation_stretch_l1356_135601


namespace range_when_p_true_range_when_one_true_one_false_l1356_135643

-- Define the propositions
def has_two_distinct_negative_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 ≠ 0

-- Theorem 1
theorem range_when_p_true (m : ℝ) :
  has_two_distinct_negative_roots m → m > 2 :=
sorry

-- Theorem 2
theorem range_when_one_true_one_false (m : ℝ) :
  (has_two_distinct_negative_roots m ↔ ¬has_no_real_roots m) →
  (m ∈ Set.Ioo 1 2 ∪ Set.Ici 3) :=
sorry

end range_when_p_true_range_when_one_true_one_false_l1356_135643


namespace function_property_l1356_135611

def IteratedFunction (f : ℕ+ → ℕ+) : ℕ → ℕ+ → ℕ+
  | 0, n => n
  | k+1, n => f (IteratedFunction f k n)

theorem function_property (f : ℕ+ → ℕ+) :
  (∀ (a b c : ℕ+), a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 →
    IteratedFunction f (a*b*c - a) (a*b*c) + 
    IteratedFunction f (a*b*c - b) (a*b*c) + 
    IteratedFunction f (a*b*c - c) (a*b*c) = a + b + c) →
  ∀ n : ℕ+, n ≥ 3 → f n = n - 1 := by
  sorry

end function_property_l1356_135611


namespace no_integer_solutions_l1356_135679

theorem no_integer_solutions : ¬∃ (x y z : ℤ),
  (x^2 - 4*x*y + 3*y^2 - z^2 = 25) ∧
  (-x^2 + 4*y*z + 3*z^2 = 36) ∧
  (x^2 + 2*x*y + 9*z^2 = 121) := by
  sorry

end no_integer_solutions_l1356_135679


namespace symmetric_point_wrt_y_axis_l1356_135647

/-- Given a point A with coordinates (-2,4), this theorem states that the point
    symmetric to A with respect to the y-axis has coordinates (2,4). -/
theorem symmetric_point_wrt_y_axis :
  let A : ℝ × ℝ := (-2, 4)
  let symmetric_point := (- A.1, A.2)
  symmetric_point = (2, 4) := by sorry

end symmetric_point_wrt_y_axis_l1356_135647


namespace negation_of_at_most_one_l1356_135667

theorem negation_of_at_most_one (P : Type → Prop) :
  (¬ (∃! x, P x)) ↔ (∃ x y, P x ∧ P y ∧ x ≠ y) :=
by sorry

end negation_of_at_most_one_l1356_135667


namespace artist_paintings_l1356_135689

theorem artist_paintings (paint_per_large : ℕ) (paint_per_small : ℕ) 
  (small_paintings : ℕ) (total_paint : ℕ) :
  paint_per_large = 3 →
  paint_per_small = 2 →
  small_paintings = 4 →
  total_paint = 17 →
  ∃ (large_paintings : ℕ), 
    large_paintings * paint_per_large + small_paintings * paint_per_small = total_paint ∧
    large_paintings = 3 :=
by sorry

end artist_paintings_l1356_135689


namespace parabola_line_intersection_l1356_135638

-- Define a parabola
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

-- Define a line
structure Line where
  m : ℝ
  b : ℝ

-- Define the concept of a directrix
def is_directrix (l : Line) (p : Parabola) : Prop := sorry

-- Define the concept of tangency
def is_tangent (l : Line) (p : Parabola) : Prop := sorry

-- Define the concept of intersection
def intersect (l : Line) (p : Parabola) : Finset ℝ := sorry

-- Main theorem
theorem parabola_line_intersection
  (p : Parabola)
  (l1 l2 : Line)
  (h1 : l1.m ≠ l2.m ∨ l1.b ≠ l2.b) -- lines are distinct
  (h2 : is_directrix l1 p)
  (h3 : ¬ is_tangent l1 p)
  (h4 : ¬ is_tangent l2 p) :
  (intersect l1 p).card + (intersect l2 p).card = 2 := by
  sorry

end parabola_line_intersection_l1356_135638


namespace board_cut_multiple_l1356_135608

/-- Given a board of 120 cm cut into two pieces, where the shorter piece is 35 cm
    and the longer piece is 15 cm longer than m times the shorter piece, m must equal 2. -/
theorem board_cut_multiple (m : ℝ) : 
  (35 : ℝ) + (m * 35 + 15) = 120 → m = 2 := by
  sorry

end board_cut_multiple_l1356_135608


namespace min_black_edges_is_four_l1356_135684

/-- Represents the coloring of a cube's edges -/
structure CubeColoring where
  edges : Fin 12 → Bool  -- True represents black, False represents red

/-- Checks if a face has an even number of black edges -/
def has_even_black_edges (c : CubeColoring) (face : Fin 6) : Bool :=
  sorry

/-- Checks if all faces have an even number of black edges -/
def all_faces_even_black (c : CubeColoring) : Prop :=
  ∀ face : Fin 6, has_even_black_edges c face

/-- Counts the number of black edges in a coloring -/
def count_black_edges (c : CubeColoring) : Nat :=
  sorry

/-- The main theorem: The minimum number of black edges required is 4 -/
theorem min_black_edges_is_four :
  (∃ c : CubeColoring, all_faces_even_black c ∧ count_black_edges c = 4) ∧
  (∀ c : CubeColoring, all_faces_even_black c → count_black_edges c ≥ 4) :=
sorry

end min_black_edges_is_four_l1356_135684


namespace division_problem_l1356_135690

/-- Proves that given a total amount of 544, if A gets 2/3 of what B gets, and B gets 1/4 of what C gets, then A gets 64. -/
theorem division_problem (total : ℚ) (a b c : ℚ) 
  (h_total : total = 544)
  (h_ab : a = (2/3) * b)
  (h_bc : b = (1/4) * c)
  (h_sum : a + b + c = total) : 
  a = 64 := by
  sorry

end division_problem_l1356_135690


namespace pure_imaginary_fraction_l1356_135674

theorem pure_imaginary_fraction (a : ℝ) : 
  (((a : ℂ) - Complex.I) / (1 + Complex.I)).re = 0 → a = 1 := by
  sorry

end pure_imaginary_fraction_l1356_135674


namespace smallest_prime_ten_less_than_square_l1356_135664

theorem smallest_prime_ten_less_than_square : ∃ (n : ℕ), 
  (∀ (m : ℕ), m > 0 ∧ Nat.Prime m ∧ (∃ (k : ℕ), m = k^2 - 10) → n ≤ m) ∧
  n > 0 ∧ Nat.Prime n ∧ (∃ (k : ℕ), n = k^2 - 10) ∧ n = 71 := by
sorry

end smallest_prime_ten_less_than_square_l1356_135664


namespace circle_x_axis_intersection_sum_l1356_135603

theorem circle_x_axis_intersection_sum (c : ℝ × ℝ) (r : ℝ) : 
  c = (3, -4) → r = 7 → 
  ∃ x₁ x₂ : ℝ, 
    ((x₁ - 3)^2 + 4^2 = r^2) ∧
    ((x₂ - 3)^2 + 4^2 = r^2) ∧
    x₁ + x₂ = 6 :=
by sorry

end circle_x_axis_intersection_sum_l1356_135603


namespace sufficient_but_not_necessary_l1356_135618

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, x - 1 = 0 → (x - 1) * (x + 2) = 0) ∧
  (∃ x : ℝ, (x - 1) * (x + 2) = 0 ∧ x - 1 ≠ 0) :=
by sorry

end sufficient_but_not_necessary_l1356_135618


namespace echo_earnings_l1356_135651

-- Define the schools and their parameters
structure School where
  name : String
  students : ℕ
  days : ℕ
  rate_multiplier : ℚ

-- Define the problem parameters
def delta : School := { name := "Delta", students := 8, days := 4, rate_multiplier := 1 }
def echo : School := { name := "Echo", students := 6, days := 6, rate_multiplier := 3/2 }
def foxtrot : School := { name := "Foxtrot", students := 7, days := 7, rate_multiplier := 1 }

def total_payment : ℚ := 1284

-- Function to calculate effective student-days
def effective_student_days (s : School) : ℚ :=
  ↑s.students * ↑s.days * s.rate_multiplier

-- Theorem statement
theorem echo_earnings :
  let total_effective_days := effective_student_days delta + effective_student_days echo + effective_student_days foxtrot
  let daily_wage := total_payment / total_effective_days
  effective_student_days echo * daily_wage = 513.6 := by
sorry

end echo_earnings_l1356_135651


namespace inequality_theorem_l1356_135685

theorem inequality_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  2 * Real.log m - n > 2 * Real.log n - m := by
  sorry

end inequality_theorem_l1356_135685


namespace combined_instruments_count_l1356_135675

/-- Represents the number of instruments owned by a person -/
structure InstrumentCount where
  flutes : ℕ
  horns : ℕ
  harps : ℕ

/-- Calculates the total number of instruments -/
def totalInstruments (ic : InstrumentCount) : ℕ :=
  ic.flutes + ic.horns + ic.harps

/-- Charlie's instrument count -/
def charlie : InstrumentCount :=
  { flutes := 1, horns := 2, harps := 1 }

/-- Carli's instrument count -/
def carli : InstrumentCount :=
  { flutes := 2 * charlie.flutes,
    horns := charlie.horns / 2,
    harps := 0 }

/-- Theorem: The combined total number of musical instruments owned by Charlie and Carli is 7 -/
theorem combined_instruments_count :
  totalInstruments charlie + totalInstruments carli = 7 := by
  sorry

end combined_instruments_count_l1356_135675


namespace seventh_root_unity_product_l1356_135695

theorem seventh_root_unity_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 8 := by
  sorry

end seventh_root_unity_product_l1356_135695


namespace sandra_feeding_days_l1356_135662

/-- The number of days Sandra can feed the puppies with the given formula -/
def feeding_days (num_puppies : ℕ) (total_portions : ℕ) (feedings_per_day : ℕ) : ℕ :=
  total_portions / (num_puppies * feedings_per_day)

/-- Theorem stating that Sandra can feed the puppies for 5 days with the given formula -/
theorem sandra_feeding_days :
  feeding_days 7 105 3 = 5 := by
  sorry

end sandra_feeding_days_l1356_135662


namespace rectangle_area_l1356_135623

-- Define the rectangle
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

-- Define the area function
def area (r : Rectangle) : ℝ :=
  (r.x2 - r.x1) * (r.y2 - r.y1)

-- Theorem statement
theorem rectangle_area :
  let r : Rectangle := { x1 := 0, y1 := 0, x2 := 3, y2 := 3 }
  area r = 9 := by sorry

end rectangle_area_l1356_135623


namespace partner_c_profit_share_l1356_135621

/-- Given the investments of partners A, B, and C, and the total profit,
    calculate C's share of the profit. -/
theorem partner_c_profit_share
  (invest_a invest_b invest_c total_profit : ℝ)
  (h1 : invest_a = 3 * invest_b)
  (h2 : invest_a = 2 / 3 * invest_c)
  (h3 : total_profit = 66000) :
  (invest_c / (invest_a + invest_b + invest_c)) * total_profit = (9 / 17) * 66000 :=
by sorry

end partner_c_profit_share_l1356_135621


namespace exponent_relations_l1356_135699

theorem exponent_relations (a : ℝ) (m n k : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : a^m = 2) 
  (h3 : a^n = 4) 
  (h4 : a^k = 32) : 
  (a^(3*m + 2*n - k) = 4) ∧ (k - 3*m - n = 0) := by
  sorry

end exponent_relations_l1356_135699


namespace yearly_income_calculation_l1356_135649

/-- Calculates the simple interest for a given principal, rate, and time (in years) -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ := 1) : ℚ :=
  principal * rate * time / 100

theorem yearly_income_calculation (totalAmount : ℚ) (part1 : ℚ) (rate1 : ℚ) (rate2 : ℚ) 
  (h1 : totalAmount = 2600)
  (h2 : part1 = 1600)
  (h3 : rate1 = 5)
  (h4 : rate2 = 6) :
  simpleInterest part1 rate1 + simpleInterest (totalAmount - part1) rate2 = 140 := by
  sorry

#eval simpleInterest 1600 5 + simpleInterest 1000 6

end yearly_income_calculation_l1356_135649


namespace smallest_n_for_exact_tax_l1356_135691

theorem smallest_n_for_exact_tax : ∃ (x : ℕ), (105 * x) % 10000 = 0 ∧ 
  (∀ (y : ℕ), y < 21 → (105 * y) % 10000 ≠ 0) :=
sorry

end smallest_n_for_exact_tax_l1356_135691


namespace jose_investment_is_45000_l1356_135676

/-- Represents the investment scenario of Tom and Jose -/
structure InvestmentScenario where
  tom_investment : ℕ
  tom_months : ℕ
  jose_months : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment based on the given scenario -/
def calculate_jose_investment (scenario : InvestmentScenario) : ℕ :=
  (scenario.jose_profit * scenario.tom_investment * scenario.tom_months) /
  (scenario.tom_months * (scenario.total_profit - scenario.jose_profit))

/-- Theorem stating that Jose's investment is 45000 given the specific scenario -/
theorem jose_investment_is_45000 (scenario : InvestmentScenario)
  (h1 : scenario.tom_investment = 30000)
  (h2 : scenario.tom_months = 12)
  (h3 : scenario.jose_months = 10)
  (h4 : scenario.total_profit = 45000)
  (h5 : scenario.jose_profit = 25000) :
  calculate_jose_investment scenario = 45000 := by
  sorry


end jose_investment_is_45000_l1356_135676


namespace jaspers_refreshments_l1356_135619

theorem jaspers_refreshments (chips drinks : ℕ) (h1 : chips = 27) (h2 : drinks = 31) :
  let hot_dogs := drinks - 12
  chips - hot_dogs = 8 := by
  sorry

end jaspers_refreshments_l1356_135619


namespace sqrt_squared_equals_original_sqrt_529441_squared_l1356_135617

theorem sqrt_squared_equals_original (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n)^2 = n := by
  sorry

theorem sqrt_529441_squared :
  (Real.sqrt 529441)^2 = 529441 := by
  apply sqrt_squared_equals_original
  norm_num

end sqrt_squared_equals_original_sqrt_529441_squared_l1356_135617


namespace alcohol_fraction_after_water_increase_l1356_135626

theorem alcohol_fraction_after_water_increase (v : ℝ) (h : v > 0) :
  let initial_alcohol := (2 / 3) * v
  let initial_water := (1 / 3) * v
  let new_water := 3 * initial_water
  let new_total := initial_alcohol + new_water
  initial_alcohol / new_total = 2 / 5 := by
sorry

end alcohol_fraction_after_water_increase_l1356_135626


namespace no_integer_in_interval_l1356_135637

theorem no_integer_in_interval (n : ℕ) : ¬∃ k : ℤ, (n : ℝ) * Real.sqrt 2 - 1 / (3 * (n : ℝ)) < (k : ℝ) ∧ (k : ℝ) < (n : ℝ) * Real.sqrt 2 + 1 / (3 * (n : ℝ)) := by
  sorry

end no_integer_in_interval_l1356_135637


namespace tadpole_survival_fraction_l1356_135620

/-- Represents the frog pond ecosystem --/
structure FrogPond where
  num_frogs : ℕ
  num_tadpoles : ℕ
  max_capacity : ℕ
  frogs_to_relocate : ℕ

/-- Calculates the fraction of tadpoles that will survive to maturity as frogs --/
def survival_fraction (pond : FrogPond) : ℚ :=
  let surviving_tadpoles := pond.max_capacity - pond.num_frogs
  ↑surviving_tadpoles / ↑pond.num_tadpoles

/-- Theorem stating the fraction of tadpoles that will survive to maturity as frogs --/
theorem tadpole_survival_fraction (pond : FrogPond) 
  (h1 : pond.num_frogs = 5)
  (h2 : pond.num_tadpoles = 3 * pond.num_frogs)
  (h3 : pond.max_capacity = 8)
  (h4 : pond.frogs_to_relocate = 7) :
  survival_fraction pond = 1 / 5 := by
  sorry

end tadpole_survival_fraction_l1356_135620


namespace tangent_slope_at_pi_over_four_l1356_135633

theorem tangent_slope_at_pi_over_four :
  let f (x : ℝ) := Real.tan x
  (deriv f) (π / 4) = 2 := by
  sorry

end tangent_slope_at_pi_over_four_l1356_135633


namespace triangle_height_theorem_l1356_135653

-- Define the triangle ABC
theorem triangle_height_theorem (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  1 + 2 * Real.cos (B + C) = 0 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  -- Conclusion
  b * Real.sin C = (Real.sqrt 3 + 1) / 2 :=
by sorry

end triangle_height_theorem_l1356_135653


namespace target_probability_l1356_135648

/-- The probability of hitting the target in a single shot -/
def p : ℝ := 0.8

/-- The probability of missing the target in a single shot -/
def q : ℝ := 1 - p

/-- The probability of hitting the target at least once in two shots -/
def prob_hit_at_least_once_in_two : ℝ := 1 - q^2

theorem target_probability :
  prob_hit_at_least_once_in_two = 0.96 →
  (5 : ℝ) * p^4 * q = 0.4096 :=
sorry

end target_probability_l1356_135648


namespace fill_time_without_leakage_l1356_135642

/-- Represents the time to fill a tank with leakage -/
def fill_time_with_leakage : ℝ := 18

/-- Represents the time to empty the tank due to leakage -/
def empty_time_leakage : ℝ := 36

/-- Represents the volume of the tank -/
def tank_volume : ℝ := 1

/-- Theorem stating the time to fill the tank without leakage -/
theorem fill_time_without_leakage :
  let fill_rate := tank_volume / fill_time_with_leakage + tank_volume / empty_time_leakage
  tank_volume / fill_rate = 12 := by
  sorry

end fill_time_without_leakage_l1356_135642


namespace average_age_after_leaving_l1356_135624

def initial_people : ℕ := 7
def initial_average_age : ℚ := 28
def leaving_person_age : ℕ := 20

theorem average_age_after_leaving :
  let total_age : ℚ := initial_people * initial_average_age
  let remaining_total_age : ℚ := total_age - leaving_person_age
  let remaining_people : ℕ := initial_people - 1
  remaining_total_age / remaining_people = 29.33 := by sorry

end average_age_after_leaving_l1356_135624


namespace quadratic_distinct_roots_l1356_135612

theorem quadratic_distinct_roots (n : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + n*x + 9 = 0 ∧ y^2 + n*y + 9 = 0) ↔ 
  (n < -6 ∨ n > 6) := by
sorry

end quadratic_distinct_roots_l1356_135612


namespace sum_of_reversed_square_digits_l1356_135692

/-- The number to be squared -/
def n : ℕ := 11111

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ := sorry

/-- Function to reverse the digits of a natural number -/
def reverse_digits (m : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the digits of the reversed square of 11111 is 25 -/
theorem sum_of_reversed_square_digits : sum_of_digits (reverse_digits (n^2)) = 25 := by sorry

end sum_of_reversed_square_digits_l1356_135692


namespace adam_shopping_cost_l1356_135663

/-- The total cost of Adam's shopping given the number of sandwiches, cost per sandwich, and cost of water. -/
def total_cost (num_sandwiches : ℕ) (sandwich_price : ℕ) (water_price : ℕ) : ℕ :=
  num_sandwiches * sandwich_price + water_price

/-- Theorem stating that Adam's total shopping cost is $11. -/
theorem adam_shopping_cost :
  total_cost 3 3 2 = 11 := by
  sorry

end adam_shopping_cost_l1356_135663


namespace optimal_labeled_price_l1356_135698

/-- Represents the pricing strategy of a retailer --/
structure RetailPricing where
  list_price : ℝ
  purchase_discount : ℝ
  sale_discount : ℝ
  profit_margin : ℝ
  labeled_price : ℝ

/-- The pricing strategy satisfies the retailer's conditions --/
def satisfies_conditions (rp : RetailPricing) : Prop :=
  rp.purchase_discount = 0.3 ∧
  rp.sale_discount = 0.25 ∧
  rp.profit_margin = 0.3 ∧
  rp.labeled_price > 0 ∧
  rp.list_price > 0

/-- The final selling price after discount --/
def selling_price (rp : RetailPricing) : ℝ :=
  rp.labeled_price * (1 - rp.sale_discount)

/-- The purchase price for the retailer --/
def purchase_price (rp : RetailPricing) : ℝ :=
  rp.list_price * (1 - rp.purchase_discount)

/-- The profit calculation --/
def profit (rp : RetailPricing) : ℝ :=
  selling_price rp - purchase_price rp

/-- The theorem stating that the labeled price should be 135% of the list price --/
theorem optimal_labeled_price (rp : RetailPricing) 
  (h : satisfies_conditions rp) : 
  rp.labeled_price = 1.35 * rp.list_price ↔ 
  profit rp = rp.profit_margin * selling_price rp :=
sorry

end optimal_labeled_price_l1356_135698


namespace P_sufficient_not_necessary_for_Q_l1356_135632

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}

theorem P_sufficient_not_necessary_for_Q :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by sorry

end P_sufficient_not_necessary_for_Q_l1356_135632


namespace rope_length_is_35_l1356_135604

/-- The length of the rope in meters -/
def rope_length : ℝ := 35

/-- The time ratio between walking with and against the tractor -/
def time_ratio : ℝ := 7

/-- The equation for walking in the same direction as the tractor -/
def same_direction_equation (x S : ℝ) : Prop :=
  x + time_ratio * S = 140

/-- The equation for walking in the opposite direction of the tractor -/
def opposite_direction_equation (x S : ℝ) : Prop :=
  x - S = 20

theorem rope_length_is_35 :
  ∃ S : ℝ, same_direction_equation rope_length S ∧ opposite_direction_equation rope_length S :=
sorry

end rope_length_is_35_l1356_135604


namespace complement_of_A_in_U_l1356_135644

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 4}

theorem complement_of_A_in_U : 
  (U \ A) = {2, 3, 5} := by sorry

end complement_of_A_in_U_l1356_135644


namespace product_14_sum_5_or_minus_5_l1356_135646

theorem product_14_sum_5_or_minus_5 (a b c d : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 14 →
  a + b + c + d = 5 ∨ a + b + c + d = -5 := by
sorry

end product_14_sum_5_or_minus_5_l1356_135646


namespace candy_distribution_l1356_135609

theorem candy_distribution (total_candies : ℕ) (candies_per_student : ℕ) (num_students : ℕ) :
  total_candies = 81 →
  candies_per_student = 9 →
  total_candies = candies_per_student * num_students →
  num_students = 9 := by
sorry

end candy_distribution_l1356_135609


namespace ratio_bounds_in_acute_triangle_l1356_135681

theorem ratio_bounds_in_acute_triangle (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  -- Sum of angles in a triangle
  A + B + C = π ∧
  -- A = 2B
  A = 2 * B ∧
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  c / (Real.sin C) = a / (Real.sin A) →
  -- Conclusion: a/b is bounded by √2 and √3
  Real.sqrt 2 < a / b ∧ a / b < Real.sqrt 3 :=
by sorry

end ratio_bounds_in_acute_triangle_l1356_135681


namespace stool_height_l1356_135640

-- Define the constants
def ceiling_height : ℝ := 300  -- in cm
def bulb_below_ceiling : ℝ := 15  -- in cm
def alice_height : ℝ := 160  -- in cm
def alice_reach : ℝ := 50  -- in cm

-- Define the theorem
theorem stool_height : 
  ∃ (h : ℝ), 
    h = ceiling_height - bulb_below_ceiling - (alice_height + alice_reach) ∧ 
    h = 75 :=
by sorry

end stool_height_l1356_135640


namespace journey_theorem_l1356_135693

/-- Represents a two-segment journey with different speeds -/
structure Journey where
  time_at_5mph : ℝ
  time_at_15mph : ℝ
  total_time : ℝ
  total_distance : ℝ

/-- The average speed of the entire journey is 10 mph -/
def average_speed (j : Journey) : Prop :=
  j.total_distance / j.total_time = 10

/-- The total time is the sum of time spent at each speed -/
def total_time_sum (j : Journey) : Prop :=
  j.total_time = j.time_at_5mph + j.time_at_15mph

/-- The total distance is the sum of distances covered at each speed -/
def total_distance_sum (j : Journey) : Prop :=
  j.total_distance = 5 * j.time_at_5mph + 15 * j.time_at_15mph

/-- The fraction of time spent at 15 mph is half of the total time -/
def half_time_at_15mph (j : Journey) : Prop :=
  j.time_at_15mph / j.total_time = 1 / 2

theorem journey_theorem (j : Journey) 
  (h1 : average_speed j) 
  (h2 : total_time_sum j) 
  (h3 : total_distance_sum j) : 
  half_time_at_15mph j := by
  sorry

end journey_theorem_l1356_135693


namespace smallest_cube_ending_888_l1356_135629

theorem smallest_cube_ending_888 :
  ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 888 ∧ ∀ (m : ℕ), m > 0 ∧ m^3 % 1000 = 888 → n ≤ m :=
by sorry

end smallest_cube_ending_888_l1356_135629


namespace pizza_theorem_l1356_135628

/-- The number of pizzas ordered for a class celebration --/
def pizza_problem (num_boys : ℕ) (num_girls : ℕ) (boys_pizzas : ℕ) : Prop :=
  num_girls = 11 ∧
  num_boys > num_girls ∧
  boys_pizzas = 10 ∧
  ∃ (total_pizzas : ℚ),
    total_pizzas = boys_pizzas + (num_girls : ℚ) * (boys_pizzas : ℚ) / (2 * num_boys : ℚ) ∧
    total_pizzas = 11

theorem pizza_theorem :
  ∃ (num_boys : ℕ), pizza_problem num_boys 11 10 :=
sorry

end pizza_theorem_l1356_135628


namespace counterexample_exists_l1356_135688

theorem counterexample_exists : ∃ n : ℕ, 
  (¬ Nat.Prime n) ∧ (Nat.Prime (n - 3) ∨ Nat.Prime (n - 2)) := by
  sorry

end counterexample_exists_l1356_135688


namespace drink_conversion_l1356_135686

theorem drink_conversion (x : ℚ) : 
  (4 / (4 + x) * 63 = 3 / 7 * (63 + 21)) → x = 3 := by
  sorry

end drink_conversion_l1356_135686


namespace unique_y_value_l1356_135639

theorem unique_y_value (x : ℝ) (h : x^2 + 4 * (x / (x + 3))^2 = 64) : 
  ((x + 3)^2 * (x - 2)) / (2 * x + 3) = 250 / 3 := by
  sorry

end unique_y_value_l1356_135639


namespace cube_plus_three_square_plus_three_plus_one_l1356_135610

theorem cube_plus_three_square_plus_three_plus_one : 101^3 + 3*(101^2) + 3*101 + 1 = 1061208 := by
  sorry

end cube_plus_three_square_plus_three_plus_one_l1356_135610


namespace arithmetic_expression_evaluation_l1356_135641

theorem arithmetic_expression_evaluation : 3 + 2 * (8 - 3) = 13 := by
  sorry

end arithmetic_expression_evaluation_l1356_135641


namespace contrapositive_equivalence_l1356_135625

theorem contrapositive_equivalence : 
  (∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) ↔ 
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) :=
by sorry

end contrapositive_equivalence_l1356_135625


namespace remainder_divisibility_l1356_135680

theorem remainder_divisibility (x : ℤ) : x % 52 = 19 → x % 7 = 5 := by
  sorry

end remainder_divisibility_l1356_135680


namespace emma_remaining_time_l1356_135659

-- Define the wrapping rates and initial joint work time
def emma_rate : ℚ := 1 / 6
def troy_rate : ℚ := 1 / 8
def joint_work_time : ℚ := 2

-- Define the function to calculate the remaining time for Emma
def remaining_time_for_emma (emma_rate troy_rate joint_work_time : ℚ) : ℚ :=
  let joint_completion := (emma_rate + troy_rate) * joint_work_time
  let remaining_work := 1 - joint_completion
  remaining_work / emma_rate

-- Theorem statement
theorem emma_remaining_time :
  remaining_time_for_emma emma_rate troy_rate joint_work_time = 5 / 2 := by
  sorry

end emma_remaining_time_l1356_135659


namespace skew_lines_sufficient_not_necessary_l1356_135606

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line in 3D space
  -- This is a placeholder structure

/-- Two lines are skew if they are not coplanar -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- Two lines have no common point if they don't intersect -/
def have_no_common_point (l1 l2 : Line3D) : Prop :=
  sorry

/-- Theorem stating that "are_skew" is a sufficient but not necessary condition for "have_no_common_point" -/
theorem skew_lines_sufficient_not_necessary :
  ∃ (l1 l2 l3 l4 : Line3D),
    (are_skew l1 l2 → have_no_common_point l1 l2) ∧
    (have_no_common_point l3 l4 ∧ ¬are_skew l3 l4) :=
  sorry

end skew_lines_sufficient_not_necessary_l1356_135606


namespace probability_h_in_mathematics_l1356_135687

def word : String := "Mathematics"

def count_letter (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem probability_h_in_mathematics :
  (count_letter word 'h' : ℚ) / word.length = 1 / 11 := by
  sorry

end probability_h_in_mathematics_l1356_135687


namespace hyperbola_distance_l1356_135615

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 20 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define a point on the hyperbola
def P : ℝ × ℝ := sorry

-- Distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_distance :
  hyperbola P.1 P.2 → distance P F1 = 9 → distance P F2 = 17 := by sorry

end hyperbola_distance_l1356_135615


namespace largest_prime_factor_l1356_135696

theorem largest_prime_factor : ∃ (p : ℕ), Nat.Prime p ∧ 
  p ∣ (15^3 + 10^4 - 5^5) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (15^3 + 10^4 - 5^5) → q ≤ p :=
by sorry

end largest_prime_factor_l1356_135696


namespace remainder_is_224_l1356_135613

/-- The polynomial f(x) = x^5 - 8x^4 + 16x^3 + 25x^2 - 50x + 24 -/
def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 16*x^3 + 25*x^2 - 50*x + 24

/-- The remainder when f(x) is divided by (x - 4) -/
def remainder : ℝ := f 4

theorem remainder_is_224 : remainder = 224 := by
  sorry

end remainder_is_224_l1356_135613


namespace chessboard_symmetry_l1356_135607

-- Define a chessboard
structure Chessboard :=
  (ranks : Fin 8)
  (files : Fin 8)

-- Define a chess square
structure Square :=
  (file : Char)
  (rank : Nat)

-- Define symmetry on the chessboard
def symmetric (s1 s2 : Square) (b : Chessboard) : Prop :=
  s1.file = s2.file ∧ s1.rank + s2.rank = 9

-- Define the line of symmetry
def lineOfSymmetry (b : Chessboard) : Prop :=
  ∀ (s1 s2 : Square), symmetric s1 s2 b → (s1.rank = 4 ∧ s2.rank = 5) ∨ (s1.rank = 5 ∧ s2.rank = 4)

-- Theorem statement
theorem chessboard_symmetry (b : Chessboard) :
  lineOfSymmetry b ∧
  symmetric (Square.mk 'e' 2) (Square.mk 'e' 7) b ∧
  symmetric (Square.mk 'h' 5) (Square.mk 'h' 4) b :=
sorry

end chessboard_symmetry_l1356_135607


namespace not_right_triangle_4_6_8_l1356_135666

/-- Checks if three line segments can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- Theorem stating that the line segments 4, 6, and 8 cannot form a right triangle -/
theorem not_right_triangle_4_6_8 : ¬ is_right_triangle 4 6 8 := by
  sorry

end not_right_triangle_4_6_8_l1356_135666


namespace min_sum_of_internally_tangent_circles_l1356_135614

/-- Given two circles C₁ and C₂ with equations x² + y² + 2ax + a² - 4 = 0 and x² + y² - 2by - 1 + b² = 0 respectively, 
    where a, b ∈ ℝ, and C₁ and C₂ have only one common tangent line, 
    the minimum value of a + b is -√2. -/
theorem min_sum_of_internally_tangent_circles (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 = 0 → x^2 + y^2 - 2*b*y - 1 + b^2 = 0 → False) ∧ 
  (∃ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∧ x^2 + y^2 - 2*b*y - 1 + b^2 = 0) →
  (a + b ≥ -Real.sqrt 2) ∧ (∃ a₀ b₀ : ℝ, a₀ + b₀ = -Real.sqrt 2) :=
by sorry

end min_sum_of_internally_tangent_circles_l1356_135614


namespace belt_and_road_population_scientific_notation_l1356_135634

theorem belt_and_road_population_scientific_notation :
  (4600000000 : ℝ) = 4.6 * (10 ^ 9) := by sorry

end belt_and_road_population_scientific_notation_l1356_135634


namespace yurts_are_xarps_and_zarqs_l1356_135682

-- Define the sets
variable (U : Type) -- Universe set
variable (Xarp Zarq Yurt Wint : Set U)

-- Define the conditions
variable (h1 : Xarp ⊆ Zarq)
variable (h2 : Yurt ⊆ Zarq)
variable (h3 : Xarp ⊆ Wint)
variable (h4 : Yurt ⊆ Xarp)

-- Theorem to prove
theorem yurts_are_xarps_and_zarqs : Yurt ⊆ Xarp ∩ Zarq :=
sorry

end yurts_are_xarps_and_zarqs_l1356_135682


namespace cost_operation_l1356_135645

theorem cost_operation (t : ℝ) (b b' : ℝ) : 
  (∀ C, C = t * b^4) →
  (∃ e, e = 16 * t * b^4) →
  (∃ e, e = t * b'^4) →
  b' = 2 * b :=
sorry

end cost_operation_l1356_135645


namespace hexagon_area_2016_l1356_135657

/-- The area of the hexagon formed by constructing squares on the sides of a right triangle -/
def hexagon_area (a b : ℕ) : ℕ := 2 * (a^2 + b^2 + a*b)

/-- The proposition to be proved -/
theorem hexagon_area_2016 :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ hexagon_area a b = 2016 ∧
  (∀ (x y : ℕ), x > 0 → y > 0 → hexagon_area x y = 2016 → (x = 12 ∧ y = 24) ∨ (x = 24 ∧ y = 12)) :=
sorry

end hexagon_area_2016_l1356_135657


namespace train_passing_platform_l1356_135694

theorem train_passing_platform (train_length platform_length : ℝ) 
  (time_to_pass_point : ℝ) (h1 : train_length = 1400) 
  (h2 : platform_length = 700) (h3 : time_to_pass_point = 100) :
  (train_length + platform_length) / (train_length / time_to_pass_point) = 150 :=
by sorry

end train_passing_platform_l1356_135694


namespace min_sum_inverse_squares_min_sum_inverse_squares_value_min_sum_inverse_squares_equality_l1356_135668

theorem min_sum_inverse_squares (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c → a^2 + b^2 + c^2 = 1 →
  1/x^2 + 1/y^2 + 1/z^2 ≤ 1/a^2 + 1/b^2 + 1/c^2 :=
by sorry

theorem min_sum_inverse_squares_value (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  1/x^2 + 1/y^2 + 1/z^2 ≥ 9 :=
by sorry

theorem min_sum_inverse_squares_equality :
  ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 
  x^2 + y^2 + z^2 = 1 ∧ 1/x^2 + 1/y^2 + 1/z^2 = 9 :=
by sorry

end min_sum_inverse_squares_min_sum_inverse_squares_value_min_sum_inverse_squares_equality_l1356_135668
