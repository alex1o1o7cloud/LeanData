import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l2112_211223

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (2 * a)) + (1 / (2 * b)) + (1 / (2 * c)) ≥ (1 / (a + b)) + (1 / (b + c)) + (1 / (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2112_211223


namespace NUMINAMATH_CALUDE_reflection_of_M_l2112_211276

/-- Reflection of a point about the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem reflection_of_M :
  let M : ℝ × ℝ := (3, 2)
  reflect_x M = (3, -2) := by sorry

end NUMINAMATH_CALUDE_reflection_of_M_l2112_211276


namespace NUMINAMATH_CALUDE_linear_function_second_quadrant_increasing_l2112_211282

/-- A linear function passing through the second quadrant with increasing y as x increases -/
def LinearFunctionSecondQuadrantIncreasing (k b : ℝ) : Prop :=
  k > 0 ∧ b > 0

/-- The property of a function passing through the second quadrant -/
def PassesThroughSecondQuadrant (f : ℝ → ℝ) : Prop :=
  ∃ x y, x < 0 ∧ y > 0 ∧ f x = y

/-- The property of a function being increasing -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

/-- Theorem stating that a linear function with positive slope and y-intercept
    passes through the second quadrant and is increasing -/
theorem linear_function_second_quadrant_increasing (k b : ℝ) :
  LinearFunctionSecondQuadrantIncreasing k b ↔
  PassesThroughSecondQuadrant (λ x => k * x + b) ∧
  IsIncreasing (λ x => k * x + b) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_second_quadrant_increasing_l2112_211282


namespace NUMINAMATH_CALUDE_lisa_marble_distribution_l2112_211238

/-- The minimum number of additional marbles needed -/
def additional_marbles_needed (friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let required_marbles := friends * (friends + 1) / 2
  max (required_marbles - initial_marbles) 0

/-- Theorem stating the solution to Lisa's marble distribution problem -/
theorem lisa_marble_distribution (friends : ℕ) (initial_marbles : ℕ)
    (h1 : friends = 12)
    (h2 : initial_marbles = 50) :
    additional_marbles_needed friends initial_marbles = 28 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marble_distribution_l2112_211238


namespace NUMINAMATH_CALUDE_remaining_macaroons_weight_l2112_211207

def macaroon_problem (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (num_bags : ℕ) (bags_eaten : ℕ) : ℕ :=
  let total_weight := total_macaroons * weight_per_macaroon
  let macaroons_per_bag := total_macaroons / num_bags
  let weight_per_bag := macaroons_per_bag * weight_per_macaroon
  total_weight - (bags_eaten * weight_per_bag)

theorem remaining_macaroons_weight :
  macaroon_problem 12 5 4 1 = 45 := by
  sorry

end NUMINAMATH_CALUDE_remaining_macaroons_weight_l2112_211207


namespace NUMINAMATH_CALUDE_candy_count_correct_l2112_211214

/-- Represents the number of pieces in each box of chocolates -/
def chocolate_boxes : List Nat := [500, 350, 700, 400, 450, 600]

/-- Represents the number of pieces in each box of lollipops -/
def lollipop_boxes : List Nat := [200, 300, 250, 350]

/-- Represents the number of pieces in each box of gummy bears -/
def gummy_bear_boxes : List Nat := [500, 550, 400, 600, 450]

/-- The total number of candy pieces in all boxes -/
def total_candies : Nat :=
  chocolate_boxes.sum + lollipop_boxes.sum + gummy_bear_boxes.sum

theorem candy_count_correct : total_candies = 6600 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_correct_l2112_211214


namespace NUMINAMATH_CALUDE_square_difference_401_399_l2112_211262

theorem square_difference_401_399 : 401^2 - 399^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_401_399_l2112_211262


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_with_8_0_digits_l2112_211296

/-- A function that checks if all digits of a natural number are either 8 or 0 -/
def all_digits_eight_or_zero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 8 ∨ d = 0

/-- The theorem statement -/
theorem largest_multiple_of_15_with_8_0_digits :
  ∃ n : ℕ, n > 0 ∧ 15 ∣ n ∧ all_digits_eight_or_zero n ∧
  (∀ m : ℕ, m > n → ¬(15 ∣ m ∧ all_digits_eight_or_zero m)) ∧
  n / 15 = 592 := by
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_with_8_0_digits_l2112_211296


namespace NUMINAMATH_CALUDE_electric_distance_average_costs_annual_mileage_threshold_l2112_211224

-- Define variables and constants
variable (x : ℝ) -- Average charging cost per km for electric vehicle
def fuel_cost_diff : ℝ := 0.6 -- Difference in cost per km between fuel and electric
def charging_cost : ℝ := 300 -- Charging cost for electric vehicle
def refueling_cost : ℝ := 300 -- Refueling cost for fuel vehicle
def distance_ratio : ℝ := 4 -- Ratio of electric vehicle distance to fuel vehicle distance
def other_cost_fuel : ℝ := 4800 -- Other annual costs for fuel vehicle
def other_cost_electric : ℝ := 7800 -- Other annual costs for electric vehicle

-- Theorem statements
theorem electric_distance (hx : x > 0) : 
  (charging_cost : ℝ) / x = 300 / x :=
sorry

theorem average_costs (hx : x > 0) : 
  x = 0.2 ∧ x + fuel_cost_diff = 0.8 :=
sorry

theorem annual_mileage_threshold (y : ℝ) :
  0.2 * y + other_cost_electric < 0.8 * y + other_cost_fuel ↔ y > 5000 :=
sorry

end NUMINAMATH_CALUDE_electric_distance_average_costs_annual_mileage_threshold_l2112_211224


namespace NUMINAMATH_CALUDE_message_reconstruction_existence_l2112_211281

/-- Represents a text as a list of characters -/
def Text := List Char

/-- Represents a permutation of characters -/
def Permutation := Char → Char

/-- Represents a substitution of characters -/
def Substitution := Char → Char

/-- Apply a permutation to a text -/
def applyPermutation (p : Permutation) (t : Text) : Text :=
  t.map p

/-- Apply a substitution to a text -/
def applySubstitution (s : Substitution) (t : Text) : Text :=
  t.map s

/-- Check if a substitution is bijective -/
def isBijectiveSubstitution (s : Substitution) : Prop :=
  Function.Injective s ∧ Function.Surjective s

theorem message_reconstruction_existence :
  ∃ (original : Text) (p : Permutation) (s : Substitution),
    let text1 := "МИМОПРАСТЕТИРАСИСПДАИСАФЕИИБОЕТКЖРГЛЕОЛОИШИСАННСЙСАООЛТЛЕЯТУИЦВЫИПИЯДПИЩПЬПСЕЮЯ".data
    let text2 := "УЩФМШПДРЕЦЧЕШЮЧДАКЕЧМДВКШБЕЕЧДФЭПЙЩГШФЩЦЕЮЩФПМЕЧПМРРМЕОЧХЕШРГИФРЯЯЛКДФФЕЕ".data
    applyPermutation p original = text1 ∧
    applySubstitution s original = text2 ∧
    isBijectiveSubstitution s ∧
    original = "ШЕСТАЯОЛИМПИАДАПОКРИПТОГРАФИИПОСВЯЩЕННАЯСЕМЬДЕСЯТИПЯТИЛЕТИЮСПЕЦИАЛЬНОЙСЛУЖБЫРОССИИ".data :=
by sorry


end NUMINAMATH_CALUDE_message_reconstruction_existence_l2112_211281


namespace NUMINAMATH_CALUDE_journey_length_l2112_211285

theorem journey_length :
  ∀ (total : ℝ),
  (total / 4 : ℝ) + 30 + (total / 3 : ℝ) = total →
  total = 72 := by
sorry

end NUMINAMATH_CALUDE_journey_length_l2112_211285


namespace NUMINAMATH_CALUDE_six_possible_values_for_A_l2112_211291

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the sum operation in the problem -/
def SumOperation (A B X Y : Digit) : Prop :=
  (A.val * 1000000 + B.val * 1000 + A.val) + 
  (B.val * 1000000 + A.val * 1000 + B.val) = 
  (X.val * 10000000 + X.val * 1000000 + X.val * 10000 + Y.val * 1000 + X.val * 100 + X.val)

/-- The main theorem stating that there are exactly 6 possible values for A -/
theorem six_possible_values_for_A :
  ∃! (s : Finset Digit), 
    (∀ A ∈ s, ∃ (B X Y : Digit), A ≠ B ∧ A ≠ X ∧ A ≠ Y ∧ B ≠ X ∧ B ≠ Y ∧ X ≠ Y ∧ SumOperation A B X Y) ∧
    s.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_possible_values_for_A_l2112_211291


namespace NUMINAMATH_CALUDE_investment_change_l2112_211290

theorem investment_change (initial_investment : ℝ) 
  (loss_rate1 loss_rate3 gain_rate2 : ℝ) : 
  initial_investment = 200 →
  loss_rate1 = 0.1 →
  gain_rate2 = 0.15 →
  loss_rate3 = 0.05 →
  let year1 := initial_investment * (1 - loss_rate1)
  let year2 := year1 * (1 + gain_rate2)
  let year3 := year2 * (1 - loss_rate3)
  let percent_change := (year3 - initial_investment) / initial_investment * 100
  ∃ ε > 0, |percent_change + 1.68| < ε :=
by sorry

end NUMINAMATH_CALUDE_investment_change_l2112_211290


namespace NUMINAMATH_CALUDE_georginas_parrot_learning_rate_l2112_211237

/-- The number of phrases Georgina's parrot knows now -/
def current_phrases : ℕ := 17

/-- The number of phrases the parrot knew when Georgina bought it -/
def initial_phrases : ℕ := 3

/-- The number of days Georgina has had the parrot -/
def days_owned : ℕ := 49

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of phrases Georgina teaches her parrot per week -/
def phrases_per_week : ℚ :=
  (current_phrases - initial_phrases) / (days_owned / days_per_week)

theorem georginas_parrot_learning_rate :
  phrases_per_week = 2 := by sorry

end NUMINAMATH_CALUDE_georginas_parrot_learning_rate_l2112_211237


namespace NUMINAMATH_CALUDE_no_integer_solution_l2112_211217

theorem no_integer_solution : ∀ (x y z : ℤ), x ≠ 0 → 2*x^4 + 2*x^2*y^2 + y^4 ≠ z^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2112_211217


namespace NUMINAMATH_CALUDE_converse_proposition_l2112_211209

theorem converse_proposition : ∀ x : ℝ, (1 / (x - 1) ≥ 3) → (x ≤ 4 / 3) := by sorry

end NUMINAMATH_CALUDE_converse_proposition_l2112_211209


namespace NUMINAMATH_CALUDE_passing_percentage_is_40_l2112_211292

/-- The maximum marks possible in the exam -/
def max_marks : ℕ := 550

/-- The marks obtained by the student -/
def obtained_marks : ℕ := 200

/-- The number of marks by which the student failed -/
def fail_margin : ℕ := 20

/-- The passing percentage for the exam -/
def passing_percentage : ℚ :=
  (obtained_marks + fail_margin : ℚ) / max_marks * 100

theorem passing_percentage_is_40 :
  passing_percentage = 40 := by sorry

end NUMINAMATH_CALUDE_passing_percentage_is_40_l2112_211292


namespace NUMINAMATH_CALUDE_subset_condition_main_result_l2112_211247

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem subset_condition (a : ℝ) : B a ⊆ A ↔ a = 0 ∨ a = 1/3 ∨ a = 1/5 := by
  sorry

def solution_set : Set ℝ := {0, 1/3, 1/5}

theorem main_result : {a : ℝ | B a ⊆ A} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_main_result_l2112_211247


namespace NUMINAMATH_CALUDE_second_machine_time_l2112_211268

theorem second_machine_time (t1 t_combined : ℝ) (h1 : t1 = 9) (h2 : t_combined = 4.235294117647059) : 
  let t2 := (t1 * t_combined) / (t1 - t_combined)
  t2 = 8 := by sorry

end NUMINAMATH_CALUDE_second_machine_time_l2112_211268


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l2112_211229

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and between a line and a plane
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)

-- Define the "contained in" relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (a b : Line) (α : Plane) 
  (h1 : parallel_plane a α) 
  (h2 : parallel_line a b) 
  (h3 : ¬ contained_in b α) : 
  parallel_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l2112_211229


namespace NUMINAMATH_CALUDE_number_puzzle_l2112_211241

theorem number_puzzle : ∃ x : ℝ, 47 - 3 * x = 14 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2112_211241


namespace NUMINAMATH_CALUDE_complex_square_at_one_one_l2112_211246

theorem complex_square_at_one_one : 
  ∀ z : ℂ, (z.re = 1 ∧ z.im = 1) → z^2 = 2*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_square_at_one_one_l2112_211246


namespace NUMINAMATH_CALUDE_mary_book_count_l2112_211274

/-- The number of books Jason has -/
def jason_books : ℕ := 18

/-- The total number of books Jason and Mary have together -/
def total_books : ℕ := 60

/-- The number of books Mary has -/
def mary_books : ℕ := total_books - jason_books

theorem mary_book_count : mary_books = 42 := by
  sorry

end NUMINAMATH_CALUDE_mary_book_count_l2112_211274


namespace NUMINAMATH_CALUDE_bryan_bookshelves_l2112_211242

/-- The number of books on each bookshelf -/
def books_per_shelf : ℕ := 23

/-- The number of magazines on each bookshelf -/
def magazines_per_shelf : ℕ := 61

/-- The total number of books and magazines -/
def total_items : ℕ := 2436

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 29

theorem bryan_bookshelves :
  (books_per_shelf + magazines_per_shelf) * num_bookshelves = total_items :=
sorry

end NUMINAMATH_CALUDE_bryan_bookshelves_l2112_211242


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2112_211299

theorem right_triangle_perimeter : ∃ (a c : ℕ), 
  11^2 + a^2 = c^2 ∧ 11 + a + c = 132 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2112_211299


namespace NUMINAMATH_CALUDE_bouquet_39_roses_cost_l2112_211208

/-- Represents the cost of a bouquet of roses -/
structure BouquetCost where
  baseCost : ℝ
  additionalCostPerRose : ℝ

/-- Calculates the total cost of a bouquet given the number of roses -/
def totalCost (bc : BouquetCost) (numRoses : ℕ) : ℝ :=
  bc.baseCost + bc.additionalCostPerRose * numRoses

/-- Theorem: Given the conditions, a bouquet of 39 roses costs $58.75 -/
theorem bouquet_39_roses_cost
  (bc : BouquetCost)
  (h1 : bc.baseCost = 10)
  (h2 : totalCost bc 12 = 25) :
  totalCost bc 39 = 58.75 := by
  sorry

#check bouquet_39_roses_cost

end NUMINAMATH_CALUDE_bouquet_39_roses_cost_l2112_211208


namespace NUMINAMATH_CALUDE_fifth_power_sum_l2112_211244

theorem fifth_power_sum (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 2) :
  a^5 + b^5 = 19/4 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l2112_211244


namespace NUMINAMATH_CALUDE_train_length_l2112_211252

/-- The length of a train given its speed, the speed of a person it passes, and the time it takes to pass them. -/
theorem train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 63 →
  person_speed = 3 →
  passing_time = 53.99568034557235 →
  ∃ (length : ℝ), abs (length - 899.93) < 0.01 ∧
  length = (train_speed - person_speed) * (5 / 18) * passing_time :=
sorry

end NUMINAMATH_CALUDE_train_length_l2112_211252


namespace NUMINAMATH_CALUDE_sum_to_135_mod_7_l2112_211288

/-- The sum of integers from 1 to n -/
def sum_to (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that the sum of integers from 1 to 135, when divided by 7, has a remainder of 3 -/
theorem sum_to_135_mod_7 : sum_to 135 % 7 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_to_135_mod_7_l2112_211288


namespace NUMINAMATH_CALUDE_photo_lineup_arrangements_l2112_211279

def students : ℕ := 4
def teachers : ℕ := 3

def arrangements_teachers_together : ℕ := 720
def arrangements_teachers_together_students_split : ℕ := 144
def arrangements_teachers_apart : ℕ := 1440

theorem photo_lineup_arrangements :
  (students = 4 ∧ teachers = 3) →
  (arrangements_teachers_together = 720 ∧
   arrangements_teachers_together_students_split = 144 ∧
   arrangements_teachers_apart = 1440) := by
  sorry

end NUMINAMATH_CALUDE_photo_lineup_arrangements_l2112_211279


namespace NUMINAMATH_CALUDE_coefficient_x2y2_in_expansion_l2112_211265

-- Define the binomial expansion function
def binomialExpand (n : ℕ) (x : ℝ) : ℝ := (1 + x) ^ n

-- Define the coefficient extraction function
def coefficientOf (term : ℕ × ℕ) (expansion : ℝ → ℝ → ℝ) : ℝ :=
  sorry -- Placeholder for the actual implementation

theorem coefficient_x2y2_in_expansion :
  coefficientOf (2, 2) (fun x y => binomialExpand 3 x * binomialExpand 4 y) = 18 := by
  sorry

#check coefficient_x2y2_in_expansion

end NUMINAMATH_CALUDE_coefficient_x2y2_in_expansion_l2112_211265


namespace NUMINAMATH_CALUDE_set_membership_implies_value_l2112_211294

theorem set_membership_implies_value (a : ℝ) : 
  3 ∈ ({a, a^2 - 2*a} : Set ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_value_l2112_211294


namespace NUMINAMATH_CALUDE_cube_cut_possible_4_5_cube_cut_impossible_4_7_l2112_211234

/-- Represents a cut of a cube using four planes -/
structure CubeCut where
  planes : Fin 4 → Plane

/-- The maximum distance between any two points in a part resulting from a cube cut -/
def max_distance (cut : CubeCut) : ℝ := sorry

theorem cube_cut_possible_4_5 :
  ∃ (cut : CubeCut), max_distance cut < 4/5 := by sorry

theorem cube_cut_impossible_4_7 :
  ¬ ∃ (cut : CubeCut), max_distance cut < 4/7 := by sorry

end NUMINAMATH_CALUDE_cube_cut_possible_4_5_cube_cut_impossible_4_7_l2112_211234


namespace NUMINAMATH_CALUDE_allowance_spent_on_games_l2112_211213

theorem allowance_spent_on_games (total : ℝ) (books_frac snacks_frac music_frac : ℝ) : 
  total = 50 ∧ 
  books_frac = 1/4 ∧ 
  snacks_frac = 1/5 ∧ 
  music_frac = 2/5 → 
  total - (books_frac * total + snacks_frac * total + music_frac * total) = 7.5 := by
sorry

end NUMINAMATH_CALUDE_allowance_spent_on_games_l2112_211213


namespace NUMINAMATH_CALUDE_charlie_dana_difference_l2112_211226

/-- Represents the number of games won by each player -/
structure GameWins where
  perry : ℕ
  dana : ℕ
  charlie : ℕ
  phil : ℕ

/-- The conditions of the golf game results -/
def golf_results (g : GameWins) : Prop :=
  g.perry = g.dana + 5 ∧
  g.charlie < g.dana ∧
  g.phil = g.charlie + 3 ∧
  g.phil = 12 ∧
  g.perry = g.phil + 4

theorem charlie_dana_difference (g : GameWins) (h : golf_results g) :
  g.dana - g.charlie = 2 := by
  sorry

end NUMINAMATH_CALUDE_charlie_dana_difference_l2112_211226


namespace NUMINAMATH_CALUDE_driving_equation_correct_l2112_211227

/-- Represents a driving scenario where the actual speed is faster than planned. -/
structure DrivingScenario where
  distance : ℝ
  planned_speed : ℝ
  actual_speed : ℝ
  time_saved : ℝ

/-- The equation correctly represents the driving scenario. -/
theorem driving_equation_correct (scenario : DrivingScenario) 
  (h1 : scenario.distance = 240)
  (h2 : scenario.actual_speed = 1.5 * scenario.planned_speed)
  (h3 : scenario.time_saved = 1)
  (h4 : scenario.planned_speed > 0) :
  scenario.distance / scenario.planned_speed - scenario.distance / scenario.actual_speed = scenario.time_saved := by
  sorry

#check driving_equation_correct

end NUMINAMATH_CALUDE_driving_equation_correct_l2112_211227


namespace NUMINAMATH_CALUDE_count_sevens_20_to_119_l2112_211240

/-- Count of digit 7 in a number -/
def countSevens (n : ℕ) : ℕ := sorry

/-- Sum of countSevens for a range of natural numbers -/
def sumCountSevens (start finish : ℕ) : ℕ := sorry

theorem count_sevens_20_to_119 : sumCountSevens 20 119 = 19 := by sorry

end NUMINAMATH_CALUDE_count_sevens_20_to_119_l2112_211240


namespace NUMINAMATH_CALUDE_nellie_legos_l2112_211289

theorem nellie_legos (L : ℕ) : 
  L - 57 - 24 = 299 → L = 380 := by
sorry

end NUMINAMATH_CALUDE_nellie_legos_l2112_211289


namespace NUMINAMATH_CALUDE_common_tangents_from_guiding_circles_l2112_211210

/-- Represents an ellipse with its foci and semi-major axis -/
structure Ellipse where
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  semiMajorAxis : ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Enum representing the possible number of common tangents -/
inductive NumCommonTangents
  | zero
  | one
  | two

/-- Function to determine the number of intersections between two circles -/
def circleIntersections (c1 c2 : Circle) : NumCommonTangents :=
  sorry

/-- Function to get the guiding circle of an ellipse for a given focus -/
def guidingCircle (e : Ellipse) (f : ℝ × ℝ) : Circle :=
  sorry

/-- Theorem stating that the number of common tangents between two ellipses
    sharing a focus is determined by the intersection of their guiding circles -/
theorem common_tangents_from_guiding_circles 
  (e1 e2 : Ellipse) 
  (h : e1.focus1 = e2.focus1) :
  ∃ (f : ℝ × ℝ), 
    let c1 := guidingCircle e1 f
    let c2 := guidingCircle e2 f
    circleIntersections c1 c2 = NumCommonTangents.zero ∨
    circleIntersections c1 c2 = NumCommonTangents.one ∨
    circleIntersections c1 c2 = NumCommonTangents.two :=
  sorry

end NUMINAMATH_CALUDE_common_tangents_from_guiding_circles_l2112_211210


namespace NUMINAMATH_CALUDE_sum_of_roots_even_function_l2112_211249

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define a function that has exactly 4 real roots
def HasFourRealRoots (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)

-- Theorem statement
theorem sum_of_roots_even_function
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_four_roots : HasFourRealRoots f) :
  ∃ a b c d : ℝ, (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧ a + b + c + d = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_even_function_l2112_211249


namespace NUMINAMATH_CALUDE_jacket_pricing_l2112_211269

theorem jacket_pricing (x : ℝ) : 
  (0.8 * (1 + 0.5) * x = x + 28) ↔ 
  (∃ (markup : ℝ) (discount : ℝ) (profit : ℝ), 
    markup = 0.5 ∧ 
    discount = 0.2 ∧ 
    profit = 28 ∧ 
    (1 - discount) * (1 + markup) * x - x = profit) :=
by sorry

end NUMINAMATH_CALUDE_jacket_pricing_l2112_211269


namespace NUMINAMATH_CALUDE_logarithm_power_sum_l2112_211263

theorem logarithm_power_sum (a b : ℝ) (ha : a = Real.log 25) (hb : b = Real.log 36) :
  (5 : ℝ) ^ (a / b) + (6 : ℝ) ^ (b / a) = 11 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_power_sum_l2112_211263


namespace NUMINAMATH_CALUDE_infinitely_many_a_for_positive_integer_l2112_211286

theorem infinitely_many_a_for_positive_integer (n : ℕ) :
  ∃ (f : ℕ → ℤ), Function.Injective f ∧
  ∀ (k : ℕ), (n^6 + 3 * (f k) : ℤ) > 0 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_a_for_positive_integer_l2112_211286


namespace NUMINAMATH_CALUDE_product_inequality_l2112_211260

theorem product_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l2112_211260


namespace NUMINAMATH_CALUDE_molecular_weight_5_moles_AlBr3_l2112_211267

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The number of Aluminum atoms in AlBr3 -/
def num_Al : ℕ := 1

/-- The number of Bromine atoms in AlBr3 -/
def num_Br : ℕ := 3

/-- The number of moles of AlBr3 -/
def num_moles : ℝ := 5

/-- The molecular weight of AlBr3 in g/mol -/
def molecular_weight_AlBr3 : ℝ :=
  num_Al * atomic_weight_Al + num_Br * atomic_weight_Br

/-- Theorem stating that the molecular weight of 5 moles of AlBr3 is 1333.40 grams -/
theorem molecular_weight_5_moles_AlBr3 :
  num_moles * molecular_weight_AlBr3 = 1333.40 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_5_moles_AlBr3_l2112_211267


namespace NUMINAMATH_CALUDE_inequality_proof_l2112_211284

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * b * c) + 1 ≥ 3 * (1 / (a^2 + b^2 + c^2) + 1 / (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2112_211284


namespace NUMINAMATH_CALUDE_curve_tangent_perpendicular_l2112_211257

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x + 1

-- Define the tangent line
def tangent_slope (a : ℝ) : ℝ := -a

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x + y + 10 = 0

-- State the theorem
theorem curve_tangent_perpendicular (a : ℝ) (h : a ≠ 0) :
  (tangent_slope a * 2 = -1) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_curve_tangent_perpendicular_l2112_211257


namespace NUMINAMATH_CALUDE_freds_remaining_balloons_l2112_211239

/-- The number of green balloons Fred has after giving some away -/
def remaining_balloons (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Fred's remaining balloons equals the difference between initial and given away -/
theorem freds_remaining_balloons :
  remaining_balloons 709 221 = 488 := by
  sorry

end NUMINAMATH_CALUDE_freds_remaining_balloons_l2112_211239


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2112_211270

/-- Given a group of 6 persons where one person is replaced by a new person weighing 79.8 kg,
    and the average weight increases by 1.8 kg, prove that the replaced person weighed 69 kg. -/
theorem weight_of_replaced_person
  (initial_count : ℕ)
  (new_person_weight : ℝ)
  (average_increase : ℝ)
  (h1 : initial_count = 6)
  (h2 : new_person_weight = 79.8)
  (h3 : average_increase = 1.8) :
  ∃ (replaced_weight : ℝ),
    replaced_weight = 69 ∧
    new_person_weight = replaced_weight + (initial_count : ℝ) * average_increase :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2112_211270


namespace NUMINAMATH_CALUDE_equation_solution_l2112_211258

theorem equation_solution (x : ℝ) : 
  1 - 6/x + 9/x^2 - 4/x^3 = 0 → (3/x = 3 ∨ 3/x = 3/4) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2112_211258


namespace NUMINAMATH_CALUDE_negation_of_implication_l2112_211211

theorem negation_of_implication (a b c : ℝ) :
  ¬(a + b + c = 1 → a^2 + b^2 + c^2 ≤ 1/9) ↔ (a + b + c ≠ 1 → a^2 + b^2 + c^2 > 1/9) := by
sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2112_211211


namespace NUMINAMATH_CALUDE_number_of_incorrect_statements_l2112_211216

-- Define the triangles
def triangle1 (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 9^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 12^2 ∧
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2)

def triangle2 (a b c : ℝ) : Prop :=
  a = 7 ∧ b = 24 ∧ c = 25

def triangle3 (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 6^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 8^2 ∧
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2)

def triangle4 (a b c : ℝ) : Prop :=
  (a = 3 ∧ b = 3 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 3)

-- Define the statements
def statement1 (A B C : ℝ × ℝ) : Prop :=
  triangle1 A B C → abs ((B.2 - A.2) * C.1 + (A.1 - B.1) * C.2 + (B.1 * A.2 - A.1 * B.2)) / 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 9

def statement2 (a b c : ℝ) : Prop :=
  triangle2 a b c → a^2 + b^2 = c^2

def statement3 (A B C : ℝ × ℝ) : Prop :=
  triangle3 A B C → ∃ (M : ℝ × ℝ), 
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    Real.sqrt ((M.1 - C.1)^2 + (M.2 - C.2)^2) = 5

def statement4 (a b c : ℝ) : Prop :=
  triangle4 a b c → a + b + c = 13

-- Theorem to prove
theorem number_of_incorrect_statements :
  ∃ (A1 B1 C1 A3 B3 C3 : ℝ × ℝ) (a2 b2 c2 a4 b4 c4 : ℝ),
    (¬ statement1 A1 B1 C1) ∧
    statement2 a2 b2 c2 ∧
    (¬ statement3 A3 B3 C3) ∧
    (¬ statement4 a4 b4 c4) := by
  sorry

end NUMINAMATH_CALUDE_number_of_incorrect_statements_l2112_211216


namespace NUMINAMATH_CALUDE_fraction_problem_l2112_211218

theorem fraction_problem (N : ℝ) (F : ℝ) : 
  N = 8 → 0.5 * N = F * N + 2 → F = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2112_211218


namespace NUMINAMATH_CALUDE_triangle_with_specific_circumcircle_l2112_211231

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def circumscribed_circle_diameter (a b c : ℕ) : ℚ :=
  (a * b * c : ℚ) / ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c) : ℚ) * 4

theorem triangle_with_specific_circumcircle :
  ∀ a b c : ℕ,
    is_triangle a b c →
    circumscribed_circle_diameter a b c = 25/4 →
    (a = 5 ∧ b = 5 ∧ c = 6) ∨ (a = 5 ∧ b = 6 ∧ c = 5) ∨ (a = 6 ∧ b = 5 ∧ c = 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_specific_circumcircle_l2112_211231


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l2112_211221

theorem absolute_value_simplification : |(-4^2 + 6)| = 10 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l2112_211221


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l2112_211248

theorem quadratic_form_equivalence : ∃ (a b c : ℝ), 
  (∀ x, x * (x + 2) = 5 * (x - 2) ↔ a * x^2 + b * x + c = 0) ∧ 
  a = 1 ∧ b = -3 ∧ c = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l2112_211248


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l2112_211235

theorem smallest_positive_integer_satisfying_congruences :
  ∃! x : ℕ+, 
    (45 * x.val + 9) % 25 = 3 ∧
    (2 * x.val) % 5 = 3 ∧
    ∀ y : ℕ+, 
      ((45 * y.val + 9) % 25 = 3 ∧ (2 * y.val) % 5 = 3) → x ≤ y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l2112_211235


namespace NUMINAMATH_CALUDE_no_real_solutions_l2112_211245

theorem no_real_solutions : ¬∃ (x : ℝ), (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2112_211245


namespace NUMINAMATH_CALUDE_gladys_age_ratio_l2112_211225

def gladys_age : ℕ := 30

def billy_age : ℕ := gladys_age / 3

def lucas_age : ℕ := 8 - 3

def sum_billy_lucas : ℕ := billy_age + lucas_age

theorem gladys_age_ratio : 
  gladys_age / sum_billy_lucas = 2 :=
by sorry

end NUMINAMATH_CALUDE_gladys_age_ratio_l2112_211225


namespace NUMINAMATH_CALUDE_triangle_area_l2112_211212

/-- The area of a triangle with side lengths √29, √13, and √34 is 19/2 -/
theorem triangle_area (a b c : ℝ) (ha : a = Real.sqrt 29) (hb : b = Real.sqrt 13) (hc : c = Real.sqrt 34) :
  (1/2) * b * c * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*b*c))^2) = 19/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2112_211212


namespace NUMINAMATH_CALUDE_probability_two_white_is_three_tenths_l2112_211230

def total_balls : ℕ := 5
def white_balls : ℕ := 3
def drawn_balls : ℕ := 2

def probability_two_white : ℚ := (white_balls.choose drawn_balls : ℚ) / (total_balls.choose drawn_balls)

theorem probability_two_white_is_three_tenths :
  probability_two_white = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_two_white_is_three_tenths_l2112_211230


namespace NUMINAMATH_CALUDE_cubic_function_derivative_l2112_211256

/-- Given a cubic function f(x) = ax³ + 3x² + 2, prove that if f'(-1) = 4, then a = 10/3 -/
theorem cubic_function_derivative (a : ℝ) :
  let f := λ x : ℝ => a * x^3 + 3 * x^2 + 2
  let f' := λ x : ℝ => 3 * a * x^2 + 6 * x
  f' (-1) = 4 → a = 10/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_l2112_211256


namespace NUMINAMATH_CALUDE_root_product_l2112_211283

theorem root_product (a b : ℝ) : 
  (a^2 + 2*a - 2023 = 0) → 
  (b^2 + 2*b - 2023 = 0) → 
  (a + 1) * (b + 1) = -2024 := by
sorry

end NUMINAMATH_CALUDE_root_product_l2112_211283


namespace NUMINAMATH_CALUDE_vector_linear_combination_l2112_211236

/-- Given two vectors in ℝ², prove that their linear combination results in the expected vector. -/
theorem vector_linear_combination (a b : ℝ × ℝ) (h1 : a = (-1, 0)) (h2 : b = (0, 2)) :
  (2 : ℝ) • a - (3 : ℝ) • b = (-2, -6) := by sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l2112_211236


namespace NUMINAMATH_CALUDE_jose_join_time_l2112_211280

/-- Represents the problem of determining when Jose joined Tom's business --/
theorem jose_join_time (tom_investment jose_investment total_profit jose_profit : ℚ) 
  (h1 : tom_investment = 3000)
  (h2 : jose_investment = 4500)
  (h3 : total_profit = 5400)
  (h4 : jose_profit = 3000) :
  let x := (12 * tom_investment * (total_profit - jose_profit)) / 
           (jose_investment * jose_profit) - 12
  x = 2 := by sorry

end NUMINAMATH_CALUDE_jose_join_time_l2112_211280


namespace NUMINAMATH_CALUDE_inequality_proof_l2112_211222

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 - x^2)⁻¹ + (1 - y^2)⁻¹ ≥ 2 * (1 - x*y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2112_211222


namespace NUMINAMATH_CALUDE_min_rectangles_to_cover_l2112_211251

/-- Represents a corner in the shape --/
inductive Corner
| Type1
| Type2

/-- Represents the shape with its corner configuration --/
structure Shape where
  type1_corners : Nat
  type2_corners : Nat

/-- Represents a rectangle that can cover cells and corners --/
structure Rectangle where
  covered_corners : List Corner

/-- Defines the properties of the shape as given in the problem --/
def problem_shape : Shape :=
  { type1_corners := 12
  , type2_corners := 12 }

/-- Theorem stating the minimum number of rectangles needed to cover the shape --/
theorem min_rectangles_to_cover (s : Shape) 
  (h1 : s.type1_corners = problem_shape.type1_corners) 
  (h2 : s.type2_corners = problem_shape.type2_corners) :
  ∃ (rectangles : List Rectangle), 
    (rectangles.length = 12) ∧ 
    (∀ c : Corner, c ∈ Corner.Type1 :: List.replicate s.type1_corners Corner.Type1 ++ 
                   Corner.Type2 :: List.replicate s.type2_corners Corner.Type2 → 
      ∃ r ∈ rectangles, c ∈ r.covered_corners) :=
by sorry

end NUMINAMATH_CALUDE_min_rectangles_to_cover_l2112_211251


namespace NUMINAMATH_CALUDE_cakes_per_friend_l2112_211264

def total_cakes : ℕ := 8
def num_friends : ℕ := 4

theorem cakes_per_friend :
  total_cakes / num_friends = 2 :=
by sorry

end NUMINAMATH_CALUDE_cakes_per_friend_l2112_211264


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2112_211253

-- Problem 1
theorem problem_1 : Real.sqrt 5 ^ 2 + |(-3)| - (Real.pi + Real.sqrt 3) ^ 0 = 7 := by sorry

-- Problem 2
theorem problem_2 : 
  Set.Ioo (-1 : ℝ) 2 = {x : ℝ | 5 * x - 10 ≤ 0 ∧ x + 3 > -2 * x} := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2112_211253


namespace NUMINAMATH_CALUDE_pet_shop_total_cost_l2112_211200

/-- Calculates the total cost of purchasing all pets with discounts -/
def total_cost_with_discounts (puppy1_price puppy2_price kitten1_price kitten2_price 
                               parakeet1_price parakeet2_price parakeet3_price : ℚ) : ℚ :=
  let puppy_total := puppy1_price + puppy2_price
  let puppy_discount := puppy_total * (5 / 100)
  let puppy_cost := puppy_total - puppy_discount

  let kitten_total := kitten1_price + kitten2_price
  let kitten_discount := kitten_total * (10 / 100)
  let kitten_cost := kitten_total - kitten_discount

  let parakeet_total := parakeet1_price + parakeet2_price + parakeet3_price
  let parakeet_discount := min parakeet1_price (min parakeet2_price parakeet3_price) / 2
  let parakeet_cost := parakeet_total - parakeet_discount

  puppy_cost + kitten_cost + parakeet_cost

/-- The theorem stating the total cost of purchasing all pets with discounts -/
theorem pet_shop_total_cost :
  total_cost_with_discounts 72 78 48 52 10 12 14 = 263.5 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_total_cost_l2112_211200


namespace NUMINAMATH_CALUDE_average_price_per_book_l2112_211278

theorem average_price_per_book (books_shop1 : ℕ) (cost_shop1 : ℕ) (books_shop2 : ℕ) (cost_shop2 : ℕ) 
  (h1 : books_shop1 = 42)
  (h2 : cost_shop1 = 520)
  (h3 : books_shop2 = 22)
  (h4 : cost_shop2 = 248) :
  (cost_shop1 + cost_shop2) / (books_shop1 + books_shop2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_price_per_book_l2112_211278


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2112_211275

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 2 → x + y ≥ 3) ∧
  (∃ x y : ℝ, x + y ≥ 3 ∧ ¬(x ≥ 1 ∧ y ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2112_211275


namespace NUMINAMATH_CALUDE_max_draws_for_cmwmc_l2112_211277

/-- Represents the number of tiles of each letter in the bag -/
structure TileCounts :=
  (c : Nat)
  (m : Nat)
  (w : Nat)

/-- Represents the number of tiles needed to spell the word -/
structure WordCounts :=
  (c : Nat)
  (m : Nat)
  (w : Nat)

/-- The maximum number of tiles that need to be drawn -/
def maxDraws (bag : TileCounts) (word : WordCounts) : Nat :=
  bag.c + bag.m + bag.w - (word.c - 1) - (word.m - 1) - (word.w - 1)

/-- Theorem stating the maximum number of draws for the given problem -/
theorem max_draws_for_cmwmc :
  let bag := TileCounts.mk 8 8 8
  let word := WordCounts.mk 2 2 1
  maxDraws bag word = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_draws_for_cmwmc_l2112_211277


namespace NUMINAMATH_CALUDE_saree_price_calculation_l2112_211255

theorem saree_price_calculation (P : ℝ) : 
  P * (1 - 0.20) * (1 - 0.15) = 306 → P = 450 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l2112_211255


namespace NUMINAMATH_CALUDE_expression_factorization_l2112_211261

theorem expression_factorization (a b c : ℝ) : 
  2*(a+b)*(b+c)*(a+3*b+2*c) + 2*(b+c)*(c+a)*(b+3*c+2*a) + 
  2*(c+a)*(a+b)*(c+3*a+2*b) + 9*(a+b)*(b+c)*(c+a) = 
  (a + 3*b + 2*c)*(b + 3*c + 2*a)*(c + 3*a + 2*b) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l2112_211261


namespace NUMINAMATH_CALUDE_system_solution_condition_l2112_211219

theorem system_solution_condition (n p : ℕ) :
  (∃ x y : ℕ+, x + p * y = n ∧ x + y = p^2) ↔
  (p > 1 ∧ (p - 1) ∣ (n - 1) ∧ ∀ k : ℕ+, n ≠ p^(k : ℕ)) :=
sorry

end NUMINAMATH_CALUDE_system_solution_condition_l2112_211219


namespace NUMINAMATH_CALUDE_min_pages_for_baseball_cards_l2112_211215

/-- Represents the number of cards that can be held by each type of page -/
structure PageCapacity where
  x : Nat
  y : Nat

/-- Calculates the minimum number of pages needed to hold all cards -/
def minPages (totalCards : Nat) (capacity : PageCapacity) : Nat :=
  let fullXPages := totalCards / capacity.x
  let remainingCards := totalCards % capacity.x
  if remainingCards = 0 then
    fullXPages
  else if remainingCards ≤ capacity.y then
    fullXPages + 1
  else
    fullXPages + 2

/-- Theorem stating the minimum number of pages needed for the given problem -/
theorem min_pages_for_baseball_cards :
  let totalCards := 1040
  let capacity : PageCapacity := { x := 12, y := 10 }
  minPages totalCards capacity = 87 := by
  sorry

#eval minPages 1040 { x := 12, y := 10 }

end NUMINAMATH_CALUDE_min_pages_for_baseball_cards_l2112_211215


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l2112_211203

theorem largest_integer_less_than_100_remainder_5_mod_8 :
  ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l2112_211203


namespace NUMINAMATH_CALUDE_portrait_problem_l2112_211272

theorem portrait_problem (total_students : ℕ) (before_lunch : ℕ) (after_lunch : ℕ) 
  (h1 : total_students = 24)
  (h2 : before_lunch = total_students / 3)
  (h3 : after_lunch = 10) :
  total_students - (before_lunch + after_lunch) = 6 := by
  sorry

end NUMINAMATH_CALUDE_portrait_problem_l2112_211272


namespace NUMINAMATH_CALUDE_smallest_sum_a_b_l2112_211254

theorem smallest_sum_a_b (a b : ℕ+) 
  (h : (1 : ℚ) / a + (1 : ℚ) / (2 * a) + (1 : ℚ) / (3 * a) = (1 : ℚ) / (b^2 - 2*b)) : 
  ∀ (x y : ℕ+), 
    ((1 : ℚ) / x + (1 : ℚ) / (2 * x) + (1 : ℚ) / (3 * x) = (1 : ℚ) / (y^2 - 2*y)) → 
    (x + y : ℕ) ≥ (a + b : ℕ) ∧ (a + b : ℕ) = 50 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_a_b_l2112_211254


namespace NUMINAMATH_CALUDE_wire_cutting_l2112_211273

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) : 
  total_length = 35 →
  ratio = 2 / 5 →
  ∃ (shorter_piece longer_piece : ℝ),
    shorter_piece + longer_piece = total_length ∧
    longer_piece = ratio * shorter_piece ∧
    shorter_piece = 25 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l2112_211273


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l2112_211220

theorem gcd_lcm_problem (a b : ℕ+) : 
  Nat.gcd a b = 21 ∧ Nat.lcm a b = 3969 → 
  (a = 21 ∧ b = 3969) ∨ (a = 147 ∧ b = 567) ∨ (a = 3969 ∧ b = 21) ∨ (a = 567 ∧ b = 147) :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l2112_211220


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l2112_211295

/-- The cost ratio of a muffin to a banana given Susie and Calvin's purchases -/
theorem muffin_banana_cost_ratio :
  ∀ (m b c : ℚ),
  (5 * m + 2 * b + 3 * c = 1) →  -- Normalize Susie's purchase to 1
  (4 * m + 18 * b + c = 3) →     -- Calvin's purchase is 3 times Susie's
  (c = 2 * b) →                  -- A cookie costs twice as much as a banana
  (m / b = 4 / 11) :=
by sorry

end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l2112_211295


namespace NUMINAMATH_CALUDE_lentil_dishes_count_l2112_211298

/-- Represents the menu of a vegan restaurant -/
structure VeganMenu :=
  (total_dishes : ℕ)
  (beans_lentils : ℕ)
  (beans_seitan : ℕ)
  (only_beans : ℕ)
  (only_seitan : ℕ)
  (only_lentils : ℕ)

/-- The conditions of the vegan restaurant menu problem -/
def menu_conditions (m : VeganMenu) : Prop :=
  m.total_dishes = 10 ∧
  m.beans_lentils = 2 ∧
  m.beans_seitan = 2 ∧
  m.only_beans = (m.total_dishes - m.beans_lentils - m.beans_seitan) / 2 ∧
  m.only_beans = 3 * m.only_seitan

/-- Theorem stating that the number of dishes including lentils is 2 -/
theorem lentil_dishes_count (m : VeganMenu) (h : menu_conditions m) : 
  m.beans_lentils + m.only_lentils = 2 := by
  sorry


end NUMINAMATH_CALUDE_lentil_dishes_count_l2112_211298


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2112_211243

variable (a b : ℝ)
variable (x y : ℝ)

theorem simplify_expression_1 : 4 * a^2 + 2 * (3 * a * b - 2 * a^2) - (7 * a * b - 1) = -a * b + 1 := by
  sorry

theorem simplify_expression_2 : 3 * (x^2 * y - 1/2 * x * y^2) - 1/2 * (4 * x^2 * y - 3 * x * y^2) = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2112_211243


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2112_211297

theorem parallelogram_base_length 
  (area : ℝ) (height : ℝ) (base : ℝ) 
  (h1 : area = 32) 
  (h2 : height = 8) 
  (h3 : area = base * height) : 
  base = 4 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2112_211297


namespace NUMINAMATH_CALUDE_desired_annual_profit_l2112_211259

def annual_fixed_costs : ℕ := 50200000
def average_cost_per_vehicle : ℕ := 5000
def forecasted_sales : ℕ := 20000
def selling_price_per_car : ℕ := 9035

theorem desired_annual_profit :
  (selling_price_per_car * forecasted_sales) - 
  (annual_fixed_costs + average_cost_per_vehicle * forecasted_sales) = 30500000 := by
  sorry

end NUMINAMATH_CALUDE_desired_annual_profit_l2112_211259


namespace NUMINAMATH_CALUDE_sine_phase_shift_l2112_211201

/-- The phase shift of the sine function y = sin(4x + π/2) is π/8 units to the left. -/
theorem sine_phase_shift :
  let f : ℝ → ℝ := λ x ↦ Real.sin (4 * x + π / 2)
  ∃ (φ : ℝ), φ = π / 8 ∧
    ∀ x, f x = Real.sin (4 * (x + φ)) := by
  sorry

end NUMINAMATH_CALUDE_sine_phase_shift_l2112_211201


namespace NUMINAMATH_CALUDE_books_sold_to_store_l2112_211228

def book_problem (initial_books : ℕ) (book_club_months : ℕ) (bookstore_books : ℕ) 
  (yard_sale_books : ℕ) (daughter_books : ℕ) (mother_books : ℕ) (donated_books : ℕ) 
  (final_books : ℕ) : ℕ :=
  let total_acquired := initial_books + book_club_months + bookstore_books + 
                        yard_sale_books + daughter_books + mother_books
  let before_selling := total_acquired - donated_books
  before_selling - final_books

theorem books_sold_to_store : 
  book_problem 72 12 5 2 1 4 12 81 = 3 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_to_store_l2112_211228


namespace NUMINAMATH_CALUDE_power_of_one_third_l2112_211266

theorem power_of_one_third (a b : ℕ) : 
  (2^a : ℕ) * (5^b : ℕ) = 200 → 
  (∀ k : ℕ, 2^k ∣ 200 → k ≤ a) →
  (∀ k : ℕ, 5^k ∣ 200 → k ≤ b) →
  (1/3 : ℚ)^(b - a) = 3 := by sorry

end NUMINAMATH_CALUDE_power_of_one_third_l2112_211266


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_and_cube_root_l2112_211204

theorem min_value_sum_reciprocals_and_cube_root (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1/x + 1/y + 1/z + (x*y*z)^(1/3 : ℝ) ≥ 4 ∧ 
  (1/x + 1/y + 1/z + (x*y*z)^(1/3 : ℝ) = 4 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_and_cube_root_l2112_211204


namespace NUMINAMATH_CALUDE_power_function_m_values_l2112_211206

/-- A function is a power function if it's of the form f(x) = ax^n, where a ≠ 0 and n is a real number -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^n

/-- The given function f(x) = (m^2 - m - 1)x^3 -/
def f (m : ℝ) : ℝ → ℝ := fun x ↦ (m^2 - m - 1) * x^3

/-- Theorem: If f(x) = (m^2 - m - 1)x^3 is a power function, then m = -1 or m = 2 -/
theorem power_function_m_values (m : ℝ) : IsPowerFunction (f m) → m = -1 ∨ m = 2 := by
  sorry


end NUMINAMATH_CALUDE_power_function_m_values_l2112_211206


namespace NUMINAMATH_CALUDE_greatest_good_set_size_l2112_211287

/-- A set S of positive integers is "good" if there exists a coloring of positive integers
    with k colors such that no element from S can be written as the sum of two distinct
    positive integers having the same color. -/
def IsGood (S : Set ℕ) (k : ℕ) : Prop :=
  ∃ (c : ℕ → Fin k), ∀ s ∈ S, ∀ x y : ℕ, x < y → x + y = s → c x ≠ c y

/-- The set S defined as {a+1, a+2, ..., a+t} for some positive integer a -/
def S (a t : ℕ) : Set ℕ := {n : ℕ | a + 1 ≤ n ∧ n ≤ a + t}

theorem greatest_good_set_size (k : ℕ) (h : k > 1) :
  (∃ t : ℕ, ∀ a : ℕ, a > 0 → IsGood (S a t) k ∧
    ∀ t' : ℕ, t' > t → ∃ a : ℕ, a > 0 ∧ ¬IsGood (S a t') k) ∧
  (∀ t : ℕ, (∀ a : ℕ, a > 0 → IsGood (S a t) k) → t ≤ 2 * k - 2) :=
sorry

end NUMINAMATH_CALUDE_greatest_good_set_size_l2112_211287


namespace NUMINAMATH_CALUDE_family_age_difference_l2112_211293

/-- Represents a family with changing composition over time -/
structure Family where
  initialSize : ℕ
  initialAvgAge : ℕ
  timePassed : ℕ
  currentSize : ℕ
  currentAvgAge : ℕ
  youngestChildAge : ℕ

/-- The age difference between the two youngest children in the family -/
def ageDifference (f : Family) : ℕ := sorry

theorem family_age_difference (f : Family)
  (h1 : f.initialSize = 4)
  (h2 : f.initialAvgAge = 24)
  (h3 : f.timePassed = 10)
  (h4 : f.currentSize = 6)
  (h5 : f.currentAvgAge = 24)
  (h6 : f.youngestChildAge = 3) :
  ageDifference f = 2 := by sorry

end NUMINAMATH_CALUDE_family_age_difference_l2112_211293


namespace NUMINAMATH_CALUDE_days_without_visits_l2112_211205

def days_in_year : ℕ := 366

def visit_period_1 : ℕ := 6
def visit_period_2 : ℕ := 8
def visit_period_3 : ℕ := 10

def days_with_visits (period : ℕ) : ℕ := days_in_year / period

def lcm_two (a b : ℕ) : ℕ := Nat.lcm a b
def lcm_three (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

def days_with_two_visits (period1 period2 : ℕ) : ℕ := days_in_year / (lcm_two period1 period2)

def days_with_three_visits (period1 period2 period3 : ℕ) : ℕ := days_in_year / (lcm_three period1 period2 period3)

theorem days_without_visits :
  days_in_year - 
  ((days_with_visits visit_period_1 + days_with_visits visit_period_2 + days_with_visits visit_period_3) -
   (days_with_two_visits visit_period_1 visit_period_2 + 
    days_with_two_visits visit_period_1 visit_period_3 + 
    days_with_two_visits visit_period_2 visit_period_3) +
   days_with_three_visits visit_period_1 visit_period_2 visit_period_3) = 257 :=
by sorry

end NUMINAMATH_CALUDE_days_without_visits_l2112_211205


namespace NUMINAMATH_CALUDE_subtraction_absolute_value_l2112_211233

theorem subtraction_absolute_value : ∃ (x y : ℝ), 
  (|9 - 4| - |x - y| = 3) ∧ (|x - y| = 2) :=
by sorry

end NUMINAMATH_CALUDE_subtraction_absolute_value_l2112_211233


namespace NUMINAMATH_CALUDE_factor_condition_l2112_211232

theorem factor_condition (t : ℚ) :
  (∃ k : ℚ, ∀ x, 4*x^2 + 11*x - 3 = (x - t) * k) ↔ (t = 1/4 ∨ t = -3) := by
sorry

end NUMINAMATH_CALUDE_factor_condition_l2112_211232


namespace NUMINAMATH_CALUDE_glenville_population_l2112_211250

theorem glenville_population (h p : ℕ) : 
  (∃ h p, 13 * h + 6 * p = 48) ∧
  (∃ h p, 13 * h + 6 * p = 52) ∧
  (∃ h p, 13 * h + 6 * p = 65) ∧
  (∃ h p, 13 * h + 6 * p = 75) ∧
  (∀ h p, 13 * h + 6 * p ≠ 70) :=
by sorry

end NUMINAMATH_CALUDE_glenville_population_l2112_211250


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l2112_211271

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (correct_marks : ℕ) 
  (incorrect_marks : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 65)
  (h3 : correct_marks = 3)
  (h4 : incorrect_marks = 2) :
  ∃ (correct_sums : ℕ) (incorrect_sums : ℕ),
    correct_sums + incorrect_sums = total_sums ∧
    (correct_sums : ℤ) * correct_marks - incorrect_sums * incorrect_marks = total_marks ∧
    correct_sums = 25 :=
by sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l2112_211271


namespace NUMINAMATH_CALUDE_special_triangle_properties_l2112_211202

/-- An acute triangle ABC with specific properties -/
structure SpecialTriangle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Properties of the triangle
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  law_of_sines : a / Real.sin A = b / Real.sin B
  special_relation : 2 * a * Real.sin B = Real.sqrt 3 * b
  side_a : a = Real.sqrt 7
  side_c : c = 2

/-- The main theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  t.A = π/3 ∧ 
  (1/2 : ℝ) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l2112_211202
