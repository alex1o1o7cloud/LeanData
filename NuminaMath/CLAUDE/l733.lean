import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_digits_special_product_l733_73389

/-- Sum of digits function -/
def sum_of_digits (x : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem sum_of_digits_special_product (m n : ℕ) (d : ℕ) :
  m > 0 → n > 0 → d > 0 → d ≤ n → d = (Nat.digits 10 m).length →
  sum_of_digits ((10^n - 1) * m) = 9 * n := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_special_product_l733_73389


namespace NUMINAMATH_CALUDE_brick_width_calculation_l733_73343

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  height : ℝ
  thickness : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.height * w.thickness

theorem brick_width_calculation (brick : BrickDimensions) (wall : WallDimensions) 
    (h1 : brick.length = 25)
    (h2 : brick.height = 6)
    (h3 : wall.length = 800)
    (h4 : wall.height = 600)
    (h5 : wall.thickness = 22.5)
    (h6 : (6400 : ℝ) * brickVolume brick = wallVolume wall) :
    brick.width = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l733_73343


namespace NUMINAMATH_CALUDE_ExistEvenOddComposition_l733_73338

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the property of being an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of a function not being identically zero
def NotIdenticallyZero (f : ℝ → ℝ) : Prop := ∃ x, f x ≠ 0

-- State the theorem
theorem ExistEvenOddComposition :
  ∃ (p q : ℝ → ℝ), IsEven p ∧ IsOdd (p ∘ q) ∧ NotIdenticallyZero (p ∘ q) := by
  sorry

end NUMINAMATH_CALUDE_ExistEvenOddComposition_l733_73338


namespace NUMINAMATH_CALUDE_max_knights_between_knights_theorem_l733_73376

/-- Represents the seating arrangement around a round table -/
structure SeatingArrangement where
  knights : ℕ
  samurais : ℕ
  knights_with_samurai_right : ℕ

/-- The maximum number of knights that could be seated next to two other knights -/
def max_knights_between_knights (arrangement : SeatingArrangement) : ℕ :=
  arrangement.knights - (arrangement.knights_with_samurai_right + 1)

/-- Theorem stating the maximum number of knights between knights for the given arrangement -/
theorem max_knights_between_knights_theorem (arrangement : SeatingArrangement) 
  (h1 : arrangement.knights = 40)
  (h2 : arrangement.samurais = 10)
  (h3 : arrangement.knights_with_samurai_right = 7) :
  max_knights_between_knights arrangement = 32 := by
  sorry

#eval max_knights_between_knights ⟨40, 10, 7⟩

end NUMINAMATH_CALUDE_max_knights_between_knights_theorem_l733_73376


namespace NUMINAMATH_CALUDE_system_solution_l733_73363

/-- A solution to the system of equations is a triple (x, y, z) that satisfies all three equations. -/
def IsSolution (x y z : ℝ) : Prop :=
  x + y - z = 4 ∧
  x^2 + y^2 - z^2 = 12 ∧
  x^3 + y^3 - z^3 = 34

/-- The theorem states that the only solutions to the system of equations are (2, 3, 1) and (3, 2, 1). -/
theorem system_solution :
  ∀ x y z : ℝ, IsSolution x y z ↔ ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 2 ∧ z = 1)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l733_73363


namespace NUMINAMATH_CALUDE_multiplication_equation_solution_l733_73331

theorem multiplication_equation_solution : 
  ∃ x : ℕ, 18396 * x = 183868020 ∧ x = 9990 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equation_solution_l733_73331


namespace NUMINAMATH_CALUDE_scientific_notation_450_million_l733_73399

theorem scientific_notation_450_million :
  (450000000 : ℝ) = 4.5 * (10 : ℝ)^8 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_450_million_l733_73399


namespace NUMINAMATH_CALUDE_second_place_limit_l733_73353

/-- Represents an election with five candidates -/
structure Election where
  totalVoters : ℕ
  nonParticipationRate : ℚ
  invalidVotes : ℕ
  winnerVoteShare : ℚ
  winnerMargin : ℕ

/-- Conditions for a valid election -/
def validElection (e : Election) : Prop :=
  e.nonParticipationRate = 15/100 ∧
  e.invalidVotes = 250 ∧
  e.winnerVoteShare = 38/100 ∧
  e.winnerMargin = 300

/-- Calculate the percentage of valid votes for the second-place candidate -/
def secondPlacePercentage (e : Election) : ℚ :=
  let validVotes := e.totalVoters * (1 - e.nonParticipationRate) - e.invalidVotes
  let secondPlaceVotes := e.totalVoters * e.winnerVoteShare - e.winnerMargin
  secondPlaceVotes / validVotes * 100

/-- Theorem stating that as the number of voters approaches infinity, 
    the percentage of valid votes for the second-place candidate approaches 44.71% -/
theorem second_place_limit (ε : ℚ) (hε : ε > 0) : 
  ∃ N : ℕ, ∀ e : Election, validElection e → e.totalVoters ≥ N → 
    |secondPlacePercentage e - 4471/100| < ε :=
sorry

end NUMINAMATH_CALUDE_second_place_limit_l733_73353


namespace NUMINAMATH_CALUDE_expression_evaluation_l733_73300

theorem expression_evaluation :
  let x : ℝ := (1/2)^2023
  let y : ℝ := 2^2022
  (2*x + y)^2 - (2*x + y)*(2*x - y) - 2*y*(x + y) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l733_73300


namespace NUMINAMATH_CALUDE_dog_food_preferences_l733_73318

theorem dog_food_preferences (total : ℕ) (carrot : ℕ) (chicken : ℕ) (both : ℕ) 
  (h1 : total = 85)
  (h2 : carrot = 12)
  (h3 : chicken = 62)
  (h4 : both = 8) :
  total - (carrot + chicken - both) = 19 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_preferences_l733_73318


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l733_73346

theorem least_positive_integer_congruence :
  ∃! x : ℕ+, x.val + 3649 ≡ 304 [ZMOD 15] ∧
  ∀ y : ℕ+, y.val + 3649 ≡ 304 [ZMOD 15] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l733_73346


namespace NUMINAMATH_CALUDE_product_equals_fraction_l733_73342

def product_term (n : ℕ) : ℚ :=
  (2 * (n^4 - 1)) / (2 * (n^4 + 1))

def product_result : ℚ :=
  (product_term 2) * (product_term 3) * (product_term 4) * 
  (product_term 5) * (product_term 6) * (product_term 7)

theorem product_equals_fraction : product_result = 4400 / 135 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l733_73342


namespace NUMINAMATH_CALUDE_lucy_groceries_l733_73322

theorem lucy_groceries (cookies : ℕ) (cake : ℕ) : 
  cookies = 2 → cake = 12 → cookies + cake = 14 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_l733_73322


namespace NUMINAMATH_CALUDE_function_symmetry_l733_73375

-- Define the function f
variable (f : ℝ → ℝ)
-- Define the constant a
variable (a : ℝ)

-- State the theorem
theorem function_symmetry 
  (h : ∀ x : ℝ, f (a - x) = -f (a + x)) : 
  ∀ x : ℝ, f (2 * a - x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l733_73375


namespace NUMINAMATH_CALUDE_urn_probability_l733_73313

theorem urn_probability (N : ℝ) : N = 21 →
  (3 / 8) * (9 / (9 + N)) + (5 / 8) * (N / (9 + N)) = 0.55 := by
  sorry

#check urn_probability

end NUMINAMATH_CALUDE_urn_probability_l733_73313


namespace NUMINAMATH_CALUDE_product_squares_relation_l733_73373

theorem product_squares_relation (a b : ℝ) (h : a * b = 2 * (a^2 + b^2)) :
  2 * a * b - (a^2 + b^2) = a * b := by
  sorry

end NUMINAMATH_CALUDE_product_squares_relation_l733_73373


namespace NUMINAMATH_CALUDE_emily_calculation_l733_73367

theorem emily_calculation (n : ℕ) : n = 42 → (n + 1)^2 = n^2 + 85 → (n - 1)^2 = n^2 - 83 := by
  sorry

end NUMINAMATH_CALUDE_emily_calculation_l733_73367


namespace NUMINAMATH_CALUDE_trig_product_equals_one_l733_73321

theorem trig_product_equals_one : 
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
sorry

end NUMINAMATH_CALUDE_trig_product_equals_one_l733_73321


namespace NUMINAMATH_CALUDE_lens_break_probability_l733_73315

def prob_break_first : ℝ := 0.3
def prob_break_second_given_not_first : ℝ := 0.4
def prob_break_third_given_not_first_two : ℝ := 0.9

theorem lens_break_probability :
  let prob_break_second := (1 - prob_break_first) * prob_break_second_given_not_first
  let prob_break_third := (1 - prob_break_first) * (1 - prob_break_second_given_not_first) * prob_break_third_given_not_first_two
  prob_break_first + prob_break_second + prob_break_third = 0.958 := by
  sorry

end NUMINAMATH_CALUDE_lens_break_probability_l733_73315


namespace NUMINAMATH_CALUDE_square_last_digit_l733_73341

theorem square_last_digit (n : ℕ) 
  (h : (n^2 / 10) % 10 = 7) : 
  n^2 % 10 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_last_digit_l733_73341


namespace NUMINAMATH_CALUDE_sum_reciprocal_products_equals_three_eighths_l733_73349

theorem sum_reciprocal_products_equals_three_eighths :
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) +
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_products_equals_three_eighths_l733_73349


namespace NUMINAMATH_CALUDE_karen_tom_race_l733_73396

/-- Karen's race against Tom -/
theorem karen_tom_race (karen_speed : ℝ) (karen_delay : ℝ) (lead_distance : ℝ) (tom_distance : ℝ) :
  karen_speed = 60 →
  karen_delay = 4 / 60 →
  lead_distance = 4 →
  tom_distance = 24 →
  ∃ (tom_speed : ℝ), tom_speed = 45 ∧ 
    karen_speed * (tom_distance / karen_speed + lead_distance / karen_speed) = 
    tom_speed * (tom_distance / karen_speed + lead_distance / karen_speed + karen_delay) :=
by sorry

end NUMINAMATH_CALUDE_karen_tom_race_l733_73396


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l733_73323

def set_A : Set ℝ := {x | 2 * x + 1 > 0}
def set_B : Set ℝ := {x | |x - 1| < 2}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x | -1/2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l733_73323


namespace NUMINAMATH_CALUDE_solution_set_xf_x_minus_one_l733_73354

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then x^2 - 3*x - 4 else -((-x)^2 - 3*(-x) - 4)

theorem solution_set_xf_x_minus_one (x : ℝ) :
  x * f (x - 1) > 0 ↔ x ∈ Set.Iio (-3) ∪ Set.Ioo 0 1 ∪ Set.Ioi 5 :=
sorry

end NUMINAMATH_CALUDE_solution_set_xf_x_minus_one_l733_73354


namespace NUMINAMATH_CALUDE_french_students_count_l733_73337

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 69)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 15) :
  ∃ french : ℕ, french = total - german + both - neither :=
by
  sorry

end NUMINAMATH_CALUDE_french_students_count_l733_73337


namespace NUMINAMATH_CALUDE_two_children_gender_combinations_l733_73335

-- Define the Gender type
inductive Gender
  | Male
  | Female

-- Define a type for a pair of children
def ChildPair := (Gender × Gender)

-- Define the set of all possible gender combinations
def allGenderCombinations : Set ChildPair :=
  {(Gender.Male, Gender.Male), (Gender.Male, Gender.Female),
   (Gender.Female, Gender.Male), (Gender.Female, Gender.Female)}

-- Theorem statement
theorem two_children_gender_combinations :
  ∀ (family : Set ChildPair),
  (∀ pair : ChildPair, pair ∈ family ↔ pair ∈ allGenderCombinations) ↔
  family = allGenderCombinations :=
by sorry

end NUMINAMATH_CALUDE_two_children_gender_combinations_l733_73335


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l733_73328

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l733_73328


namespace NUMINAMATH_CALUDE_sara_balloons_l733_73377

theorem sara_balloons (tom_balloons : ℕ) (total_balloons : ℕ) 
  (h1 : tom_balloons = 9)
  (h2 : total_balloons = 17) :
  total_balloons - tom_balloons = 8 := by sorry

end NUMINAMATH_CALUDE_sara_balloons_l733_73377


namespace NUMINAMATH_CALUDE_min_value_theorem_l733_73332

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 3 * x + 2 * y = 1 → 2 / x + 3 / y ≥ 2 / a + 3 / b) →
  2 / a + 3 / b = 24 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l733_73332


namespace NUMINAMATH_CALUDE_worksheet_problems_l733_73392

theorem worksheet_problems (total_worksheets : ℕ) (graded_worksheets : ℕ) (problems_left : ℕ) :
  total_worksheets = 17 →
  graded_worksheets = 8 →
  problems_left = 63 →
  (total_worksheets - graded_worksheets) * (problems_left / (total_worksheets - graded_worksheets)) = 7 :=
by sorry

end NUMINAMATH_CALUDE_worksheet_problems_l733_73392


namespace NUMINAMATH_CALUDE_new_person_age_l733_73378

theorem new_person_age (initial_group_size : ℕ) (age_decrease : ℕ) (replaced_person_age : ℕ) :
  initial_group_size = 10 →
  age_decrease = 3 →
  replaced_person_age = 42 →
  ∃ (new_person_age : ℕ),
    new_person_age = initial_group_size * age_decrease + replaced_person_age - initial_group_size * age_decrease :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_age_l733_73378


namespace NUMINAMATH_CALUDE_folded_rectangle_perimeter_ratio_l733_73368

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem folded_rectangle_perimeter_ratio :
  let original := Rectangle.mk 8 4
  let folded := Rectangle.mk 4 2
  (perimeter folded) / (perimeter original) = 1/2 := by sorry

end NUMINAMATH_CALUDE_folded_rectangle_perimeter_ratio_l733_73368


namespace NUMINAMATH_CALUDE_closest_result_is_180_l733_73339

theorem closest_result_is_180 (options : List ℝ := [160, 180, 190, 200, 240]) : 
  let result := (0.000345 * 7650000) / 15
  options.argmin (λ x => |x - result|) = some 180 := by
  sorry

end NUMINAMATH_CALUDE_closest_result_is_180_l733_73339


namespace NUMINAMATH_CALUDE_ac_eq_b_squared_necessary_not_sufficient_l733_73390

/-- Definition of a geometric progression for three real numbers -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The main theorem stating that ac = b^2 is necessary but not sufficient for a, b, c to be in geometric progression -/
theorem ac_eq_b_squared_necessary_not_sufficient :
  (∀ a b c : ℝ, isGeometricProgression a b c → a * c = b^2) ∧
  (∃ a b c : ℝ, a * c = b^2 ∧ ¬isGeometricProgression a b c) := by
  sorry

end NUMINAMATH_CALUDE_ac_eq_b_squared_necessary_not_sufficient_l733_73390


namespace NUMINAMATH_CALUDE_set_operations_and_intersection_condition_l733_73310

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}

theorem set_operations_and_intersection_condition (a : ℝ) :
  (A ∪ B = {x | 1 < x ∧ x ≤ 8}) ∧
  ((Set.univ \ A) ∩ B = {x | 1 < x ∧ x < 2}) ∧
  (A ∩ C a ≠ ∅ → a < 8) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_intersection_condition_l733_73310


namespace NUMINAMATH_CALUDE_chord_equation_l733_73350

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Represents a line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Main theorem -/
theorem chord_equation (e : Ellipse) (p : Point) (l : Line) : 
  e.a^2 = 8 ∧ e.b^2 = 4 ∧ p.x = 2 ∧ p.y = -1 ∧ 
  (∃ p1 p2 : Point, 
    pointOnEllipse p1 e ∧ 
    pointOnEllipse p2 e ∧
    p.x = (p1.x + p2.x) / 2 ∧ 
    p.y = (p1.y + p2.y) / 2 ∧
    pointOnLine p1 l ∧ 
    pointOnLine p2 l) →
  l.slope = 1 ∧ l.intercept = -3 := by
  sorry

end NUMINAMATH_CALUDE_chord_equation_l733_73350


namespace NUMINAMATH_CALUDE_geometric_progression_perfect_square_sum_l733_73351

/-- A geometric progression starting with 1 -/
def GeometricProgression (r : ℕ) (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => r^i)

/-- The sum of a list of natural numbers -/
def ListSum (l : List ℕ) : ℕ :=
  l.foldl (·+·) 0

/-- A number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem geometric_progression_perfect_square_sum :
  ∃ r₁ r₂ n₁ n₂ : ℕ,
    r₁ ≠ r₂ ∧
    n₁ ≥ 3 ∧
    n₂ ≥ 3 ∧
    IsPerfectSquare (ListSum (GeometricProgression r₁ n₁)) ∧
    IsPerfectSquare (ListSum (GeometricProgression r₂ n₂)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_perfect_square_sum_l733_73351


namespace NUMINAMATH_CALUDE_expression_one_value_expression_two_value_l733_73345

-- Given tan θ = 2
variable (θ : Real) (h : Real.tan θ = 2)

-- Theorem for the first expression
theorem expression_one_value : 
  (4 * Real.sin θ - 2 * Real.cos θ) / (3 * Real.sin θ + 5 * Real.cos θ) = 6/11 := by sorry

-- Theorem for the second expression
theorem expression_two_value :
  1 - 4 * Real.sin θ * Real.cos θ + 2 * (Real.cos θ)^2 = -1/5 := by sorry

end NUMINAMATH_CALUDE_expression_one_value_expression_two_value_l733_73345


namespace NUMINAMATH_CALUDE_base7_addition_l733_73394

/-- Addition of numbers in base 7 -/
def base7_add (a b c : ℕ) : ℕ :=
  (a + b + c) % 7^3

/-- Conversion from base 7 to decimal -/
def base7_to_decimal (n : ℕ) : ℕ :=
  (n / 7^2) * 7^2 + ((n / 7) % 7) * 7 + (n % 7)

theorem base7_addition :
  base7_add (base7_to_decimal 26) (base7_to_decimal 64) (base7_to_decimal 135) = base7_to_decimal 261 :=
sorry

end NUMINAMATH_CALUDE_base7_addition_l733_73394


namespace NUMINAMATH_CALUDE_wand_original_price_l733_73370

theorem wand_original_price (price_paid : ℝ) (original_price : ℝ) 
  (h1 : price_paid = 8)
  (h2 : price_paid = original_price / 8) : 
  original_price = 64 := by
  sorry

end NUMINAMATH_CALUDE_wand_original_price_l733_73370


namespace NUMINAMATH_CALUDE_prism_with_27_edges_has_11_faces_l733_73393

/-- A prism is a polyhedron with two congruent parallel faces (bases) and faces that connect the bases (lateral faces). -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges -/
def num_faces (p : Prism) : ℕ :=
  let base_edges := p.edges / 3
  2 + base_edges

theorem prism_with_27_edges_has_11_faces (p : Prism) (h : p.edges = 27) :
  num_faces p = 11 := by
  sorry

end NUMINAMATH_CALUDE_prism_with_27_edges_has_11_faces_l733_73393


namespace NUMINAMATH_CALUDE_eva_is_speed_skater_l733_73398

-- Define the people and sports
inductive Person : Type
| Ben : Person
| Filip : Person
| Eva : Person
| Andrea : Person

inductive Sport : Type
| SpeedSkating : Sport
| Skiing : Sport
| Hockey : Sport
| Snowboarding : Sport

-- Define the positions at the table
inductive Position : Type
| Top : Position
| Right : Position
| Bottom : Position
| Left : Position

-- Define the seating arrangement
def SeatingArrangement : Type := Person → Position

-- Define the sport assignment
def SportAssignment : Type := Person → Sport

-- Define the conditions
def Conditions (seating : SeatingArrangement) (sports : SportAssignment) : Prop :=
  ∃ (skier hockey_player : Person),
    -- The skier sat at Andrea's left hand
    seating Person.Andrea = Position.Top ∧ seating skier = Position.Left
    -- The speed skater sat opposite Ben
    ∧ seating Person.Ben = Position.Left
    ∧ sports Person.Ben ≠ Sport.SpeedSkating
    -- Eva and Filip sat next to each other
    ∧ (seating Person.Eva = Position.Right ∧ seating Person.Filip = Position.Bottom
    ∨ seating Person.Eva = Position.Bottom ∧ seating Person.Filip = Position.Right)
    -- A woman sat at the hockey player's left hand
    ∧ ((seating hockey_player = Position.Right ∧ seating Person.Andrea = Position.Top)
    ∨ (seating hockey_player = Position.Bottom ∧ seating Person.Eva = Position.Right))

-- The theorem to prove
theorem eva_is_speed_skater (seating : SeatingArrangement) (sports : SportAssignment) :
  Conditions seating sports → sports Person.Eva = Sport.SpeedSkating :=
sorry

end NUMINAMATH_CALUDE_eva_is_speed_skater_l733_73398


namespace NUMINAMATH_CALUDE_money_left_after_debts_l733_73369

def lottery_winnings : ℕ := 100
def payment_to_colin : ℕ := 20

def payment_to_helen (colin_payment : ℕ) : ℕ := 2 * colin_payment

def payment_to_benedict (helen_payment : ℕ) : ℕ := helen_payment / 2

def total_payments (colin : ℕ) (helen : ℕ) (benedict : ℕ) : ℕ := colin + helen + benedict

theorem money_left_after_debts :
  lottery_winnings - total_payments payment_to_colin (payment_to_helen payment_to_colin) (payment_to_benedict (payment_to_helen payment_to_colin)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_debts_l733_73369


namespace NUMINAMATH_CALUDE_x_greater_than_x_squared_only_half_satisfies_l733_73320

theorem x_greater_than_x_squared (x : ℝ) : x > x^2 ↔ x ∈ (Set.Ioo 0 1) := by sorry

theorem only_half_satisfies :
  ∀ x ∈ ({-2, -(1/2), 0, 1/2, 2} : Set ℝ), x > x^2 ↔ x = 1/2 := by sorry

end NUMINAMATH_CALUDE_x_greater_than_x_squared_only_half_satisfies_l733_73320


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l733_73382

/-- For a parabola with equation y^2 = 4x, the distance between its focus and directrix is 2. -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y^2 = 4*x → ∃ (f d : ℝ × ℝ),
    (f.1 = 1 ∧ f.2 = 0) ∧  -- focus
    (d.1 = -1 ∧ ∀ t, d.2 = t) ∧  -- directrix
    (f.1 - d.1 = 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l733_73382


namespace NUMINAMATH_CALUDE_company_growth_rate_inequality_l733_73317

theorem company_growth_rate_inequality (p q x : ℝ) : 
  (1 + p) * (1 + q) = (1 + x)^2 → x ≤ (p + q) / 2 := by
  sorry

end NUMINAMATH_CALUDE_company_growth_rate_inequality_l733_73317


namespace NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l733_73355

theorem sin_minus_cos_sqrt_two (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi →
  (Real.sin x - Real.cos x = Real.sqrt 2 ↔ x = 3 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l733_73355


namespace NUMINAMATH_CALUDE_geometric_sequence_tan_result_l733_73384

/-- Given a geometric sequence {a_n} with the specified conditions, 
    prove that tan((a_4 * a_6 / 3) * π) = -√3 -/
theorem geometric_sequence_tan_result (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 2 * a 3 * a 4 = -a 7^2 →                        -- given condition
  a 7^2 = 64 →                                      -- given condition
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_tan_result_l733_73384


namespace NUMINAMATH_CALUDE_joan_remaining_books_l733_73385

/-- Given an initial number of books and a number of books sold, 
    calculate the remaining number of books. -/
def remaining_books (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

/-- Theorem: Given 33 initial books and 26 books sold, 
    the remaining number of books is 7. -/
theorem joan_remaining_books :
  remaining_books 33 26 = 7 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_books_l733_73385


namespace NUMINAMATH_CALUDE_train_seat_count_l733_73333

/-- Calculates the total number of seats on trains at a station -/
def total_seats (num_trains : ℕ) (cars_per_train : ℕ) (seats_per_car : ℕ) : ℕ :=
  num_trains * cars_per_train * seats_per_car

/-- Theorem: The total number of seats on 3 trains, each with 12 cars and 24 seats per car, is 864 -/
theorem train_seat_count : total_seats 3 12 24 = 864 := by
  sorry

end NUMINAMATH_CALUDE_train_seat_count_l733_73333


namespace NUMINAMATH_CALUDE_circle_circumference_bounds_l733_73397

/-- The circumference of a circle with diameter 1 is between 3 and 4 -/
theorem circle_circumference_bounds :
  ∀ C : ℝ, C = π * 1 → 3 < C ∧ C < 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_bounds_l733_73397


namespace NUMINAMATH_CALUDE_conjugate_complex_equation_l733_73366

/-- Two complex numbers are conjugates if their real parts are equal and their imaginary parts are opposites -/
def are_conjugates (a b : ℂ) : Prop := a.re = b.re ∧ a.im = -b.im

/-- The main theorem -/
theorem conjugate_complex_equation (a b : ℂ) :
  are_conjugates a b → (a + b)^2 - 3 * a * b * I = 4 - 6 * I →
  ((a = 1 + I ∧ b = 1 - I) ∨
   (a = -1 - I ∧ b = -1 + I) ∨
   (a = 1 - I ∧ b = 1 + I) ∨
   (a = -1 + I ∧ b = -1 - I)) :=
by sorry

end NUMINAMATH_CALUDE_conjugate_complex_equation_l733_73366


namespace NUMINAMATH_CALUDE_distances_to_other_vertices_l733_73312

/-- A circle with radius 5 and an inscribed square -/
structure CircleSquare where
  center : ℝ × ℝ
  radius : ℝ
  square_vertices : Fin 4 → ℝ × ℝ

/-- A point on the circle -/
def PointOnCircle (cs : CircleSquare) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the distances to other vertices -/
theorem distances_to_other_vertices (cs : CircleSquare) 
  (h_radius : cs.radius = 5)
  (h_inscribed : ∀ v, distance cs.center (cs.square_vertices v) = cs.radius)
  (h_on_circle : distance cs.center (PointOnCircle cs) = cs.radius)
  (h_distance_to_one : ∃ v, distance (PointOnCircle cs) (cs.square_vertices v) = 6) :
  ∃ (v1 v2 v3 : Fin 4), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
    distance (PointOnCircle cs) (cs.square_vertices v1) = Real.sqrt 2 ∧
    distance (PointOnCircle cs) (cs.square_vertices v2) = 8 ∧
    distance (PointOnCircle cs) (cs.square_vertices v3) = 7 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_CALUDE_distances_to_other_vertices_l733_73312


namespace NUMINAMATH_CALUDE_least_common_multiple_4_5_6_9_l733_73305

theorem least_common_multiple_4_5_6_9 : ∃ (n : ℕ), n > 0 ∧ 
  4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n ∧ 9 ∣ n ∧ 
  ∀ (m : ℕ), m > 0 → 4 ∣ m → 5 ∣ m → 6 ∣ m → 9 ∣ m → n ≤ m :=
by
  use 180
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_4_5_6_9_l733_73305


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l733_73374

theorem max_sum_of_factors (A B C : ℕ) : 
  A > 0 → B > 0 → C > 0 →
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2310 →
  A + B + C ≤ 42 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l733_73374


namespace NUMINAMATH_CALUDE_perpendicular_vectors_result_l733_73372

def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := λ m ↦ (4, m)

theorem perpendicular_vectors_result (m : ℝ) 
  (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) : 
  (5 : ℝ) • a - (3 : ℝ) • (b m) = (-7, -16) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_result_l733_73372


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l733_73388

/-- Given that g(x) = ax^2 + bx + c divides f(x) = x^3 + px^2 + qx + r, 
    where a ≠ 0, b ≠ 0, c ≠ 0, prove that (ap - b) / a = (aq - c) / b = ar / c -/
theorem polynomial_division_theorem 
  (a b c p q r : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_divides : ∃ k, x^3 + p*x^2 + q*x + r = (a*x^2 + b*x + c) * k) : 
  (a*p - b) / a = (a*q - c) / b ∧ (a*q - c) / b = a*r / c := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l733_73388


namespace NUMINAMATH_CALUDE_sqrt_18_div_sqrt_2_equals_3_l733_73371

theorem sqrt_18_div_sqrt_2_equals_3 : Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_div_sqrt_2_equals_3_l733_73371


namespace NUMINAMATH_CALUDE_find_common_ratio_l733_73334

/-- Given a table of n^2 (n ≥ 4) positive numbers arranged in n rows and n columns,
    where each row forms an arithmetic sequence and each column forms a geometric sequence
    with the same common ratio q, prove that q = 1/2 given the specified conditions. -/
theorem find_common_ratio (n : ℕ) (a : ℕ → ℕ → ℝ) (q : ℝ) 
    (h_n : n ≥ 4)
    (h_positive : ∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → a i j > 0)
    (h_arithmetic_row : ∀ i k, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ k ∧ k < n → 
      a i (k + 1) - a i k = a i (k + 2) - a i (k + 1))
    (h_geometric_col : ∀ i j, 1 ≤ i ∧ i < n ∧ 1 ≤ j ∧ j ≤ n → 
      a (i + 1) j = q * a i j)
    (h_a26 : a 2 6 = 1)
    (h_a42 : a 4 2 = 1/8)
    (h_a44 : a 4 4 = 3/16) :
  q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_find_common_ratio_l733_73334


namespace NUMINAMATH_CALUDE_racket_purchase_cost_l733_73308

/-- Calculates the total cost of two rackets with given discounts and sales tax -/
def totalCost (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (salesTax : ℝ) : ℝ :=
  let price1 := originalPrice * (1 - discount1)
  let price2 := originalPrice * (1 - discount2)
  let subtotal := price1 + price2
  subtotal * (1 + salesTax)

/-- Theorem stating the total cost of two rackets under specific conditions -/
theorem racket_purchase_cost :
  totalCost 60 0.2 0.5 0.05 = 81.90 := by
  sorry

end NUMINAMATH_CALUDE_racket_purchase_cost_l733_73308


namespace NUMINAMATH_CALUDE_gcf_factorial_seven_eight_l733_73314

theorem gcf_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_factorial_seven_eight_l733_73314


namespace NUMINAMATH_CALUDE_prisoner_selection_l733_73324

/-- Given 25 prisoners, prove the number of ways to choose 3 in order and without order. -/
theorem prisoner_selection (n : ℕ) (h : n = 25) : 
  (n * (n - 1) * (n - 2) = 13800) ∧ (Nat.choose n 3 = 2300) := by
  sorry

end NUMINAMATH_CALUDE_prisoner_selection_l733_73324


namespace NUMINAMATH_CALUDE_max_distance_to_point_l733_73380

theorem max_distance_to_point (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (w : ℂ), Complex.abs w = 1 ∧ Complex.abs (w - (1 + Complex.I)) = Real.sqrt 2 + 1 :=
sorry

end NUMINAMATH_CALUDE_max_distance_to_point_l733_73380


namespace NUMINAMATH_CALUDE_solution_product_l733_73360

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x - 3) * (3 * x + 7) = x^2 - 12 * x + 27

-- State the theorem
theorem solution_product (a b : ℝ) : 
  a ≠ b ∧ equation a ∧ equation b → (a + 2) * (b + 2) = -30 := by
  sorry

end NUMINAMATH_CALUDE_solution_product_l733_73360


namespace NUMINAMATH_CALUDE_star_calculation_l733_73358

def star (a b : ℝ) : ℝ := a * b + a + b

theorem star_calculation : star 1 2 + star 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l733_73358


namespace NUMINAMATH_CALUDE_oil_price_reduction_60_percent_l733_73325

/-- The percentage reduction in oil price -/
def oil_price_reduction (original_price reduced_price : ℚ) : ℚ :=
  (original_price - reduced_price) / original_price * 100

/-- The amount of oil that can be bought with a fixed amount of money -/
def oil_amount (price : ℚ) (money : ℚ) : ℚ := money / price

theorem oil_price_reduction_60_percent 
  (reduced_price : ℚ) 
  (additional_amount : ℚ) 
  (fixed_money : ℚ) :
  reduced_price = 30 →
  additional_amount = 10 →
  fixed_money = 1500 →
  oil_amount reduced_price fixed_money = oil_amount reduced_price (fixed_money / 2) + additional_amount →
  oil_price_reduction ((fixed_money / 2) / additional_amount) reduced_price = 60 := by
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_60_percent_l733_73325


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l733_73306

theorem complex_magnitude_problem (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = (1 - i) / (2 + i) →
  Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l733_73306


namespace NUMINAMATH_CALUDE_unique_point_distance_to_line_l733_73327

/-- Given a circle C: (x - √a)² + (y - a)² = 1 where a ≥ 0, if there exists only one point P on C
    such that the distance from P to the line l: y = 2x - 6 equals √5 - 1, then a = 1. -/
theorem unique_point_distance_to_line (a : ℝ) (h1 : a ≥ 0) :
  (∃! P : ℝ × ℝ, (P.1 - Real.sqrt a)^2 + (P.2 - a)^2 = 1 ∧
    |2 * P.1 - P.2 - 6| / Real.sqrt 5 = Real.sqrt 5 - 1) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_point_distance_to_line_l733_73327


namespace NUMINAMATH_CALUDE_trajectory_and_equilateral_triangle_l733_73357

-- Define the points
def H : ℝ × ℝ := (-3, 0)
def T : ℝ × ℝ := (-1, 0)

-- Define the trajectory C
def C : Set (ℝ × ℝ) := {(x, y) | y^2 = 4*x ∧ x > 0}

-- Define the conditions
def on_y_axis (P : ℝ × ℝ) : Prop := P.1 = 0
def on_positive_x_axis (Q : ℝ × ℝ) : Prop := Q.2 = 0 ∧ Q.1 > 0
def on_line (P Q M : ℝ × ℝ) : Prop := ∃ t : ℝ, M = (1 - t) • P + t • Q

def orthogonal (HP PM : ℝ × ℝ) : Prop := HP.1 * PM.1 + HP.2 * PM.2 = 0
def vector_ratio (PM MQ : ℝ × ℝ) : Prop := PM = (-3/2) • MQ

-- Main theorem
theorem trajectory_and_equilateral_triangle 
  (P Q M : ℝ × ℝ) 
  (hP : on_y_axis P) 
  (hQ : on_positive_x_axis Q) 
  (hM : on_line P Q M) 
  (hOrth : orthogonal (H.1 - P.1, H.2 - P.2) (M.1 - P.1, M.2 - P.2))
  (hRatio : vector_ratio (M.1 - P.1, M.2 - P.2) (Q.1 - M.1, Q.2 - M.2)) :
  (M ∈ C) ∧ 
  (∀ (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)) (E : ℝ × ℝ),
    (A ∈ C ∧ B ∈ C ∧ T ∈ l ∧ A ∈ l ∧ B ∈ l ∧ E.2 = 0) →
    (∃ (x₀ : ℝ), E.1 = x₀ ∧ 
      (norm (A - E) = norm (B - E) ∧ norm (A - E) = norm (A - B)) →
      x₀ = 11/3)) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_equilateral_triangle_l733_73357


namespace NUMINAMATH_CALUDE_e_general_term_l733_73379

/-- A sequence is a DQ sequence if it can be expressed as the sum of an arithmetic sequence
and a geometric sequence, both with positive integer terms. -/
def is_dq_sequence (e : ℕ → ℕ) : Prop :=
  ∃ (a b : ℕ → ℕ) (d q : ℕ),
    (∀ n, a n = a 1 + (n - 1) * d) ∧
    (∀ n, b n = b 1 * q^(n - 1)) ∧
    (∀ n, e n = a n + b n) ∧
    (∀ n, a n > 0 ∧ b n > 0)

/-- The sequence e_n satisfies the given conditions -/
def e_satisfies_conditions (e : ℕ → ℕ) : Prop :=
  is_dq_sequence e ∧
  e 1 = 3 ∧ e 2 = 6 ∧ e 3 = 11 ∧ e 4 = 20 ∧ e 5 = 37

theorem e_general_term (e : ℕ → ℕ) (h : e_satisfies_conditions e) :
  ∀ n : ℕ, e n = n + 2^n :=
by sorry

end NUMINAMATH_CALUDE_e_general_term_l733_73379


namespace NUMINAMATH_CALUDE_cosine_amplitude_l733_73383

theorem cosine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.cos (b * x) ≤ 3) ∧ (∃ x, a * Real.cos (b * x) = 3) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l733_73383


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l733_73387

theorem imaginary_part_of_z (z : ℂ) : z = -2 * Complex.I * (-1 + Real.sqrt 3 * Complex.I) → z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l733_73387


namespace NUMINAMATH_CALUDE_multiple_of_q_in_equation_l733_73391

theorem multiple_of_q_in_equation (p q m : ℚ) 
  (h1 : p / q = 3 / 4)
  (h2 : 3 * p + m * q = 25 / 4) :
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_q_in_equation_l733_73391


namespace NUMINAMATH_CALUDE_planted_field_fraction_l733_73336

theorem planted_field_fraction (a b h x : ℝ) (ha : a = 5) (hb : b = 12) (hh : h = 3) :
  let c := (a^2 + b^2).sqrt
  let s := x^2
  let triangle_area := a * b / 2
  h = (2 * triangle_area) / c - (b * x) / c →
  (triangle_area - s) / triangle_area = 431 / 480 :=
by sorry

end NUMINAMATH_CALUDE_planted_field_fraction_l733_73336


namespace NUMINAMATH_CALUDE_twenty_seven_power_minus_log_eight_two_equals_zero_l733_73362

theorem twenty_seven_power_minus_log_eight_two_equals_zero :
  Real.rpow 27 (-1/3) - Real.log 2 / Real.log 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_twenty_seven_power_minus_log_eight_two_equals_zero_l733_73362


namespace NUMINAMATH_CALUDE_only_piston_and_bottles_are_translations_l733_73365

/-- Represents a type of motion --/
inductive Motion
| Translation
| Rotation
| Other

/-- Represents the different phenomena described in the problem --/
inductive Phenomenon
| ChildSwinging
| PistonMovement
| PendulumSwinging
| BottlesOnConveyorBelt

/-- Determines the type of motion for a given phenomenon --/
def motionType (p : Phenomenon) : Motion :=
  match p with
  | Phenomenon.ChildSwinging => Motion.Rotation
  | Phenomenon.PistonMovement => Motion.Translation
  | Phenomenon.PendulumSwinging => Motion.Rotation
  | Phenomenon.BottlesOnConveyorBelt => Motion.Translation

/-- Theorem stating that only the piston movement and bottles on conveyor belt are translations --/
theorem only_piston_and_bottles_are_translations :
  (∀ p : Phenomenon, motionType p = Motion.Translation ↔ 
    (p = Phenomenon.PistonMovement ∨ p = Phenomenon.BottlesOnConveyorBelt)) :=
by sorry

end NUMINAMATH_CALUDE_only_piston_and_bottles_are_translations_l733_73365


namespace NUMINAMATH_CALUDE_fraction_problem_l733_73326

theorem fraction_problem (x : ℚ) : x * 8 + 2 = 8 → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l733_73326


namespace NUMINAMATH_CALUDE_total_spent_equals_sum_of_games_l733_73359

/-- The total amount Tom spent on video games -/
def total_spent : ℝ := 35.52

/-- The cost of the football game -/
def football_cost : ℝ := 14.02

/-- The cost of the strategy game -/
def strategy_cost : ℝ := 9.46

/-- The cost of the Batman game -/
def batman_cost : ℝ := 12.04

/-- Theorem: The total amount Tom spent on video games is equal to the sum of the costs of the football game, strategy game, and Batman game -/
theorem total_spent_equals_sum_of_games : 
  total_spent = football_cost + strategy_cost + batman_cost := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_sum_of_games_l733_73359


namespace NUMINAMATH_CALUDE_probability_Sa_before_Sb_l733_73311

/-- Represents a three-letter string -/
structure ThreeLetterString :=
  (letters : Fin 3 → Char)

/-- The probability of a letter being received correctly -/
def correct_probability : ℚ := 2/3

/-- The probability of a letter being received incorrectly -/
def incorrect_probability : ℚ := 1/3

/-- The transmitted string aaa -/
def aaa : ThreeLetterString :=
  { letters := λ _ => 'a' }

/-- The transmitted string bbb -/
def bbb : ThreeLetterString :=
  { letters := λ _ => 'b' }

/-- The received string when aaa is transmitted -/
def Sa : ThreeLetterString :=
  sorry

/-- The received string when bbb is transmitted -/
def Sb : ThreeLetterString :=
  sorry

/-- The probability that Sa comes before Sb in alphabetical order -/
def p : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem probability_Sa_before_Sb : p = 532/729 :=
  sorry

end NUMINAMATH_CALUDE_probability_Sa_before_Sb_l733_73311


namespace NUMINAMATH_CALUDE_initial_boys_count_l733_73330

/-- Given an initial group of boys with an average weight of 102 kg,
    adding a new person weighing 40 kg reduces the average by 2 kg.
    This function calculates the initial number of boys. -/
def initial_number_of_boys : ℕ :=
  let initial_avg : ℚ := 102
  let new_person_weight : ℚ := 40
  let avg_decrease : ℚ := 2
  let n : ℕ := 30  -- The number we want to prove
  n

/-- Theorem stating that the initial number of boys is 30 -/
theorem initial_boys_count :
  let n := initial_number_of_boys
  let initial_avg : ℚ := 102
  let new_person_weight : ℚ := 40
  let avg_decrease : ℚ := 2
  (n : ℚ) * initial_avg + new_person_weight = (n + 1) * (initial_avg - avg_decrease) :=
by sorry

end NUMINAMATH_CALUDE_initial_boys_count_l733_73330


namespace NUMINAMATH_CALUDE_factors_multiple_of_300_eq_1320_l733_73309

/-- The number of natural-number factors of 2^12 * 3^15 * 5^9 that are multiples of 300 -/
def factors_multiple_of_300 : ℕ :=
  (12 - 2 + 1) * (15 - 1 + 1) * (9 - 2 + 1)

/-- Theorem stating that the number of natural-number factors of 2^12 * 3^15 * 5^9
    that are multiples of 300 is equal to 1320 -/
theorem factors_multiple_of_300_eq_1320 :
  factors_multiple_of_300 = 1320 := by
  sorry

end NUMINAMATH_CALUDE_factors_multiple_of_300_eq_1320_l733_73309


namespace NUMINAMATH_CALUDE_least_b_proof_l733_73356

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The least possible value of b given the conditions -/
def least_b : ℕ := 12

theorem least_b_proof (a b : ℕ+) 
  (ha : num_factors a = 4) 
  (hb : num_factors b = a.val) 
  (hdiv : a ∣ b) : 
  ∀ c : ℕ+, 
    (num_factors c = a.val) → 
    (a ∣ c) → 
    least_b ≤ c.val :=
by sorry

end NUMINAMATH_CALUDE_least_b_proof_l733_73356


namespace NUMINAMATH_CALUDE_max_portfolios_is_six_l733_73347

/-- Represents the number of items Stacy purchases -/
structure Purchase where
  pens : ℕ
  pads : ℕ
  portfolios : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  2 * p.pens + 5 * p.pads + 15 * p.portfolios

/-- Checks if a purchase is valid according to the problem constraints -/
def isValidPurchase (p : Purchase) : Prop :=
  p.pens ≥ 1 ∧ p.pads ≥ 1 ∧ p.portfolios ≥ 1 ∧ totalCost p = 100

/-- The maximum number of portfolios that can be purchased -/
def maxPortfolios : ℕ := 6

/-- Theorem stating that 6 is the maximum number of portfolios that can be purchased -/
theorem max_portfolios_is_six :
  (∀ p : Purchase, isValidPurchase p → p.portfolios ≤ maxPortfolios) ∧
  (∃ p : Purchase, isValidPurchase p ∧ p.portfolios = maxPortfolios) := by
  sorry


end NUMINAMATH_CALUDE_max_portfolios_is_six_l733_73347


namespace NUMINAMATH_CALUDE_fair_rides_l733_73302

theorem fair_rides (total_tickets : ℕ) (spent_tickets : ℕ) (ride_cost : ℕ) : 
  total_tickets = 79 → spent_tickets = 23 → ride_cost = 7 → 
  (total_tickets - spent_tickets) / ride_cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_fair_rides_l733_73302


namespace NUMINAMATH_CALUDE_quadratic_solution_l733_73381

theorem quadratic_solution : 
  ∀ x : ℝ, x * (x - 7) = 0 ↔ x = 0 ∨ x = 7 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l733_73381


namespace NUMINAMATH_CALUDE_ozone_experiment_properties_l733_73319

/-- Represents the experimental setup and data for the ozone effect study on mice. -/
structure OzoneExperiment where
  total_mice : Nat
  control_group : Nat
  experimental_group : Nat
  weight_increases : List ℝ
  k_squared_threshold : ℝ

/-- Represents the distribution of X, where X is the number of specified two mice
    assigned to the control group. -/
def distribution_X (exp : OzoneExperiment) : Fin 3 → ℝ := sorry

/-- Calculates the expected value of X. -/
def expected_value_X (exp : OzoneExperiment) : ℝ := sorry

/-- Calculates the median of all mice weight increases. -/
def median_weight_increase (exp : OzoneExperiment) : ℝ := sorry

/-- Represents the contingency table based on the median. -/
structure ContingencyTable where
  control_less : Nat
  control_greater_equal : Nat
  experimental_less : Nat
  experimental_greater_equal : Nat

/-- Constructs the contingency table based on the median. -/
def create_contingency_table (exp : OzoneExperiment) : ContingencyTable := sorry

/-- Calculates the K^2 value based on the contingency table. -/
def calculate_k_squared (table : ContingencyTable) : ℝ := sorry

/-- The main theorem stating the properties of the ozone experiment. -/
theorem ozone_experiment_properties (exp : OzoneExperiment) :
  exp.total_mice = 40 ∧
  exp.control_group = 20 ∧
  exp.experimental_group = 20 ∧
  distribution_X exp 0 = 19/78 ∧
  distribution_X exp 1 = 20/39 ∧
  distribution_X exp 2 = 19/78 ∧
  expected_value_X exp = 1 ∧
  median_weight_increase exp = 23.4 ∧
  let table := create_contingency_table exp
  table.control_less = 6 ∧
  table.control_greater_equal = 14 ∧
  table.experimental_less = 14 ∧
  table.experimental_greater_equal = 6 ∧
  calculate_k_squared table = 6.4 ∧
  calculate_k_squared table > exp.k_squared_threshold := by
  sorry

end NUMINAMATH_CALUDE_ozone_experiment_properties_l733_73319


namespace NUMINAMATH_CALUDE_final_net_worth_l733_73348

/-- Represents a person's assets --/
structure Assets where
  cash : Int
  has_house : Bool
  has_vehicle : Bool

/-- Represents a transaction between two people --/
inductive Transaction
  | sell_house (price : Int)
  | sell_vehicle (price : Int)

/-- Performs a transaction and updates the assets of both parties --/
def perform_transaction (a b : Assets) (t : Transaction) : Assets × Assets :=
  match t with
  | Transaction.sell_house price => 
    ({ cash := a.cash + price, has_house := false, has_vehicle := a.has_vehicle },
     { cash := b.cash - price, has_house := true, has_vehicle := b.has_vehicle })
  | Transaction.sell_vehicle price => 
    ({ cash := a.cash - price, has_house := a.has_house, has_vehicle := true },
     { cash := b.cash + price, has_house := b.has_house, has_vehicle := false })

/-- Calculates the net worth of a person given their assets and the values of the house and vehicle --/
def net_worth (a : Assets) (house_value vehicle_value : Int) : Int :=
  a.cash + (if a.has_house then house_value else 0) + (if a.has_vehicle then vehicle_value else 0)

/-- The main theorem stating the final net worth of Mr. A and Mr. B --/
theorem final_net_worth (initial_a initial_b : Assets) 
  (house_value vehicle_value : Int) (transactions : List Transaction) : 
  initial_a.cash = 20000 → initial_a.has_house = true → initial_a.has_vehicle = false →
  initial_b.cash = 22000 → initial_b.has_house = false → initial_b.has_vehicle = true →
  house_value = 20000 → vehicle_value = 10000 →
  transactions = [
    Transaction.sell_house 25000,
    Transaction.sell_vehicle 12000,
    Transaction.sell_house 18000,
    Transaction.sell_vehicle 9000
  ] →
  let (final_a, final_b) := transactions.foldl 
    (fun (acc : Assets × Assets) (t : Transaction) => perform_transaction acc.1 acc.2 t) 
    (initial_a, initial_b)
  net_worth final_a house_value vehicle_value = 40000 ∧ 
  net_worth final_b house_value vehicle_value = 8000 := by
  sorry


end NUMINAMATH_CALUDE_final_net_worth_l733_73348


namespace NUMINAMATH_CALUDE_g_of_60_l733_73364

/-- Given a function g satisfying the specified properties, prove that g(60) = 11.25 -/
theorem g_of_60 (g : ℝ → ℝ) 
    (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → g (x * y) = g x / y)
    (h2 : g 45 = 15) : 
  g 60 = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_g_of_60_l733_73364


namespace NUMINAMATH_CALUDE_cone_volume_l733_73307

/-- The volume of a cone with lateral surface forming a sector of radius √31 and arc length 4π -/
theorem cone_volume (r h : ℝ) : 
  r > 0 → h > 0 → 
  (h^2 + r^2 : ℝ) = 31 → 
  2 * π * r = 4 * π → 
  (1/3 : ℝ) * π * r^2 * h = 4 * Real.sqrt 3 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l733_73307


namespace NUMINAMATH_CALUDE_units_digit_characteristic_l733_73361

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A predicate to check if a natural number is even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem units_digit_characteristic (p : ℕ) 
  (h1 : p > 0) 
  (h2 : isEven p) 
  (h3 : unitsDigit (p^3) - unitsDigit (p^2) = 0)
  (h4 : unitsDigit (p + 4) = 0) : 
  unitsDigit p = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_characteristic_l733_73361


namespace NUMINAMATH_CALUDE_not_always_sufficient_condition_l733_73301

theorem not_always_sufficient_condition : 
  ¬(∀ (a b c : ℝ), a > b → a * c^2 > b * c^2) :=
by sorry

end NUMINAMATH_CALUDE_not_always_sufficient_condition_l733_73301


namespace NUMINAMATH_CALUDE_sin_cos_sum_17_13_l733_73304

theorem sin_cos_sum_17_13 : 
  Real.sin (17 * π / 180) * Real.cos (13 * π / 180) + 
  Real.cos (17 * π / 180) * Real.sin (13 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_17_13_l733_73304


namespace NUMINAMATH_CALUDE_complex_equation_imag_part_l733_73340

theorem complex_equation_imag_part : 
  ∃ (z : ℂ), (3 - 4*I) * z = Complex.abs (4 + 3*I) ∧ z.im = 4/5 := by sorry

end NUMINAMATH_CALUDE_complex_equation_imag_part_l733_73340


namespace NUMINAMATH_CALUDE_sun_division_problem_l733_73329

/-- Proves that the total amount is 156 rupees given the conditions of the problem -/
theorem sun_division_problem (x y z : ℝ) : 
  (∀ (r : ℝ), r > 0 → y = 0.45 * r ∧ z = 0.5 * r) →  -- For each rupee x gets, y gets 45 paisa and z gets 50 paisa
  y = 36 →  -- y's share is Rs. 36
  x + y + z = 156 := by  -- The total amount is Rs. 156
sorry

end NUMINAMATH_CALUDE_sun_division_problem_l733_73329


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l733_73316

/-- Given a quadrilateral in the Cartesian plane, the sum of the x-coordinates
    of the midpoints of its sides is equal to the sum of the x-coordinates of its vertices. -/
theorem midpoint_sum_equals_vertex_sum (p q r s : ℝ) :
  let vertex_sum := p + q + r + s
  let midpoint_sum := (p + q) / 2 + (q + r) / 2 + (r + s) / 2 + (s + p) / 2
  midpoint_sum = vertex_sum := by sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l733_73316


namespace NUMINAMATH_CALUDE_pole_height_is_seven_meters_l733_73352

/-- Represents the geometry of a leaning telephone pole supported by a cable --/
structure LeaningPole where
  /-- Angle between the pole and the horizontal ground in degrees --/
  angle : ℝ
  /-- Distance from the pole base to the cable attachment point on the ground in meters --/
  cable_ground_distance : ℝ
  /-- Height of the person touching the cable in meters --/
  person_height : ℝ
  /-- Distance the person walks from the pole base towards the cable attachment point in meters --/
  person_distance : ℝ

/-- Calculates the height of the leaning pole given the geometry --/
def calculate_pole_height (pole : LeaningPole) : ℝ :=
  sorry

/-- Theorem stating that for the given conditions, the pole height is 7 meters --/
theorem pole_height_is_seven_meters (pole : LeaningPole) 
  (h_angle : pole.angle = 85)
  (h_cable : pole.cable_ground_distance = 4)
  (h_person_height : pole.person_height = 1.75)
  (h_person_distance : pole.person_distance = 3)
  : calculate_pole_height pole = 7 := by
  sorry

end NUMINAMATH_CALUDE_pole_height_is_seven_meters_l733_73352


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_specific_l733_73386

/-- The length of a bridge given specific train characteristics and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof that the bridge length is 205 meters given the specific conditions -/
theorem bridge_length_specific : bridge_length 170 45 30 = 205 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_specific_l733_73386


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l733_73344

theorem min_value_trig_expression (θ φ : ℝ) :
  (3 * Real.cos θ + 4 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 4 * Real.cos φ - 20)^2 ≥ 235.97 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l733_73344


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l733_73395

/-- The number of players in the basketball team -/
def total_players : ℕ := 18

/-- The number of players in a lineup excluding the point guard -/
def lineup_size : ℕ := 7

/-- The number of different lineups that can be chosen -/
def number_of_lineups : ℕ := total_players * (Nat.choose (total_players - 1) lineup_size)

/-- Theorem stating the number of different lineups -/
theorem basketball_lineup_count : number_of_lineups = 349464 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l733_73395


namespace NUMINAMATH_CALUDE_min_t_value_l733_73303

def f (x : ℝ) := x^3 - 3*x - 1

theorem min_t_value (t : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3) 2 → x₂ ∈ Set.Icc (-3) 2 → |f x₁ - f x₂| ≤ t) ↔ t ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_min_t_value_l733_73303
