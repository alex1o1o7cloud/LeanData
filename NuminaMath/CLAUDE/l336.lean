import Mathlib

namespace NUMINAMATH_CALUDE_edward_work_hours_l336_33690

theorem edward_work_hours (hourly_rate : ℝ) (max_regular_hours : ℕ) (total_earnings : ℝ) :
  hourly_rate = 7 →
  max_regular_hours = 40 →
  total_earnings = 210 →
  ∃ (hours_worked : ℕ), hours_worked = 30 ∧ (hours_worked : ℝ) * hourly_rate = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_edward_work_hours_l336_33690


namespace NUMINAMATH_CALUDE_empty_set_implies_m_zero_l336_33618

theorem empty_set_implies_m_zero (m : ℝ) : (∀ x : ℝ, m * x ≠ 1) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_set_implies_m_zero_l336_33618


namespace NUMINAMATH_CALUDE_layla_score_comparison_l336_33637

/-- Represents a player in the game -/
inductive Player : Type
| Layla : Player
| Nahima : Player
| Ramon : Player
| Aria : Player

/-- Represents a round in the game -/
inductive Round : Type
| First : Round
| Second : Round
| Third : Round

/-- The scoring function for the game -/
def score (p : Player) (r : Round) : ℕ → ℕ :=
  match r with
  | Round.First => (· * 2)
  | Round.Second => (· * 3)
  | Round.Third => id

/-- The total score of a player across all rounds -/
def totalScore (p : Player) (s1 s2 s3 : ℕ) : ℕ :=
  score p Round.First s1 + score p Round.Second s2 + score p Round.Third s3

theorem layla_score_comparison :
  ∀ (nahima_total ramon_total aria_total : ℕ),
  totalScore Player.Layla 120 90 (760 - score Player.Layla Round.First 120 - score Player.Layla Round.Second 90) = 760 →
  nahima_total + ramon_total + aria_total = 1330 - 760 →
  760 - score Player.Layla Round.First 120 - score Player.Layla Round.Second 90 =
    nahima_total + ramon_total + aria_total - 320 :=
by sorry

end NUMINAMATH_CALUDE_layla_score_comparison_l336_33637


namespace NUMINAMATH_CALUDE_largest_value_l336_33658

theorem largest_value (a b c d e : ℝ) 
  (ha : a = 1 - 0.1)
  (hb : b = 1 - 0.01)
  (hc : c = 1 - 0.001)
  (hd : d = 1 - 0.0001)
  (he : e = 1 - 0.00001) :
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l336_33658


namespace NUMINAMATH_CALUDE_factorization_coefficient_sum_l336_33674

theorem factorization_coefficient_sum : ∃ (a b c d e f g : ℤ),
  (125 : ℤ) * X^6 - 216 * Y^6 = (a * X^2 + b * Y^2) * (c * X^2 + d * Y^2) * (e * X^4 + f * X^2 * Y^2 + g * Y^4) ∧
  a + b + c + d + e + f + g = 41 :=
sorry

end NUMINAMATH_CALUDE_factorization_coefficient_sum_l336_33674


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_over_four_l336_33665

theorem tan_theta_minus_pi_over_four (θ : ℝ) (h : Real.cos θ - 3 * Real.sin θ = 0) :
  Real.tan (θ - π/4) = -1/2 := by sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_over_four_l336_33665


namespace NUMINAMATH_CALUDE_factor_tree_X_value_l336_33667

/-- Represents a node in the factor tree -/
structure TreeNode where
  value : ℕ

/-- Represents the factor tree structure -/
structure FactorTree where
  X : TreeNode
  F : TreeNode
  G : TreeNode
  H : TreeNode

/-- The main theorem to prove -/
theorem factor_tree_X_value (tree : FactorTree) : tree.X.value = 6776 :=
  sorry

/-- Axioms representing the given conditions -/
axiom F_value (tree : FactorTree) : tree.F.value = 7 * 4

axiom G_value (tree : FactorTree) : tree.G.value = 11 * tree.H.value

axiom H_value (tree : FactorTree) : tree.H.value = 11 * 2

axiom X_value (tree : FactorTree) : tree.X.value = tree.F.value * tree.G.value

end NUMINAMATH_CALUDE_factor_tree_X_value_l336_33667


namespace NUMINAMATH_CALUDE_nursing_home_medicine_boxes_l336_33613

/-- The total number of boxes of medicine received by the nursing home -/
def total_boxes (vitamin_boxes supplement_boxes : ℕ) : ℕ :=
  vitamin_boxes + supplement_boxes

/-- Theorem stating that the nursing home received 760 boxes of medicine -/
theorem nursing_home_medicine_boxes : 
  total_boxes 472 288 = 760 := by
  sorry

end NUMINAMATH_CALUDE_nursing_home_medicine_boxes_l336_33613


namespace NUMINAMATH_CALUDE_gcf_of_180_240_300_l336_33620

theorem gcf_of_180_240_300 : Nat.gcd 180 (Nat.gcd 240 300) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_240_300_l336_33620


namespace NUMINAMATH_CALUDE_sample_survey_most_appropriate_l336_33608

/-- Represents a survey method --/
inductive SurveyMethod
  | InterestGroup
  | FamiliarFriends
  | AllStudents
  | SampleSurvey

/-- Criteria for evaluating survey methods --/
structure SurveyCriteria where
  representativeness : Bool
  practicality : Bool
  efficiency : Bool

/-- Evaluates a survey method based on given criteria --/
def evaluateSurveyMethod (method : SurveyMethod) : SurveyCriteria :=
  match method with
  | SurveyMethod.InterestGroup => { representativeness := false, practicality := true, efficiency := true }
  | SurveyMethod.FamiliarFriends => { representativeness := false, practicality := true, efficiency := true }
  | SurveyMethod.AllStudents => { representativeness := true, practicality := false, efficiency := false }
  | SurveyMethod.SampleSurvey => { representativeness := true, practicality := true, efficiency := true }

/-- Determines if a survey method is appropriate based on all criteria being met --/
def isAppropriateMethod (criteria : SurveyCriteria) : Bool :=
  criteria.representativeness ∧ criteria.practicality ∧ criteria.efficiency

/-- Theorem stating that the sample survey method is the most appropriate --/
theorem sample_survey_most_appropriate :
  ∀ (method : SurveyMethod),
    method = SurveyMethod.SampleSurvey ↔ isAppropriateMethod (evaluateSurveyMethod method) :=
  sorry


end NUMINAMATH_CALUDE_sample_survey_most_appropriate_l336_33608


namespace NUMINAMATH_CALUDE_toy_piles_total_l336_33651

theorem toy_piles_total (small_pile large_pile : ℕ) : 
  large_pile = 2 * small_pile → 
  large_pile = 80 → 
  small_pile + large_pile = 120 :=
by sorry

end NUMINAMATH_CALUDE_toy_piles_total_l336_33651


namespace NUMINAMATH_CALUDE_other_polynomial_form_l336_33600

/-- Given two polynomials with a specified difference, this theorem proves the form of the other polynomial. -/
theorem other_polynomial_form (a b c d : ℝ) 
  (diff : ℝ) -- The difference between the two polynomials
  (poly1 : ℝ) -- One of the polynomials
  (h1 : diff = c^2 * d^2 - a^2 * b^2) -- Condition on the difference
  (h2 : poly1 = a^2 * b^2 + c^2 * d^2 - 2*a*b*c*d) -- Condition on one polynomial
  : ∃ (poly2 : ℝ), (poly2 = 2*c^2*d^2 - 2*a*b*c*d ∨ poly2 = 2*a^2*b^2 - 2*a*b*c*d) ∧ 
    ((poly1 - poly2 = diff) ∨ (poly2 - poly1 = diff)) :=
by
  sorry

end NUMINAMATH_CALUDE_other_polynomial_form_l336_33600


namespace NUMINAMATH_CALUDE_roselyn_remaining_books_l336_33642

/-- The number of books Roselyn has after giving books to Mara and Rebecca -/
def books_remaining (initial_books rebecca_books : ℕ) : ℕ :=
  initial_books - (rebecca_books + 3 * rebecca_books)

/-- Theorem stating that Roselyn has 60 books after giving books to Mara and Rebecca -/
theorem roselyn_remaining_books :
  books_remaining 220 40 = 60 := by
  sorry

end NUMINAMATH_CALUDE_roselyn_remaining_books_l336_33642


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l336_33692

theorem sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((1/2 : ℝ)^a < (1/2 : ℝ)^b → Real.log (a + 1) > Real.log b) ∧
  ¬(Real.log (a + 1) > Real.log b → (1/2 : ℝ)^a < (1/2 : ℝ)^b) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l336_33692


namespace NUMINAMATH_CALUDE_squash_players_l336_33671

/-- Given a class of children with information about their sport participation,
    calculate the number of children who play squash. -/
theorem squash_players (total : ℕ) (tennis : ℕ) (neither : ℕ) (both : ℕ) :
  total = 38 →
  tennis = 19 →
  neither = 10 →
  both = 12 →
  ∃ (squash : ℕ), squash = 21 ∧ 
    squash = total - neither - (tennis - both) := by
  sorry

#check squash_players

end NUMINAMATH_CALUDE_squash_players_l336_33671


namespace NUMINAMATH_CALUDE_rectangle_to_square_l336_33632

theorem rectangle_to_square (k : ℕ) (h1 : k > 5) :
  (∃ n : ℕ, k * (k - 5) = n^2) → k * (k - 5) = 6^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l336_33632


namespace NUMINAMATH_CALUDE_perfect_square_condition_l336_33668

/-- If 100x^2 - kxy + 49y^2 is a perfect square, then k = ±140 -/
theorem perfect_square_condition (x y k : ℝ) :
  (∃ (z : ℝ), 100 * x^2 - k * x * y + 49 * y^2 = z^2) →
  (k = 140 ∨ k = -140) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l336_33668


namespace NUMINAMATH_CALUDE_flight_speed_l336_33686

/-- Given a flight distance and time, calculate the speed -/
theorem flight_speed (distance : ℝ) (time : ℝ) (h1 : distance = 256) (h2 : time = 8) :
  distance / time = 32 := by
  sorry

end NUMINAMATH_CALUDE_flight_speed_l336_33686


namespace NUMINAMATH_CALUDE_white_surface_area_fraction_l336_33635

theorem white_surface_area_fraction (large_cube_edge : ℕ) (small_cube_edge : ℕ) 
  (total_small_cubes : ℕ) (white_small_cubes : ℕ) (black_small_cubes : ℕ) :
  large_cube_edge = 4 →
  small_cube_edge = 1 →
  total_small_cubes = 64 →
  white_small_cubes = 56 →
  black_small_cubes = 8 →
  white_small_cubes + black_small_cubes = total_small_cubes →
  black_small_cubes = large_cube_edge^2 →
  (((6 * large_cube_edge^2) - large_cube_edge^2) : ℚ) / (6 * large_cube_edge^2) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_area_fraction_l336_33635


namespace NUMINAMATH_CALUDE_even_painted_faces_5x5x1_l336_33623

/-- Represents a 3D rectangular block -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with an even number of painted faces in a given block -/
def count_even_painted_faces (b : Block) : ℕ :=
  sorry

/-- The theorem stating that a 5x5x1 block has 12 cubes with an even number of painted faces -/
theorem even_painted_faces_5x5x1 :
  let b : Block := { length := 5, width := 5, height := 1 }
  count_even_painted_faces b = 12 := by
  sorry

end NUMINAMATH_CALUDE_even_painted_faces_5x5x1_l336_33623


namespace NUMINAMATH_CALUDE_negative_two_and_negative_half_are_reciprocals_l336_33602

-- Define the concept of reciprocals
def are_reciprocals (a b : ℚ) : Prop := a * b = 1

-- Theorem statement
theorem negative_two_and_negative_half_are_reciprocals :
  are_reciprocals (-2) (-1/2) := by
  sorry

end NUMINAMATH_CALUDE_negative_two_and_negative_half_are_reciprocals_l336_33602


namespace NUMINAMATH_CALUDE_complement_union_theorem_l336_33612

open Set

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l336_33612


namespace NUMINAMATH_CALUDE_pebble_collection_sum_l336_33661

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of an arithmetic sequence with n terms, starting at a and increasing by d each term -/
def arithmetic_sum (n a d : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem pebble_collection_sum :
  arithmetic_sum 12 1 1 = 78 := by
  sorry

end NUMINAMATH_CALUDE_pebble_collection_sum_l336_33661


namespace NUMINAMATH_CALUDE_intersection_equality_theorem_l336_33648

/-- The set A of solutions to x^2 + 2x - 3 = 0 -/
def A : Set ℝ := {x | x^2 + 2*x - 3 = 0}

/-- The set B of solutions to x^2 - (k+1)x + k = 0 for a given k -/
def B (k : ℝ) : Set ℝ := {x | x^2 - (k+1)*x + k = 0}

/-- The theorem stating that the set of k values satisfying A ∩ B = B is {1, -3} -/
theorem intersection_equality_theorem :
  {k : ℝ | A ∩ B k = B k} = {1, -3} := by sorry

end NUMINAMATH_CALUDE_intersection_equality_theorem_l336_33648


namespace NUMINAMATH_CALUDE_david_average_marks_l336_33639

def david_marks : List ℝ := [70, 63, 80, 63, 65]

theorem david_average_marks :
  (david_marks.sum / david_marks.length : ℝ) = 68.2 := by
  sorry

end NUMINAMATH_CALUDE_david_average_marks_l336_33639


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l336_33614

/-- Calculates the total amount to pay after applying discounts to two bills -/
def total_amount_after_discount (bill1 bill2 discount1 discount2 : ℚ) : ℚ :=
  (bill1 * (1 - discount1 / 100)) + (bill2 * (1 - discount2 / 100))

/-- Theorem stating that the total amount Bob and Kate pay after discounts is $53 -/
theorem restaurant_bill_theorem :
  total_amount_after_discount 30 25 5 2 = 53 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l336_33614


namespace NUMINAMATH_CALUDE_lisa_coffee_consumption_l336_33666

/-- The number of cups of coffee Lisa drank -/
def cups_of_coffee : ℕ := sorry

/-- The amount of caffeine in milligrams per cup of coffee -/
def caffeine_per_cup : ℕ := 80

/-- Lisa's daily caffeine limit in milligrams -/
def daily_limit : ℕ := 200

/-- The amount of caffeine Lisa consumed over her daily limit in milligrams -/
def excess_caffeine : ℕ := 40

/-- Theorem stating that Lisa drank 3 cups of coffee -/
theorem lisa_coffee_consumption : cups_of_coffee = 3 := by sorry

end NUMINAMATH_CALUDE_lisa_coffee_consumption_l336_33666


namespace NUMINAMATH_CALUDE_prism_volume_l336_33625

/-- The volume of a right prism with an equilateral triangular base -/
theorem prism_volume (a : ℝ) (h : ℝ) : 
  a = 5 → -- Side length of the equilateral triangle base
  (a * h * 2 + a^2 * Real.sqrt 3 / 4) = 40 → -- Sum of areas of three adjacent faces
  a * a * Real.sqrt 3 / 4 * h = 625 / 160 * (3 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_prism_volume_l336_33625


namespace NUMINAMATH_CALUDE_color_partition_impossibility_l336_33626

theorem color_partition_impossibility : ¬ ∃ (A B C : Set ℕ),
  (∀ n : ℕ, n > 1 → (n ∈ A ∨ n ∈ B ∨ n ∈ C)) ∧
  (A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅) ∧
  (A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅) ∧
  (∀ x y, x ∈ A → y ∈ B → x * y ∈ C) ∧
  (∀ x z, x ∈ A → z ∈ C → x * z ∈ B) ∧
  (∀ y z, y ∈ B → z ∈ C → y * z ∈ A) :=
sorry

end NUMINAMATH_CALUDE_color_partition_impossibility_l336_33626


namespace NUMINAMATH_CALUDE_later_purchase_cost_l336_33604

/-- The cost of a single bat in dollars -/
def bat_cost : ℕ := 500

/-- The cost of a single ball in dollars -/
def ball_cost : ℕ := 100

/-- The number of bats in the later purchase -/
def num_bats : ℕ := 3

/-- The number of balls in the later purchase -/
def num_balls : ℕ := 5

/-- The total cost of the later purchase -/
def total_cost : ℕ := num_bats * bat_cost + num_balls * ball_cost

theorem later_purchase_cost : total_cost = 2000 := by
  sorry

end NUMINAMATH_CALUDE_later_purchase_cost_l336_33604


namespace NUMINAMATH_CALUDE_ratio_MBQ_ABQ_l336_33689

-- Define the points
variable (A B C P Q M : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- BP and BQ bisect ∠ABC
axiom bisect_ABC : angle A B P = angle P B C ∧ angle A B Q = angle Q B C

-- BM trisects ∠PBQ
axiom trisect_PBQ : angle P B M = angle M B Q ∧ 3 * angle M B Q = angle P B Q

-- Theorem to prove
theorem ratio_MBQ_ABQ : angle M B Q / angle A B Q = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_ratio_MBQ_ABQ_l336_33689


namespace NUMINAMATH_CALUDE_derivative_sin_cos_x_l336_33628

theorem derivative_sin_cos_x (x : ℝ) : 
  deriv (λ x => Real.sin x * Real.cos x) x = Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_cos_x_l336_33628


namespace NUMINAMATH_CALUDE_schedule_theorem_l336_33694

/-- The number of lessons to be scheduled -/
def total_lessons : ℕ := 6

/-- The number of morning periods -/
def morning_periods : ℕ := 4

/-- The number of afternoon periods -/
def afternoon_periods : ℕ := 2

/-- The number of ways to arrange the schedule -/
def schedule_arrangements : ℕ := 192

theorem schedule_theorem :
  (morning_periods.choose 1) * (afternoon_periods.choose 1) * (total_lessons - 2).factorial = schedule_arrangements :=
sorry

end NUMINAMATH_CALUDE_schedule_theorem_l336_33694


namespace NUMINAMATH_CALUDE_park_diagonal_ratio_l336_33607

theorem park_diagonal_ratio :
  ∀ (long_side : ℝ) (short_side : ℝ) (diagonal : ℝ),
    short_side = long_side / 2 →
    long_side + short_side - diagonal = long_side / 3 →
    long_side / diagonal = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_park_diagonal_ratio_l336_33607


namespace NUMINAMATH_CALUDE_fold_points_area_theorem_l336_33660

/-- Represents a triangle with sides DE, DF, and angle E -/
structure Triangle where
  de : ℝ
  df : ℝ
  angleE : ℝ

/-- Represents the area of the set of fold points -/
structure FoldPointsArea where
  u : ℕ
  v : ℕ
  w : ℕ

/-- Calculates the area of the set of fold points for a given triangle -/
def calculateFoldPointsArea (t : Triangle) : FoldPointsArea :=
  sorry

theorem fold_points_area_theorem (t : Triangle) 
  (h1 : t.de = 48)
  (h2 : t.df = 96)
  (h3 : t.angleE = 90)
  : let area := calculateFoldPointsArea t
    area.u = 432 ∧ 
    area.v = 518 ∧ 
    area.w = 3 ∧ 
    area.u + area.v + area.w = 953 :=
  sorry

end NUMINAMATH_CALUDE_fold_points_area_theorem_l336_33660


namespace NUMINAMATH_CALUDE_cubic_divisibility_l336_33610

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluation of the cubic polynomial at a given point -/
def CubicPolynomial.eval (p : CubicPolynomial) (x : ℤ) : ℤ :=
  x^3 + p.a * x^2 + p.b * x + p.c

/-- Condition that one root is the product of the other two -/
def has_product_root (p : CubicPolynomial) : Prop :=
  ∃ (u v : ℚ), u ≠ 0 ∧ v ≠ 0 ∧ 
    (u + v + u*v = -p.a) ∧
    (u*v*(1 + u + v) = p.b) ∧
    (u^2 * v^2 = -p.c)

/-- Main theorem statement -/
theorem cubic_divisibility (p : CubicPolynomial) (h : has_product_root p) :
  (2 * p.eval (-1)) ∣ (p.eval 1 + p.eval (-1) - 2 * (1 + p.eval 0)) :=
sorry

end NUMINAMATH_CALUDE_cubic_divisibility_l336_33610


namespace NUMINAMATH_CALUDE_fraction_addition_l336_33629

theorem fraction_addition : (4 : ℚ) / 510 + 25 / 34 = 379 / 510 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l336_33629


namespace NUMINAMATH_CALUDE_quadratic_equation_single_solution_l336_33693

theorem quadratic_equation_single_solution :
  ∀ b : ℝ, b ≠ 0 →
  (∃! x : ℝ, b * x^2 - 24 * x + 6 = 0) →
  (∃ x : ℝ, b * x^2 - 24 * x + 6 = 0 ∧ x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_single_solution_l336_33693


namespace NUMINAMATH_CALUDE_fish_population_estimate_l336_33675

/-- Estimate the number of fish in a reservoir using the capture-recapture method. -/
theorem fish_population_estimate
  (M : ℕ) -- Number of fish initially captured, marked, and released
  (m : ℕ) -- Number of fish captured in the second round
  (n : ℕ) -- Number of marked fish found in the second capture
  (h1 : M > 0)
  (h2 : m > 0)
  (h3 : n > 0)
  (h4 : n ≤ m)
  (h5 : n ≤ M) :
  ∃ x : ℚ, x = (M * m : ℚ) / n ∧ x > 0 :=
sorry

end NUMINAMATH_CALUDE_fish_population_estimate_l336_33675


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l336_33684

/-- The sum of the digits of 10^93 - 93 -/
def sum_of_digits : ℕ := 826

/-- The number represented by 10^93 - 93 -/
def large_number : ℕ := 10^93 - 93

/-- Function to calculate the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem sum_of_digits_of_large_number :
  digit_sum large_number = sum_of_digits :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l336_33684


namespace NUMINAMATH_CALUDE_condition_one_condition_two_l336_33644

-- Define set A
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}

-- Define set B
def B : Set ℝ := {x | x < -1 ∨ x > 3}

-- Theorem for condition 1
theorem condition_one (a : ℝ) : A a ∩ B = A a → a < -3 ∨ a > 3 := by
  sorry

-- Theorem for condition 2
theorem condition_two (a : ℝ) : (A a ∩ B).Nonempty → a < -1 ∨ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_condition_one_condition_two_l336_33644


namespace NUMINAMATH_CALUDE_alice_least_money_l336_33624

-- Define the set of people
inductive Person : Type
  | Alice : Person
  | Bob : Person
  | Charlie : Person
  | Dana : Person
  | Eve : Person

-- Define the money function
variable (money : Person → ℝ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q

axiom charlie_most : ∀ (p : Person), p ≠ Person.Charlie → money p < money Person.Charlie

axiom bob_more_than_alice : money Person.Alice < money Person.Bob

axiom dana_more_than_alice : money Person.Alice < money Person.Dana

axiom eve_between_alice_and_bob : 
  money Person.Alice < money Person.Eve ∧ money Person.Eve < money Person.Bob

-- State the theorem
theorem alice_least_money : 
  ∀ (p : Person), p ≠ Person.Alice → money Person.Alice < money p := by
  sorry

end NUMINAMATH_CALUDE_alice_least_money_l336_33624


namespace NUMINAMATH_CALUDE_seven_row_triangle_pieces_l336_33653

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Calculates the sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Represents the structure of the triangle -/
structure TriangleStructure where
  rows : ℕ
  rodIncrease : ℕ
  extraConnectorRow : ℕ

/-- Calculates the total number of pieces in the triangle -/
def totalPieces (t : TriangleStructure) : ℕ :=
  let totalRods := arithmeticSum 3 t.rodIncrease t.rows
  let totalConnectors := triangularNumber (t.rows + t.extraConnectorRow)
  totalRods + totalConnectors

/-- The main theorem to prove -/
theorem seven_row_triangle_pieces :
  let t : TriangleStructure := {
    rows := 7,
    rodIncrease := 3,
    extraConnectorRow := 1
  }
  totalPieces t = 120 := by sorry

end NUMINAMATH_CALUDE_seven_row_triangle_pieces_l336_33653


namespace NUMINAMATH_CALUDE_equilateral_triangle_height_equals_rectangle_width_l336_33696

theorem equilateral_triangle_height_equals_rectangle_width (w : ℝ) :
  let rectangle_area := 2 * w^2
  let triangle_side := (2 * w^2 * 4 / Real.sqrt 3).sqrt
  let triangle_height := triangle_side * Real.sqrt 3 / 2
  triangle_height = w * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_height_equals_rectangle_width_l336_33696


namespace NUMINAMATH_CALUDE_james_future_age_l336_33621

def justin_age : ℕ := 26
def jessica_age_at_justin_birth : ℕ := 6
def james_age_diff_jessica : ℕ := 7
def years_in_future : ℕ := 5

theorem james_future_age :
  justin_age + jessica_age_at_justin_birth + james_age_diff_jessica + years_in_future = 44 :=
by sorry

end NUMINAMATH_CALUDE_james_future_age_l336_33621


namespace NUMINAMATH_CALUDE_hyperbola_parameter_sum_l336_33643

/-- Theorem about the sum of parameters for a specific hyperbola -/
theorem hyperbola_parameter_sum :
  let center : ℝ × ℝ := (1, 3)
  let focus : ℝ × ℝ := (1, 9)
  let vertex : ℝ × ℝ := (1, 0)
  let h : ℝ := center.1
  let k : ℝ := center.2
  let a : ℝ := |k - vertex.2|
  let c : ℝ := |k - focus.2|
  let b : ℝ := Real.sqrt (c^2 - a^2)
  h + k + a + b = 7 + 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_parameter_sum_l336_33643


namespace NUMINAMATH_CALUDE_prom_couples_count_l336_33654

theorem prom_couples_count (total_students : ℕ) (solo_students : ℕ) (couples : ℕ) : 
  total_students = 123 → 
  solo_students = 3 → 
  couples = (total_students - solo_students) / 2 → 
  couples = 60 := by
  sorry

end NUMINAMATH_CALUDE_prom_couples_count_l336_33654


namespace NUMINAMATH_CALUDE_five_fourths_of_eight_thirds_l336_33609

theorem five_fourths_of_eight_thirds (x : ℚ) : 
  x = 8 / 3 → (5 / 4) * x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_eight_thirds_l336_33609


namespace NUMINAMATH_CALUDE_product_inequality_l336_33631

theorem product_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq : a + d = b + c) 
  (abs_ineq : |a - d| < |b - c|) : 
  a * d < b * c := by
sorry

end NUMINAMATH_CALUDE_product_inequality_l336_33631


namespace NUMINAMATH_CALUDE_dust_retention_proof_l336_33691

/-- The average annual dust retention of a locust leaf in milligrams. -/
def locust_dust_retention : ℝ := 22

/-- The average annual dust retention of a ginkgo leaf in milligrams. -/
def ginkgo_dust_retention : ℝ := 2 * locust_dust_retention - 4

theorem dust_retention_proof :
  11 * ginkgo_dust_retention = 20 * locust_dust_retention :=
by sorry

end NUMINAMATH_CALUDE_dust_retention_proof_l336_33691


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l336_33664

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving the given rectangular floor is Rs. 28,875 -/
theorem paving_cost_calculation :
  paving_cost 5.5 3.75 1400 = 28875 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l336_33664


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l336_33657

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1 ∧ m1 ≠ 0 ∧ m2 ≠ 0

/-- The theorem states that a = 2 is a necessary and sufficient condition
    for the lines 2x - ay + 1 = 0 and (a-1)x - y + a = 0 to be parallel -/
theorem parallel_iff_a_eq_two (a : ℝ) :
  parallel 2 (-a) 1 (a-1) (-1) a ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l336_33657


namespace NUMINAMATH_CALUDE_valid_digits_l336_33640

/-- Given a digit x, construct the number 20x06 -/
def construct_number (x : Nat) : Nat := 20000 + x * 100 + 6

/-- Predicate to check if a given digit satisfies the divisibility condition -/
def is_valid_digit (x : Nat) : Prop :=
  x < 10 ∧ (construct_number x) % 7 = 0

theorem valid_digits :
  ∀ x, is_valid_digit x ↔ (x = 0 ∨ x = 7) :=
sorry

end NUMINAMATH_CALUDE_valid_digits_l336_33640


namespace NUMINAMATH_CALUDE_six_students_two_restricted_pairs_l336_33659

/-- Represents a pair of students who refuse to stand next to each other -/
structure RestrictedPair :=
  (student1 : ℕ)
  (student2 : ℕ)

/-- Calculates the number of ways to arrange students with restrictions -/
def arrange_students_with_restrictions (n : ℕ) (restricted_pairs : List RestrictedPair) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements for the given problem -/
theorem six_students_two_restricted_pairs :
  arrange_students_with_restrictions 6 
    [⟨1, 2⟩, ⟨3, 4⟩] = 336 :=
by sorry

end NUMINAMATH_CALUDE_six_students_two_restricted_pairs_l336_33659


namespace NUMINAMATH_CALUDE_triangle_side_length_l336_33678

theorem triangle_side_length (A B C : ℝ × ℝ) :
  let AC := Real.sqrt 3
  let AB := 2
  let angle_B := 60 * Real.pi / 180
  let BC := Real.sqrt ((AC^2 + AB^2) - 2 * AC * AB * Real.cos angle_B)
  AC = Real.sqrt 3 ∧ AB = 2 ∧ angle_B = 60 * Real.pi / 180 →
  BC = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l336_33678


namespace NUMINAMATH_CALUDE_blocks_needed_per_color_l336_33663

/-- Represents the dimensions of a clay block -/
structure BlockDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a cylindrical pot -/
structure PotDimensions where
  height : ℝ
  diameter : ℝ

/-- Calculates the number of blocks needed for each color -/
def blocksPerColor (block : BlockDimensions) (pot : PotDimensions) (layerHeight : ℝ) : ℕ :=
  sorry

/-- Theorem stating that 7 blocks of each color are needed -/
theorem blocks_needed_per_color 
  (block : BlockDimensions)
  (pot : PotDimensions)
  (layerHeight : ℝ)
  (h1 : block.length = 4)
  (h2 : block.width = 3)
  (h3 : block.height = 2)
  (h4 : pot.height = 10)
  (h5 : pot.diameter = 5)
  (h6 : layerHeight = 2.5) :
  blocksPerColor block pot layerHeight = 7 := by
  sorry

end NUMINAMATH_CALUDE_blocks_needed_per_color_l336_33663


namespace NUMINAMATH_CALUDE_edward_tickets_l336_33697

/-- The number of tickets Edward won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 3

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 4

/-- The number of candies Edward could buy -/
def candies_bought : ℕ := 2

/-- The number of tickets Edward won playing 'skee ball' -/
def skee_ball_tickets : ℕ := sorry

theorem edward_tickets : skee_ball_tickets = 5 := by
  sorry

end NUMINAMATH_CALUDE_edward_tickets_l336_33697


namespace NUMINAMATH_CALUDE_pyramidal_stack_logs_example_l336_33603

/-- Calculates the total number of logs in a pyramidal stack. -/
def pyramidal_stack_logs (bottom_row : ℕ) (top_row : ℕ) (difference : ℕ) : ℕ :=
  let n := (bottom_row - top_row) / difference + 1
  n * (bottom_row + top_row) / 2

/-- Proves that the total number of logs in the given pyramidal stack is 60. -/
theorem pyramidal_stack_logs_example : pyramidal_stack_logs 15 5 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_pyramidal_stack_logs_example_l336_33603


namespace NUMINAMATH_CALUDE_shopkeeper_cheating_profit_l336_33619

/-- The percentage by which the shopkeeper increases the weight when buying from the supplier -/
def supplier_increase_percent : ℝ := 20

/-- The profit percentage the shopkeeper aims to achieve -/
def target_profit_percent : ℝ := 32

/-- The percentage by which the shopkeeper increases the weight when selling to the customer -/
def customer_increase_percent : ℝ := 26.67

theorem shopkeeper_cheating_profit (initial_weight : ℝ) (h : initial_weight > 0) :
  let actual_weight := initial_weight * (1 + supplier_increase_percent / 100)
  let selling_weight := actual_weight * (1 + customer_increase_percent / 100)
  (selling_weight - actual_weight) / initial_weight * 100 = target_profit_percent := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_cheating_profit_l336_33619


namespace NUMINAMATH_CALUDE_sine_inequality_solution_set_l336_33622

theorem sine_inequality_solution_set 
  (a : ℝ) 
  (h1 : -1 < a) 
  (h2 : a < 0) 
  (θ : ℝ) 
  (h3 : θ = Real.arcsin a) : 
  {x : ℝ | ∃ (n : ℤ), (2*n - 1)*π - θ < x ∧ x < 2*n*π + θ} = 
  {x : ℝ | Real.sin x < a} := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_solution_set_l336_33622


namespace NUMINAMATH_CALUDE_cos_negative_480_deg_l336_33655

theorem cos_negative_480_deg : Real.cos (-(480 * π / 180)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_480_deg_l336_33655


namespace NUMINAMATH_CALUDE_absolute_value_fraction_l336_33679

theorem absolute_value_fraction (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 10*a*b) :
  |((a - b) / (a + b))| = Real.sqrt (2/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_l336_33679


namespace NUMINAMATH_CALUDE_gcd_16_12_is_4_l336_33634

theorem gcd_16_12_is_4 : Nat.gcd 16 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_16_12_is_4_l336_33634


namespace NUMINAMATH_CALUDE_gasoline_tank_problem_l336_33676

/-- Proves properties of a gasoline tank given initial and final fill levels -/
theorem gasoline_tank_problem (x : ℚ) 
  (h1 : 5/6 * x - 2/3 * x = 18) 
  (h2 : x > 0) : 
  x = 108 ∧ 18 * 4 = 72 := by
  sorry

#check gasoline_tank_problem

end NUMINAMATH_CALUDE_gasoline_tank_problem_l336_33676


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_ten_l336_33649

theorem product_of_five_consecutive_integers_divisible_by_ten (n : ℕ) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_ten_l336_33649


namespace NUMINAMATH_CALUDE_new_person_weight_l336_33611

theorem new_person_weight (n : ℕ) (initial_avg weight_replaced increase : ℝ) :
  n = 8 →
  initial_avg = 57 →
  weight_replaced = 55 →
  increase = 1.5 →
  (n * initial_avg + (weight_replaced + increase * n) - weight_replaced) / n = initial_avg + increase →
  weight_replaced + increase * n = 67 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l336_33611


namespace NUMINAMATH_CALUDE_complex_equation_solution_l336_33669

theorem complex_equation_solution (z : ℂ) : z * (1 + 2*I) = 3 + I → z = 1 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l336_33669


namespace NUMINAMATH_CALUDE_flower_problem_l336_33699

theorem flower_problem (total : ℕ) (roses_fraction : ℚ) (tulips : ℕ) (carnations : ℕ) : 
  total = 40 →
  roses_fraction = 2 / 5 →
  tulips = 10 →
  carnations = total - (roses_fraction * total).num - tulips →
  carnations = 14 := by
sorry

end NUMINAMATH_CALUDE_flower_problem_l336_33699


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l336_33615

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define an isosceles triangle
def Isosceles (t : Triangle A B C) : Prop :=
  ‖A - B‖ = ‖B - C‖

-- Define the angle measure function
def AngleMeasure (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem isosceles_triangle_angle_measure 
  (A B C D : ℝ × ℝ) 
  (t : Triangle A B C) 
  (h_isosceles : Isosceles t) 
  (h_angle_C : AngleMeasure B C A = 50) :
  AngleMeasure C B D = 115 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l336_33615


namespace NUMINAMATH_CALUDE_marias_trip_l336_33605

theorem marias_trip (total_distance : ℝ) (h1 : total_distance = 360) : 
  let first_stop := total_distance / 2
  let remaining_after_first := total_distance - first_stop
  let second_stop := remaining_after_first / 4
  let distance_after_second := remaining_after_first - second_stop
  distance_after_second = 135 := by
sorry

end NUMINAMATH_CALUDE_marias_trip_l336_33605


namespace NUMINAMATH_CALUDE_sin_2theta_value_l336_33688

theorem sin_2theta_value (θ : Real) :
  (π < θ ∧ θ < 3*π/2) →  -- θ is in the third quadrant
  (Real.sin θ)^4 + (Real.cos θ)^4 = 5/9 →
  Real.sin (2*θ) = 2*Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l336_33688


namespace NUMINAMATH_CALUDE_trees_planted_per_cut_l336_33677

/-- Proves that the number of new trees planted for each tree cut is 5 --/
theorem trees_planted_per_cut (initial_trees : ℕ) (cut_percentage : ℚ) (final_trees : ℕ) : 
  initial_trees = 400 → 
  cut_percentage = 1/5 →
  final_trees = 720 →
  (final_trees - (initial_trees - initial_trees * cut_percentage)) / (initial_trees * cut_percentage) = 5 := by
  sorry

end NUMINAMATH_CALUDE_trees_planted_per_cut_l336_33677


namespace NUMINAMATH_CALUDE_remainder_5432876543_mod_101_l336_33645

theorem remainder_5432876543_mod_101 : 5432876543 % 101 = 79 := by
  sorry

end NUMINAMATH_CALUDE_remainder_5432876543_mod_101_l336_33645


namespace NUMINAMATH_CALUDE_systems_solutions_l336_33641

-- Define the systems of equations
def system1 (x y : ℝ) : Prop :=
  y = 2 * x - 5 ∧ 3 * x + 4 * y = 2

def system2 (x y : ℝ) : Prop :=
  3 * x - y = 8 ∧ (y - 1) / 3 = (x + 5) / 5

-- State the theorem
theorem systems_solutions :
  (∃ (x y : ℝ), system1 x y ∧ x = 2 ∧ y = -1) ∧
  (∃ (x y : ℝ), system2 x y ∧ x = 5 ∧ y = 7) := by
  sorry

end NUMINAMATH_CALUDE_systems_solutions_l336_33641


namespace NUMINAMATH_CALUDE_min_value_x_l336_33672

theorem min_value_x (x : ℝ) : 2 * (x + 1) ≥ x + 1 → x ≥ -1 ∧ ∀ y, (∀ z, 2 * (z + 1) ≥ z + 1 → z ≥ y) → y ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_l336_33672


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l336_33606

theorem binomial_expansion_sum (n : ℕ) : 
  (∃ p q : ℕ, p = (3 + 1)^n ∧ q = 2^n ∧ p + q = 272) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l336_33606


namespace NUMINAMATH_CALUDE_age_relationships_l336_33601

-- Define variables for ages
variable (a b c d : ℝ)

-- Define the conditions
def condition1 : Prop := a + b = b + c + d + 18
def condition2 : Prop := a / c = 3 / 2

-- Define the theorem
theorem age_relationships 
  (h1 : condition1 a b c d) 
  (h2 : condition2 a c) : 
  c = (2/3) * a ∧ d = (1/3) * a - 18 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_age_relationships_l336_33601


namespace NUMINAMATH_CALUDE_power_of_two_equation_l336_33617

theorem power_of_two_equation (m : ℕ) : 
  2^2002 - 2^2000 - 2^1999 + 2^1998 = m * 2^1998 → m = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l336_33617


namespace NUMINAMATH_CALUDE_percentage_of_450_to_325x_l336_33673

theorem percentage_of_450_to_325x (x : ℝ) (h : x ≠ 0) :
  (450 / (325 * x)) * 100 = 138.46153846 / x := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_450_to_325x_l336_33673


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l336_33670

theorem quadratic_inequality_solution (c : ℝ) :
  (∀ x : ℝ, x^2 + 5*x - 2*c ≤ 0 ↔ -6 ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l336_33670


namespace NUMINAMATH_CALUDE_parallelogram_z_range_l336_33683

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (4, -2)

-- Define the function z
def z (x y : ℝ) : ℝ := 2*x - 5*y

-- Statement of the theorem
theorem parallelogram_z_range :
  ∀ (x y : ℝ), 
  (∃ (t₁ t₂ t₃ : ℝ), 0 ≤ t₁ ∧ 0 ≤ t₂ ∧ 0 ≤ t₃ ∧ t₁ + t₂ + t₃ ≤ 1 ∧
    (x, y) = t₁ • A + t₂ • B + t₃ • C + (1 - t₁ - t₂ - t₃) • (C + A - B)) →
  -14 ≤ z x y ∧ z x y ≤ 20 :=
by sorry


end NUMINAMATH_CALUDE_parallelogram_z_range_l336_33683


namespace NUMINAMATH_CALUDE_ternary_2101211_equals_octal_444_l336_33685

/-- Converts a ternary number represented as a list of digits to its decimal value. -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Converts a decimal number to its octal representation as a list of digits. -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Theorem stating that the ternary number 2101211 is equal to the octal number 444. -/
theorem ternary_2101211_equals_octal_444 :
  decimal_to_octal (ternary_to_decimal [1, 1, 2, 1, 0, 1, 2]) = [4, 4, 4] := by
  sorry

#eval ternary_to_decimal [1, 1, 2, 1, 0, 1, 2]
#eval decimal_to_octal (ternary_to_decimal [1, 1, 2, 1, 0, 1, 2])

end NUMINAMATH_CALUDE_ternary_2101211_equals_octal_444_l336_33685


namespace NUMINAMATH_CALUDE_steve_earnings_l336_33650

/-- Calculates an author's earnings from book sales after agent's commission --/
def author_earnings (copies_sold : ℕ) (price_per_copy : ℚ) (agent_commission_rate : ℚ) : ℚ :=
  let total_revenue := copies_sold * price_per_copy
  let agent_commission := total_revenue * agent_commission_rate
  total_revenue - agent_commission

/-- Proves that given the specified conditions, the author's earnings are $1,800,000 --/
theorem steve_earnings :
  author_earnings 1000000 2 (1/10) = 1800000 := by
  sorry

end NUMINAMATH_CALUDE_steve_earnings_l336_33650


namespace NUMINAMATH_CALUDE_weight_of_ton_l336_33633

/-- The weight of a ton in pounds -/
def ton_weight : ℝ := 2000

theorem weight_of_ton (elephant_weight : ℝ) (donkey_weight : ℝ) 
  (h1 : elephant_weight = 3 * ton_weight)
  (h2 : donkey_weight = 0.1 * elephant_weight)
  (h3 : elephant_weight + donkey_weight = 6600) :
  ton_weight = 2000 := by
  sorry

#check weight_of_ton

end NUMINAMATH_CALUDE_weight_of_ton_l336_33633


namespace NUMINAMATH_CALUDE_intersection_theorem_l336_33695

/-- The line x + y = k intersects the circle x^2 + y^2 = 4 at points A and B. -/
def intersectionPoints (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  (A.1 + A.2 = k) ∧ (B.1 + B.2 = k) ∧
  (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4)

/-- The length of AB equals the length of OA + OB, where O is the origin. -/
def lengthCondition (A B : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 + B.1)^2 + (A.2 + B.2)^2

/-- Main theorem: If the conditions are satisfied, then k = 2. -/
theorem intersection_theorem (k : ℝ) (A B : ℝ × ℝ) 
  (h1 : k > 0)
  (h2 : intersectionPoints k A B)
  (h3 : lengthCondition A B) : 
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l336_33695


namespace NUMINAMATH_CALUDE_range_of_f_triangle_properties_l336_33698

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x - 1/2

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

def triangle : Triangle where
  A := Real.pi / 3
  a := 2 * Real.sqrt 3
  b := 2
  c := 4

-- Theorem statements
theorem range_of_f : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc (-1/2) 1 := sorry

theorem triangle_properties (t : Triangle) (h1 : 0 < t.A) (h2 : t.A < Real.pi / 2) 
  (h3 : t.a = 2 * Real.sqrt 3) (h4 : t.c = 4) (h5 : f t.A = 1) : 
  t.A = Real.pi / 3 ∧ t.b = 2 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3) := sorry

end

end NUMINAMATH_CALUDE_range_of_f_triangle_properties_l336_33698


namespace NUMINAMATH_CALUDE_leahs_coins_value_l336_33616

theorem leahs_coins_value :
  ∀ (p n : ℕ),
  p + n = 18 →
  n + 2 = p →
  5 * n + p = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l336_33616


namespace NUMINAMATH_CALUDE_angle_measure_l336_33630

theorem angle_measure (x : Real) : 
  (0.4 * (180 - x) = 90 - x) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l336_33630


namespace NUMINAMATH_CALUDE_variance_is_five_ninths_l336_33627

/-- A random variable with a discrete distribution over {-1, 0, 1} -/
structure DiscreteRV where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_to_one : a + b + c = 1
  arithmetic_seq : 2 * b = a + c

/-- Expected value of the random variable -/
def expected_value (ξ : DiscreteRV) : ℝ := -1 * ξ.a + 1 * ξ.c

/-- Variance of the random variable -/
def variance (ξ : DiscreteRV) : ℝ :=
  (-1 - expected_value ξ)^2 * ξ.a +
  (0 - expected_value ξ)^2 * ξ.b +
  (1 - expected_value ξ)^2 * ξ.c

theorem variance_is_five_ninths (ξ : DiscreteRV) 
  (h : expected_value ξ = 1/3) : variance ξ = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_variance_is_five_ninths_l336_33627


namespace NUMINAMATH_CALUDE_prove_b_value_l336_33680

theorem prove_b_value (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 35 * b) : b = 63 := by
  sorry

end NUMINAMATH_CALUDE_prove_b_value_l336_33680


namespace NUMINAMATH_CALUDE_susanas_chocolate_chips_l336_33636

theorem susanas_chocolate_chips 
  (viviana_chocolate : ℕ) 
  (susana_chocolate : ℕ) 
  (viviana_vanilla : ℕ) 
  (susana_vanilla : ℕ) 
  (h1 : viviana_chocolate = susana_chocolate + 5)
  (h2 : susana_vanilla = (3 * viviana_vanilla) / 4)
  (h3 : viviana_vanilla = 20)
  (h4 : viviana_chocolate + susana_chocolate + viviana_vanilla + susana_vanilla = 90) :
  susana_chocolate = 25 := by
sorry

end NUMINAMATH_CALUDE_susanas_chocolate_chips_l336_33636


namespace NUMINAMATH_CALUDE_product_of_sums_and_differences_l336_33682

theorem product_of_sums_and_differences (P Q R S : ℝ) : P * Q * R * S = 1 :=
  by
  have h1 : P = Real.sqrt 2011 + Real.sqrt 2010 := by sorry
  have h2 : Q = -Real.sqrt 2011 - Real.sqrt 2010 := by sorry
  have h3 : R = Real.sqrt 2011 - Real.sqrt 2010 := by sorry
  have h4 : S = Real.sqrt 2010 - Real.sqrt 2011 := by sorry
  sorry

#check product_of_sums_and_differences

end NUMINAMATH_CALUDE_product_of_sums_and_differences_l336_33682


namespace NUMINAMATH_CALUDE_g_composition_half_l336_33681

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

-- State the theorem
theorem g_composition_half : g (g (1/2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_half_l336_33681


namespace NUMINAMATH_CALUDE_derivative_of_f_l336_33652

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log 2

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2^x * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l336_33652


namespace NUMINAMATH_CALUDE_allie_billie_meeting_l336_33662

/-- The distance Allie skates before meeting Billie -/
def allie_distance (ab_distance : ℝ) (allie_speed billie_speed : ℝ) (allie_angle : ℝ) : ℝ :=
  let x := 160
  x

theorem allie_billie_meeting 
  (ab_distance : ℝ) 
  (allie_speed billie_speed : ℝ) 
  (allie_angle : ℝ) 
  (h1 : ab_distance = 100)
  (h2 : allie_speed = 8)
  (h3 : billie_speed = 7)
  (h4 : allie_angle = 60 * π / 180)
  (h5 : ∀ x, x > 0 → x ≠ 160 → 
    (x / allie_speed ≠ 
    Real.sqrt (x^2 + ab_distance^2 - 2 * x * ab_distance * Real.cos allie_angle) / billie_speed)) :
  allie_distance ab_distance allie_speed billie_speed allie_angle = 160 := by
  sorry

end NUMINAMATH_CALUDE_allie_billie_meeting_l336_33662


namespace NUMINAMATH_CALUDE_number_square_relationship_l336_33646

theorem number_square_relationship (n : ℕ) (h : n = 14) : n + n^2 = 210 := by
  sorry

end NUMINAMATH_CALUDE_number_square_relationship_l336_33646


namespace NUMINAMATH_CALUDE_gummy_bear_cost_l336_33638

theorem gummy_bear_cost
  (total_cost : ℝ)
  (chocolate_chip_cost : ℝ)
  (chocolate_bar_cost : ℝ)
  (num_chocolate_bars : ℕ)
  (num_gummy_bears : ℕ)
  (num_chocolate_chips : ℕ)
  (h1 : total_cost = 150)
  (h2 : chocolate_chip_cost = 5)
  (h3 : chocolate_bar_cost = 3)
  (h4 : num_chocolate_bars = 10)
  (h5 : num_gummy_bears = 10)
  (h6 : num_chocolate_chips = 20)
  : (total_cost - num_chocolate_bars * chocolate_bar_cost - num_chocolate_chips * chocolate_chip_cost) / num_gummy_bears = 2 := by
  sorry

end NUMINAMATH_CALUDE_gummy_bear_cost_l336_33638


namespace NUMINAMATH_CALUDE_no_valid_coloring_l336_33656

theorem no_valid_coloring :
  ¬ ∃ (f : ℕ+ → Fin 3),
    (∀ c : Fin 3, ∃ n : ℕ+, f n = c) ∧
    (∀ a b : ℕ+, f a ≠ f b → f (a * b) ≠ f a ∧ f (a * b) ≠ f b) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l336_33656


namespace NUMINAMATH_CALUDE_marks_age_difference_l336_33687

theorem marks_age_difference (mark_current_age : ℕ) (aaron_current_age : ℕ) : 
  mark_current_age = 28 →
  mark_current_age + 4 = 2 * (aaron_current_age + 4) + 2 →
  (mark_current_age - 3) - 3 * (aaron_current_age - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_marks_age_difference_l336_33687


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l336_33647

/-- Represents the gain percentage of a trader selling pens -/
def gain_percentage (num_sold : ℕ) (num_gain : ℕ) : ℚ :=
  (num_gain : ℚ) / (num_sold : ℚ) * 100

/-- Theorem stating that selling 90 pens and gaining the cost of 30 pens results in a 33.33% gain -/
theorem trader_gain_percentage : 
  gain_percentage 90 30 = 33.33 := by sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l336_33647
