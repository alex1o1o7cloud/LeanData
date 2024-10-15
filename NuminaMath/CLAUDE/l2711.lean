import Mathlib

namespace NUMINAMATH_CALUDE_phillips_remaining_money_l2711_271106

/-- Calculates the remaining money after Phillip's shopping trip --/
def remaining_money (initial_amount : ℚ) 
  (orange_price : ℚ) (orange_quantity : ℚ)
  (apple_price : ℚ) (apple_quantity : ℚ)
  (candy_price : ℚ)
  (egg_price : ℚ) (egg_quantity : ℚ)
  (milk_price : ℚ)
  (sales_tax_rate : ℚ)
  (apple_discount_rate : ℚ) : ℚ :=
  sorry

/-- Theorem stating that Phillip's remaining money is $51.91 --/
theorem phillips_remaining_money :
  remaining_money 95 3 2 3.5 4 6 6 2 4 0.08 0.15 = 51.91 :=
  sorry

end NUMINAMATH_CALUDE_phillips_remaining_money_l2711_271106


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2711_271122

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Theorem statement
theorem arithmetic_sequence_difference
  (a : ℕ → ℝ) (d : ℝ) (h : isArithmeticSequence a d) (h_d : d = 2) :
  a 5 - a 2 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2711_271122


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l2711_271169

/-- Proves that in a 60-litre mixture, if adding 60 litres of water changes
    the milk-to-water ratio to 1:2, then the initial milk-to-water ratio was 2:1 -/
theorem initial_milk_water_ratio (m w : ℝ) : 
  m + w = 60 →  -- Total initial volume is 60 litres
  2 * m = w + 60 →  -- After adding 60 litres of water, milk:water = 1:2
  m / w = 2 / 1 :=  -- Initial ratio of milk to water is 2:1
by sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l2711_271169


namespace NUMINAMATH_CALUDE_factorization_equality_l2711_271141

theorem factorization_equality (x y : ℝ) : 25*x - x*y^2 = x*(5+y)*(5-y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2711_271141


namespace NUMINAMATH_CALUDE_intersection_circles_power_l2711_271138

/-- Given two circles centered on the x-axis that intersect at points M(3a-b, 5) and N(9, 2a+3b), prove that a^b = 1/8 -/
theorem intersection_circles_power (a b : ℝ) : 
  (3 * a - b = 9) → (2 * a + 3 * b = -5) → a^b = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_circles_power_l2711_271138


namespace NUMINAMATH_CALUDE_reading_time_difference_l2711_271176

-- Define the reading speeds and book length
def xanthia_speed : ℝ := 150  -- pages per hour
def molly_speed : ℝ := 75     -- pages per hour
def book_length : ℝ := 300    -- pages

-- Define the time difference in minutes
def time_difference : ℝ := 120 -- minutes

-- Theorem statement
theorem reading_time_difference :
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = time_difference := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l2711_271176


namespace NUMINAMATH_CALUDE_classmate_heights_most_suitable_l2711_271121

-- Define the survey options
inductive SurveyOption
  | LightBulbLifespan
  | WaterQualityGanRiver
  | TVProgramViewership
  | ClassmateHeights

-- Define the characteristic of being suitable for a comprehensive survey
def SuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.ClassmateHeights => True
  | _ => False

-- Theorem statement
theorem classmate_heights_most_suitable :
  SuitableForComprehensiveSurvey SurveyOption.ClassmateHeights ∧
  (∀ option : SurveyOption, option ≠ SurveyOption.ClassmateHeights →
    ¬SuitableForComprehensiveSurvey option) :=
by sorry

end NUMINAMATH_CALUDE_classmate_heights_most_suitable_l2711_271121


namespace NUMINAMATH_CALUDE_power_of_power_l2711_271185

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2711_271185


namespace NUMINAMATH_CALUDE_cos_450_degrees_l2711_271152

theorem cos_450_degrees : Real.cos (450 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_450_degrees_l2711_271152


namespace NUMINAMATH_CALUDE_express_x_in_terms_of_y_l2711_271110

theorem express_x_in_terms_of_y (x y : ℝ) (h : 3 * x - 4 * y = 8) :
  x = (4 * y + 8) / 3 := by
  sorry

end NUMINAMATH_CALUDE_express_x_in_terms_of_y_l2711_271110


namespace NUMINAMATH_CALUDE_smallest_prime_2018_factorial_l2711_271133

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem smallest_prime_2018_factorial :
  ∃ p : ℕ, 
    Prime p ∧ 
    p = 509 ∧ 
    is_divisible (factorial 2018) (p^3) ∧ 
    ¬is_divisible (factorial 2018) (p^4) ∧
    ∀ q : ℕ, Prime q → q < p → 
      ¬(is_divisible (factorial 2018) (q^3) ∧ ¬is_divisible (factorial 2018) (q^4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_2018_factorial_l2711_271133


namespace NUMINAMATH_CALUDE_circle_center_satisfies_conditions_l2711_271186

/-- The center of a circle satisfying given conditions -/
theorem circle_center_satisfies_conditions :
  let center : ℝ × ℝ := (-18, -11)
  let line1 : ℝ → ℝ → ℝ := λ x y => 3 * x - 4 * y - 20
  let line2 : ℝ → ℝ → ℝ := λ x y => 3 * x - 4 * y + 40
  let midline : ℝ → ℝ → ℝ := λ x y => 3 * x - 4 * y + 10
  let line3 : ℝ → ℝ → ℝ := λ x y => x - 3 * y - 15
  (midline center.1 center.2 = 0) ∧ (line3 center.1 center.2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_satisfies_conditions_l2711_271186


namespace NUMINAMATH_CALUDE_monthly_fee_calculation_l2711_271174

def cost_per_minute : ℚ := 25 / 100

def total_bill : ℚ := 1202 / 100

def minutes_used : ℚ := 2808 / 100

theorem monthly_fee_calculation :
  ∃ (monthly_fee : ℚ),
    monthly_fee + cost_per_minute * minutes_used = total_bill ∧
    monthly_fee = 5 := by
  sorry

end NUMINAMATH_CALUDE_monthly_fee_calculation_l2711_271174


namespace NUMINAMATH_CALUDE_union_of_given_sets_l2711_271145

theorem union_of_given_sets :
  let A : Set ℕ := {0, 1}
  let B : Set ℕ := {1, 2}
  A ∪ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_given_sets_l2711_271145


namespace NUMINAMATH_CALUDE_convex_polyhedron_has_32_faces_l2711_271144

/-- A convex polyhedron with pentagonal and hexagonal faces -/
structure ConvexPolyhedron where
  /-- Number of pentagonal faces -/
  pentagonFaces : ℕ
  /-- Number of hexagonal faces -/
  hexagonFaces : ℕ
  /-- Three faces meet at each vertex -/
  threeAtVertex : True
  /-- Each pentagon shares edges with 5 hexagons -/
  pentagonSharing : pentagonFaces * 5 = hexagonFaces * 3
  /-- Each hexagon shares edges with 3 pentagons -/
  hexagonSharing : hexagonFaces * 3 = pentagonFaces * 5

/-- The total number of faces in the polyhedron -/
def ConvexPolyhedron.totalFaces (p : ConvexPolyhedron) : ℕ :=
  p.pentagonFaces + p.hexagonFaces

/-- Theorem: The convex polyhedron has exactly 32 faces -/
theorem convex_polyhedron_has_32_faces (p : ConvexPolyhedron) :
  p.totalFaces = 32 := by
  sorry

#eval ConvexPolyhedron.totalFaces ⟨12, 20, trivial, rfl, rfl⟩

end NUMINAMATH_CALUDE_convex_polyhedron_has_32_faces_l2711_271144


namespace NUMINAMATH_CALUDE_csc_135_deg_l2711_271161

-- Define the cosecant function
noncomputable def csc (θ : Real) : Real := 1 / Real.sin θ

-- State the theorem
theorem csc_135_deg : csc (135 * π / 180) = Real.sqrt 2 := by
  -- Define the given conditions
  have sin_135 : Real.sin (135 * π / 180) = 1 / Real.sqrt 2 := by sorry
  have cos_135 : Real.cos (135 * π / 180) = -(1 / Real.sqrt 2) := by sorry

  -- Prove the theorem
  sorry

end NUMINAMATH_CALUDE_csc_135_deg_l2711_271161


namespace NUMINAMATH_CALUDE_lineup_ways_proof_l2711_271167

/-- The number of ways to arrange 5 people in a line with restrictions -/
def lineupWays : ℕ := 72

/-- The number of people in the line -/
def totalPeople : ℕ := 5

/-- The number of positions where the youngest person can be placed -/
def youngestPositions : ℕ := 3

/-- The number of choices for the first position -/
def firstPositionChoices : ℕ := 4

/-- The number of ways to arrange the remaining people after placing the youngest -/
def remainingArrangements : ℕ := 6

theorem lineup_ways_proof :
  lineupWays = firstPositionChoices * youngestPositions * remainingArrangements :=
sorry

end NUMINAMATH_CALUDE_lineup_ways_proof_l2711_271167


namespace NUMINAMATH_CALUDE_homework_selection_is_systematic_l2711_271112

/-- Represents a sampling method --/
inductive SamplingMethod
  | Stratified
  | Lottery
  | Random
  | Systematic

/-- Represents a school's homework selection process --/
structure HomeworkSelection where
  selectFromEachClass : Bool
  selectionCriteria : Nat → Bool
  studentsArranged : Bool
  largeStudentPopulation : Bool

/-- Determines the sampling method based on the selection process --/
def determineSamplingMethod (selection : HomeworkSelection) : SamplingMethod :=
  sorry

/-- Theorem stating that the given selection process is Systematic Sampling --/
theorem homework_selection_is_systematic 
  (selection : HomeworkSelection)
  (h1 : selection.selectFromEachClass = true)
  (h2 : selection.selectionCriteria = λ id => id % 10 = 5)
  (h3 : selection.studentsArranged = true)
  (h4 : selection.largeStudentPopulation = true) :
  determineSamplingMethod selection = SamplingMethod.Systematic :=
sorry

end NUMINAMATH_CALUDE_homework_selection_is_systematic_l2711_271112


namespace NUMINAMATH_CALUDE_square_measurement_error_l2711_271172

theorem square_measurement_error (S : ℝ) (S' : ℝ) (h : S > 0) :
  S'^2 = S^2 * (1 + 0.0404) → (S' - S) / S * 100 = 2 := by sorry

end NUMINAMATH_CALUDE_square_measurement_error_l2711_271172


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2711_271105

theorem quadratic_factorization (x : ℝ) : 
  (x^2 - 6*x - 11 = 0) ↔ ((x - 3)^2 = 20) := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2711_271105


namespace NUMINAMATH_CALUDE_cookie_count_equivalence_l2711_271155

/-- Represents the shape of a cookie -/
inductive CookieShape
  | Circle
  | Rectangle
  | Parallelogram
  | Triangle
  | Square

/-- Represents a friend who bakes cookies -/
structure Friend where
  name : String
  shape : CookieShape

/-- Represents the dimensions of a cookie -/
structure CookieDimensions where
  base : ℝ
  height : ℝ

theorem cookie_count_equivalence 
  (friends : List Friend)
  (carlos_dims : CookieDimensions)
  (lisa_side : ℝ)
  (carlos_count : ℕ)
  (h1 : friends.length = 5)
  (h2 : ∃ f ∈ friends, f.name = "Carlos" ∧ f.shape = CookieShape.Triangle)
  (h3 : ∃ f ∈ friends, f.name = "Lisa" ∧ f.shape = CookieShape.Square)
  (h4 : carlos_dims.base = 4)
  (h5 : carlos_dims.height = 5)
  (h6 : carlos_count = 20)
  (h7 : lisa_side = 5)
  : (200 : ℝ) / (lisa_side ^ 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_equivalence_l2711_271155


namespace NUMINAMATH_CALUDE_cube_surface_area_l2711_271165

/-- Given a cube made up of 6 squares, each with a perimeter of 24 cm,
    prove that its surface area is 216 cm². -/
theorem cube_surface_area (cube_side_length : ℝ) (square_perimeter : ℝ) : 
  square_perimeter = 24 →
  cube_side_length = square_perimeter / 4 →
  6 * cube_side_length ^ 2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2711_271165


namespace NUMINAMATH_CALUDE_f_behavior_at_infinity_l2711_271123

def f (x : ℝ) := -3 * x^3 + 4 * x^2 + 1

theorem f_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → f x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → f x > M) :=
by sorry

end NUMINAMATH_CALUDE_f_behavior_at_infinity_l2711_271123


namespace NUMINAMATH_CALUDE_mice_breeding_experiment_l2711_271168

/-- Calculates the final number of mice after two generations -/
def final_mice_count (initial_mice : ℕ) (pups_per_mouse : ℕ) (pups_eaten : ℕ) : ℕ :=
  let first_gen_pups := initial_mice * pups_per_mouse
  let total_after_first_gen := initial_mice + first_gen_pups
  let surviving_pups_per_mouse := pups_per_mouse - pups_eaten
  let second_gen_pups := total_after_first_gen * surviving_pups_per_mouse
  total_after_first_gen + second_gen_pups

/-- Theorem stating that the final number of mice is 280 given the initial conditions -/
theorem mice_breeding_experiment :
  final_mice_count 8 6 2 = 280 := by
  sorry

end NUMINAMATH_CALUDE_mice_breeding_experiment_l2711_271168


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l2711_271163

theorem multiplication_addition_equality : 24 * 44 + 56 * 24 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l2711_271163


namespace NUMINAMATH_CALUDE_science_tech_group_size_l2711_271125

theorem science_tech_group_size :
  ∀ (girls boys : ℕ),
  girls = 18 →
  girls = 2 * boys - 2 →
  girls + boys = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_science_tech_group_size_l2711_271125


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2711_271199

theorem perfect_square_condition (a b k : ℝ) :
  (∃ (n : ℝ), a^2 + k*a*b + 9*b^2 = n^2) → (k = 6 ∨ k = -6) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2711_271199


namespace NUMINAMATH_CALUDE_f_monotone_increasing_implies_a_bound_l2711_271190

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 2 * Real.sin x * Real.cos x + a * Real.cos x

def monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem f_monotone_increasing_implies_a_bound :
  ∀ a : ℝ, monotone_increasing (f a) (π/4) (3*π/4) → a ≤ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_implies_a_bound_l2711_271190


namespace NUMINAMATH_CALUDE_smallest_x_value_l2711_271127

theorem smallest_x_value (x y : ℕ+) (h : (0.8 : ℚ) = y / (186 + x)) : 
  x ≥ 4 ∧ ∃ (y' : ℕ+), (0.8 : ℚ) = y' / (186 + 4) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2711_271127


namespace NUMINAMATH_CALUDE_angle_D_value_l2711_271189

-- Define the angles as real numbers
variable (A B C D F : ℝ)

-- State the theorem
theorem angle_D_value (h1 : A + B = 180)
                      (h2 : C = D)
                      (h3 : B = 90)
                      (h4 : F = 50)
                      (h5 : A + C + F = 180) : D = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_value_l2711_271189


namespace NUMINAMATH_CALUDE_mary_james_seating_probability_l2711_271114

/-- The number of chairs in the row -/
def total_chairs : ℕ := 10

/-- The number of chairs Mary can choose from -/
def mary_choices : ℕ := 9

/-- The number of chairs James can choose from -/
def james_choices : ℕ := 10

/-- The probability that Mary and James do not sit next to each other -/
def prob_not_adjacent : ℚ := 8/9

theorem mary_james_seating_probability :
  prob_not_adjacent = 1 - (mary_choices.pred / (mary_choices * james_choices)) :=
by sorry

end NUMINAMATH_CALUDE_mary_james_seating_probability_l2711_271114


namespace NUMINAMATH_CALUDE_job_age_is_five_l2711_271160

def freddy_age : ℕ := 18
def stephanie_age : ℕ := freddy_age + 2

theorem job_age_is_five :
  ∃ (job_age : ℕ), stephanie_age = 4 * job_age ∧ job_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_job_age_is_five_l2711_271160


namespace NUMINAMATH_CALUDE_sum_of_digits_twice_square_222222222_l2711_271128

def n : ℕ := 9

def Y : ℕ := 2 * (10^n - 1) / 9

def Y_squared : ℕ := Y * Y

def doubled_Y_squared : ℕ := 2 * Y_squared

def sum_of_digits (x : ℕ) : ℕ :=
  if x < 10 then x else x % 10 + sum_of_digits (x / 10)

theorem sum_of_digits_twice_square_222222222 :
  sum_of_digits doubled_Y_squared = 126 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_twice_square_222222222_l2711_271128


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_three_l2711_271171

theorem trigonometric_expression_equals_three (α : ℝ) 
  (h : Real.tan (3 * Real.pi + α) = 3) : 
  (Real.sin (α - 3 * Real.pi) + Real.cos (Real.pi - α) + 
   Real.sin (Real.pi / 2 - α) - 2 * Real.cos (Real.pi / 2 + α)) / 
  (-Real.sin (-α) + Real.cos (Real.pi + α)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_three_l2711_271171


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l2711_271101

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 + 4*x - 4 = 0
def equation2 (x : ℝ) : Prop := (x - 1)^2 = 2*(x - 1)

-- Theorem for the first equation
theorem solutions_equation1 :
  ∃ (x1 x2 : ℝ), 
    equation1 x1 ∧ equation1 x2 ∧ 
    x1 = -2 + 2 * Real.sqrt 2 ∧ 
    x2 = -2 - 2 * Real.sqrt 2 :=
sorry

-- Theorem for the second equation
theorem solutions_equation2 :
  ∃ (x1 x2 : ℝ), 
    equation2 x1 ∧ equation2 x2 ∧ 
    x1 = 1 ∧ x2 = 3 :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l2711_271101


namespace NUMINAMATH_CALUDE_cosine_angle_equality_l2711_271149

theorem cosine_angle_equality (n : ℤ) : 
  (0 ≤ n ∧ n ≤ 180) ∧ (Real.cos (n * π / 180) = Real.cos (1124 * π / 180)) ↔ n = 44 := by
  sorry

end NUMINAMATH_CALUDE_cosine_angle_equality_l2711_271149


namespace NUMINAMATH_CALUDE_exists_player_in_interval_l2711_271157

/-- Represents a round-robin tournament with 2n+1 players -/
structure Tournament (n : ℕ) where
  /-- The number of matches where the weaker player wins -/
  k : ℕ
  /-- The strength of each player, which are all different -/
  strength : Fin (2*n+1) → ℕ
  strength_injective : Function.Injective strength
  /-- The result of each match, where true means the first player won -/
  result : Fin (2*n+1) → Fin (2*n+1) → Bool
  /-- Each player plays exactly one match against every other player -/
  played_all : ∀ i j, i ≠ j → (result i j = true ∧ result j i = false) ∨ (result i j = false ∧ result j i = true)
  /-- Exactly k matches are won by the weaker player -/
  weaker_wins : (Finset.univ.filter (λ (p : Fin (2*n+1) × Fin (2*n+1)) => 
    p.1 ≠ p.2 ∧ strength p.1 < strength p.2 ∧ result p.1 p.2 = true)).card = k

/-- The number of victories for a player -/
def victories (t : Tournament n) (i : Fin (2*n+1)) : ℕ :=
  (Finset.univ.filter (λ j => j ≠ i ∧ t.result i j = true)).card

/-- The main theorem -/
theorem exists_player_in_interval (n : ℕ) (t : Tournament n) :
  ∃ i : Fin (2*n+1), n - Real.sqrt (2 * t.k) ≤ victories t i ∧ victories t i ≤ n + Real.sqrt (2 * t.k) := by
  sorry

end NUMINAMATH_CALUDE_exists_player_in_interval_l2711_271157


namespace NUMINAMATH_CALUDE_fraction_simplification_l2711_271148

theorem fraction_simplification (x : ℝ) (h : x = 7) : 
  (x^6 - 36*x^3 + 324) / (x^3 - 18) = 325 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2711_271148


namespace NUMINAMATH_CALUDE_x_range_lower_bound_l2711_271192

theorem x_range_lower_bound (x y : ℝ) (h : x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0) :
  x ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_x_range_lower_bound_l2711_271192


namespace NUMINAMATH_CALUDE_shaded_area_is_30_l2711_271164

/-- An isosceles right triangle with legs of length 10 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  is_leg_length_10 : leg_length = 10

/-- A partition of the triangle into 25 congruent smaller triangles -/
structure Partition (t : IsoscelesRightTriangle) where
  num_small_triangles : ℕ
  is_25_triangles : num_small_triangles = 25

/-- The shaded region covering 15 of the smaller triangles -/
structure ShadedRegion (p : Partition t) where
  num_shaded_triangles : ℕ
  is_15_triangles : num_shaded_triangles = 15

/-- The theorem stating that the area of the shaded region is 30 -/
theorem shaded_area_is_30 (t : IsoscelesRightTriangle) (p : Partition t) (s : ShadedRegion p) :
  (t.leg_length ^ 2 / 2) * (s.num_shaded_triangles / p.num_small_triangles) = 30 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_is_30_l2711_271164


namespace NUMINAMATH_CALUDE_day_crew_load_fraction_l2711_271158

/-- Fraction of boxes loaded by day crew given night crew conditions -/
theorem day_crew_load_fraction (D W : ℚ) : 
  D > 0 → W > 0 →
  (D * W) / ((D * W) + ((3/4 * D) * (4/7 * W))) = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_day_crew_load_fraction_l2711_271158


namespace NUMINAMATH_CALUDE_arcade_change_machine_l2711_271104

theorem arcade_change_machine (total_bills : ℕ) (one_dollar_bills : ℕ) : 
  total_bills = 200 → one_dollar_bills = 175 → 
  (total_bills - one_dollar_bills) * 5 + one_dollar_bills = 300 := by
  sorry

end NUMINAMATH_CALUDE_arcade_change_machine_l2711_271104


namespace NUMINAMATH_CALUDE_complex_addition_l2711_271109

theorem complex_addition (z₁ z₂ : ℂ) (h₁ : z₁ = 1 + 2*I) (h₂ : z₂ = 3 + 4*I) : 
  z₁ + z₂ = 4 + 6*I := by
sorry

end NUMINAMATH_CALUDE_complex_addition_l2711_271109


namespace NUMINAMATH_CALUDE_complement_implies_sum_l2711_271175

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A (m : ℝ) : Set ℝ := {x : ℝ | (x - 1) * (x - m) > 0}

-- Define the complement of A in U
def C_UA (m n : ℝ) : Set ℝ := Set.Icc (-1) (-n)

-- Theorem statement
theorem complement_implies_sum (m n : ℝ) : 
  C_UA m n = Set.compl (A m) → m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_complement_implies_sum_l2711_271175


namespace NUMINAMATH_CALUDE_fiscal_revenue_scientific_notation_l2711_271115

/-- Converts a number in billions to scientific notation -/
def to_scientific_notation (x : ℝ) : ℝ × ℤ :=
  (1.14, 9)

/-- The fiscal revenue in billions -/
def fiscal_revenue : ℝ := 1.14

theorem fiscal_revenue_scientific_notation :
  to_scientific_notation fiscal_revenue = (1.14, 9) := by
  sorry

end NUMINAMATH_CALUDE_fiscal_revenue_scientific_notation_l2711_271115


namespace NUMINAMATH_CALUDE_smallest_n_value_l2711_271102

/-- Counts the number of factors of 5 in k! -/
def count_factors_of_5 (k : ℕ) : ℕ := sorry

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 3000 →
  a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  ¬(10 ∣ m) →
  (∀ n' : ℕ, n' < n → ∃ m' : ℕ, a.factorial * b.factorial * c.factorial ≠ m' * (10 ^ n')) →
  n = 747 := by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l2711_271102


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2711_271130

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2711_271130


namespace NUMINAMATH_CALUDE_power_mod_seven_l2711_271120

theorem power_mod_seven : 3^2023 % 7 = 3 := by sorry

end NUMINAMATH_CALUDE_power_mod_seven_l2711_271120


namespace NUMINAMATH_CALUDE_sin_x_minus_pi_third_l2711_271142

theorem sin_x_minus_pi_third (x : ℝ) (h : Real.cos (x + π / 6) = 1 / 3) :
  Real.sin (x - π / 3) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_x_minus_pi_third_l2711_271142


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2711_271178

/-- A function satisfying the given functional equation. -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((x + y) / 2) = (f x + f y) / 2

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = ax + b for some constants a and b. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2711_271178


namespace NUMINAMATH_CALUDE_max_value_expression_l2711_271177

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hbc : b > c) (hca : c > a) (ha_neq_zero : a ≠ 0) : 
  ∃ (x : ℝ), x = ((2*a + b)^2 + (b - 2*c)^2 + (c - a)^2) / a^2 ∧ 
  x ≤ 44 ∧ 
  ∀ (y : ℝ), y = ((2*a + b)^2 + (b - 2*c)^2 + (c - a)^2) / a^2 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2711_271177


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2711_271137

theorem cyclic_inequality (x y z p q : ℝ) 
  (h1 : y = x^2 + p*x + q)
  (h2 : z = y^2 + p*y + q)
  (h3 : x = z^2 + p*z + q) :
  x^2*y + y^2*z + z^2*x ≥ x^2*z + y^2*x + z^2*y := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2711_271137


namespace NUMINAMATH_CALUDE_watch_payment_in_dimes_l2711_271107

/-- The number of dimes in one dollar -/
def dimes_per_dollar : ℕ := 10

/-- The cost of the watch in dollars -/
def watch_cost : ℕ := 5

/-- Theorem: If a watch costs 5 dollars and is paid for entirely in dimes, 
    the number of dimes used is 50. -/
theorem watch_payment_in_dimes : 
  watch_cost * dimes_per_dollar = 50 := by sorry

end NUMINAMATH_CALUDE_watch_payment_in_dimes_l2711_271107


namespace NUMINAMATH_CALUDE_truck_load_after_deliveries_l2711_271147

theorem truck_load_after_deliveries :
  let initial_load : ℝ := 50000
  let first_unload_percentage : ℝ := 0.1
  let second_unload_percentage : ℝ := 0.2
  let after_first_delivery := initial_load * (1 - first_unload_percentage)
  let final_load := after_first_delivery * (1 - second_unload_percentage)
  final_load = 36000 := by sorry

end NUMINAMATH_CALUDE_truck_load_after_deliveries_l2711_271147


namespace NUMINAMATH_CALUDE_james_tennis_balls_l2711_271139

theorem james_tennis_balls (total_containers : Nat) (balls_per_container : Nat) : 
  total_containers = 5 → 
  balls_per_container = 10 → 
  2 * (total_containers * balls_per_container) = 100 := by
sorry

end NUMINAMATH_CALUDE_james_tennis_balls_l2711_271139


namespace NUMINAMATH_CALUDE_remainder_17_pow_2047_mod_23_l2711_271126

theorem remainder_17_pow_2047_mod_23 :
  (17 : ℤ) ^ 2047 % 23 = 11 := by sorry

end NUMINAMATH_CALUDE_remainder_17_pow_2047_mod_23_l2711_271126


namespace NUMINAMATH_CALUDE_multiplication_value_proof_l2711_271150

theorem multiplication_value_proof (x : ℚ) : (3 / 4) * x = 9 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_value_proof_l2711_271150


namespace NUMINAMATH_CALUDE_at_least_two_positive_roots_l2711_271183

def f (x : ℝ) : ℝ := x^11 + 8*x^10 + 15*x^9 - 1729*x^8 + 1379*x^7 - 172*x^6

theorem at_least_two_positive_roots :
  ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a ≠ b ∧ f a = 0 ∧ f b = 0 :=
sorry

end NUMINAMATH_CALUDE_at_least_two_positive_roots_l2711_271183


namespace NUMINAMATH_CALUDE_prime_difference_product_l2711_271173

theorem prime_difference_product (a b : ℕ) : 
  Nat.Prime a → Nat.Prime b → a - b = 35 → a * b = 74 := by
  sorry

end NUMINAMATH_CALUDE_prime_difference_product_l2711_271173


namespace NUMINAMATH_CALUDE_petya_more_likely_to_win_petya_wins_in_game_l2711_271188

/-- Represents a game between Petya and Vasya with two boxes of candies. -/
structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

/-- Defines the game setup with the given conditions. -/
def game : CandyGame :=
  { total_candies := 25,
    prob_two_caramels := 0.54 }

/-- Calculates the probability of Vasya winning (getting two chocolate candies). -/
def prob_vasya_wins (g : CandyGame) : ℝ :=
  1 - g.prob_two_caramels

/-- Theorem stating that Petya has a higher chance of winning than Vasya. -/
theorem petya_more_likely_to_win (g : CandyGame) :
  prob_vasya_wins g < 1 - prob_vasya_wins g :=
by sorry

/-- Corollary proving that Petya has a higher chance of winning in the specific game setup. -/
theorem petya_wins_in_game : prob_vasya_wins game < 1 - prob_vasya_wins game :=
by sorry

end NUMINAMATH_CALUDE_petya_more_likely_to_win_petya_wins_in_game_l2711_271188


namespace NUMINAMATH_CALUDE_min_values_theorem_l2711_271198

/-- Given positive real numbers a and b satisfying 4a + b = ab, 
    prove the following statements about their minimum values. -/
theorem min_values_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4*a + b = a*b) :
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4*a₀ + b₀ = a₀*b₀ ∧ ∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → a₀*b₀ ≤ a'*b') ∧
  (∃ (a₁ b₁ : ℝ), a₁ > 0 ∧ b₁ > 0 ∧ 4*a₁ + b₁ = a₁*b₁ ∧ ∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → a₁ + b₁ ≤ a' + b') ∧
  (∃ (a₂ b₂ : ℝ), a₂ > 0 ∧ b₂ > 0 ∧ 4*a₂ + b₂ = a₂*b₂ ∧ ∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → 1/a₂^2 + 4/b₂^2 ≤ 1/a'^2 + 4/b'^2) ∧
  (∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → a'*b' ≥ 16) ∧
  (∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → a' + b' ≥ 9) ∧
  (∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → 1/a'^2 + 4/b'^2 ≥ 1/5) :=
by sorry

end NUMINAMATH_CALUDE_min_values_theorem_l2711_271198


namespace NUMINAMATH_CALUDE_no_solution_for_fermat_like_equation_l2711_271140

theorem no_solution_for_fermat_like_equation :
  ∀ (x y z k : ℕ), x < k → y < k → x^k + y^k ≠ z^k := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_fermat_like_equation_l2711_271140


namespace NUMINAMATH_CALUDE_rain_probabilities_l2711_271103

theorem rain_probabilities (p_monday p_tuesday : ℝ) 
  (h_monday : p_monday = 0.4)
  (h_tuesday : p_tuesday = 0.3)
  (h_independent : True)  -- This represents the independence assumption
  : (p_monday * p_tuesday = 0.12) ∧ 
    ((1 - p_monday) * (1 - p_tuesday) = 0.42) := by
  sorry

end NUMINAMATH_CALUDE_rain_probabilities_l2711_271103


namespace NUMINAMATH_CALUDE_max_value_abc_expression_l2711_271154

theorem max_value_abc_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≥ b * c^2) (hbc : b ≥ c * a^2) (hca : c ≥ a * b^2) :
  a * b * c * (a - b * c^2) * (b - c * a^2) * (c - a * b^2) ≤ 0 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    a₀ ≥ b₀ * c₀^2 ∧ b₀ ≥ c₀ * a₀^2 ∧ c₀ ≥ a₀ * b₀^2 ∧
    a₀ * b₀ * c₀ * (a₀ - b₀ * c₀^2) * (b₀ - c₀ * a₀^2) * (c₀ - a₀ * b₀^2) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_abc_expression_l2711_271154


namespace NUMINAMATH_CALUDE_nested_square_root_evaluation_l2711_271136

theorem nested_square_root_evaluation (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x + Real.sqrt (x + Real.sqrt x)) = Real.sqrt (x + Real.sqrt (x + x^(1/2))) := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_evaluation_l2711_271136


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2711_271131

theorem trigonometric_inequality (A B C : Real) (h : A + B + C = Real.pi) :
  (Real.cos (A - B))^2 + (Real.cos (B - C))^2 + (Real.cos (C - A))^2 ≥ 
  24 * Real.cos A * Real.cos B * Real.cos C :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l2711_271131


namespace NUMINAMATH_CALUDE_robin_hair_growth_l2711_271179

/-- Calculates hair growth given initial length, cut length, and final length -/
def hair_growth (initial_length cut_length final_length : ℕ) : ℕ :=
  final_length - (initial_length - cut_length)

/-- Theorem: Given the problem conditions, hair growth is 12 inches -/
theorem robin_hair_growth :
  hair_growth 16 11 17 = 12 := by sorry

end NUMINAMATH_CALUDE_robin_hair_growth_l2711_271179


namespace NUMINAMATH_CALUDE_tangent_point_determines_b_l2711_271191

-- Define the curve and line
def curve (x a b : ℝ) : ℝ := x^3 + a*x + b
def line (x k : ℝ) : ℝ := k*x + 1

-- Define the tangent condition
def is_tangent (a b k : ℝ) : Prop :=
  ∃ x, curve x a b = line x k ∧ 
       (deriv (fun x => curve x a b)) x = k

theorem tangent_point_determines_b :
  ∀ a b k : ℝ, 
    is_tangent a b k →  -- The line is tangent to the curve
    curve 1 a b = 3 →   -- The point of tangency is (1, 3)
    b = 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_determines_b_l2711_271191


namespace NUMINAMATH_CALUDE_intersection_implies_m_and_n_l2711_271113

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x + 2| < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - m) * (x - 2) < 0}

-- State the theorem
theorem intersection_implies_m_and_n (m n : ℝ) :
  A ∩ B m = Set.Ioo (-1) n → m = -1 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_and_n_l2711_271113


namespace NUMINAMATH_CALUDE_fast_reader_time_l2711_271100

/-- Given two people, where one reads 4 times faster than the other, 
    prove that if the slower reader takes 90 minutes to read a book, 
    the faster reader will take 22.5 minutes to read the same book. -/
theorem fast_reader_time (slow_reader_time : ℝ) (speed_ratio : ℝ) 
    (h1 : slow_reader_time = 90) 
    (h2 : speed_ratio = 4) : 
  slow_reader_time / speed_ratio = 22.5 := by
  sorry

#check fast_reader_time

end NUMINAMATH_CALUDE_fast_reader_time_l2711_271100


namespace NUMINAMATH_CALUDE_distance_to_place_distance_calculation_l2711_271118

/-- Calculates the distance to a place given rowing speeds and time -/
theorem distance_to_place (still_water_speed : ℝ) (current_velocity : ℝ) (total_time : ℝ) : ℝ :=
  let downstream_speed := still_water_speed + current_velocity
  let upstream_speed := still_water_speed - current_velocity
  let downstream_time := (total_time * upstream_speed) / (downstream_speed + upstream_speed)
  let distance := downstream_time * downstream_speed
  distance

/-- The distance to the place is approximately 10.83 km -/
theorem distance_calculation : 
  abs (distance_to_place 8 2.5 3 - 10.83) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_place_distance_calculation_l2711_271118


namespace NUMINAMATH_CALUDE_karen_rolls_count_l2711_271196

/-- The number of egg rolls Omar rolled -/
def omar_rolls : ℕ := 219

/-- The total number of egg rolls Omar and Karen rolled -/
def total_rolls : ℕ := 448

/-- The number of egg rolls Karen rolled -/
def karen_rolls : ℕ := total_rolls - omar_rolls

theorem karen_rolls_count : karen_rolls = 229 := by
  sorry

end NUMINAMATH_CALUDE_karen_rolls_count_l2711_271196


namespace NUMINAMATH_CALUDE_fruit_stand_problem_l2711_271162

def fruit_problem (apple_price orange_price : ℚ) 
                  (total_fruits : ℕ) 
                  (initial_avg_price desired_avg_price : ℚ) : Prop :=
  let oranges_to_remove := 10
  let remaining_fruits := total_fruits - oranges_to_remove
  ∃ (apples oranges : ℕ),
    apples + oranges = total_fruits ∧
    (apple_price * apples + orange_price * oranges) / total_fruits = initial_avg_price ∧
    (apple_price * apples + orange_price * (oranges - oranges_to_remove)) / remaining_fruits = desired_avg_price

theorem fruit_stand_problem :
  fruit_problem (40/100) (60/100) 20 (56/100) (52/100) :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_stand_problem_l2711_271162


namespace NUMINAMATH_CALUDE_abc_inequality_l2711_271184

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a / (a * b + 1) + b / (b * c + 1) + c / (c * a + 1) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2711_271184


namespace NUMINAMATH_CALUDE_probability_of_guessing_two_questions_correctly_l2711_271170

theorem probability_of_guessing_two_questions_correctly :
  let num_questions : ℕ := 2
  let options_per_question : ℕ := 4
  let prob_one_correct : ℚ := 1 / options_per_question
  prob_one_correct ^ num_questions = (1 : ℚ) / 16 := by sorry

end NUMINAMATH_CALUDE_probability_of_guessing_two_questions_correctly_l2711_271170


namespace NUMINAMATH_CALUDE_cos_alpha_minus_beta_l2711_271153

theorem cos_alpha_minus_beta (α β : Real) 
  (h1 : α > -π/4 ∧ α < π/4) 
  (h2 : β > -π/4 ∧ β < π/4) 
  (h3 : Real.cos (2*α + 2*β) = -7/9) 
  (h4 : Real.sin α * Real.sin β = 1/4) : 
  Real.cos (α - β) = 5/6 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_beta_l2711_271153


namespace NUMINAMATH_CALUDE_fabian_marbles_comparison_l2711_271159

theorem fabian_marbles_comparison (fabian_marbles kyle_marbles miles_marbles : ℕ) : 
  fabian_marbles = 15 →
  fabian_marbles = 3 * kyle_marbles →
  kyle_marbles + miles_marbles = 8 →
  fabian_marbles = 5 * miles_marbles :=
by
  sorry

end NUMINAMATH_CALUDE_fabian_marbles_comparison_l2711_271159


namespace NUMINAMATH_CALUDE_total_turtles_received_l2711_271117

theorem total_turtles_received (martha_turtles : ℕ) (marion_extra_turtles : ℕ) : 
  martha_turtles = 40 → 
  marion_extra_turtles = 20 → 
  martha_turtles + (martha_turtles + marion_extra_turtles) = 100 := by
sorry

end NUMINAMATH_CALUDE_total_turtles_received_l2711_271117


namespace NUMINAMATH_CALUDE_rhombus_area_l2711_271108

/-- The area of a rhombus with side length 13 units and one interior angle of 60 degrees is (169√3)/2 square units. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 13) (h2 : θ = π / 3) :
  s^2 * Real.sin θ = (169 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l2711_271108


namespace NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l2711_271182

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the point P₀
def P₀ : ℝ × ℝ := (-1, -4)

-- Define the slope of the line parallel to the tangent
def parallel_slope : ℝ := 4

theorem tangent_point_and_perpendicular_line :
  -- The tangent line at P₀ is parallel to 4x - y - 1 = 0
  curve_derivative (P₀.1) = parallel_slope →
  -- P₀ is in the third quadrant
  P₀.1 < 0 ∧ P₀.2 < 0 →
  -- P₀ lies on the curve
  curve P₀.1 = P₀.2 →
  -- The equation of the perpendicular line passing through P₀
  ∃ (a b c : ℝ), a * P₀.1 + b * P₀.2 + c = 0 ∧
                 a = 1 ∧ b = 4 ∧ c = 17 ∧
                 -- The perpendicular line is indeed perpendicular to the tangent
                 a * parallel_slope + b = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l2711_271182


namespace NUMINAMATH_CALUDE_committee_count_l2711_271195

theorem committee_count (n m : ℕ) (hn : n = 8) (hm : m = 4) :
  (n.choose 1) * ((n - 1).choose 1) * ((n - 2).choose (m - 2)) = 840 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_l2711_271195


namespace NUMINAMATH_CALUDE_intersection_and_slope_l2711_271156

theorem intersection_and_slope (k : ℝ) :
  (∃ y : ℝ, -3 * 3 + y = k ∧ 3 + y = 8) →
  k = -4 ∧ 
  (∀ x y : ℝ, x + y = 8 → y = -x + 8) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_slope_l2711_271156


namespace NUMINAMATH_CALUDE_dawn_hourly_income_l2711_271119

/-- Calculates the hourly income for Dawn's painting project -/
theorem dawn_hourly_income (num_paintings : ℕ) 
                            (sketch_time painting_time finish_time : ℝ) 
                            (watercolor_payment sketch_payment finish_payment : ℝ) : 
  num_paintings = 12 ∧ 
  sketch_time = 1.5 ∧ 
  painting_time = 2 ∧ 
  finish_time = 0.5 ∧
  watercolor_payment = 3600 ∧ 
  sketch_payment = 1200 ∧ 
  finish_payment = 300 → 
  (watercolor_payment + sketch_payment + finish_payment) / 
  (num_paintings * (sketch_time + painting_time + finish_time)) = 106.25 := by
sorry

end NUMINAMATH_CALUDE_dawn_hourly_income_l2711_271119


namespace NUMINAMATH_CALUDE_fermat_sum_of_two_squares_l2711_271111

theorem fermat_sum_of_two_squares (p : ℕ) (h_prime : Nat.Prime p) (h_mod : p % 4 = 1) :
  ∃ a b : ℤ, p = a^2 + b^2 := by sorry

end NUMINAMATH_CALUDE_fermat_sum_of_two_squares_l2711_271111


namespace NUMINAMATH_CALUDE_bert_crossword_theorem_l2711_271146

/-- Represents a crossword puzzle --/
structure Crossword where
  size : Nat × Nat
  words : Nat

/-- Represents Bert's crossword solving habits --/
structure CrosswordHabit where
  puzzlesPerDay : Nat
  daysToUsePencil : Nat
  wordsPerPencil : Nat

/-- Calculate the average words per puzzle --/
def avgWordsPerPuzzle (habit : CrosswordHabit) : Nat :=
  habit.wordsPerPencil / (habit.puzzlesPerDay * habit.daysToUsePencil)

/-- Calculate the estimated words for a given puzzle size --/
def estimatedWords (baseSize : Nat × Nat) (baseWords : Nat) (newSize : Nat × Nat) : Nat :=
  let baseArea := baseSize.1 * baseSize.2
  let newArea := newSize.1 * newSize.2
  (baseWords * newArea) / baseArea

/-- Main theorem about Bert's crossword habits --/
theorem bert_crossword_theorem (habit : CrosswordHabit)
  (h1 : habit.puzzlesPerDay = 1)
  (h2 : habit.daysToUsePencil = 14)
  (h3 : habit.wordsPerPencil = 1050) :
  avgWordsPerPuzzle habit = 75 ∧
  estimatedWords (15, 15) 75 (21, 21) - 75 = 72 := by
  sorry

end NUMINAMATH_CALUDE_bert_crossword_theorem_l2711_271146


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2711_271197

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, (3 * y - 4 > 2 * y + 5) → y ≥ 10 ∧ (3 * 10 - 4 > 2 * 10 + 5) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2711_271197


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2711_271187

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 9

-- State the theorem
theorem quadratic_minimum :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2711_271187


namespace NUMINAMATH_CALUDE_triangle_problem_l2711_271194

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a = 3 →
  b = 2 →
  Real.cos A = 1/2 →
  -- (I)
  Real.sin B = Real.sqrt 3 / 3 ∧
  -- (II)
  c = 1 + Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2711_271194


namespace NUMINAMATH_CALUDE_simple_interest_rate_equivalence_l2711_271151

theorem simple_interest_rate_equivalence (P : ℝ) (P_pos : P > 0) :
  let initial_rate : ℝ := 5 / 100
  let initial_time : ℝ := 8
  let new_time : ℝ := 5
  let new_rate : ℝ := 8 / 100
  (P * initial_rate * initial_time) = (P * new_rate * new_time) := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_equivalence_l2711_271151


namespace NUMINAMATH_CALUDE_infinitely_many_common_divisors_l2711_271193

theorem infinitely_many_common_divisors :
  Set.Infinite {n : ℕ | ∃ d : ℕ, d > 1 ∧ d ∣ (2*n - 3) ∧ d ∣ (3*n - 2)} :=
by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_common_divisors_l2711_271193


namespace NUMINAMATH_CALUDE_find_a_l2711_271116

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2^x else a + 2*x

-- State the theorem
theorem find_a : ∃ a : ℝ, (f a (f a (-1)) = 2) ∧ (a = 1) := by
  sorry

end NUMINAMATH_CALUDE_find_a_l2711_271116


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_c_l2711_271124

theorem quadratic_roots_imply_c (c : ℚ) : 
  (∀ x : ℚ, 2 * x^2 + 14 * x + c = 0 ↔ x = (-14 + Real.sqrt 10) / 4 ∨ x = (-14 - Real.sqrt 10) / 4) →
  c = 93 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_c_l2711_271124


namespace NUMINAMATH_CALUDE_expression_evaluation_l2711_271181

theorem expression_evaluation : 
  let x : ℚ := -1/2
  let expr := (x - 2) / ((x^2 + 4*x + 4) * ((x^2 + x - 6) / (x + 2) - x + 2))
  expr = 2/3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2711_271181


namespace NUMINAMATH_CALUDE_range_of_x_l2711_271135

theorem range_of_x (x : ℝ) : 
  (Real.sqrt ((1 - 2*x)^2) = 2*x - 1) → x ≥ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l2711_271135


namespace NUMINAMATH_CALUDE_max_square_plots_l2711_271129

/-- Represents the dimensions of the field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available fencing -/
def availableFence : ℕ := 2250

/-- Calculates the number of square plots given the side length -/
def numPlots (dimensions : FieldDimensions) (sideLength : ℕ) : ℕ :=
  (dimensions.width / sideLength) * (dimensions.length / sideLength)

/-- Calculates the required fencing for a given configuration -/
def requiredFencing (dimensions : FieldDimensions) (sideLength : ℕ) : ℕ :=
  (dimensions.width / sideLength - 1) * dimensions.length +
  (dimensions.length / sideLength - 1) * dimensions.width

/-- Checks if a given side length is valid for the field dimensions -/
def isValidSideLength (dimensions : FieldDimensions) (sideLength : ℕ) : Prop :=
  sideLength > 0 ∧
  dimensions.width % sideLength = 0 ∧
  dimensions.length % sideLength = 0

theorem max_square_plots (dimensions : FieldDimensions)
    (h1 : dimensions.width = 30)
    (h2 : dimensions.length = 45) :
    (∃ (sideLength : ℕ),
      isValidSideLength dimensions sideLength ∧
      requiredFencing dimensions sideLength ≤ availableFence ∧
      numPlots dimensions sideLength = 150 ∧
      (∀ (s : ℕ), isValidSideLength dimensions s →
        requiredFencing dimensions s ≤ availableFence →
        numPlots dimensions s ≤ 150)) :=
  sorry

end NUMINAMATH_CALUDE_max_square_plots_l2711_271129


namespace NUMINAMATH_CALUDE_darcy_laundry_theorem_l2711_271166

/-- Given the number of shirts and shorts Darcy has, and the number he has folded,
    calculate the number of remaining pieces to fold. -/
def remaining_to_fold (total_shirts : ℕ) (total_shorts : ℕ) 
                      (folded_shirts : ℕ) (folded_shorts : ℕ) : ℕ :=
  (total_shirts - folded_shirts) + (total_shorts - folded_shorts)

/-- Theorem stating that with 20 shirts and 8 shorts, 
    if 12 shirts and 5 shorts are folded, 
    11 pieces remain to be folded. -/
theorem darcy_laundry_theorem : 
  remaining_to_fold 20 8 12 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_darcy_laundry_theorem_l2711_271166


namespace NUMINAMATH_CALUDE_probability_same_color_l2711_271180

def num_green_balls : ℕ := 7
def num_white_balls : ℕ := 7

def total_balls : ℕ := num_green_balls + num_white_balls

def same_color_combinations : ℕ := (num_green_balls.choose 2) + (num_white_balls.choose 2)
def total_combinations : ℕ := total_balls.choose 2

theorem probability_same_color :
  (same_color_combinations : ℚ) / total_combinations = 42 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_l2711_271180


namespace NUMINAMATH_CALUDE_total_watch_time_l2711_271134

/-- Calculate the total watch time for John's videos in a week -/
theorem total_watch_time
  (short_video_length : ℕ)
  (long_video_multiplier : ℕ)
  (short_videos_per_day : ℕ)
  (long_videos_per_day : ℕ)
  (days_per_week : ℕ)
  (retention_rate : ℝ)
  (h1 : short_video_length = 2)
  (h2 : long_video_multiplier = 6)
  (h3 : short_videos_per_day = 2)
  (h4 : long_videos_per_day = 1)
  (h5 : days_per_week = 7)
  (h6 : 0 < retention_rate)
  (h7 : retention_rate ≤ 100)
  : ℝ :=
by
  sorry

#check total_watch_time

end NUMINAMATH_CALUDE_total_watch_time_l2711_271134


namespace NUMINAMATH_CALUDE_rectangle_circumference_sum_l2711_271143

/-- Calculates the sum of coins around the circumference of a rectangle formed by coins -/
def circumference_sum (horizontal : Nat) (vertical : Nat) (coin_value : Nat) : Nat :=
  let horizontal_edge := 2 * (horizontal - 2)
  let vertical_edge := 2 * (vertical - 2)
  let corners := 4
  (horizontal_edge + vertical_edge + corners) * coin_value

/-- Theorem stating that the sum of coins around the circumference of a 6x4 rectangle of 100-won coins is 1600 won -/
theorem rectangle_circumference_sum :
  circumference_sum 6 4 100 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circumference_sum_l2711_271143


namespace NUMINAMATH_CALUDE_cannot_obtain_123_l2711_271132

/-- Represents an arithmetic expression using numbers 1, 2, 3, 4, 5 and operations +, -, * -/
inductive Expr
| Num : Fin 5 → Expr
| Add : Expr → Expr → Expr
| Sub : Expr → Expr → Expr
| Mul : Expr → Expr → Expr

/-- Evaluates an arithmetic expression -/
def eval : Expr → Int
| Expr.Num n => n.val.succ
| Expr.Add e1 e2 => eval e1 + eval e2
| Expr.Sub e1 e2 => eval e1 - eval e2
| Expr.Mul e1 e2 => eval e1 * eval e2

/-- Theorem stating that it's impossible to obtain 123 using the given constraints -/
theorem cannot_obtain_123 : ¬ ∃ e : Expr, eval e = 123 := by
  sorry

end NUMINAMATH_CALUDE_cannot_obtain_123_l2711_271132
