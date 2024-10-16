import Mathlib

namespace NUMINAMATH_CALUDE_sandys_correct_sums_l1662_166286

theorem sandys_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (correct_marks : ℕ) 
  (incorrect_marks : ℕ) 
  (h1 : total_sums = 30) 
  (h2 : total_marks = 55) 
  (h3 : correct_marks = 3) 
  (h4 : incorrect_marks = 2) : 
  ∃ (correct_sums : ℕ), 
    correct_sums * correct_marks - (total_sums - correct_sums) * incorrect_marks = total_marks ∧ 
    correct_sums = 23 := by
  sorry

end NUMINAMATH_CALUDE_sandys_correct_sums_l1662_166286


namespace NUMINAMATH_CALUDE_fraction_C_is_simplest_l1662_166238

-- Define the fractions
def fraction_A (m n : ℚ) : ℚ := 2 * m / (10 * m * n)
def fraction_B (m n : ℚ) : ℚ := (m^2 - n^2) / (m + n)
def fraction_C (m n : ℚ) : ℚ := (m^2 + n^2) / (m + n)
def fraction_D (a : ℚ) : ℚ := 2 * a / a^2

-- Define what it means for a fraction to be in simplest form
def is_simplest_form (f : ℚ) : Prop := 
  ∀ (g : ℚ), g ≠ 1 → g ≠ -1 → f ≠ g * (↑(f.num) / ↑(f.den))

-- Theorem statement
theorem fraction_C_is_simplest : 
  ∀ (m n : ℚ), m + n ≠ 0 → is_simplest_form (fraction_C m n) := by sorry

end NUMINAMATH_CALUDE_fraction_C_is_simplest_l1662_166238


namespace NUMINAMATH_CALUDE_race_distance_l1662_166225

/-- Race conditions and proof of distance -/
theorem race_distance (d x y z : ℝ) 
  (h1 : d / x = (d - 25) / y)  -- X beats Y by 25 meters
  (h2 : d / y = (d - 15) / z)  -- Y beats Z by 15 meters
  (h3 : d / x = (d - 37) / z)  -- X beats Z by 37 meters
  (h4 : d > 0) : d = 125 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l1662_166225


namespace NUMINAMATH_CALUDE_power_of_two_inequality_l1662_166257

theorem power_of_two_inequality (k l m : ℕ) :
  2^(k+1) + 2^(k+m) + 2^(l+m) ≤ 2^(k+l+m+1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_inequality_l1662_166257


namespace NUMINAMATH_CALUDE_some_athletes_not_honor_society_l1662_166209

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Athlete : U → Prop)
variable (Disciplined : U → Prop)
variable (HonorSocietyMember : U → Prop)

-- Define the conditions
variable (h1 : ∃ x, Athlete x ∧ ¬Disciplined x)
variable (h2 : ∀ x, HonorSocietyMember x → Disciplined x)

-- Theorem to prove
theorem some_athletes_not_honor_society :
  ∃ x, Athlete x ∧ ¬HonorSocietyMember x :=
sorry

end NUMINAMATH_CALUDE_some_athletes_not_honor_society_l1662_166209


namespace NUMINAMATH_CALUDE_cos_2alpha_eq_neg_four_fifths_l1662_166249

theorem cos_2alpha_eq_neg_four_fifths (α : Real) 
  (h : (Real.tan α + 1) / (Real.tan α - 1) = 2) : 
  Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_eq_neg_four_fifths_l1662_166249


namespace NUMINAMATH_CALUDE_polynomial_equality_l1662_166203

-- Define the polynomials
variable (x : ℝ)
def f (x : ℝ) : ℝ := x^3 - 3*x - 1
def h (x : ℝ) : ℝ := -x^3 + 5*x^2 + 3*x

-- State the theorem
theorem polynomial_equality :
  (∀ x, f x + h x = 5*x^2 - 1) ∧ 
  (∀ x, f x = x^3 - 3*x - 1) →
  (∀ x, h x = -x^3 + 5*x^2 + 3*x) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1662_166203


namespace NUMINAMATH_CALUDE_muscovy_female_percentage_l1662_166284

theorem muscovy_female_percentage (total_ducks : ℕ) (muscovy_percentage : ℚ) (female_muscovy : ℕ) :
  total_ducks = 40 →
  muscovy_percentage = 1/2 →
  female_muscovy = 6 →
  (female_muscovy : ℚ) / (muscovy_percentage * total_ducks) = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_muscovy_female_percentage_l1662_166284


namespace NUMINAMATH_CALUDE_least_whole_number_for_ratio_l1662_166207

theorem least_whole_number_for_ratio : 
  ∃ x : ℕ, x > 0 ∧ 
    (∀ y : ℕ, y > 0 → y < x → (6 - y : ℚ) / (7 - y) ≥ 16 / 21) ∧
    (6 - x : ℚ) / (7 - x) < 16 / 21 :=
by
  use 3
  sorry

end NUMINAMATH_CALUDE_least_whole_number_for_ratio_l1662_166207


namespace NUMINAMATH_CALUDE_calculation_proof_l1662_166243

theorem calculation_proof : (2013 : ℚ) / (25 * 52 - 46 * 15) * 10 = 33 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1662_166243


namespace NUMINAMATH_CALUDE_guess_who_i_am_l1662_166231

theorem guess_who_i_am : ∃ x y : ℕ,
  120 = 4 * x ∧
  87 = y - 40 ∧
  x = 30 ∧
  y = 127 := by
sorry

end NUMINAMATH_CALUDE_guess_who_i_am_l1662_166231


namespace NUMINAMATH_CALUDE_length_of_chord_line_equation_l1662_166218

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line passing through (1,0)
def line_through_P (m b : ℝ) (x y : ℝ) : Prop := y = m*(x - 1)

-- Define the intersection points of the line and parabola
def intersection_points (m b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | parabola x y ∧ line_through_P m b x y}

-- Part 1: Length of chord AB when slope is 1
theorem length_of_chord (A B : ℝ × ℝ) :
  A ∈ intersection_points 1 0 →
  B ∈ intersection_points 1 0 →
  A ≠ B →
  ‖A - B‖ = 2 * Real.sqrt 6 :=
sorry

-- Part 2: Equation of line when PA = -2PB
theorem line_equation (A B : ℝ × ℝ) (m : ℝ) :
  A ∈ intersection_points m 0 →
  B ∈ intersection_points m 0 →
  A ≠ B →
  (A.1 - 1, A.2) = (-2 * (B.1 - 1), -2 * B.2) →
  (m = 1/2 ∨ m = -1/2) :=
sorry

end NUMINAMATH_CALUDE_length_of_chord_line_equation_l1662_166218


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1662_166292

theorem negation_of_existential_proposition :
  (¬ (∃ x : ℝ, 2 * x - 3 > 1)) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1662_166292


namespace NUMINAMATH_CALUDE_divisible_pair_count_l1662_166250

/-- Given a set of 2117 cards numbered from 1 to 2117, this function calculates
    the number of ways to choose two cards such that their sum is divisible by 100. -/
def count_divisible_pairs : ℕ := 
  let card_count := 2117
  sorry

/-- Theorem stating that the number of ways to choose two cards with a sum
    divisible by 100 from a set of 2117 cards numbered 1 to 2117 is 23058. -/
theorem divisible_pair_count : count_divisible_pairs = 23058 := by
  sorry

end NUMINAMATH_CALUDE_divisible_pair_count_l1662_166250


namespace NUMINAMATH_CALUDE_smallest_divisible_by_14_15_16_l1662_166217

theorem smallest_divisible_by_14_15_16 : ∃ n : ℕ, n > 0 ∧ 14 ∣ n ∧ 15 ∣ n ∧ 16 ∣ n ∧ ∀ m : ℕ, m > 0 → 14 ∣ m → 15 ∣ m → 16 ∣ m → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_14_15_16_l1662_166217


namespace NUMINAMATH_CALUDE_walk_distance_proof_l1662_166290

/-- Calculates the distance traveled given a constant speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that walking at 4 miles per hour for 2 hours results in a distance of 8 miles -/
theorem walk_distance_proof :
  let speed : ℝ := 4
  let time : ℝ := 2
  distance_traveled speed time = 8 := by sorry

end NUMINAMATH_CALUDE_walk_distance_proof_l1662_166290


namespace NUMINAMATH_CALUDE_march_text_messages_l1662_166236

def T (n : ℕ) : ℕ := ((n^2) + 1) * n.factorial

theorem march_text_messages : T 5 = 3120 := by
  sorry

end NUMINAMATH_CALUDE_march_text_messages_l1662_166236


namespace NUMINAMATH_CALUDE_no_perfect_square_solution_l1662_166215

theorem no_perfect_square_solution : 
  ¬ ∃ (n : ℕ+) (m : ℕ), n^2 + 12*n - 2006 = m^2 := by
sorry

end NUMINAMATH_CALUDE_no_perfect_square_solution_l1662_166215


namespace NUMINAMATH_CALUDE_tom_dance_lesson_payment_l1662_166251

/-- The amount Tom pays for dance lessons -/
def tom_payment (total_lessons : ℕ) (cost_per_lesson : ℕ) (free_lessons : ℕ) : ℕ :=
  (total_lessons - free_lessons) * cost_per_lesson

/-- Proof that Tom pays $80 for dance lessons -/
theorem tom_dance_lesson_payment :
  tom_payment 10 10 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_tom_dance_lesson_payment_l1662_166251


namespace NUMINAMATH_CALUDE_avocados_for_guacamole_l1662_166222

/-- The number of avocados needed for one serving of guacamole -/
def avocados_per_serving (initial_avocados sister_avocados total_servings : ℕ) : ℕ :=
  (initial_avocados + sister_avocados) / total_servings

/-- Theorem stating that 3 avocados are needed for one serving of guacamole -/
theorem avocados_for_guacamole :
  avocados_per_serving 5 4 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_avocados_for_guacamole_l1662_166222


namespace NUMINAMATH_CALUDE_contest_team_mistakes_l1662_166265

/-- The number of incorrect answers for a team in a math contest -/
def team_incorrect_answers (total_questions : ℕ) (riley_mistakes : ℕ) (ofelia_correct_offset : ℕ) : ℕ :=
  let riley_correct := total_questions - riley_mistakes
  let ofelia_correct := riley_correct / 2 + ofelia_correct_offset
  let ofelia_mistakes := total_questions - ofelia_correct
  riley_mistakes + ofelia_mistakes

/-- Theorem stating the number of incorrect answers for Riley and Ofelia's team -/
theorem contest_team_mistakes :
  team_incorrect_answers 35 3 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_contest_team_mistakes_l1662_166265


namespace NUMINAMATH_CALUDE_turquoise_more_blue_count_l1662_166234

/-- Represents the results of a survey about the color turquoise -/
structure TurquoiseSurvey where
  total : ℕ
  more_green : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of people who believe turquoise is "more blue" -/
def more_blue (survey : TurquoiseSurvey) : ℕ :=
  survey.total - (survey.more_green - survey.both) - survey.neither - survey.both

/-- Theorem stating that in the given survey, 65 people believe turquoise is "more blue" -/
theorem turquoise_more_blue_count :
  ∃ (survey : TurquoiseSurvey),
    survey.total = 150 ∧
    survey.more_green = 95 ∧
    survey.both = 35 ∧
    survey.neither = 25 ∧
    more_blue survey = 65 := by
  sorry

#eval more_blue ⟨150, 95, 35, 25⟩

end NUMINAMATH_CALUDE_turquoise_more_blue_count_l1662_166234


namespace NUMINAMATH_CALUDE_park_to_grocery_distance_l1662_166237

/-- The distance from Talia's house to the park, in miles -/
def distance_house_to_park : ℝ := 5

/-- The distance from Talia's house to the grocery store, in miles -/
def distance_house_to_grocery : ℝ := 8

/-- The total distance Talia drives, in miles -/
def total_distance : ℝ := 16

/-- The distance from the park to the grocery store, in miles -/
def distance_park_to_grocery : ℝ := total_distance - distance_house_to_park - distance_house_to_grocery

theorem park_to_grocery_distance :
  distance_park_to_grocery = 3 := by sorry

end NUMINAMATH_CALUDE_park_to_grocery_distance_l1662_166237


namespace NUMINAMATH_CALUDE_bacteria_growth_l1662_166283

theorem bacteria_growth (n : ℕ) : n = 4 ↔ (n > 0 ∧ 5 * 3^n > 200 ∧ ∀ m : ℕ, m > 0 → m < n → 5 * 3^m ≤ 200) :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_l1662_166283


namespace NUMINAMATH_CALUDE_simplify_expression_l1662_166223

theorem simplify_expression (x : ℝ) (h : x^2 ≥ 49) :
  (7 - Real.sqrt (x^2 - 49))^2 = x^2 - 14 * Real.sqrt (x^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1662_166223


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_equals_four_l1662_166262

def A (a : ℝ) : Set ℝ := {x | a * x^2 + a * x + 1 = 0}

theorem unique_solution_implies_a_equals_four (a : ℝ) :
  (∃! x, x ∈ A a) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_equals_four_l1662_166262


namespace NUMINAMATH_CALUDE_line_divides_l_shape_in_half_l1662_166260

/-- L-shaped region in the xy-plane -/
structure LShapedRegion where
  vertices : List (ℝ × ℝ) := [(0,0), (0,4), (4,4), (4,2), (6,2), (6,0)]

/-- Line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculate the area of a polygon given its vertices -/
def calculateArea (vertices : List (ℝ × ℝ)) : ℝ := sorry

/-- Calculate the slope of a line -/
def calculateSlope (l : Line) : ℝ := sorry

/-- Check if a line divides a region in half -/
def divides_in_half (l : Line) (r : LShapedRegion) : Prop := sorry

/-- Theorem: The line through (0,0) and (2,4) divides the L-shaped region in half -/
theorem line_divides_l_shape_in_half :
  let l : Line := { point1 := (0, 0), point2 := (2, 4) }
  let r : LShapedRegion := {}
  divides_in_half l r ∧ calculateSlope l = 2 := by sorry

end NUMINAMATH_CALUDE_line_divides_l_shape_in_half_l1662_166260


namespace NUMINAMATH_CALUDE_smallest_n_divisible_l1662_166282

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 
  24 ∣ n^2 ∧ 
  1024 ∣ n^3 ∧ 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(24 ∣ m^2 ∧ 1024 ∣ m^3)) → 
  n = 48 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_l1662_166282


namespace NUMINAMATH_CALUDE_prob_at_least_two_different_fruits_l1662_166246

def num_meals : ℕ := 4
def num_fruits : ℕ := 4

def prob_same_fruit_all_day : ℚ := (1 / num_fruits) ^ num_meals * num_fruits

theorem prob_at_least_two_different_fruits :
  1 - prob_same_fruit_all_day = 63 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_different_fruits_l1662_166246


namespace NUMINAMATH_CALUDE_guests_not_responded_l1662_166221

def total_guests : ℕ := 200
def yes_percentage : ℚ := 83 / 100
def no_percentage : ℚ := 9 / 100

theorem guests_not_responded : 
  (total_guests : ℚ) - 
  (yes_percentage * total_guests + no_percentage * total_guests) = 16 := by
  sorry

end NUMINAMATH_CALUDE_guests_not_responded_l1662_166221


namespace NUMINAMATH_CALUDE_binomial_distributions_l1662_166206

/-- A random variable follows a binomial distribution if it represents the number of successes
    in a fixed number of independent Bernoulli trials with the same probability of success. -/
def IsBinomialDistribution (X : ℕ → ℝ) : Prop :=
  ∃ (n : ℕ) (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧
    ∀ k, 0 ≤ k ∧ k ≤ n → X k = (n.choose k : ℝ) * p^k * (1-p)^(n-k)

/-- The probability mass function for the number of shots needed to hit the target for the first time -/
def GeometricDistribution (p : ℝ) (X : ℕ → ℝ) : Prop :=
  0 < p ∧ p ≤ 1 ∧ ∀ k, k > 0 → X k = (1-p)^(k-1) * p

/-- The distribution of computer virus infections -/
def VirusInfection (n : ℕ) (X : ℕ → ℝ) : Prop :=
  IsBinomialDistribution X

/-- The distribution of hitting a target in n shots -/
def TargetHits (n : ℕ) (X : ℕ → ℝ) : Prop :=
  IsBinomialDistribution X

/-- The distribution of cars refueling at a gas station -/
def CarRefueling (X : ℕ → ℝ) : Prop :=
  IsBinomialDistribution X

theorem binomial_distributions (n : ℕ) (p : ℝ) (X₁ X₂ X₃ X₄ : ℕ → ℝ) :
  VirusInfection n X₁ ∧
  GeometricDistribution p X₂ ∧
  TargetHits n X₃ ∧
  CarRefueling X₄ →
  IsBinomialDistribution X₁ ∧
  ¬IsBinomialDistribution X₂ ∧
  IsBinomialDistribution X₃ ∧
  IsBinomialDistribution X₄ :=
sorry

end NUMINAMATH_CALUDE_binomial_distributions_l1662_166206


namespace NUMINAMATH_CALUDE_complex_solutions_of_x_squared_equals_negative_four_l1662_166216

theorem complex_solutions_of_x_squared_equals_negative_four :
  ∀ x : ℂ, x^2 = -4 ↔ x = 2*I ∨ x = -2*I :=
sorry

end NUMINAMATH_CALUDE_complex_solutions_of_x_squared_equals_negative_four_l1662_166216


namespace NUMINAMATH_CALUDE_find_V_l1662_166259

-- Define the relationship between U, V, and W
def relationship (k : ℝ) (U V W : ℝ) : Prop :=
  U = k * (V / W)

-- Define the theorem
theorem find_V (k : ℝ) :
  relationship k 16 2 (1/4) →
  relationship k 25 (5/2) (1/5) :=
by sorry

end NUMINAMATH_CALUDE_find_V_l1662_166259


namespace NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l1662_166273

theorem negation_of_cube_odd_is_odd :
  ¬(∀ x : ℤ, Odd x → Odd (x^3)) ↔ ∃ x : ℤ, Odd x ∧ ¬Odd (x^3) :=
sorry

end NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l1662_166273


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1662_166267

/-- Given a square with perimeter 144 units divided into 4 congruent rectangles by vertical lines,
    the perimeter of one rectangle is 90 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (num_rectangles : ℕ) : 
  square_perimeter = 144 → 
  num_rectangles = 4 → 
  ∃ (rectangle_perimeter : ℝ), rectangle_perimeter = 90 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1662_166267


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_31_l1662_166233

/-- Represents a quadrilateral with vertices A, B, C, D and intersection point O of diagonals -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)

/-- The area of a quadrilateral given its side lengths and angle between diagonals -/
def area_quadrilateral (q : Quadrilateral) (AB BC CD DA : ℝ) (angle_COB : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the area of the given quadrilateral is 31 -/
theorem quadrilateral_area_is_31 (q : Quadrilateral) 
  (h1 : area_quadrilateral q 10 6 8 2 (π/4) = 31) : 
  area_quadrilateral q 10 6 8 2 (π/4) = 31 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_31_l1662_166233


namespace NUMINAMATH_CALUDE_molecular_weight_CCl4_proof_l1662_166204

/-- The molecular weight of CCl4 in g/mol -/
def molecular_weight_CCl4 : ℝ := 152

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 7

/-- The total weight of the given moles of CCl4 in g/mol -/
def given_total_weight : ℝ := 1064

/-- Theorem stating that the molecular weight of CCl4 is correct given the condition -/
theorem molecular_weight_CCl4_proof :
  molecular_weight_CCl4 * given_moles = given_total_weight :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_CCl4_proof_l1662_166204


namespace NUMINAMATH_CALUDE_smallest_prime_cube_sum_fourth_power_l1662_166268

theorem smallest_prime_cube_sum_fourth_power :
  ∃ (p : ℕ), Prime p ∧ 
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a^2 + p^3 = b^4) ∧
  (∀ (q : ℕ), Prime q → q < p → 
    ¬∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c^2 + q^3 = d^4) ∧
  p = 23 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_cube_sum_fourth_power_l1662_166268


namespace NUMINAMATH_CALUDE_balanced_split_theorem_l1662_166289

/-- A finite collection of positive real numbers is balanced if each number
    is less than the sum of the others. -/
def IsBalanced (s : Finset ℝ) : Prop :=
  ∀ x ∈ s, x < (s.sum id - x)

/-- A finite collection of positive real numbers can be split into three parts
    with the property that the sum of the numbers in each part is less than
    the sum of the numbers in the two other parts. -/
def CanSplitIntoThreeParts (s : Finset ℝ) : Prop :=
  ∃ (a b c : Finset ℝ), a ∪ b ∪ c = s ∧ a ∩ b = ∅ ∧ b ∩ c = ∅ ∧ a ∩ c = ∅ ∧
    a.sum id < b.sum id + c.sum id ∧
    b.sum id < a.sum id + c.sum id ∧
    c.sum id < a.sum id + b.sum id

/-- The main theorem -/
theorem balanced_split_theorem (m : ℕ) (hm : m ≥ 3) :
  (∀ (s : Finset ℝ), s.card = m → IsBalanced s → CanSplitIntoThreeParts s) ↔ m ≠ 4 :=
sorry

end NUMINAMATH_CALUDE_balanced_split_theorem_l1662_166289


namespace NUMINAMATH_CALUDE_number_ordering_l1662_166261

theorem number_ordering (a b : ℝ) (ha : a > 0) (hb : 0 < b ∧ b < 1) :
  a^b > b^a ∧ b^a > Real.log b := by sorry

end NUMINAMATH_CALUDE_number_ordering_l1662_166261


namespace NUMINAMATH_CALUDE_component_unqualified_l1662_166220

/-- Determines if a component is qualified based on its diameter -/
def is_qualified (measured_diameter : ℝ) (specified_diameter : ℝ) (tolerance : ℝ) : Prop :=
  measured_diameter ≥ specified_diameter - tolerance ∧ 
  measured_diameter ≤ specified_diameter + tolerance

/-- Theorem stating that the component with measured diameter 19.9 mm is unqualified -/
theorem component_unqualified : 
  ¬ is_qualified 19.9 20 0.02 := by
  sorry

end NUMINAMATH_CALUDE_component_unqualified_l1662_166220


namespace NUMINAMATH_CALUDE_parallelogram_area_minimum_l1662_166291

theorem parallelogram_area_minimum (z : ℂ) : 
  (∃ (area : ℝ), area = (36:ℝ)/(37:ℝ) ∧ 
    area = 2 * Complex.abs (z * Complex.I * (1/z - z))) →
  (Complex.re z > 0) →
  (Complex.im z < 0) →
  (∃ (d : ℝ), d = Complex.abs (z + 1/z) ∧
    ∀ (w : ℂ), (Complex.re w > 0) → (Complex.im w < 0) →
    (∃ (area : ℝ), area = (36:ℝ)/(37:ℝ) ∧ 
      area = 2 * Complex.abs (w * Complex.I * (1/w - w))) →
    d ≤ Complex.abs (w + 1/w)) →
  (Complex.abs (z + 1/z))^2 = (12:ℝ)/(37:ℝ) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_minimum_l1662_166291


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l1662_166275

theorem fourth_rectangle_area (P Q R S : ℝ × ℝ) : 
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = 25 →
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 49 →
  (S.1 - R.1)^2 + (S.2 - R.2)^2 = 64 →
  (Q.2 - P.2) * (R.1 - P.1) = (Q.1 - P.1) * (R.2 - P.2) →
  (S.2 - P.2) * (R.1 - P.1) = (S.1 - P.1) * (R.2 - P.2) →
  (S.1 - P.1)^2 + (S.2 - P.2)^2 = 89 := by
sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_l1662_166275


namespace NUMINAMATH_CALUDE_union_of_sets_l1662_166255

def A (a : ℝ) : Set ℝ := {1, 2^a}
def B (a b : ℝ) : Set ℝ := {a, b}

theorem union_of_sets (a b : ℝ) :
  A a ∩ B a b = {1/4} →
  A a ∪ B a b = {-2, 1, 1/4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1662_166255


namespace NUMINAMATH_CALUDE_day_crew_load_fraction_l1662_166245

theorem day_crew_load_fraction (D : ℝ) (W_d : ℝ) (W_d_pos : W_d > 0) : 
  let night_boxes_per_worker := (1 / 2) * D
  let night_workers := (4 / 5) * W_d
  let day_total := D * W_d
  let night_total := night_boxes_per_worker * night_workers
  (day_total) / (day_total + night_total) = 5 / 7 := by
sorry

end NUMINAMATH_CALUDE_day_crew_load_fraction_l1662_166245


namespace NUMINAMATH_CALUDE_ternary_decimal_conversion_decimal_base7_conversion_l1662_166205

-- Define a function to convert from base 3 to base 10
def ternary_to_decimal (t : List Nat) : Nat :=
  t.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (t.length - 1 - i))) 0

-- Define a function to convert from base 10 to base 7
def decimal_to_base7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem ternary_decimal_conversion :
  ternary_to_decimal [1, 0, 2, 1, 2] = 104 := by sorry

theorem decimal_base7_conversion :
  decimal_to_base7 1234 = [3, 4, 1, 2] := by sorry

end NUMINAMATH_CALUDE_ternary_decimal_conversion_decimal_base7_conversion_l1662_166205


namespace NUMINAMATH_CALUDE_total_cats_l1662_166281

theorem total_cats (white : ℕ) (black_percentage : ℚ) (grey : ℕ) :
  white = 2 →
  black_percentage = 1/4 →
  grey = 10 →
  ∃ (total : ℕ), 
    total = white + (black_percentage * total).floor + grey ∧
    total = 16 :=
by sorry

end NUMINAMATH_CALUDE_total_cats_l1662_166281


namespace NUMINAMATH_CALUDE_marble_probability_l1662_166232

/-- Given a box of marbles with the following properties:
  - There are 120 marbles in total
  - Each marble is either red, green, blue, or white
  - The probability of drawing a white marble is 1/4
  - The probability of drawing a green marble is 1/3
  This theorem proves that the probability of drawing either a red or blue marble is 5/12. -/
theorem marble_probability (total_marbles : ℕ) (p_white p_green : ℚ)
  (h_total : total_marbles = 120)
  (h_white : p_white = 1/4)
  (h_green : p_green = 1/3) :
  1 - (p_white + p_green) = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l1662_166232


namespace NUMINAMATH_CALUDE_octal_sum_theorem_l1662_166241

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- Converts a decimal number to its octal representation as a list of digits -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The main theorem stating that the sum of 642₈ and 157₈ in base 8 is 1021₈ -/
theorem octal_sum_theorem :
  decimal_to_octal (octal_to_decimal [6, 4, 2] + octal_to_decimal [1, 5, 7]) = [1, 0, 2, 1] :=
sorry

end NUMINAMATH_CALUDE_octal_sum_theorem_l1662_166241


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_condition_l1662_166285

theorem not_necessary_not_sufficient_condition (x : ℝ) :
  ¬(∀ x, x^2 - 2*x = 0 → x = 0) ∧ ¬(∀ x, x = 0 → x^2 - 2*x = 0) := by
  sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_condition_l1662_166285


namespace NUMINAMATH_CALUDE_extended_line_point_l1662_166269

-- Define points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (17, 7)

-- Define the ratio of BC to AB
def ratio : ℚ := 2 / 5

-- Define point C
def C : ℝ × ℝ := (22.6, 9.4)

-- Theorem statement
theorem extended_line_point : 
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  BC.1 = ratio * AB.1 ∧ BC.2 = ratio * AB.2 := by sorry

end NUMINAMATH_CALUDE_extended_line_point_l1662_166269


namespace NUMINAMATH_CALUDE_bus_trip_speed_l1662_166226

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) : 
  distance = 500 ∧ 
  speed_increase = 10 ∧ 
  time_decrease = 2 →
  ∃ (v : ℝ), v > 0 ∧ 
    distance / v - distance / (v + speed_increase) = time_decrease ∧ 
    v = 45.25 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l1662_166226


namespace NUMINAMATH_CALUDE_trig_identity_l1662_166288

theorem trig_identity (x y z : ℝ) : 
  Real.sin (x - y + z) * Real.cos y - Real.cos (x - y + z) * Real.sin y = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1662_166288


namespace NUMINAMATH_CALUDE_parabola_equation_l1662_166213

/-- A parabola with vertex at the origin, focus on the y-axis, and a point P(m, 1) on the parabola that is 5 units away from the focus has the standard equation x^2 = 16y. -/
theorem parabola_equation (m : ℝ) : 
  let p : ℝ → ℝ → Prop := λ x y => x^2 = 16*y  -- Standard equation of the parabola
  let focus : ℝ × ℝ := (0, 4)  -- Focus on y-axis, 4 units above origin
  let vertex : ℝ × ℝ := (0, 0)  -- Vertex at origin
  let point_on_parabola : ℝ × ℝ := (m, 1)  -- Given point on parabola
  (vertex = (0, 0)) →  -- Vertex condition
  (focus.1 = 0) →  -- Focus on y-axis condition
  ((point_on_parabola.1 - focus.1)^2 + (point_on_parabola.2 - focus.2)^2 = 5^2) →  -- Distance condition
  p point_on_parabola.1 point_on_parabola.2  -- Conclusion: point satisfies parabola equation
  := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1662_166213


namespace NUMINAMATH_CALUDE_product_of_fractions_l1662_166228

theorem product_of_fractions : (1 : ℚ) / 3 * 4 / 7 * 9 / 11 = 12 / 77 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1662_166228


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1662_166287

/-- A quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The function is decreasing on the interval (-∞, 4] -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 4 → f a x > f a y

/-- The range of values for a -/
def a_range (a : ℝ) : Prop := a ≤ -3

theorem quadratic_function_theorem (a : ℝ) :
  is_decreasing_on_interval a → a_range a :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1662_166287


namespace NUMINAMATH_CALUDE_yans_distance_ratio_l1662_166299

theorem yans_distance_ratio (w : ℝ) (x y : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) :
  (y / w = x / w + (x + y) / (6 * w)) → (x / y = 5 / 7) :=
by sorry

end NUMINAMATH_CALUDE_yans_distance_ratio_l1662_166299


namespace NUMINAMATH_CALUDE_tangent_line_length_l1662_166266

/-- The length of a tangent line from a point to a circle --/
theorem tangent_line_length 
  (l : ℝ → ℝ → Prop) 
  (C : ℝ → ℝ → Prop) 
  (a : ℝ) :
  (∀ x y, l x y ↔ x + a * y - 1 = 0) →
  (∀ x y, C x y ↔ x^2 + y^2 - 4*x - 2*y + 1 = 0) →
  (∀ x y, l x y → C x y → x = 2 ∧ y = 1) →
  l (-4) a →
  ∃ B : ℝ × ℝ, C B.1 B.2 ∧ 
    (B.1 + 4)^2 + (B.2 + 1)^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_length_l1662_166266


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_3pi_over_5_l1662_166208

theorem cos_2alpha_plus_3pi_over_5 (α : ℝ) 
  (h : Real.sin (π / 5 - α) = 1 / 3) : 
  Real.cos (2 * α + 3 * π / 5) = - 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_3pi_over_5_l1662_166208


namespace NUMINAMATH_CALUDE_ten_trees_road_length_l1662_166272

/-- The length of a road with trees planted at equal intervals --/
def road_length (num_trees : ℕ) (interval : ℕ) : ℕ :=
  (num_trees - 1) * interval

/-- Theorem: The length of a road with 10 trees planted at 10-meter intervals is 90 meters --/
theorem ten_trees_road_length :
  road_length 10 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ten_trees_road_length_l1662_166272


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_seconds_l1662_166263

-- Define the motion equation
def s (t : ℝ) : ℝ := (2 * t + 3) ^ 2

-- Define the instantaneous velocity (derivative of s)
def v (t : ℝ) : ℝ := 4 * (2 * t + 3)

-- Theorem statement
theorem instantaneous_velocity_at_2_seconds : v 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_seconds_l1662_166263


namespace NUMINAMATH_CALUDE_triangle_sequence_solution_l1662_166230

theorem triangle_sequence_solution (b d c k : ℤ) 
  (h1 : b % d = 0)
  (h2 : c % k = 0)
  (h3 : b^2 + (b+2*d)^2 = (c+6*k)^2) :
  ∃ (b d c k : ℤ), c = 0 ∧ 
    b % d = 0 ∧ 
    c % k = 0 ∧ 
    b^2 + (b+2*d)^2 = (c+6*k)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sequence_solution_l1662_166230


namespace NUMINAMATH_CALUDE_evaluate_trigonometric_expression_l1662_166201

theorem evaluate_trigonometric_expression :
  let angle_27 : Real := 27 * Real.pi / 180
  let angle_18 : Real := 18 * Real.pi / 180
  let angle_63 : Real := 63 * Real.pi / 180
  (Real.cos angle_63 = Real.sin angle_27) →
  (angle_27 = 45 * Real.pi / 180 - angle_18) →
  (Real.cos angle_27 - Real.sqrt 2 * Real.sin angle_18) / Real.cos angle_63 = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_trigonometric_expression_l1662_166201


namespace NUMINAMATH_CALUDE_max_students_distribution_l1662_166298

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 640) (h2 : pencils = 520) :
  (∃ (students : ℕ), students > 0 ∧ pens % students = 0 ∧ pencils % students = 0) ∧
  (∀ (n : ℕ), n > 0 ∧ pens % n = 0 ∧ pencils % n = 0 → n ≤ 40) ∧
  (pens % 40 = 0 ∧ pencils % 40 = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l1662_166298


namespace NUMINAMATH_CALUDE_kayla_apples_l1662_166229

theorem kayla_apples (total : ℕ) (kylie : ℕ) (kayla : ℕ) : 
  total = 340 →
  kayla = 4 * kylie + 10 →
  total = kylie + kayla →
  kayla = 274 := by
sorry

end NUMINAMATH_CALUDE_kayla_apples_l1662_166229


namespace NUMINAMATH_CALUDE_equation_has_two_solutions_l1662_166277

-- Define the equation
def equation (x : ℝ) : Prop := |x - 2| = |x - 4| + |x - 6|

-- Define the set of solutions
def solution_set : Set ℝ := {x : ℝ | equation x}

-- Theorem statement
theorem equation_has_two_solutions : 
  ∃ (a b : ℝ), a ≠ b ∧ solution_set = {a, b} :=
sorry

end NUMINAMATH_CALUDE_equation_has_two_solutions_l1662_166277


namespace NUMINAMATH_CALUDE_prob_different_colors_l1662_166274

/-- Represents the color of a chip -/
inductive ChipColor
  | Blue
  | Red
  | Yellow

/-- Represents the state of the bag after the first draw -/
structure BagState where
  blue : Nat
  red : Nat
  yellow : Nat

/-- The initial state of the bag -/
def initialBag : BagState :=
  { blue := 6, red := 5, yellow := 4 }

/-- The state of the bag after drawing a blue chip -/
def bagAfterBlue : BagState :=
  { blue := 7, red := 5, yellow := 4 }

/-- The probability of drawing two chips of different colors -/
def probDifferentColors : ℚ := 593 / 900

/-- The theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors :
  let totalChips := initialBag.blue + initialBag.red + initialBag.yellow
  let probFirstBlue := initialBag.blue / totalChips
  let probFirstRed := initialBag.red / totalChips
  let probFirstYellow := initialBag.yellow / totalChips
  let probSecondNotBlueAfterBlue := (bagAfterBlue.red + bagAfterBlue.yellow) / (bagAfterBlue.blue + bagAfterBlue.red + bagAfterBlue.yellow)
  let probSecondNotRedAfterRed := (initialBag.blue + initialBag.yellow) / totalChips
  let probSecondNotYellowAfterYellow := (initialBag.blue + initialBag.red) / totalChips
  probFirstBlue * probSecondNotBlueAfterBlue +
  probFirstRed * probSecondNotRedAfterRed +
  probFirstYellow * probSecondNotYellowAfterYellow = probDifferentColors :=
by
  sorry


end NUMINAMATH_CALUDE_prob_different_colors_l1662_166274


namespace NUMINAMATH_CALUDE_custom_mult_three_two_l1662_166200

/-- Custom multiplication operation -/
def custom_mult (a b : ℤ) : ℤ := a^2 + a*b - b^2

/-- Theorem stating that 3*2 equals 11 under the custom multiplication -/
theorem custom_mult_three_two : custom_mult 3 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_three_two_l1662_166200


namespace NUMINAMATH_CALUDE_is_cylinder_l1662_166227

/-- Represents the shape of a view in an orthographic projection --/
inductive ViewShape
  | Circle
  | Rectangle

/-- Represents the three orthographic views of a solid --/
structure OrthographicViews where
  top : ViewShape
  front : ViewShape
  side : ViewShape

/-- Represents different types of solids --/
inductive Solid
  | Sphere
  | Cylinder
  | Cone
  | Cuboid

/-- Given the three orthographic views of a solid, determine if it is a cylinder --/
theorem is_cylinder (views : OrthographicViews) :
  views.top = ViewShape.Circle ∧ 
  views.front = ViewShape.Rectangle ∧ 
  views.side = ViewShape.Rectangle → 
  ∃ (s : Solid), s = Solid.Cylinder :=
sorry

end NUMINAMATH_CALUDE_is_cylinder_l1662_166227


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1662_166294

/-- Given a cylinder whose lateral surface unfolds into a square with side length 2π,
    the total surface area of the cylinder is 2π + 4π². -/
theorem cylinder_surface_area (h : ℝ) (r : ℝ) :
  h = 2 * π → 2 * π * r = 2 * π →
  2 * π * r * (r + h) = 2 * π + 4 * π^2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l1662_166294


namespace NUMINAMATH_CALUDE_min_value_sum_squares_over_sum_l1662_166253

theorem min_value_sum_squares_over_sum (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → a + b + c = 9 →
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_over_sum_l1662_166253


namespace NUMINAMATH_CALUDE_constant_term_is_99_l1662_166254

-- Define the function q'
def q' (q : ℝ) (c : ℝ) : ℝ := 3 * q - 3 + c

-- Define the condition that (5')' = 132
axiom condition : q' (q' 5 99) 99 = 132

-- Theorem to prove
theorem constant_term_is_99 : ∃ c : ℝ, q' (q' 5 c) c = 132 ∧ c = 99 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_is_99_l1662_166254


namespace NUMINAMATH_CALUDE_shareholders_profit_decrease_l1662_166240

/-- Represents the problem of calculating the percentage decrease in shareholders' profit --/
theorem shareholders_profit_decrease (total_machines : ℝ) (operational_machines : ℝ) 
  (annual_output : ℝ) (profit_percentage : ℝ) :
  total_machines = 14 →
  operational_machines = total_machines - 7.14 →
  annual_output = 70000 →
  profit_percentage = 0.125 →
  let new_output := (operational_machines / total_machines) * annual_output
  let original_profit := profit_percentage * annual_output
  let new_profit := profit_percentage * new_output
  let percentage_decrease := ((original_profit - new_profit) / original_profit) * 100
  percentage_decrease = 51 := by
sorry

end NUMINAMATH_CALUDE_shareholders_profit_decrease_l1662_166240


namespace NUMINAMATH_CALUDE_range_of_a_l1662_166211

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + 4 * x + a > 0) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1662_166211


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1662_166258

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 1 + a 5 + a 9 = 6) : 
  a 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1662_166258


namespace NUMINAMATH_CALUDE_pet_store_puppies_l1662_166278

theorem pet_store_puppies (initial_birds initial_puppies initial_cats initial_spiders : ℕ)
  (sold_birds adopted_puppies loose_spiders : ℕ) (final_total : ℕ) :
  initial_birds = 12 →
  initial_cats = 5 →
  initial_spiders = 15 →
  sold_birds = initial_birds / 2 →
  adopted_puppies = 3 →
  loose_spiders = 7 →
  final_total = 25 →
  final_total = initial_birds - sold_birds + initial_cats + 
                (initial_spiders - loose_spiders) + (initial_puppies - adopted_puppies) →
  initial_puppies = 9 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l1662_166278


namespace NUMINAMATH_CALUDE_wholesale_price_calculation_l1662_166264

theorem wholesale_price_calculation (retail_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  retail_price = 132 ∧ 
  discount_rate = 0.1 ∧ 
  profit_rate = 0.2 →
  ∃ wholesale_price : ℝ,
    wholesale_price = 99 ∧
    retail_price * (1 - discount_rate) = wholesale_price * (1 + profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_wholesale_price_calculation_l1662_166264


namespace NUMINAMATH_CALUDE_min_a5_in_geometric_sequence_l1662_166280

/-- A geometric sequence of 6 terms -/
def GeometricSequence (a : Fin 6 → ℝ) : Prop :=
  ∃ q : ℝ, ∀ i : Fin 5, a (i + 1) = a i * q

theorem min_a5_in_geometric_sequence 
  (a : Fin 6 → ℝ) 
  (h_geometric : GeometricSequence a) 
  (h_terms : ∃ i j : Fin 6, a i = 1 ∧ a j = 9) :
  ∃ a5_min : ℝ, a5_min = -27 ∧ ∀ a' : Fin 6 → ℝ, 
    GeometricSequence a' → 
    (∃ i j : Fin 6, a' i = 1 ∧ a' j = 9) → 
    a' 5 ≥ a5_min :=
sorry

end NUMINAMATH_CALUDE_min_a5_in_geometric_sequence_l1662_166280


namespace NUMINAMATH_CALUDE_number_division_problem_l1662_166247

theorem number_division_problem : ∃ x : ℝ, x / 5 = 80 + x / 6 ∧ x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1662_166247


namespace NUMINAMATH_CALUDE_sum_fourth_powers_of_roots_l1662_166270

/-- Given a cubic polynomial x^3 - x^2 + x - 3 = 0 with roots p, q, and r,
    prove that p^4 + q^4 + r^4 = 11 -/
theorem sum_fourth_powers_of_roots (p q r : ℂ) : 
  p^3 - p^2 + p - 3 = 0 → 
  q^3 - q^2 + q - 3 = 0 → 
  r^3 - r^2 + r - 3 = 0 → 
  p^4 + q^4 + r^4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_fourth_powers_of_roots_l1662_166270


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1662_166202

theorem sufficient_not_necessary (x : ℝ) : 
  (|x - 1| < 2 → x < 3) ∧ ¬(x < 3 → |x - 1| < 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1662_166202


namespace NUMINAMATH_CALUDE_protest_days_calculation_l1662_166252

/-- Calculates the number of days of protest given the conditions of the problem. -/
def daysOfProtest (
  numCities : ℕ)
  (arrestsPerDay : ℕ)
  (preTrialDays : ℕ)
  (sentenceDays : ℕ)
  (totalJailWeeks : ℕ) : ℕ :=
  let totalJailDays := totalJailWeeks * 7
  let daysPerPerson := preTrialDays + sentenceDays / 2
  let totalArrests := totalJailDays / daysPerPerson
  let totalProtestDays := totalArrests / arrestsPerDay
  totalProtestDays / numCities

/-- Theorem stating that given the conditions of the problem, there were 30 days of protest. -/
theorem protest_days_calculation :
  daysOfProtest 21 10 4 14 9900 = 30 := by
  sorry

end NUMINAMATH_CALUDE_protest_days_calculation_l1662_166252


namespace NUMINAMATH_CALUDE_expression_simplification_l1662_166256

theorem expression_simplification (m : ℝ) (h : m ≠ 2) :
  (m + 2 - 5 / (m - 2)) / ((m - 3) / (2 * m - 4)) = 2 * m + 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1662_166256


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_m_l1662_166239

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_purely_imaginary_m (m : ℝ) :
  is_purely_imaginary ((m^2 - m) + m * I) → m = 1 := by sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_m_l1662_166239


namespace NUMINAMATH_CALUDE_jump_rope_results_l1662_166293

def passing_score : ℕ := 140

def scores : List ℤ := [-25, 17, 23, 0, -39, -11, 9, 34]

def score_difference (scores : List ℤ) : ℕ :=
  (scores.maximum?.getD 0 - scores.minimum?.getD 0).toNat

def average_score (scores : List ℤ) : ℚ :=
  passing_score + (scores.sum : ℚ) / scores.length

def calculate_points (score : ℤ) : ℤ :=
  if score > 0 then 2 * score else -score

def total_score (scores : List ℤ) : ℤ :=
  scores.map calculate_points |>.sum

theorem jump_rope_results :
  score_difference scores = 73 ∧
  average_score scores = 141 ∧
  total_score scores = 91 := by
  sorry

end NUMINAMATH_CALUDE_jump_rope_results_l1662_166293


namespace NUMINAMATH_CALUDE_max_sum_constraint_l1662_166276

theorem max_sum_constraint (x y z : ℕ) (h : x + y + z = 1000) :
  11 * x * y + 3 * x + 2012 * y * z ≤ 503000000 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_constraint_l1662_166276


namespace NUMINAMATH_CALUDE_g_range_g_range_achieves_bounds_l1662_166235

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x/3))^2 + 2*Real.pi * Real.arcsin (x/3) - 3*(Real.arcsin (x/3))^2 + (Real.pi^2/4)*(x^2 - 3*x + 9)

theorem g_range :
  ∀ y ∈ Set.range g, π^2/4 ≤ y ∧ y ≤ 37*π^2/4 :=
by sorry

theorem g_range_achieves_bounds :
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3) 3 ∧ x₂ ∈ Set.Icc (-3) 3 ∧ 
    g x₁ = π^2/4 ∧ g x₂ = 37*π^2/4 :=
by sorry

end NUMINAMATH_CALUDE_g_range_g_range_achieves_bounds_l1662_166235


namespace NUMINAMATH_CALUDE_water_overflow_l1662_166212

/-- Given a tap producing water at a constant rate and a water tank with a fixed capacity,
    calculate the amount of water that overflows after a certain time. -/
theorem water_overflow (flow_rate : ℕ) (time : ℕ) (tank_capacity : ℕ) : 
  flow_rate = 200 → time = 24 → tank_capacity = 4000 → 
  flow_rate * time - tank_capacity = 800 := by
  sorry

end NUMINAMATH_CALUDE_water_overflow_l1662_166212


namespace NUMINAMATH_CALUDE_remaining_amount_is_10_95_l1662_166248

def initial_amount : ℝ := 60

def frame_price : ℝ := 15
def frame_discount : ℝ := 0.1

def wheel_price : ℝ := 25
def wheel_discount : ℝ := 0.05

def seat_price : ℝ := 8
def seat_discount : ℝ := 0.15

def tape_price : ℝ := 5
def tape_discount : ℝ := 0

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_cost : ℝ :=
  discounted_price frame_price frame_discount +
  discounted_price wheel_price wheel_discount +
  discounted_price seat_price seat_discount +
  discounted_price tape_price tape_discount

theorem remaining_amount_is_10_95 :
  initial_amount - total_cost = 10.95 :=
by sorry

end NUMINAMATH_CALUDE_remaining_amount_is_10_95_l1662_166248


namespace NUMINAMATH_CALUDE_granger_cisco_equal_spots_sum_equal_total_granger_cisco_ratio_one_to_one_l1662_166279

/-- The number of spots Granger has -/
def granger_spots : ℕ := 54

/-- The number of spots Cisco has -/
def cisco_spots : ℕ := 54

/-- The total number of spots Granger and Cisco have combined -/
def total_spots : ℕ := 108

/-- Theorem stating that Granger and Cisco have the same number of spots -/
theorem granger_cisco_equal : granger_spots = cisco_spots := by sorry

/-- Theorem stating that the sum of Granger's and Cisco's spots equals the total -/
theorem spots_sum_equal_total : granger_spots + cisco_spots = total_spots := by sorry

/-- Theorem proving that the ratio of Granger's spots to Cisco's spots is 1:1 -/
theorem granger_cisco_ratio_one_to_one : 
  granger_spots / cisco_spots = 1 := by sorry

end NUMINAMATH_CALUDE_granger_cisco_equal_spots_sum_equal_total_granger_cisco_ratio_one_to_one_l1662_166279


namespace NUMINAMATH_CALUDE_four_digit_integer_problem_l1662_166297

theorem four_digit_integer_problem (n : ℕ) (a b c d : ℕ) :
  n = a * 1000 + b * 100 + c * 10 + d →
  a ≥ 1 →
  a ≤ 9 →
  b ≤ 9 →
  c ≤ 9 →
  d ≤ 9 →
  a + b + c + d = 17 →
  b + c = 10 →
  a - d = 3 →
  n % 13 = 0 →
  n = 5732 := by sorry

end NUMINAMATH_CALUDE_four_digit_integer_problem_l1662_166297


namespace NUMINAMATH_CALUDE_solution_difference_l1662_166295

theorem solution_difference (p q : ℝ) : 
  (p - 5) * (p + 5) = 26 * p - 130 →
  (q - 5) * (q + 5) = 26 * q - 130 →
  p ≠ q →
  p > q →
  p - q = 16 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l1662_166295


namespace NUMINAMATH_CALUDE_school_students_l1662_166219

/-- The number of students in a school -/
theorem school_students (boys : ℕ) (girls : ℕ) : 
  boys = 272 → girls = boys + 106 → boys + girls = 650 := by
  sorry

end NUMINAMATH_CALUDE_school_students_l1662_166219


namespace NUMINAMATH_CALUDE_marks_garden_flowers_l1662_166296

/-- The number of flowers in Mark's garden -/
def total_flowers : ℕ := by sorry

/-- The number of yellow flowers -/
def yellow_flowers : ℕ := 10

/-- The number of purple flowers -/
def purple_flowers : ℕ := yellow_flowers + (yellow_flowers * 8 / 10)

/-- The number of green flowers -/
def green_flowers : ℕ := (yellow_flowers + purple_flowers) * 25 / 100

/-- The number of red flowers -/
def red_flowers : ℕ := (yellow_flowers + purple_flowers + green_flowers) * 35 / 100

theorem marks_garden_flowers :
  total_flowers = yellow_flowers + purple_flowers + green_flowers + red_flowers ∧
  total_flowers = 47 := by sorry

end NUMINAMATH_CALUDE_marks_garden_flowers_l1662_166296


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l1662_166210

/-- Percentage of employees who attended the company picnic -/
def picnic_attendance (men_percentage : Real) (women_percentage : Real) 
  (men_attendance : Real) (women_attendance : Real) : Real :=
  men_percentage * men_attendance + (1 - men_percentage) * women_attendance

theorem company_picnic_attendance :
  picnic_attendance 0.45 0.55 0.20 0.40 = 0.31 := by
  sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l1662_166210


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l1662_166271

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 6 * Real.sqrt 3) + Real.sqrt (12 - 6 * Real.sqrt 3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l1662_166271


namespace NUMINAMATH_CALUDE_total_time_is_186_l1662_166242

def total_time (mac_download : ℕ) (windows_multiplier : ℕ)
  (ny_audio_glitch_count ny_audio_glitch_duration : ℕ)
  (ny_video_glitch_count ny_video_glitch_duration : ℕ)
  (ny_unglitched_multiplier : ℕ)
  (berlin_audio_glitch_count berlin_audio_glitch_duration : ℕ)
  (berlin_video_glitch_count berlin_video_glitch_duration : ℕ)
  (berlin_unglitched_multiplier : ℕ) : ℕ :=
  let windows_download := mac_download * windows_multiplier
  let total_download := mac_download + windows_download

  let ny_audio_glitch := ny_audio_glitch_count * ny_audio_glitch_duration
  let ny_video_glitch := ny_video_glitch_count * ny_video_glitch_duration
  let ny_total_glitch := ny_audio_glitch + ny_video_glitch
  let ny_unglitched := ny_total_glitch * ny_unglitched_multiplier
  let ny_total := ny_total_glitch + ny_unglitched

  let berlin_audio_glitch := berlin_audio_glitch_count * berlin_audio_glitch_duration
  let berlin_video_glitch := berlin_video_glitch_count * berlin_video_glitch_duration
  let berlin_total_glitch := berlin_audio_glitch + berlin_video_glitch
  let berlin_unglitched := berlin_total_glitch * berlin_unglitched_multiplier
  let berlin_total := berlin_total_glitch + berlin_unglitched

  total_download + ny_total + berlin_total

theorem total_time_is_186 :
  total_time 10 3 2 6 1 8 3 3 4 2 5 2 = 186 := by sorry

end NUMINAMATH_CALUDE_total_time_is_186_l1662_166242


namespace NUMINAMATH_CALUDE_total_triangles_is_seventeen_l1662_166224

/-- Represents a 2x2 square grid where each square is divided diagonally into two right-angled triangles -/
structure DiagonallyDividedGrid :=
  (size : Nat)
  (is_two_by_two : size = 2)
  (diagonally_divided : Bool)

/-- Counts the total number of triangles in the grid, including all possible combinations -/
def count_triangles (grid : DiagonallyDividedGrid) : Nat :=
  sorry

/-- Theorem stating that the total number of triangles in the described grid is 17 -/
theorem total_triangles_is_seventeen (grid : DiagonallyDividedGrid) 
  (h1 : grid.diagonally_divided = true) : 
  count_triangles grid = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_seventeen_l1662_166224


namespace NUMINAMATH_CALUDE_inequality_solution_l1662_166214

theorem inequality_solution (x : ℝ) : 
  (2 / (x - 2) - 5 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ 
  (x < 2 ∨ (3 < x ∧ x < 4) ∨ 5 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1662_166214


namespace NUMINAMATH_CALUDE_first_watermelon_weight_l1662_166244

theorem first_watermelon_weight (total_weight second_weight : ℝ) 
  (h1 : total_weight = 14.02)
  (h2 : second_weight = 4.11) :
  total_weight - second_weight = 9.91 := by
  sorry

end NUMINAMATH_CALUDE_first_watermelon_weight_l1662_166244
