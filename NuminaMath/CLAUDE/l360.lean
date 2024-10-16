import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_coefficients_l360_36030

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^9 = a₀ + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + 
    a₄*(x - 1)^4 + a₅*(x - 1)^5 + a₆*(x - 1)^6 + a₇*(x - 1)^7 + a₈*(x - 1)^8 + a₉*(x - 1)^9) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l360_36030


namespace NUMINAMATH_CALUDE_fraction_subtraction_property_l360_36014

theorem fraction_subtraction_property (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b - c / d = (a - c) / (b + d)) ↔ (a / c = (b / d)^2) :=
sorry

end NUMINAMATH_CALUDE_fraction_subtraction_property_l360_36014


namespace NUMINAMATH_CALUDE_flower_bed_fraction_is_one_sixth_l360_36021

/-- Represents a rectangular yard with flower beds -/
structure YardWithFlowerBeds where
  yard_width : ℝ
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ

/-- The fraction of the yard occupied by the flower beds -/
def flower_bed_fraction (y : YardWithFlowerBeds) : ℚ :=
  1/6

/-- Theorem stating that the fraction of the yard occupied by the flower beds is 1/6 -/
theorem flower_bed_fraction_is_one_sixth (y : YardWithFlowerBeds)
    (h1 : y.trapezoid_short_side = 20)
    (h2 : y.trapezoid_long_side = 30) :
    flower_bed_fraction y = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_is_one_sixth_l360_36021


namespace NUMINAMATH_CALUDE_meteorological_satellite_requires_comprehensive_survey_l360_36065

/-- Represents a type of survey --/
inductive SurveyType
| Comprehensive
| Sampling

/-- Represents a scenario for data collection --/
structure DataCollectionScenario where
  description : String
  requiredSurveyType : SurveyType

/-- Represents a component of a meteorological satellite --/
structure SatelliteComponent where
  id : Nat
  quality : Bool  -- True if the component meets quality standards, False otherwise

/-- Definition of a meteorological satellite --/
structure MeteorologicalSatellite where
  components : List SatelliteComponent

/-- Function to determine if a satellite is functional --/
def isSatelliteFunctional (satellite : MeteorologicalSatellite) : Bool :=
  satellite.components.all (fun c => c.quality)

/-- Theorem stating that the quality assessment of meteorological satellite components requires a comprehensive survey --/
theorem meteorological_satellite_requires_comprehensive_survey 
  (scenario : DataCollectionScenario) 
  (h1 : scenario.description = "The quality of components of a meteorological satellite about to be launched") :
  scenario.requiredSurveyType = SurveyType.Comprehensive :=
by
  sorry


end NUMINAMATH_CALUDE_meteorological_satellite_requires_comprehensive_survey_l360_36065


namespace NUMINAMATH_CALUDE_tuesday_flower_sales_ratio_l360_36002

/-- Represents the number of flowers sold -/
structure FlowerSales where
  roses : ℕ
  lilacs : ℕ
  gardenias : ℕ

/-- Represents the ratio of two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of roses to lilacs -/
def roseToLilacRatio (sales : FlowerSales) : Ratio :=
  { numerator := sales.roses, denominator := sales.lilacs }

theorem tuesday_flower_sales_ratio : 
  ∀ (sales : FlowerSales), 
    sales.lilacs = 10 →
    sales.gardenias = sales.lilacs / 2 →
    sales.roses + sales.lilacs + sales.gardenias = 45 →
    (roseToLilacRatio sales).numerator = 3 ∧ (roseToLilacRatio sales).denominator = 1 := by
  sorry


end NUMINAMATH_CALUDE_tuesday_flower_sales_ratio_l360_36002


namespace NUMINAMATH_CALUDE_least_possible_difference_l360_36076

theorem least_possible_difference (x y z : ℤ) : 
  Even x → Odd y → Odd z → x < y → y < z → y - x > 5 → 
  ∀ (s : ℤ), z - x ≥ s → s ≥ 9 :=
by
  sorry

end NUMINAMATH_CALUDE_least_possible_difference_l360_36076


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l360_36050

def polynomial_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) : ℕ → ℤ
  | 0 => a₀
  | 1 => a₁
  | 2 => a₂
  | 3 => a₃
  | 4 => a₄
  | 5 => a₅
  | 6 => a₆
  | 7 => a₇
  | _ => 0

theorem sum_of_even_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ (x : ℤ), (3*x - 1)^7 = 
    a₀*x^7 + a₁*x^6 + a₂*x^5 + a₃*x^4 + a₄*x^3 + a₅*x^2 + a₆*x + a₇) →
  a₀ + a₂ + a₄ + a₆ = 4128 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l360_36050


namespace NUMINAMATH_CALUDE_joan_attended_395_games_l360_36039

/-- The number of baseball games Joan attended -/
def games_attended (total_games missed_games : ℕ) : ℕ :=
  total_games - missed_games

/-- Proof that Joan attended 395 baseball games -/
theorem joan_attended_395_games (total_games night_games missed_games : ℕ) 
  (h1 : total_games = 864)
  (h2 : night_games = 128)
  (h3 : missed_games = 469) :
  games_attended total_games missed_games = 395 := by
  sorry

#eval games_attended 864 469

end NUMINAMATH_CALUDE_joan_attended_395_games_l360_36039


namespace NUMINAMATH_CALUDE_solve_for_a_l360_36028

-- Define the operation *
def star (a b : ℝ) : ℝ := 2 * a - b^2

-- Theorem statement
theorem solve_for_a : ∃ (a : ℝ), star a 3 = 7 ∧ a = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l360_36028


namespace NUMINAMATH_CALUDE_only_vegetarian_count_l360_36090

/-- Represents the number of people in a family with different eating habits -/
structure FamilyEatingHabits where
  only_non_veg : ℕ
  both_veg_and_non_veg : ℕ
  total_veg : ℕ

/-- Theorem stating the number of people who eat only vegetarian -/
theorem only_vegetarian_count (f : FamilyEatingHabits) 
  (h1 : f.only_non_veg = 6)
  (h2 : f.both_veg_and_non_veg = 9)
  (h3 : f.total_veg = 20) :
  f.total_veg - f.both_veg_and_non_veg = 11 := by
  sorry

end NUMINAMATH_CALUDE_only_vegetarian_count_l360_36090


namespace NUMINAMATH_CALUDE_grid_exists_l360_36084

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_primes (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ ∀ r, is_prime r → r ≤ p ∨ r ≥ q

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem grid_exists : ∃ (a b c d e f g h : ℕ),
  (∀ x ∈ [a, b, c, d, e, f, g, h], x > 0 ∧ x < 10) ∧
  a ≠ 0 ∧ e ≠ 0 ∧
  (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ 1000 * a + 100 * b + 10 * c + d = p^q) ∧
  (∃ (p q : ℕ), consecutive_primes p q ∧ 1000 * e + 100 * f + 10 * c + d = p * q) ∧
  is_perfect_square (1000 * e + 100 * e + 10 * g + g) ∧
  is_multiple_of (1000 * e + 100 * h + 10 * g + g) 37 ∧
  1000 * a + 100 * b + 10 * c + d = 2187 ∧
  1000 * e + 100 * f + 10 * c + d = 7387 ∧
  1000 * e + 100 * e + 10 * g + g = 7744 ∧
  1000 * e + 100 * h + 10 * g + g = 7744 :=
by
  sorry


end NUMINAMATH_CALUDE_grid_exists_l360_36084


namespace NUMINAMATH_CALUDE_circumcenter_rational_l360_36026

-- Define a triangle with rational coordinates
structure RationalTriangle where
  a : ℚ × ℚ
  b : ℚ × ℚ
  c : ℚ × ℚ

-- Define the center of the circumscribed circle
def circumcenter (t : RationalTriangle) : ℚ × ℚ :=
  sorry

-- Theorem statement
theorem circumcenter_rational (t : RationalTriangle) :
  ∃ (x y : ℚ), circumcenter t = (x, y) :=
sorry

end NUMINAMATH_CALUDE_circumcenter_rational_l360_36026


namespace NUMINAMATH_CALUDE_sum_product_reciprocals_inequality_l360_36054

theorem sum_product_reciprocals_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (a + b + c) * (1/a + 1/b + 1/c) > 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_reciprocals_inequality_l360_36054


namespace NUMINAMATH_CALUDE_age_difference_l360_36068

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 17) : a = c + 17 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l360_36068


namespace NUMINAMATH_CALUDE_num_paths_5x4_grid_l360_36019

/-- The number of paths on a grid from point C to point D -/
def num_paths (grid_width grid_height path_length right_steps up_steps : ℕ) : ℕ :=
  Nat.choose path_length up_steps

/-- Theorem stating the number of paths on a 5x4 grid with specific constraints -/
theorem num_paths_5x4_grid : num_paths 5 4 8 5 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_num_paths_5x4_grid_l360_36019


namespace NUMINAMATH_CALUDE_diego_orange_weight_l360_36001

/-- Given Diego's fruit purchases, prove the weight of oranges he bought. -/
theorem diego_orange_weight (total_capacity : ℕ) (watermelon_weight : ℕ) (grape_weight : ℕ) (apple_weight : ℕ) 
  (h1 : total_capacity = 20)
  (h2 : watermelon_weight = 1)
  (h3 : grape_weight = 1)
  (h4 : apple_weight = 17) :
  total_capacity - (watermelon_weight + grape_weight + apple_weight) = 1 := by
  sorry

end NUMINAMATH_CALUDE_diego_orange_weight_l360_36001


namespace NUMINAMATH_CALUDE_trapezoid_segment_equality_l360_36007

-- Define the points
variable (A B C D M N : Point)

-- Define the trapezoid
def is_trapezoid (A B C D : Point) : Prop := sorry

-- Define that M is on CD and N is on AB
def point_on_segment (P Q R : Point) : Prop := sorry

-- Define angle equality
def angle_eq (P Q R S T U : Point) : Prop := sorry

-- Define segment equality
def segment_eq (P Q R S : Point) : Prop := sorry

-- Theorem statement
theorem trapezoid_segment_equality 
  (h1 : is_trapezoid A B C D)
  (h2 : point_on_segment C D M)
  (h3 : point_on_segment A B N)
  (h4 : segment_eq A N B N)
  (h5 : angle_eq A B N C D M) :
  segment_eq C M M D :=
sorry

end NUMINAMATH_CALUDE_trapezoid_segment_equality_l360_36007


namespace NUMINAMATH_CALUDE_min_height_box_l360_36075

/-- Represents a rectangular box with a square base -/
structure Box where
  base : ℝ
  height : ℝ

/-- Calculates the surface area of a box -/
def surfaceArea (b : Box) : ℝ :=
  2 * b.base^2 + 4 * b.base * b.height

/-- Theorem stating the minimum height of the box under given conditions -/
theorem min_height_box :
  ∀ (b : Box),
    b.height = b.base + 4 →
    surfaceArea b ≥ 150 →
    ∀ (b' : Box),
      b'.height = b'.base + 4 →
      surfaceArea b' ≥ 150 →
      b.height ≤ b'.height →
      b.height = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_height_box_l360_36075


namespace NUMINAMATH_CALUDE_ellipse_sum_l360_36087

theorem ellipse_sum (h k a b : ℝ) : 
  (∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) →  -- Ellipse equation
  (h = 3 ∧ k = -5) →                                   -- Center at (3, -5)
  (a = 7 ∨ b = 7) →                                    -- Semi-major axis is 7
  (a = 2 ∨ b = 2) →                                    -- Semi-minor axis is 2
  (a > b) →                                            -- Ensure a is semi-major axis
  h + k + a + b = 7 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_sum_l360_36087


namespace NUMINAMATH_CALUDE_trig_identities_l360_36011

theorem trig_identities (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 ∧
  Real.sin α ^ 2 + Real.sin α * Real.cos α - 2 * Real.cos α ^ 2 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l360_36011


namespace NUMINAMATH_CALUDE_ingrid_income_proof_l360_36067

/-- The annual income of John in dollars -/
def john_income : ℝ := 56000

/-- The tax rate for John as a decimal -/
def john_tax_rate : ℝ := 0.30

/-- The tax rate for Ingrid as a decimal -/
def ingrid_tax_rate : ℝ := 0.40

/-- The combined tax rate for John and Ingrid as a decimal -/
def combined_tax_rate : ℝ := 0.3569

/-- Ingrid's income in dollars -/
def ingrid_income : ℝ := 73924.13

/-- Theorem stating that given the conditions, Ingrid's income is correct -/
theorem ingrid_income_proof :
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) = combined_tax_rate :=
by sorry

end NUMINAMATH_CALUDE_ingrid_income_proof_l360_36067


namespace NUMINAMATH_CALUDE_evaluate_expression_l360_36031

theorem evaluate_expression (x : ℝ) (h : x = 3) : x - x * (x ^ x) = -78 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l360_36031


namespace NUMINAMATH_CALUDE_ping_pong_theorem_l360_36071

/-- Represents the number of ping-pong balls in the box -/
def total_balls : ℕ := 7

/-- Represents the number of unused balls initially -/
def initial_unused : ℕ := 5

/-- Represents the number of used balls initially -/
def initial_used : ℕ := 2

/-- Represents the number of balls taken out and used -/
def balls_taken : ℕ := 3

/-- Represents the set of possible values for X (number of used balls after the process) -/
def possible_X : Set ℕ := {3, 4, 5}

/-- Represents the probability of X being 3 -/
def prob_X_3 : ℚ := 1/7

theorem ping_pong_theorem :
  (∀ x : ℕ, x ∈ possible_X ↔ (x ≥ initial_used ∧ x ≤ initial_used + balls_taken)) ∧
  (Nat.choose initial_unused 1 * Nat.choose initial_used 2 : ℚ) / Nat.choose total_balls balls_taken = prob_X_3 :=
by sorry

end NUMINAMATH_CALUDE_ping_pong_theorem_l360_36071


namespace NUMINAMATH_CALUDE_friendship_subset_exists_l360_36063

/-- Represents a friendship relation between students -/
def FriendshipRelation (S : Type) := S → S → Prop

/-- A school is valid if it satisfies the friendship condition -/
def ValidSchool (S : Type) (friendship : FriendshipRelation S) (students : Finset S) : Prop :=
  ∀ s ∈ students, ∃ t ∈ students, s ≠ t ∧ friendship s t

theorem friendship_subset_exists 
  (S : Type) 
  (friendship : FriendshipRelation S) 
  (students : Finset S) 
  (h_valid : ValidSchool S friendship students)
  (h_count : students.card = 101) :
  ∀ n : ℕ, 1 < n → n < 101 → 
    ∃ subset : Finset S, subset.card = n ∧ subset ⊆ students ∧
      ∀ s ∈ subset, ∃ t ∈ subset, s ≠ t ∧ friendship s t :=
by
  sorry


end NUMINAMATH_CALUDE_friendship_subset_exists_l360_36063


namespace NUMINAMATH_CALUDE_junior_boy_girl_ratio_l360_36057

/-- Represents the number of participants in each category -/
structure Participants where
  juniorBoys : ℕ
  seniorBoys : ℕ
  juniorGirls : ℕ
  seniorGirls : ℕ

/-- The ratio of boys to total participants is 55% -/
def boyRatio (p : Participants) : Prop :=
  (p.juniorBoys + p.seniorBoys : ℚ) / (p.juniorBoys + p.seniorBoys + p.juniorGirls + p.seniorGirls) = 55 / 100

/-- The ratio of junior boys to senior boys equals the ratio of all juniors to all seniors -/
def juniorSeniorRatio (p : Participants) : Prop :=
  (p.juniorBoys : ℚ) / p.seniorBoys = (p.juniorBoys + p.juniorGirls : ℚ) / (p.seniorBoys + p.seniorGirls)

/-- The main theorem: given the conditions, prove that the ratio of junior boys to junior girls is 11:9 -/
theorem junior_boy_girl_ratio (p : Participants) 
  (hBoyRatio : boyRatio p) (hJuniorSeniorRatio : juniorSeniorRatio p) : 
  (p.juniorBoys : ℚ) / p.juniorGirls = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_junior_boy_girl_ratio_l360_36057


namespace NUMINAMATH_CALUDE_saturday_ice_cream_amount_l360_36018

/-- The amount of ice cream eaten on Saturday night, given the amount eaten on Friday and the total amount eaten over both nights. -/
def ice_cream_saturday (friday : ℝ) (total : ℝ) : ℝ :=
  total - friday

theorem saturday_ice_cream_amount :
  ice_cream_saturday 3.25 3.5 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_saturday_ice_cream_amount_l360_36018


namespace NUMINAMATH_CALUDE_inequality_solution_l360_36055

theorem inequality_solution (x : ℝ) :
  x ≠ 5 → (x * (x - 2) / ((x - 5)^2) ≥ 15 ↔ x ∈ Set.Iio 5 ∪ Set.Ioi 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l360_36055


namespace NUMINAMATH_CALUDE_smallest_with_twenty_divisors_l360_36059

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer has exactly 20 positive divisors -/
def has_twenty_divisors (n : ℕ+) : Prop := num_divisors n = 20

theorem smallest_with_twenty_divisors : 
  (∀ m : ℕ+, m < 432 → ¬(has_twenty_divisors m)) ∧ has_twenty_divisors 432 := by sorry

end NUMINAMATH_CALUDE_smallest_with_twenty_divisors_l360_36059


namespace NUMINAMATH_CALUDE_opposite_of_A_is_F_l360_36056

-- Define the labels for the cube faces
inductive CubeFace
  | A | B | C | D | E | F

-- Define a structure for the cube
structure Cube where
  faces : Finset CubeFace
  opposite : CubeFace → CubeFace

-- Define the properties of the cube
axiom cube_has_six_faces : ∀ (c : Cube), c.faces.card = 6

axiom cube_has_unique_opposite : ∀ (c : Cube) (f : CubeFace), 
  f ∈ c.faces → c.opposite f ∈ c.faces ∧ c.opposite (c.opposite f) = f

axiom cube_opposite_distinct : ∀ (c : Cube) (f : CubeFace), 
  f ∈ c.faces → c.opposite f ≠ f

-- Theorem to prove
theorem opposite_of_A_is_F (c : Cube) : 
  CubeFace.A ∈ c.faces → c.opposite CubeFace.A = CubeFace.F := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_A_is_F_l360_36056


namespace NUMINAMATH_CALUDE_six_by_six_grid_squares_l360_36098

/-- The number of squares of size n×n in a 6×6 grid -/
def squaresOfSize (n : ℕ) : ℕ := (6 - n) * (6 - n)

/-- The total number of squares in a 6×6 grid -/
def totalSquares : ℕ :=
  squaresOfSize 1 + squaresOfSize 2 + squaresOfSize 3 + squaresOfSize 4

theorem six_by_six_grid_squares :
  totalSquares = 54 := by
  sorry

end NUMINAMATH_CALUDE_six_by_six_grid_squares_l360_36098


namespace NUMINAMATH_CALUDE_basketball_team_selection_l360_36043

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := sorry

theorem basketball_team_selection :
  let total_players : ℕ := 16
  let quadruplets : ℕ := 4
  let starters : ℕ := 7
  let quadruplets_in_lineup : ℕ := 3
  
  (choose quadruplets quadruplets_in_lineup) * 
  (choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 1980 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l360_36043


namespace NUMINAMATH_CALUDE_binomial_sixteen_nine_l360_36009

theorem binomial_sixteen_nine (h1 : Nat.choose 15 7 = 6435)
                              (h2 : Nat.choose 15 8 = 6435)
                              (h3 : Nat.choose 17 9 = 24310) :
  Nat.choose 16 9 = 11440 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sixteen_nine_l360_36009


namespace NUMINAMATH_CALUDE_e_opposite_x_l360_36040

/-- Represents the faces of a cube --/
inductive Face : Type
  | X | A | B | C | D | E

/-- Represents the net of a cube --/
structure CubeNet where
  central : Face
  left : Face
  right : Face
  bottom : Face
  connected_to_right : Face
  connected_to_left : Face

/-- Defines the specific cube net given in the problem --/
def given_net : CubeNet :=
  { central := Face.X
  , left := Face.A
  , right := Face.B
  , bottom := Face.D
  , connected_to_right := Face.C
  , connected_to_left := Face.E
  }

/-- Defines the concept of opposite faces in a cube --/
def opposite (f1 f2 : Face) : Prop := sorry

/-- Theorem stating that in the given net, E is opposite to X --/
theorem e_opposite_x (net : CubeNet) : 
  net = given_net → opposite Face.E Face.X :=
sorry

end NUMINAMATH_CALUDE_e_opposite_x_l360_36040


namespace NUMINAMATH_CALUDE_special_function_value_l360_36069

/-- A function satisfying certain properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 4) ≤ f x + 4) ∧
  (∀ x : ℝ, f (x + 2) ≥ f x + 2) ∧
  (f 1 = 0)

/-- Theorem stating the value of f(2013) for a special function f -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) : f 2013 = 2012 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l360_36069


namespace NUMINAMATH_CALUDE_income_calculation_l360_36060

/-- Calculates a person's income given the income to expenditure ratio and savings amount. -/
def calculate_income (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) : ℕ :=
  (income_ratio * savings) / (income_ratio - expenditure_ratio)

/-- Proves that given the specified conditions, the person's income is 18000. -/
theorem income_calculation :
  let income_ratio : ℕ := 9
  let expenditure_ratio : ℕ := 8
  let savings : ℕ := 2000
  calculate_income income_ratio expenditure_ratio savings = 18000 := by
  sorry

#eval calculate_income 9 8 2000

end NUMINAMATH_CALUDE_income_calculation_l360_36060


namespace NUMINAMATH_CALUDE_four_digit_sum_l360_36029

/-- Given four distinct non-zero digits, the sum of all possible four-digit numbers formed using these digits without repetition is 73,326 if and only if the digits are 1, 2, 3, and 5. -/
theorem four_digit_sum (a b c d : ℕ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
  6 * (a + b + c + d) * 1111 = 73326 →
  ({a, b, c, d} : Set ℕ) = {1, 2, 3, 5} :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_l360_36029


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l360_36034

theorem polynomial_division_theorem (x : ℝ) : 
  12 * x^3 + 18 * x^2 + 27 * x + 17 = 
  (4 * x + 3) * (3 * x^2 + 2.25 * x + 5/16) + 29/16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l360_36034


namespace NUMINAMATH_CALUDE_complex_equation_solution_l360_36005

theorem complex_equation_solution (z : ℂ) :
  (1 - Complex.I)^2 / z = 1 + Complex.I → z = -1 - Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l360_36005


namespace NUMINAMATH_CALUDE_blue_marbles_count_l360_36095

theorem blue_marbles_count (total : ℕ) (red : ℕ) (prob_red_or_white : ℚ) 
  (h1 : total = 20)
  (h2 : red = 9)
  (h3 : prob_red_or_white = 3/4) :
  ∃ blue : ℕ, blue = 5 ∧ 
    (blue + red : ℚ) / total + prob_red_or_white = 1 := by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l360_36095


namespace NUMINAMATH_CALUDE_units_digit_problem_l360_36033

theorem units_digit_problem : (7 * 13 * 1957 - 7^4) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l360_36033


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_parabola_standard_equation_l360_36073

-- Ellipse
def ellipse_equation (x y : ℝ) := x^2 / 25 + y^2 / 9 = 1

theorem ellipse_standard_equation 
  (foci_on_x_axis : Bool) 
  (major_axis_length : ℝ) 
  (eccentricity : ℝ) :
  foci_on_x_axis ∧ 
  major_axis_length = 10 ∧ 
  eccentricity = 4/5 →
  ∀ x y : ℝ, ellipse_equation x y :=
sorry

-- Parabola
def parabola_equation (x y : ℝ) := x^2 = -8*y

theorem parabola_standard_equation 
  (vertex : ℝ × ℝ) 
  (directrix : ℝ → ℝ) :
  vertex = (0, 0) ∧ 
  (∀ x : ℝ, directrix x = 2) →
  ∀ x y : ℝ, parabola_equation x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_parabola_standard_equation_l360_36073


namespace NUMINAMATH_CALUDE_candy_distribution_l360_36064

theorem candy_distribution (e : ℚ) 
  (frank_candies : ℚ) (gail_candies : ℚ) (hank_candies : ℚ) : 
  frank_candies = 4 * e →
  gail_candies = 4 * frank_candies →
  hank_candies = 6 * gail_candies →
  e + frank_candies + gail_candies + hank_candies = 876 →
  e = 7.5 := by
sorry


end NUMINAMATH_CALUDE_candy_distribution_l360_36064


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l360_36058

/-- Represents the distance on a map in centimeters -/
def map_distance : ℝ := 65

/-- Represents the scale factor of the map (km per cm) -/
def scale_factor : ℝ := 20

/-- Calculates the actual distance in kilometers given the map distance and scale factor -/
def actual_distance (map_dist : ℝ) (scale : ℝ) : ℝ := map_dist * scale

theorem stockholm_uppsala_distance :
  actual_distance map_distance scale_factor = 1300 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l360_36058


namespace NUMINAMATH_CALUDE_infinite_square_divisibility_l360_36088

def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | (n + 4) => 2 * a (n + 3) + a (n + 2) - 2 * a (n + 1) - a n

theorem infinite_square_divisibility :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (n : ℤ)^2 ∣ a n := by sorry

end NUMINAMATH_CALUDE_infinite_square_divisibility_l360_36088


namespace NUMINAMATH_CALUDE_fourth_power_sum_equals_51_to_fourth_l360_36080

theorem fourth_power_sum_equals_51_to_fourth : 
  ∃! (n : ℕ+), 50^4 + 43^4 + 36^4 + 6^4 = n^4 :=
by sorry

end NUMINAMATH_CALUDE_fourth_power_sum_equals_51_to_fourth_l360_36080


namespace NUMINAMATH_CALUDE_ant_growth_rate_l360_36022

/-- The growth rate of an ant population over 5 hours -/
theorem ant_growth_rate (initial_population final_population : ℝ) 
  (h1 : initial_population = 50)
  (h2 : final_population = 1600)
  (h3 : final_population = initial_population * (growth_rate ^ 5)) :
  growth_rate = (32 : ℝ) ^ (1/5) :=
by sorry

end NUMINAMATH_CALUDE_ant_growth_rate_l360_36022


namespace NUMINAMATH_CALUDE_min_value_theorem_l360_36081

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 26 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ 2 / a₀ + 3 / b₀ = 26 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l360_36081


namespace NUMINAMATH_CALUDE_max_discount_rate_l360_36049

theorem max_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) : 
  cost_price = 4 →
  selling_price = 5 →
  min_profit_margin = 0.1 →
  ∃ (max_discount : ℝ),
    max_discount = 12 ∧
    ∀ (discount : ℝ),
      0 ≤ discount →
      discount ≤ max_discount →
      selling_price * (1 - discount / 100) - cost_price ≥ min_profit_margin * cost_price ∧
      ∀ (other_discount : ℝ),
        other_discount > max_discount →
        selling_price * (1 - other_discount / 100) - cost_price < min_profit_margin * cost_price :=
by sorry

end NUMINAMATH_CALUDE_max_discount_rate_l360_36049


namespace NUMINAMATH_CALUDE_distribute_six_balls_two_boxes_limit_four_l360_36053

/-- The number of ways to distribute n distinguishable balls into 2 indistinguishable boxes,
    where no box can hold more than m balls. -/
def distributeWithLimit (n : ℕ) (m : ℕ) : ℕ := sorry

/-- The theorem stating that there are 25 ways to distribute 6 distinguishable balls
    into 2 indistinguishable boxes, where no box can hold more than 4 balls. -/
theorem distribute_six_balls_two_boxes_limit_four :
  distributeWithLimit 6 4 = 25 := by sorry

end NUMINAMATH_CALUDE_distribute_six_balls_two_boxes_limit_four_l360_36053


namespace NUMINAMATH_CALUDE_michael_truck_meet_once_l360_36010

/-- Represents the meeting of Michael and the truck -/
structure Meeting where
  time : ℝ
  position : ℝ

/-- Represents the problem setup -/
structure Setup where
  michael_speed : ℝ
  pail_spacing : ℝ
  truck_speed : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the truck -/
def count_meetings (s : Setup) : ℕ :=
  sorry

/-- The main theorem stating that Michael and the truck meet exactly once -/
theorem michael_truck_meet_once (s : Setup) 
  (h1 : s.michael_speed = 6)
  (h2 : s.pail_spacing = 300)
  (h3 : s.truck_speed = 15)
  (h4 : s.truck_stop_time = 45)
  (h5 : s.initial_distance = 300) : 
  count_meetings s = 1 := by
  sorry

end NUMINAMATH_CALUDE_michael_truck_meet_once_l360_36010


namespace NUMINAMATH_CALUDE_fraction_product_l360_36091

theorem fraction_product : (2/3 : ℚ) * (5/11 : ℚ) * (3/8 : ℚ) = (5/44 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l360_36091


namespace NUMINAMATH_CALUDE_matrix_power_500_l360_36045

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; -1, 1]

theorem matrix_power_500 : A ^ 500 = !![1, 0; -500, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_500_l360_36045


namespace NUMINAMATH_CALUDE_average_increase_l360_36041

theorem average_increase (initial_average : ℝ) (fourth_test_score : ℝ) :
  initial_average = 81 ∧ fourth_test_score = 89 →
  (3 * initial_average + fourth_test_score) / 4 = initial_average + 2 := by
  sorry

end NUMINAMATH_CALUDE_average_increase_l360_36041


namespace NUMINAMATH_CALUDE_exists_starting_station_l360_36096

/-- Represents a gasoline station with its fuel amount -/
structure GasStation where
  fuel : ℝ

/-- Represents a circular highway with gasoline stations -/
structure CircularHighway where
  stations : List GasStation
  length : ℝ
  h_positive_length : length > 0

/-- The total fuel available in all stations -/
def total_fuel (highway : CircularHighway) : ℝ :=
  (highway.stations.map (fun s => s.fuel)).sum

/-- Checks if it's possible to complete a lap starting from a given station index -/
def can_complete_lap (highway : CircularHighway) (start_index : ℕ) : Prop :=
  ∃ (direction : Bool), 
    let station_sequence := if direction then 
        highway.stations.rotateLeft start_index
      else 
        (highway.stations.rotateLeft start_index).reverse
    station_sequence.foldl 
      (fun (acc : ℝ) (station : GasStation) => 
        acc + station.fuel - (highway.length / highway.stations.length))
      0 
    ≥ 0

/-- The main theorem to be proved -/
theorem exists_starting_station (highway : CircularHighway) 
  (h_fuel : total_fuel highway = 2 * highway.length) :
  ∃ (i : ℕ), i < highway.stations.length ∧ can_complete_lap highway i :=
sorry

end NUMINAMATH_CALUDE_exists_starting_station_l360_36096


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_18_24_30_l360_36079

def gcd3 (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm3 (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem gcd_lcm_sum_18_24_30 : 
  gcd3 18 24 30 + lcm3 18 24 30 = 366 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_18_24_30_l360_36079


namespace NUMINAMATH_CALUDE_fifth_root_of_unity_sum_l360_36013

theorem fifth_root_of_unity_sum (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^12 + ω^15 + ω^18 + ω^21 + ω^24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_unity_sum_l360_36013


namespace NUMINAMATH_CALUDE_cube_remainder_mod_nine_l360_36070

theorem cube_remainder_mod_nine (n : ℤ) :
  (n % 9 = 2 ∨ n % 9 = 5 ∨ n % 9 = 8) → n^3 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_remainder_mod_nine_l360_36070


namespace NUMINAMATH_CALUDE_relationship_abc_l360_36086

theorem relationship_abc : 
  let a : ℝ := Real.rpow 0.6 0.6
  let b : ℝ := Real.rpow 0.6 1.5
  let c : ℝ := Real.rpow 1.5 0.6
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l360_36086


namespace NUMINAMATH_CALUDE_petes_age_relation_l360_36052

/-- 
Given Pete's current age and his son's current age, 
this theorem proves how many years it will take for Pete 
to be exactly three times older than his son.
-/
theorem petes_age_relation (pete_age son_age : ℕ) 
  (h1 : pete_age = 35) (h2 : son_age = 9) : 
  ∃ (years : ℕ), pete_age + years = 3 * (son_age + years) ∧ years = 4 := by
  sorry

end NUMINAMATH_CALUDE_petes_age_relation_l360_36052


namespace NUMINAMATH_CALUDE_sqrt_3_minus_pi_squared_l360_36015

theorem sqrt_3_minus_pi_squared (π : ℝ) (h : π > 3) : 
  Real.sqrt ((3 - π)^2) = π - 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_minus_pi_squared_l360_36015


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l360_36042

theorem nested_fraction_equality : 1 + (1 / (1 + (1 / (1 + (1 / 2))))) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l360_36042


namespace NUMINAMATH_CALUDE_sqrt_real_range_l360_36006

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_real_range_l360_36006


namespace NUMINAMATH_CALUDE_paths_from_C_to_D_l360_36051

/-- The number of paths on a grid from (0,0) to (m,n) where only right and up moves are allowed -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- The dimensions of the grid -/
def gridWidth : ℕ := 7
def gridHeight : ℕ := 9

/-- The theorem stating the number of paths from C to D -/
theorem paths_from_C_to_D : gridPaths gridWidth gridHeight = 11440 := by
  sorry

end NUMINAMATH_CALUDE_paths_from_C_to_D_l360_36051


namespace NUMINAMATH_CALUDE_sampling_method_l360_36066

/-- Represents a bag of milk powder with a three-digit number -/
def BagNumber := Fin 800

/-- The random number table row -/
def RandomRow : List ℕ := [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79]

/-- Selects valid numbers from the random row -/
def selectValidNumbers (row : List ℕ) : List BagNumber := sorry

/-- The sampling method -/
theorem sampling_method (randomRow : List ℕ) :
  randomRow = RandomRow →
  (selectValidNumbers randomRow).take 5 = [⟨785, sorry⟩, ⟨567, sorry⟩, ⟨199, sorry⟩, ⟨507, sorry⟩, ⟨175, sorry⟩] := by
  sorry

end NUMINAMATH_CALUDE_sampling_method_l360_36066


namespace NUMINAMATH_CALUDE_union_of_sets_l360_36012

theorem union_of_sets (p q : ℝ) :
  let A := {x : ℝ | x^2 + p*x + q = 0}
  let B := {x : ℝ | x^2 - p*x - 2*q = 0}
  (A ∩ B = {-1}) →
  (A ∪ B = {-1, -2, 4}) := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l360_36012


namespace NUMINAMATH_CALUDE_quadrilateral_area_sum_l360_36093

/-- Represents a quadrilateral PQRS -/
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

/-- Checks if a quadrilateral is convex -/
def is_convex (quad : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
def area (quad : Quadrilateral) : ℝ := sorry

/-- Checks if a number has no perfect square factors greater than 1 -/
def no_perfect_square_factors (n : ℝ) : Prop := sorry

theorem quadrilateral_area_sum (quad : Quadrilateral) (a b c : ℝ) :
  is_convex quad →
  distance quad.P quad.Q = 7 →
  distance quad.Q quad.R = 3 →
  distance quad.R quad.S = 9 →
  distance quad.S quad.P = 9 →
  angle quad.R quad.S quad.P = π / 3 →
  ∃ (a b c : ℝ), area quad = Real.sqrt a + b * Real.sqrt c ∧
                  no_perfect_square_factors a ∧
                  no_perfect_square_factors c →
  a + b + c = 608.25 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_sum_l360_36093


namespace NUMINAMATH_CALUDE_initial_liquid_x_percentage_l360_36027

theorem initial_liquid_x_percentage
  (initial_water_percentage : Real)
  (initial_solution_weight : Real)
  (evaporated_water : Real)
  (added_solution : Real)
  (final_liquid_x_percentage : Real)
  (h1 : initial_water_percentage = 70)
  (h2 : initial_solution_weight = 8)
  (h3 : evaporated_water = 3)
  (h4 : added_solution = 3)
  (h5 : final_liquid_x_percentage = 41.25)
  : Real := by
  sorry

#check initial_liquid_x_percentage

end NUMINAMATH_CALUDE_initial_liquid_x_percentage_l360_36027


namespace NUMINAMATH_CALUDE_overtime_hours_l360_36038

/-- Queenie's daily wage as a part-time clerk -/
def daily_wage : ℕ := 150

/-- Queenie's overtime pay rate per hour -/
def overtime_rate : ℕ := 5

/-- Number of days Queenie worked -/
def days_worked : ℕ := 5

/-- Total amount Queenie received -/
def total_pay : ℕ := 770

/-- Calculate the number of overtime hours Queenie worked -/
theorem overtime_hours : 
  (total_pay - daily_wage * days_worked) / overtime_rate = 4 := by
  sorry

end NUMINAMATH_CALUDE_overtime_hours_l360_36038


namespace NUMINAMATH_CALUDE_banana_count_l360_36024

/-- The number of bananas Melissa had initially -/
def initial_bananas : ℕ := 88

/-- The number of bananas Melissa shared -/
def shared_bananas : ℕ := 4

/-- The number of bananas Melissa had left after sharing -/
def remaining_bananas : ℕ := 84

theorem banana_count : initial_bananas = shared_bananas + remaining_bananas := by
  sorry

end NUMINAMATH_CALUDE_banana_count_l360_36024


namespace NUMINAMATH_CALUDE_animal_arrangement_count_l360_36085

def num_chickens : ℕ := 3
def num_dogs : ℕ := 3
def num_cats : ℕ := 4
def num_rabbits : ℕ := 2
def total_animals : ℕ := num_chickens + num_dogs + num_cats + num_rabbits

def arrangement_count : ℕ := 41472

theorem animal_arrangement_count :
  (Nat.factorial 4) * 
  (Nat.factorial num_chickens) * 
  (Nat.factorial num_dogs) * 
  (Nat.factorial num_cats) * 
  (Nat.factorial num_rabbits) = arrangement_count :=
by sorry

end NUMINAMATH_CALUDE_animal_arrangement_count_l360_36085


namespace NUMINAMATH_CALUDE_prob_at_least_one_girl_l360_36017

/-- The probability of selecting at least one girl from a group of 4 boys and 3 girls when choosing 2 people -/
theorem prob_at_least_one_girl (num_boys : ℕ) (num_girls : ℕ) : 
  num_boys = 4 → num_girls = 3 → 
  (1 - (Nat.choose num_boys 2 : ℚ) / (Nat.choose (num_boys + num_girls) 2 : ℚ)) = 5/7 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_girl_l360_36017


namespace NUMINAMATH_CALUDE_ball_travel_distance_l360_36094

/-- Represents an elliptical billiard table -/
structure EllipticalTable where
  majorAxis : ℝ
  focalDistance : ℝ

/-- Possible distances traveled by a ball on an elliptical table -/
def possibleDistances (table : EllipticalTable) : Set ℝ :=
  {4, 3, 1}

/-- Theorem: The distance traveled by a ball on a specific elliptical table -/
theorem ball_travel_distance (table : EllipticalTable) 
  (h1 : table.majorAxis = 2)
  (h2 : table.focalDistance = 1) :
  ∃ d ∈ possibleDistances table, d = 4 ∨ d = 3 ∨ d = 1 :=
by sorry

end NUMINAMATH_CALUDE_ball_travel_distance_l360_36094


namespace NUMINAMATH_CALUDE_certain_number_is_88_l360_36023

theorem certain_number_is_88 (x : ℝ) (y : ℝ) : 
  x = y + 0.25 * y → x = 110 → y = 88 := by
sorry

end NUMINAMATH_CALUDE_certain_number_is_88_l360_36023


namespace NUMINAMATH_CALUDE_power_fraction_equality_l360_36044

theorem power_fraction_equality : (2^2014 + 2^2012) / (2^2014 - 2^2012) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l360_36044


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_sum_of_squares_l360_36082

theorem two_numbers_sum_and_sum_of_squares (a b : ℝ) :
  (∃ (x y : ℚ), x > 0 ∧ y > 0 ∧ (x : ℝ) + y = a ∧ (x : ℝ)^2 + y^2 = b) ↔
  (∃ (k : ℕ), 2*b - a^2 = (k : ℝ)^2 ∧ k > 0) :=
sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_sum_of_squares_l360_36082


namespace NUMINAMATH_CALUDE_tan_equality_implies_specific_angles_l360_36097

theorem tan_equality_implies_specific_angles (m : ℤ) :
  -180 < m ∧ m < 180 →
  Real.tan (↑m * π / 180) = Real.tan (405 * π / 180) →
  m = 45 ∨ m = -135 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_implies_specific_angles_l360_36097


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l360_36077

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) (h_sum : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) :
  a 5 + a 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l360_36077


namespace NUMINAMATH_CALUDE_paint_cans_used_l360_36016

def initial_capacity : ℕ := 36
def reduced_capacity : ℕ := 28
def lost_cans : ℕ := 4

theorem paint_cans_used : ℕ := by
  -- Prove that the number of cans used to paint 28 rooms is 14
  sorry

end NUMINAMATH_CALUDE_paint_cans_used_l360_36016


namespace NUMINAMATH_CALUDE_star_expression_equals_six_l360_36032

-- Define the new * operation
def star (a b : ℝ) : ℝ := (a + 1) * (b - 1)

-- Define the *2 operation
def star_squared (a : ℝ) : ℝ := star a a

-- Theorem statement
theorem star_expression_equals_six :
  let x : ℝ := 2
  star 3 (star_squared x) - star 2 x + 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_star_expression_equals_six_l360_36032


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_sixteen_l360_36048

theorem sum_of_solutions_eq_sixteen : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 8)^2 = 36 ∧ (x₂ - 8)^2 = 36 ∧ x₁ + x₂ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_sixteen_l360_36048


namespace NUMINAMATH_CALUDE_dot_product_theorem_l360_36099

def a : ℝ × ℝ := (1, 2)

theorem dot_product_theorem (b : ℝ × ℝ) 
  (h : (2 • a - b) = (3, 1)) : a • b = 5 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_theorem_l360_36099


namespace NUMINAMATH_CALUDE_meaningful_sqrt_over_x_l360_36089

theorem meaningful_sqrt_over_x (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 3)) / x) ↔ x ≥ -3 ∧ x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_over_x_l360_36089


namespace NUMINAMATH_CALUDE_sum_max_value_sum_max_x_product_max_value_product_max_x_l360_36025

/-- Represents a point on an ellipse --/
structure EllipsePoint where
  x : ℝ
  y : ℝ
  a : ℝ
  b : ℝ
  h_ellipse : x^2 / a^2 + y^2 / b^2 = 1
  h_positive : a > 0 ∧ b > 0

/-- The sum of x and y coordinates has a maximum value --/
theorem sum_max_value (p : EllipsePoint) :
  ∃ m : ℝ, m = Real.sqrt (p.a^2 + p.b^2) ∧
    ∀ q : EllipsePoint, q.a = p.a ∧ q.b = p.b → q.x + q.y ≤ m :=
sorry

/-- The sum of x and y coordinates reaches its maximum when x has a specific value --/
theorem sum_max_x (p : EllipsePoint) :
  ∃ x_max : ℝ, x_max = p.a^2 / Real.sqrt (p.a^2 + p.b^2) ∧
    ∀ q : EllipsePoint, q.a = p.a ∧ q.b = p.b ∧ q.x + q.y = Real.sqrt (p.a^2 + p.b^2) →
      q.x = x_max :=
sorry

/-- The product of x and y coordinates has a maximum value --/
theorem product_max_value (p : EllipsePoint) :
  ∃ m : ℝ, m = p.a * p.b / 2 ∧
    ∀ q : EllipsePoint, q.a = p.a ∧ q.b = p.b → q.x * q.y ≤ m :=
sorry

/-- The product of x and y coordinates reaches its maximum when x has a specific value --/
theorem product_max_x (p : EllipsePoint) :
  ∃ x_max : ℝ, x_max = p.a * Real.sqrt 2 / 2 ∧
    ∀ q : EllipsePoint, q.a = p.a ∧ q.b = p.b ∧ q.x * q.y = p.a * p.b / 2 →
      q.x = x_max :=
sorry

end NUMINAMATH_CALUDE_sum_max_value_sum_max_x_product_max_value_product_max_x_l360_36025


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l360_36061

/-- A line in the xy-plane is represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the xy-plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- A point lies on a line if its coordinates satisfy the line's equation. -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The problem statement as a theorem. -/
theorem parallel_line_through_point :
  let l1 : Line := { slope := -2, intercept := 3 }
  let p : Point := { x := 1, y := 2 }
  ∃ l2 : Line, parallel l1 l2 ∧ pointOnLine p l2 ∧ l2.slope = -2 ∧ l2.intercept = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l360_36061


namespace NUMINAMATH_CALUDE_reciprocal_equals_self_l360_36037

theorem reciprocal_equals_self (q : ℚ) : q⁻¹ = q → q = 1 ∨ q = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equals_self_l360_36037


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l360_36003

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![3, -2]

theorem vector_magnitude_proof :
  Real.sqrt ((2 * a 0 - b 0)^2 + (2 * a 1 - b 1)^2) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l360_36003


namespace NUMINAMATH_CALUDE_system_solution_l360_36047

theorem system_solution (x y m : ℤ) :
  x + 2*y - 6 = 0 ∧ 
  x - 2*y + m*x + 5 = 0 →
  m = -1 ∨ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l360_36047


namespace NUMINAMATH_CALUDE_factorial_345_trailing_zeros_l360_36000

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_345_trailing_zeros :
  trailingZeros 345 = 84 :=
sorry

end NUMINAMATH_CALUDE_factorial_345_trailing_zeros_l360_36000


namespace NUMINAMATH_CALUDE_special_polygon_sum_angles_l360_36004

/-- A polygon where 3 diagonals can be drawn from one vertex -/
structure SpecialPolygon where
  /-- The number of diagonals that can be drawn from one vertex -/
  diagonals_from_vertex : ℕ
  /-- The condition that 3 diagonals can be drawn from one vertex -/
  diag_condition : diagonals_from_vertex = 3

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- Theorem: The sum of interior angles of a SpecialPolygon is 720° -/
theorem special_polygon_sum_angles (p : SpecialPolygon) : 
  sum_interior_angles (p.diagonals_from_vertex + 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_sum_angles_l360_36004


namespace NUMINAMATH_CALUDE_B_power_101_l360_36035

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_101 : B^101 = B^2 := by sorry

end NUMINAMATH_CALUDE_B_power_101_l360_36035


namespace NUMINAMATH_CALUDE_min_value_theorem_l360_36072

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) 
  (heq : a^2 * (b + 4*b^2 + 2*a^2) = 8 - 2*b^3) :
  ∃ (m : ℝ), m = 8 * Real.sqrt 3 ∧ 
  (∀ (x y : ℝ), x > 0 → y > 1 → x^2 * (y + 4*y^2 + 2*x^2) = 8 - 2*y^3 → 
    8*x^2 + 4*y^2 + 3*y ≥ m) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 1 ∧ x^2 * (y + 4*y^2 + 2*x^2) = 8 - 2*y^3 ∧ 
    8*x^2 + 4*y^2 + 3*y = m) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l360_36072


namespace NUMINAMATH_CALUDE_bertha_family_without_children_bertha_family_consistent_l360_36008

/-- Represents the family structure of Bertha and her descendants -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  daughters_with_children : ℕ
  daughters_per_mother : ℕ

/-- The actual family structure of Bertha -/
def bertha_family : BerthaFamily :=
  { daughters := 8,
    granddaughters := 34,
    daughters_with_children := 5,
    daughters_per_mother := 6 }

/-- Theorem stating the number of Bertha's daughters and granddaughters without children -/
theorem bertha_family_without_children :
  bertha_family.daughters + bertha_family.granddaughters - bertha_family.daughters_with_children = 37 :=
by sorry

/-- Consistency check for the family structure -/
theorem bertha_family_consistent :
  bertha_family.daughters + bertha_family.granddaughters = 42 ∧
  bertha_family.granddaughters = bertha_family.daughters_with_children * bertha_family.daughters_per_mother :=
by sorry

end NUMINAMATH_CALUDE_bertha_family_without_children_bertha_family_consistent_l360_36008


namespace NUMINAMATH_CALUDE_right_triangle_identification_l360_36083

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_identification :
  is_right_triangle 3 4 5 ∧
  is_right_triangle 5 12 13 ∧
  is_right_triangle 6 8 10 ∧
  ¬ is_right_triangle 4 6 8 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l360_36083


namespace NUMINAMATH_CALUDE_special_numbers_are_one_and_nine_l360_36092

/-- The number of divisors of a natural number -/
def divisor_count (n : ℕ) : ℕ := sorry

/-- The set of natural numbers that are equal to the square of their divisor count -/
def special_numbers : Set ℕ := {n : ℕ | n = (divisor_count n)^2}

/-- Theorem stating that the set of special numbers is equal to {1, 9} -/
theorem special_numbers_are_one_and_nine : special_numbers = {1, 9} := by sorry

end NUMINAMATH_CALUDE_special_numbers_are_one_and_nine_l360_36092


namespace NUMINAMATH_CALUDE_gcd_45_105_l360_36078

theorem gcd_45_105 : Nat.gcd 45 105 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45_105_l360_36078


namespace NUMINAMATH_CALUDE_ellipse_chord_length_l360_36074

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  e : ℝ  -- Eccentricity

/-- Represents a line in the form y = mx + c -/
structure Line where
  m : ℝ  -- Slope
  c : ℝ  -- y-intercept

theorem ellipse_chord_length (C : Ellipse) (L : Line) :
  C.b = 1 ∧ C.e = Real.sqrt 3 / 2 ∧ L.m = 1 ∧ L.c = 1 →
  (∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
  Real.sqrt ((8/5)^2 + (8/5)^2) = 8 * Real.sqrt 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_length_l360_36074


namespace NUMINAMATH_CALUDE_four_digit_number_at_1496_l360_36046

/-- Given a list of increasing positive integers starting with digit 2,
    returns the four-digit number formed by the nth to (n+3)th digits in the list -/
def fourDigitNumber (n : ℕ) : ℕ :=
  sorry

/-- The list of increasing positive integers starting with digit 2 -/
def digitTwoList : List ℕ :=
  sorry

theorem four_digit_number_at_1496 :
  fourDigitNumber 1496 = 5822 :=
sorry

end NUMINAMATH_CALUDE_four_digit_number_at_1496_l360_36046


namespace NUMINAMATH_CALUDE_prove_triangle_cotangent_formula_l360_36062

def triangle_cotangent_formula (A B C a b c p r S : Real) : Prop :=
  let ctg_half (x : Real) := (p - x) / r
  A + B + C = Real.pi ∧
  p = (a + b + c) / 2 ∧
  S = Real.sqrt (p * (p - a) * (p - b) * (p - c)) ∧
  S = p * r ∧
  ctg_half a + ctg_half b + ctg_half c = ctg_half a * ctg_half b * ctg_half c

theorem prove_triangle_cotangent_formula (A B C a b c p r S : Real) :
  triangle_cotangent_formula A B C a b c p r S := by
  sorry

end NUMINAMATH_CALUDE_prove_triangle_cotangent_formula_l360_36062


namespace NUMINAMATH_CALUDE_total_legs_of_bokyungs_animals_l360_36036

-- Define the number of legs for puppies and chicks
def puppy_legs : ℕ := 4
def chick_legs : ℕ := 2

-- Define the number of puppies and chicks Bokyung has
def num_puppies : ℕ := 3
def num_chicks : ℕ := 7

-- Theorem to prove
theorem total_legs_of_bokyungs_animals : 
  num_puppies * puppy_legs + num_chicks * chick_legs = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_of_bokyungs_animals_l360_36036


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l360_36020

theorem diophantine_equation_solutions :
  ∀ m n : ℕ+, 7^(m : ℕ) - 3 * 2^(n : ℕ) = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) := by
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l360_36020
