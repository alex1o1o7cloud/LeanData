import Mathlib

namespace complement_of_A_in_B_l2861_286186

def A : Set ℕ := {2, 3}
def B : Set ℕ := {0, 1, 2, 3, 4}

theorem complement_of_A_in_B :
  (B \ A) = {0, 1, 4} := by sorry

end complement_of_A_in_B_l2861_286186


namespace sum_of_digits_double_permutation_l2861_286187

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Permutation relation between natural numbers -/
def isPermutationOf (a b : ℕ) : Prop := sorry

theorem sum_of_digits_double_permutation (A B : ℕ) 
  (h : isPermutationOf A B) : 
  sumOfDigits (2 * A) = sumOfDigits (2 * B) := by sorry

end sum_of_digits_double_permutation_l2861_286187


namespace count_special_numbers_eq_18_l2861_286125

/-- Counts the number of ways to arrange n items with given multiplicities. -/
def multinomial_coefficient (n : ℕ) (multiplicities : List ℕ) : ℕ :=
  Nat.factorial n / (multiplicities.map Nat.factorial).prod

/-- The number of five-digit numbers composed of 2 zeros, 2 ones, and 1 two. -/
def count_special_numbers : ℕ :=
  -- Case 1: First digit is 2
  (multinomial_coefficient 4 [2, 2]) +
  -- Case 2: First digit is 1
  (multinomial_coefficient 4 [2, 1, 1])

theorem count_special_numbers_eq_18 :
  count_special_numbers = 18 := by
  sorry

#eval count_special_numbers

end count_special_numbers_eq_18_l2861_286125


namespace average_cost_before_gratuity_l2861_286109

/-- Proves that for a group of 7 people with a total bill of $840 including 20% gratuity,
    the average cost per person before gratuity is $100. -/
theorem average_cost_before_gratuity 
  (num_people : ℕ) 
  (total_bill : ℝ) 
  (gratuity_rate : ℝ) :
  num_people = 7 →
  total_bill = 840 →
  gratuity_rate = 0.20 →
  (total_bill / (1 + gratuity_rate)) / num_people = 100 := by
sorry

end average_cost_before_gratuity_l2861_286109


namespace tangent_slope_at_negative_two_l2861_286153

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- Define the point of interest
def point : ℝ × ℝ := (-2, -8)

-- State the theorem
theorem tangent_slope_at_negative_two :
  (deriv f) point.1 = 12 := by sorry

end tangent_slope_at_negative_two_l2861_286153


namespace arithmetic_expression_evaluation_l2861_286115

theorem arithmetic_expression_evaluation :
  (∀ x y z : ℤ, x + y = z → (x = 6 ∧ y = -13 ∧ z = -7) ∨ (x = -5 ∧ y = -3 ∧ z = -8)) ∧
  (6 + (-13) = -7) ∧
  (6 + (-13) ≠ 7) ∧
  (6 + (-13) ≠ -19) ∧
  (-5 + (-3) ≠ 8) :=
by sorry

end arithmetic_expression_evaluation_l2861_286115


namespace condition_relationship_l2861_286106

theorem condition_relationship (x : ℝ) : 
  (∀ x, x = Real.sqrt (x + 2) → x^2 = x + 2) ∧ 
  (∃ x, x^2 = x + 2 ∧ x ≠ Real.sqrt (x + 2)) := by
  sorry

end condition_relationship_l2861_286106


namespace always_winnable_l2861_286164

/-- Represents a move in the card game -/
def move (deck : List ℕ) : List ℕ :=
  match deck with
  | [] => []
  | x :: xs => (xs.take x).reverse ++ [x] ++ xs.drop x

/-- Predicate to check if 1 is at the top of the deck -/
def hasOneOnTop (deck : List ℕ) : Prop :=
  match deck with
  | 1 :: _ => True
  | _ => False

/-- Theorem stating that the game is always winnable -/
theorem always_winnable (n : ℕ) (deck : List ℕ) :
  (deck.length = n) →
  (∀ i, i ∈ deck ↔ 1 ≤ i ∧ i ≤ n) →
  ∃ k, hasOneOnTop ((move^[k]) deck) :=
sorry


end always_winnable_l2861_286164


namespace symmetry_about_origin_l2861_286166

-- Define a function f and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the property of f_inv being the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Define the third function g
def g (x : ℝ) : ℝ := -f_inv (-x)

-- Theorem stating that g is symmetric to f_inv about the origin
theorem symmetry_about_origin :
  ∀ x, g (-x) = -g x :=
sorry

end symmetry_about_origin_l2861_286166


namespace p_3_eq_10_p_condition_l2861_286142

/-- A polynomial function p: ℝ → ℝ satisfying specific conditions -/
def p : ℝ → ℝ := fun x ↦ x^2 + 1

/-- The first condition: p(3) = 10 -/
theorem p_3_eq_10 : p 3 = 10 := by sorry

/-- The second condition: p(x)p(y) = p(x) + p(y) + p(xy) - 2 for all real x and y -/
theorem p_condition (x y : ℝ) : p x * p y = p x + p y + p (x * y) - 2 := by sorry

end p_3_eq_10_p_condition_l2861_286142


namespace complement_intersect_equal_l2861_286145

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 3}
def B : Set ℕ := {1, 3, 4}

theorem complement_intersect_equal : (U \ B) ∩ A = {0, 2} := by sorry

end complement_intersect_equal_l2861_286145


namespace unique_arrangements_count_l2861_286179

/-- The number of letters in the word -/
def word_length : ℕ := 7

/-- The number of identical letters (B and S) -/
def identical_letters : ℕ := 2

/-- Calculates the number of unique arrangements for the given word -/
def unique_arrangements : ℕ := (Nat.factorial word_length) / (Nat.factorial identical_letters)

/-- Theorem stating that the number of unique arrangements is 2520 -/
theorem unique_arrangements_count : unique_arrangements = 2520 := by
  sorry

end unique_arrangements_count_l2861_286179


namespace discount_percentage_is_five_percent_l2861_286149

def cameras_cost : ℝ := 2 * 110
def frames_cost : ℝ := 3 * 120
def total_cost : ℝ := cameras_cost + frames_cost
def discounted_price : ℝ := 551

theorem discount_percentage_is_five_percent :
  (total_cost - discounted_price) / total_cost * 100 = 5 := by
  sorry

end discount_percentage_is_five_percent_l2861_286149


namespace negative_square_cubed_l2861_286100

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end negative_square_cubed_l2861_286100


namespace number_of_pupils_l2861_286121

theorem number_of_pupils (total : ℕ) (parents : ℕ) (teachers : ℕ) 
  (h1 : total = 1541)
  (h2 : parents = 73)
  (h3 : teachers = 744) :
  total - (parents + teachers) = 724 := by
sorry

end number_of_pupils_l2861_286121


namespace perimeter_ABCD_l2861_286199

-- Define the points A, B, C, D, E
variable (A B C D E : ℝ × ℝ)

-- Define the properties of the triangles
def is_right_angled (X Y Z : ℝ × ℝ) : Prop := sorry
def angle_equals_45_deg (X Y Z : ℝ × ℝ) : Prop := sorry
def is_45_45_90_triangle (X Y Z : ℝ × ℝ) : Prop := sorry

-- Define the distance function
def distance (X Y : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter function for a quadrilateral
def perimeter_quadrilateral (W X Y Z : ℝ × ℝ) : ℝ :=
  distance W X + distance X Y + distance Y Z + distance Z W

-- State the theorem
theorem perimeter_ABCD (h1 : is_right_angled A B E)
                       (h2 : is_right_angled B C E)
                       (h3 : is_right_angled C D E)
                       (h4 : angle_equals_45_deg A E B)
                       (h5 : angle_equals_45_deg B E C)
                       (h6 : angle_equals_45_deg C E D)
                       (h7 : distance A E = 32)
                       (h8 : is_45_45_90_triangle A B E)
                       (h9 : is_45_45_90_triangle B C E)
                       (h10 : is_45_45_90_triangle C D E) :
  perimeter_quadrilateral A B C D = 32 + 32 * Real.sqrt 2 := by sorry

end perimeter_ABCD_l2861_286199


namespace morse_code_symbols_l2861_286196

/-- The number of possible symbols for a given sequence length -/
def symbolCount (n : ℕ) : ℕ := 2^n

/-- The total number of distinct Morse code symbols with lengths 1 to 4 -/
def totalSymbols : ℕ := symbolCount 1 + symbolCount 2 + symbolCount 3 + symbolCount 4

theorem morse_code_symbols : totalSymbols = 30 := by
  sorry

end morse_code_symbols_l2861_286196


namespace star_three_neg_five_l2861_286146

-- Define the new operation "*"
def star (a b : ℚ) : ℚ := a * b + a - b

-- Theorem statement
theorem star_three_neg_five : star 3 (-5) = -7 := by sorry

end star_three_neg_five_l2861_286146


namespace quadratic_real_roots_l2861_286197

theorem quadratic_real_roots (a b c : ℝ) :
  (∃ x : ℝ, (a^2 + b^2 + c^2) * x^2 + 2*(a + b + c) * x + 3 = 0) ↔
  (a = b ∧ b = c ∧ a ≠ 0) :=
by sorry

end quadratic_real_roots_l2861_286197


namespace coordinates_are_precise_l2861_286170

-- Define a type for location descriptions
inductive LocationDescription
  | indoor : String → String → String → LocationDescription  -- Building, room, etc.
  | roadSection : String → LocationDescription  -- Road name
  | coordinates : Float → Float → LocationDescription  -- Longitude and Latitude
  | direction : Float → String → LocationDescription  -- Angle and cardinal direction

-- Function to check if a location description is precise
def isPreciseLocation (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.coordinates _ _ => True
  | _ => False

-- Theorem statement
theorem coordinates_are_precise (locations : List LocationDescription) :
  ∃ (loc : LocationDescription), loc ∈ locations ∧ isPreciseLocation loc ↔
    ∃ (lon lat : Float), LocationDescription.coordinates lon lat ∈ locations :=
sorry

end coordinates_are_precise_l2861_286170


namespace fourth_sample_is_20_l2861_286134

def random_numbers : List ℕ := [71, 11, 5, 65, 9, 95, 86, 68, 76, 83, 20, 37, 90, 57, 16, 3, 11, 63, 14, 90]

def is_valid_sample (n : ℕ) : Bool :=
  1 ≤ n ∧ n ≤ 50

def get_fourth_sample (numbers : List ℕ) : ℕ :=
  (numbers.filter is_valid_sample).nthLe 3 sorry

theorem fourth_sample_is_20 :
  get_fourth_sample random_numbers = 20 := by sorry

end fourth_sample_is_20_l2861_286134


namespace variance_of_letters_l2861_286168

def letters : List ℕ := [10, 6, 8, 5, 6]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let μ := mean l
  (l.map (fun x => ((x : ℚ) - μ)^2)).sum / l.length

theorem variance_of_letters :
  variance letters = 16/5 := by sorry

end variance_of_letters_l2861_286168


namespace image_of_two_is_five_l2861_286178

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem image_of_two_is_five : f 2 = 5 := by
  sorry

end image_of_two_is_five_l2861_286178


namespace sin_sum_of_complex_exponentials_l2861_286133

theorem sin_sum_of_complex_exponentials (α β : ℝ) :
  Complex.exp (Complex.I * α) = 3/5 + 4/5 * Complex.I ∧
  Complex.exp (Complex.I * β) = -12/13 + 5/13 * Complex.I →
  Real.sin (α + β) = -33/65 := by
sorry

end sin_sum_of_complex_exponentials_l2861_286133


namespace problem_235_l2861_286160

theorem problem_235 (x y : ℝ) : 
  y + Real.sqrt (x^2 + y^2) = 16 ∧ x - y = 2 → x = 8 ∧ y = 6 := by
  sorry

end problem_235_l2861_286160


namespace square_side_length_l2861_286167

theorem square_side_length (area : ℝ) (side : ℝ) :
  area = 169 →
  side * side = area →
  side = 13 := by
  sorry

end square_side_length_l2861_286167


namespace log_inequality_l2861_286177

theorem log_inequality : (1 : ℝ) / 3 < Real.log 3 - Real.log 2 ∧ Real.log 3 - Real.log 2 < (1 : ℝ) / 2 := by
  sorry

end log_inequality_l2861_286177


namespace math_books_same_box_probability_l2861_286131

/-- Represents a box with a given capacity -/
structure Box where
  capacity : ℕ

/-- Represents the collection of boxes -/
def boxes : List Box := [⟨4⟩, ⟨5⟩, ⟨6⟩]

/-- Total number of textbooks -/
def total_textbooks : ℕ := 15

/-- Number of mathematics textbooks -/
def math_textbooks : ℕ := 4

/-- Calculates the probability of all mathematics textbooks being in the same box -/
noncomputable def prob_math_books_same_box : ℚ := sorry

/-- Theorem stating the probability of all mathematics textbooks being in the same box -/
theorem math_books_same_box_probability :
  prob_math_books_same_box = 1 / 91 := by sorry

end math_books_same_box_probability_l2861_286131


namespace A_D_independent_l2861_286107

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events A and D
def A : Set Ω := {ω | ω.1 = 0}
def D : Set Ω := {ω | ω.1.val + ω.2.val + 2 = 7}

-- Theorem statement
theorem A_D_independent : P (A ∩ D) = P A * P D := by sorry

end A_D_independent_l2861_286107


namespace hyperbola_m_range_l2861_286159

-- Define the equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (|m| - 1) - y^2 / (m - 2) = 1

-- Define the condition for the equation to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, hyperbola_equation x y m

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  (-1 < m ∧ m < 1) ∨ m > 2

-- Theorem statement
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ m_range m := by sorry

end hyperbola_m_range_l2861_286159


namespace geometric_sequence_common_ratio_l2861_286188

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n > 0

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : GeometricSequence a q)
  (h_arith : a 3 - 3 * a 1 = a 2 - a 3) :
  q = 3 := by
  sorry

end geometric_sequence_common_ratio_l2861_286188


namespace alcohol_solution_percentage_l2861_286104

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) 
  (added_alcohol : ℝ) (added_water : ℝ) : 
  initial_volume = 40 →
  initial_percentage = 5 →
  added_alcohol = 4.5 →
  added_water = 5.5 →
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol + added_water
  let final_percentage := (final_alcohol / final_volume) * 100
  final_percentage = 13 := by
sorry

end alcohol_solution_percentage_l2861_286104


namespace max_ab_value_l2861_286156

theorem max_ab_value (a b : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) → 
  a * b ≤ (1 : ℝ) / 4 := by sorry

end max_ab_value_l2861_286156


namespace jose_distance_l2861_286173

/-- Given a speed of 2 kilometers per hour and a time of 2 hours, 
    the distance traveled is equal to 4 kilometers. -/
theorem jose_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 2 → time = 2 → distance = speed * time → distance = 4 := by
  sorry


end jose_distance_l2861_286173


namespace not_equivalent_squared_and_equal_l2861_286112

variable {X : Type*}
variable (x : X)
variable (A B : X → ℝ)

theorem not_equivalent_squared_and_equal :
  ¬(∀ x, A x ^ 2 = B x ^ 2 ↔ A x = B x) :=
sorry

end not_equivalent_squared_and_equal_l2861_286112


namespace harrys_morning_routine_time_l2861_286117

/-- Harry's morning routine time calculation -/
theorem harrys_morning_routine_time :
  let buying_time : ℕ := 15
  let eating_time : ℕ := 2 * buying_time
  let total_time : ℕ := buying_time + eating_time
  total_time = 45 :=
by sorry

end harrys_morning_routine_time_l2861_286117


namespace min_beta_value_l2861_286174

theorem min_beta_value (α β : ℕ+) 
  (h1 : (43 : ℚ) / 197 < α / β)
  (h2 : α / β < (17 : ℚ) / 77) :
  ∀ β' : ℕ+, ((43 : ℚ) / 197 < α / β' ∧ α / β' < (17 : ℚ) / 77) → β' ≥ 32 := by
  sorry

end min_beta_value_l2861_286174


namespace min_value_fraction_l2861_286144

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y - 1 = 0) :
  (x + 2*y) / (x*y) ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + y₀ - 1 = 0 ∧ (x₀ + 2*y₀) / (x₀*y₀) = 9 :=
sorry

end min_value_fraction_l2861_286144


namespace a_2008_mod_4_l2861_286152

def sequence_a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => sequence_a n * sequence_a (n + 1) + 1

theorem a_2008_mod_4 : sequence_a 2008 % 4 = 3 := by
  sorry

end a_2008_mod_4_l2861_286152


namespace upperclassmen_sport_players_l2861_286111

/-- Represents the number of students who play a sport in a college --/
structure SportPlayers where
  total : ℕ
  freshmen : ℕ
  upperclassmen : ℕ
  freshmenPercent : ℚ
  upperclassmenPercent : ℚ
  totalNonPlayersPercent : ℚ

/-- Theorem stating that given the conditions, 383 upperclassmen play a sport --/
theorem upperclassmen_sport_players (sp : SportPlayers)
  (h1 : sp.total = 800)
  (h2 : sp.freshmenPercent = 35 / 100)
  (h3 : sp.upperclassmenPercent = 75 / 100)
  (h4 : sp.totalNonPlayersPercent = 395 / 1000)
  : sp.upperclassmen = 383 := by
  sorry

#check upperclassmen_sport_players

end upperclassmen_sport_players_l2861_286111


namespace intersection_range_l2861_286195

-- Define the function f(x) = |x^2 - 4x + 3|
def f (x : ℝ) : ℝ := abs (x^2 - 4*x + 3)

-- Define the property of having at least three intersections
def has_at_least_three_intersections (b : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = b ∧ f x₂ = b ∧ f x₃ = b

-- State the theorem
theorem intersection_range :
  ∀ b : ℝ, has_at_least_three_intersections b ↔ 0 < b ∧ b ≤ 1 := by sorry

end intersection_range_l2861_286195


namespace cube_sum_odd_implies_product_odd_l2861_286192

theorem cube_sum_odd_implies_product_odd (n m : ℤ) : 
  Odd (n^3 + m^3) → Odd (n * m) := by
sorry

end cube_sum_odd_implies_product_odd_l2861_286192


namespace real_part_of_z_is_negative_four_l2861_286157

theorem real_part_of_z_is_negative_four :
  let i : ℂ := Complex.I
  let z : ℂ := (3 + 4 * i) * i
  (z.re : ℝ) = -4 := by
  sorry

end real_part_of_z_is_negative_four_l2861_286157


namespace rectangle_perimeter_l2861_286148

theorem rectangle_perimeter (a b : ℕ) : 
  a ≠ b →  -- non-square condition
  a * b = 2 * (2 * a + 2 * b) - 8 →  -- area condition
  2 * (a + b) = 36 :=  -- perimeter conclusion
by
  sorry

end rectangle_perimeter_l2861_286148


namespace min_value_theorem_l2861_286143

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 4) :
  (∀ u v : ℝ, u > 0 → v > 0 → Real.log 2 * u + Real.log 8 * v = Real.log 4 → 
    1/x + 1/(3*y) ≤ 1/u + 1/(3*v)) ∧ 
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ Real.log 2 * x₀ + Real.log 8 * y₀ = Real.log 4 ∧ 
    1/x₀ + 1/(3*y₀) = 2) :=
by sorry

end min_value_theorem_l2861_286143


namespace lg_sum_equals_zero_l2861_286140

-- Define lg as the logarithm with base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem statement
theorem lg_sum_equals_zero : lg 2 + lg 0.5 = 0 := by sorry

end lg_sum_equals_zero_l2861_286140


namespace circle_area_ratio_l2861_286169

theorem circle_area_ratio : 
  ∀ (r₁ r₂ : ℝ), r₁ > 0 → r₂ > 0 → r₂ = 3 * r₁ →
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 8 := by
  sorry

end circle_area_ratio_l2861_286169


namespace tom_weekly_earnings_l2861_286138

/-- Calculates the weekly earnings from crab fishing given the number of buckets, crabs per bucket, price per crab, and days in a week. -/
def weekly_crab_earnings (buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_in_week : ℕ) : ℕ :=
  buckets * crabs_per_bucket * price_per_crab * days_in_week

/-- Proves that Tom's weekly earnings from crab fishing is $3360. -/
theorem tom_weekly_earnings : 
  weekly_crab_earnings 8 12 5 7 = 3360 := by
  sorry

end tom_weekly_earnings_l2861_286138


namespace expression_value_l2861_286137

theorem expression_value (x : ℝ) (h : x = 5) : 2 * x + 3 - 2 = 11 := by
  sorry

end expression_value_l2861_286137


namespace melissa_games_played_l2861_286122

def points_per_game : ℕ := 12
def total_points : ℕ := 36

theorem melissa_games_played : 
  total_points / points_per_game = 3 := by
  sorry

end melissa_games_played_l2861_286122


namespace mean_proportional_234_104_l2861_286175

theorem mean_proportional_234_104 : ∃ x : ℝ, x^2 = 234 * 104 ∧ x = 156 := by
  sorry

end mean_proportional_234_104_l2861_286175


namespace total_laundry_pieces_l2861_286172

def start_time : Nat := 8
def end_time : Nat := 12
def pieces_per_hour : Nat := 20

theorem total_laundry_pieces :
  (end_time - start_time) * pieces_per_hour = 80 := by
  sorry

end total_laundry_pieces_l2861_286172


namespace x_positive_sufficient_not_necessary_for_x_squared_plus_x_positive_l2861_286165

theorem x_positive_sufficient_not_necessary_for_x_squared_plus_x_positive :
  (∃ x : ℝ, x > 0 ∧ x^2 + x > 0) ∧
  (∃ x : ℝ, x^2 + x > 0 ∧ ¬(x > 0)) :=
by sorry

end x_positive_sufficient_not_necessary_for_x_squared_plus_x_positive_l2861_286165


namespace quadratic_general_form_l2861_286183

theorem quadratic_general_form :
  ∀ x : ℝ, x^2 = 3*x + 1 ↔ x^2 - 3*x - 1 = 0 := by
sorry

end quadratic_general_form_l2861_286183


namespace inequality_solution_1_inequality_solution_2_l2861_286123

-- Problem 1
theorem inequality_solution_1 : 
  {x : ℝ | -x^2 + 3*x + 10 < 0} = {x : ℝ | x > 5 ∨ x < -2} := by sorry

-- Problem 2
theorem inequality_solution_2 (a : ℝ) : 
  {x : ℝ | x^2 - 2*a*x + (a-1)*(a+1) ≤ 0} = {x : ℝ | a-1 ≤ x ∧ x ≤ a+1} := by sorry

end inequality_solution_1_inequality_solution_2_l2861_286123


namespace wheel_probability_l2861_286194

theorem wheel_probability (W X Y Z : ℝ) : 
  W = 3/8 → X = 1/4 → Y = 1/8 → W + X + Y + Z = 1 → Z = 1/4 := by
  sorry

end wheel_probability_l2861_286194


namespace x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l2861_286128

theorem x_eq_one_sufficient_not_necessary_for_x_squared_eq_one :
  (∃ x : ℝ, x = 1 → x^2 = 1) ∧ 
  (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) :=
sorry

end x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l2861_286128


namespace evaluate_expression_l2861_286120

theorem evaluate_expression (a : ℝ) : 
  let x : ℝ := a + 5
  (2*x - a + 4) = (a + 14) := by sorry

end evaluate_expression_l2861_286120


namespace triangle_side_length_l2861_286176

/-- A square with side length 10 cm is divided into two right trapezoids and a right triangle. -/
structure DividedSquare where
  /-- Side length of the square -/
  side_length : ℝ
  /-- Height of the trapezoids -/
  trapezoid_height : ℝ
  /-- Area difference between the trapezoids -/
  area_difference : ℝ
  /-- Length of one side of the right triangle -/
  triangle_side : ℝ
  /-- The side length is 10 cm -/
  side_length_eq : side_length = 10
  /-- The area difference between trapezoids is 10 cm² -/
  area_difference_eq : area_difference = 10
  /-- The trapezoids have equal height -/
  trapezoid_height_eq : trapezoid_height = side_length / 2

/-- The theorem to be proved -/
theorem triangle_side_length (s : DividedSquare) : s.triangle_side = 4 := by
  sorry

end triangle_side_length_l2861_286176


namespace coal_extraction_theorem_l2861_286116

/-- Represents the working time ratio and coal extraction for a year -/
structure YearData where
  ratio : Fin 4 → ℚ
  coal_extracted : ℚ

/-- Given the data for three years, calculate the total coal extraction for 4 months -/
def total_coal_extraction (year1 year2 year3 : YearData) : ℚ :=
  4 * (year1.coal_extracted * (year1.ratio 0 + year1.ratio 1 + year1.ratio 2 + year1.ratio 3) / 
      (year1.ratio 0 + year1.ratio 1 + year1.ratio 2 + year1.ratio 3) +
      year2.coal_extracted * (year2.ratio 0 + year2.ratio 1 + year2.ratio 2 + year2.ratio 3) / 
      (year2.ratio 0 + year2.ratio 1 + year2.ratio 2 + year2.ratio 3) +
      year3.coal_extracted * (year3.ratio 0 + year3.ratio 1 + year3.ratio 2 + year3.ratio 3) / 
      (year3.ratio 0 + year3.ratio 1 + year3.ratio 2 + year3.ratio 3)) / 3

theorem coal_extraction_theorem (year1 year2 year3 : YearData) 
  (h1 : year1.ratio 0 = 4 ∧ year1.ratio 1 = 1 ∧ year1.ratio 2 = 2 ∧ year1.ratio 3 = 5 ∧ year1.coal_extracted = 10)
  (h2 : year2.ratio 0 = 2 ∧ year2.ratio 1 = 3 ∧ year2.ratio 2 = 2 ∧ year2.ratio 3 = 1 ∧ year2.coal_extracted = 7)
  (h3 : year3.ratio 0 = 5 ∧ year3.ratio 1 = 2 ∧ year3.ratio 2 = 1 ∧ year3.ratio 3 = 4 ∧ year3.coal_extracted = 14) :
  total_coal_extraction year1 year2 year3 = 12 := by
  sorry

end coal_extraction_theorem_l2861_286116


namespace smallest_three_digit_number_with_sum_condition_l2861_286124

theorem smallest_three_digit_number_with_sum_condition 
  (x y z : Nat) 
  (h1 : x < 10 ∧ y < 10 ∧ z < 10) 
  (h2 : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (h3 : x + y + z = 10) 
  (h4 : x < y ∧ y < z) : 
  100 * x + 10 * y + z = 127 := by
sorry

end smallest_three_digit_number_with_sum_condition_l2861_286124


namespace boat_speed_in_still_water_l2861_286147

/-- Given a boat that travels 10 km/hr downstream and 4 km/hr upstream, 
    its speed in still water is 7 km/hr. -/
theorem boat_speed_in_still_water 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h_downstream : downstream_speed = 10) 
  (h_upstream : upstream_speed = 4) : 
  (downstream_speed + upstream_speed) / 2 = 7 := by
sorry

end boat_speed_in_still_water_l2861_286147


namespace sqrt_x_minus_3_meaningful_range_l2861_286162

-- Define the property of being meaningful for a square root
def is_meaningful (x : ℝ) : Prop := x ≥ 0

-- State the theorem
theorem sqrt_x_minus_3_meaningful_range (x : ℝ) :
  is_meaningful (x - 3) → x ≥ 3 :=
by
  sorry

end sqrt_x_minus_3_meaningful_range_l2861_286162


namespace regular_pentagons_are_similar_l2861_286110

/-- A regular pentagon is a polygon with 5 sides of equal length and 5 equal angles -/
structure RegularPentagon where
  side_length : ℝ
  angle_measure : ℝ
  side_length_pos : side_length > 0
  angle_measure_pos : angle_measure > 0
  angle_sum : angle_measure * 5 = 540

/-- Two shapes are similar if they have the same shape but not necessarily the same size -/
def AreSimilar (p1 p2 : RegularPentagon) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ p2.side_length = k * p1.side_length

/-- Theorem: Any two regular pentagons are similar -/
theorem regular_pentagons_are_similar (p1 p2 : RegularPentagon) : AreSimilar p1 p2 := by
  sorry

end regular_pentagons_are_similar_l2861_286110


namespace aunt_may_milk_sales_l2861_286113

/-- Represents the milk production and sales for Aunt May's farm --/
structure MilkProduction where
  morning : ℕ  -- Morning milk production in gallons
  evening : ℕ  -- Evening milk production in gallons
  leftover : ℕ  -- Leftover milk from yesterday in gallons
  remaining : ℕ  -- Remaining milk after selling in gallons

/-- Calculates the amount of milk sold to the ice cream factory --/
def milk_sold (p : MilkProduction) : ℕ :=
  p.morning + p.evening + p.leftover - p.remaining

/-- Theorem stating the amount of milk sold to the ice cream factory --/
theorem aunt_may_milk_sales (p : MilkProduction)
  (h_morning : p.morning = 365)
  (h_evening : p.evening = 380)
  (h_leftover : p.leftover = 15)
  (h_remaining : p.remaining = 148) :
  milk_sold p = 612 := by
  sorry

#eval milk_sold { morning := 365, evening := 380, leftover := 15, remaining := 148 }

end aunt_may_milk_sales_l2861_286113


namespace min_difference_is_one_l2861_286161

/-- Triangle with integer side lengths and specific conditions -/
structure Triangle where
  DE : ℕ
  EF : ℕ
  FD : ℕ
  perimeter_eq : DE + EF + FD = 3010
  side_order : DE < EF ∧ EF ≤ FD

/-- The smallest possible difference between EF and DE is 1 -/
theorem min_difference_is_one (t : Triangle) : 
  (∀ t' : Triangle, t'.EF - t'.DE ≥ 1) ∧ (∃ t' : Triangle, t'.EF - t'.DE = 1) :=
by sorry

end min_difference_is_one_l2861_286161


namespace complement_intersection_problem_l2861_286129

universe u

theorem complement_intersection_problem :
  let U : Set ℕ := {1, 2, 3, 4, 5}
  let M : Set ℕ := {3, 4, 5}
  let N : Set ℕ := {2, 3}
  (U \ N) ∩ M = {4, 5} := by
  sorry

end complement_intersection_problem_l2861_286129


namespace combined_tax_rate_l2861_286185

/-- Calculates the combined tax rate for Mork and Mindy -/
theorem combined_tax_rate (mork_rate mindy_rate : ℚ) (income_ratio : ℚ) :
  mork_rate = 2/5 →
  mindy_rate = 1/4 →
  income_ratio = 4 →
  (mork_rate + income_ratio * mindy_rate) / (1 + income_ratio) = 7/25 := by
  sorry

end combined_tax_rate_l2861_286185


namespace leadership_team_selection_l2861_286158

theorem leadership_team_selection (n : ℕ) (h : n = 20) :
  (n.choose 2) * ((n - 2).choose 1) = 3420 := by
  sorry

end leadership_team_selection_l2861_286158


namespace greta_is_oldest_l2861_286118

-- Define the set of people
inductive Person : Type
| Ada : Person
| Darwyn : Person
| Max : Person
| Greta : Person
| James : Person

-- Define the age relation
def younger_than (a b : Person) : Prop := sorry

-- Define the conditions
axiom ada_younger_than_darwyn : younger_than Person.Ada Person.Darwyn
axiom max_younger_than_greta : younger_than Person.Max Person.Greta
axiom james_older_than_darwyn : younger_than Person.Darwyn Person.James
axiom max_same_age_as_james : ∀ p, younger_than Person.Max p ↔ younger_than Person.James p

-- Define the oldest person property
def is_oldest (p : Person) : Prop :=
  ∀ q : Person, q ≠ p → younger_than q p

-- Theorem statement
theorem greta_is_oldest : is_oldest Person.Greta := by
  sorry

end greta_is_oldest_l2861_286118


namespace blue_paint_cans_l2861_286130

def paint_mixture (total_cans : ℕ) (blue_ratio green_ratio : ℕ) : ℕ := 
  (blue_ratio * total_cans) / (blue_ratio + green_ratio)

theorem blue_paint_cans : paint_mixture 45 4 5 = 20 := by
  sorry

end blue_paint_cans_l2861_286130


namespace cubic_sum_theorem_l2861_286189

theorem cubic_sum_theorem (x y z : ℝ) 
  (h1 : x + y + z = 2) 
  (h2 : x*y + x*z + y*z = -5) 
  (h3 : x*y*z = -6) : 
  x^3 + y^3 + z^3 = 18 := by
sorry

end cubic_sum_theorem_l2861_286189


namespace solution_set_abs_inequality_l2861_286198

theorem solution_set_abs_inequality (x : ℝ) :
  (|1 - 2*x| < 3) ↔ (x ∈ Set.Ioo (-1) 2) :=
sorry

end solution_set_abs_inequality_l2861_286198


namespace exists_x_where_inequality_fails_l2861_286155

theorem exists_x_where_inequality_fails : ∃ x : ℝ, x > 0 ∧ 2^x - x^2 < 0 := by
  sorry

end exists_x_where_inequality_fails_l2861_286155


namespace gcd_of_quadratic_and_linear_l2861_286184

theorem gcd_of_quadratic_and_linear (b : ℤ) (h : 3150 ∣ b) :
  Nat.gcd (Int.natAbs (b^2 + 9*b + 54)) (Int.natAbs (b + 4)) = 2 :=
by sorry

end gcd_of_quadratic_and_linear_l2861_286184


namespace complex_equation_solution_l2861_286136

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l2861_286136


namespace sixth_operation_result_l2861_286135

def operation (a b : ℕ) : ℕ := (a + b) * a - a

theorem sixth_operation_result : operation 7 8 = 98 := by
  sorry

end sixth_operation_result_l2861_286135


namespace incorrect_observation_value_l2861_286101

/-- Given a set of observations with known properties, calculate the incorrect value --/
theorem incorrect_observation_value
  (n : ℕ)  -- Total number of observations
  (original_mean : ℝ)  -- Original mean of observations
  (correct_value : ℝ)  -- The correct value of the misrecorded observation
  (new_mean : ℝ)  -- New mean after correction
  (hn : n = 40)  -- There are 40 observations
  (hom : original_mean = 36)  -- The original mean was 36
  (hcv : correct_value = 34)  -- The correct value of the misrecorded observation is 34
  (hnm : new_mean = 36.45)  -- The new mean after correction is 36.45
  : ∃ (incorrect_value : ℝ), incorrect_value = 52 := by
  sorry


end incorrect_observation_value_l2861_286101


namespace plan_d_more_economical_l2861_286114

/-- The cost per gigabyte for Plan C in cents -/
def plan_c_cost_per_gb : ℚ := 15

/-- The initial fee for Plan D in cents -/
def plan_d_initial_fee : ℚ := 3000

/-- The cost per gigabyte for Plan D in cents -/
def plan_d_cost_per_gb : ℚ := 8

/-- The minimum number of gigabytes for Plan D to be more economical -/
def min_gb_for_plan_d : ℕ := 429

theorem plan_d_more_economical :
  (∀ n : ℕ, n ≥ min_gb_for_plan_d →
    plan_d_initial_fee + n * plan_d_cost_per_gb < n * plan_c_cost_per_gb) ∧
  (∀ n : ℕ, n < min_gb_for_plan_d →
    plan_d_initial_fee + n * plan_d_cost_per_gb ≥ n * plan_c_cost_per_gb) :=
by sorry

end plan_d_more_economical_l2861_286114


namespace right_triangle_leg_length_l2861_286102

theorem right_triangle_leg_length 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (leg_a : a = 7) 
  (hypotenuse : c = 25) : 
  b = 24 := by
sorry

end right_triangle_leg_length_l2861_286102


namespace article_cost_l2861_286150

/-- The cost of an article satisfying given profit conditions -/
theorem article_cost : ∃ (C : ℝ), 
  (C = 70) ∧ 
  (∃ (S : ℝ), S = 1.25 * C) ∧ 
  (∃ (S_new : ℝ), S_new = 0.8 * C + 0.3 * (0.8 * C) ∧ S_new = 1.25 * C - 14.70) :=
sorry

end article_cost_l2861_286150


namespace cubic_diophantine_equation_solution_l2861_286119

theorem cubic_diophantine_equation_solution :
  ∀ x y : ℕ+, x^3 - y^3 = x * y + 61 → (x = 6 ∧ y = 5) :=
by
  sorry

end cubic_diophantine_equation_solution_l2861_286119


namespace root_difference_for_arithmetic_progression_cubic_l2861_286139

theorem root_difference_for_arithmetic_progression_cubic (a b c d : ℝ) :
  (∃ x y z : ℝ, 
    (49 * x^3 - 105 * x^2 + 63 * x - 10 = 0) ∧
    (49 * y^3 - 105 * y^2 + 63 * y - 10 = 0) ∧
    (49 * z^3 - 105 * z^2 + 63 * z - 10 = 0) ∧
    (y - x = z - y) ∧
    (x < y) ∧ (y < z)) →
  (z - x = 2 * Real.sqrt 11 / 7) :=
by sorry

end root_difference_for_arithmetic_progression_cubic_l2861_286139


namespace yard_length_26_trees_l2861_286154

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 26 equally spaced trees and 20 meters between trees is 500 meters -/
theorem yard_length_26_trees : 
  yard_length 26 20 = 500 := by
  sorry

end yard_length_26_trees_l2861_286154


namespace find_z2_l2861_286127

def complex_i : ℂ := Complex.I

theorem find_z2 (z1 z2 : ℂ) : 
  ((z1 - 2) * (1 + complex_i) = 1 - complex_i) →
  (z2.im = 2) →
  ((z1 * z2).im = 0) →
  z2 = 4 + 2 * complex_i :=
by sorry

end find_z2_l2861_286127


namespace quadratic_always_positive_l2861_286163

theorem quadratic_always_positive (b : ℝ) :
  (∀ x : ℝ, x^2 + b*x + 1 > 0) → -2 < b ∧ b < 2 := by
  sorry

end quadratic_always_positive_l2861_286163


namespace photo_frame_border_area_l2861_286180

/-- The area of the border surrounding a rectangular photograph -/
theorem photo_frame_border_area (photo_height photo_width border_width : ℕ) : 
  photo_height = 12 →
  photo_width = 15 →
  border_width = 3 →
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - photo_height * photo_width = 198 := by
  sorry

#check photo_frame_border_area

end photo_frame_border_area_l2861_286180


namespace election_winner_votes_l2861_286191

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) 
  (h1 : winner_percentage = 56 / 100) 
  (h2 : vote_difference = 288) 
  (h3 : ↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = vote_difference) :
  ↑total_votes * winner_percentage = 1344 :=
sorry

end election_winner_votes_l2861_286191


namespace square_perimeter_l2861_286108

theorem square_perimeter (s : ℝ) (h : s * s = 625) : 4 * s = 100 := by
  sorry

end square_perimeter_l2861_286108


namespace circle_ratio_l2861_286171

theorem circle_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  π * b^2 - π * a^2 = 4 * (π * a^2) → a / b = Real.sqrt 5 / 5 := by
  sorry

end circle_ratio_l2861_286171


namespace flowers_planted_per_day_l2861_286126

theorem flowers_planted_per_day (total_people : ℕ) (total_days : ℕ) (total_flowers : ℕ) 
  (h1 : total_people = 5)
  (h2 : total_days = 2)
  (h3 : total_flowers = 200)
  (h4 : total_people > 0)
  (h5 : total_days > 0) :
  total_flowers / (total_people * total_days) = 20 := by
sorry

end flowers_planted_per_day_l2861_286126


namespace alcohol_solution_concentration_l2861_286182

/-- Given a 6-liter solution that is 40% alcohol, prove that adding 1.2 liters
    of pure alcohol will result in a solution that is 50% alcohol. -/
theorem alcohol_solution_concentration (initial_volume : ℝ) (initial_concentration : ℝ)
    (added_alcohol : ℝ) (target_concentration : ℝ) :
  initial_volume = 6 →
  initial_concentration = 0.4 →
  added_alcohol = 1.2 →
  target_concentration = 0.5 →
  (initial_volume * initial_concentration + added_alcohol) /
    (initial_volume + added_alcohol) = target_concentration := by
  sorry

#check alcohol_solution_concentration

end alcohol_solution_concentration_l2861_286182


namespace rhombus_area_in_rectangle_l2861_286132

/-- The area of a rhombus formed by intersecting equilateral triangles in a rectangle --/
theorem rhombus_area_in_rectangle (a b : ℝ) (h1 : a = 4 * Real.sqrt 3) (h2 : b = 3 * Real.sqrt 3) :
  let triangle_height := (Real.sqrt 3 / 2) * a
  let overlap := 2 * triangle_height - b
  let rhombus_area := (1 / 2) * overlap * a
  rhombus_area = 54 := by sorry

end rhombus_area_in_rectangle_l2861_286132


namespace parabola_equation_l2861_286193

/-- A parabola with vertex at the origin and directrix x = -1 has the equation y^2 = 4x -/
theorem parabola_equation (p : ℝ → ℝ → Prop) : 
  (∀ x y, p x y ↔ y^2 = 4*x) → 
  (∀ x, p x 0 ↔ x = 0) →  -- vertex at origin
  (∀ y, p (-1) y ↔ False) →  -- directrix at x = -1
  ∀ x y, p x y ↔ y^2 = 4*x := by
sorry

end parabola_equation_l2861_286193


namespace smallest_x_for_1260x_perfect_square_l2861_286190

theorem smallest_x_for_1260x_perfect_square : 
  ∃ (x : ℕ+), 
    (∀ (y : ℕ+), ∃ (N : ℤ), 1260 * y = N^2 → x ≤ y) ∧
    (∃ (N : ℤ), 1260 * x = N^2) ∧
    x = 35 := by
  sorry

end smallest_x_for_1260x_perfect_square_l2861_286190


namespace min_value_sum_of_squares_l2861_286151

theorem min_value_sum_of_squares (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_9 : x + y + z = 9) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 :=
by sorry

end min_value_sum_of_squares_l2861_286151


namespace max_brownies_is_294_l2861_286141

/-- Represents the dimensions of a rectangular pan of brownies -/
structure BrowniePan where
  m : ℕ  -- length
  n : ℕ  -- width

/-- Calculates the number of interior pieces in a brownie pan -/
def interiorPieces (pan : BrowniePan) : ℕ :=
  (pan.m - 2) * (pan.n - 2)

/-- Calculates the number of perimeter pieces in a brownie pan -/
def perimeterPieces (pan : BrowniePan) : ℕ :=
  2 * pan.m + 2 * pan.n - 4

/-- Checks if the interior pieces are twice the perimeter pieces -/
def validCutting (pan : BrowniePan) : Prop :=
  interiorPieces pan = 2 * perimeterPieces pan

/-- Calculates the total number of brownies in a pan -/
def totalBrownies (pan : BrowniePan) : ℕ :=
  pan.m * pan.n

/-- Theorem: The maximum number of brownies is 294 given the conditions -/
theorem max_brownies_is_294 :
  ∃ (pan : BrowniePan), validCutting pan ∧
    (∀ (other : BrowniePan), validCutting other → totalBrownies other ≤ totalBrownies pan) ∧
    totalBrownies pan = 294 :=
  sorry

end max_brownies_is_294_l2861_286141


namespace smallest_four_digit_congruence_solution_l2861_286105

theorem smallest_four_digit_congruence_solution :
  let x : ℕ := 1011
  (∀ y : ℕ, y < x → y < 1000 ∨ ¬(5 * y ≡ 25 [ZMOD 20] ∧ 
                                 3 * y + 10 ≡ 19 [ZMOD 7] ∧ 
                                 y + 3 ≡ 2 * y [ZMOD 12])) ∧
  (5 * x ≡ 25 [ZMOD 20] ∧ 
   3 * x + 10 ≡ 19 [ZMOD 7] ∧ 
   x + 3 ≡ 2 * x [ZMOD 12]) :=
by sorry

end smallest_four_digit_congruence_solution_l2861_286105


namespace transform_minus3_minus8i_l2861_286181

def rotate90 (z : ℂ) : ℂ := z * Complex.I

def dilate2 (z : ℂ) : ℂ := 2 * z

def transform (z : ℂ) : ℂ := dilate2 (rotate90 z)

theorem transform_minus3_minus8i :
  transform (-3 - 8 * Complex.I) = 16 - 6 * Complex.I := by
  sorry

end transform_minus3_minus8i_l2861_286181


namespace fraction_evaluation_l2861_286103

theorem fraction_evaluation (x : ℝ) (h : x = 3) : (x^6 + 8*x^3 + 16) / (x^3 + 4) = 31 := by
  sorry

end fraction_evaluation_l2861_286103
