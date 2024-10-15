import Mathlib

namespace NUMINAMATH_CALUDE_quiz_score_of_dropped_student_l60_6002

theorem quiz_score_of_dropped_student 
  (initial_students : ℕ)
  (initial_average : ℚ)
  (curve_adjustment : ℕ)
  (remaining_students : ℕ)
  (final_average : ℚ)
  (h1 : initial_students = 16)
  (h2 : initial_average = 61.5)
  (h3 : curve_adjustment = 5)
  (h4 : remaining_students = 15)
  (h5 : final_average = 64) :
  ∃ (dropped_score : ℕ), 
    (initial_students : ℚ) * initial_average - dropped_score + 
    (remaining_students : ℚ) * curve_adjustment = 
    (remaining_students : ℚ) * final_average ∧ 
    dropped_score = 99 := by
  sorry

end NUMINAMATH_CALUDE_quiz_score_of_dropped_student_l60_6002


namespace NUMINAMATH_CALUDE_maggots_eaten_first_correct_l60_6022

/-- The number of maggots eaten by the beetle in the first feeding -/
def maggots_eaten_first : ℕ := 17

/-- The total number of maggots served -/
def total_maggots : ℕ := 20

/-- The number of maggots eaten in the second feeding -/
def maggots_eaten_second : ℕ := 3

/-- Theorem stating that the number of maggots eaten in the first feeding is correct -/
theorem maggots_eaten_first_correct : 
  maggots_eaten_first = total_maggots - maggots_eaten_second := by
  sorry

end NUMINAMATH_CALUDE_maggots_eaten_first_correct_l60_6022


namespace NUMINAMATH_CALUDE_novel_reading_time_l60_6034

theorem novel_reading_time (total_pages : ℕ) (rate_alice rate_bob rate_chandra : ℚ) :
  total_pages = 760 ∧ 
  rate_alice = 1 / 20 ∧ 
  rate_bob = 1 / 45 ∧ 
  rate_chandra = 1 / 30 →
  ∃ t : ℚ, t = 7200 ∧ 
    t * rate_alice + t * rate_bob + t * rate_chandra = total_pages :=
by sorry

end NUMINAMATH_CALUDE_novel_reading_time_l60_6034


namespace NUMINAMATH_CALUDE_parallelepipeds_in_4x4x4_cube_l60_6018

/-- The number of distinct rectangular parallelepipeds in a cube of size n --/
def count_parallelepipeds (n : ℕ) : ℕ :=
  (n + 1).choose 2 ^ 3

/-- Theorem stating that in a 4 × 4 × 4 cube, there are 1000 distinct rectangular parallelepipeds --/
theorem parallelepipeds_in_4x4x4_cube :
  count_parallelepipeds 4 = 1000 := by sorry

end NUMINAMATH_CALUDE_parallelepipeds_in_4x4x4_cube_l60_6018


namespace NUMINAMATH_CALUDE_larger_box_capacity_l60_6073

/-- Represents a rectangular box with integer dimensions -/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box -/
def Box.volume (b : Box) : ℕ := b.length * b.width * b.height

/-- Represents the number of marbles a box can hold -/
def marbles_capacity (b : Box) (marbles : ℕ) : Prop :=
  b.volume = marbles

theorem larger_box_capacity 
  (kevin_box : Box)
  (kevin_marbles : ℕ)
  (laura_box : Box)
  (h1 : kevin_box.length = 3 ∧ kevin_box.width = 3 ∧ kevin_box.height = 8)
  (h2 : marbles_capacity kevin_box kevin_marbles)
  (h3 : kevin_marbles = 216)
  (h4 : laura_box.length = 3 * kevin_box.length ∧ 
        laura_box.width = 3 * kevin_box.width ∧ 
        laura_box.height = 3 * kevin_box.height) :
  marbles_capacity laura_box 5832 :=
sorry

end NUMINAMATH_CALUDE_larger_box_capacity_l60_6073


namespace NUMINAMATH_CALUDE_subtract_negative_l60_6048

theorem subtract_negative : -3 - 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l60_6048


namespace NUMINAMATH_CALUDE_inequality_proof_l60_6066

theorem inequality_proof (n : ℕ) (hn : n > 1) :
  1 / Real.exp 1 - 1 / (n * Real.exp 1) < (1 - 1 / n : ℝ)^n ∧
  (1 - 1 / n : ℝ)^n < 1 / Real.exp 1 - 1 / (2 * n * Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l60_6066


namespace NUMINAMATH_CALUDE_fraction_of_25_problem_l60_6027

theorem fraction_of_25_problem : ∃ x : ℚ, 
  x * 25 = 80 / 100 * 40 - 12 ∧ 
  x = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_25_problem_l60_6027


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l60_6043

theorem square_perimeter_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ k : ℝ, b = k * a * Real.sqrt 2) → (4 * b) / (4 * a) = 5 → 
  b / (a * Real.sqrt 2) = 5 * Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l60_6043


namespace NUMINAMATH_CALUDE_smallest_n_with_common_divisors_l60_6097

def M : ℕ := 30030

theorem smallest_n_with_common_divisors (n : ℕ) : n = 9440 ↔ 
  (∀ k : ℕ, k ≤ 20 → ∃ d : ℕ, d > 1 ∧ d ∣ (n + k) ∧ d ∣ M) ∧
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 20 ∧ ∀ d : ℕ, d > 1 → d ∣ (m + k) → ¬(d ∣ M)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_common_divisors_l60_6097


namespace NUMINAMATH_CALUDE_students_taking_neither_proof_l60_6010

def students_taking_neither (total students_music students_art students_dance
                             students_music_art students_art_dance students_music_dance
                             students_all_three : ℕ) : ℕ :=
  total - (students_music + students_art + students_dance
           - students_music_art - students_art_dance - students_music_dance
           + students_all_three)

theorem students_taking_neither_proof :
  students_taking_neither 2500 200 150 100 75 50 40 25 = 2190 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_neither_proof_l60_6010


namespace NUMINAMATH_CALUDE_carol_invitations_proof_l60_6062

/-- The number of invitations Carol is sending out -/
def total_invitations : ℕ := 12

/-- The number of packs Carol bought -/
def number_of_packs : ℕ := 3

/-- The number of invitations in each pack -/
def invitations_per_pack : ℕ := total_invitations / number_of_packs

theorem carol_invitations_proof :
  invitations_per_pack = 4 ∧
  total_invitations = number_of_packs * invitations_per_pack :=
by sorry

end NUMINAMATH_CALUDE_carol_invitations_proof_l60_6062


namespace NUMINAMATH_CALUDE_smallest_winning_m_l60_6028

/-- Represents the state of a square on the board -/
inductive Color
| White
| Green

/-- Represents the game board -/
def Board := Array Color

/-- Represents a player in the game -/
inductive Player
| Ana
| Banana

/-- Ana's strategy function type -/
def AnaStrategy := Board → Fin 2024 → Bool

/-- Banana's strategy function type -/
def BananaStrategy := Board → Nat → Nat

/-- Simulates a single game with given strategies and m -/
def playGame (m : Nat) (anaStrat : AnaStrategy) (bananaStrat : BananaStrategy) : Bool :=
  sorry

/-- Checks if Ana has a winning strategy for a given m -/
def anaHasWinningStrategy (m : Nat) : Bool :=
  sorry

/-- The main theorem stating the smallest m for which Ana can guarantee winning -/
theorem smallest_winning_m :
  (∀ m : Nat, m < 88 → ¬ anaHasWinningStrategy m) ∧
  anaHasWinningStrategy 88 :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_m_l60_6028


namespace NUMINAMATH_CALUDE_farm_theorem_l60_6096

def farm_problem (num_pigs num_hens : ℕ) : Prop :=
  let num_heads := num_pigs + num_hens
  let num_legs := 4 * num_pigs + 2 * num_hens
  (num_pigs = 11) ∧ (∃ k : ℕ, num_legs = 2 * num_heads + k) ∧ (num_legs - 2 * num_heads = 22)

theorem farm_theorem : ∃ num_hens : ℕ, farm_problem 11 num_hens :=
by sorry

end NUMINAMATH_CALUDE_farm_theorem_l60_6096


namespace NUMINAMATH_CALUDE_yoongi_score_l60_6074

theorem yoongi_score (yoongi eunji yuna : ℕ) 
  (h1 : eunji = yoongi - 25)
  (h2 : yuna = eunji - 20)
  (h3 : yuna = 8) :
  yoongi = 53 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_score_l60_6074


namespace NUMINAMATH_CALUDE_prove_num_sodas_l60_6031

def sandwich_cost : ℚ := 149/100
def soda_cost : ℚ := 87/100
def total_cost : ℚ := 646/100
def num_sandwiches : ℕ := 2

def num_sodas : ℕ := 4

theorem prove_num_sodas : 
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = total_cost := by
  sorry

#eval num_sodas

end NUMINAMATH_CALUDE_prove_num_sodas_l60_6031


namespace NUMINAMATH_CALUDE_defense_attorney_implication_l60_6089

-- Define propositions
variable (P : Prop) -- P represents "the defendant is guilty"
variable (Q : Prop) -- Q represents "the defendant had an accomplice"

-- Theorem statement
theorem defense_attorney_implication : ¬(P → Q) → (P ∧ ¬Q) := by
  sorry

end NUMINAMATH_CALUDE_defense_attorney_implication_l60_6089


namespace NUMINAMATH_CALUDE_alice_departure_time_l60_6093

/-- Proof that Alice must leave 30 minutes after Bob to arrive in city B just before him. -/
theorem alice_departure_time (distance : ℝ) (bob_speed : ℝ) (alice_speed : ℝ) 
  (h1 : distance = 220)
  (h2 : bob_speed = 40)
  (h3 : alice_speed = 44) :
  (distance / bob_speed - distance / alice_speed) * 60 = 30 := by
  sorry

#check alice_departure_time

end NUMINAMATH_CALUDE_alice_departure_time_l60_6093


namespace NUMINAMATH_CALUDE_basketball_score_proof_l60_6033

theorem basketball_score_proof (two_points three_points free_throws : ℕ) : 
  (3 * three_points = 2 * two_points) →
  (free_throws = 2 * three_points) →
  (2 * two_points + 3 * three_points + free_throws = 72) →
  free_throws = 18 := by
sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l60_6033


namespace NUMINAMATH_CALUDE_michael_has_eight_robots_l60_6005

/-- The number of animal robots Michael has -/
def michaels_robots : ℕ := sorry

/-- The number of animal robots Tom has -/
def toms_robots : ℕ := 16

/-- Tom has twice as many animal robots as Michael -/
axiom twice_as_many : toms_robots = 2 * michaels_robots

theorem michael_has_eight_robots : michaels_robots = 8 := by sorry

end NUMINAMATH_CALUDE_michael_has_eight_robots_l60_6005


namespace NUMINAMATH_CALUDE_suspension_ratio_l60_6083

/-- The number of fingers and toes a typical person has -/
def typical_fingers_and_toes : ℕ := 20

/-- The number of days Kris is suspended for each bullying instance -/
def suspension_days_per_instance : ℕ := 3

/-- The number of bullying instances Kris is responsible for -/
def bullying_instances : ℕ := 20

/-- Kris's total suspension days -/
def total_suspension_days : ℕ := suspension_days_per_instance * bullying_instances

theorem suspension_ratio :
  total_suspension_days / typical_fingers_and_toes = 3 :=
by sorry

end NUMINAMATH_CALUDE_suspension_ratio_l60_6083


namespace NUMINAMATH_CALUDE_grandmas_farm_l60_6013

theorem grandmas_farm (chickens ducks : ℕ) : 
  chickens = 4 * ducks ∧ chickens = ducks + 600 → chickens = 800 ∧ ducks = 200 := by
  sorry

end NUMINAMATH_CALUDE_grandmas_farm_l60_6013


namespace NUMINAMATH_CALUDE_ladder_wood_length_l60_6014

/-- Calculates the total length of wood needed for ladder rungs -/
theorem ladder_wood_length 
  (rung_length : ℚ)      -- Length of each rung in inches
  (rung_spacing : ℚ)     -- Space between rungs in inches
  (climb_height : ℚ)     -- Height to climb in feet
  (h1 : rung_length = 18)
  (h2 : rung_spacing = 6)
  (h3 : climb_height = 50) :
  (climb_height * 12 / rung_spacing) * (rung_length / 12) = 150 :=
by sorry

end NUMINAMATH_CALUDE_ladder_wood_length_l60_6014


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l60_6061

theorem volleyball_team_selection (n : ℕ) (k : ℕ) (t : ℕ) :
  n = 15 →
  k = 6 →
  t = 3 →
  (Nat.choose n k) - (Nat.choose (n - t) k) = 4081 :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l60_6061


namespace NUMINAMATH_CALUDE_gas_pressure_change_l60_6003

/-- Given inverse proportionality of pressure and volume at constant temperature,
    prove that the pressure in a 6-liter container is 4 kPa, given initial conditions. -/
theorem gas_pressure_change (p₁ p₂ v₁ v₂ : ℝ) : 
  p₁ > 0 → v₁ > 0 → p₂ > 0 → v₂ > 0 →  -- Ensuring positive values
  (p₁ * v₁ = p₂ * v₂) →  -- Inverse proportionality
  p₁ = 8 → v₁ = 3 → v₂ = 6 →  -- Initial conditions and new volume
  p₂ = 4 := by sorry

end NUMINAMATH_CALUDE_gas_pressure_change_l60_6003


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_squared_l60_6050

theorem right_triangle_max_ratio_squared (a b x y : ℝ) : 
  0 < a → 0 < b → a ≥ b → 
  x^2 + b^2 = a^2 + y^2 → -- right triangle condition
  x + y = a → 
  x ≥ 0 → y ≥ 0 → 
  (a / b)^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_squared_l60_6050


namespace NUMINAMATH_CALUDE_cloth_selling_price_l60_6041

theorem cloth_selling_price 
  (meters : ℕ) 
  (profit_per_meter : ℕ) 
  (cost_price_per_meter : ℕ) 
  (h1 : meters = 75)
  (h2 : profit_per_meter = 15)
  (h3 : cost_price_per_meter = 51) :
  meters * (cost_price_per_meter + profit_per_meter) = 4950 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l60_6041


namespace NUMINAMATH_CALUDE_one_pencil_one_pen_cost_l60_6004

def pencil_cost : ℝ → ℝ → Prop := λ p q ↦ 3 * p + 2 * q = 3.75
def pen_cost : ℝ → ℝ → Prop := λ p q ↦ 2 * p + 3 * q = 4.05

theorem one_pencil_one_pen_cost (p q : ℝ) 
  (h1 : pencil_cost p q) (h2 : pen_cost p q) : 
  p + q = 1.56 := by
  sorry

end NUMINAMATH_CALUDE_one_pencil_one_pen_cost_l60_6004


namespace NUMINAMATH_CALUDE_inverse_square_theorem_l60_6068

/-- A function representing the inverse square relationship between x and y -/
def inverse_square_relation (k : ℝ) (x y : ℝ) : Prop :=
  x = k / (y ^ 2)

/-- Theorem stating the relationship between x and y -/
theorem inverse_square_theorem (k : ℝ) :
  (inverse_square_relation k 1 3) →
  (inverse_square_relation k 0.5625 4) :=
by sorry

end NUMINAMATH_CALUDE_inverse_square_theorem_l60_6068


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l60_6058

def U : Set Nat := {1,2,3,4,5,6,7}
def A : Set Nat := {2,4,5}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {1,3,6,7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l60_6058


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l60_6035

/-- Circle O₁ with equation x² + y² = 1 -/
def circle_O₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle O₂ with equation x² + y² - 6x + 8y + 9 = 0 -/
def circle_O₂ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 + 8*p.2 + 9 = 0}

/-- The center of circle O₁ -/
def center_O₁ : ℝ × ℝ := (0, 0)

/-- The radius of circle O₁ -/
def radius_O₁ : ℝ := 1

/-- The center of circle O₂ -/
def center_O₂ : ℝ × ℝ := (3, -4)

/-- The radius of circle O₂ -/
def radius_O₂ : ℝ := 4

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c₁ c₂ : Set (ℝ × ℝ)) (center₁ center₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  ((center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2).sqrt = r₁ + r₂

theorem circles_externally_tangent :
  externally_tangent circle_O₁ circle_O₂ center_O₁ center_O₂ radius_O₁ radius_O₂ := by
  sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l60_6035


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l60_6047

theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : a = 1) (h2 : b = 3) 
  (h3 : c^2 = a^2 + b^2) : c = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l60_6047


namespace NUMINAMATH_CALUDE_rainy_days_count_l60_6057

/-- Proves that the number of rainy days in a week is 2, given the conditions of Mo's drinking habits. -/
theorem rainy_days_count (n : ℤ) : 
  (∃ (R NR : ℕ), 
    R + NR = 7 ∧ 
    n * R + 3 * NR = 20 ∧ 
    3 * NR = n * R + 10) → 
  (∃ (R : ℕ), R = 2) :=
by sorry

end NUMINAMATH_CALUDE_rainy_days_count_l60_6057


namespace NUMINAMATH_CALUDE_sum_of_integers_l60_6017

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val^2 + y.val^2 = 130) 
  (h2 : x.val * y.val = 45) : 
  x.val + y.val = 2 * Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l60_6017


namespace NUMINAMATH_CALUDE_simplify_fraction_expression_l60_6029

theorem simplify_fraction_expression (a b : ℝ) (h : a + b ≠ 0) (h' : a + 2*b ≠ 0) (h'' : a^2 - b^2 ≠ 0) :
  (((a - b) / (a + 2*b)) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2))) - 2 = -a / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_expression_l60_6029


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_l60_6078

theorem quadratic_roots_imply_m (m : ℝ) : 
  (∃ x : ℂ, 5 * x^2 - 2 * x + m = 0 ∧ x = (2 + Complex.I * Real.sqrt 78) / 10) ∧
  (∃ x : ℂ, 5 * x^2 - 2 * x + m = 0 ∧ x = (2 - Complex.I * Real.sqrt 78) / 10) →
  m = 41 / 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_l60_6078


namespace NUMINAMATH_CALUDE_divisor_implies_value_l60_6049

theorem divisor_implies_value (k : ℕ) : 
  21^k ∣ 435961 → 7^k - k^7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_implies_value_l60_6049


namespace NUMINAMATH_CALUDE_badminton_match_31_probability_l60_6081

def badminton_match_probability (p : ℝ) : ℝ :=
  4 * p^3 * (1 - p)

theorem badminton_match_31_probability :
  badminton_match_probability (2/3) = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_badminton_match_31_probability_l60_6081


namespace NUMINAMATH_CALUDE_same_digit_sum_in_arithmetic_progression_l60_6008

-- Define an arithmetic progression of natural numbers
def arithmeticProgression (a d : ℕ) : ℕ → ℕ := λ n => a + n * d

-- Define the sum of digits function
def sumOfDigits : ℕ → ℕ := sorry

theorem same_digit_sum_in_arithmetic_progression (a d : ℕ) :
  ∃ (k l : ℕ), k ≠ l ∧ sumOfDigits (arithmeticProgression a d k) = sumOfDigits (arithmeticProgression a d l) := by
  sorry

end NUMINAMATH_CALUDE_same_digit_sum_in_arithmetic_progression_l60_6008


namespace NUMINAMATH_CALUDE_final_book_count_l60_6082

/-- Represents the number of books in each genre -/
structure BookCollection :=
  (novels : ℕ)
  (science : ℕ)
  (cookbooks : ℕ)
  (philosophy : ℕ)
  (history : ℕ)
  (selfHelp : ℕ)

/-- Represents the donation percentages for each genre -/
structure DonationPercentages :=
  (novels : ℚ)
  (science : ℚ)
  (cookbooks : ℚ)
  (philosophy : ℚ)
  (history : ℚ)
  (selfHelp : ℚ)

def initialCollection : BookCollection :=
  { novels := 75
  , science := 55
  , cookbooks := 40
  , philosophy := 35
  , history := 25
  , selfHelp := 20 }

def donationPercentages : DonationPercentages :=
  { novels := 3/5
  , science := 3/4
  , cookbooks := 1/2
  , philosophy := 3/10
  , history := 1/4
  , selfHelp := 1 }

def recyclePercentage : ℚ := 1/20

def newBooksAcquired : ℕ := 24

theorem final_book_count
  (total : ℕ)
  (h1 : total = initialCollection.novels + initialCollection.science +
                initialCollection.cookbooks + initialCollection.philosophy +
                initialCollection.history + initialCollection.selfHelp)
  (h2 : total = 250) :
  ∃ (donatedBooks recycledBooks remainingBooks : ℕ),
    donatedBooks = ⌊initialCollection.novels * donationPercentages.novels⌋ +
                   ⌊initialCollection.science * donationPercentages.science⌋ +
                   ⌊initialCollection.cookbooks * donationPercentages.cookbooks⌋ +
                   ⌊initialCollection.philosophy * donationPercentages.philosophy⌋ +
                   ⌊initialCollection.history * donationPercentages.history⌋ +
                   ⌊initialCollection.selfHelp * donationPercentages.selfHelp⌋ ∧
    recycledBooks = ⌊(donatedBooks : ℚ) * recyclePercentage⌋ ∧
    remainingBooks = total - donatedBooks + recycledBooks ∧
    remainingBooks + newBooksAcquired = 139 :=
by sorry

end NUMINAMATH_CALUDE_final_book_count_l60_6082


namespace NUMINAMATH_CALUDE_not_p_and_not_q_true_l60_6009

theorem not_p_and_not_q_true (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬(p ∨ q)) : 
  (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_not_q_true_l60_6009


namespace NUMINAMATH_CALUDE_exactly_one_valid_sequence_of_length_15_l60_6054

/-- Represents a sequence of A's and B's -/
inductive ABSequence
  | empty : ABSequence
  | cons_a : ABSequence → ABSequence
  | cons_b : ABSequence → ABSequence

/-- Returns true if the given sequence satisfies the run length conditions -/
def valid_sequence (s : ABSequence) : Bool :=
  sorry

/-- Returns the length of the sequence -/
def sequence_length (s : ABSequence) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem exactly_one_valid_sequence_of_length_15 :
  ∃! (s : ABSequence), valid_sequence s ∧ sequence_length s = 15 :=
  sorry

end NUMINAMATH_CALUDE_exactly_one_valid_sequence_of_length_15_l60_6054


namespace NUMINAMATH_CALUDE_dans_potatoes_l60_6000

/-- The number of potatoes Dan has after rabbits eat some -/
def remaining_potatoes (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

theorem dans_potatoes : remaining_potatoes 7 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_dans_potatoes_l60_6000


namespace NUMINAMATH_CALUDE_abs_of_negative_2023_l60_6059

theorem abs_of_negative_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_of_negative_2023_l60_6059


namespace NUMINAMATH_CALUDE_complex_square_roots_l60_6015

theorem complex_square_roots : ∃ (z₁ z₂ : ℂ),
  z₁^2 = -100 - 49*I ∧ 
  z₂^2 = -100 - 49*I ∧ 
  z₁ = (7*Real.sqrt 2)/2 - (7*Real.sqrt 2)/2*I ∧
  z₂ = -(7*Real.sqrt 2)/2 + (7*Real.sqrt 2)/2*I ∧
  ∀ (z : ℂ), z^2 = -100 - 49*I → (z = z₁ ∨ z = z₂) := by
sorry

end NUMINAMATH_CALUDE_complex_square_roots_l60_6015


namespace NUMINAMATH_CALUDE_puzzle_solution_l60_6076

/-- Represents the animals in the puzzle -/
inductive Animal : Type
  | Cat | Chicken | Crab | Bear | Goat

/-- Represents the puzzle grid -/
def Grid := Animal → Nat

/-- Checks if the grid satisfies the sum conditions -/
def satisfies_sums (g : Grid) : Prop :=
  g Animal.Crab + g Animal.Crab + g Animal.Crab + g Animal.Crab + g Animal.Crab = 10 ∧
  g Animal.Goat + g Animal.Goat + g Animal.Crab + g Animal.Bear + g Animal.Bear = 16 ∧
  g Animal.Crab + g Animal.Chicken + g Animal.Chicken + g Animal.Goat + g Animal.Crab = 17 ∧
  g Animal.Cat + g Animal.Bear + g Animal.Goat + g Animal.Goat + g Animal.Crab = 13

/-- Checks if all animals have different values -/
def all_different (g : Grid) : Prop :=
  ∀ a b : Animal, a ≠ b → g a ≠ g b

/-- The main theorem stating the unique solution -/
theorem puzzle_solution :
  ∃! g : Grid, satisfies_sums g ∧ all_different g ∧
    g Animal.Cat = 1 ∧ g Animal.Chicken = 5 ∧ g Animal.Crab = 2 ∧
    g Animal.Bear = 4 ∧ g Animal.Goat = 3 :=
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l60_6076


namespace NUMINAMATH_CALUDE_value_k_std_dev_below_mean_two_std_dev_below_mean_l60_6039

/-- For a normal distribution with mean μ and standard deviation σ,
    the value that is exactly k standard deviations less than the mean is μ - k * σ -/
theorem value_k_std_dev_below_mean (μ σ k : ℝ) :
  μ - k * σ = μ - k * σ := by sorry

/-- For a normal distribution with mean 12 and standard deviation 1.2,
    the value that is exactly 2 standard deviations less than the mean is 9.6 -/
theorem two_std_dev_below_mean :
  let μ : ℝ := 12  -- mean
  let σ : ℝ := 1.2 -- standard deviation
  let k : ℝ := 2   -- number of standard deviations below mean
  μ - k * σ = 9.6 := by sorry

end NUMINAMATH_CALUDE_value_k_std_dev_below_mean_two_std_dev_below_mean_l60_6039


namespace NUMINAMATH_CALUDE_simplify_expression_l60_6077

theorem simplify_expression (x : ℝ) : (3 * x^4)^2 * (2 * x^2)^3 = 72 * x^14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l60_6077


namespace NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l60_6060

theorem product_from_lcm_and_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 90) 
  (h2 : Nat.gcd a b = 10) : 
  a * b = 900 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l60_6060


namespace NUMINAMATH_CALUDE_amusement_park_trip_distance_l60_6032

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- The total distance covered by Amanda and her friends -/
theorem amusement_park_trip_distance : 
  let d1 := distance 40 1.5
  let d2 := distance 50 1
  let d3 := distance 30 2.25
  d1 + d2 + d3 = 177.5 := by sorry

end NUMINAMATH_CALUDE_amusement_park_trip_distance_l60_6032


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l60_6026

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = (3/4) * x ∨ y = -(3/4) * x

/-- Theorem: The asymptotes of the given hyperbola -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l60_6026


namespace NUMINAMATH_CALUDE_binomial_fraction_integer_l60_6016

theorem binomial_fraction_integer (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  (∃ m : ℕ, n + 2 = m * (k + 2)) ↔ 
  ∃ z : ℤ, z = (2*n - 3*k - 2) * (n.choose k) / (k + 2) :=
sorry

end NUMINAMATH_CALUDE_binomial_fraction_integer_l60_6016


namespace NUMINAMATH_CALUDE_total_height_increase_two_centuries_l60_6045

/-- Represents the increase in height (in meters) per decade for a specific species of plants -/
def height_increase_per_decade : ℕ := 90

/-- Represents the number of decades in 2 centuries -/
def decades_in_two_centuries : ℕ := 20

/-- Theorem stating that the total increase in height over 2 centuries is 1800 meters -/
theorem total_height_increase_two_centuries : 
  height_increase_per_decade * decades_in_two_centuries = 1800 := by
  sorry

end NUMINAMATH_CALUDE_total_height_increase_two_centuries_l60_6045


namespace NUMINAMATH_CALUDE_solution_pairs_l60_6090

theorem solution_pairs (x y : ℝ) (h1 : x + y = 1) (h2 : x^3 + y^3 = 19) :
  (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_l60_6090


namespace NUMINAMATH_CALUDE_arrangement_count_l60_6052

theorem arrangement_count (n_boys n_girls : ℕ) (h_boys : n_boys = 5) (h_girls : n_girls = 6) :
  (Nat.factorial n_girls) * (Nat.choose (n_girls + 1) n_boys) * (Nat.factorial n_boys) =
  (Nat.factorial n_girls) * 2520 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l60_6052


namespace NUMINAMATH_CALUDE_proposition_truth_l60_6011

-- Define the propositions
def proposition1 (m : ℝ) : Prop := m > 0 ↔ ∃ (x y : ℝ), x^2 + m*y^2 = 1 ∧ ¬(x^2 + y^2 = 1)

def proposition2 (a : ℝ) : Prop := (a = 1 → ∃ (k : ℝ), ∀ (x y : ℝ), a*x + y - 1 = k*(x + a*y - 2)) ∧
                                   ¬(∀ (a : ℝ), a = 1 → ∃ (k : ℝ), ∀ (x y : ℝ), a*x + y - 1 = k*(x + a*y - 2))

def proposition3 (m : ℝ) : Prop := (∀ (x₁ x₂ : ℝ), x₁ < x₂ → x₁^3 + m*x₁ < x₂^3 + m*x₂) ↔ m > 0

def proposition4 (p q : Prop) : Prop := ((p ∨ q) → (p ∧ q)) ∧ ((p ∧ q) → (p ∨ q))

-- Theorem stating which propositions are true and which are false
theorem proposition_truth : 
  (∃ (m : ℝ), ¬proposition1 m) ∧ 
  (∀ (a : ℝ), proposition2 a) ∧
  (∃ (m : ℝ), ¬proposition3 m) ∧
  (∀ (p q : Prop), proposition4 p q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l60_6011


namespace NUMINAMATH_CALUDE_ratio_problem_l60_6021

theorem ratio_problem (A B C : ℚ) 
  (sum_eq : A + B + C = 98)
  (ratio_bc : B / C = 5 / 8)
  (b_eq : B = 30) :
  A / B = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l60_6021


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_condition_l60_6051

theorem quadratic_perfect_square_condition (a b : ℤ) :
  (∃ S : Set ℤ, (Set.Infinite S) ∧ (∀ x ∈ S, ∃ y : ℤ, x^2 + a*x + b = y^2)) ↔ a^2 = 4*b :=
sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_condition_l60_6051


namespace NUMINAMATH_CALUDE_P_intersect_Q_eq_P_l60_6025

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}

theorem P_intersect_Q_eq_P : P ∩ Q = P := by
  sorry

end NUMINAMATH_CALUDE_P_intersect_Q_eq_P_l60_6025


namespace NUMINAMATH_CALUDE_degree_of_example_monomial_l60_6095

/-- Represents a monomial with coefficient and exponents for x and y -/
structure Monomial where
  coeff : ℤ
  x_exp : ℕ
  y_exp : ℕ

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ := m.x_exp + m.y_exp

/-- The specific monomial -5x^2y -/
def example_monomial : Monomial := ⟨-5, 2, 1⟩

theorem degree_of_example_monomial :
  degree example_monomial = 3 := by sorry

end NUMINAMATH_CALUDE_degree_of_example_monomial_l60_6095


namespace NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l60_6088

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 9 = 0

-- Define the ellipse C
def ellipse_C (x y θ : ℝ) : Prop := x = 2 * Real.sqrt 3 * Real.cos θ ∧ y = Real.sqrt 3 * Real.sin θ

-- Define the foci
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define a point on the line l
def point_on_l (M : ℝ × ℝ) : Prop := line_l M.1 M.2

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 45 + y^2 / 36 = 1

-- Theorem statement
theorem shortest_major_axis_ellipse :
  ∀ (M : ℝ × ℝ), point_on_l M →
  ∀ (x y : ℝ), ellipse_equation x y →
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
    (x^2 / a^2 + y^2 / b^2 = 1) ∧
    (∀ (a' b' : ℝ), a' > b' ∧ b' > 0 →
      (x^2 / a'^2 + y^2 / b'^2 = 1) →
      point_on_l (x, y) →
      a ≤ a') :=
by sorry

end NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l60_6088


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l60_6040

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.log (Real.cos x)) / x else 0

theorem derivative_f_at_zero :
  deriv f 0 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l60_6040


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l60_6079

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4 / (1 - r))) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l60_6079


namespace NUMINAMATH_CALUDE_distance_sum_squares_l60_6099

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := x - m * y + 1 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + y - m + 3 = 0

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, -3)

-- Theorem statement
theorem distance_sum_squares (m : ℝ) (P : ℝ × ℝ) :
  l₁ m P.1 P.2 ∧ l₂ m P.1 P.2 →
  l₁ m A.1 A.2 ∧ l₂ m B.1 B.2 →
  (P.1 - A.1)^2 + (P.2 - A.2)^2 + (P.1 - B.1)^2 + (P.2 - B.2)^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_distance_sum_squares_l60_6099


namespace NUMINAMATH_CALUDE_lines_skew_iff_a_neq_4_l60_6070

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ¬ (∃ (t u : ℝ), l1.point + t • l1.direction = l2.point + u • l2.direction)

/-- The main theorem -/
theorem lines_skew_iff_a_neq_4 (a : ℝ) :
  let l1 : Line3D := ⟨(2, 3, a), (3, 4, 5)⟩
  let l2 : Line3D := ⟨(5, 2, 1), (6, 3, 2)⟩
  are_skew l1 l2 ↔ a ≠ 4 := by
  sorry


end NUMINAMATH_CALUDE_lines_skew_iff_a_neq_4_l60_6070


namespace NUMINAMATH_CALUDE_calculation_result_l60_6063

theorem calculation_result (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : m = -2)  -- m is a negative number with an absolute value of 2
  : m + c * d + a + b + (c * d) ^ 2010 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l60_6063


namespace NUMINAMATH_CALUDE_marcella_shoes_l60_6053

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_remaining_pairs (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - shoes_lost

theorem marcella_shoes :
  max_remaining_pairs 20 9 = 11 := by
  sorry

end NUMINAMATH_CALUDE_marcella_shoes_l60_6053


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l60_6071

-- Define an isosceles triangle with side lengths 3 and 6
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 3 ∧ b = 6 ∧ c = 6) ∨ (a = 6 ∧ b = 3 ∧ c = 3)

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → Perimeter a b c = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l60_6071


namespace NUMINAMATH_CALUDE_scientific_notation_of_1650000000_l60_6080

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The original number to be expressed in scientific notation -/
def originalNumber : ℝ := 1650000000

/-- The number of significant figures to keep -/
def sigFigs : ℕ := 3

theorem scientific_notation_of_1650000000 :
  toScientificNotation originalNumber sigFigs =
    ScientificNotation.mk 1.65 9 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1650000000_l60_6080


namespace NUMINAMATH_CALUDE_grunters_lineup_count_l60_6038

/-- Represents the number of players in each position --/
structure TeamComposition where
  guards : ℕ
  forwards : ℕ
  centers : ℕ

/-- Represents the starting lineup requirements --/
structure LineupRequirements where
  total_starters : ℕ
  guards : ℕ
  forwards : ℕ
  centers : ℕ

/-- Calculates the number of possible lineups --/
def calculate_lineups (team : TeamComposition) (req : LineupRequirements) : ℕ :=
  (team.guards.choose req.guards) * (team.forwards.choose req.forwards) * (team.centers.choose req.centers)

theorem grunters_lineup_count :
  let team : TeamComposition := ⟨5, 6, 3⟩  -- 4+1 guards, 5+1 forwards, 3 centers
  let req : LineupRequirements := ⟨5, 2, 2, 1⟩  -- 5 total, 2 guards, 2 forwards, 1 center
  calculate_lineups team req = 60 := by
  sorry

#check grunters_lineup_count

end NUMINAMATH_CALUDE_grunters_lineup_count_l60_6038


namespace NUMINAMATH_CALUDE_power_subtraction_l60_6072

theorem power_subtraction : (81 : ℝ) ^ (1/4) - (16 : ℝ) ^ (1/2) = -1 := by sorry

end NUMINAMATH_CALUDE_power_subtraction_l60_6072


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_product_l60_6007

theorem arithmetic_sequence_sum_product (a b c : ℝ) : 
  (b - a = c - b) →  -- arithmetic sequence condition
  (a + b + c = 12) →  -- sum condition
  (a * b * c = 48) →  -- product condition
  ((a = 2 ∧ b = 4 ∧ c = 6) ∨ (a = 6 ∧ b = 4 ∧ c = 2)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_product_l60_6007


namespace NUMINAMATH_CALUDE_sum_of_roots_l60_6012

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  c^2 + a*c + b = 0 →
  d^2 + a*d + b = 0 →
  a^2 + c*a + d = 0 →
  b^2 + c*b + d = 0 →
  a + b + c + d = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l60_6012


namespace NUMINAMATH_CALUDE_paint_usage_l60_6094

theorem paint_usage (total_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ)
  (h1 : total_paint = 360)
  (h2 : first_week_fraction = 1 / 9)
  (h3 : second_week_fraction = 1 / 5) :
  let first_week_usage := first_week_fraction * total_paint
  let remaining_paint := total_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  let total_usage := first_week_usage + second_week_usage
  total_usage = 104 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_l60_6094


namespace NUMINAMATH_CALUDE_complementary_angle_of_25_41_l60_6006

-- Define a type for angles in degrees and minutes
structure Angle where
  degrees : ℕ
  minutes : ℕ

-- Define addition for Angle
def Angle.add (a b : Angle) : Angle :=
  let totalMinutes := a.minutes + b.minutes
  let extraDegrees := totalMinutes / 60
  { degrees := a.degrees + b.degrees + extraDegrees
  , minutes := totalMinutes % 60 }

-- Define subtraction for Angle
def Angle.sub (a b : Angle) : Angle :=
  let totalMinutes := (a.degrees * 60 + a.minutes) - (b.degrees * 60 + b.minutes)
  { degrees := totalMinutes / 60
  , minutes := totalMinutes % 60 }

-- Define the given angle
def givenAngle : Angle := { degrees := 25, minutes := 41 }

-- Define 90 degrees
def rightAngle : Angle := { degrees := 90, minutes := 0 }

-- Theorem statement
theorem complementary_angle_of_25_41 :
  Angle.sub rightAngle givenAngle = { degrees := 64, minutes := 19 } := by
  sorry


end NUMINAMATH_CALUDE_complementary_angle_of_25_41_l60_6006


namespace NUMINAMATH_CALUDE_matchboxes_per_box_l60_6092

/-- Proves that the number of matchboxes in each box is 20, given the total number of boxes,
    sticks per matchbox, and total number of sticks. -/
theorem matchboxes_per_box 
  (total_boxes : ℕ) 
  (sticks_per_matchbox : ℕ) 
  (total_sticks : ℕ) 
  (h1 : total_boxes = 4)
  (h2 : sticks_per_matchbox = 300)
  (h3 : total_sticks = 24000) :
  total_sticks / sticks_per_matchbox / total_boxes = 20 := by
  sorry

#eval 24000 / 300 / 4  -- Should output 20

end NUMINAMATH_CALUDE_matchboxes_per_box_l60_6092


namespace NUMINAMATH_CALUDE_cos_2alpha_problem_l60_6055

theorem cos_2alpha_problem (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →
  Real.cos (2 * α) = -(Real.sqrt 5 / 3) := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_problem_l60_6055


namespace NUMINAMATH_CALUDE_rectangle_dimension_l60_6023

theorem rectangle_dimension (x : ℝ) : 
  (3*x - 5 > 0) ∧ (x + 7 > 0) ∧ ((3*x - 5) * (x + 7) = 15*x - 14) → x = 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_l60_6023


namespace NUMINAMATH_CALUDE_labor_practice_problem_l60_6056

-- Define the problem parameters
def type_a_capacity : ℕ := 35
def type_b_capacity : ℕ := 30
def type_a_rental : ℕ := 400
def type_b_rental : ℕ := 320
def max_rental : ℕ := 3000

-- Define the theorem
theorem labor_practice_problem :
  ∃ (teachers students : ℕ) (type_a_buses : ℕ),
    -- Conditions from the problem
    students = 30 * teachers + 7 ∧
    31 * teachers = students + 1 ∧
    -- Solution part 1
    teachers = 8 ∧
    students = 247 ∧
    -- Solution part 2
    3 ≤ type_a_buses ∧ type_a_buses ≤ 5 ∧
    type_a_capacity * type_a_buses + type_b_capacity * (teachers - type_a_buses) ≥ teachers + students ∧
    type_a_rental * type_a_buses + type_b_rental * (teachers - type_a_buses) ≤ max_rental ∧
    -- Solution part 3
    (∀ m : ℕ, 3 ≤ m ∧ m ≤ 5 →
      type_a_rental * 3 + type_b_rental * 5 ≤ type_a_rental * m + type_b_rental * (8 - m)) :=
by sorry


end NUMINAMATH_CALUDE_labor_practice_problem_l60_6056


namespace NUMINAMATH_CALUDE_tea_mixture_price_l60_6086

theorem tea_mixture_price (price1 price2 mixture_price : ℚ) (ratio1 ratio2 ratio3 : ℚ) :
  price1 = 126 →
  price2 = 135 →
  mixture_price = 153 →
  ratio1 = 1 →
  ratio2 = 1 →
  ratio3 = 2 →
  ∃ price3 : ℚ,
    price3 = 175.5 ∧
    (ratio1 * price1 + ratio2 * price2 + ratio3 * price3) / (ratio1 + ratio2 + ratio3) = mixture_price :=
by sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l60_6086


namespace NUMINAMATH_CALUDE_greater_root_of_quadratic_l60_6067

theorem greater_root_of_quadratic (x : ℝ) :
  x^2 - 5*x - 36 = 0 → x ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greater_root_of_quadratic_l60_6067


namespace NUMINAMATH_CALUDE_power_of_64_l60_6037

theorem power_of_64 : (64 : ℝ) ^ (3/2) = 512 := by sorry

end NUMINAMATH_CALUDE_power_of_64_l60_6037


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l60_6020

theorem repeating_decimal_to_fraction :
  ∀ (x : ℚ), (∃ (n : ℕ), x = 0.56 + 0.0056 * (1 - (1/100)^n) / (1 - 1/100)) →
  x = 56/99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l60_6020


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_3_l60_6044

theorem no_solution_implies_a_leq_3 :
  (∀ x : ℝ, ¬(x > 3 ∧ x < a)) → a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_3_l60_6044


namespace NUMINAMATH_CALUDE_volume_removed_percentage_l60_6085

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def prismVolume (d : PrismDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the side length of the small cube removed from each corner -/
def smallCubeSide : ℝ := 4

/-- Calculates the volume of the small cube -/
def smallCubeVolume : ℝ :=
  smallCubeSide ^ 3

/-- The number of corners in a rectangular prism -/
def numCorners : ℕ := 8

/-- Theorem: The percentage of volume removed from the rectangular prism -/
theorem volume_removed_percentage
  (d : PrismDimensions)
  (h1 : d.length = 20)
  (h2 : d.width = 14)
  (h3 : d.height = 12) :
  (numCorners * smallCubeVolume) / (prismVolume d) * 100 = 512 / 3360 * 100 := by
  sorry

end NUMINAMATH_CALUDE_volume_removed_percentage_l60_6085


namespace NUMINAMATH_CALUDE_janice_age_l60_6030

def current_year : ℕ := 2021
def mark_birth_year : ℕ := 1976

def graham_age_difference : ℕ := 3

theorem janice_age :
  let mark_age : ℕ := current_year - mark_birth_year
  let graham_age : ℕ := mark_age - graham_age_difference
  let janice_age : ℕ := graham_age / 2
  janice_age = 21 := by sorry

end NUMINAMATH_CALUDE_janice_age_l60_6030


namespace NUMINAMATH_CALUDE_food_waste_scientific_notation_l60_6046

/-- The amount of food wasted in China annually in kilograms -/
def food_waste : ℕ := 500000000000

/-- Prove that the food waste in China is equivalent to 5 × 10^10 kg -/
theorem food_waste_scientific_notation : food_waste = 5 * (10 ^ 10) := by
  sorry

end NUMINAMATH_CALUDE_food_waste_scientific_notation_l60_6046


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l60_6065

theorem bernoulli_inequality (x r : ℝ) (hx : x > 0) (hr : r > 1) :
  (1 + x)^r > 1 + r * x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l60_6065


namespace NUMINAMATH_CALUDE_fathers_age_l60_6091

/-- The father's age given the son's age ratio conditions -/
theorem fathers_age (man_age : ℝ) (father_age : ℝ) : 
  man_age = (2 / 5) * father_age ∧ 
  man_age + 14 = (1 / 2) * (father_age + 14) → 
  father_age = 70 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l60_6091


namespace NUMINAMATH_CALUDE_election_winner_percentage_l60_6087

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 930 →
  margin = 360 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 62 / 100 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l60_6087


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l60_6024

/-- The distance from the origin to the point (12, -5) in a rectangular coordinate system is 13 units. -/
theorem distance_from_origin_to_point : Real.sqrt (12^2 + (-5)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l60_6024


namespace NUMINAMATH_CALUDE_sages_can_succeed_l60_6075

/-- Represents the color of a hat -/
def HatColor := Fin 1000

/-- Represents the signal a sage can show (white or black card) -/
def Signal := Bool

/-- Represents the configuration of hats on the sages -/
def HatConfiguration := Fin 11 → HatColor

/-- A strategy is a function that takes the colors of the other hats and returns a signal -/
def Strategy := (Fin 10 → HatColor) → Signal

/-- The result of applying a strategy is a function that determines the hat color based on the signals of others -/
def StrategyResult := (Fin 10 → Signal) → HatColor

/-- A successful strategy correctly determines the hat color for all possible configurations -/
def SuccessfulStrategy (strategy : Fin 11 → Strategy) (result : Fin 11 → StrategyResult) : Prop :=
  ∀ (config : HatConfiguration),
    ∀ (i : Fin 11),
      result i (λ j => if j < i then strategy j (λ k => config (k.succ)) 
                       else strategy j.succ (λ k => if k < j then config k else config k.succ)) = config i

theorem sages_can_succeed : ∃ (strategy : Fin 11 → Strategy) (result : Fin 11 → StrategyResult),
  SuccessfulStrategy strategy result := by
  sorry

end NUMINAMATH_CALUDE_sages_can_succeed_l60_6075


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l60_6001

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem starting_lineup_combinations : choose 13 4 = 715 := by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l60_6001


namespace NUMINAMATH_CALUDE_playground_max_area_l60_6042

theorem playground_max_area :
  ∀ (width height : ℕ),
    width + height = 75 →
    width * height ≤ 1406 :=
by
  sorry

end NUMINAMATH_CALUDE_playground_max_area_l60_6042


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l60_6019

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns true if all numbers in a list are non-prime, false otherwise -/
def allNonPrime (list : List ℕ) : Prop := sorry

theorem smallest_prime_after_seven_nonprimes :
  ∃ (n : ℕ), 
    (isPrime (nthPrime n)) ∧ 
    (allNonPrime [nthPrime (n-1) + 1, nthPrime (n-1) + 2, nthPrime (n-1) + 3, 
                  nthPrime (n-1) + 4, nthPrime (n-1) + 5, nthPrime (n-1) + 6, 
                  nthPrime (n-1) + 7]) ∧
    (nthPrime n = 67) ∧
    (∀ (m : ℕ), m < n → 
      ¬(isPrime (nthPrime m) ∧ 
        allNonPrime [nthPrime (m-1) + 1, nthPrime (m-1) + 2, nthPrime (m-1) + 3, 
                     nthPrime (m-1) + 4, nthPrime (m-1) + 5, nthPrime (m-1) + 6, 
                     nthPrime (m-1) + 7])) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l60_6019


namespace NUMINAMATH_CALUDE_complex_sum_argument_l60_6084

theorem complex_sum_argument : ∃ (r : ℝ), 
  Complex.exp (11 * Real.pi * Complex.I / 60) + 
  Complex.exp (23 * Real.pi * Complex.I / 60) + 
  Complex.exp (35 * Real.pi * Complex.I / 60) + 
  Complex.exp (47 * Real.pi * Complex.I / 60) + 
  Complex.exp (59 * Real.pi * Complex.I / 60) + 
  Complex.exp (Real.pi * Complex.I / 60) = 
  r * Complex.exp (7 * Real.pi * Complex.I / 12) :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_argument_l60_6084


namespace NUMINAMATH_CALUDE_unique_number_l60_6064

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000) ∧ (n < 10000) ∧
  (n % 10 = (n / 100) % 10) ∧
  (n - (n % 10 * 1000 + (n / 10) % 10 * 100 + (n / 100) % 10 * 10 + n / 1000) = 7812)

theorem unique_number : ∃! n : ℕ, is_valid_number n ∧ n = 1979 :=
  sorry

end NUMINAMATH_CALUDE_unique_number_l60_6064


namespace NUMINAMATH_CALUDE_johns_piggy_bank_l60_6036

theorem johns_piggy_bank (quarters dimes nickels : ℕ) : 
  quarters = 22 →
  dimes = quarters + 3 →
  nickels = quarters - 6 →
  quarters + dimes + nickels = 63 := by
sorry

end NUMINAMATH_CALUDE_johns_piggy_bank_l60_6036


namespace NUMINAMATH_CALUDE_square_minus_product_equals_one_l60_6098

theorem square_minus_product_equals_one (a : ℝ) (h : a = -4) : a^2 - (a+1)*(a-1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_equals_one_l60_6098


namespace NUMINAMATH_CALUDE_function_identity_l60_6069

theorem function_identity (f : ℝ → ℝ) (h : ∀ x, 2 * f x - f (-x) = 3 * x) :
  ∀ x, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l60_6069
