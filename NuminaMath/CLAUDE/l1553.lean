import Mathlib

namespace NUMINAMATH_CALUDE_exactly_three_props_true_l1553_155336

/-- Property P for a sequence -/
def has_property_P (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i → i < j → j ≤ n → (∃ k ≤ n, a k = a j + a i) ∨ (∃ k ≤ n, a k = a j - a i)

/-- The sequence is strictly increasing and starts with a non-negative number -/
def is_valid_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j) ∧ 0 ≤ a 1

/-- Proposition 1: The sequence 0, 2, 4, 6 has property P -/
def prop_1 : Prop :=
  let a : ℕ → ℕ := fun i => 2 * (i - 1)
  has_property_P a 4

/-- Proposition 2: If sequence A has property P, then a₁ = 0 -/
def prop_2 : Prop :=
  ∀ a : ℕ → ℕ, ∀ n ≥ 3, is_valid_sequence a n → has_property_P a n → a 1 = 0

/-- Proposition 3: If sequence A has property P and a₁ ≠ 0, then aₙ - aₙ₋ₖ = aₖ for k = 1, 2, ..., n-1 -/
def prop_3 : Prop :=
  ∀ a : ℕ → ℕ, ∀ n ≥ 3, is_valid_sequence a n → has_property_P a n → a 1 ≠ 0 →
    ∀ k, 1 ≤ k → k < n → a n - a (n - k) = a k

/-- Proposition 4: If the sequence a₁, a₂, a₃ (0 ≤ a₁ < a₂ < a₃) has property P, then a₃ = a₁ + a₂ -/
def prop_4 : Prop :=
  ∀ a : ℕ → ℕ, is_valid_sequence a 3 → has_property_P a 3 → a 3 = a 1 + a 2

theorem exactly_three_props_true : (prop_1 ∧ ¬prop_2 ∧ prop_3 ∧ prop_4) := by sorry

end NUMINAMATH_CALUDE_exactly_three_props_true_l1553_155336


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l1553_155372

/-- Given a circle C₁ with equation (x+1)² + (y-1)² = 1 and a line L with equation x - y - 1 = 0,
    the circle C₂ symmetric to C₁ about L has equation (x-2)² + (y+2)² = 1 -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∀ X Y : ℝ, (X + 1)^2 + (Y - 1)^2 = 1 → 
    (X - Y - 1 = 0 → (x + 1 = Y ∧ y - 1 = X) → (x - 2)^2 + (y + 2)^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l1553_155372


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l1553_155367

/-- Given a circle with center (a, b) and radius r, returns the equation of the circle symmetric to it with respect to the line y = x -/
def symmetricCircle (a b r : ℝ) : (ℝ × ℝ → Prop) :=
  fun p => (p.1 - b)^2 + (p.2 - a)^2 = r^2

/-- The original circle (x-1)^2 + (y-2)^2 = 1 -/
def originalCircle : (ℝ × ℝ → Prop) :=
  fun p => (p.1 - 1)^2 + (p.2 - 2)^2 = 1

theorem symmetric_circle_equation :
  symmetricCircle 1 2 1 = fun p => (p.1 - 2)^2 + (p.2 - 1)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l1553_155367


namespace NUMINAMATH_CALUDE_complex_cube_equality_l1553_155348

theorem complex_cube_equality (a b c : ℝ) : 
  ((2 * a - b - c : ℂ) + (b - c) * Complex.I * Real.sqrt 3) ^ 3 = 
  ((2 * b - c - a : ℂ) + (c - a) * Complex.I * Real.sqrt 3) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_equality_l1553_155348


namespace NUMINAMATH_CALUDE_circle_equation_l1553_155340

/-- Given points A and B, and a circle whose center lies on a line, prove the equation of the circle. -/
theorem circle_equation (A B C : ℝ × ℝ) (r : ℝ) : 
  A = (1, -1) →
  B = (-1, 1) →
  C.1 + C.2 = 2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 →
  ∀ x y : ℝ, (x - C.1)^2 + (y - C.2)^2 = r^2 ↔ (x - 1)^2 + (y - 1)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1553_155340


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1553_155346

theorem inverse_variation_problem (a b : ℝ) (k : ℝ) (h1 : a * b^3 = k) (h2 : 8 * 1^3 = k) :
  a * 4^3 = k → a = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1553_155346


namespace NUMINAMATH_CALUDE_identical_solutions_iff_k_neg_one_l1553_155338

/-- 
Proves that the equations y = x^2 and y = 2x + k have two identical solutions 
if and only if k = -1.
-/
theorem identical_solutions_iff_k_neg_one (k : ℝ) : 
  (∃ x y : ℝ, y = x^2 ∧ y = 2*x + k ∧ 
   (∀ x' y' : ℝ, y' = x'^2 ∧ y' = 2*x' + k → x' = x ∧ y' = y)) ↔ 
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_identical_solutions_iff_k_neg_one_l1553_155338


namespace NUMINAMATH_CALUDE_downstream_speed_calculation_l1553_155357

/-- The speed downstream of a boat, given its speed in still water and the speed of the current. -/
def speed_downstream (speed_still_water speed_current : ℝ) : ℝ :=
  speed_still_water + speed_current

/-- Theorem stating that the speed downstream is 77 kmph when the boat's speed in still water is 60 kmph and the current speed is 17 kmph. -/
theorem downstream_speed_calculation :
  speed_downstream 60 17 = 77 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_calculation_l1553_155357


namespace NUMINAMATH_CALUDE_percentage_male_worker_ants_l1553_155312

theorem percentage_male_worker_ants (total_ants : ℕ) (female_worker_ants : ℕ) 
  (h1 : total_ants = 110)
  (h2 : female_worker_ants = 44) : 
  (((total_ants / 2 - female_worker_ants : ℚ) / (total_ants / 2)) * 100 = 20) := by
  sorry

end NUMINAMATH_CALUDE_percentage_male_worker_ants_l1553_155312


namespace NUMINAMATH_CALUDE_circle_radius_from_chords_l1553_155383

/-- Given a circle with two chords AB and AC, where AB = a, AC = b, and the length of arc AC is twice the length of arc AB, the radius of the circle is a²/√(4a² - b²). -/
theorem circle_radius_from_chords (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : b < 2*a) : 
  ∃ (R : ℝ), R > 0 ∧ R = a^2 / Real.sqrt (4*a^2 - b^2) := by
  sorry


end NUMINAMATH_CALUDE_circle_radius_from_chords_l1553_155383


namespace NUMINAMATH_CALUDE_eldest_age_l1553_155382

theorem eldest_age (x : ℝ) (h1 : 5*x - 7 + 7*x - 7 + 8*x - 7 = 59) : 8*x = 32 := by
  sorry

#check eldest_age

end NUMINAMATH_CALUDE_eldest_age_l1553_155382


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l1553_155328

theorem parametric_to_ordinary_equation (t : ℝ) :
  let x := Real.exp t + Real.exp (-t)
  let y := 2 * (Real.exp t - Real.exp (-t))
  (x^2 / 4) - (y^2 / 16) = 1 ∧ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l1553_155328


namespace NUMINAMATH_CALUDE_watch_correction_proof_l1553_155311

/-- Represents the time loss of a watch in minutes per day -/
def timeLossPerDay : ℝ := 3

/-- Represents the number of days between April 1 at 12 noon and April 10 at 6 P.M. -/
def daysElapsed : ℝ := 9.25

/-- Calculates the positive correction in minutes for the watch -/
def watchCorrection (loss : ℝ) (days : ℝ) : ℝ := loss * days

theorem watch_correction_proof :
  watchCorrection timeLossPerDay daysElapsed = 27.75 := by
  sorry

end NUMINAMATH_CALUDE_watch_correction_proof_l1553_155311


namespace NUMINAMATH_CALUDE_candidate_a_democratic_vote_percentage_l1553_155398

theorem candidate_a_democratic_vote_percentage
  (total_voters : ℝ)
  (democrat_percentage : ℝ)
  (republican_percentage : ℝ)
  (republican_vote_for_a : ℝ)
  (total_vote_for_a : ℝ)
  (h1 : democrat_percentage = 0.70)
  (h2 : republican_percentage = 1 - democrat_percentage)
  (h3 : republican_vote_for_a = 0.30)
  (h4 : total_vote_for_a = 0.65) :
  ∃ (democrat_vote_for_a : ℝ),
    democrat_vote_for_a * democrat_percentage +
    republican_vote_for_a * republican_percentage = total_vote_for_a ∧
    democrat_vote_for_a = 0.80 :=
sorry

end NUMINAMATH_CALUDE_candidate_a_democratic_vote_percentage_l1553_155398


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1553_155393

theorem inequality_equivalence (x : ℝ) : 
  -2 < (x^2 - 18*x + 24) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 18*x + 24) / (x^2 - 4*x + 8) < 2 ↔ 
  -2 < x ∧ x < 10/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1553_155393


namespace NUMINAMATH_CALUDE_elephants_after_three_years_is_zero_l1553_155332

/-- Represents the different types of animals in the zoo -/
inductive Animal
| Giraffe
| Penguin
| Elephant
| Lion
| Bear

/-- Represents the state of the zoo -/
structure ZooState where
  animalCount : Animal → ℕ
  budget : ℕ

/-- The cost of each animal type -/
def animalCost : Animal → ℕ
| Animal.Giraffe => 1000
| Animal.Penguin => 500
| Animal.Elephant => 1200
| Animal.Lion => 1100
| Animal.Bear => 1300

/-- The initial state of the zoo -/
def initialState : ZooState :=
  { animalCount := λ a => match a with
      | Animal.Giraffe => 5
      | Animal.Penguin => 10
      | Animal.Elephant => 0
      | Animal.Lion => 5
      | Animal.Bear => 0
    budget := 10000 }

/-- The maximum capacity of the zoo -/
def maxCapacity : ℕ := 300

/-- Theorem stating that the number of elephants after three years is zero -/
theorem elephants_after_three_years_is_zero :
  (initialState.animalCount Animal.Elephant) = 0 → 
  ∀ (finalState : ZooState),
    (finalState.animalCount Animal.Elephant) = 0 := by
  sorry

#check elephants_after_three_years_is_zero

end NUMINAMATH_CALUDE_elephants_after_three_years_is_zero_l1553_155332


namespace NUMINAMATH_CALUDE_total_handshakes_l1553_155323

/-- Represents the number of people in the meeting -/
def total_people : ℕ := 40

/-- Represents the number of people who mostly know each other -/
def group1_size : ℕ := 25

/-- Represents the number of strangers within group1 -/
def strangers_in_group1 : ℕ := 5

/-- Represents the number of people who know no one -/
def group2_size : ℕ := 15

/-- Calculates the number of handshakes between strangers in group1 -/
def handshakes_in_group1 : ℕ := strangers_in_group1 * (strangers_in_group1 - 1) / 2

/-- Calculates the number of handshakes involving group2 -/
def handshakes_involving_group2 : ℕ := group2_size * (total_people - 1)

/-- The main theorem stating the total number of handshakes -/
theorem total_handshakes : 
  handshakes_in_group1 + handshakes_involving_group2 = 595 := by sorry

end NUMINAMATH_CALUDE_total_handshakes_l1553_155323


namespace NUMINAMATH_CALUDE_square_root_sum_implies_product_l1553_155316

theorem square_root_sum_implies_product (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8) →
  ((10 + x) * (30 - x) = 144) := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_implies_product_l1553_155316


namespace NUMINAMATH_CALUDE_weekly_pig_feed_l1553_155391

def feed_per_pig_per_day : ℕ := 10
def number_of_pigs : ℕ := 2
def days_in_week : ℕ := 7

theorem weekly_pig_feed : 
  feed_per_pig_per_day * number_of_pigs * days_in_week = 140 := by
  sorry

end NUMINAMATH_CALUDE_weekly_pig_feed_l1553_155391


namespace NUMINAMATH_CALUDE_exists_function_satisfying_properties_l1553_155307

/-- A strictly increasing function from natural numbers to natural numbers -/
def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m < n → f m < f n

/-- The property that f(f(f(n))) = n + 2f(n) for all n -/
def TripleCompositionProperty (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f (f (f n)) = n + 2 * (f n)

/-- The main theorem stating the existence of a function satisfying both properties -/
theorem exists_function_satisfying_properties :
  ∃ f : ℕ → ℕ, StrictlyIncreasing f ∧ TripleCompositionProperty f :=
sorry

end NUMINAMATH_CALUDE_exists_function_satisfying_properties_l1553_155307


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_ln_positive_l1553_155313

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) := by sorry

theorem negation_of_ln_positive :
  (¬ ∃ x > 0, Real.log x > 0) ↔ (∀ x > 0, Real.log x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_ln_positive_l1553_155313


namespace NUMINAMATH_CALUDE_multiply_binomials_l1553_155324

theorem multiply_binomials (a b : ℝ) : (3*a + 2*b) * (a - 2*b) = 3*a^2 - 4*a*b - 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_binomials_l1553_155324


namespace NUMINAMATH_CALUDE_coffee_mix_ratio_l1553_155377

theorem coffee_mix_ratio (x y : ℝ) (h : x > 0 ∧ y > 0) :
  (50 * x + 40 * y) / (x + y) = (55 * x + 34 * y) / (x + y) ↔ x / y = 6 / 5 :=
by sorry

end NUMINAMATH_CALUDE_coffee_mix_ratio_l1553_155377


namespace NUMINAMATH_CALUDE_probability_two_green_marbles_l1553_155378

def red_marbles : ℕ := 3
def green_marbles : ℕ := 4
def white_marbles : ℕ := 13

def total_marbles : ℕ := red_marbles + green_marbles + white_marbles

theorem probability_two_green_marbles :
  (green_marbles / total_marbles) * ((green_marbles - 1) / (total_marbles - 1)) = 3 / 95 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_green_marbles_l1553_155378


namespace NUMINAMATH_CALUDE_books_to_buy_l1553_155376

/-- Given that 3 books cost $18.72 and you have $37.44, prove that you can buy 6 books. -/
theorem books_to_buy (cost_of_three : ℝ) (total_money : ℝ) : 
  cost_of_three = 18.72 → total_money = 37.44 → 
  (total_money / (cost_of_three / 3)) = 6 := by
sorry

end NUMINAMATH_CALUDE_books_to_buy_l1553_155376


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1553_155362

theorem inequality_solution_set : 
  {x : ℝ | x + 7 > -2*x + 1} = {x : ℝ | x > -2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1553_155362


namespace NUMINAMATH_CALUDE_faye_science_problems_l1553_155387

theorem faye_science_problems :
  ∀ (math_problems finished_problems remaining_problems : ℕ),
    math_problems = 46 →
    finished_problems = 40 →
    remaining_problems = 15 →
    math_problems + (finished_problems + remaining_problems - math_problems) = 
      finished_problems + remaining_problems :=
by
  sorry

end NUMINAMATH_CALUDE_faye_science_problems_l1553_155387


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l1553_155305

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 49) 
  (h2 : x * y = 15) : 
  x^2 + y^2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l1553_155305


namespace NUMINAMATH_CALUDE_final_price_after_reductions_ball_price_reduction_l1553_155390

/-- Calculates the final price of an item after two successive price reductions -/
theorem final_price_after_reductions (original_price : ℝ) 
  (first_reduction_percent : ℝ) (second_reduction_percent : ℝ) : 
  original_price * (1 - first_reduction_percent / 100) * (1 - second_reduction_percent / 100) = 8 :=
by
  sorry

/-- The specific case of the ball price reduction problem -/
theorem ball_price_reduction : 
  20 * (1 - 20 / 100) * (1 - 50 / 100) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_final_price_after_reductions_ball_price_reduction_l1553_155390


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l1553_155386

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l1553_155386


namespace NUMINAMATH_CALUDE_food_bank_donation_ratio_l1553_155337

theorem food_bank_donation_ratio :
  let foster_chickens : ℕ := 45
  let american_water := 2 * foster_chickens
  let hormel_chickens := 3 * foster_chickens
  let del_monte_water := american_water - 30
  let total_items : ℕ := 375
  let boudin_chickens := total_items - (foster_chickens + american_water + hormel_chickens + del_monte_water)
  (boudin_chickens : ℚ) / hormel_chickens = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_food_bank_donation_ratio_l1553_155337


namespace NUMINAMATH_CALUDE_chessboard_uniquely_determined_l1553_155301

/-- Represents a chessboard with numbers 1 to 64 -/
def Chessboard := Fin 8 → Fin 8 → Fin 64

/-- The sum of numbers in a rectangle of two cells -/
def RectangleSum (board : Chessboard) (r1 c1 r2 c2 : Fin 8) : ℕ :=
  (board r1 c1).val + 1 + (board r2 c2).val + 1

/-- Predicate to check if two positions are on the same diagonal -/
def OnSameDiagonal (r1 c1 r2 c2 : Fin 8) : Prop :=
  r1 + c1 = r2 + c2 ∨ r1 + c2 = r2 + c1

/-- Main theorem -/
theorem chessboard_uniquely_determined 
  (board : Chessboard) 
  (sums_known : ∀ (r1 c1 r2 c2 : Fin 8), r1 = r2 ∧ c1.val + 1 = c2.val ∨ r1.val + 1 = r2.val ∧ c1 = c2 → 
    ∃ (s : ℕ), s = RectangleSum board r1 c1 r2 c2)
  (one_and_sixtyfour_on_diagonal : ∃ (r1 c1 r2 c2 : Fin 8), 
    board r1 c1 = 0 ∧ board r2 c2 = 63 ∧ OnSameDiagonal r1 c1 r2 c2) :
  ∀ (r c : Fin 8), ∃! (n : Fin 64), board r c = n :=
sorry

end NUMINAMATH_CALUDE_chessboard_uniquely_determined_l1553_155301


namespace NUMINAMATH_CALUDE_range_of_a_when_no_solutions_l1553_155347

/-- The range of a when x^2 + (2a-1)x + a ≠ 0 for all x ∈ (-2,0) -/
theorem range_of_a_when_no_solutions (a : ℝ) : 
  (∀ x ∈ Set.Ioo (-2) 0, x^2 + (2*a - 1)*x + a ≠ 0) ↔ 
  a ∈ Set.Icc 0 ((2 + Real.sqrt 3) / 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_no_solutions_l1553_155347


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1553_155364

theorem cone_lateral_surface_area 
  (r : ℝ) (V : ℝ) (h : ℝ) (l : ℝ) (S : ℝ) :
  r = 3 →
  V = 12 * Real.pi →
  V = (1/3) * Real.pi * r^2 * h →
  l^2 = r^2 + h^2 →
  S = Real.pi * r * l →
  S = 15 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1553_155364


namespace NUMINAMATH_CALUDE_ap_sum_terms_l1553_155344

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a₁ : ℤ     -- First term
  d : ℤ      -- Common difference

/-- Calculates the sum of the first n terms of an arithmetic progression -/
def sum_of_terms (ap : ArithmeticProgression) (n : ℕ) : ℤ :=
  n * (2 * ap.a₁ + (n - 1) * ap.d) / 2

/-- Theorem: The number of terms needed for the sum to equal 3069 in the given arithmetic progression is either 9 or 31 -/
theorem ap_sum_terms (ap : ArithmeticProgression) 
  (h1 : ap.a₁ = 429) 
  (h2 : ap.d = -22) : 
  (∃ n : ℕ, sum_of_terms ap n = 3069) → (n = 9 ∨ n = 31) :=
sorry

end NUMINAMATH_CALUDE_ap_sum_terms_l1553_155344


namespace NUMINAMATH_CALUDE_series_sum_equals_closed_form_l1553_155321

/-- The sum of the series Σ(n=1 to ∞) (-1)^(n+1)/(3n-2) -/
noncomputable def seriesSum : ℝ := ∑' n, ((-1 : ℝ)^(n+1)) / (3*n - 2)

/-- The closed form of the series sum -/
noncomputable def closedForm : ℝ := (1/3) * (Real.log 2 + 2 * Real.pi / Real.sqrt 3)

/-- Theorem stating that the series sum equals the closed form -/
theorem series_sum_equals_closed_form : seriesSum = closedForm := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_closed_form_l1553_155321


namespace NUMINAMATH_CALUDE_four_bottles_cost_l1553_155302

/-- The cost of a certain number of bottles of mineral water -/
def cost (bottles : ℕ) : ℚ :=
  if bottles = 3 then 3/2 else (3/2 * bottles) / 3

/-- Theorem: The cost of 4 bottles of mineral water is 2 euros -/
theorem four_bottles_cost : cost 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_four_bottles_cost_l1553_155302


namespace NUMINAMATH_CALUDE_y0_minus_one_is_perfect_square_l1553_155366

theorem y0_minus_one_is_perfect_square 
  (x y : ℕ → ℕ) 
  (h : ∀ n, (x n : ℝ) + Real.sqrt 2 * (y n) = Real.sqrt 2 * (3 + 2 * Real.sqrt 2) ^ (2 ^ n)) : 
  ∃ k : ℕ, y 0 - 1 = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_y0_minus_one_is_perfect_square_l1553_155366


namespace NUMINAMATH_CALUDE_garden_length_l1553_155334

/-- A rectangular garden with given perimeter and breadth has a specific length. -/
theorem garden_length (perimeter breadth : ℝ) (h_perimeter : perimeter = 600) (h_breadth : breadth = 150) :
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter → perimeter / 2 - breadth = 150 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l1553_155334


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1553_155300

/-- Given a point P with coordinates (4, 2a+10), prove that if P lies on the x-axis, then a = -5 -/
theorem point_on_x_axis (a : ℝ) : 
  let P : ℝ × ℝ := (4, 2*a + 10)
  (P.2 = 0) → a = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1553_155300


namespace NUMINAMATH_CALUDE_safari_count_l1553_155341

theorem safari_count (antelopes : ℕ) (h1 : antelopes = 80) : ∃ (rabbits hyenas wild_dogs leopards giraffes lions elephants zebras hippos : ℕ),
  rabbits = antelopes + 34 ∧
  hyenas = antelopes + rabbits - 42 ∧
  wild_dogs = hyenas + 50 ∧
  leopards * 2 = rabbits ∧
  giraffes = antelopes + 15 ∧
  lions = leopards + giraffes ∧
  elephants = 3 * lions ∧
  4 * zebras = 3 * antelopes ∧
  hippos = zebras + zebras / 10 ∧
  antelopes + rabbits + hyenas + wild_dogs + leopards + giraffes + lions + elephants + zebras + hippos = 1334 :=
by
  sorry


end NUMINAMATH_CALUDE_safari_count_l1553_155341


namespace NUMINAMATH_CALUDE_arithmetic_sequence_logarithm_l1553_155360

theorem arithmetic_sequence_logarithm (x : ℝ) : 
  (∃ r : ℝ, Real.log 2 + r = Real.log (2^x - 1) ∧ 
             Real.log (2^x - 1) + r = Real.log (2^x + 3)) → 
  x = Real.log 5 / Real.log 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_logarithm_l1553_155360


namespace NUMINAMATH_CALUDE_divisibility_problem_l1553_155335

theorem divisibility_problem :
  {n : ℤ | (n - 2) ∣ (n^2 + 3*n + 27)} = {1, 3, 39, -35} := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1553_155335


namespace NUMINAMATH_CALUDE_f_one_half_equals_sixteen_l1553_155325

-- Define the function f
noncomputable def f : ℝ → ℝ := fun t => 1 / ((1 - t) / 2)^2

-- State the theorem
theorem f_one_half_equals_sixteen :
  (∀ x, f (1 - 2 * x) = 1 / x^2) → f (1/2) = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_f_one_half_equals_sixteen_l1553_155325


namespace NUMINAMATH_CALUDE_zero_in_interval_l1553_155342

noncomputable def f (x : ℝ) := Real.exp x + 4 * x - 3

theorem zero_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo (1/4 : ℝ) (1/2 : ℝ) ∧ f x = 0 :=
by
  have h1 : f (1/4) < 0 := by sorry
  have h2 : f (1/2) > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1553_155342


namespace NUMINAMATH_CALUDE_custom_baseball_caps_l1553_155319

theorem custom_baseball_caps (jack_circumference bill_circumference : ℝ)
  (h1 : jack_circumference = 12)
  (h2 : bill_circumference = 10)
  (h3 : ∃ f : ℝ, charlie_circumference = f * jack_circumference + 9)
  (h4 : bill_circumference = (2/3) * charlie_circumference) :
  ∃ f : ℝ, charlie_circumference = f * jack_circumference + 9 ∧ f = (1/2) :=
by
  sorry
where
  charlie_circumference : ℝ := bill_circumference / (2/3)

end NUMINAMATH_CALUDE_custom_baseball_caps_l1553_155319


namespace NUMINAMATH_CALUDE_certain_amount_proof_l1553_155304

theorem certain_amount_proof (A : ℝ) : 
  (0.25 * 680 = 0.20 * 1000 - A) → A = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_proof_l1553_155304


namespace NUMINAMATH_CALUDE_worker_c_work_rate_l1553_155388

/-- Given workers A, B, and C, and their work rates, prove that C's work rate is 1/3 of the total work per hour. -/
theorem worker_c_work_rate
  (total_work : ℝ) -- Total work to be done
  (rate_a : ℝ) -- A's work rate
  (rate_b : ℝ) -- B's work rate
  (rate_c : ℝ) -- C's work rate
  (h1 : rate_a = total_work / 3) -- A can do the work in 3 hours
  (h2 : rate_b + rate_c = total_work / 2) -- B and C together can do the work in 2 hours
  (h3 : rate_a + rate_b = total_work / 2) -- A and B together can do the work in 2 hours
  : rate_c = total_work / 3 := by
  sorry

end NUMINAMATH_CALUDE_worker_c_work_rate_l1553_155388


namespace NUMINAMATH_CALUDE_ball_cost_l1553_155371

/-- Proves that if Kyoko buys 3 balls for a total cost of $4.62, then each ball costs $1.54. -/
theorem ball_cost (total_cost : ℝ) (num_balls : ℕ) (cost_per_ball : ℝ) 
  (h1 : total_cost = 4.62)
  (h2 : num_balls = 3)
  (h3 : cost_per_ball = total_cost / num_balls) : 
  cost_per_ball = 1.54 := by
  sorry

end NUMINAMATH_CALUDE_ball_cost_l1553_155371


namespace NUMINAMATH_CALUDE_negative_of_negative_two_l1553_155369

theorem negative_of_negative_two : -(Int.neg 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_of_negative_two_l1553_155369


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_zero_l1553_155375

-- Define a cubic polynomial
def cubic_polynomial (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem cubic_polynomial_sum_zero 
  (a b c d : ℝ) 
  (h1 : cubic_polynomial a b c d 0 = 2 * d)
  (h2 : cubic_polynomial a b c d 1 = 3 * d)
  (h3 : cubic_polynomial a b c d (-1) = 5 * d) :
  cubic_polynomial a b c d 3 + cubic_polynomial a b c d (-3) = 0 := by
sorry


end NUMINAMATH_CALUDE_cubic_polynomial_sum_zero_l1553_155375


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l1553_155355

theorem baseball_card_value_decrease (x : ℝ) : 
  (1 - x / 100) * (1 - 30 / 100) = 1 - 44.00000000000001 / 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l1553_155355


namespace NUMINAMATH_CALUDE_min_pig_count_l1553_155303

def is_valid_pig_count (pigs : ℕ) (total : ℕ) : Prop :=
  pigs > 0 ∧ total > 0 ∧ 
  (54 * total ≤ 100 * pigs) ∧ (100 * pigs ≤ 57 * total)

theorem min_pig_count :
  ∃ (min_pigs : ℕ), 
    (∃ (total : ℕ), is_valid_pig_count min_pigs total) ∧
    (∀ (pigs : ℕ), pigs < min_pigs → 
      ∀ (total : ℕ), ¬is_valid_pig_count pigs total) ∧
    min_pigs = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_pig_count_l1553_155303


namespace NUMINAMATH_CALUDE_complete_graph_10_coloring_l1553_155358

/-- A complete graph with 10 vertices -/
def CompleteGraph10 := Fin 10

/-- The type of edge colorings for CompleteGraph10 -/
def EdgeColoring (k : ℕ) := CompleteGraph10 → CompleteGraph10 → Fin k

/-- Predicate to check if k vertices form a k-colored subgraph -/
def is_k_colored_subgraph (k : ℕ) (coloring : EdgeColoring k) (vertices : Finset CompleteGraph10) : Prop :=
  vertices.card = k ∧
  ∀ (v w : CompleteGraph10), v ∈ vertices → w ∈ vertices → v ≠ w →
    ∃ (c : Fin k), ∀ (x y : CompleteGraph10), x ∈ vertices → y ∈ vertices → x ≠ y →
      coloring x y = c → x = v ∧ y = w

/-- Main theorem: k-coloring of CompleteGraph10 is possible iff k ≥ 5 -/
theorem complete_graph_10_coloring (k : ℕ) :
  (∃ (coloring : EdgeColoring k),
    ∀ (vertices : Finset CompleteGraph10),
      vertices.card = k → is_k_colored_subgraph k coloring vertices) ↔
  k ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_complete_graph_10_coloring_l1553_155358


namespace NUMINAMATH_CALUDE_regular_decagon_interior_angle_measure_l1553_155365

/-- The measure of one interior angle of a regular decagon in degrees. -/
def regular_decagon_interior_angle : ℝ := 144

/-- Theorem: The measure of one interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle_measure :
  regular_decagon_interior_angle = 144 := by
  sorry

end NUMINAMATH_CALUDE_regular_decagon_interior_angle_measure_l1553_155365


namespace NUMINAMATH_CALUDE_complex_number_problem_l1553_155359

theorem complex_number_problem (a : ℝ) (z : ℂ) (i : ℂ) : 
  a < 0 → 
  i^2 = -1 → 
  z = a * i / (1 - 2 * i) → 
  Complex.abs z = Real.sqrt 5 → 
  a = -5 := by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1553_155359


namespace NUMINAMATH_CALUDE_problem_solution_l1553_155345

theorem problem_solution (m a b c d : ℚ) 
  (h1 : |m + 1| = 4)
  (h2 : a + b = 0)
  (h3 : c * d = 1) :
  a + b + 3 * c * d - m = 0 ∨ a + b + 3 * c * d - m = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1553_155345


namespace NUMINAMATH_CALUDE_best_estimate_and_error_prob_l1553_155395

/-- Represents a measurement with an error margin and probability --/
structure Measurement where
  value : ℝ
  error_margin : ℝ
  error_prob : ℝ

/-- The problem setup --/
def river_length_problem (gsa awra : Measurement) : Prop :=
  gsa.value = 402 ∧
  gsa.error_margin = 0.5 ∧
  gsa.error_prob = 0.04 ∧
  awra.value = 403 ∧
  awra.error_margin = 0.5 ∧
  awra.error_prob = 0.04

/-- The theorem to prove --/
theorem best_estimate_and_error_prob
  (gsa awra : Measurement)
  (h : river_length_problem gsa awra) :
  ∃ (estimate error_prob : ℝ),
    estimate = 402.5 ∧
    error_prob = 0.04 :=
  sorry

end NUMINAMATH_CALUDE_best_estimate_and_error_prob_l1553_155395


namespace NUMINAMATH_CALUDE_marble_ratio_l1553_155397

/-- The number of marbles Wolfgang bought -/
def wolfgang_marbles : ℕ := 16

/-- The number of marbles Ludo bought -/
def ludo_marbles : ℕ := 20

/-- The number of marbles Michael bought -/
def michael_marbles : ℕ := (2 * (wolfgang_marbles + ludo_marbles)) / 3

/-- The total number of marbles -/
def total_marbles : ℕ := wolfgang_marbles + ludo_marbles + michael_marbles

/-- Each person's share when divided equally -/
def equal_share : ℕ := 20

theorem marble_ratio :
  (ludo_marbles : ℚ) / wolfgang_marbles = 5 / 4 ∧
  total_marbles = 3 * equal_share :=
sorry

end NUMINAMATH_CALUDE_marble_ratio_l1553_155397


namespace NUMINAMATH_CALUDE_inequality_proof_l1553_155399

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  a * (b/a)^(1/3 : ℝ) + b * (c/b)^(1/3 : ℝ) + c * (a/c)^(1/3 : ℝ) ≤ a*b + b*c + c*a + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1553_155399


namespace NUMINAMATH_CALUDE_part_one_simplification_part_two_simplification_l1553_155356

-- Part 1
theorem part_one_simplification :
  (1 / 2)⁻¹ - (Real.sqrt 2019 - 1)^0 = 1 := by sorry

-- Part 2
theorem part_two_simplification (x y : ℝ) :
  (x - y)^2 - (x + 2*y) * (x - 2*y) = -2*x*y + 5*y^2 := by sorry

end NUMINAMATH_CALUDE_part_one_simplification_part_two_simplification_l1553_155356


namespace NUMINAMATH_CALUDE_arithmetic_mean_property_l1553_155373

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression (α : Type*) [Add α] [Mul α] where
  a₁ : α  -- First term
  d : α   -- Common difference

variable {α : Type*} [LinearOrderedField α]

/-- The nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression α) (n : ℕ) : α :=
  ap.a₁ + (n - 1 : α) * ap.d

/-- Theorem: In an arithmetic progression, any term (starting from the second)
    is the arithmetic mean of two terms equidistant from it. -/
theorem arithmetic_mean_property (ap : ArithmeticProgression α) (k p : ℕ) 
    (h1 : k ≥ 2) (h2 : p > 0) :
  ap.nthTerm k = (ap.nthTerm (k - p) + ap.nthTerm (k + p)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_property_l1553_155373


namespace NUMINAMATH_CALUDE_simplify_and_factorize_l1553_155343

theorem simplify_and_factorize (x : ℝ) : 
  3 * x^2 + 4 * x + 5 - (7 - 3 * x^2 - 5 * x) = (x + 2) * (6 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_factorize_l1553_155343


namespace NUMINAMATH_CALUDE_sqrt_square_abs_sqrt_neg_nine_squared_l1553_155322

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by sorry

theorem sqrt_neg_nine_squared : Real.sqrt ((-9) ^ 2) = 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_sqrt_neg_nine_squared_l1553_155322


namespace NUMINAMATH_CALUDE_wedding_chairs_l1553_155379

theorem wedding_chairs (rows : ℕ) (chairs_per_row : ℕ) (extra_chairs : ℕ) : 
  rows = 7 → chairs_per_row = 12 → extra_chairs = 11 → 
  rows * chairs_per_row + extra_chairs = 95 := by
sorry

end NUMINAMATH_CALUDE_wedding_chairs_l1553_155379


namespace NUMINAMATH_CALUDE_samir_stairs_count_l1553_155352

theorem samir_stairs_count (s : ℕ) : 
  (s + (s / 2 + 18) = 495) → s = 318 := by
  sorry

end NUMINAMATH_CALUDE_samir_stairs_count_l1553_155352


namespace NUMINAMATH_CALUDE_tangent_slope_angle_sin_plus_cos_l1553_155326

theorem tangent_slope_angle_sin_plus_cos (x : Real) : 
  let f : Real → Real := λ x => Real.sin x + Real.cos x
  let f' : Real → Real := λ x => -Real.sin x + Real.cos x
  let slope : Real := f' (π/4)
  let slope_angle : Real := Real.arctan slope
  x = π/4 → slope_angle = 0 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_sin_plus_cos_l1553_155326


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1553_155308

/-- Given an arithmetic sequence {aₙ}, prove that if a₅ + a₁₁ = 30 and a₄ = 7, then a₁₂ = 23 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h_sum : a 5 + a 11 = 30)
  (h_a4 : a 4 = 7) :
  a 12 = 23 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1553_155308


namespace NUMINAMATH_CALUDE_logarithm_equation_solution_l1553_155320

theorem logarithm_equation_solution (x : ℝ) (h1 : Real.log x + Real.log (x - 3) = 1) (h2 : x > 0) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equation_solution_l1553_155320


namespace NUMINAMATH_CALUDE_coupe_price_proof_l1553_155306

/-- The amount for which Melissa sold the coupe -/
def coupe_price : ℝ := 30000

/-- The amount for which Melissa sold the SUV -/
def suv_price : ℝ := 2 * coupe_price

/-- The commission rate -/
def commission_rate : ℝ := 0.02

/-- The total commission from both sales -/
def total_commission : ℝ := 1800

theorem coupe_price_proof :
  commission_rate * (coupe_price + suv_price) = total_commission :=
sorry

end NUMINAMATH_CALUDE_coupe_price_proof_l1553_155306


namespace NUMINAMATH_CALUDE_fraction_equality_l1553_155370

theorem fraction_equality : (20 * 21) / (2 + 0 + 2 + 1) = 84 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1553_155370


namespace NUMINAMATH_CALUDE_line_through_point_equation_line_with_slope_equation_l1553_155339

-- Define a line in 2D space
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to calculate the area of a triangle formed by a line and coordinate axes
def triangleArea (l : Line) : ℝ :=
  sorry

-- Function to check if a line passes through a point
def linePassesPoint (l : Line) (p : Point) : Prop :=
  sorry

-- Theorem for condition 1
theorem line_through_point_equation (l : Line) (A : Point) :
  triangleArea l = 3 ∧ linePassesPoint l A ∧ A.x = -3 ∧ A.y = 4 →
  (∃ a b c, a * l.slope + b = 0 ∧ a = 2 ∧ b = 3 ∧ c = -6) ∨
  (∃ a b c, a * l.slope + b = 0 ∧ a = 8 ∧ b = 3 ∧ c = 12) :=
sorry

-- Theorem for condition 2
theorem line_with_slope_equation (l : Line) :
  triangleArea l = 3 ∧ l.slope = 1/6 →
  (∃ b, l.intercept = b ∧ (b = 1 ∨ b = -1)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_equation_line_with_slope_equation_l1553_155339


namespace NUMINAMATH_CALUDE_kgood_existence_l1553_155354

def IsKGood (k : ℕ) (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

theorem kgood_existence (k : ℕ) :
  (k ≥ 2 → ∃ f : ℕ+ → ℕ+, IsKGood k f) ∧
  (k = 1 → ¬∃ f : ℕ+ → ℕ+, IsKGood k f) :=
sorry

end NUMINAMATH_CALUDE_kgood_existence_l1553_155354


namespace NUMINAMATH_CALUDE_race_heartbeats_l1553_155389

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  heart_rate * pace * race_distance

theorem race_heartbeats :
  total_heartbeats 140 6 30 = 25200 := by
  sorry

end NUMINAMATH_CALUDE_race_heartbeats_l1553_155389


namespace NUMINAMATH_CALUDE_projectile_distance_l1553_155329

theorem projectile_distance (v1 v2 t : ℝ) (h1 : v1 = 470) (h2 : v2 = 500) (h3 : t = 90 / 60) :
  v1 * t + v2 * t = 1455 :=
by sorry

end NUMINAMATH_CALUDE_projectile_distance_l1553_155329


namespace NUMINAMATH_CALUDE_min_colors_condition_1_min_colors_condition_2_l1553_155380

variable (n : ℕ)

/-- The set of all lattice points in n-dimensional space -/
def X : Set (Fin n → ℤ) := Set.univ

/-- Distance between two lattice points -/
def distance (A B : Fin n → ℤ) : ℕ :=
  (Finset.univ.sum fun i => (A i - B i).natAbs)

/-- A coloring of X is valid for Condition 1 if any two points of the same color have distance ≥ 2 -/
def valid_coloring_1 (c : (Fin n → ℤ) → Fin r) : Prop :=
  ∀ A B : Fin n → ℤ, c A = c B → distance n A B ≥ 2

/-- A coloring of X is valid for Condition 2 if any two points of the same color have distance ≥ 3 -/
def valid_coloring_2 (c : (Fin n → ℤ) → Fin r) : Prop :=
  ∀ A B : Fin n → ℤ, c A = c B → distance n A B ≥ 3

/-- The minimum number of colors needed to satisfy Condition 1 is 2 -/
theorem min_colors_condition_1 :
  (∃ c : (Fin n → ℤ) → Fin 2, valid_coloring_1 n c) ∧
  (∀ r < 2, ¬∃ c : (Fin n → ℤ) → Fin r, valid_coloring_1 n c) :=
sorry

/-- The minimum number of colors needed to satisfy Condition 2 is 2n + 1 -/
theorem min_colors_condition_2 :
  (∃ c : (Fin n → ℤ) → Fin (2 * n + 1), valid_coloring_2 n c) ∧
  (∀ r < 2 * n + 1, ¬∃ c : (Fin n → ℤ) → Fin r, valid_coloring_2 n c) :=
sorry

end NUMINAMATH_CALUDE_min_colors_condition_1_min_colors_condition_2_l1553_155380


namespace NUMINAMATH_CALUDE_ratio_problem_l1553_155310

theorem ratio_problem (p q r s : ℚ) 
  (h1 : p / q = 8)
  (h2 : r / q = 5)
  (h3 : r / s = 3 / 4) :
  s^2 / p^2 = 25 / 36 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1553_155310


namespace NUMINAMATH_CALUDE_thousand_to_100_equals_googol_cubed_l1553_155381

-- Define googol
def googol : ℕ := 10^100

-- Theorem statement
theorem thousand_to_100_equals_googol_cubed :
  1000^100 = googol^3 := by
  sorry

end NUMINAMATH_CALUDE_thousand_to_100_equals_googol_cubed_l1553_155381


namespace NUMINAMATH_CALUDE_max_value_sqrt_x_10_minus_x_l1553_155394

theorem max_value_sqrt_x_10_minus_x :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 10 → Real.sqrt (x * (10 - x)) ≤ 5) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 10 ∧ Real.sqrt (x * (10 - x)) = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_x_10_minus_x_l1553_155394


namespace NUMINAMATH_CALUDE_sum_of_odd_coefficients_l1553_155314

theorem sum_of_odd_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₀*x^6 + a₁*x^5 + a₂*x^4 + a₃*x^3 + a₄*x^2 + a₅*x + a₆) →
  a₁ + a₃ + a₅ = -364 := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_coefficients_l1553_155314


namespace NUMINAMATH_CALUDE_second_project_questions_l1553_155350

/-- Calculates the number of questions for the second project given the total questions per day,
    number of days, and questions for the first project. -/
def questions_for_second_project (questions_per_day : ℕ) (days : ℕ) (questions_first_project : ℕ) : ℕ :=
  questions_per_day * days - questions_first_project

/-- Proves that given the specified conditions, the number of questions for the second project is 476. -/
theorem second_project_questions :
  questions_for_second_project 142 7 518 = 476 := by
  sorry

end NUMINAMATH_CALUDE_second_project_questions_l1553_155350


namespace NUMINAMATH_CALUDE_circle_center_l1553_155318

/-- The center of a circle given by the equation 4x^2 + 8x + 4y^2 - 12y + 20 = 0 is (-1, 3/2) -/
theorem circle_center (x y : ℝ) : 
  (4 * x^2 + 8 * x + 4 * y^2 - 12 * y + 20 = 0) → 
  (∃ r : ℝ, (x + 1)^2 + (y - 3/2)^2 = r^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l1553_155318


namespace NUMINAMATH_CALUDE_min_value_of_F_l1553_155353

theorem min_value_of_F (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (min : ℝ), min = 6/11 ∧ ∀ (F : ℝ), F = 2*x^2 + 3*y^2 + z^2 → F ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_F_l1553_155353


namespace NUMINAMATH_CALUDE_number_problem_l1553_155374

theorem number_problem (x : ℝ) (n : ℝ) (h1 : x > 0) (h2 : x / 50 + x / n = 0.06 * x) : n = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1553_155374


namespace NUMINAMATH_CALUDE_third_batch_size_l1553_155330

/-- Proves that given the conditions of the problem, the number of students in the third batch is 60 -/
theorem third_batch_size :
  let batch1_size : ℕ := 40
  let batch2_size : ℕ := 50
  let batch1_avg : ℚ := 45
  let batch2_avg : ℚ := 55
  let batch3_avg : ℚ := 65
  let total_avg : ℚ := 56333333333333336 / 1000000000000000
  let batch3_size : ℕ := 60
  (batch1_size * batch1_avg + batch2_size * batch2_avg + batch3_size * batch3_avg) / 
    (batch1_size + batch2_size + batch3_size) = total_avg :=
by sorry


end NUMINAMATH_CALUDE_third_batch_size_l1553_155330


namespace NUMINAMATH_CALUDE_least_number_divisible_by_all_l1553_155363

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 6) % 24 = 0 ∧ (n + 6) % 32 = 0 ∧ (n + 6) % 36 = 0 ∧ (n + 6) % 54 = 0

theorem least_number_divisible_by_all : 
  is_divisible_by_all 858 ∧ ∀ m : ℕ, m < 858 → ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_all_l1553_155363


namespace NUMINAMATH_CALUDE_probability_of_specific_arrangement_l1553_155351

def total_tiles : ℕ := 7
def x_tiles : ℕ := 4
def o_tiles : ℕ := 3

def specific_arrangement : List Char := ['X', 'X', 'O', 'X', 'O', 'X', 'O']

theorem probability_of_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = 
  (1 : ℚ) / 35 :=
sorry

end NUMINAMATH_CALUDE_probability_of_specific_arrangement_l1553_155351


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1553_155396

/-- A quadratic function f(x) = x^2 - 2bx + b^2 + b - 5 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + b^2 + b - 5

/-- The derivative of f with respect to x -/
def f_derivative (b : ℝ) (x : ℝ) : ℝ := 2*x - 2*b

theorem quadratic_function_range (b : ℝ) :
  (∃ x, f b x = 0) ∧ (∀ x < (3.5 : ℝ), f_derivative b x < 0) →
  3.5 ≤ b ∧ b ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1553_155396


namespace NUMINAMATH_CALUDE_piggy_bank_ratio_l1553_155361

theorem piggy_bank_ratio (T A S X Y : ℝ) (hT : T = 450) (hA : A = 30) 
  (hS : S > A) (hX : X > S) (hY : Y > X) (hTotal : A + S + X + Y = T) :
  ∃ (r : ℝ), r = (T - A - X - Y) / A ∧ r = S / A :=
sorry

end NUMINAMATH_CALUDE_piggy_bank_ratio_l1553_155361


namespace NUMINAMATH_CALUDE_emily_beads_count_l1553_155309

/-- The number of necklaces Emily made -/
def necklaces : ℕ := 26

/-- The number of beads required for each necklace -/
def beads_per_necklace : ℕ := 2

/-- The total number of beads Emily had -/
def total_beads : ℕ := necklaces * beads_per_necklace

theorem emily_beads_count : total_beads = 52 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l1553_155309


namespace NUMINAMATH_CALUDE_weighted_average_theorem_l1553_155333

def group1_avg : ℝ := 30
def group1_weight : ℝ := 2
def group2_avg : ℝ := 40
def group2_weight : ℝ := 3
def group3_avg : ℝ := 20
def group3_weight : ℝ := 1

def total_weighted_sum : ℝ := group1_avg * group1_weight + group2_avg * group2_weight + group3_avg * group3_weight
def total_weight : ℝ := group1_weight + group2_weight + group3_weight

theorem weighted_average_theorem : total_weighted_sum / total_weight = 200 / 6 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_theorem_l1553_155333


namespace NUMINAMATH_CALUDE_decimal_51_to_binary_binary_to_decimal_51_l1553_155385

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- Converts a list of bits to a natural number -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_51_to_binary :
  to_binary 51 = [true, true, false, false, true, true] :=
by sorry

theorem binary_to_decimal_51 :
  from_binary [true, true, false, false, true, true] = 51 :=
by sorry

end NUMINAMATH_CALUDE_decimal_51_to_binary_binary_to_decimal_51_l1553_155385


namespace NUMINAMATH_CALUDE_unique_k_exists_l1553_155368

-- Define the sequence sum function
def S (n : ℕ) : ℤ := n^2 - 9*n

-- Define the k-th term of the sequence
def a (k : ℕ) : ℤ := S k - S (k-1)

-- State the theorem
theorem unique_k_exists (k : ℕ) :
  (∃ k, 5 < a k ∧ a k < 8) → k = 8 :=
sorry

end NUMINAMATH_CALUDE_unique_k_exists_l1553_155368


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1553_155317

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  a*x^2 + b*x + c = 0 ∧ a = 1 ∧ b = -5 ∧ c = 6 →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1553_155317


namespace NUMINAMATH_CALUDE_area_preserved_l1553_155331

-- Define the transformation
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - 1, p.2 + 2)

-- Define a quadrilateral as a set of four points in ℝ²
def Quadrilateral := Fin 4 → ℝ × ℝ

-- Define the area of a quadrilateral
noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Define F and F'
def F : Quadrilateral := sorry
def F' : Quadrilateral := fun i => f (F i)

-- Theorem statement
theorem area_preserved (h : area F = 6) : area F' = area F := by sorry

end NUMINAMATH_CALUDE_area_preserved_l1553_155331


namespace NUMINAMATH_CALUDE_initial_population_village1_is_correct_l1553_155392

/-- The initial population of the first village -/
def initial_population_village1 : ℕ := 78000

/-- The yearly decrease in population of the first village -/
def yearly_decrease_village1 : ℕ := 1200

/-- The initial population of the second village -/
def initial_population_village2 : ℕ := 42000

/-- The yearly increase in population of the second village -/
def yearly_increase_village2 : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years_until_equal : ℕ := 18

/-- Theorem stating that the initial population of the first village is correct -/
theorem initial_population_village1_is_correct :
  initial_population_village1 - years_until_equal * yearly_decrease_village1 =
  initial_population_village2 + years_until_equal * yearly_increase_village2 :=
by sorry

end NUMINAMATH_CALUDE_initial_population_village1_is_correct_l1553_155392


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1553_155384

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 5) / (x - 3) = 4 ∧ x = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1553_155384


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1553_155315

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 74 →
  E = 4 * F + 18 →
  D + E + F = 180 →
  F = 17.6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1553_155315


namespace NUMINAMATH_CALUDE_unique_solution_iff_m_not_neg_two_and_not_zero_l1553_155349

/-- Given an equation (m^2 + 2m + 3)x = 3(x + 2) + m - 4, it has a unique solution
    with respect to x if and only if m ≠ -2 and m ≠ 0 -/
theorem unique_solution_iff_m_not_neg_two_and_not_zero (m : ℝ) :
  (∃! x : ℝ, (m^2 + 2*m + 3)*x = 3*(x + 2) + m - 4) ↔ (m ≠ -2 ∧ m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_m_not_neg_two_and_not_zero_l1553_155349


namespace NUMINAMATH_CALUDE_complex_abs_3_minus_10i_l1553_155327

theorem complex_abs_3_minus_10i :
  let z : ℂ := 3 - 10*I
  Complex.abs z = Real.sqrt 109 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_3_minus_10i_l1553_155327
