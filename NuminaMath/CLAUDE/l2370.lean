import Mathlib

namespace balanced_equation_oxygen_coefficient_l2370_237019

/-- Represents a chemical element in a molecule --/
inductive Element
  | As
  | S
  | O

/-- Represents a molecule in a chemical equation --/
structure Molecule where
  elements : List (Element × Nat)

/-- Represents a side of a chemical equation --/
structure EquationSide where
  molecules : List (Molecule × Nat)

/-- Represents a chemical equation --/
structure ChemicalEquation where
  leftSide : EquationSide
  rightSide : EquationSide

/-- Checks if a chemical equation is balanced --/
def isBalanced (eq : ChemicalEquation) : Bool :=
  sorry

/-- Checks if coefficients are the smallest possible integers --/
def hasSmallestCoefficients (eq : ChemicalEquation) : Bool :=
  sorry

/-- The coefficient of O₂ in the balanced equation --/
def oxygenCoefficient (eq : ChemicalEquation) : Nat :=
  sorry

theorem balanced_equation_oxygen_coefficient :
  ∀ (eq : ChemicalEquation),
    eq.leftSide.molecules = [
      (Molecule.mk [(Element.As, 2), (Element.S, 3)], 2),
      (Molecule.mk [(Element.O, 2)], oxygenCoefficient eq)
    ] →
    eq.rightSide.molecules = [
      (Molecule.mk [(Element.As, 2), (Element.O, 3)], 4),
      (Molecule.mk [(Element.S, 1), (Element.O, 2)], 6)
    ] →
    isBalanced eq →
    hasSmallestCoefficients eq →
    oxygenCoefficient eq = 9 :=
  sorry

end balanced_equation_oxygen_coefficient_l2370_237019


namespace cuboid_area_example_l2370_237066

/-- The surface area of a cuboid -/
def cuboid_surface_area (length breadth height : ℝ) : ℝ :=
  2 * (length * breadth + breadth * height + length * height)

/-- Theorem: The surface area of a cuboid with length 12, breadth 6, and height 10 is 504 -/
theorem cuboid_area_example : cuboid_surface_area 12 6 10 = 504 := by
  sorry

end cuboid_area_example_l2370_237066


namespace rectangle_area_l2370_237089

/-- Rectangle ABCD with given properties -/
structure Rectangle where
  -- Length of the rectangle
  length : ℝ
  -- Width of the rectangle
  width : ℝ
  -- Point E on AB
  BE : ℝ
  -- Point F on CD
  CF : ℝ
  -- Length is thrice the width
  length_eq : length = 3 * width
  -- BE is twice CF
  BE_eq : BE = 2 * CF
  -- BE is less than AB (length)
  BE_lt_length : BE < length
  -- CF is less than CD (width)
  CF_lt_width : CF < width
  -- AB is 18 cm
  AB_eq : length = 18
  -- BE is 12 cm
  BE_eq_12 : BE = 12

/-- Theorem stating the area of the rectangle -/
theorem rectangle_area (rect : Rectangle) : rect.length * rect.width = 108 := by
  sorry

#check rectangle_area

end rectangle_area_l2370_237089


namespace smallest_four_digit_divisible_by_35_l2370_237000

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_35 :
  ∃ (n : ℕ), is_four_digit n ∧ n % 35 = 0 ∧ ∀ (m : ℕ), is_four_digit m ∧ m % 35 = 0 → n ≤ m :=
by
  use 1050
  sorry

end smallest_four_digit_divisible_by_35_l2370_237000


namespace no_real_solutions_l2370_237008

theorem no_real_solutions :
  ∀ x : ℝ, (5 * x^2 - 3 * x + 2) / (x + 2) ≠ 2 * x - 3 :=
by
  sorry

end no_real_solutions_l2370_237008


namespace division_remainder_zero_l2370_237002

theorem division_remainder_zero : 
  1234567 % 112 = 0 := by sorry

end division_remainder_zero_l2370_237002


namespace sqrt_difference_equality_l2370_237068

theorem sqrt_difference_equality : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - 3 * Real.sqrt 3 := by
  sorry

end sqrt_difference_equality_l2370_237068


namespace pool_volume_l2370_237090

/-- Proves that the pool holds 84 gallons of water given the specified conditions. -/
theorem pool_volume (bucket_fill_time : ℕ) (bucket_capacity : ℕ) (total_fill_time : ℕ) :
  bucket_fill_time = 20 →
  bucket_capacity = 2 →
  total_fill_time = 14 * 60 →
  (total_fill_time / bucket_fill_time) * bucket_capacity = 84 := by
  sorry

end pool_volume_l2370_237090


namespace area_under_curve_l2370_237009

/-- The area enclosed by the curve y = x^2 + 1, the coordinate axes, and the line x = 1 is 4/3 -/
theorem area_under_curve : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 1
  ∫ x in (0 : ℝ)..1, f x = 4/3 := by sorry

end area_under_curve_l2370_237009


namespace tourist_contact_probability_l2370_237097

/-- The probability that at least one tourist from the first group can contact at least one tourist from the second group -/
def contact_probability (p : ℝ) : ℝ := 1 - (1 - p)^42

/-- Theorem stating the probability of contact between two groups of tourists -/
theorem tourist_contact_probability (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  contact_probability p = 
  1 - (1 - p)^(6 * 7) :=
by sorry

end tourist_contact_probability_l2370_237097


namespace sum_of_odd_and_multiples_of_three_l2370_237040

/-- The number of six-digit odd numbers -/
def A : ℕ := 450000

/-- The number of six-digit multiples of 3 -/
def B : ℕ := 300000

/-- The sum of six-digit odd numbers and six-digit multiples of 3 is 750000 -/
theorem sum_of_odd_and_multiples_of_three : A + B = 750000 := by
  sorry

end sum_of_odd_and_multiples_of_three_l2370_237040


namespace hyperbola_perimeter_l2370_237033

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the properties of the hyperbola and points
def hyperbola_properties (F₁ F₂ P Q : ℝ × ℝ) : Prop :=
  ∃ (l : Set (ℝ × ℝ)),
    hyperbola P.1 P.2 ∧ 
    hyperbola Q.1 Q.2 ∧
    P ∈ l ∧ Q ∈ l ∧
    F₁.1 < P.1 ∧ F₁.1 < Q.1 ∧
    F₂.1 > F₁.1 ∧
    ‖P - Q‖ = 4

-- Theorem statement
theorem hyperbola_perimeter (F₁ F₂ P Q : ℝ × ℝ) 
  (h : hyperbola_properties F₁ F₂ P Q) :
  ‖P - F₂‖ + ‖Q - F₂‖ + ‖P - Q‖ = 12 :=
sorry

end hyperbola_perimeter_l2370_237033


namespace cookie_problem_solution_l2370_237015

/-- Represents the number of cookies decorated by each person in one cycle -/
structure DecoratingCycle where
  grandmother : ℕ
  mary : ℕ
  john : ℕ

/-- Represents the problem setup -/
structure CookieDecoratingProblem where
  cycle : DecoratingCycle
  trays : ℕ
  cookies_per_tray : ℕ
  grandmother_time_per_cookie : ℕ

def solve_cookie_problem (problem : CookieDecoratingProblem) :
  (ℕ × ℕ × ℕ) :=
sorry

theorem cookie_problem_solution
  (problem : CookieDecoratingProblem)
  (h_cycle : problem.cycle = ⟨5, 3, 2⟩)
  (h_trays : problem.trays = 5)
  (h_cookies_per_tray : problem.cookies_per_tray = 12)
  (h_grandmother_time : problem.grandmother_time_per_cookie = 4) :
  solve_cookie_problem problem = (4, 140, 40) :=
sorry

end cookie_problem_solution_l2370_237015


namespace product_mod_sixty_l2370_237018

theorem product_mod_sixty (m : ℕ) : 
  198 * 953 ≡ m [ZMOD 60] → 0 ≤ m → m < 60 → m = 54 := by
  sorry

end product_mod_sixty_l2370_237018


namespace triangle_problem_l2370_237006

theorem triangle_problem (a b c A B C : ℝ) 
  (h1 : 3 * a * Real.cos A = c * Real.cos B + b * Real.cos C)
  (h2 : a = 1)
  (h3 : Real.cos B + Real.cos C = 1) :
  Real.cos A = 1/3 ∧ c = Real.sqrt 3 := by
  sorry

end triangle_problem_l2370_237006


namespace corrected_mean_is_89_42857142857143_l2370_237024

def initial_scores : List ℝ := [85, 90, 87, 93, 89, 84, 88]

def corrected_scores : List ℝ := 
  [85, 90, 87 + 5, 93, 89, 84 + 5, 88]

theorem corrected_mean_is_89_42857142857143 : 
  (corrected_scores.sum / corrected_scores.length : ℝ) = 89.42857142857143 := by
  sorry

end corrected_mean_is_89_42857142857143_l2370_237024


namespace jonah_fish_exchange_l2370_237059

/-- The number of new fish Jonah received in exchange -/
def exchange_fish (initial : ℕ) (added : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial + added - eaten)

/-- Theorem stating the number of new fish Jonah received -/
theorem jonah_fish_exchange :
  exchange_fish 14 2 6 11 = 1 := by
  sorry

end jonah_fish_exchange_l2370_237059


namespace inequality_proof_l2370_237096

theorem inequality_proof (x y : ℝ) : x + y + Real.sqrt (x * y) ≤ 3 * (x + y - Real.sqrt (x * y)) := by
  sorry

end inequality_proof_l2370_237096


namespace prob_odd_divisor_18_factorial_l2370_237030

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of positive integer divisors of n -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- The number of odd positive integer divisors of n -/
def numOddDivisors (n : ℕ) : ℕ := sorry

/-- The probability of a randomly chosen positive integer divisor of n being odd -/
def probOddDivisor (n : ℕ) : ℚ := (numOddDivisors n : ℚ) / (numDivisors n : ℚ)

theorem prob_odd_divisor_18_factorial :
  probOddDivisor (factorial 18) = 1 / 17 := by sorry

end prob_odd_divisor_18_factorial_l2370_237030


namespace viewer_ratio_l2370_237051

def voltaire_daily_viewers : ℕ := 50
def earnings_per_view : ℚ := 1/2
def leila_weekly_earnings : ℕ := 350
def days_per_week : ℕ := 7

theorem viewer_ratio : 
  let voltaire_weekly_viewers := voltaire_daily_viewers * days_per_week
  let leila_weekly_viewers := (leila_weekly_earnings : ℚ) / earnings_per_view
  (leila_weekly_viewers : ℚ) / (voltaire_weekly_viewers : ℚ) = 2 := by
  sorry

end viewer_ratio_l2370_237051


namespace pond_length_l2370_237074

/-- Given a rectangular field with length 112 m and width half of its length,
    and a square-shaped pond inside the field with an area 1/98 of the field's area,
    prove that the length of the pond is 8 meters. -/
theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_area : ℝ) :
  field_length = 112 →
  field_width = field_length / 2 →
  pond_area = (field_length * field_width) / 98 →
  Real.sqrt pond_area = 8 := by
  sorry

end pond_length_l2370_237074


namespace books_after_donation_l2370_237005

theorem books_after_donation (boris_initial : Nat) (cameron_initial : Nat)
  (h1 : boris_initial = 24)
  (h2 : cameron_initial = 30) :
  boris_initial - boris_initial / 4 + cameron_initial - cameron_initial / 3 = 38 := by
  sorry

#check books_after_donation

end books_after_donation_l2370_237005


namespace probability_at_least_one_six_all_different_l2370_237045

-- Define the number of faces on a die
def num_faces : ℕ := 6

-- Define the total number of possible outcomes when rolling three dice
def total_outcomes : ℕ := num_faces ^ 3

-- Define the number of favorable outcomes (at least one 6 and all different)
def favorable_outcomes : ℕ := 60

-- Define the number of outcomes with at least one 6
def outcomes_with_six : ℕ := total_outcomes - (num_faces - 1) ^ 3

-- Theorem statement
theorem probability_at_least_one_six_all_different :
  (favorable_outcomes : ℚ) / outcomes_with_six = 60 / 91 := by
  sorry


end probability_at_least_one_six_all_different_l2370_237045


namespace largest_n_with_unique_k_l2370_237043

theorem largest_n_with_unique_k : ∀ n : ℕ,
  n > 112 →
  ¬(∃! k : ℤ, (8 : ℚ)/15 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 7/13) ∧
  (∃! k : ℤ, (8 : ℚ)/15 < (112 : ℚ)/(112 + k) ∧ (112 : ℚ)/(112 + k) < 7/13) :=
by sorry

end largest_n_with_unique_k_l2370_237043


namespace number_solution_l2370_237049

theorem number_solution : ∃ x : ℝ, (5020 - (502 / x) = 5015) ∧ x = 100.4 := by
  sorry

end number_solution_l2370_237049


namespace sqrt_equation_solution_l2370_237094

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 7 → x = 54 := by
  sorry

end sqrt_equation_solution_l2370_237094


namespace exists_fibonacci_divisible_by_10000_l2370_237025

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem exists_fibonacci_divisible_by_10000 :
  ∃ k, k ≤ 10^8 + 1 ∧ fibonacci k % 10000 = 0 := by
  sorry

end exists_fibonacci_divisible_by_10000_l2370_237025


namespace second_to_fourth_l2370_237070

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem: If A(a,b) is in the second quadrant, then B(b,a) is in the fourth quadrant -/
theorem second_to_fourth (a b : ℝ) :
  is_in_second_quadrant (Point.mk a b) →
  is_in_fourth_quadrant (Point.mk b a) := by
  sorry

end second_to_fourth_l2370_237070


namespace skew_lines_sufficient_not_necessary_l2370_237095

-- Define the concept of a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop := sorry

-- Define what it means for two lines to have no common point
def no_common_point (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem skew_lines_sufficient_not_necessary :
  ∀ (l1 l2 : Line3D),
    (are_skew l1 l2 → no_common_point l1 l2) ∧
    ¬(no_common_point l1 l2 → are_skew l1 l2) :=
by sorry

end skew_lines_sufficient_not_necessary_l2370_237095


namespace otimes_k_otimes_k_l2370_237022

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := x^3 + y - 2*x

-- Theorem statement
theorem otimes_k_otimes_k (k : ℝ) : otimes k (otimes k k) = 2*k^3 - 3*k := by
  sorry

end otimes_k_otimes_k_l2370_237022


namespace calculation_proofs_l2370_237020

theorem calculation_proofs :
  (∃ (x : ℝ), x = Real.sqrt 12 * Real.sqrt (1/3) - Real.sqrt 18 + |Real.sqrt 2 - 2| ∧ x = 4 - 4 * Real.sqrt 2) ∧
  (∃ (y : ℝ), y = (7 + 4 * Real.sqrt 3) * (7 - 4 * Real.sqrt 3) - (Real.sqrt 3 - 1)^2 ∧ y = 2 * Real.sqrt 3 - 3) := by
  sorry

end calculation_proofs_l2370_237020


namespace inequality_solution_l2370_237004

theorem inequality_solution (x : ℝ) : 
  (x + 2) / (x + 3) > (4 * x + 5) / (3 * x + 10) ↔ 
  (x > -10/3 ∧ x < -3) ∨ (x > (-1 - Real.sqrt 61) / 6 ∧ x < (-1 + Real.sqrt 61) / 6) :=
by sorry

end inequality_solution_l2370_237004


namespace min_third_side_of_triangle_l2370_237067

theorem min_third_side_of_triangle (a b c : ℕ) : 
  (a + b + c) % 2 = 1 → -- perimeter is odd
  (a = b + 5 ∨ b = a + 5 ∨ a = c + 5 ∨ c = a + 5 ∨ b = c + 5 ∨ c = b + 5) → -- difference between two sides is 5
  c ≥ 6 -- minimum length of the third side is 6
  :=
by sorry

end min_third_side_of_triangle_l2370_237067


namespace mango_apple_not_orange_count_l2370_237076

/-- Given information about fruit preferences --/
structure FruitPreferences where
  apple : Nat
  orange_mango_not_apple : Nat
  all_fruits : Nat
  total_apple : Nat

/-- Calculate the number of people who like mango and apple and dislike orange --/
def mango_apple_not_orange (prefs : FruitPreferences) : Nat :=
  prefs.total_apple - prefs.all_fruits - prefs.orange_mango_not_apple

/-- Theorem stating the result of the calculation --/
theorem mango_apple_not_orange_count 
  (prefs : FruitPreferences) 
  (h1 : prefs.apple = 40)
  (h2 : prefs.orange_mango_not_apple = 7)
  (h3 : prefs.all_fruits = 4)
  (h4 : prefs.total_apple = 47) :
  mango_apple_not_orange prefs = 36 := by
  sorry

#eval mango_apple_not_orange ⟨40, 7, 4, 47⟩

end mango_apple_not_orange_count_l2370_237076


namespace remaining_time_for_finger_exerciser_l2370_237091

theorem remaining_time_for_finger_exerciser (total_time piano_time writing_time history_time : ℕ) :
  total_time = 120 ∧ piano_time = 30 ∧ writing_time = 25 ∧ history_time = 38 →
  total_time - (piano_time + writing_time + history_time) = 27 := by
sorry

end remaining_time_for_finger_exerciser_l2370_237091


namespace officers_from_six_people_l2370_237017

/-- The number of ways to choose three distinct officers from a group of 6 people -/
def choose_officers (n : ℕ) : ℕ :=
  if n ≥ 3 then n * (n - 1) * (n - 2) else 0

/-- Theorem stating that choosing three distinct officers from 6 people results in 120 ways -/
theorem officers_from_six_people :
  choose_officers 6 = 120 := by
  sorry

end officers_from_six_people_l2370_237017


namespace initial_observations_l2370_237056

theorem initial_observations (initial_average : ℝ) (new_observation : ℝ) (average_decrease : ℝ) :
  initial_average = 12 →
  new_observation = 5 →
  average_decrease = 1 →
  ∃ n : ℕ, 
    (n : ℝ) * initial_average + new_observation = (n + 1) * (initial_average - average_decrease) ∧
    n = 6 :=
by sorry

end initial_observations_l2370_237056


namespace roster_adjustment_count_l2370_237082

/-- Represents the number of class officers -/
def num_officers : ℕ := 5

/-- Represents the number of days in the duty roster -/
def num_days : ℕ := 5

/-- The number of ways to arrange the original Monday and Friday officers -/
def arrange_mon_fri : ℕ := 6

/-- The number of ways to choose an officer for each of Tuesday, Wednesday, and Thursday -/
def arrange_tue_thu : ℕ := 2

/-- The number of ways to arrange the remaining two officers for each of Tuesday, Wednesday, and Thursday -/
def arrange_remaining : ℕ := 2

/-- Theorem stating the total number of ways to adjust the roster -/
theorem roster_adjustment_count :
  (arrange_mon_fri * arrange_tue_thu * arrange_remaining) = 24 :=
sorry

end roster_adjustment_count_l2370_237082


namespace problem_solution_l2370_237052

theorem problem_solution (a : ℚ) : a + a/3 - a/9 = 10/3 → a = 30/11 := by
  sorry

end problem_solution_l2370_237052


namespace system_solution_l2370_237010

theorem system_solution (a b c d x y z : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : x + y + z = 1)
  (h2 : a * x + b * y + c * z = d)
  (h3 : a^2 * x + b^2 * y + c^2 * z = d^2) :
  x = (d - c) * (b - d) / ((a - c) * (b - a)) ∧
  y = (a - d) * (c - d) / ((b - a) * (b - c)) ∧
  z = (a - d) * (d - b) / ((a - c) * (c - b)) := by
  sorry

end system_solution_l2370_237010


namespace church_trip_distance_l2370_237064

def trip_distance (speed1 speed2 speed3 : Real) (time : Real) : Real :=
  (speed1 * time + speed2 * time + speed3 * time)

theorem church_trip_distance :
  let speed1 : Real := 16
  let speed2 : Real := 12
  let speed3 : Real := 20
  let time : Real := 15 / 60
  trip_distance speed1 speed2 speed3 time = 12 := by
  sorry

end church_trip_distance_l2370_237064


namespace remainder_a_37_mod_45_l2370_237086

def sequence_number (n : ℕ) : ℕ :=
  -- Definition of a_n: integer obtained by writing all integers from 1 to n sequentially
  sorry

theorem remainder_a_37_mod_45 : sequence_number 37 % 45 = 37 := by
  sorry

end remainder_a_37_mod_45_l2370_237086


namespace power_sum_and_division_simplification_l2370_237021

theorem power_sum_and_division_simplification :
  3^123 + 9^5 / 9^3 = 3^123 + 81 :=
by sorry

end power_sum_and_division_simplification_l2370_237021


namespace correct_answer_l2370_237032

theorem correct_answer (x : ℝ) (h : 2 * x = 60) : x / 2 = 15 := by
  sorry

end correct_answer_l2370_237032


namespace stating_parallelogram_count_theorem_l2370_237047

/-- 
Given a triangle ABC where each side is divided into n equal parts and lines are drawn 
parallel to the sides through each division point, the function returns the total number 
of parallelograms formed in the resulting figure.
-/
def parallelogram_count (n : ℕ) : ℕ :=
  3 * Nat.choose (n + 2) 4

/-- 
Theorem stating that the number of parallelograms in a triangle with sides divided into 
n equal parts and lines drawn parallel to sides through division points is 
3 * (n+2 choose 4).
-/
theorem parallelogram_count_theorem (n : ℕ) : 
  parallelogram_count n = 3 * Nat.choose (n + 2) 4 := by
  sorry

#eval parallelogram_count 5  -- Example evaluation

end stating_parallelogram_count_theorem_l2370_237047


namespace hyperbola_asymptotes_l2370_237044

/-- A hyperbola with center at the origin, focus on the y-axis, and eccentricity √5 -/
structure Hyperbola where
  /-- The eccentricity of the hyperbola -/
  e : ℝ
  /-- The eccentricity is √5 -/
  h_e : e = Real.sqrt 5
  /-- The center is at the origin -/
  center : ℝ × ℝ
  h_center : center = (0, 0)
  /-- The focus is on the y-axis -/
  focus : ℝ × ℝ
  h_focus : focus.1 = 0

/-- The equations of the asymptotes of the hyperbola -/
def asymptotes (h : Hyperbola) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (1/2) * p.1 ∨ p.2 = -(1/2) * p.1}

/-- Theorem: The asymptotes of the given hyperbola are y = ± (1/2)x -/
theorem hyperbola_asymptotes (h : Hyperbola) : 
  asymptotes h = {p : ℝ × ℝ | p.2 = (1/2) * p.1 ∨ p.2 = -(1/2) * p.1} := by
  sorry

end hyperbola_asymptotes_l2370_237044


namespace polynomial_d_abs_l2370_237071

/-- A polynomial with complex roots 3 + i and 3 - i -/
def polynomial (a b c d e : ℤ) : ℂ → ℂ := fun z ↦ 
  a * (z - (3 + Complex.I))^4 + b * (z - (3 + Complex.I))^3 + 
  c * (z - (3 + Complex.I))^2 + d * (z - (3 + Complex.I)) + e

/-- The coefficients have no common factors other than 1 -/
def coprime (a b c d e : ℤ) : Prop := 
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs) d.natAbs) e.natAbs = 1

theorem polynomial_d_abs (a b c d e : ℤ) 
  (h1 : polynomial a b c d e (3 + Complex.I) = 0)
  (h2 : coprime a b c d e) : 
  Int.natAbs d = 40 := by
  sorry

end polynomial_d_abs_l2370_237071


namespace perfect_square_and_cube_is_sixth_power_l2370_237038

theorem perfect_square_and_cube_is_sixth_power (n : ℕ) :
  (∃ a : ℕ, n = a^2) ∧ (∃ b : ℕ, n = b^3) → ∃ c : ℕ, n = c^6 := by
  sorry

end perfect_square_and_cube_is_sixth_power_l2370_237038


namespace min_value_problem_l2370_237072

theorem min_value_problem (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_6 : x + y + z = 6) : 
  (x^2 + 2*y^2)/(x + y) + (x^2 + 2*z^2)/(x + z) + (y^2 + 2*z^2)/(y + z) ≥ 6 := by
  sorry

end min_value_problem_l2370_237072


namespace scrap_metal_collection_l2370_237014

theorem scrap_metal_collection (a b : Nat) :
  a < 10 ∧ b < 10 ∧ 
  (900 + 10 * a + b) - (100 * a + 10 * b + 9) = 216 →
  900 + 10 * a + b = 975 ∧ 100 * a + 10 * b + 9 = 759 :=
by sorry

end scrap_metal_collection_l2370_237014


namespace cubic_equation_value_l2370_237031

theorem cubic_equation_value (x : ℝ) (h : 2 * x^2 - 3 * x - 2022 = 0) :
  2 * x^3 - x^2 - 2025 * x - 2020 = 2 := by
  sorry

end cubic_equation_value_l2370_237031


namespace unique_condition_implies_sum_l2370_237060

-- Define the set of possible values
def S : Set ℕ := {1, 2, 5}

-- Define the conditions
def condition1 (a b c : ℕ) : Prop := a ≠ 5
def condition2 (a b c : ℕ) : Prop := b = 5
def condition3 (a b c : ℕ) : Prop := c ≠ 2

-- Main theorem
theorem unique_condition_implies_sum (a b c : ℕ) :
  a ∈ S → b ∈ S → c ∈ S →
  a ≠ b → b ≠ c → a ≠ c →
  (condition1 a b c ∨ condition2 a b c ∨ condition3 a b c) →
  (¬condition1 a b c ∨ ¬condition2 a b c) →
  (¬condition1 a b c ∨ ¬condition3 a b c) →
  (¬condition2 a b c ∨ ¬condition3 a b c) →
  100 * a + 10 * b + c = 521 :=
by sorry

end unique_condition_implies_sum_l2370_237060


namespace relay_team_arrangements_l2370_237029

/-- The number of ways to arrange 4 people in a line with one fixed in the second position -/
def fixed_second_arrangements : ℕ := 6

/-- The total number of team members -/
def team_size : ℕ := 4

/-- The position where Jordan is fixed -/
def jordans_position : ℕ := 2

theorem relay_team_arrangements :
  (team_size = 4) →
  (jordans_position = 2) →
  (fixed_second_arrangements = 6) := by
sorry

end relay_team_arrangements_l2370_237029


namespace seventh_fibonacci_is_eight_l2370_237001

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem seventh_fibonacci_is_eight :
  fibonacci 6 = 8 := by
  sorry

end seventh_fibonacci_is_eight_l2370_237001


namespace intersection_M_N_l2370_237069

-- Define set M
def M : Set ℝ := {x | Real.sqrt (x + 1) ≥ 0}

-- Define set N
def N : Set ℝ := {x | x^2 + x - 2 < 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end intersection_M_N_l2370_237069


namespace refrigerator_price_l2370_237098

theorem refrigerator_price (refrigerator washing_machine : ℕ) 
  (h1 : washing_machine = refrigerator - 1490)
  (h2 : refrigerator + washing_machine = 7060) : 
  refrigerator = 4275 := by
  sorry

end refrigerator_price_l2370_237098


namespace tournament_games_32_teams_l2370_237007

/-- The number of games required in a single-elimination tournament --/
def games_required (n : ℕ) : ℕ := n - 1

/-- A theorem stating that a single-elimination tournament with 32 teams requires 31 games --/
theorem tournament_games_32_teams :
  games_required 32 = 31 :=
by sorry

end tournament_games_32_teams_l2370_237007


namespace curve_is_hyperbola_iff_l2370_237058

/-- A curve in the xy-plane parameterized by k -/
def curve (k : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 / (4 + k) + y^2 / (1 - k) = 1}

/-- The condition for the curve to be a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  (4 + k) * (1 - k) < 0

/-- The range of k for which the curve is a hyperbola -/
def hyperbola_range : Set ℝ :=
  {k | k < -4 ∨ k > 1}

/-- Theorem stating that the curve is a hyperbola if and only if k is in the hyperbola_range -/
theorem curve_is_hyperbola_iff (k : ℝ) :
  is_hyperbola k ↔ k ∈ hyperbola_range := by sorry

end curve_is_hyperbola_iff_l2370_237058


namespace remainder_of_7n_mod_4_l2370_237023

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 2) : (7 * n) % 4 = 2 := by
  sorry

end remainder_of_7n_mod_4_l2370_237023


namespace salty_cookies_left_l2370_237003

/-- Given the initial number of salty cookies and the number of salty cookies eaten,
    prove that the number of salty cookies left is equal to their difference. -/
theorem salty_cookies_left (initial : ℕ) (eaten : ℕ) (h : eaten ≤ initial) :
  initial - eaten = initial - eaten :=
by sorry

end salty_cookies_left_l2370_237003


namespace no_savings_on_group_purchase_l2370_237077

def window_price : ℕ := 120

def free_windows (n : ℕ) : ℕ := (n / 10) * 2

def cost (n : ℕ) : ℕ := (n - free_windows n) * window_price

def alice_windows : ℕ := 9
def bob_windows : ℕ := 11
def celina_windows : ℕ := 10

theorem no_savings_on_group_purchase :
  cost (alice_windows + bob_windows + celina_windows) =
  cost alice_windows + cost bob_windows + cost celina_windows :=
by sorry

end no_savings_on_group_purchase_l2370_237077


namespace container_volume_l2370_237012

/-- Given a cube with surface area 864 square units placed inside a cuboidal container
    with a 1 unit gap on all sides, the volume of the container is 2744 cubic units. -/
theorem container_volume (cube_surface_area : ℝ) (gap : ℝ) :
  cube_surface_area = 864 →
  gap = 1 →
  (cube_surface_area / 6).sqrt + 2 * gap ^ 3 = 2744 :=
by sorry

end container_volume_l2370_237012


namespace sum_of_absolute_coefficients_l2370_237062

theorem sum_of_absolute_coefficients (x a a₁ a₂ a₃ a₄ : ℝ) 
  (h : (1 - 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4)
  (ha : a > 0)
  (ha₂ : a₂ > 0)
  (ha₄ : a₄ > 0)
  (ha₁ : a₁ < 0)
  (ha₃ : a₃ < 0) :
  |a| + |a₁| + |a₂| + |a₃| + |a₄| = 81 := by
  sorry

end sum_of_absolute_coefficients_l2370_237062


namespace decrease_amount_l2370_237037

theorem decrease_amount (x y : ℝ) : x = 50 → (1/5) * x - y = 5 → y = 5 := by sorry

end decrease_amount_l2370_237037


namespace toothpicks_count_l2370_237028

/-- The number of small triangles in the base row of the large triangle -/
def base_triangles : ℕ := 1001

/-- The total number of small triangles in the large triangle -/
def total_triangles : ℕ := (base_triangles * (base_triangles + 1)) / 2

/-- The number of toothpicks required to construct the large triangle -/
def toothpicks_required : ℕ := (3 * total_triangles) / 2 + 3 * base_triangles

/-- Theorem stating that the number of toothpicks required is 755255 -/
theorem toothpicks_count : toothpicks_required = 755255 := by
  sorry

end toothpicks_count_l2370_237028


namespace decimal_to_binary_2008_l2370_237093

theorem decimal_to_binary_2008 :
  ∃ (binary : List Bool),
    binary.length = 11 ∧
    (binary.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0 = 2008) ∧
    binary = [true, true, true, true, true, false, true, true, false, false, false] := by
  sorry

end decimal_to_binary_2008_l2370_237093


namespace max_value_fraction_l2370_237011

theorem max_value_fraction (a b : ℝ) (h1 : a * b = 1) (h2 : a > b) (h3 : b ≥ 2/3) :
  (a - b) / (a^2 + b^2) ≤ 30/97 := by
  sorry

end max_value_fraction_l2370_237011


namespace johns_wrong_marks_l2370_237016

/-- Proves that John's wrongly entered marks are 102 given the conditions of the problem -/
theorem johns_wrong_marks (n : ℕ) (actual_marks wrong_marks : ℝ) 
  (h1 : n = 80)  -- Number of students in the class
  (h2 : actual_marks = 62)  -- John's actual marks
  (h3 : (wrong_marks - actual_marks) / n = 1/2)  -- Average increase due to wrong entry
  : wrong_marks = 102 := by
  sorry

end johns_wrong_marks_l2370_237016


namespace max_distribution_girls_l2370_237075

theorem max_distribution_girls (bags : Nat) (eyeliners : Nat) 
  (h1 : bags = 2923) (h2 : eyeliners = 3239) : 
  Nat.gcd bags eyeliners = 1 := by
  sorry

end max_distribution_girls_l2370_237075


namespace squared_sum_geq_one_l2370_237027

theorem squared_sum_geq_one (a b c : ℝ) (h : a * b + b * c + c * a = 1) :
  a^2 + b^2 + c^2 ≥ 1 := by
  sorry

end squared_sum_geq_one_l2370_237027


namespace root_difference_l2370_237088

/-- The polynomial coefficients -/
def a : ℚ := 8
def b : ℚ := -22
def c : ℚ := 15
def d : ℚ := -2

/-- The polynomial function -/
def f (x : ℚ) : ℚ := a * x^3 + b * x^2 + c * x + d

/-- The roots of the polynomial are in geometric progression -/
axiom roots_in_geometric_progression : ∃ (r₁ r₂ r₃ : ℚ), 
  (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) ∧ 
  (∃ (q : ℚ), r₂ = r₁ * q ∧ r₃ = r₂ * q)

/-- The theorem to be proved -/
theorem root_difference : 
  ∃ (r₁ r₃ : ℚ), (f r₁ = 0 ∧ f r₃ = 0) ∧ 
  (∀ r, f r = 0 → r₁ ≤ r ∧ r ≤ r₃) ∧
  (r₃ - r₁ = 33 / 14) := by
sorry

end root_difference_l2370_237088


namespace felix_brother_lift_multiple_l2370_237065

theorem felix_brother_lift_multiple :
  ∀ (felix_weight brother_weight : ℝ),
  felix_weight > 0 →
  brother_weight > 0 →
  1.5 * felix_weight = 150 →
  brother_weight = 2 * felix_weight →
  600 / brother_weight = 3 := by
  sorry

end felix_brother_lift_multiple_l2370_237065


namespace smallest_angle_theorem_l2370_237073

open Real

theorem smallest_angle_theorem : 
  let θ : ℝ := 90
  ∀ φ : ℝ, φ > 0 → φ < θ → 
    cos (φ * π / 180) ≠ sin (50 * π / 180) + cos (32 * π / 180) - sin (22 * π / 180) - cos (16 * π / 180) →
    cos (θ * π / 180) = sin (50 * π / 180) + cos (32 * π / 180) - sin (22 * π / 180) - cos (16 * π / 180) :=
by sorry

end smallest_angle_theorem_l2370_237073


namespace track_length_l2370_237057

/-- The length of a circular track given race conditions -/
theorem track_length (s t a : ℝ) (h₁ : s > 0) (h₂ : t > 0) (h₃ : a > 0) : 
  ∃ x : ℝ, x > 0 ∧ x = (s / (120 * t)) * (Real.sqrt (a^2 + 240 * a * t) - a) :=
by sorry

end track_length_l2370_237057


namespace mistaken_multiplication_l2370_237054

theorem mistaken_multiplication (x : ℝ) : 67 * x - 59 * x = 4828 → x = 603.5 := by
  sorry

end mistaken_multiplication_l2370_237054


namespace line_intercept_product_l2370_237013

/-- Given a line with equation y + 3 = -2(x + 5), 
    the product of its x-intercept and y-intercept is 84.5 -/
theorem line_intercept_product : 
  ∀ (x y : ℝ), y + 3 = -2 * (x + 5) → 
  ∃ (x_int y_int : ℝ), 
    (x_int + 5 = -13/2) ∧ 
    (y_int + 3 = -2 * 5) ∧ 
    (x_int * y_int = 84.5) := by
  sorry


end line_intercept_product_l2370_237013


namespace tan_alpha_value_l2370_237053

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = -5) : 
  Real.tan α = -23/16 := by
  sorry

end tan_alpha_value_l2370_237053


namespace train_length_calculation_l2370_237055

/-- Calculates the length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 60 →
  man_speed = 6 →
  passing_time = 23.998080153587715 →
  ∃ (train_length : ℝ), abs (train_length - 440) < 0.01 := by
  sorry


end train_length_calculation_l2370_237055


namespace parallel_vectors_x_value_l2370_237050

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (-1, 3)
  parallel a b → x = -1/3 := by
sorry

end parallel_vectors_x_value_l2370_237050


namespace speed_is_48_l2370_237042

-- Define the duration of the drive in hours
def drive_duration : ℚ := 7/4

-- Define the distance driven in km
def distance_driven : ℚ := 84

-- Theorem stating that the speed is 48 km/h
theorem speed_is_48 : distance_driven / drive_duration = 48 := by
  sorry

end speed_is_48_l2370_237042


namespace polynomial_division_remainder_l2370_237085

def dividend (x : ℝ) : ℝ := 3 * x^5 - 2 * x^3 + 5 * x^2 - 9
def divisor (x : ℝ) : ℝ := x^2 - 2 * x + 1
def remainder (x : ℝ) : ℝ := 19 * x - 22

theorem polynomial_division_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, dividend x = divisor x * q x + remainder x :=
by sorry

end polynomial_division_remainder_l2370_237085


namespace jacks_water_bottles_l2370_237061

/-- Represents the problem of determining how many bottles of water Jack initially bought. -/
theorem jacks_water_bottles :
  ∀ (initial_bottles : ℕ),
    (100 : ℚ) - (2 : ℚ) * (initial_bottles : ℚ) - (2 : ℚ) * (2 : ℚ) * (initial_bottles : ℚ) - (5 : ℚ) = (71 : ℚ) →
    initial_bottles = 4 := by
  sorry

end jacks_water_bottles_l2370_237061


namespace unique_c_value_l2370_237092

theorem unique_c_value : ∃! (c : ℝ), c ≠ 0 ∧
  (∃! (b : ℝ), b > 0 ∧
    (∃! (x : ℝ), x^2 + (b + 1/b + 1) * x + c = 0)) ∧
  c = 1/2 := by
  sorry

end unique_c_value_l2370_237092


namespace ratio_problem_l2370_237034

theorem ratio_problem (a b : ℚ) (h : (12 * a - 5 * b) / (14 * a - 3 * b) = 4 / 7) :
  a / b = 23 / 28 := by
sorry

end ratio_problem_l2370_237034


namespace bird_count_proof_l2370_237026

/-- The number of birds initially on a branch -/
def initial_birds (initial_parrots : ℕ) (initial_crows : ℕ) : ℕ :=
  initial_parrots + initial_crows

theorem bird_count_proof 
  (initial_parrots : ℕ) 
  (initial_crows : ℕ) 
  (remaining_parrots : ℕ) 
  (remaining_crow : ℕ) 
  (h1 : initial_parrots = 7)
  (h2 : remaining_parrots = 2)
  (h3 : remaining_crow = 1)
  (h4 : initial_parrots - remaining_parrots = initial_crows - remaining_crow) :
  initial_birds initial_parrots initial_crows = 13 := by
  sorry

end bird_count_proof_l2370_237026


namespace min_value_quadratic_l2370_237041

/-- The function f(x) = (3/2)x^2 - 9x + 7 attains its minimum value when x = 3 -/
theorem min_value_quadratic (x : ℝ) : 
  (∀ y : ℝ, (3/2 : ℝ) * x^2 - 9*x + 7 ≤ (3/2 : ℝ) * y^2 - 9*y + 7) ↔ x = 3 := by
  sorry

end min_value_quadratic_l2370_237041


namespace award_distribution_theorem_l2370_237048

-- Define the number of awards and students
def num_awards : ℕ := 7
def num_students : ℕ := 4

-- Function to calculate the number of ways to distribute awards
def distribute_awards (awards : ℕ) (students : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

-- Theorem stating that the number of ways to distribute awards is 3920
theorem award_distribution_theorem :
  distribute_awards num_awards num_students = 3920 :=
by sorry

end award_distribution_theorem_l2370_237048


namespace container_weight_sum_l2370_237035

theorem container_weight_sum (x y z : ℝ) 
  (h1 : x + y = 162) 
  (h2 : y + z = 168) 
  (h3 : z + x = 174) : 
  x + y + z = 252 := by
sorry

end container_weight_sum_l2370_237035


namespace no_partition_sum_product_l2370_237063

theorem no_partition_sum_product : ¬ ∃ (x y : ℕ), 
  1 ≤ x ∧ x ≤ 15 ∧ 1 ≤ y ∧ y ≤ 15 ∧ x ≠ y ∧
  x * y = (List.range 16).sum - x - y := by
  sorry

end no_partition_sum_product_l2370_237063


namespace intersection_of_A_and_B_l2370_237046

def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l2370_237046


namespace raisin_distribution_l2370_237079

/-- Given 5 boxes of raisins with a total of 437 raisins, where one box has 72 raisins,
    another has 74 raisins, and the remaining three boxes have an equal number of raisins,
    prove that each of the three equal boxes contains 97 raisins. -/
theorem raisin_distribution (total_raisins : ℕ) (total_boxes : ℕ) 
  (box1_raisins : ℕ) (box2_raisins : ℕ) (h1 : total_raisins = 437) 
  (h2 : total_boxes = 5) (h3 : box1_raisins = 72) (h4 : box2_raisins = 74) :
  ∃ (equal_box_raisins : ℕ), 
    equal_box_raisins * 3 + box1_raisins + box2_raisins = total_raisins ∧ 
    equal_box_raisins = 97 := by
  sorry

end raisin_distribution_l2370_237079


namespace arithmetic_progression_ratio_l2370_237087

/-- Given three numbers in arithmetic progression where the largest is 70
    and the difference between the smallest and largest is 40,
    prove that their ratio is 3:5:7 -/
theorem arithmetic_progression_ratio :
  ∀ (a b c : ℕ),
  c = 70 →
  c - a = 40 →
  b - a = c - b →
  ∃ (k : ℕ), k > 0 ∧ a = 3 * k ∧ b = 5 * k ∧ c = 7 * k :=
by sorry

end arithmetic_progression_ratio_l2370_237087


namespace major_selection_theorem_l2370_237084

-- Define the number of majors
def total_majors : ℕ := 7

-- Define the number of majors to be selected
def selected_majors : ℕ := 3

-- Define a function to calculate the number of ways to select and order majors
def ways_to_select_and_order (total : ℕ) (select : ℕ) (excluded : ℕ) : ℕ :=
  (Nat.choose total select - Nat.choose (total - excluded) (select - excluded)) * Nat.factorial select

-- Theorem statement
theorem major_selection_theorem : 
  ways_to_select_and_order total_majors selected_majors 2 = 180 := by
  sorry

end major_selection_theorem_l2370_237084


namespace f_greater_than_one_range_l2370_237036

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/2)^x else x^(1/2)

theorem f_greater_than_one_range :
  {a : ℝ | f a > 1} = Set.Ioi 1 ∪ Set.Iic 0 := by sorry

end f_greater_than_one_range_l2370_237036


namespace specific_figure_perimeter_l2370_237039

/-- A figure composed of a triangle and an adjacent quadrilateral -/
structure TriangleQuadrilateralFigure where
  /-- The three sides of the triangle -/
  triangle_side1 : ℝ
  triangle_side2 : ℝ
  triangle_side3 : ℝ
  /-- The side length of the quadrilateral (all sides equal) -/
  quad_side : ℝ

/-- The perimeter of the TriangleQuadrilateralFigure -/
def perimeter (figure : TriangleQuadrilateralFigure) : ℝ :=
  figure.triangle_side1 + figure.triangle_side2 + figure.triangle_side3 + 4 * figure.quad_side

/-- Theorem stating that the perimeter of the specific figure is 44 -/
theorem specific_figure_perimeter :
  ∃ (figure : TriangleQuadrilateralFigure),
    figure.triangle_side1 = 6 ∧
    figure.triangle_side2 = 8 ∧
    figure.triangle_side3 = 10 ∧
    figure.quad_side = 5 ∧
    perimeter figure = 44 :=
sorry

end specific_figure_perimeter_l2370_237039


namespace M_inter_N_eq_N_l2370_237080

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {x : ℝ | x^2 ≤ x}

-- Theorem statement
theorem M_inter_N_eq_N : M ∩ N = N := by
  sorry

end M_inter_N_eq_N_l2370_237080


namespace y1_less_than_y2_l2370_237081

/-- Linear function f(x) = -2x - 7 -/
def f (x : ℝ) : ℝ := -2 * x - 7

theorem y1_less_than_y2 (x₁ y₁ y₂ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f (x₁ - 1) = y₂) : 
  y₁ < y₂ := by
  sorry

end y1_less_than_y2_l2370_237081


namespace sum_of_two_numbers_l2370_237078

theorem sum_of_two_numbers (x y : ℝ) 
  (sum_eq : x + y = 10)
  (diff_eq : x - y = 8)
  (sq_diff_eq : x^2 - y^2 = 80) : 
  x + y = 10 := by
  sorry

end sum_of_two_numbers_l2370_237078


namespace perpendicular_vectors_l2370_237083

def a : Fin 2 → ℝ := ![4, 4]
def b (m : ℝ) : Fin 2 → ℝ := ![5, m]
def c : Fin 2 → ℝ := ![1, 3]

theorem perpendicular_vectors (m : ℝ) :
  (∀ i : Fin 2, (a i - 2 * c i) * b m i = 0) ↔ m = 5 := by
  sorry

end perpendicular_vectors_l2370_237083


namespace pill_supply_duration_l2370_237099

/-- Proves that a supply of pills lasts for a specific number of months -/
theorem pill_supply_duration (total_pills : ℕ) (days_per_pill : ℕ) (days_per_month : ℕ) 
  (h1 : total_pills = 120)
  (h2 : days_per_pill = 2)
  (h3 : days_per_month = 30) :
  (total_pills * days_per_pill) / days_per_month = 8 := by
  sorry

#check pill_supply_duration

end pill_supply_duration_l2370_237099
