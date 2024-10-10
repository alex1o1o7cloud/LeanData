import Mathlib

namespace line_parabola_intersection_l830_83095

/-- Given a line y = kx + m intersecting a parabola y^2 = 4x at two points,
    if the midpoint of these intersection points has y-coordinate 2,
    then k = 1. -/
theorem line_parabola_intersection (k m x₀ : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    -- Line equation
    y₁ = k * x₁ + m ∧
    y₂ = k * x₂ + m ∧
    -- Parabola equation
    y₁^2 = 4 * x₁ ∧
    y₂^2 = 4 * x₂ ∧
    -- Midpoint condition
    (x₁ + x₂) / 2 = x₀ ∧
    (y₁ + y₂) / 2 = 2) →
  k = 1 := by
sorry

end line_parabola_intersection_l830_83095


namespace max_diagonals_theorem_l830_83057

/-- The maximum number of non-intersecting or perpendicular diagonals in a regular n-gon -/
def max_diagonals (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 2 else n - 3

/-- Theorem stating the maximum number of non-intersecting or perpendicular diagonals in a regular n-gon -/
theorem max_diagonals_theorem (n : ℕ) (h : n ≥ 3) :
  max_diagonals n = if n % 2 = 0 then n - 2 else n - 3 :=
by sorry

end max_diagonals_theorem_l830_83057


namespace remainder_444_power_444_mod_13_l830_83099

theorem remainder_444_power_444_mod_13 : 444^444 ≡ 1 [ZMOD 13] := by
  sorry

end remainder_444_power_444_mod_13_l830_83099


namespace inequality_solution_set_l830_83089

def solution_set (x : ℝ) : Prop := -1 < x ∧ x < 2

theorem inequality_solution_set :
  ∀ x : ℝ, x * (x - 1) < 2 ↔ solution_set x :=
sorry

end inequality_solution_set_l830_83089


namespace marshmallow_roasting_l830_83085

/-- The number of marshmallows Joe's dad has -/
def dads_marshmallows : ℕ := 21

/-- The number of marshmallows Joe has -/
def joes_marshmallows : ℕ := 4 * dads_marshmallows

/-- The number of marshmallows Joe's dad roasts -/
def dads_roasted : ℕ := dads_marshmallows / 3

/-- The number of marshmallows Joe roasts -/
def joes_roasted : ℕ := joes_marshmallows / 2

/-- The total number of marshmallows roasted -/
def total_roasted : ℕ := dads_roasted + joes_roasted

theorem marshmallow_roasting :
  total_roasted = 49 := by sorry

end marshmallow_roasting_l830_83085


namespace congruence_solution_l830_83045

theorem congruence_solution (n : ℕ) : n < 47 → (13 * n) % 47 = 9 % 47 ↔ n = 4 := by
  sorry

end congruence_solution_l830_83045


namespace matrix_multiplication_result_l830_83053

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -1; 2, 4]
  A * B = !![17, 1; 16, -12] := by sorry

end matrix_multiplication_result_l830_83053


namespace total_score_is_26_l830_83064

-- Define the scores for Keith, Larry, and Danny
def keith_score : ℕ := 3
def larry_score : ℕ := 3 * keith_score
def danny_score : ℕ := larry_score + 5

-- Define the total score
def total_score : ℕ := keith_score + larry_score + danny_score

-- Theorem to prove
theorem total_score_is_26 : total_score = 26 := by
  sorry

end total_score_is_26_l830_83064


namespace jerry_weighted_mean_l830_83020

/-- Represents different currencies --/
inductive Currency
| USD
| EUR
| GBP
| CAD

/-- Represents a monetary amount with its currency --/
structure Money where
  amount : Float
  currency : Currency

/-- Represents a gift with its source --/
structure Gift where
  amount : Money
  isFamily : Bool

/-- Exchange rates to USD --/
def exchangeRate (c : Currency) : Float :=
  match c with
  | Currency.USD => 1
  | Currency.EUR => 1.20
  | Currency.GBP => 1.38
  | Currency.CAD => 0.82

/-- Convert Money to USD --/
def toUSD (m : Money) : Float :=
  m.amount * exchangeRate m.currency

/-- List of all gifts Jerry received --/
def jerryGifts : List Gift := [
  { amount := { amount := 9.73, currency := Currency.USD }, isFamily := true },
  { amount := { amount := 9.43, currency := Currency.EUR }, isFamily := true },
  { amount := { amount := 22.16, currency := Currency.USD }, isFamily := false },
  { amount := { amount := 23.51, currency := Currency.USD }, isFamily := false },
  { amount := { amount := 18.72, currency := Currency.EUR }, isFamily := false },
  { amount := { amount := 15.53, currency := Currency.GBP }, isFamily := false },
  { amount := { amount := 22.84, currency := Currency.USD }, isFamily := false },
  { amount := { amount := 7.25, currency := Currency.USD }, isFamily := true },
  { amount := { amount := 20.37, currency := Currency.CAD }, isFamily := true }
]

/-- Calculate weighted mean of Jerry's gifts --/
def weightedMean (gifts : List Gift) : Float :=
  let familyGifts := gifts.filter (λ g => g.isFamily)
  let friendGifts := gifts.filter (λ g => ¬g.isFamily)
  let familySum := familyGifts.foldl (λ acc g => acc + toUSD g.amount) 0
  let friendSum := friendGifts.foldl (λ acc g => acc + toUSD g.amount) 0
  familySum * 0.4 + friendSum * 0.6

/-- Theorem: The weighted mean of Jerry's birthday money in USD is $85.4442 --/
theorem jerry_weighted_mean :
  weightedMean jerryGifts = 85.4442 := by
  sorry

end jerry_weighted_mean_l830_83020


namespace map_distance_to_real_distance_l830_83098

/-- Proves that for a map with scale 1:500,000, a 4 cm distance on the map represents 20 km in reality -/
theorem map_distance_to_real_distance 
  (scale : ℚ) 
  (map_distance : ℚ) 
  (h_scale : scale = 1 / 500000)
  (h_map_distance : map_distance = 4)
  : map_distance * scale * 100000 = 20 := by
  sorry

end map_distance_to_real_distance_l830_83098


namespace three_X_seven_l830_83037

/-- Operation X is defined as a X b = b + 10*a - a^2 + 2*a*b -/
def X (a b : ℤ) : ℤ := b + 10*a - a^2 + 2*a*b

/-- The value of 3X7 is 70 -/
theorem three_X_seven : X 3 7 = 70 := by
  sorry

end three_X_seven_l830_83037


namespace quadratic_inequality_solution_l830_83071

theorem quadratic_inequality_solution (a b k : ℝ) : 
  (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ (x < 1 ∨ x > b)) →
  b > 1 →
  (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ k^2 + k + 2) →
  a = 1 ∧ b = 2 ∧ -3 ≤ k ∧ k ≤ 2 :=
by sorry

end quadratic_inequality_solution_l830_83071


namespace complement_of_intersection_AB_l830_83075

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x ≥ 1}

-- Define set B
def B : Set ℝ := {x | -1 ≤ x ∧ x < 2}

-- State the theorem
theorem complement_of_intersection_AB : 
  (A ∩ B)ᶜ = {x : ℝ | x < 1 ∨ x ≥ 2} := by sorry

end complement_of_intersection_AB_l830_83075


namespace largest_constant_inequality_l830_83044

theorem largest_constant_inequality (D : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 4 ≥ D * (x + y)) ↔ D ≤ 2 * Real.sqrt 2 := by
  sorry

end largest_constant_inequality_l830_83044


namespace horses_meet_after_nine_days_l830_83005

/-- The distance from Chang'an to Qi in li -/
def total_distance : ℝ := 1125

/-- The distance traveled by the good horse on the first day in li -/
def good_horse_initial : ℝ := 103

/-- The daily increase in distance for the good horse in li -/
def good_horse_increase : ℝ := 13

/-- The distance traveled by the mediocre horse on the first day in li -/
def mediocre_horse_initial : ℝ := 97

/-- The daily decrease in distance for the mediocre horse in li -/
def mediocre_horse_decrease : ℝ := 0.5

/-- The number of days it takes for the horses to meet -/
def meeting_days : ℕ := 9

/-- Theorem stating that the horses meet after 9 days -/
theorem horses_meet_after_nine_days :
  (good_horse_initial * meeting_days + (meeting_days * (meeting_days - 1) / 2) * good_horse_increase +
   mediocre_horse_initial * meeting_days - (meeting_days * (meeting_days - 1) / 2) * mediocre_horse_decrease) =
  2 * total_distance := by
  sorry

#check horses_meet_after_nine_days

end horses_meet_after_nine_days_l830_83005


namespace reciprocal_of_negative_2023_l830_83031

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end reciprocal_of_negative_2023_l830_83031


namespace unique_k_term_l830_83096

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n^2 - 7*n

/-- The k-th term of the sequence -/
def a (k : ℕ) : ℤ := S k - S (k-1)

theorem unique_k_term (k : ℕ) (h : 9 < a k ∧ a k < 12) : k = 9 := by
  sorry

end unique_k_term_l830_83096


namespace danny_thrice_jane_age_l830_83006

theorem danny_thrice_jane_age (danny_age jane_age : ℕ) (h1 : danny_age = 40) (h2 : jane_age = 26) :
  ∃ x : ℕ, x ≤ jane_age ∧ danny_age - x = 3 * (jane_age - x) ∧ x = 19 :=
by sorry

end danny_thrice_jane_age_l830_83006


namespace product_equality_l830_83035

theorem product_equality (x y : ℤ) (h1 : 24 * x = 173 * y) (h2 : 173 * y = 1730) : y = 10 := by
  sorry

end product_equality_l830_83035


namespace circle_cutting_theorem_l830_83056

-- Define the circle C1 with center O and radius r
def C1 (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define a point A on the circumference of C1
def A_on_C1 (O : ℝ × ℝ) (r : ℝ) (A : ℝ × ℝ) : Prop :=
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2

-- Define the existence of a line that cuts C1 into two parts
def cutting_line_exists (O : ℝ × ℝ) (r : ℝ) (A : ℝ × ℝ) : Prop :=
  ∃ (B C : ℝ × ℝ), 
    B ∈ C1 O r ∧ C ∈ C1 O r ∧ 
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = r^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = r^2

-- Theorem statement
theorem circle_cutting_theorem (O : ℝ × ℝ) (r : ℝ) (A : ℝ × ℝ) :
  A_on_C1 O r A → cutting_line_exists O r A := by
  sorry

end circle_cutting_theorem_l830_83056


namespace students_who_left_l830_83094

/-- Proves the number of students who left given initial, new, and final student counts -/
theorem students_who_left (initial : ℕ) (new : ℕ) (final : ℕ) 
  (h_initial : initial = 10)
  (h_new : new = 42)
  (h_final : final = 48) :
  initial + new - final = 4 := by
  sorry

end students_who_left_l830_83094


namespace three_numbers_sum_l830_83087

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →
  (a + b + c) / 3 = a + 8 →
  (a + b + c) / 3 = c - 9 →
  c - a = 26 →
  a + b + c = 81 := by
sorry

end three_numbers_sum_l830_83087


namespace find_number_l830_83029

theorem find_number : ∃ N : ℕ,
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := 2 * diff
  N / sum = quotient ∧ N % sum = 20 ∧ N = 220020 := by
  sorry

end find_number_l830_83029


namespace square_gt_abs_square_l830_83002

theorem square_gt_abs_square (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end square_gt_abs_square_l830_83002


namespace volume_removed_is_two_l830_83088

/-- Represents a cube with corner cuts -/
structure CutCube where
  side : ℝ
  cut_depth : ℝ
  face_square_side : ℝ

/-- Calculates the volume of material removed from a cut cube -/
def volume_removed (c : CutCube) : ℝ :=
  8 * (c.side - c.face_square_side) * (c.side - c.face_square_side) * c.cut_depth

/-- Theorem stating the volume removed from a 2x2x2 cube with specific cuts is 2 cubic units -/
theorem volume_removed_is_two :
  let c : CutCube := ⟨2, 1, 1⟩
  volume_removed c = 2 := by
  sorry


end volume_removed_is_two_l830_83088


namespace sum_seven_more_likely_than_eight_l830_83032

def dice_sum_probability (sum : Nat) : Rat :=
  (Finset.filter (fun (x, y) => x + y = sum) (Finset.product (Finset.range 6) (Finset.range 6))).card / 36

theorem sum_seven_more_likely_than_eight :
  dice_sum_probability 7 > dice_sum_probability 8 := by
  sorry

end sum_seven_more_likely_than_eight_l830_83032


namespace x_less_than_y_l830_83084

theorem x_less_than_y :
  let x : ℝ := Real.sqrt 7 - Real.sqrt 3
  let y : ℝ := Real.sqrt 6 - Real.sqrt 2
  x < y := by sorry

end x_less_than_y_l830_83084


namespace distinct_collections_proof_l830_83038

/-- The number of letters in "COMPUTATION" -/
def word_length : ℕ := 11

/-- The number of vowels in "COMPUTATION" -/
def num_vowels : ℕ := 5

/-- The number of consonants in "COMPUTATION" -/
def num_consonants : ℕ := 6

/-- The number of indistinguishable T's in "COMPUTATION" -/
def num_ts : ℕ := 2

/-- The number of vowels removed -/
def vowels_removed : ℕ := 3

/-- The number of consonants removed -/
def consonants_removed : ℕ := 4

/-- The function to calculate the number of distinct possible collections -/
def distinct_collections : ℕ := 110

theorem distinct_collections_proof :
  distinct_collections = 110 :=
sorry

end distinct_collections_proof_l830_83038


namespace ten_digit_square_plus_one_has_identical_digits_l830_83072

-- Define a function to check if a number is ten digits
def isTenDigits (n : ℕ) : Prop := 1000000000 ≤ n ∧ n < 10000000000

-- Define a function to check if a number has at least two identical digits
def hasAtLeastTwoIdenticalDigits (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ (∃ i j : ℕ, i ≠ j ∧ (n / 10^i % 10 = d) ∧ (n / 10^j % 10 = d))

theorem ten_digit_square_plus_one_has_identical_digits (n : ℕ) :
  isTenDigits (n^2 + 1) → hasAtLeastTwoIdenticalDigits (n^2 + 1) := by
  sorry

end ten_digit_square_plus_one_has_identical_digits_l830_83072


namespace joes_investment_rate_l830_83058

/-- Represents a simple interest bond investment -/
structure SimpleInterestBond where
  initialValue : ℝ
  interestRate : ℝ

/-- Calculates the value of a simple interest bond after a given number of years -/
def bondValue (bond : SimpleInterestBond) (years : ℝ) : ℝ :=
  bond.initialValue * (1 + bond.interestRate * years)

/-- Theorem: Given the conditions of Joe's investment, the interest rate is 1/13 -/
theorem joes_investment_rate : ∃ (bond : SimpleInterestBond),
  bondValue bond 3 = 260 ∧
  bondValue bond 8 = 360 ∧
  bond.interestRate = 1 / 13 := by
  sorry

end joes_investment_rate_l830_83058


namespace expression_simplification_l830_83011

theorem expression_simplification : 
  ((0.3 * 0.8) / 0.2) + (0.1 * 0.5) ^ 2 - 1 / (0.5 * 0.8)^2 = -5.0475 := by
  sorry

end expression_simplification_l830_83011


namespace pie_order_cost_l830_83042

/-- Represents the cost of fruit for Michael's pie order -/
def total_cost (peach_pies apple_pies blueberry_pies : ℕ) 
  (fruit_per_pie : ℕ) 
  (peach_price apple_price blueberry_price : ℚ) : ℚ :=
  (peach_pies * fruit_per_pie : ℚ) * peach_price +
  (apple_pies * fruit_per_pie : ℚ) * apple_price +
  (blueberry_pies * fruit_per_pie : ℚ) * blueberry_price

/-- Theorem stating that the total cost of fruit for Michael's pie order is $51.00 -/
theorem pie_order_cost : 
  total_cost 5 4 3 3 2 1 1 = 51 := by
  sorry

end pie_order_cost_l830_83042


namespace inequality_and_equality_condition_l830_83041

theorem inequality_and_equality_condition (x y : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  8 * x * y ≤ 5 * x * (1 - x) + 5 * y * (1 - y) ∧
  (8 * x * y = 5 * x * (1 - x) + 5 * y * (1 - y) ↔ x = 1/2 ∧ y = 1/2) :=
by sorry

end inequality_and_equality_condition_l830_83041


namespace cricketer_average_increase_l830_83008

/-- Represents a cricketer's batting statistics -/
structure CricketerStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional inning -/
def newAverage (stats : CricketerStats) (newInningRuns : ℕ) : ℚ :=
  (stats.totalRuns + newInningRuns) / (stats.innings + 1)

/-- Theorem: If a cricketer's average increases by 8 after scoring 140 in the 15th inning, 
    the new average is 28 -/
theorem cricketer_average_increase 
  (stats : CricketerStats) 
  (h1 : stats.innings = 14)
  (h2 : newAverage stats 140 = stats.average + 8)
  : newAverage stats 140 = 28 := by
  sorry

#check cricketer_average_increase

end cricketer_average_increase_l830_83008


namespace divisors_of_720_l830_83026

theorem divisors_of_720 : ∃ (n : ℕ), n = 720 ∧ (Finset.card (Finset.filter (λ x => n % x = 0) (Finset.range (n + 1)))) = 30 := by
  sorry

end divisors_of_720_l830_83026


namespace lawrence_county_kids_count_l830_83007

/-- The number of kids from Lawrence county who go to camp -/
def kids_at_camp : ℕ := 610769

/-- The number of kids from Lawrence county who stay home -/
def kids_at_home : ℕ := 590796

/-- The number of kids from outside the county who attended the camp -/
def outside_kids_at_camp : ℕ := 22

/-- The total number of kids in Lawrence county -/
def total_kids : ℕ := kids_at_camp + kids_at_home

theorem lawrence_county_kids_count :
  total_kids = 1201565 :=
sorry

end lawrence_county_kids_count_l830_83007


namespace puppies_sold_calculation_l830_83061

-- Define the given conditions
def initial_puppies : ℕ := 18
def puppies_per_cage : ℕ := 5
def cages_used : ℕ := 3

-- Define the theorem
theorem puppies_sold_calculation :
  initial_puppies - (cages_used * puppies_per_cage) = 3 := by
  sorry

end puppies_sold_calculation_l830_83061


namespace scientific_notation_equivalence_l830_83003

theorem scientific_notation_equivalence : 22000000 = 2.2 * (10 ^ 7) := by
  sorry

end scientific_notation_equivalence_l830_83003


namespace center_is_B_l830_83066

-- Define the points
variable (A B C D P Q K L : ℝ × ℝ)

-- Define the conditions
axiom on_line : ∃ (t : ℝ), A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1 ∧
  B = (1 - t) • A + t • D ∧
  C = (1 - t) • B + t • D

axiom AB_eq_BC : dist A B = dist B C

axiom perp_B : (B.2 - A.2) * (P.1 - B.1) = (B.1 - A.1) * (P.2 - B.2) ∧
               (B.2 - A.2) * (Q.1 - B.1) = (B.1 - A.1) * (Q.2 - B.2)

axiom perp_C : (C.2 - B.2) * (K.1 - C.1) = (C.1 - B.1) * (K.2 - C.2) ∧
               (C.2 - B.2) * (L.1 - C.1) = (C.1 - B.1) * (L.2 - C.2)

axiom on_circle_AD : dist A P + dist P D = dist A D ∧
                     dist A Q + dist Q D = dist A D

axiom on_circle_BD : dist B K + dist K D = dist B D ∧
                     dist B L + dist L D = dist B D

-- State the theorem
theorem center_is_B : 
  dist B P = dist B K ∧ dist B K = dist B L ∧ dist B L = dist B Q :=
sorry

end center_is_B_l830_83066


namespace friends_game_sales_l830_83055

/-- The amount of money received by Zachary -/
def zachary_amount : ℚ := 40 * 5

/-- The amount of money received by Jason -/
def jason_amount : ℚ := zachary_amount * (1 + 30 / 100)

/-- The amount of money received by Ryan -/
def ryan_amount : ℚ := jason_amount + 50

/-- The amount of money received by Emily -/
def emily_amount : ℚ := ryan_amount * (1 - 20 / 100)

/-- The amount of money received by Lily -/
def lily_amount : ℚ := emily_amount + 70

/-- The total amount of money received by all five friends -/
def total_amount : ℚ := zachary_amount + jason_amount + ryan_amount + emily_amount + lily_amount

theorem friends_game_sales : total_amount = 1336 := by
  sorry

end friends_game_sales_l830_83055


namespace complex_absolute_value_sum_l830_83062

theorem complex_absolute_value_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) = 2 * Real.sqrt 34 := by
  sorry

end complex_absolute_value_sum_l830_83062


namespace factorial_simplification_l830_83097

theorem factorial_simplification : 
  Nat.factorial 15 / (Nat.factorial 11 + 3 * Nat.factorial 10) = 25740 := by
  sorry

end factorial_simplification_l830_83097


namespace rationalize_denominator_l830_83052

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := by
  sorry

end rationalize_denominator_l830_83052


namespace mans_speed_against_current_l830_83047

/-- Given a man's speed with the current and the speed of the current,
    calculates the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Proves that given the specified conditions, the man's speed against the current is 8.6 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 15 3.2 = 8.6 := by
  sorry

#eval speed_against_current 15 3.2

end mans_speed_against_current_l830_83047


namespace real_part_of_complex_product_l830_83091

theorem real_part_of_complex_product : 
  let z : ℂ := (1 + 2*Complex.I) * (3 - Complex.I)
  Complex.re z = 5 := by
  sorry

end real_part_of_complex_product_l830_83091


namespace morgan_pens_count_l830_83034

/-- The number of pens Morgan has -/
def total_pens (red blue black green purple : ℕ) : ℕ :=
  red + blue + black + green + purple

/-- Theorem: Morgan has 231 pens in total -/
theorem morgan_pens_count : total_pens 65 45 58 36 27 = 231 := by
  sorry

end morgan_pens_count_l830_83034


namespace emily_orange_count_l830_83018

/-- The number of oranges each person has -/
structure OrangeCount where
  betty : ℕ
  sandra : ℕ
  emily : ℕ

/-- The conditions of the orange distribution problem -/
def orange_distribution (oc : OrangeCount) : Prop :=
  oc.emily = 7 * oc.sandra ∧
  oc.sandra = 3 * oc.betty ∧
  oc.betty = 12

/-- Theorem stating that Emily has 252 oranges given the conditions -/
theorem emily_orange_count (oc : OrangeCount) 
  (h : orange_distribution oc) : oc.emily = 252 := by
  sorry


end emily_orange_count_l830_83018


namespace circle_intersection_angle_equality_l830_83092

-- Define the types for points and circles
variable (Point Circle : Type)
[MetricSpace Point]

-- Define the intersection function
variable (intersect : Circle → Circle → Set Point)

-- Define the center of a circle
variable (center : Circle → Point)

-- Define the angle between three points
variable (angle : Point → Point → Point → ℝ)

-- State the theorem
theorem circle_intersection_angle_equality
  (c₁ c₂ c₃ : Circle)
  (P Q A B C D : Point)
  (h₁ : P ∈ intersect c₁ c₂)
  (h₂ : Q ∈ intersect c₁ c₂)
  (h₃ : center c₃ = P)
  (h₄ : A ∈ intersect c₁ c₃)
  (h₅ : B ∈ intersect c₁ c₃)
  (h₆ : C ∈ intersect c₂ c₃)
  (h₇ : D ∈ intersect c₂ c₃) :
  angle A Q D = angle B Q C :=
sorry

end circle_intersection_angle_equality_l830_83092


namespace product_equals_specific_number_l830_83024

theorem product_equals_specific_number : 333333 * (333333 + 1) = 111111222222 := by
  sorry

end product_equals_specific_number_l830_83024


namespace rhombus_longer_diagonal_l830_83063

/-- A rhombus with side length 65 and shorter diagonal 72 has a longer diagonal of 108 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diag : ℝ) (longer_diag : ℝ) : 
  side = 65 → shorter_diag = 72 → longer_diag = 108 → 
  side^2 = (shorter_diag/2)^2 + (longer_diag/2)^2 := by
  sorry

#check rhombus_longer_diagonal

end rhombus_longer_diagonal_l830_83063


namespace negative_two_squared_times_negative_one_to_2015_l830_83001

theorem negative_two_squared_times_negative_one_to_2015 : -2^2 * (-1)^2015 = 4 := by
  sorry

end negative_two_squared_times_negative_one_to_2015_l830_83001


namespace steve_salary_calculation_l830_83012

def steve_take_home_pay (salary : ℝ) (tax_rate : ℝ) (healthcare_rate : ℝ) (union_dues : ℝ) : ℝ :=
  salary - (salary * tax_rate) - (salary * healthcare_rate) - union_dues

theorem steve_salary_calculation :
  steve_take_home_pay 40000 0.20 0.10 800 = 27200 := by
  sorry

end steve_salary_calculation_l830_83012


namespace pump_emptying_time_l830_83017

theorem pump_emptying_time (time_B time_together : ℝ) 
  (hB : time_B = 6)
  (hTogether : time_together = 2.4)
  (h_rate : 1 / time_A + 1 / time_B = 1 / time_together) :
  time_A = 4 := by
  sorry

end pump_emptying_time_l830_83017


namespace susan_ate_six_candies_l830_83021

/-- The number of candies Susan ate during the week -/
def candies_eaten (bought_tuesday bought_thursday bought_friday remaining : ℕ) : ℕ :=
  bought_tuesday + bought_thursday + bought_friday - remaining

/-- Theorem: Susan ate 6 candies during the week -/
theorem susan_ate_six_candies : candies_eaten 3 5 2 4 = 6 := by
  sorry

end susan_ate_six_candies_l830_83021


namespace least_prime_factor_of_9_4_minus_9_3_l830_83040

theorem least_prime_factor_of_9_4_minus_9_3 :
  Nat.minFac (9^4 - 9^3) = 2 := by
  sorry

end least_prime_factor_of_9_4_minus_9_3_l830_83040


namespace initial_gold_percentage_l830_83004

/-- Given an alloy weighing 48 ounces, adding 12 ounces of pure gold results in a new alloy that is 40% gold.
    This theorem proves that the initial percentage of gold in the alloy is 25%. -/
theorem initial_gold_percentage (initial_weight : ℝ) (added_gold : ℝ) (final_percentage : ℝ) :
  initial_weight = 48 →
  added_gold = 12 →
  final_percentage = 40 →
  (initial_weight * (25 / 100) + added_gold) / (initial_weight + added_gold) = final_percentage / 100 :=
by sorry

end initial_gold_percentage_l830_83004


namespace number_of_girls_in_college_l830_83048

theorem number_of_girls_in_college (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) : 
  total_students = 240 → ratio_boys = 5 → ratio_girls = 7 → 
  (ratio_girls * total_students) / (ratio_boys + ratio_girls) = 140 := by
  sorry

end number_of_girls_in_college_l830_83048


namespace gift_wrapping_l830_83023

theorem gift_wrapping (total_gifts : ℕ) (total_rolls : ℕ) (first_roll_gifts : ℕ) (third_roll_gifts : ℕ) :
  total_gifts = 12 →
  total_rolls = 3 →
  first_roll_gifts = 3 →
  third_roll_gifts = 4 →
  ∃ (second_roll_gifts : ℕ),
    first_roll_gifts + second_roll_gifts + third_roll_gifts = total_gifts ∧
    second_roll_gifts = 5 :=
by
  sorry

end gift_wrapping_l830_83023


namespace line_intersection_y_axis_l830_83093

/-- A line passing through two points intersects the y-axis at a specific point -/
theorem line_intersection_y_axis (x₁ y₁ x₂ y₂ : ℝ) (hx : x₁ ≠ x₂) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (x₁ = 2 ∧ y₁ = 9 ∧ x₂ = 4 ∧ y₂ = 17) →
  (0, m * 0 + b) = (0, 1) :=
sorry

end line_intersection_y_axis_l830_83093


namespace employee_salary_problem_l830_83039

theorem employee_salary_problem (num_employees : ℕ) (manager_salary : ℕ) (avg_increase : ℕ) :
  num_employees = 15 →
  manager_salary = 4200 →
  avg_increase = 150 →
  (∃ (avg_salary : ℕ),
    num_employees * avg_salary + manager_salary = (num_employees + 1) * (avg_salary + avg_increase) ∧
    avg_salary = 1800) :=
by sorry

end employee_salary_problem_l830_83039


namespace profit_calculation_min_model_A_bicycles_l830_83068

-- Define the profit functions for models A and B
def profit_A : ℝ := 150
def profit_B : ℝ := 100

-- Define the purchase prices
def price_A : ℝ := 500
def price_B : ℝ := 800

-- Define the total number of bicycles and budget
def total_bicycles : ℕ := 20
def max_budget : ℝ := 13000

-- Theorem for part 1
theorem profit_calculation :
  3 * profit_A + 2 * profit_B = 650 ∧
  profit_A + 2 * profit_B = 350 := by sorry

-- Theorem for part 2
theorem min_model_A_bicycles :
  ∀ m : ℕ,
  (m ≤ total_bicycles ∧ 
   price_A * m + price_B * (total_bicycles - m) ≤ max_budget) →
  m ≥ 10 := by sorry

end profit_calculation_min_model_A_bicycles_l830_83068


namespace sqrt_equation_solution_l830_83083

theorem sqrt_equation_solution :
  ∃! x : ℝ, (Real.sqrt x + 2 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 2*x) ∧ 
             (x = 729/144) := by
  sorry

end sqrt_equation_solution_l830_83083


namespace range_of_even_function_l830_83016

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem range_of_even_function (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →
  (∀ x, x ∈ Set.Icc (a - 3) (2 * a) ↔ f a b x ≠ 0) →
  Set.range (f a b) = Set.Icc 3 7 := by
  sorry

#check range_of_even_function

end range_of_even_function_l830_83016


namespace second_term_value_l830_83010

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem second_term_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 2 + a 3 = 9) →
  a 1 = 1 →
  a 2 = 3 := by
sorry

end second_term_value_l830_83010


namespace commission_percentage_proof_l830_83022

def commission_percentage_below_threshold (total_sales : ℚ) (threshold : ℚ) (remitted_amount : ℚ) (commission_percentage_above_threshold : ℚ) : ℚ :=
  let sales_above_threshold := total_sales - threshold
  let commission_above_threshold := sales_above_threshold * commission_percentage_above_threshold / 100
  let total_commission := total_sales - remitted_amount
  let commission_below_threshold := total_commission - commission_above_threshold
  commission_below_threshold / threshold * 100

theorem commission_percentage_proof :
  let total_sales : ℚ := 32500
  let threshold : ℚ := 10000
  let remitted_amount : ℚ := 31100
  let commission_percentage_above_threshold : ℚ := 4
  commission_percentage_below_threshold total_sales threshold remitted_amount commission_percentage_above_threshold = 5 := by
  sorry

#eval commission_percentage_below_threshold 32500 10000 31100 4

end commission_percentage_proof_l830_83022


namespace at_least_three_pass_six_students_l830_83078

def exam_pass_probability : ℚ := 1/3

def at_least_three_pass (n : ℕ) (p : ℚ) : ℚ :=
  1 - (Nat.choose n 0 * (1 - p)^n +
       Nat.choose n 1 * p * (1 - p)^(n-1) +
       Nat.choose n 2 * p^2 * (1 - p)^(n-2))

theorem at_least_three_pass_six_students :
  at_least_three_pass 6 exam_pass_probability = 353/729 := by
  sorry

end at_least_three_pass_six_students_l830_83078


namespace bus_driver_regular_rate_l830_83000

/-- Represents the bus driver's compensation structure and work hours -/
structure BusDriverCompensation where
  regularRate : ℝ
  overtimeRate : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Theorem stating the bus driver's regular rate given the compensation conditions -/
theorem bus_driver_regular_rate 
  (comp : BusDriverCompensation)
  (h1 : comp.regularHours = 40)
  (h2 : comp.overtimeHours = 17)
  (h3 : comp.overtimeRate = comp.regularRate * 1.75)
  (h4 : comp.totalCompensation = 1116)
  (h5 : comp.totalCompensation = comp.regularRate * comp.regularHours + 
        comp.overtimeRate * comp.overtimeHours) : 
  comp.regularRate = 16 := by
  sorry

#check bus_driver_regular_rate

end bus_driver_regular_rate_l830_83000


namespace cousins_distribution_eq_67_l830_83019

/-- The number of ways to distribute n indistinguishable objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 cousins into 4 rooms -/
def cousins_distribution : ℕ := distribute 5 4

theorem cousins_distribution_eq_67 : cousins_distribution = 67 := by sorry

end cousins_distribution_eq_67_l830_83019


namespace james_number_problem_l830_83069

theorem james_number_problem (x : ℝ) : 3 * ((3 * x + 15) - 5) = 141 → x = 37 / 3 := by
  sorry

end james_number_problem_l830_83069


namespace equal_cupcake_distribution_l830_83076

theorem equal_cupcake_distribution (total_cupcakes : ℕ) (num_children : ℕ) 
  (h1 : total_cupcakes = 96) (h2 : num_children = 8) :
  total_cupcakes / num_children = 12 := by
  sorry

end equal_cupcake_distribution_l830_83076


namespace flagpole_height_l830_83030

/-- The height of a flagpole given specific measurements of surrounding stakes -/
theorem flagpole_height (AB OC OD OH : ℝ) (hAB : AB = 120) 
  (hHC : OH^2 + OC^2 = 170^2) (hHD : OH^2 + OD^2 = 100^2) (hCD : OC^2 + OD^2 = AB^2) :
  OH = 50 * Real.sqrt 7 := by
  sorry

end flagpole_height_l830_83030


namespace line_equation_through_points_l830_83013

/-- The equation x - y + 1 = 0 represents the line passing through the points (-1, 0) and (0, 1) -/
theorem line_equation_through_points : 
  ∀ (x y : ℝ), (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) → x - y + 1 = 0 :=
by sorry

end line_equation_through_points_l830_83013


namespace unique_common_tangent_common_tangent_segments_bisect_l830_83051

-- Define the parabolas
def C₁ (x : ℝ) : ℝ := x^2 + 2*x
def C₂ (a x : ℝ) : ℝ := -x^2 + a

-- Define the common tangent line
def commonTangent (k b : ℝ) (x : ℝ) : ℝ := k*x + b

-- Define the tangency points
structure TangencyPoint where
  x : ℝ
  y : ℝ

-- Theorem for part 1
theorem unique_common_tangent (a : ℝ) :
  a = -1/2 →
  ∃! (k b : ℝ), ∀ (x : ℝ),
    (C₁ x = commonTangent k b x ∧ C₂ a x = commonTangent k b x) →
    k = 1 ∧ b = -1/4 :=
sorry

-- Theorem for part 2
theorem common_tangent_segments_bisect (a : ℝ) :
  a ≠ -1/2 →
  ∃ (A B C D : TangencyPoint),
    (C₁ A.x = commonTangent k₁ b₁ A.x ∧ C₂ a A.x = commonTangent k₁ b₁ A.x) ∧
    (C₁ B.x = commonTangent k₁ b₁ B.x ∧ C₂ a B.x = commonTangent k₁ b₁ B.x) ∧
    (C₁ C.x = commonTangent k₂ b₂ C.x ∧ C₂ a C.x = commonTangent k₂ b₂ C.x) ∧
    (C₁ D.x = commonTangent k₂ b₂ D.x ∧ C₂ a D.x = commonTangent k₂ b₂ D.x) →
    (A.x + C.x) / 2 = -1/2 ∧ (A.y + C.y) / 2 = (a - 1) / 2 ∧
    (B.x + D.x) / 2 = -1/2 ∧ (B.y + D.y) / 2 = (a - 1) / 2 :=
sorry

end unique_common_tangent_common_tangent_segments_bisect_l830_83051


namespace orthographic_projection_properties_l830_83082

-- Define the basic structure for a view in orthographic projection
structure View where
  length : ℝ
  width : ℝ
  height : ℝ

-- Define the orthographic projection
structure OrthographicProjection where
  main_view : View
  top_view : View
  left_view : View

-- Define the properties of orthographic projection
def is_valid_orthographic_projection (op : OrthographicProjection) : Prop :=
  -- Main view and top view have aligned lengths
  op.main_view.length = op.top_view.length ∧
  -- Main view and left view are height level
  op.main_view.height = op.left_view.height ∧
  -- Left view and top view have equal widths
  op.left_view.width = op.top_view.width

-- Theorem statement
theorem orthographic_projection_properties (op : OrthographicProjection) 
  (h : is_valid_orthographic_projection op) :
  op.main_view.length = op.top_view.length ∧
  op.main_view.height = op.left_view.height ∧
  op.left_view.width = op.top_view.width := by
  sorry

end orthographic_projection_properties_l830_83082


namespace implication_equiv_contrapositive_l830_83079

-- Define the propositions
variable (P Q : Prop)

-- Define the original implication
def original : Prop := P → Q

-- Define the contrapositive
def contrapositive : Prop := ¬Q → ¬P

-- Theorem stating the equivalence of the original implication and its contrapositive
theorem implication_equiv_contrapositive :
  original P Q ↔ contrapositive P Q :=
sorry

end implication_equiv_contrapositive_l830_83079


namespace division_chain_l830_83028

theorem division_chain : (180 / 6) / 3 / 2 = 5 := by sorry

end division_chain_l830_83028


namespace geometric_sequence_increasing_condition_l830_83009

/-- A sequence a_n is geometric if there exists a constant r such that a_{n+1} = r * a_n for all n. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence a_n is increasing if a_n < a_{n+1} for all n. -/
def IsIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The condition a₁ < a₂ < a₄ for a sequence a_n. -/
def Condition (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 4

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) 
  (h : IsGeometric a) :
  (IsIncreasing a → Condition a) ∧ 
  ¬(Condition a → IsIncreasing a) := by
  sorry

end geometric_sequence_increasing_condition_l830_83009


namespace number_ratio_l830_83015

theorem number_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 143) (h4 : y = 104) :
  y / x = 8 / 3 := by
sorry

end number_ratio_l830_83015


namespace largest_b_value_l830_83050

theorem largest_b_value : ∃ b_max : ℝ,
  (∀ b : ℝ, (3 * b + 7) * (b - 2) = 4 * b → b ≤ b_max) ∧
  ((3 * b_max + 7) * (b_max - 2) = 4 * b_max) ∧
  b_max = 81.5205 / 30 :=
by sorry

end largest_b_value_l830_83050


namespace time_to_fill_cistern_l830_83036

/-- Given a cistern that can be partially filled by a pipe, this theorem proves
    the time required to fill the entire cistern. -/
theorem time_to_fill_cistern (partial_fill_time : ℝ) (partial_fill_fraction : ℝ) 
    (h1 : partial_fill_time = 4)
    (h2 : partial_fill_fraction = 1 / 11) : 
  partial_fill_time / partial_fill_fraction = 44 := by
  sorry

#check time_to_fill_cistern

end time_to_fill_cistern_l830_83036


namespace election_vote_difference_l830_83059

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 10000 → 
  candidate_percentage = 30/100 → 
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 4000 := by
sorry

end election_vote_difference_l830_83059


namespace xu_jun_current_age_l830_83033

-- Define Xu Jun's current age
def xu_jun_age : ℕ := sorry

-- Define the teacher's current age
def teacher_age : ℕ := sorry

-- Condition 1: Two years ago, the teacher's age was 3 times Xu Jun's age
axiom condition1 : teacher_age - 2 = 3 * (xu_jun_age - 2)

-- Condition 2: In 8 years, the teacher's age will be twice Xu Jun's age
axiom condition2 : teacher_age + 8 = 2 * (xu_jun_age + 8)

-- Theorem to prove
theorem xu_jun_current_age : xu_jun_age = 12 := by sorry

end xu_jun_current_age_l830_83033


namespace initial_money_calculation_l830_83043

theorem initial_money_calculation (initial_money : ℚ) : 
  (2 / 5 : ℚ) * initial_money = 100 → initial_money = 250 := by
  sorry

#check initial_money_calculation

end initial_money_calculation_l830_83043


namespace sum_of_cubes_roots_l830_83077

/-- For a quadratic equation x^2 + ax + a + 1 = 0, the sum of cubes of its roots equals 1 iff a = -1 -/
theorem sum_of_cubes_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + a*x₁ + a + 1 = 0 ∧ x₂^2 + a*x₂ + a + 1 = 0 ∧ x₁^3 + x₂^3 = 1) ↔ a = -1 :=
by sorry

end sum_of_cubes_roots_l830_83077


namespace carpet_length_l830_83054

theorem carpet_length (floor_area : ℝ) (carpet_coverage : ℝ) (carpet_width : ℝ) :
  floor_area = 120 →
  carpet_coverage = 0.3 →
  carpet_width = 4 →
  (carpet_coverage * floor_area) / carpet_width = 9 := by
  sorry

end carpet_length_l830_83054


namespace air_quality_probability_l830_83086

theorem air_quality_probability (p_single : ℝ) (p_consecutive : ℝ) (p_next : ℝ) : 
  p_single = 0.75 → p_consecutive = 0.6 → p_next = p_consecutive / p_single → p_next = 0.8 := by
  sorry

end air_quality_probability_l830_83086


namespace inequality_range_l830_83046

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) → 
  (a ≤ -1 ∨ a ≥ 4) := by
sorry

end inequality_range_l830_83046


namespace same_solution_value_of_b_l830_83049

theorem same_solution_value_of_b : ∀ x b : ℝ, 
  (3 * x + 9 = 6) ∧ 
  (5 * b * x - 15 = 5) → 
  b = -4 := by sorry

end same_solution_value_of_b_l830_83049


namespace line_slope_intercept_product_l830_83080

theorem line_slope_intercept_product (m b : ℝ) : 
  m = 3/4 → b = -2 → m * b < -3/2 := by
  sorry

end line_slope_intercept_product_l830_83080


namespace scarf_parity_l830_83074

theorem scarf_parity (initial_count : ℕ) (actions : ℕ) (final_count : ℕ) : 
  initial_count % 2 = 0 → 
  actions % 2 = 1 → 
  (∃ (changes : List ℤ), 
    changes.length = actions ∧ 
    (∀ c ∈ changes, c = 1 ∨ c = -1) ∧
    final_count = initial_count + changes.sum) →
  final_count % 2 = 1 :=
by sorry

#check scarf_parity 20 17 10

end scarf_parity_l830_83074


namespace apple_calculation_l830_83090

/-- The number of apples Pinky, Danny, and Benny collectively have after accounting for Lucy's sales -/
def total_apples (pinky_apples danny_apples lucy_sales benny_apples : ℝ) : ℝ :=
  pinky_apples + danny_apples + benny_apples - lucy_sales

/-- Theorem stating the total number of apples after Lucy's sales -/
theorem apple_calculation :
  total_apples 36.5 73.2 15.7 48.8 = 142.8 := by
  sorry

end apple_calculation_l830_83090


namespace sum_of_digits_a_l830_83073

def a : ℕ := 10^101 - 100

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_a : sum_of_digits a = 891 := by
  sorry

end sum_of_digits_a_l830_83073


namespace complex_fraction_simplification_l830_83081

theorem complex_fraction_simplification (i : ℂ) :
  i^2 = -1 →
  (2 + i) * (3 - 4*i) / (2 - i) = 5 := by
sorry

end complex_fraction_simplification_l830_83081


namespace odd_function_condition_l830_83060

/-- The function f(x) defined as (3^(x+1) - 1) / (3^x - 1) + a * (sin x + cos x)^2 --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (3^(x+1) - 1) / (3^x - 1) + a * (Real.sin x + Real.cos x)^2

/-- Theorem stating that for f to be an odd function, a must equal -2 --/
theorem odd_function_condition (a : ℝ) : 
  (∀ x, f a x = -f a (-x)) ↔ a = -2 := by sorry

end odd_function_condition_l830_83060


namespace henry_total_games_l830_83014

def wins : ℕ := 2
def losses : ℕ := 2
def draws : ℕ := 10

theorem henry_total_games : wins + losses + draws = 14 := by
  sorry

end henry_total_games_l830_83014


namespace interval_property_l830_83070

def f (x : ℝ) : ℝ := |x - 1|

theorem interval_property (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc a b ∧ x₂ ∈ Set.Icc a b ∧ x₁ < x₂ ∧ f x₁ ≥ f x₂) →
  a < 1 := by
  sorry

end interval_property_l830_83070


namespace school_population_theorem_l830_83065

theorem school_population_theorem (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 150 →
  boys + girls = total →
  girls = (boys * total) / 100 →
  boys = 60 := by
  sorry

end school_population_theorem_l830_83065


namespace total_whales_count_l830_83027

/-- The total number of whales observed across three trips -/
def total_whales : ℕ := by sorry

/-- The number of male whales observed in the first trip -/
def first_trip_males : ℕ := 28

/-- The number of female whales observed in the first trip -/
def first_trip_females : ℕ := 2 * first_trip_males

/-- The number of baby whales observed in the second trip -/
def second_trip_babies : ℕ := 8

/-- The number of whales in each family group (baby + two parents) -/
def whales_per_family : ℕ := 3

/-- The number of male whales observed in the third trip -/
def third_trip_males : ℕ := first_trip_males / 2

/-- The number of female whales observed in the third trip -/
def third_trip_females : ℕ := first_trip_females

/-- Theorem stating that the total number of whales observed is 178 -/
theorem total_whales_count : total_whales = 178 := by sorry

end total_whales_count_l830_83027


namespace greatest_power_of_two_l830_83025

theorem greatest_power_of_two (n : ℕ) : ∃ k : ℕ, 
  (2^k : ℤ) ∣ (10^1002 - 4^501) ∧ 
  ∀ m : ℕ, (2^m : ℤ) ∣ (10^1002 - 4^501) → m ≤ k :=
by
  -- The proof goes here
  sorry

end greatest_power_of_two_l830_83025


namespace circle_equation_proof_l830_83067

-- Define the circle
def circle_center : ℝ × ℝ := (-2, 1)

-- Define the diameter endpoints
def diameter_endpoint_x : ℝ → ℝ × ℝ := λ a => (a, 0)
def diameter_endpoint_y : ℝ → ℝ × ℝ := λ b => (0, b)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y = 0

-- Theorem statement
theorem circle_equation_proof :
  ∃ (a b : ℝ),
    (diameter_endpoint_x a).1 + (diameter_endpoint_y b).1 = 2 * circle_center.1 ∧
    (diameter_endpoint_x a).2 + (diameter_endpoint_y b).2 = 2 * circle_center.2 →
    ∀ (x y : ℝ),
      (x - circle_center.1)^2 + (y - circle_center.2)^2 = 
        ((diameter_endpoint_x a).1 - (diameter_endpoint_y b).1)^2 / 4 +
        ((diameter_endpoint_x a).2 - (diameter_endpoint_y b).2)^2 / 4 →
      circle_equation x y :=
by
  sorry

end circle_equation_proof_l830_83067
