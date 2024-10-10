import Mathlib

namespace cube_root_of_negative_64_l985_98518

theorem cube_root_of_negative_64 : ∃ x : ℝ, x^3 = -64 ∧ x = -4 := by sorry

end cube_root_of_negative_64_l985_98518


namespace books_count_l985_98569

def total_books (beatrix alannah queen kingston : ℕ) : ℕ :=
  beatrix + alannah + queen + kingston

theorem books_count :
  ∀ (beatrix alannah queen kingston : ℕ),
    beatrix = 30 →
    alannah = beatrix + 20 →
    queen = alannah + alannah / 5 →
    kingston = 2 * (beatrix + queen) →
    total_books beatrix alannah queen kingston = 320 := by
  sorry

end books_count_l985_98569


namespace big_sale_commission_proof_l985_98562

/-- Calculates the commission amount for a big sale given the following conditions:
  * new_average: Matt's new average commission after the big sale
  * total_sales: Total number of sales including the big sale
  * average_increase: The amount by which the big sale raised the average commission
-/
def big_sale_commission (new_average : ℚ) (total_sales : ℕ) (average_increase : ℚ) : ℚ :=
  new_average * total_sales - (new_average - average_increase) * (total_sales - 1)

/-- Theorem stating that given Matt's new average commission is $250, he has made 6 sales,
    and the big sale commission raises his average by $150, the commission amount for the
    big sale is $1000. -/
theorem big_sale_commission_proof :
  big_sale_commission 250 6 150 = 1000 := by
  sorry

end big_sale_commission_proof_l985_98562


namespace simplify_expression_l985_98557

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 := by
  sorry

end simplify_expression_l985_98557


namespace coord_sum_of_point_B_l985_98556

/-- Given points A(0, 0) and B(x, 3) where the slope of AB is 4/5,
    prove that the sum of B's coordinates is 6.75 -/
theorem coord_sum_of_point_B (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  let slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  slope = 4/5 → x + 3 = 6.75 := by
sorry

end coord_sum_of_point_B_l985_98556


namespace train_crossing_time_l985_98502

theorem train_crossing_time (train_length : ℝ) (platform1_length : ℝ) (platform2_length : ℝ) (time2 : ℝ) :
  train_length = 230 →
  platform1_length = 130 →
  platform2_length = 250 →
  time2 = 20 →
  let speed := (train_length + platform2_length) / time2
  let time1 := (train_length + platform1_length) / speed
  time1 = 15 := by sorry

end train_crossing_time_l985_98502


namespace integer_count_equality_l985_98594

theorem integer_count_equality : 
  ∃! (count : ℕ), count = 39999 ∧ 
  (∀ n : ℤ, (2 + ⌊(200 * n : ℚ) / 201⌋ = ⌈(198 * n : ℚ) / 199⌉) ↔ 
    (∃ k : ℤ, 0 ≤ k ∧ k < count ∧ n ≡ k [ZMOD 39999])) :=
by sorry

end integer_count_equality_l985_98594


namespace function_inequality_l985_98540

theorem function_inequality (f : ℤ → ℤ) 
  (h1 : ∀ k : ℤ, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2) 
  (h2 : f 4 = 25) : 
  ∀ k : ℤ, k ≥ 4 → f k ≥ k^2 := by
sorry

end function_inequality_l985_98540


namespace stating_six_suitcases_attempts_stating_ten_suitcases_attempts_l985_98507

/-- 
Given n suitcases and n keys, where it is unknown which key opens which suitcase,
this function calculates the minimum number of attempts needed to ensure all suitcases are opened.
-/
def minAttempts (n : ℕ) : ℕ := (n - 1) * n / 2

/-- 
Theorem stating that for 6 suitcases and 6 keys, the minimum number of attempts is 15.
-/
theorem six_suitcases_attempts : minAttempts 6 = 15 := by sorry

/-- 
Theorem stating that for 10 suitcases and 10 keys, the minimum number of attempts is 45.
-/
theorem ten_suitcases_attempts : minAttempts 10 = 45 := by sorry

end stating_six_suitcases_attempts_stating_ten_suitcases_attempts_l985_98507


namespace kishore_savings_l985_98590

def total_expenses : ℕ := 16200
def savings_rate : ℚ := 1 / 10

theorem kishore_savings :
  ∀ (salary : ℕ),
  (salary : ℚ) = (total_expenses : ℚ) + savings_rate * (salary : ℚ) →
  savings_rate * (salary : ℚ) = 1800 := by
  sorry

end kishore_savings_l985_98590


namespace complex_expression_equals_81_l985_98501

theorem complex_expression_equals_81 :
  3 * ((-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4) = 81 := by
  sorry

end complex_expression_equals_81_l985_98501


namespace flour_needed_proof_l985_98513

/-- The amount of flour Katie needs in pounds -/
def katie_flour : ℝ := 3

/-- The additional amount of flour Sheila needs compared to Katie in pounds -/
def sheila_additional : ℝ := 2

/-- The multiplier for John's flour needs compared to Sheila's -/
def john_multiplier : ℝ := 1.5

/-- The amount of flour Sheila needs in pounds -/
def sheila_flour : ℝ := katie_flour + sheila_additional

/-- The amount of flour John needs in pounds -/
def john_flour : ℝ := john_multiplier * sheila_flour

/-- The total amount of flour needed by Katie, Sheila, and John -/
def total_flour : ℝ := katie_flour + sheila_flour + john_flour

theorem flour_needed_proof : total_flour = 15.5 := by
  sorry

end flour_needed_proof_l985_98513


namespace fresh_to_dried_grapes_l985_98530

/-- Given fresh grapes with 60% water content and dried grapes with 20% water content,
    prove that 15 kg of dried grapes comes from 30 kg of fresh grapes. -/
theorem fresh_to_dried_grapes (fresh_water_content : ℝ) (dried_water_content : ℝ) 
  (dried_weight : ℝ) (fresh_weight : ℝ) : 
  fresh_water_content = 0.6 →
  dried_water_content = 0.2 →
  dried_weight = 15 →
  (1 - fresh_water_content) * fresh_weight = (1 - dried_water_content) * dried_weight →
  fresh_weight = 30 := by
sorry

end fresh_to_dried_grapes_l985_98530


namespace arman_work_hours_l985_98545

/-- Proves that Arman worked 35 hours last week given the conditions of his work schedule and pay. -/
theorem arman_work_hours : ∀ (last_week_hours : ℝ),
  (last_week_hours * 10 + 40 * 10.5 = 770) →
  last_week_hours = 35 := by
  sorry

end arman_work_hours_l985_98545


namespace function_equation_solution_l985_98542

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x^2 * y^2) →
  (∀ x : ℝ, f x = x^2 ∨ f x = -x^2) :=
by sorry

end function_equation_solution_l985_98542


namespace number_ratio_l985_98552

theorem number_ratio (x : ℝ) (h : 3 * (2 * x + 9) = 69) : 2 * x / x = 2 := by
  sorry

end number_ratio_l985_98552


namespace arithmetic_progression_solution_l985_98547

theorem arithmetic_progression_solution (a : ℝ) : 
  (3 - 2*a = a - 6 - 3) → a = 4 := by
  sorry

end arithmetic_progression_solution_l985_98547


namespace f_is_decreasing_l985_98574

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 2

-- Define the property of being an even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of being a decreasing function on an interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- State the theorem
theorem f_is_decreasing (a b : ℝ) :
  is_even_function (f a b) ∧ (Set.Icc (1 + a) 2).Nonempty →
  is_decreasing_on (f a b) 1 2 := by
  sorry

end f_is_decreasing_l985_98574


namespace brianna_marbles_l985_98572

theorem brianna_marbles (x : ℕ) : 
  x - 4 - (2 * 4) - (4 / 2) = 10 → x = 24 := by
  sorry

end brianna_marbles_l985_98572


namespace pioneer_camp_group_l985_98514

theorem pioneer_camp_group (x y z w : ℕ) : 
  x + y + z + w = 23 →
  10 * x + 11 * y + 12 * z + 13 * w = 253 →
  z = (3 : ℕ) / 2 * w →
  z = 6 := by
  sorry

end pioneer_camp_group_l985_98514


namespace robins_hair_growth_l985_98520

/-- Calculates the hair growth given initial length, final length, and cut length -/
def hair_growth (initial_length final_length cut_length : ℕ) : ℕ :=
  cut_length + final_length - initial_length

/-- Theorem: Given Robin's hair scenario, the hair growth is 8 inches -/
theorem robins_hair_growth :
  hair_growth 14 2 20 = 8 := by
  sorry

end robins_hair_growth_l985_98520


namespace girls_fraction_is_37_75_l985_98533

/-- Represents a school with a given number of students and boy-to-girl ratio -/
structure School where
  total_students : ℕ
  boys_ratio : ℕ
  girls_ratio : ℕ

/-- Calculates the number of girls in a school -/
def girls_count (s : School) : ℚ :=
  (s.total_students : ℚ) * s.girls_ratio / (s.boys_ratio + s.girls_ratio)

/-- Calculates the fraction of girls in a gathering of two schools -/
def girls_fraction (s1 s2 : School) : ℚ :=
  (girls_count s1 + girls_count s2) / (s1.total_students + s2.total_students)

theorem girls_fraction_is_37_75 (school_a school_b : School)
  (ha : school_a.total_students = 240 ∧ school_a.boys_ratio = 3 ∧ school_a.girls_ratio = 2)
  (hb : school_b.total_students = 210 ∧ school_b.boys_ratio = 2 ∧ school_b.girls_ratio = 3) :
  girls_fraction school_a school_b = 37 / 75 := by
  sorry

end girls_fraction_is_37_75_l985_98533


namespace quadratic_polynomial_negative_root_l985_98566

-- Define a quadratic polynomial type
def QuadraticPolynomial (α : Type*) [Ring α] := α → α

-- Define the property of having two distinct real roots
def HasTwoDistinctRealRoots (P : QuadraticPolynomial ℝ) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ P r₁ = 0 ∧ P r₂ = 0

-- Define the inequality condition
def SatisfiesInequality (P : QuadraticPolynomial ℝ) : Prop :=
  ∀ (a b : ℝ), (abs a ≥ 2017 ∧ abs b ≥ 2017) → P (a^2 + b^2) ≥ P (2*a*b)

-- Define the property of having at least one negative root
def HasNegativeRoot (P : QuadraticPolynomial ℝ) : Prop :=
  ∃ (r : ℝ), r < 0 ∧ P r = 0

-- The main theorem
theorem quadratic_polynomial_negative_root 
  (P : QuadraticPolynomial ℝ) 
  (h1 : HasTwoDistinctRealRoots P) 
  (h2 : SatisfiesInequality P) : 
  HasNegativeRoot P :=
sorry

end quadratic_polynomial_negative_root_l985_98566


namespace derivative_cos_ln_l985_98505

open Real

theorem derivative_cos_ln (x : ℝ) (h : x > 0) :
  deriv (λ x => cos (log x)) x = -1/x * sin (log x) := by
  sorry

end derivative_cos_ln_l985_98505


namespace initial_amount_is_750_l985_98525

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

/-- Final amount calculation using simple interest -/
def final_amount (principal rate time : ℝ) : ℝ := principal + simple_interest principal rate time

/-- Theorem stating that given the conditions, the initial amount must be 750 -/
theorem initial_amount_is_750 :
  ∀ (P : ℝ),
  final_amount P 0.06 5 = 975 →
  P = 750 := by
  sorry

end initial_amount_is_750_l985_98525


namespace expression_evaluation_l985_98579

theorem expression_evaluation (a b c d : ℚ) : 
  a = 3 → 
  b = a + 3 → 
  c = b - 8 → 
  d = a + 5 → 
  a + 2 ≠ 0 → 
  b - 4 ≠ 0 → 
  c + 5 ≠ 0 → 
  d - 3 ≠ 0 → 
  (a + 3) / (a + 2) * (b - 2) / (b - 4) * (c + 9) / (c + 5) * (d + 1) / (d - 3) = 1512 / 75 := by
sorry

end expression_evaluation_l985_98579


namespace mikes_books_l985_98568

theorem mikes_books (tim_books : ℕ) (total_books : ℕ) (h1 : tim_books = 22) (h2 : total_books = 42) :
  total_books - tim_books = 20 := by
  sorry

end mikes_books_l985_98568


namespace sum_21_implies_n_6_l985_98523

/-- Represents a sequence where a₁ = 1 and aₙ₊₁ = aₙ + 1 -/
def ArithmeticSequence (n : ℕ) : ℕ :=
  n

/-- Sum of the first n terms of the arithmetic sequence -/
def Sn (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Theorem: If Sn = 21, then n = 6 -/
theorem sum_21_implies_n_6 : Sn 6 = 21 :=
  by sorry

end sum_21_implies_n_6_l985_98523


namespace club_membership_theorem_l985_98576

/-- Represents the number of students in various club combinations -/
structure ClubMembership where
  total : ℕ
  music : ℕ
  science : ℕ
  sports : ℕ
  none : ℕ
  onlyMusic : ℕ
  onlyScience : ℕ
  onlySports : ℕ
  musicScience : ℕ
  scienceSports : ℕ
  musicSports : ℕ
  allThree : ℕ

/-- Theorem stating that given the conditions, the number of students in all three clubs is 1 -/
theorem club_membership_theorem (c : ClubMembership) : 
  c.total = 40 ∧ 
  c.music = c.total / 4 ∧ 
  c.science = c.total / 5 ∧ 
  c.sports = 8 ∧ 
  c.none = 7 ∧ 
  c.onlyMusic = 6 ∧ 
  c.onlyScience = 5 ∧ 
  c.onlySports = 2 ∧ 
  c.music = c.onlyMusic + c.musicScience + c.musicSports + c.allThree ∧ 
  c.science = c.onlyScience + c.musicScience + c.scienceSports + c.allThree ∧ 
  c.sports = c.onlySports + c.scienceSports + c.musicSports + c.allThree ∧ 
  c.total = c.none + c.onlyMusic + c.onlyScience + c.onlySports + c.musicScience + c.scienceSports + c.musicSports + c.allThree →
  c.allThree = 1 := by
sorry


end club_membership_theorem_l985_98576


namespace exponent_properties_l985_98560

theorem exponent_properties (a b : ℝ) (n : ℕ) :
  (a * b) ^ n = a ^ n * b ^ n ∧
  2 ^ 5 * (-1/2) ^ 5 = -1 ∧
  (-0.125) ^ 2022 * 2 ^ 2021 * 4 ^ 2020 = 1/32 := by
  sorry

end exponent_properties_l985_98560


namespace class_size_proof_l985_98592

/-- Proves that the number of students in a class is 27 given specific score distributions and averages -/
theorem class_size_proof (n : ℕ) : 
  (5 : ℝ) * 95 + (3 : ℝ) * 0 + ((n : ℝ) - 8) * 45 = (n : ℝ) * 49.25925925925926 → 
  n = 27 := by
  sorry

end class_size_proof_l985_98592


namespace square_difference_identity_l985_98559

theorem square_difference_identity (a b : ℝ) : a^2 - b^2 = (a + b) * (a - b) := by
  sorry

end square_difference_identity_l985_98559


namespace system_solution_unique_l985_98537

theorem system_solution_unique : 
  ∃! (x y : ℝ), (x - 2*y = 0) ∧ (3*x + 2*y = 8) ∧ (x = 2) ∧ (y = 1) :=
by
  sorry

end system_solution_unique_l985_98537


namespace function_inequality_l985_98585

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x > 0, x * (f' x) + x^2 < f x) :
  2 * f 1 > f 2 + 2 ∧ 3 * f 1 > f 3 + 3 := by
  sorry

end function_inequality_l985_98585


namespace equation_solutions_l985_98526

theorem equation_solutions : ∃ (x₁ x₂ : ℚ), 
  (x₁ = -1/2 ∧ x₂ = 3/4) ∧ 
  (∀ x : ℚ, 4*x*(2*x+1) = 3*(2*x+1) ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end equation_solutions_l985_98526


namespace arithmetic_sequence_property_l985_98597

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 = 45) : 
  a 5 = 9 := by
  sorry

end arithmetic_sequence_property_l985_98597


namespace dan_helmet_craters_l985_98508

/-- The number of craters in helmets owned by Dan, Daniel, and Rin. -/
structure HelmetsWithCraters where
  dan : ℕ
  daniel : ℕ
  rin : ℕ

/-- The conditions of the helmet crater problem. -/
def helmet_crater_conditions (h : HelmetsWithCraters) : Prop :=
  h.dan = h.daniel + 10 ∧
  h.rin = h.dan + h.daniel + 15 ∧
  h.rin = 75

/-- The theorem stating that Dan's helmet has 35 craters given the conditions. -/
theorem dan_helmet_craters (h : HelmetsWithCraters) 
  (hc : helmet_crater_conditions h) : h.dan = 35 := by
  sorry

end dan_helmet_craters_l985_98508


namespace gcd_n_cube_plus_nine_and_n_plus_two_l985_98573

theorem gcd_n_cube_plus_nine_and_n_plus_two (n : ℕ) (h : n > 2^3) :
  Nat.gcd (n^3 + 3^2) (n + 2) = 1 := by
  sorry

end gcd_n_cube_plus_nine_and_n_plus_two_l985_98573


namespace cyclic_inequality_l985_98541

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (1 / (x^2 + y*z)) + (1 / (y^2 + z*x)) + (1 / (z^2 + x*y)) ≤ 
  (1 / 2) * ((1 / (x*y)) + (1 / (y*z)) + (1 / (z*x))) := by
  sorry

end cyclic_inequality_l985_98541


namespace midpoint_trajectory_l985_98522

/-- The trajectory of the midpoint between a moving point on the unit circle and the fixed point (3, 0) -/
theorem midpoint_trajectory :
  ∀ (a b x y : ℝ),
  a^2 + b^2 = 1 →  -- point (a, b) is on the unit circle
  x = (a + 3) / 2 →  -- x-coordinate of midpoint
  y = b / 2 →  -- y-coordinate of midpoint
  x^2 + y^2 - 3*x + 2 = 0 := by
sorry

end midpoint_trajectory_l985_98522


namespace pens_bought_l985_98581

/-- Represents the cost of a single notebook in dollars -/
def notebook_cost : ℝ := sorry

/-- Represents the number of pens Maria bought -/
def num_pens : ℝ := sorry

/-- Theorem stating the relationship between the number of pens, total cost, and notebook cost -/
theorem pens_bought (notebook_cost num_pens : ℝ) : 
  (10 * notebook_cost + 2 * num_pens = 30) → 
  (num_pens = (30 - 10 * notebook_cost) / 2) := by
  sorry

end pens_bought_l985_98581


namespace jesse_banana_sharing_l985_98599

theorem jesse_banana_sharing :
  ∀ (total_bananas : ℕ) (bananas_per_friend : ℕ) (num_friends : ℕ),
    total_bananas = 21 →
    bananas_per_friend = 7 →
    total_bananas = bananas_per_friend * num_friends →
    num_friends = 3 := by
  sorry

end jesse_banana_sharing_l985_98599


namespace sale_price_ratio_l985_98553

theorem sale_price_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end sale_price_ratio_l985_98553


namespace maria_coin_stacks_maria_coin_stacks_proof_l985_98538

/-- Given that Maria has a total of 15 coins and each stack contains 3 coins,
    prove that the number of stacks she has is 5. -/
theorem maria_coin_stacks : ℕ → ℕ → ℕ → Prop :=
  fun (total_coins : ℕ) (coins_per_stack : ℕ) (num_stacks : ℕ) =>
    total_coins = 15 ∧ coins_per_stack = 3 →
    num_stacks * coins_per_stack = total_coins →
    num_stacks = 5

#check maria_coin_stacks

/-- Proof of the theorem -/
theorem maria_coin_stacks_proof : maria_coin_stacks 15 3 5 := by
  sorry

end maria_coin_stacks_maria_coin_stacks_proof_l985_98538


namespace complex_fraction_simplification_l985_98580

theorem complex_fraction_simplification :
  let z₁ : ℂ := Complex.mk 3 5
  let z₂ : ℂ := Complex.mk (-2) 3
  z₁ / z₂ = Complex.mk (-21/13) (-19/13) := by
  sorry

end complex_fraction_simplification_l985_98580


namespace book_combinations_l985_98591

theorem book_combinations (n m : ℕ) (h1 : n = 15) (h2 : m = 3) : Nat.choose n m = 455 := by
  sorry

end book_combinations_l985_98591


namespace arithmetic_sequence_sum_l985_98570

/-- Given an arithmetic sequence {a_n} where a_3 + a_11 = 40, prove that a_6 + a_7 + a_8 = 60 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 3 + a 11 = 40 →                                     -- given condition
  a 6 + a 7 + a 8 = 60 :=                               -- conclusion to prove
by sorry

end arithmetic_sequence_sum_l985_98570


namespace june_election_win_l985_98546

theorem june_election_win (total_students : ℕ) (boy_percentage : ℚ) 
  (june_boy_vote_percentage : ℚ) (june_girl_vote_percentage : ℚ) :
  total_students = 200 →
  boy_percentage = 60 / 100 →
  june_boy_vote_percentage = 675 / 1000 →
  june_girl_vote_percentage = 1 / 4 →
  ∃ (june_total_vote_percentage : ℚ), 
    june_total_vote_percentage = 505 / 1000 ∧ 
    june_total_vote_percentage > 1 / 2 :=
by sorry

end june_election_win_l985_98546


namespace personal_preference_invalid_l985_98554

/-- Represents the principles of sample selection --/
structure SampleSelectionPrinciples where
  representativeness : Bool
  randomness : Bool
  adequateSize : Bool

/-- Represents a sample selection method --/
inductive SampleSelectionMethod
  | Random
  | Representative
  | LargeEnough
  | PersonalPreference

/-- Checks if a sample selection method adheres to the principles --/
def isValidMethod (principles : SampleSelectionPrinciples) (method : SampleSelectionMethod) : Prop :=
  match method with
  | .Random => principles.randomness
  | .Representative => principles.representativeness
  | .LargeEnough => principles.adequateSize
  | .PersonalPreference => False

/-- Theorem stating that personal preference is not a valid sample selection method --/
theorem personal_preference_invalid (principles : SampleSelectionPrinciples) :
  ¬(isValidMethod principles SampleSelectionMethod.PersonalPreference) := by
  sorry


end personal_preference_invalid_l985_98554


namespace charity_ticket_revenue_l985_98543

theorem charity_ticket_revenue :
  ∀ (full_price : ℕ) (full_count half_count : ℕ),
  full_count + half_count = 200 →
  full_count = 3 * half_count →
  full_count * full_price + half_count * (full_price / 2) = 3501 →
  full_count * full_price = 3000 :=
by
  sorry

end charity_ticket_revenue_l985_98543


namespace train_braking_problem_l985_98596

/-- The braking distance function for a train -/
def S (t : ℝ) : ℝ := 27 * t - 0.45 * t^2

/-- The derivative of the braking distance function -/
def S' (t : ℝ) : ℝ := 27 - 0.9 * t

theorem train_braking_problem :
  (∃ t : ℝ, S' t = 0 ∧ t = 30) ∧
  S 30 = 405 := by
  sorry

end train_braking_problem_l985_98596


namespace ghi_equilateral_same_circumcenter_l985_98550

-- Define the points
variable (A B C D E F G H I G' H' I' : ℝ × ℝ)

-- Define the triangles
def triangle (P Q R : ℝ × ℝ) := Set.insert P (Set.insert Q (Set.singleton R))

-- Define equilateral triangle
def is_equilateral (t : Set (ℝ × ℝ)) : Prop :=
  ∃ P Q R, t = triangle P Q R ∧ 
    dist P Q = dist Q R ∧ dist Q R = dist R P

-- Define reflection
def reflect (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define circumcenter
def circumcenter (t : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Assumptions
variable (h1 : is_equilateral (triangle A B D))
variable (h2 : is_equilateral (triangle A C E))
variable (h3 : is_equilateral (triangle B C F))
variable (h4 : G = circumcenter (triangle A B D))
variable (h5 : H = circumcenter (triangle A C E))
variable (h6 : I = circumcenter (triangle B C F))
variable (h7 : G' = reflect B A G)
variable (h8 : H' = reflect C B H)
variable (h9 : I' = reflect A C I)

-- Theorem statements
theorem ghi_equilateral :
  is_equilateral (triangle G H I) := sorry

theorem same_circumcenter :
  circumcenter (triangle G H I) = circumcenter (triangle G' H' I') := sorry

end ghi_equilateral_same_circumcenter_l985_98550


namespace choose_starters_with_twins_l985_98524

def total_players : ℕ := 12
def twin_players : ℕ := 2
def starters : ℕ := 5

theorem choose_starters_with_twins :
  (total_players.choose starters) = (total_players - twin_players).choose (starters - twin_players) :=
sorry

end choose_starters_with_twins_l985_98524


namespace exponential_inequality_l985_98582

theorem exponential_inequality (m n : ℝ) (h1 : m > n) (h2 : n > 0) : (0.3 : ℝ) ^ m < (0.3 : ℝ) ^ n := by
  sorry

end exponential_inequality_l985_98582


namespace units_digit_of_product_l985_98598

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_product :
  units_digit (27 * 46) = 2 := by sorry

end units_digit_of_product_l985_98598


namespace peppers_required_per_day_l985_98567

/-- Represents the number of jalapeno pepper strips per sandwich -/
def strips_per_sandwich : ℕ := 4

/-- Represents the number of slices one jalapeno pepper can make -/
def slices_per_pepper : ℕ := 8

/-- Represents the time in minutes between serving each sandwich -/
def minutes_per_sandwich : ℕ := 5

/-- Represents the number of hours in a workday -/
def hours_per_day : ℕ := 8

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating the number of jalapeno peppers required for an 8-hour day -/
theorem peppers_required_per_day : 
  (hours_per_day * minutes_per_hour / minutes_per_sandwich) * 
  (strips_per_sandwich : ℚ) / slices_per_pepper = 48 := by
  sorry


end peppers_required_per_day_l985_98567


namespace sqrt_product_simplification_l985_98527

theorem sqrt_product_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (30 * p) * Real.sqrt (5 * p) * Real.sqrt (6 * p) = 30 * p * Real.sqrt p :=
by sorry

end sqrt_product_simplification_l985_98527


namespace probability_white_and_red_l985_98555

/-- The probability of drawing one white ball and one red ball from a box 
    containing 7 white balls, 8 black balls, and 1 red ball, 
    when two balls are drawn at random. -/
theorem probability_white_and_red (white : ℕ) (black : ℕ) (red : ℕ) : 
  white = 7 → black = 8 → red = 1 → 
  (white * red : ℚ) / (Nat.choose (white + black + red) 2) = 7 / 120 := by
  sorry

end probability_white_and_red_l985_98555


namespace units_digit_of_3_pow_1789_units_digit_of_1777_pow_1777_pow_1777_l985_98521

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem for the first part
theorem units_digit_of_3_pow_1789 :
  unitsDigit (3^1789) = 3 := by sorry

-- Theorem for the second part
theorem units_digit_of_1777_pow_1777_pow_1777 :
  unitsDigit (1777^(1777^1777)) = 7 := by sorry

end units_digit_of_3_pow_1789_units_digit_of_1777_pow_1777_pow_1777_l985_98521


namespace product_of_integers_l985_98577

theorem product_of_integers (x y : ℤ) (h1 : x + y = 8) (h2 : x^2 + y^2 = 34) : x * y = 15 := by
  sorry

end product_of_integers_l985_98577


namespace sum_of_valid_a_l985_98549

theorem sum_of_valid_a : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, 
    (∃ x : ℝ, x > 0 ∧ x ≠ 3 ∧ (3*x - a)/(x - 3) + (x + 1)/(3 - x) = 1) ∧
    (∀ y : ℝ, (y + 9 ≤ 2*(y + 2) ∧ (2*y - a)/3 > 1) ↔ y ≥ 5)) ∧
  (∀ a : ℤ, 
    ((∃ x : ℝ, x > 0 ∧ x ≠ 3 ∧ (3*x - a)/(x - 3) + (x + 1)/(3 - x) = 1) ∧
    (∀ y : ℝ, (y + 9 ≤ 2*(y + 2) ∧ (2*y - a)/3 > 1) ↔ y ≥ 5)) → a ∈ S) ∧
  Finset.sum S id = 13 :=
by sorry

end sum_of_valid_a_l985_98549


namespace function_properties_l985_98584

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h₁ : ∃ x, f x ≠ 0)
variable (h₂ : ∀ x, f (x + 3) = -f (3 - x))
variable (h₃ : ∀ x, f (x + 4) = -f (4 - x))

-- Theorem statement
theorem function_properties :
  (∀ x, f (-x) = -f x) ∧ (∃ p > 0, ∀ x, f (x + p) = f x) :=
sorry

end function_properties_l985_98584


namespace least_bamboo_sticks_l985_98509

/-- Represents the number of bamboo sticks each panda takes initially -/
structure BambooDistribution where
  s1 : ℕ
  s2 : ℕ
  s3 : ℕ
  s4 : ℕ

/-- Represents the final number of bamboo sticks each panda has -/
structure FinalDistribution where
  p1 : ℕ
  p2 : ℕ
  p3 : ℕ
  p4 : ℕ

/-- Calculates the final distribution based on the initial distribution -/
def calculateFinalDistribution (initial : BambooDistribution) : FinalDistribution :=
  { p1 := (2 * initial.s1) / 3 + initial.s2 / 2 + initial.s3 / 6 + (8 * initial.s4) / 9
  , p2 := (2 * initial.s1) / 3 + initial.s2 / 2 + initial.s3 / 6 + initial.s4 / 9
  , p3 := (2 * initial.s1) / 3 + initial.s2 / 2 + initial.s3 / 6 + initial.s4 / 9
  , p4 := (2 * initial.s1) / 3 + initial.s2 / 2 + initial.s3 / 6 + initial.s4 / 9
  }

/-- Checks if the final distribution satisfies the 4:3:2:1 ratio -/
def isValidRatio (final : FinalDistribution) : Prop :=
  4 * final.p4 = final.p1 ∧
  3 * final.p4 = final.p2 ∧
  2 * final.p4 = final.p3

/-- The main theorem stating the least possible total number of bamboo sticks -/
theorem least_bamboo_sticks :
  ∃ (initial : BambooDistribution),
    let final := calculateFinalDistribution initial
    isValidRatio final ∧
    initial.s1 + initial.s2 + initial.s3 + initial.s4 = 93 ∧
    ∀ (other : BambooDistribution),
      let otherFinal := calculateFinalDistribution other
      isValidRatio otherFinal →
      other.s1 + other.s2 + other.s3 + other.s4 ≥ 93 :=
by sorry


end least_bamboo_sticks_l985_98509


namespace x_values_l985_98503

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x^2 + 1/y = 13) (h2 : y^2 + 1/x = 8) :
  x = Real.sqrt 13 ∨ x = -Real.sqrt 13 :=
by sorry

end x_values_l985_98503


namespace arithmetic_sequence_1005th_term_l985_98564

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  p : ℝ
  r : ℝ
  first_term : ℝ := p
  second_term : ℝ := 11
  third_term : ℝ := 4*p - r
  fourth_term : ℝ := 4*p + r

/-- The nth term of the arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * (seq.second_term - seq.first_term)

/-- Theorem stating that the 1005th term of the sequence is 6029 -/
theorem arithmetic_sequence_1005th_term (seq : ArithmeticSequence) :
  nth_term seq 1005 = 6029 := by
  sorry

end arithmetic_sequence_1005th_term_l985_98564


namespace abc_right_triangle_l985_98532

/-- Parabola defined by y^2 = 4x -/
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

/-- Point A -/
def A : ℝ × ℝ := (1, 2)

/-- Point P -/
def P : ℝ × ℝ := (5, -2)

/-- B and C are on the parabola -/
def on_parabola (B C : ℝ × ℝ) : Prop := parabola B ∧ parabola C

/-- Line BC passes through P -/
def line_through_P (B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, B.1 + t * (C.1 - B.1) = P.1 ∧ B.2 + t * (C.2 - B.2) = P.2

/-- Triangle ABC is right-angled -/
def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

theorem abc_right_triangle (B C : ℝ × ℝ) :
  on_parabola B C → line_through_P B C → is_right_triangle A B C := by sorry

end abc_right_triangle_l985_98532


namespace combination_problem_l985_98500

theorem combination_problem (n : ℕ) (h : Nat.choose n 13 = Nat.choose n 7) :
  Nat.choose n 2 = 190 := by
  sorry

end combination_problem_l985_98500


namespace min_root_product_sum_l985_98504

def f (x : ℝ) : ℝ := x^4 + 14*x^3 + 52*x^2 + 56*x + 16

theorem min_root_product_sum (z₁ z₂ z₃ z₄ : ℝ) 
  (hroots : (∀ x, f x = 0 ↔ x = z₁ ∨ x = z₂ ∨ x = z₃ ∨ x = z₄)) :
  (∀ (σ : Equiv.Perm (Fin 4)), 
    |z₁ * z₂ + z₃ * z₄| ≥ 8 ∧
    |z₁ * z₃ + z₂ * z₄| ≥ 8 ∧
    |z₁ * z₄ + z₂ * z₃| ≥ 8) ∧
  (∃ (σ : Equiv.Perm (Fin 4)), 
    |z₁ * z₂ + z₃ * z₄| = 8 ∨
    |z₁ * z₃ + z₂ * z₄| = 8 ∨
    |z₁ * z₄ + z₂ * z₃| = 8) :=
by sorry

end min_root_product_sum_l985_98504


namespace circle_properties_l985_98528

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 6 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 6 = 0

-- Define the common chord equation
def common_chord (x y : ℝ) : Prop := 3*x - 2*y = 0

-- Theorem stating the properties of the circles
theorem circle_properties :
  -- The circles intersect
  (∃ x y : ℝ, C₁ x y ∧ C₂ x y) ∧
  -- The common chord equation is correct
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → common_chord x y) ∧
  -- The length of the common chord is (2√1182) / 13
  (let chord_length := (2 * Real.sqrt 1182) / 13
   ∃ x₁ y₁ x₂ y₂ : ℝ,
     C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧
     common_chord x₁ y₁ ∧ common_chord x₂ y₂ ∧
     Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = chord_length) :=
by sorry


end circle_properties_l985_98528


namespace arithmetic_mean_difference_l985_98519

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 25) : 
  r - p = 30 := by
sorry

end arithmetic_mean_difference_l985_98519


namespace simplify_expression_l985_98595

theorem simplify_expression (a : ℝ) : 5*a + 2*a + 3*a - 2*a = 8*a := by
  sorry

end simplify_expression_l985_98595


namespace quadratic_equation_real_roots_quadratic_equation_real_roots_for_m_1_l985_98565

theorem quadratic_equation_real_roots (m : ℝ) : 
  (∃ x : ℝ, (x + 2)^2 = m + 2) ↔ m ≥ -2 :=
by sorry

-- Example for m = 1
theorem quadratic_equation_real_roots_for_m_1 : 
  ∃ x : ℝ, (x + 2)^2 = 1 + 2 :=
by sorry

end quadratic_equation_real_roots_quadratic_equation_real_roots_for_m_1_l985_98565


namespace probability_specific_marble_draw_l985_98589

/-- Represents the number of marbles of each color in the jar -/
structure MarbleCount where
  red : ℕ
  green : ℕ
  white : ℕ

/-- Calculates the probability of drawing two red marbles followed by one green marble -/
def probability_two_red_one_green (mc : MarbleCount) : ℚ :=
  let total := mc.red + mc.green + mc.white
  (mc.red : ℚ) / total *
  ((mc.red - 1) : ℚ) / (total - 1) *
  (mc.green : ℚ) / (total - 2)

/-- The main theorem stating the probability for the given marble counts -/
theorem probability_specific_marble_draw :
  probability_two_red_one_green ⟨3, 4, 12⟩ = 12 / 2907 := by
  sorry

#eval probability_two_red_one_green ⟨3, 4, 12⟩

end probability_specific_marble_draw_l985_98589


namespace equation_solutions_l985_98544

def solution_set : Set (ℤ × ℤ) :=
  {(2, 1), (1, 0), (2, 2), (0, 0), (1, 2), (0, 1)}

theorem equation_solutions :
  ∀ (x y : ℤ), (x + y = x^2 - x*y + y^2) ↔ (x, y) ∈ solution_set :=
sorry

end equation_solutions_l985_98544


namespace round_trip_speed_l985_98536

/-- Proves that given a round trip where the return journey takes twice as long as the outward journey,
    and the average speed of the entire trip is 32 miles per hour, the speed of the outward journey is 21⅓ miles per hour. -/
theorem round_trip_speed (d : ℝ) (v : ℝ) (h1 : v > 0) (h2 : d > 0) : 
  (2 * d) / (d / v + 2 * d / v) = 32 → v = 64 / 3 := by
  sorry

#eval (64 : ℚ) / 3  -- To show that 64/3 is indeed equal to 21⅓

end round_trip_speed_l985_98536


namespace sin_300_degrees_l985_98511

theorem sin_300_degrees : 
  Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by sorry

end sin_300_degrees_l985_98511


namespace simple_interest_calculation_l985_98539

/-- Given a principal amount where the compound interest for 2 years at 5% per annum is 51.25,
    prove that the simple interest for the same period and rate is 250. -/
theorem simple_interest_calculation (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) = 51.25 → P * 0.05 * 2 = 250 := by
  sorry

end simple_interest_calculation_l985_98539


namespace uneaten_fish_l985_98506

def fish_cells : List Nat := [3, 4, 16, 12, 20, 6]

def cat_eating_rate : Nat := 3

theorem uneaten_fish (eaten_count : Nat) (total_time : Nat) :
  eaten_count = 5 →
  total_time * cat_eating_rate = (fish_cells.take eaten_count).sum →
  total_time > 0 →
  (fish_cells.take eaten_count).sum % cat_eating_rate = 1 →
  fish_cells[eaten_count]! = 6 := by
  sorry

end uneaten_fish_l985_98506


namespace bus_distance_l985_98561

/-- Represents the distance traveled by each mode of transportation -/
structure TravelDistances where
  total : ℝ
  plane : ℝ
  train : ℝ
  bus : ℝ

/-- The conditions of the travel problem -/
def travel_conditions (d : TravelDistances) : Prop :=
  d.total = 900 ∧
  d.plane = d.total / 3 ∧
  d.train = 2 / 3 * d.bus ∧
  d.total = d.plane + d.train + d.bus

/-- The theorem stating that under the given conditions, the bus travel distance is 360 km -/
theorem bus_distance (d : TravelDistances) (h : travel_conditions d) : d.bus = 360 := by
  sorry

end bus_distance_l985_98561


namespace simplify_fraction_l985_98535

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  (x^2 + 1) / (x - 1) - 2*x / (x - 1) = x - 1 := by
  sorry

end simplify_fraction_l985_98535


namespace chord_length_from_arc_and_angle_l985_98531

theorem chord_length_from_arc_and_angle (m : ℝ) (h : m > 0) :
  let arc_length := m
  let central_angle : ℝ := 120 * π / 180
  let radius := arc_length / central_angle
  let chord_length := 2 * radius * Real.sin (central_angle / 2)
  chord_length = (3 * Real.sqrt 3 / (4 * π)) * m :=
by sorry

end chord_length_from_arc_and_angle_l985_98531


namespace cost_of_fencing_square_l985_98588

/-- The cost of fencing a square -/
theorem cost_of_fencing_square (cost_per_side : ℕ) (h : cost_per_side = 79) : 
  4 * cost_per_side = 316 := by
  sorry

end cost_of_fencing_square_l985_98588


namespace correct_statements_l985_98534

theorem correct_statements :
  (∀ x : ℝ, x^2 > 0 → x ≠ 0) ∧
  (∀ x : ℝ, x > 1 → x^2 > x) :=
by sorry

end correct_statements_l985_98534


namespace collage_glue_drops_l985_98510

/-- Calculates the total number of glue drops needed for a collage -/
def total_glue_drops (num_friends : ℕ) (clippings_per_friend : ℕ) (glue_drops_per_clipping : ℕ) : ℕ :=
  num_friends * clippings_per_friend * glue_drops_per_clipping

/-- Proves that for 7 friends, 3 clippings per friend, and 6 drops of glue per clipping, 
    the total number of glue drops needed is 126 -/
theorem collage_glue_drops : 
  total_glue_drops 7 3 6 = 126 := by
  sorry

end collage_glue_drops_l985_98510


namespace collinear_vectors_problem_l985_98529

/-- Given vectors a, b, and c in ℝ², prove that if a + b is collinear with c, then the y-coordinate of c is 1. -/
theorem collinear_vectors_problem (a b c : ℝ × ℝ) 
    (ha : a = (1, 2))
    (hb : b = (1, -3))
    (hc : c.1 = -2) 
    (h_collinear : ∃ (k : ℝ), k ≠ 0 ∧ (a.1 + b.1, a.2 + b.2) = (k * c.1, k * c.2)) :
  c.2 = 1 := by
  sorry

end collinear_vectors_problem_l985_98529


namespace arithmetic_calculation_l985_98593

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 - 2^3 = 190 := by
  sorry

end arithmetic_calculation_l985_98593


namespace sufficient_but_not_necessary_l985_98583

theorem sufficient_but_not_necessary (a : ℝ) : 
  (∀ a, a > 2 → 1/a < 1/2) ∧ 
  (∃ a, 1/a < 1/2 ∧ ¬(a > 2)) :=
by sorry

end sufficient_but_not_necessary_l985_98583


namespace triangle_PPB_area_l985_98563

/-- A square with side length 10 inches -/
def square_side : ℝ := 10

/-- Point P is a vertex of the square -/
def P : ℝ × ℝ := (0, 0)

/-- Point B is on the side of the square -/
def B : ℝ × ℝ := (square_side, 0)

/-- Point Q is inside the square and 8 inches above P -/
def Q : ℝ × ℝ := (0, 8)

/-- PQ is perpendicular to PB -/
axiom PQ_perp_PB : (Q.1 - P.1) * (B.1 - P.1) + (Q.2 - P.2) * (B.2 - P.2) = 0

/-- The area of triangle PPB -/
def triangle_area : ℝ := 0.5 * square_side * 8

/-- Theorem: The area of triangle PPB is 40 square inches -/
theorem triangle_PPB_area : triangle_area = 40 := by sorry

end triangle_PPB_area_l985_98563


namespace c_range_theorem_l985_98516

/-- Proposition p: c^2 < c -/
def p (c : ℝ) : Prop := c^2 < c

/-- Proposition q: ∀x∈ℝ, x^2 + 4cx + 1 > 0 -/
def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 + 4*c*x + 1 > 0

/-- The range of c given the conditions -/
def c_range (c : ℝ) : Prop := c ∈ Set.Ioc (-1/2) 0 ∪ Set.Icc (1/2) 1

theorem c_range_theorem (c : ℝ) :
  (p c ∨ q c) ∧ ¬(p c ∧ q c) → c_range c :=
by sorry

end c_range_theorem_l985_98516


namespace seventh_root_unity_product_l985_98515

theorem seventh_root_unity_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 10 := by
  sorry

end seventh_root_unity_product_l985_98515


namespace laundry_dishes_multiple_l985_98587

theorem laundry_dishes_multiple : ∃ m : ℝ, 46 = m * 20 + 6 :=
by
  -- Proof goes here
  sorry

end laundry_dishes_multiple_l985_98587


namespace cubic_function_parallel_tangents_l985_98575

/-- Given a cubic function f(x) = x³ + ax + b where a ≠ b, and the tangent lines
    to the graph of f at x=a and x=b are parallel, prove that f(1) = 1. -/
theorem cubic_function_parallel_tangents (a b : ℝ) (h : a ≠ b) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x + b
  (∃ k : ℝ, (3*a^2 + a = k) ∧ (3*b^2 + a = k)) → f 1 = 1 := by
  sorry


end cubic_function_parallel_tangents_l985_98575


namespace problem_statement_l985_98517

theorem problem_statement (a b : ℝ) (ha : a = 3) (hb : b = 2) :
  2 * (a^3 + b^3) / (a^2 - a*b + b^2) = 10 := by
  sorry

end problem_statement_l985_98517


namespace x_plus_y_value_l985_98578

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 12) 
  (h2 : x + |y| - y = 10) : 
  x + y = 26/5 := by
sorry

end x_plus_y_value_l985_98578


namespace geometric_sequence_bound_l985_98512

/-- Given two geometric sequences with specified properties, prove that the first term of the first sequence must be less than 4/3 -/
theorem geometric_sequence_bound (a b : ℝ) (r_a r_b : ℝ) : 
  (∑' i, a * r_a ^ i = 1) →
  (∑' i, b * r_b ^ i = 1) →
  (∑' i, (a * r_a ^ i) ^ 2) * (∑' i, (b * r_b ^ i) ^ 2) = ∑' i, (a * r_a ^ i) * (b * r_b ^ i) →
  a < 4/3 :=
by sorry

end geometric_sequence_bound_l985_98512


namespace cube_root_of_negative_one_twenty_seventh_l985_98586

theorem cube_root_of_negative_one_twenty_seventh :
  ((-1 / 3 : ℝ) : ℝ)^3 = -1 / 27 := by sorry

end cube_root_of_negative_one_twenty_seventh_l985_98586


namespace binomial_expansion_coefficient_l985_98558

theorem binomial_expansion_coefficient (x : ℝ) :
  (1 + 2*x)^3 = 1 + 6*x + 12*x^2 + 8*x^3 :=
by sorry

end binomial_expansion_coefficient_l985_98558


namespace grass_seed_cost_l985_98548

/-- Represents the cost and weight of a bag of grass seed -/
structure GrassSeedBag where
  weight : Nat
  cost : Float

/-- Represents a purchase of grass seed bags -/
structure Purchase where
  bags : List GrassSeedBag
  totalWeight : Nat
  totalCost : Float

def tenPoundBag : GrassSeedBag := { weight := 10, cost := 20.43 }
def twentyFivePoundBag : GrassSeedBag := { weight := 25, cost := 32.20 }

/-- The optimal purchase satisfying the given conditions -/
def optimalPurchase (fivePoundBagCost : Float) : Purchase :=
  { bags := [twentyFivePoundBag, twentyFivePoundBag, twentyFivePoundBag, 
             { weight := 5, cost := fivePoundBagCost }],
    totalWeight := 80,
    totalCost := 3 * 32.20 + fivePoundBagCost }

theorem grass_seed_cost 
  (h1 : ∀ p : Purchase, p.totalWeight ≥ 65 ∧ p.totalWeight ≤ 80 → p.totalCost ≥ 98.68)
  (h2 : (optimalPurchase 2.08).totalCost = 98.68) :
  ∃ fivePoundBagCost : Float, fivePoundBagCost = 2.08 ∧ 
    (optimalPurchase fivePoundBagCost).totalCost = 98.68 ∧
    (optimalPurchase fivePoundBagCost).totalWeight ≥ 65 ∧
    (optimalPurchase fivePoundBagCost).totalWeight ≤ 80 := by
  sorry

end grass_seed_cost_l985_98548


namespace nikolai_faster_l985_98571

/-- Represents a mountain goat with a specific jump distance -/
structure Goat where
  name : String
  jump_distance : ℕ

/-- Calculates the number of jumps needed to cover a given distance -/
def jumps_needed (g : Goat) (distance : ℕ) : ℕ :=
  (distance + g.jump_distance - 1) / g.jump_distance

theorem nikolai_faster (nikolai gennady : Goat)
  (h1 : nikolai.jump_distance = 4)
  (h2 : gennady.jump_distance = 6)
  (h3 : jumps_needed nikolai 2000 * nikolai.jump_distance = 2000)
  (h4 : jumps_needed gennady 2000 * gennady.jump_distance = 2004) :
  jumps_needed nikolai 2000 < jumps_needed gennady 2000 := by
  sorry

#eval jumps_needed (Goat.mk "Nikolai" 4) 2000
#eval jumps_needed (Goat.mk "Gennady" 6) 2000

end nikolai_faster_l985_98571


namespace alex_has_more_listens_l985_98551

/-- Calculates total listens over 3 months given initial listens and monthly growth rate -/
def totalListens (initial : ℝ) (growthRate : ℝ) : ℝ :=
  initial + initial * growthRate + initial * growthRate^2

/-- Represents the streaming statistics for a song -/
structure SongStats where
  spotify : ℝ
  appleMusic : ℝ
  youtube : ℝ

/-- Calculates total listens across all platforms -/
def overallListens (initial : SongStats) (growth : SongStats) : ℝ :=
  totalListens initial.spotify growth.spotify +
  totalListens initial.appleMusic growth.appleMusic +
  totalListens initial.youtube growth.youtube

/-- Jordan's initial listens -/
def jordanInitial : SongStats := ⟨60000, 35000, 45000⟩

/-- Jordan's monthly growth rates -/
def jordanGrowth : SongStats := ⟨2, 1.5, 1.25⟩

/-- Alex's initial listens -/
def alexInitial : SongStats := ⟨75000, 50000, 65000⟩

/-- Alex's monthly growth rates -/
def alexGrowth : SongStats := ⟨1.5, 1.8, 1.1⟩

theorem alex_has_more_listens :
  overallListens alexInitial alexGrowth > overallListens jordanInitial jordanGrowth :=
by sorry

end alex_has_more_listens_l985_98551
