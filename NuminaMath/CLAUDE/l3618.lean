import Mathlib

namespace current_calculation_l3618_361899

/-- Given complex numbers V₁, V₂, Z, V, and I, prove that I = -1 + i -/
theorem current_calculation (V₁ V₂ Z V I : ℂ) 
  (h1 : V₁ = 2 + I)
  (h2 : V₂ = -1 + 4*I)
  (h3 : Z = 2 + 2*I)
  (h4 : V = V₁ + V₂)
  (h5 : I = V / Z) :
  I = -1 + I :=
by sorry

end current_calculation_l3618_361899


namespace ceiling_neg_sqrt_64_over_9_l3618_361875

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64 / 9)⌉ = -2 := by sorry

end ceiling_neg_sqrt_64_over_9_l3618_361875


namespace f_derivative_at_zero_l3618_361830

noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem f_derivative_at_zero : 
  deriv f 0 = -120 := by sorry

end f_derivative_at_zero_l3618_361830


namespace probability_of_overlap_l3618_361818

/-- Represents the duration of the entire time frame in minutes -/
def totalDuration : ℝ := 60

/-- Represents the waiting time of the train in minutes -/
def waitingTime : ℝ := 10

/-- Represents the area of the triangle in the graphical representation -/
def triangleArea : ℝ := 50

/-- Calculates the area of the parallelogram in the graphical representation -/
def parallelogramArea : ℝ := totalDuration * waitingTime

/-- Calculates the total area of overlap (favorable outcomes) -/
def overlapArea : ℝ := triangleArea + parallelogramArea

/-- Calculates the total area of all possible outcomes -/
def totalArea : ℝ := totalDuration * totalDuration

/-- Theorem stating the probability of Alex arriving while the train is at the station -/
theorem probability_of_overlap : overlapArea / totalArea = 11 / 72 := by
  sorry

end probability_of_overlap_l3618_361818


namespace complex_fraction_simplification_l3618_361888

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i :=
by sorry

end complex_fraction_simplification_l3618_361888


namespace equality_condition_for_sum_squares_equation_l3618_361837

theorem equality_condition_for_sum_squares_equation (a b c : ℝ) :
  (a^2 + b^2 + c^2 = a*b + b*c + a*c) ↔ (a = b ∧ b = c) :=
by sorry

end equality_condition_for_sum_squares_equation_l3618_361837


namespace profit_distribution_l3618_361893

theorem profit_distribution (share_a share_b share_c : ℕ) (total_profit : ℕ) : 
  share_a + share_b + share_c = total_profit →
  2 * share_a = 3 * share_b →
  3 * share_b = 5 * share_c →
  share_c - share_b = 4000 →
  total_profit = 20000 := by
sorry

end profit_distribution_l3618_361893


namespace computer_pricing_l3618_361879

theorem computer_pricing (C : ℝ) : 
  C + 0.60 * C = 2560 → C + 0.40 * C = 2240 := by sorry

end computer_pricing_l3618_361879


namespace find_x_l3618_361882

theorem find_x (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (4 * a) ^ (2 * b) = (a ^ b * x ^ b) ^ 2 → x = 4 := by
sorry

end find_x_l3618_361882


namespace intersection_locus_is_circle_l3618_361886

/-- The locus of intersection points of two parameterized lines forms a circle -/
theorem intersection_locus_is_circle :
  ∀ (u x y : ℝ), 
    (3 * u - 4 * y + 2 = 0) →
    (2 * x - 3 * u * y - 4 = 0) →
    ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 :=
by sorry

end intersection_locus_is_circle_l3618_361886


namespace tobias_lawn_mowing_charge_tobias_lawn_mowing_charge_proof_l3618_361884

/-- Tobias' lawn mowing problem -/
theorem tobias_lawn_mowing_charge : ℕ → Prop :=
  fun x =>
    let shoe_cost : ℕ := 95
    let saving_months : ℕ := 3
    let monthly_allowance : ℕ := 5
    let shovel_charge : ℕ := 7
    let remaining_money : ℕ := 15
    let lawns_mowed : ℕ := 4
    let driveways_shoveled : ℕ := 5
    
    (saving_months * monthly_allowance + lawns_mowed * x + driveways_shoveled * shovel_charge
      = shoe_cost + remaining_money) →
    x = 15

/-- The proof of Tobias' lawn mowing charge -/
theorem tobias_lawn_mowing_charge_proof : tobias_lawn_mowing_charge 15 := by
  sorry

end tobias_lawn_mowing_charge_tobias_lawn_mowing_charge_proof_l3618_361884


namespace vacation_homework_pages_l3618_361849

/-- Represents the number of days Garin divided her homework for -/
def days : ℕ := 24

/-- Represents the number of pages Garin can solve per day -/
def pages_per_day : ℕ := 19

/-- Calculates the total number of pages in Garin's vacation homework -/
def total_pages : ℕ := days * pages_per_day

/-- Proves that the total number of pages in Garin's vacation homework is 456 -/
theorem vacation_homework_pages : total_pages = 456 := by
  sorry

end vacation_homework_pages_l3618_361849


namespace happy_dictionary_problem_l3618_361895

theorem happy_dictionary_problem (a b : ℤ) (c : ℚ) : 
  (∀ n : ℤ, n > 0 → a ≤ n) → 
  (∀ n : ℤ, n < 0 → n ≤ b) → 
  (∀ q : ℚ, q ≠ 0 → |c| ≤ |q|) → 
  a - b + c = 2 := by
sorry

end happy_dictionary_problem_l3618_361895


namespace smallest_x_abs_equation_l3618_361838

theorem smallest_x_abs_equation : 
  (∀ x : ℝ, |2*x + 5| = 21 → x ≥ -13) ∧ 
  (|2*(-13) + 5| = 21) := by
sorry

end smallest_x_abs_equation_l3618_361838


namespace quadratic_inequality_solution_set_l3618_361840

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (2 + a) * x + 2 * a > 0}
  (a < 2 → S = {x : ℝ | x < a ∨ x > 2}) ∧
  (a = 2 → S = {x : ℝ | x ≠ 2}) ∧
  (a > 2 → S = {x : ℝ | x > a ∨ x < 2}) :=
by sorry

end quadratic_inequality_solution_set_l3618_361840


namespace f_satisfies_conditions_l3618_361812

open Complex

/-- The analytic function f(z) that satisfies the given conditions -/
noncomputable def f (z : ℂ) : ℂ := z^3 - 2*I*z + (2 + 3*I)

/-- The real part of f(z) -/
def u (x y : ℝ) : ℝ := x^3 - 3*x*y^2 + 2*y

theorem f_satisfies_conditions :
  (∀ x y : ℝ, (f (x + y*I)).re = u x y) ∧
  f I = 2 := by sorry

end f_satisfies_conditions_l3618_361812


namespace part1_part2_part3_l3618_361854

-- Define the operation
def matrixOp (a b c d : ℚ) : ℚ := a * d - c * b

-- Theorem 1
theorem part1 : matrixOp (-3) (-2) 4 5 = -7 := by sorry

-- Theorem 2
theorem part2 : matrixOp 2 (-2 * x) 3 (-5 * x) = 2 → x = -1/2 := by sorry

-- Theorem 3
theorem part3 (x : ℚ) : 
  matrixOp (8 * m * x - 1) (-8/3 + 2 * x) (3/2) (-3) = matrixOp 6 (-1) (-n) x →
  m = -3/8 ∧ n = -7 := by sorry

end part1_part2_part3_l3618_361854


namespace negation_of_every_prime_is_odd_l3618_361832

theorem negation_of_every_prime_is_odd :
  (¬ ∀ p : ℕ, Prime p → Odd p) ↔ (∃ p : ℕ, Prime p ∧ ¬ Odd p) :=
sorry

end negation_of_every_prime_is_odd_l3618_361832


namespace triangle_inequality_l3618_361898

theorem triangle_inequality (A B C : Real) (h : A + B + C = π) :
  Real.sin (A / 2) + Real.sin (B / 2) + Real.sin (C / 2) ≤ 1 + (1 / 2) * (Real.cos ((A - B) / 4))^2 := by
  sorry

end triangle_inequality_l3618_361898


namespace diagonals_are_space_l3618_361811

/-- A cube with diagonals forming a 60-degree angle --/
structure CubeWithDiagonals where
  /-- The measure of the angle between two diagonals --/
  angle : ℝ
  /-- The angle between the diagonals is 60 degrees --/
  angle_is_60 : angle = 60

/-- The types of diagonals in a cube --/
inductive DiagonalType
  | Face
  | Space

/-- Theorem: If the angle between two diagonals of a cube is 60 degrees,
    then these diagonals are space diagonals --/
theorem diagonals_are_space (c : CubeWithDiagonals) :
  ∃ (d : DiagonalType), d = DiagonalType.Space :=
sorry

end diagonals_are_space_l3618_361811


namespace train_crossing_time_l3618_361806

/-- Given two trains moving in opposite directions, this theorem proves
    the time taken for them to cross each other. -/
theorem train_crossing_time
  (train_length : ℝ)
  (faster_speed : ℝ)
  (h1 : train_length = 100)
  (h2 : faster_speed = 48)
  (h3 : faster_speed > 0) :
  let slower_speed := faster_speed / 2
  let relative_speed := faster_speed + slower_speed
  let total_distance := 2 * train_length
  let time := total_distance / (relative_speed * (1000 / 3600))
  time = 10 := by sorry

end train_crossing_time_l3618_361806


namespace wendys_final_tally_l3618_361878

/-- Calculates Wendy's final point tally for recycling --/
def wendys_points (cans_points newspaper_points cans_total cans_recycled newspapers_recycled penalty_points bonus_points bonus_cans_threshold bonus_newspapers_threshold : ℕ) : ℕ :=
  let points_earned := cans_points * cans_recycled + newspaper_points * newspapers_recycled
  let points_lost := penalty_points * (cans_total - cans_recycled)
  let bonus := if cans_recycled ≥ bonus_cans_threshold ∧ newspapers_recycled ≥ bonus_newspapers_threshold then bonus_points else 0
  points_earned - points_lost + bonus

/-- Theorem stating that Wendy's final point tally is 69 --/
theorem wendys_final_tally :
  wendys_points 5 10 11 9 3 3 15 10 2 = 69 := by
  sorry

end wendys_final_tally_l3618_361878


namespace smallest_disk_not_always_circumcircle_l3618_361808

/-- Three noncollinear points in the plane -/
structure ThreePoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  noncollinear : A ≠ B ∧ B ≠ C ∧ A ≠ C

/-- The radius of the smallest disk containing three points -/
def smallest_disk_radius (p : ThreePoints) : ℝ :=
  sorry

/-- The radius of the circumcircle of three points -/
def circumcircle_radius (p : ThreePoints) : ℝ :=
  sorry

/-- Theorem stating that the smallest disk is not always the circumcircle -/
theorem smallest_disk_not_always_circumcircle :
  ∃ p : ThreePoints, smallest_disk_radius p < circumcircle_radius p :=
sorry

end smallest_disk_not_always_circumcircle_l3618_361808


namespace absent_days_calculation_l3618_361866

/-- Calculates the number of days absent given the total days, daily wage, daily fine, and total earnings -/
def days_absent (total_days : ℕ) (daily_wage : ℕ) (daily_fine : ℕ) (total_earnings : ℕ) : ℕ :=
  total_days - (total_earnings + total_days * daily_fine) / (daily_wage + daily_fine)

theorem absent_days_calculation :
  days_absent 30 10 2 216 = 7 := by
  sorry

end absent_days_calculation_l3618_361866


namespace count_divisible_by_eight_l3618_361805

theorem count_divisible_by_eight (n : ℕ) : 
  (150 < n ∧ n ≤ 400 ∧ n % 8 = 0) → 
  (Finset.filter (λ x => 150 < x ∧ x ≤ 400 ∧ x % 8 = 0) (Finset.range 401)).card = 31 := by
  sorry

end count_divisible_by_eight_l3618_361805


namespace complex_product_real_l3618_361827

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Definition of a complex number being real -/
def is_real (z : ℂ) : Prop := z.im = 0

theorem complex_product_real (m : ℝ) :
  is_real ((2 + i) * (m - 2*i)) → m = 4 := by
  sorry

end complex_product_real_l3618_361827


namespace ac_value_l3618_361834

def letter_value (c : Char) : ℕ :=
  (c.toNat - 'A'.toNat + 1)

def word_value (w : String) : ℕ :=
  (w.toList.map letter_value).sum * w.length

theorem ac_value : word_value "ac" = 8 := by
  sorry

end ac_value_l3618_361834


namespace final_savings_is_105_l3618_361845

/-- Calculates the final savings amount after a series of bets and savings --/
def finalSavings (initialWinnings : ℝ) : ℝ :=
  let firstSavings := initialWinnings * 0.5
  let secondBetAmount := initialWinnings * 0.5
  let secondBetProfit := secondBetAmount * 0.6
  let secondBetTotal := secondBetAmount + secondBetProfit
  let secondSavings := secondBetTotal * 0.5
  let remainingAfterSecond := secondBetTotal
  let thirdBetAmount := remainingAfterSecond * 0.3
  let thirdBetProfit := thirdBetAmount * 0.25
  let thirdBetTotal := thirdBetAmount + thirdBetProfit
  let thirdSavings := thirdBetTotal * 0.5
  firstSavings + secondSavings + thirdSavings

/-- The theorem stating that the final savings amount is $105.00 --/
theorem final_savings_is_105 :
  finalSavings 100 = 105 := by
  sorry

end final_savings_is_105_l3618_361845


namespace round_trip_speed_l3618_361892

/-- Given a round trip between two points A and B, this theorem proves
    that if the distance is 120 miles, the speed from A to B is 60 mph,
    and the average speed for the entire trip is 45 mph, then the speed
    from B to A must be 36 mph. -/
theorem round_trip_speed (d : ℝ) (v_ab : ℝ) (v_avg : ℝ) (v_ba : ℝ) : 
  d = 120 → v_ab = 60 → v_avg = 45 → 
  (2 * d) / (d / v_ab + d / v_ba) = v_avg →
  v_ba = 36 := by
  sorry

#check round_trip_speed

end round_trip_speed_l3618_361892


namespace intersection_M_N_l3618_361807

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 5}
def N : Set ℝ := {x | x * (x - 4) > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | (-1 < x ∧ x < 0) ∨ (4 < x ∧ x < 5)} := by
  sorry

end intersection_M_N_l3618_361807


namespace lead_is_29_points_l3618_361852

/-- The lead in points between two teams -/
def lead (our_score green_score : ℕ) : ℕ :=
  our_score - green_score

/-- Theorem: Given the final scores, prove the lead is 29 points -/
theorem lead_is_29_points : lead 68 39 = 29 := by
  sorry

end lead_is_29_points_l3618_361852


namespace exponent_multiplication_l3618_361826

theorem exponent_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end exponent_multiplication_l3618_361826


namespace range_of_a_l3618_361856

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

def prop_p (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-1) 1, ∀ y ∈ Set.Icc (-1) 1, x ≤ y → f a x ≥ f a y

def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 1 ≤ 0

theorem range_of_a : 
  {a : ℝ | (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a)} = 
  Set.Iic (-2) ∪ Set.Icc 2 3 :=
sorry

end range_of_a_l3618_361856


namespace fermats_little_theorem_l3618_361825

theorem fermats_little_theorem (p a : ℕ) (hp : Prime p) (ha : ¬(p ∣ a)) :
  a^(p-1) ≡ 1 [MOD p] := by
  sorry

end fermats_little_theorem_l3618_361825


namespace basketball_lineup_combinations_l3618_361861

def total_players : ℕ := 16
def num_quadruplets : ℕ := 4
def num_starters : ℕ := 6
def num_quadruplets_in_lineup : ℕ := 2

theorem basketball_lineup_combinations :
  (Nat.choose num_quadruplets num_quadruplets_in_lineup) *
  (Nat.choose (total_players - num_quadruplets + num_quadruplets_in_lineup)
              (num_starters - num_quadruplets_in_lineup)) = 6006 := by
  sorry

end basketball_lineup_combinations_l3618_361861


namespace distance_A_to_C_distance_A_to_C_is_300_l3618_361850

/-- The distance between city A and city C given the travel times and speeds of Eddy and Freddy -/
theorem distance_A_to_C : ℝ :=
  let eddy_time : ℝ := 3
  let freddy_time : ℝ := 4
  let distance_A_to_B : ℝ := 570
  let speed_ratio : ℝ := 2.533333333333333

  let eddy_speed : ℝ := distance_A_to_B / eddy_time
  let freddy_speed : ℝ := eddy_speed / speed_ratio
  
  freddy_speed * freddy_time

/-- The distance between city A and city C is 300 km -/
theorem distance_A_to_C_is_300 : distance_A_to_C = 300 := by
  sorry

end distance_A_to_C_distance_A_to_C_is_300_l3618_361850


namespace largest_and_smallest_A_l3618_361851

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def move_last_digit_to_front (n : ℕ) : ℕ :=
  let d := n % 10
  let r := n / 10
  d * 10^7 + r

def satisfies_conditions (b : ℕ) : Prop :=
  b > 44444444 ∧ is_coprime b 12

theorem largest_and_smallest_A :
  ∃ (a_max a_min : ℕ),
    (∀ a b : ℕ, 
      a = move_last_digit_to_front b ∧ 
      satisfies_conditions b →
      a ≤ a_max ∧ a ≥ a_min) ∧
    a_max = 99999998 ∧
    a_min = 14444446 :=
sorry

end largest_and_smallest_A_l3618_361851


namespace only_one_student_passes_l3618_361870

theorem only_one_student_passes (prob_A prob_B prob_C : ℚ)
  (hA : prob_A = 4/5)
  (hB : prob_B = 3/5)
  (hC : prob_C = 7/10) :
  (prob_A * (1 - prob_B) * (1 - prob_C)) +
  ((1 - prob_A) * prob_B * (1 - prob_C)) +
  ((1 - prob_A) * (1 - prob_B) * prob_C) = 47/250 := by
  sorry

end only_one_student_passes_l3618_361870


namespace largest_angle_in_special_triangle_l3618_361809

/-- Proves that in a triangle where two angles sum to 7/6 of a right angle,
    and one of these angles is 36° larger than the other,
    the largest angle in the triangle is 75°. -/
theorem largest_angle_in_special_triangle : 
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  a + b = 105 →
  b = a + 36 →
  max a (max b c) = 75 := by
sorry

end largest_angle_in_special_triangle_l3618_361809


namespace max_b_value_l3618_361858

theorem max_b_value (a b c : ℕ) (h_volume : a * b * c = 360) 
  (h_order : 1 < c ∧ c < b ∧ b < a) (h_prime : Nat.Prime c) :
  b ≤ 12 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = 360 ∧ 1 < c' ∧ c' < b' ∧ b' < a' ∧ Nat.Prime c' ∧ b' = 12 :=
sorry

end max_b_value_l3618_361858


namespace luisa_trip_cost_l3618_361836

/-- Represents a leg of Luisa's trip -/
structure TripLeg where
  distance : Float
  fuelEfficiency : Float
  gasPrice : Float

/-- Calculates the cost of gas for a single leg of the trip -/
def gasCost (leg : TripLeg) : Float :=
  (leg.distance / leg.fuelEfficiency) * leg.gasPrice

/-- Luisa's trip legs -/
def luisaTrip : List TripLeg := [
  { distance := 10, fuelEfficiency := 15, gasPrice := 3.50 },
  { distance := 6,  fuelEfficiency := 12, gasPrice := 3.60 },
  { distance := 7,  fuelEfficiency := 14, gasPrice := 3.40 },
  { distance := 5,  fuelEfficiency := 10, gasPrice := 3.55 },
  { distance := 3,  fuelEfficiency := 13, gasPrice := 3.55 },
  { distance := 9,  fuelEfficiency := 15, gasPrice := 3.50 }
]

/-- Calculates the total cost of Luisa's trip -/
def totalTripCost : Float :=
  luisaTrip.map gasCost |> List.sum

/-- Proves that the total cost of Luisa's trip is approximately $10.53 -/
theorem luisa_trip_cost : 
  (totalTripCost - 10.53).abs < 0.01 := by
  sorry

end luisa_trip_cost_l3618_361836


namespace smallest_three_digit_with_product_8_and_even_digit_l3618_361860

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

def has_even_digit (n : ℕ) : Prop :=
  (n / 100) % 2 = 0 ∨ ((n / 10) % 10) % 2 = 0 ∨ (n % 10) % 2 = 0

theorem smallest_three_digit_with_product_8_and_even_digit :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → has_even_digit n → 124 ≤ n :=
sorry

end smallest_three_digit_with_product_8_and_even_digit_l3618_361860


namespace two_primes_equal_l3618_361842

theorem two_primes_equal (a b c : ℕ) 
  (hp : Nat.Prime (b^c + a))
  (hq : Nat.Prime (a^b + c))
  (hr : Nat.Prime (c^a + b)) :
  ∃ (x y : ℕ), x ≠ y ∧ 
    ((x = b^c + a ∧ y = a^b + c) ∨
     (x = b^c + a ∧ y = c^a + b) ∨
     (x = a^b + c ∧ y = c^a + b)) ∧
    x = y :=
sorry

end two_primes_equal_l3618_361842


namespace rectangular_prism_volume_l3618_361897

theorem rectangular_prism_volume 
  (x y z : ℕ) 
  (h1 : x > 0 ∧ y > 0 ∧ z > 0)
  (h2 : 4 * (x + y + z - 3) = 40)
  (h3 : 2 * (x * y + x * z + y * z - 2 * (x + y + z - 3)) = 66) :
  x * y * z = 150 :=
sorry

end rectangular_prism_volume_l3618_361897


namespace tomato_problem_l3618_361831

/-- The number of tomatoes produced by the first plant -/
def first_plant : ℕ := 19

/-- The number of tomatoes produced by the second plant -/
def second_plant (x : ℕ) : ℕ := x / 2 + 5

/-- The number of tomatoes produced by the third plant -/
def third_plant (x : ℕ) : ℕ := second_plant x + 2

/-- The total number of tomatoes produced by all three plants -/
def total_tomatoes : ℕ := 60

theorem tomato_problem :
  first_plant + second_plant first_plant + third_plant first_plant = total_tomatoes :=
by sorry

end tomato_problem_l3618_361831


namespace divisible_by_91_l3618_361853

theorem divisible_by_91 (n : ℕ) : 91 ∣ (5^n * (5^n + 1) - 6^n * (3^n + 2^n)) :=
sorry

end divisible_by_91_l3618_361853


namespace two_cos_sixty_degrees_l3618_361880

theorem two_cos_sixty_degrees : 2 * Real.cos (π / 3) = 1 := by
  sorry

end two_cos_sixty_degrees_l3618_361880


namespace right_triangle_hypotenuse_l3618_361867

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → c = 5 := by
  sorry

end right_triangle_hypotenuse_l3618_361867


namespace complement_S_union_T_equals_interval_l3618_361824

open Set Real

-- Define the sets S and T
def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

-- State the theorem
theorem complement_S_union_T_equals_interval :
  (𝒰 \ S) ∪ T = Iic 1 := by sorry

end complement_S_union_T_equals_interval_l3618_361824


namespace sum_first_eight_primes_mod_ninth_prime_l3618_361872

def first_nine_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23]

theorem sum_first_eight_primes_mod_ninth_prime : 
  (List.sum (List.take 8 first_nine_primes)) % (List.get! first_nine_primes 8) = 8 := by
  sorry

end sum_first_eight_primes_mod_ninth_prime_l3618_361872


namespace gcd_n_cube_plus_27_and_n_plus_3_l3618_361803

theorem gcd_n_cube_plus_27_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 := by
  sorry

end gcd_n_cube_plus_27_and_n_plus_3_l3618_361803


namespace fish_catching_average_l3618_361863

theorem fish_catching_average (aang_fish sokka_fish toph_fish : ℕ) 
  (h1 : aang_fish = 7)
  (h2 : sokka_fish = 5)
  (h3 : toph_fish = 12) :
  (aang_fish + sokka_fish + toph_fish) / 3 = 8 := by
  sorry

end fish_catching_average_l3618_361863


namespace atom_particle_count_l3618_361810

/-- Represents an atom with a given number of protons and mass number -/
structure Atom where
  protons : ℕ
  massNumber : ℕ

/-- Calculates the total number of fundamental particles in an atom -/
def totalParticles (a : Atom) : ℕ :=
  a.protons + (a.massNumber - a.protons) + a.protons

/-- Theorem: The total number of fundamental particles in an atom with 9 protons and mass number 19 is 28 -/
theorem atom_particle_count :
  let a : Atom := { protons := 9, massNumber := 19 }
  totalParticles a = 28 := by
  sorry


end atom_particle_count_l3618_361810


namespace triangle_rotation_path_length_l3618_361891

/-- The total path length of a vertex of an equilateral triangle rotating inside a square --/
theorem triangle_rotation_path_length 
  (triangle_side : ℝ) 
  (square_side : ℝ) 
  (rotation_angle : ℝ) 
  (h1 : triangle_side = 3) 
  (h2 : square_side = 6) 
  (h3 : rotation_angle = 60 * π / 180) : 
  (4 : ℝ) * 3 * triangle_side * rotation_angle = 12 * π := by
  sorry

#check triangle_rotation_path_length

end triangle_rotation_path_length_l3618_361891


namespace basket_weight_l3618_361847

/-- Proves that the weight of an empty basket is 1.40 kg given specific conditions -/
theorem basket_weight (total_weight : Real) (remaining_weight : Real) 
  (h1 : total_weight = 11.48)
  (h2 : remaining_weight = 8.12) : 
  ∃ (basket_weight : Real) (apple_weight : Real),
    basket_weight = 1.40 ∧ 
    apple_weight > 0 ∧
    total_weight = basket_weight + 12 * apple_weight ∧
    remaining_weight = basket_weight + 8 * apple_weight :=
by
  sorry

end basket_weight_l3618_361847


namespace equation_solution_l3618_361822

theorem equation_solution : ∃! x : ℝ, x + ((2 / 3 * 3 / 8) + 4) - 8 / 16 = 4.25 := by
  sorry

end equation_solution_l3618_361822


namespace pink_crayons_l3618_361835

def crayon_box (total red blue green yellow pink purple : ℕ) : Prop :=
  total = 48 ∧
  red = 12 ∧
  blue = 8 ∧
  green = (3 * blue) / 4 ∧
  yellow = (15 * total) / 100 ∧
  pink = purple ∧
  total = red + blue + green + yellow + pink + purple

theorem pink_crayons (total red blue green yellow pink purple : ℕ) :
  crayon_box total red blue green yellow pink purple → pink = 8 := by
  sorry

end pink_crayons_l3618_361835


namespace pirate_loot_sum_l3618_361865

/-- Converts a number from base 5 to base 10 -/
def base5To10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The loot values in base 5 -/
def jewelry : List Nat := [4, 2, 1, 3]
def goldCoins : List Nat := [2, 2, 1, 3]
def rubbingAlcohol : List Nat := [4, 2, 1]

theorem pirate_loot_sum :
  base5To10 jewelry + base5To10 goldCoins + base5To10 rubbingAlcohol = 865 := by
  sorry

end pirate_loot_sum_l3618_361865


namespace round_trip_percentage_l3618_361833

/-- Proves that 80% of passengers held round-trip tickets given the conditions -/
theorem round_trip_percentage (total_passengers : ℕ) 
  (h1 : (40 : ℝ) / 100 * total_passengers = (passengers_with_car : ℝ))
  (h2 : (50 : ℝ) / 100 * (passengers_with_roundtrip : ℝ) = passengers_with_roundtrip - passengers_with_car) :
  (80 : ℝ) / 100 * total_passengers = (passengers_with_roundtrip : ℝ) := by
  sorry

end round_trip_percentage_l3618_361833


namespace total_remaining_pictures_l3618_361877

structure ColoringBook where
  purchaseDay : Nat
  totalPictures : Nat
  coloredPerDay : Nat

def daysOfColoring (book : ColoringBook) : Nat :=
  6 - book.purchaseDay

def picturesColored (book : ColoringBook) : Nat :=
  book.coloredPerDay * daysOfColoring book

def picturesRemaining (book : ColoringBook) : Nat :=
  book.totalPictures - picturesColored book

def books : List ColoringBook := [
  ⟨1, 24, 4⟩,
  ⟨2, 37, 5⟩,
  ⟨3, 50, 6⟩,
  ⟨4, 33, 3⟩,
  ⟨5, 44, 7⟩
]

theorem total_remaining_pictures :
  (books.map picturesRemaining).sum = 117 := by
  sorry

end total_remaining_pictures_l3618_361877


namespace tower_blocks_sum_l3618_361864

/-- The total number of blocks in a tower after adding more blocks -/
def total_blocks (initial : Real) (added : Real) : Real :=
  initial + added

/-- Theorem: The total number of blocks is the sum of initial and added blocks -/
theorem tower_blocks_sum (initial : Real) (added : Real) :
  total_blocks initial added = initial + added := by
  sorry

end tower_blocks_sum_l3618_361864


namespace total_stickers_is_60_l3618_361804

/-- Represents the number of folders --/
def num_folders : Nat := 3

/-- Represents the number of sheets in each folder --/
def sheets_per_folder : Nat := 10

/-- Represents the number of stickers per sheet in the red folder --/
def red_stickers : Nat := 3

/-- Represents the number of stickers per sheet in the green folder --/
def green_stickers : Nat := 2

/-- Represents the number of stickers per sheet in the blue folder --/
def blue_stickers : Nat := 1

/-- Calculates the total number of stickers used --/
def total_stickers : Nat :=
  sheets_per_folder * red_stickers +
  sheets_per_folder * green_stickers +
  sheets_per_folder * blue_stickers

/-- Theorem stating that the total number of stickers used is 60 --/
theorem total_stickers_is_60 : total_stickers = 60 := by
  sorry

end total_stickers_is_60_l3618_361804


namespace expansion_theorem_l3618_361896

theorem expansion_theorem (a₀ a₁ a₂ a₃ a₄ : ℝ) : 
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end expansion_theorem_l3618_361896


namespace dot_product_theorem_l3618_361823

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 2)

theorem dot_product_theorem (c : ℝ × ℝ) 
  (h : c = (3 * a.1 + 2 * b.1 - a.1, 3 * a.2 + 2 * b.2 - a.2)) :
  a.1 * c.1 + a.2 * c.2 = 4 := by sorry

end dot_product_theorem_l3618_361823


namespace x_twelve_equals_negative_one_l3618_361829

theorem x_twelve_equals_negative_one (x : ℝ) (h : x + 1/x = Real.sqrt 2) : x^12 = -1 := by
  sorry

end x_twelve_equals_negative_one_l3618_361829


namespace evaluate_expression_l3618_361820

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/2) (hy : y = 1/3) (hz : z = -3) :
  (2*x)^2 * (y^2)^3 * z^2 = 1/81 := by
  sorry

end evaluate_expression_l3618_361820


namespace inequality_solution_set_l3618_361857

theorem inequality_solution_set (x : ℝ) : (2 * x - 1 ≤ 3) ↔ (x ≤ 2) := by
  sorry

end inequality_solution_set_l3618_361857


namespace express_y_in_terms_of_x_l3618_361816

-- Define the variables and conditions
theorem express_y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) :
  y = x / (x - 1) := by
  sorry

end express_y_in_terms_of_x_l3618_361816


namespace min_distance_sum_of_quadratic_roots_l3618_361868

theorem min_distance_sum_of_quadratic_roots : 
  ∃ (α β : ℝ), (α^2 - 6*α + 5 = 0) ∧ (β^2 - 6*β + 5 = 0) ∧
  (∀ x : ℝ, |x - α| + |x - β| ≥ 4) ∧
  (∃ x : ℝ, |x - α| + |x - β| = 4) := by
sorry

end min_distance_sum_of_quadratic_roots_l3618_361868


namespace total_weight_N2O3_l3618_361846

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula of Dinitrogen trioxide
def N_atoms_in_N2O3 : ℕ := 2
def O_atoms_in_N2O3 : ℕ := 3

-- Define the total molecular weight
def total_molecular_weight : ℝ := 228

-- Define the molecular weight of a single molecule of N2O3
def molecular_weight_N2O3 : ℝ := 
  N_atoms_in_N2O3 * atomic_weight_N * O_atoms_in_N2O3 * atomic_weight_O

-- Theorem: The total molecular weight of some moles of N2O3 is 228 g
theorem total_weight_N2O3 : 
  ∃ (n : ℝ), n * molecular_weight_N2O3 = total_molecular_weight :=
sorry

end total_weight_N2O3_l3618_361846


namespace valid_parameterizations_l3618_361828

/-- The slope of the line -/
def m : ℚ := 5 / 3

/-- The y-intercept of the line -/
def b : ℚ := 1

/-- The line equation: y = mx + b -/
def line_equation (x y : ℚ) : Prop := y = m * x + b

/-- A parameterization of a line -/
structure Parameterization where
  initial_point : ℚ × ℚ
  direction_vector : ℚ × ℚ

/-- Check if a parameterization is valid for the given line -/
def is_valid_parameterization (p : Parameterization) : Prop :=
  let (x₀, y₀) := p.initial_point
  let (dx, dy) := p.direction_vector
  line_equation x₀ y₀ ∧ dy / dx = m

/-- The five given parameterizations -/
def param_A : Parameterization := ⟨(3, 6), (3, 5)⟩
def param_B : Parameterization := ⟨(0, 1), (5, 3)⟩
def param_C : Parameterization := ⟨(1, 8/3), (5, 3)⟩
def param_D : Parameterization := ⟨(-1, -2/3), (3, 5)⟩
def param_E : Parameterization := ⟨(1, 1), (5, 8)⟩

theorem valid_parameterizations :
  is_valid_parameterization param_A ∧
  ¬is_valid_parameterization param_B ∧
  ¬is_valid_parameterization param_C ∧
  is_valid_parameterization param_D ∧
  ¬is_valid_parameterization param_E :=
sorry

end valid_parameterizations_l3618_361828


namespace sales_ratio_l3618_361885

/-- Proves that the ratio of sales on a tough week to sales on a good week is 1:2 -/
theorem sales_ratio (tough_week_sales : ℝ) (total_sales : ℝ) : 
  tough_week_sales = 800 →
  total_sales = 10400 →
  ∃ (good_week_sales : ℝ),
    5 * good_week_sales + 3 * tough_week_sales = total_sales ∧
    tough_week_sales / good_week_sales = 1 / 2 := by
  sorry

end sales_ratio_l3618_361885


namespace hyperbola_eccentricity_l3618_361844

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → e = Real.sqrt 3 :=
by sorry

end hyperbola_eccentricity_l3618_361844


namespace angela_age_in_five_years_l3618_361801

/-- Given that Angela is four times as old as Beth, and five years ago the sum of their ages was 45 years, prove that Angela will be 49 years old in five years. -/
theorem angela_age_in_five_years (angela beth : ℕ) 
  (h1 : angela = 4 * beth) 
  (h2 : angela - 5 + beth - 5 = 45) : 
  angela + 5 = 49 := by
  sorry

end angela_age_in_five_years_l3618_361801


namespace max_product_permutation_l3618_361873

theorem max_product_permutation (a : Fin 1987 → ℕ) 
  (h_perm : Function.Bijective a) 
  (h_range : Set.range a = Finset.range 1988) : 
  (Finset.range 1988).sup (λ k => k * a k) ≥ 994^2 := by
  sorry

end max_product_permutation_l3618_361873


namespace inequality_solution_range_l3618_361889

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
  sorry

end inequality_solution_range_l3618_361889


namespace correct_quotient_proof_l3618_361800

theorem correct_quotient_proof (D : ℕ) (h1 : D - 1000 = 1200 * 4900) : D / 2100 = 2800 := by
  sorry

end correct_quotient_proof_l3618_361800


namespace quadratic_equation_m_value_l3618_361815

/-- The equation is quadratic with respect to x if and only if m^2 - 2 = 2 -/
def is_quadratic (m : ℝ) : Prop := m^2 - 2 = 2

/-- The equation is not degenerate if and only if m - 2 ≠ 0 -/
def is_not_degenerate (m : ℝ) : Prop := m - 2 ≠ 0

theorem quadratic_equation_m_value :
  ∀ m : ℝ, is_quadratic m ∧ is_not_degenerate m → m = -2 :=
by sorry

end quadratic_equation_m_value_l3618_361815


namespace unique_prime_product_l3618_361814

theorem unique_prime_product (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  r * p^3 + p^2 + p = 2 * r * q^2 + q^2 + q →
  p * q * r = 2014 := by
sorry

end unique_prime_product_l3618_361814


namespace integer_solutions_eq1_integer_solutions_eq2_l3618_361841

-- Equation 1
theorem integer_solutions_eq1 :
  ∀ x y : ℤ, 11 * x + 5 * y = 7 ↔ ∃ t : ℤ, x = 2 - 5 * t ∧ y = -3 + 11 * t :=
sorry

-- Equation 2
theorem integer_solutions_eq2 :
  ∀ x y : ℤ, 4 * x + y = 3 * x * y ↔ (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
sorry

end integer_solutions_eq1_integer_solutions_eq2_l3618_361841


namespace investment_ratio_l3618_361894

/-- Represents the business investment scenario of Krishan and Nandan -/
structure BusinessInvestment where
  nandan_investment : ℝ
  nandan_time : ℝ
  krishan_investment_multiplier : ℝ
  total_gain : ℝ
  nandan_gain : ℝ

/-- The ratio of Krishan's investment to Nandan's investment is 4:1 -/
theorem investment_ratio (b : BusinessInvestment) : 
  b.nandan_time > 0 ∧ 
  b.nandan_investment > 0 ∧ 
  b.total_gain = 26000 ∧ 
  b.nandan_gain = 2000 ∧ 
  b.total_gain = b.nandan_gain + b.krishan_investment_multiplier * b.nandan_investment * (3 * b.nandan_time) →
  b.krishan_investment_multiplier = 4 := by
  sorry

end investment_ratio_l3618_361894


namespace three_digit_equation_l3618_361843

/-- 
Given a three-digit number A7B where 7 is the tens digit, 
prove that A = 6 if A7B + 23 = 695
-/
theorem three_digit_equation (A B : ℕ) : 
  (A * 100 + 70 + B) + 23 = 695 → 
  0 ≤ A ∧ A ≤ 9 → 
  0 ≤ B ∧ B ≤ 9 → 
  A = 6 := by
sorry

end three_digit_equation_l3618_361843


namespace largest_five_digit_with_product_120_l3618_361876

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digit_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

theorem largest_five_digit_with_product_120 :
  ∀ n : ℕ, is_five_digit n → digit_product n = 120 → n ≤ 85311 :=
by sorry

end largest_five_digit_with_product_120_l3618_361876


namespace largest_three_digit_multiple_of_8_with_digit_sum_16_l3618_361887

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_digit_sum_16 :
  ∀ n : ℕ, is_three_digit n → n % 8 = 0 → digit_sum n = 16 → n ≤ 952 :=
by sorry

end largest_three_digit_multiple_of_8_with_digit_sum_16_l3618_361887


namespace man_walking_distance_l3618_361819

theorem man_walking_distance (speed : ℝ) (time : ℝ) : 
  speed > 0 →
  time > 0 →
  (speed + 1/3) * (5/6 * time) = speed * time →
  (speed - 1/3) * (time + 3.5) = speed * time →
  speed * time = 35/96 := by
  sorry

end man_walking_distance_l3618_361819


namespace simplify_expression_l3618_361817

theorem simplify_expression : 110^2 - 109 * 111 = 1 := by
  sorry

end simplify_expression_l3618_361817


namespace trapezoid_area_l3618_361874

theorem trapezoid_area (outer_area inner_area : ℝ) (h1 : outer_area = 36) (h2 : inner_area = 4) :
  let total_trapezoid_area := outer_area - inner_area
  let num_trapezoids := 4
  (total_trapezoid_area / num_trapezoids : ℝ) = 8 := by
sorry

end trapezoid_area_l3618_361874


namespace vector_difference_magnitude_l3618_361855

/-- Given two vectors in 2D Euclidean space with specific magnitudes and angle between them,
    prove that the magnitude of their difference is √7. -/
theorem vector_difference_magnitude
  (a b : ℝ × ℝ)  -- Two vectors in 2D real space
  (h1 : ‖a‖ = 2)  -- Magnitude of a is 2
  (h2 : ‖b‖ = 3)  -- Magnitude of b is 3
  (h3 : a • b = 3)  -- Dot product of a and b (equivalent to 60° angle)
  : ‖a - b‖ = Real.sqrt 7 :=
by sorry

end vector_difference_magnitude_l3618_361855


namespace stating_mans_downstream_speed_l3618_361821

/-- 
Given a man's upstream speed and the speed of a stream, 
this function calculates his downstream speed.
-/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  (upstream_speed + stream_speed) + stream_speed

/-- 
Theorem stating that given the specific conditions of the problem,
the man's downstream speed is 11 kmph.
-/
theorem mans_downstream_speed : 
  downstream_speed 8 1.5 = 11 := by
  sorry

end stating_mans_downstream_speed_l3618_361821


namespace max_tiles_on_floor_l3618_361869

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit on a floor -/
def maxTiles (floor : Dimensions) (tile : Dimensions) : ℕ :=
  max
    ((floor.length / tile.length) * (floor.width / tile.width))
    ((floor.length / tile.width) * (floor.width / tile.length))

/-- Theorem stating the maximum number of tiles that can be placed on the given floor -/
theorem max_tiles_on_floor :
  let floor := Dimensions.mk 560 240
  let tile := Dimensions.mk 60 56
  maxTiles floor tile = 40 := by
  sorry

end max_tiles_on_floor_l3618_361869


namespace max_value_of_function_max_value_attained_l3618_361871

theorem max_value_of_function (x : ℝ) : 
  (3 * Real.sin x + 2 * Real.sqrt (2 + 2 * Real.cos (2 * x))) ≤ 5 := by
  sorry

theorem max_value_attained (x : ℝ) : 
  ∃ x, 3 * Real.sin x + 2 * Real.sqrt (2 + 2 * Real.cos (2 * x)) = 5 := by
  sorry

end max_value_of_function_max_value_attained_l3618_361871


namespace apple_difference_l3618_361813

theorem apple_difference (adam_apples jackie_apples : ℕ) 
  (adam_has : adam_apples = 9) 
  (jackie_has : jackie_apples = 10) : 
  jackie_apples - adam_apples = 1 := by
sorry

end apple_difference_l3618_361813


namespace quadratic_roots_relation_l3618_361881

theorem quadratic_roots_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∀ x : ℝ, x^2 + a*x + b = 0 → (2*x)^2 + b*(2*x) + c = 0) →
  a / c = 1 / 8 := by
sorry

end quadratic_roots_relation_l3618_361881


namespace soccer_ball_donation_l3618_361839

theorem soccer_ball_donation (total_balls : ℕ) (balls_per_class : ℕ) 
  (elementary_classes_per_school : ℕ) (num_schools : ℕ) 
  (h1 : total_balls = 90) 
  (h2 : balls_per_class = 5)
  (h3 : elementary_classes_per_school = 4)
  (h4 : num_schools = 2) : 
  (total_balls / (balls_per_class * num_schools)) - elementary_classes_per_school = 5 := by
  sorry

#check soccer_ball_donation

end soccer_ball_donation_l3618_361839


namespace cartesian_equation_circle_C_arc_length_ratio_circle_C_line_l_l3618_361862

-- Define the circle C in polar coordinates
def circle_C (ρ θ : ℝ) : Prop := ρ = 6 * Real.cos θ

-- Define the line l in parametric form
def line_l (t x y : ℝ) : Prop := x = 3 + (1/2) * t ∧ y = -3 + (Real.sqrt 3 / 2) * t

-- Theorem for the Cartesian equation of circle C
theorem cartesian_equation_circle_C :
  ∀ x y : ℝ, (∃ ρ θ : ℝ, circle_C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ 
  (x - 3)^2 + y^2 = 9 :=
sorry

-- Define a function to represent the ratio of arc lengths
def arc_length_ratio (r₁ r₂ : ℝ) : Prop := r₁ / r₂ = 1 / 2

-- Theorem for the ratio of arc lengths
theorem arc_length_ratio_circle_C_line_l :
  ∃ r₁ r₂ : ℝ, arc_length_ratio r₁ r₂ ∧ 
  (∀ x y : ℝ, (x - 3)^2 + y^2 = 9 → 
    (∃ t : ℝ, line_l t x y) → 
    (r₁ + r₂ = 2 * Real.pi * 3 ∧ r₁ ≤ r₂)) :=
sorry

end cartesian_equation_circle_C_arc_length_ratio_circle_C_line_l_l3618_361862


namespace binomial_150_1_l3618_361859

theorem binomial_150_1 : Nat.choose 150 1 = 150 := by sorry

end binomial_150_1_l3618_361859


namespace two_color_theorem_l3618_361802

/-- Represents a region in the plane --/
structure Region where
  id : Nat

/-- Represents the configuration of circles and lines --/
structure Configuration where
  regions : List Region
  adjacency : Region → Region → Bool

/-- Represents a coloring of regions --/
def Coloring := Region → Bool

/-- A valid coloring is one where adjacent regions have different colors --/
def is_valid_coloring (config : Configuration) (coloring : Coloring) : Prop :=
  ∀ r1 r2, config.adjacency r1 r2 → coloring r1 ≠ coloring r2

theorem two_color_theorem (config : Configuration) :
  ∃ (coloring : Coloring), is_valid_coloring config coloring := by
  sorry

end two_color_theorem_l3618_361802


namespace pheasants_and_rabbits_l3618_361883

theorem pheasants_and_rabbits (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 35)
  (h2 : total_legs = 94) :
  ∃ (pheasants rabbits : ℕ),
    pheasants + rabbits = total_heads ∧
    2 * pheasants + 4 * rabbits = total_legs ∧
    pheasants = 23 ∧
    rabbits = 12 := by
  sorry

end pheasants_and_rabbits_l3618_361883


namespace balls_after_2017_steps_l3618_361848

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 5) ((m % 5) :: acc)
    go n []

/-- Sums the digits in a list of natural numbers -/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.foldl (· + ·) 0

/-- The sum of digits in the base-5 representation of 2017 equals 9 -/
theorem balls_after_2017_steps : sumDigits (toBase5 2017) = 9 := by
  sorry


end balls_after_2017_steps_l3618_361848


namespace circle_area_circumference_difference_l3618_361890

theorem circle_area_circumference_difference (a b c : ℝ) (h1 : a = 24) (h2 : b = 70) (h3 : c = 74) 
  (h4 : a ^ 2 + b ^ 2 = c ^ 2) : 
  let r := c / 2
  (π * r ^ 2) - (2 * π * r) = 1295 * π := by
  sorry

end circle_area_circumference_difference_l3618_361890
