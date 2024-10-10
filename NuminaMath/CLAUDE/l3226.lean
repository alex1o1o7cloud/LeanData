import Mathlib

namespace girls_in_school_l3226_322683

theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) (girls_sampled : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : girls_sampled = 95) : 
  (girls_sampled : ℚ) / (total_girls : ℚ) = (sample_size : ℚ) / (total_students : ℚ) → 
  total_girls = 760 := by
  sorry

end girls_in_school_l3226_322683


namespace shortest_side_range_l3226_322689

/-- An obtuse triangle with sides x, x+1, and x+2 -/
structure ObtuseTriangle where
  x : ℝ
  is_obtuse : 0 < x ∧ x < x + 1 ∧ x + 1 < x + 2

/-- The range of the shortest side in an obtuse triangle -/
theorem shortest_side_range (t : ObtuseTriangle) : 1 < t.x ∧ t.x < 3 := by
  sorry

end shortest_side_range_l3226_322689


namespace extreme_value_implies_a_equals_one_l3226_322603

/-- The function f reaching an extreme value at x = 1 implies a = 1 -/
theorem extreme_value_implies_a_equals_one (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 + 2 * Real.sqrt x - 3 * Real.log x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = 1 := by
  sorry


end extreme_value_implies_a_equals_one_l3226_322603


namespace man_downstream_speed_l3226_322614

/-- Calculates the downstream speed given upstream and still water speeds -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem stating that given the specified upstream and still water speeds, 
    the downstream speed is 80 kmph -/
theorem man_downstream_speed :
  downstream_speed 20 50 = 80 := by
  sorry

end man_downstream_speed_l3226_322614


namespace largest_band_size_l3226_322696

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- Represents the band and its formations --/
structure Band where
  totalMembers : ℕ
  firstFormation : BandFormation
  secondFormation : BandFormation

/-- Checks if a band satisfies all given conditions --/
def satisfiesConditions (band : Band) : Prop :=
  band.totalMembers < 100 ∧
  band.totalMembers = band.firstFormation.rows * band.firstFormation.membersPerRow + 3 ∧
  band.totalMembers = band.secondFormation.rows * band.secondFormation.membersPerRow ∧
  band.secondFormation.rows = band.firstFormation.rows - 3 ∧
  band.secondFormation.membersPerRow = band.firstFormation.membersPerRow + 1

/-- The theorem stating that 75 is the largest possible number of band members --/
theorem largest_band_size :
  ∀ band : Band, satisfiesConditions band → band.totalMembers ≤ 75 :=
by sorry

end largest_band_size_l3226_322696


namespace min_crossing_time_for_four_people_l3226_322687

/-- Represents a person with their crossing time -/
structure Person where
  crossingTime : ℕ

/-- Represents the state of the bridge crossing problem -/
structure BridgeState where
  leftSide : List Person
  rightSide : List Person

/-- Calculates the minimum time required for all people to cross the bridge -/
def minCrossingTime (people : List Person) : ℕ :=
  sorry

/-- Theorem stating the minimum crossing time for the given problem -/
theorem min_crossing_time_for_four_people :
  let people := [
    { crossingTime := 2 },
    { crossingTime := 4 },
    { crossingTime := 6 },
    { crossingTime := 8 }
  ]
  minCrossingTime people = 10 := by
  sorry

end min_crossing_time_for_four_people_l3226_322687


namespace regular_pentagon_side_length_l3226_322666

/-- The length of a side of a regular pentagon with perimeter 125 is 25 -/
theorem regular_pentagon_side_length :
  ∀ (side_length : ℝ),
    side_length > 0 →
    side_length * 5 = 125 →
    side_length = 25 := by
  sorry

end regular_pentagon_side_length_l3226_322666


namespace nearest_integer_to_power_l3226_322639

theorem nearest_integer_to_power : ∃ n : ℤ, 
  n = 376 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5)^4 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^4 - (m : ℝ)| := by
  sorry

end nearest_integer_to_power_l3226_322639


namespace value_of_a_l3226_322623

theorem value_of_a (a b c : ℝ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 := by
  sorry

end value_of_a_l3226_322623


namespace factors_of_2_pow_96_minus_1_l3226_322601

theorem factors_of_2_pow_96_minus_1 :
  ∃ (a b : ℕ), 60 < a ∧ a < 70 ∧ 60 < b ∧ b < 70 ∧
  a ≠ b ∧
  (2^96 - 1) % a = 0 ∧ (2^96 - 1) % b = 0 ∧
  (∀ c : ℕ, 60 < c → c < 70 → c ≠ a → c ≠ b → (2^96 - 1) % c ≠ 0) ∧
  a = 63 ∧ b = 65 :=
by sorry

end factors_of_2_pow_96_minus_1_l3226_322601


namespace min_sum_grid_l3226_322662

theorem min_sum_grid (a b c d : ℕ+) : 
  a + b + c + d + a * b + c * d + a * c + b * d = 2015 →
  ∃ (w x y z : ℕ+), w + x + y + z ≤ a + b + c + d ∧
                    w + x + y + z + w * x + y * z + w * y + x * z = 2015 ∧
                    w + x + y + z = 88 :=
by sorry

end min_sum_grid_l3226_322662


namespace vs_length_l3226_322648

/-- A square piece of paper PQRS with side length 8 cm is folded so that corner R 
    coincides with T, the midpoint of PS. The crease UV intersects RS at V. -/
structure FoldedSquare where
  /-- Side length of the square -/
  side_length : ℝ
  /-- Point P -/
  P : ℝ × ℝ
  /-- Point Q -/
  Q : ℝ × ℝ
  /-- Point R -/
  R : ℝ × ℝ
  /-- Point S -/
  S : ℝ × ℝ
  /-- Point T (midpoint of PS) -/
  T : ℝ × ℝ
  /-- Point V (intersection of UV and RS) -/
  V : ℝ × ℝ
  /-- PQRS forms a square with side length 8 -/
  square_constraint : 
    P.1 = Q.1 ∧ Q.2 = R.2 ∧ R.1 = S.1 ∧ S.2 = P.2 ∧
    (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = side_length^2 ∧
    side_length = 8
  /-- T is the midpoint of PS -/
  midpoint_constraint : T = ((P.1 + S.1) / 2, (P.2 + S.2) / 2)
  /-- V is on RS -/
  v_on_rs : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ V = (R.1 * (1 - t) + S.1 * t, R.2 * (1 - t) + S.2 * t)
  /-- Distance RV equals distance TV (fold constraint) -/
  fold_constraint : (R.1 - V.1)^2 + (R.2 - V.2)^2 = (T.1 - V.1)^2 + (T.2 - V.2)^2

/-- The length of VS in the folded square is 3 cm -/
theorem vs_length (fs : FoldedSquare) : 
  ((fs.V.1 - fs.S.1)^2 + (fs.V.2 - fs.S.2)^2)^(1/2) = 3 := by
  sorry

end vs_length_l3226_322648


namespace polynomial_sum_and_coefficient_sum_l3226_322635

theorem polynomial_sum_and_coefficient_sum (d : ℝ) (h : d ≠ 0) :
  (15 * d^3 + 12 * d + 7 + 18 * d^2) + (2 * d^3 + d - 6 + 3 * d^2) =
  17 * d^3 + 21 * d^2 + 13 * d + 1 ∧
  17 + 21 + 13 + 1 = 52 := by
  sorry

end polynomial_sum_and_coefficient_sum_l3226_322635


namespace problem_one_problem_two_l3226_322667

-- Problem 1
theorem problem_one : (-1/3)⁻¹ + Real.sqrt 9 + (2 - Real.pi)^0 = 1 := by sorry

-- Problem 2
theorem problem_two (a : ℝ) (ha : a ≠ 0) (ha' : a ≠ 1) : 
  ((1/a - 1) / ((a^2 - 2*a + 1) / a)) = 1 / (1 - a) := by sorry

end problem_one_problem_two_l3226_322667


namespace binary_sum_equals_1945_l3226_322615

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_sum_equals_1945 :
  let num1 := binary_to_decimal [true, true, true, true, true, true, true, true, true, true]
  let num2 := binary_to_decimal [false, true, false, true, false, true, false, true, false, true]
  let num3 := binary_to_decimal [false, false, false, false, true, true, true, true]
  num1 + num2 + num3 = 1945 := by
  sorry

end binary_sum_equals_1945_l3226_322615


namespace vector_operation_proof_l3226_322669

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, -1]

theorem vector_operation_proof :
  (3 • b - a) = ![(-3 : ℝ), -5] := by sorry

end vector_operation_proof_l3226_322669


namespace equation_solutions_l3226_322617

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 25 = 0 ↔ x = 5 ∨ x = -5) ∧
  (∀ x : ℝ, 8 * (x - 1)^3 = 27 ↔ x = 5/2) := by
  sorry

end equation_solutions_l3226_322617


namespace angle_complement_from_supplement_l3226_322694

theorem angle_complement_from_supplement (angle : ℝ) : 
  (180 - angle = 130) → (90 - angle = 40) := by
  sorry

end angle_complement_from_supplement_l3226_322694


namespace square_measurement_error_l3226_322607

theorem square_measurement_error (S : ℝ) (S' : ℝ) (h : S' > 0) :
  S'^2 = S^2 * (1 + 0.0404) → (S' - S) / S * 100 = 2 := by
  sorry

end square_measurement_error_l3226_322607


namespace incorrect_factorization_l3226_322602

theorem incorrect_factorization (x : ℝ) : x^2 + x - 2 ≠ (x - 2) * (x + 1) := by
  sorry

end incorrect_factorization_l3226_322602


namespace myrtle_has_three_hens_l3226_322657

/-- The number of hens Myrtle has -/
def num_hens : ℕ := sorry

/-- The number of eggs each hen lays per day -/
def eggs_per_hen_per_day : ℕ := 3

/-- The number of days Myrtle was gone -/
def days_gone : ℕ := 7

/-- The number of eggs the neighbor took -/
def eggs_taken_by_neighbor : ℕ := 12

/-- The number of eggs Myrtle dropped -/
def eggs_dropped : ℕ := 5

/-- The number of eggs Myrtle has remaining -/
def eggs_remaining : ℕ := 46

/-- Theorem stating that Myrtle has 3 hens -/
theorem myrtle_has_three_hens :
  num_hens = 3 :=
by sorry

end myrtle_has_three_hens_l3226_322657


namespace complex_magnitude_l3226_322655

/-- Given that (1+2i)/(a+bi) = 1 - i, where i is the imaginary unit and a and b are real numbers,
    prove that |a+bi| = √10/2 -/
theorem complex_magnitude (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (1 + 2*i) / (a + b*i) = 1 - i) : 
  Complex.abs (a + b*i) = Real.sqrt 10 / 2 := by
  sorry

end complex_magnitude_l3226_322655


namespace power_composition_l3226_322641

theorem power_composition (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3*a + 2*b) = 72 := by
  sorry

end power_composition_l3226_322641


namespace missing_angle_measure_l3226_322663

/-- A convex polygon with n sides --/
structure ConvexPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- The sum of interior angles of a convex polygon --/
def interior_angle_sum (p : ConvexPolygon) : ℝ :=
  (p.n - 2) * 180

/-- The theorem to prove --/
theorem missing_angle_measure (p : ConvexPolygon) 
  (sum_without_one : ℝ) 
  (h_sum : sum_without_one = 3025) :
  interior_angle_sum p - sum_without_one = 35 := by
  sorry

end missing_angle_measure_l3226_322663


namespace remainder_eight_pow_215_mod_9_l3226_322680

theorem remainder_eight_pow_215_mod_9 : 8^215 % 9 = 8 := by
  sorry

end remainder_eight_pow_215_mod_9_l3226_322680


namespace smallest_w_l3226_322693

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 →
  is_factor (2^5) (936 * w) →
  is_factor (3^3) (936 * w) →
  is_factor (12^2) (936 * w) →
  936 = 2^3 * 3^1 * 13^1 →
  (∀ v : ℕ, v > 0 → 
    is_factor (2^5) (936 * v) → 
    is_factor (3^3) (936 * v) → 
    is_factor (12^2) (936 * v) → 
    w ≤ v) →
  w = 36 := by
sorry

end smallest_w_l3226_322693


namespace train_length_approx_l3226_322697

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length_approx (speed : ℝ) (time : ℝ) : 
  speed = 100 → time = 3.6 → ∃ (length : ℝ), 
  (abs (length - (speed * 1000 / 3600 * time)) < 0.5) ∧ 
  (round length = 100) := by
  sorry

#check train_length_approx

end train_length_approx_l3226_322697


namespace light_pulse_reflections_l3226_322664

theorem light_pulse_reflections :
  ∃ (n : ℕ), n > 0 ∧
  (∃ (a b : ℕ), (a + 2) * (b + 2) = 4042 ∧ Nat.gcd (a + 1) (b + 1) = 1 ∧ n = a + b) ∧
  (∀ (m : ℕ), m > 0 →
    (∃ (a b : ℕ), (a + 2) * (b + 2) = 4042 ∧ Nat.gcd (a + 1) (b + 1) = 1 ∧ m = a + b) →
    m ≥ n) ∧
  n = 129 :=
by sorry

end light_pulse_reflections_l3226_322664


namespace luke_weed_eating_earnings_l3226_322627

/-- Proves that Luke made $18 weed eating given the conditions of the problem -/
theorem luke_weed_eating_earnings :
  ∀ (weed_eating_earnings : ℕ),
    9 + weed_eating_earnings = 3 * 9 →
    weed_eating_earnings = 18 :=
by
  sorry

end luke_weed_eating_earnings_l3226_322627


namespace second_set_cost_l3226_322631

/-- The cost of a set of footballs and soccer balls -/
def cost_of_set (football_price : ℝ) (soccer_price : ℝ) (num_footballs : ℕ) (num_soccers : ℕ) : ℝ :=
  football_price * (num_footballs : ℝ) + soccer_price * (num_soccers : ℝ)

/-- The theorem stating the cost of the second set of balls -/
theorem second_set_cost :
  ∀ (football_price : ℝ),
  cost_of_set football_price 50 3 1 = 155 →
  cost_of_set football_price 50 2 3 = 220 :=
by
  sorry

end second_set_cost_l3226_322631


namespace abs_neg_three_equals_three_l3226_322685

theorem abs_neg_three_equals_three : |(-3 : ℤ)| = 3 := by
  sorry

end abs_neg_three_equals_three_l3226_322685


namespace problem_solution_l3226_322605

def f (a x : ℝ) := |2*x + a| - |2*x + 3|
def g (x : ℝ) := |x - 1| - 3

theorem problem_solution :
  (∀ x : ℝ, |g x| < 2 ↔ (2 < x ∧ x < 6) ∨ (-4 < x ∧ x < 0)) ∧
  (∀ a : ℝ, (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) ↔ (0 ≤ a ∧ a ≤ 6)) :=
by sorry

end problem_solution_l3226_322605


namespace alley_width_l3226_322695

theorem alley_width (l k h w : Real) : 
  l > 0 → 
  k > 0 → 
  h > 0 → 
  w > 0 → 
  k = l * Real.sin (π / 3) → 
  h = l * Real.sin (π / 6) → 
  w = k / Real.sqrt 3 → 
  w = h * Real.sqrt 3 → 
  w = l / 2 := by sorry

end alley_width_l3226_322695


namespace sum_of_fourth_powers_l3226_322610

theorem sum_of_fourth_powers (a b c : ℝ) 
  (h1 : (a^2 - b^2) + c^2 = 8)
  (h2 : a * b * c = 2) :
  a^4 + b^4 + c^4 = 70 := by
sorry

end sum_of_fourth_powers_l3226_322610


namespace tank_capacity_proof_l3226_322690

/-- The capacity of a water tank in liters -/
def tank_capacity : ℝ := 72

/-- The amount of water in the tank when it's 40% full -/
def water_at_40_percent : ℝ := 0.4 * tank_capacity

/-- The amount of water in the tank when it's 10% empty (90% full) -/
def water_at_90_percent : ℝ := 0.9 * tank_capacity

/-- Theorem stating the tank capacity based on the given condition -/
theorem tank_capacity_proof :
  water_at_90_percent - water_at_40_percent = 36 ∧
  tank_capacity = 72 :=
sorry

end tank_capacity_proof_l3226_322690


namespace function_always_positive_l3226_322688

/-- The function f(x) = (2-a^2)x + a is always positive in the interval [0,1] if and only if 0 < a < 2 -/
theorem function_always_positive (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, (2 - a^2) * x + a > 0) ↔ (0 < a ∧ a < 2) := by
  sorry

end function_always_positive_l3226_322688


namespace largest_prime_factor_of_3401_l3226_322632

theorem largest_prime_factor_of_3401 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 3401 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 3401 → q ≤ p :=
by sorry

end largest_prime_factor_of_3401_l3226_322632


namespace f_is_perfect_square_l3226_322637

/-- The number of ordered pairs (a,b) of positive integers such that ab/(a+b) divides N -/
def f (N : ℕ+) : ℕ := sorry

/-- f(N) is always a perfect square -/
theorem f_is_perfect_square (N : ℕ+) : ∃ (k : ℕ), f N = k ^ 2 := by sorry

end f_is_perfect_square_l3226_322637


namespace roots_shifted_polynomial_l3226_322634

theorem roots_shifted_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 5*x + 7 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 + 9*x^2 + 22*x + 19 = 0 ↔ x = a - 3 ∨ x = b - 3 ∨ x = c - 3) :=
by sorry

end roots_shifted_polynomial_l3226_322634


namespace unique_scalar_for_vector_equation_l3226_322658

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def cross_product : V → V → V := sorry

theorem unique_scalar_for_vector_equation
  (cross_product : V → V → V)
  (h_cross_product : ∀ (x y z : V) (r : ℝ),
    cross_product (r • x) y = r • cross_product x y ∧
    cross_product x y = -cross_product y x ∧
    cross_product (x + y) z = cross_product x z + cross_product y z) :
  ∃! k : ℝ, ∀ (a b c d : V),
    a + b + c + d = 0 →
    k • (cross_product b a) + cross_product b c + cross_product c a + cross_product d a = 0 :=
by sorry

end unique_scalar_for_vector_equation_l3226_322658


namespace cubic_equation_sum_l3226_322650

theorem cubic_equation_sum (p q r : ℝ) : 
  p^3 - 6*p^2 + 11*p = 12 →
  q^3 - 6*q^2 + 11*q = 12 →
  r^3 - 6*r^2 + 11*r = 12 →
  p * q / r + q * r / p + r * p / q = -23/12 := by
  sorry

end cubic_equation_sum_l3226_322650


namespace sin_2alpha_value_l3226_322673

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : 3 * Real.cos (2 * α) = Real.sin (π / 4 - α)) : 
  Real.sin (2 * α) = -17 / 18 := by
  sorry

end sin_2alpha_value_l3226_322673


namespace tangent_line_condition_l3226_322618

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^3
def curve2 (a x : ℝ) : ℝ := a*x^2 + x - 9

-- Define the tangent line condition
def is_tangent_to_both (a : ℝ) : Prop :=
  ∃ (m : ℝ), ∃ (x₀ : ℝ),
    -- The line passes through (1,0)
    m * (1 - x₀) = -curve1 x₀ ∧
    -- The line is tangent to y = x^3
    m = 3 * x₀^2 ∧
    -- The line is tangent to y = ax^2 + x - 9
    m = 2 * a * x₀ + 1 ∧
    -- The point (x₀, curve1 x₀) is on both curves
    curve1 x₀ = curve2 a x₀

-- The main theorem
theorem tangent_line_condition (a : ℝ) :
  is_tangent_to_both a → a = -1 ∨ a = -7 :=
sorry

end tangent_line_condition_l3226_322618


namespace one_third_of_nine_times_seven_l3226_322606

theorem one_third_of_nine_times_seven : (1 / 3 : ℚ) * (9 * 7) = 21 := by
  sorry

end one_third_of_nine_times_seven_l3226_322606


namespace shooting_competition_sequences_l3226_322613

/-- The number of ways to arrange a multiset with 4 A's, 3 B's, 2 C's, and 1 D -/
def shooting_sequences : ℕ := 12600

/-- The total number of targets -/
def total_targets : ℕ := 10

/-- The number of targets in column A -/
def targets_A : ℕ := 4

/-- The number of targets in column B -/
def targets_B : ℕ := 3

/-- The number of targets in column C -/
def targets_C : ℕ := 2

/-- The number of targets in column D -/
def targets_D : ℕ := 1

theorem shooting_competition_sequences :
  shooting_sequences = (total_targets.factorial) / 
    (targets_A.factorial * targets_B.factorial * 
     targets_C.factorial * targets_D.factorial) :=
by sorry

end shooting_competition_sequences_l3226_322613


namespace prob_all_red_4th_draw_eq_l3226_322647

/-- The number of white balls initially in the bag -/
def white_balls : ℕ := 8

/-- The number of red balls initially in the bag -/
def red_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- The probability of drawing all red balls exactly by the 4th draw -/
def prob_all_red_4th_draw : ℚ :=
  -- Event A: Red on 1st, White on 2nd and 3rd, Red on 4th
  (red_balls / total_balls) * ((white_balls + 1) / total_balls) * ((white_balls + 1) / total_balls) * (1 / total_balls) +
  -- Event B: White on 1st, Red on 2nd, White on 3rd, Red on 4th
  (white_balls / total_balls) * (red_balls / total_balls) * ((white_balls + 1) / total_balls) * (1 / total_balls) +
  -- Event C: White on 1st and 2nd, Red on 3rd and 4th
  (white_balls / total_balls) * (white_balls / total_balls) * (red_balls / total_balls) * (1 / total_balls)

theorem prob_all_red_4th_draw_eq : prob_all_red_4th_draw = 353 / 5000 := by
  sorry

end prob_all_red_4th_draw_eq_l3226_322647


namespace work_completion_time_l3226_322612

/-- The time it takes to complete a work given two workers with different rates and a delayed start for one worker. -/
theorem work_completion_time 
  (p_rate : ℝ) (q_rate : ℝ) (p_solo_days : ℝ) 
  (hp : p_rate = 1 / 80)
  (hq : q_rate = 1 / 48)
  (hp_solo : p_solo_days = 16) : 
  p_solo_days + (1 - p_rate * p_solo_days) / (p_rate + q_rate) = 40 := by
  sorry

#check work_completion_time

end work_completion_time_l3226_322612


namespace three_digit_addition_theorem_l3226_322625

/-- Represents a three-digit number in the form xyz --/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to a natural number --/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem three_digit_addition_theorem (a b : Nat) :
  let n1 : ThreeDigitNumber := ⟨4, a, 5, by sorry, by sorry, by sorry⟩
  let n2 : ThreeDigitNumber := ⟨4, 3, 8, by sorry, by sorry, by sorry⟩
  let result : ThreeDigitNumber := ⟨8, b, 3, by sorry, by sorry, by sorry⟩
  (n1.toNat + n2.toNat = result.toNat) →
  (result.toNat % 3 = 0) →
  a + b = 1 := by
  sorry

#check three_digit_addition_theorem

end three_digit_addition_theorem_l3226_322625


namespace square_gt_of_abs_gt_l3226_322644

theorem square_gt_of_abs_gt (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end square_gt_of_abs_gt_l3226_322644


namespace two_digit_number_digit_difference_l3226_322640

/-- 
Given a two-digit number where the difference between the original number 
and the number with interchanged digits is 27, prove that the difference 
between its two digits is 3.
-/
theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 27 → x - y = 3 := by
sorry

end two_digit_number_digit_difference_l3226_322640


namespace circle_bisection_l3226_322622

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def Circle.center (c : Circle) : ℝ × ℝ := sorry

/-- A line bisects a circle if it passes through the circle's center -/
def bisects (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  l.equation x₀ y₀

/-- The main theorem -/
theorem circle_bisection (c : Circle) (l : Line) (a : ℝ) :
  c.equation = (fun x y ↦ x^2 + y^2 + 2*x - 4*y = 0) →
  l.equation = (fun x y ↦ 3*x + y + a = 0) →
  bisects l c →
  a = 1 := by sorry

end circle_bisection_l3226_322622


namespace units_digit_theorem_l3226_322692

theorem units_digit_theorem : ∃ n : ℕ, (33 * 219^89 + 89^19) % 10 = 8 := by
  sorry

end units_digit_theorem_l3226_322692


namespace largest_tile_size_l3226_322653

-- Define the courtyard dimensions in centimeters
def courtyard_length : ℕ := 378
def courtyard_width : ℕ := 525

-- Define the tile size in centimeters
def tile_size : ℕ := 21

-- Theorem statement
theorem largest_tile_size :
  (courtyard_length % tile_size = 0) ∧
  (courtyard_width % tile_size = 0) ∧
  (∀ s : ℕ, s > tile_size →
    (courtyard_length % s ≠ 0) ∨ (courtyard_width % s ≠ 0)) :=
by sorry

end largest_tile_size_l3226_322653


namespace inequality_constraint_l3226_322671

theorem inequality_constraint (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end inequality_constraint_l3226_322671


namespace rectangle_width_length_ratio_l3226_322619

theorem rectangle_width_length_ratio (w : ℝ) : 
  w > 0 ∧ 2 * w + 2 * 10 = 30 → w / 10 = 1 / 2 := by
  sorry

end rectangle_width_length_ratio_l3226_322619


namespace acute_angle_sine_equivalence_l3226_322628

theorem acute_angle_sine_equivalence (α β : Real) 
  (h_α_acute : 0 < α ∧ α < Real.pi / 2)
  (h_β_acute : 0 < β ∧ β < Real.pi / 2) :
  (α > 2 * β) ↔ (Real.sin (α - β) > Real.sin β) := by
  sorry

end acute_angle_sine_equivalence_l3226_322628


namespace perpendicular_vectors_l3226_322616

theorem perpendicular_vectors (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![-2, x]
  (∀ i, i < 2 → a i * b i = 0) → x = -1 := by
sorry

end perpendicular_vectors_l3226_322616


namespace square_sum_product_equals_k_squared_l3226_322691

theorem square_sum_product_equals_k_squared (k : ℕ) : 
  2012^2 + 2010 * 2011 * 2013 * 2014 = k^2 ∧ k > 0 → k = 4048142 := by
  sorry

end square_sum_product_equals_k_squared_l3226_322691


namespace sin_value_fourth_quadrant_l3226_322649

theorem sin_value_fourth_quadrant (α : Real) (h1 : α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) 
  (h2 : Real.tan α = -5/12) : Real.sin α = -5/13 := by
  sorry

end sin_value_fourth_quadrant_l3226_322649


namespace january_oil_bill_l3226_322656

/-- Proves that January's oil bill is $120 given the specified conditions --/
theorem january_oil_bill (feb_bill jan_bill : ℚ) : 
  (feb_bill / jan_bill = 3 / 2) → 
  ((feb_bill + 20) / jan_bill = 5 / 3) →
  jan_bill = 120 := by
  sorry

end january_oil_bill_l3226_322656


namespace shooting_range_problem_l3226_322630

theorem shooting_range_problem :
  ∀ n k : ℕ,
  (10 < n) →
  (n < 20) →
  (5 * k = 3 * (n - k)) →
  (n = 16 ∧ k = 6) :=
by sorry

end shooting_range_problem_l3226_322630


namespace a_faster_than_b_l3226_322698

/-- Represents a person sawing wood -/
structure Sawyer where
  name : String
  sections : ℕ
  pieces : ℕ

/-- Calculates the number of cuts required for a single piece of wood -/
def cuts (s : Sawyer) : ℕ := s.sections - 1

/-- Calculates the total number of cuts made by a sawyer -/
def totalCuts (s : Sawyer) : ℕ := (s.pieces / s.sections) * cuts s

/-- Defines what it means for one sawyer to be faster than another -/
def isFasterThan (s1 s2 : Sawyer) : Prop := totalCuts s1 > totalCuts s2

theorem a_faster_than_b :
  let a : Sawyer := ⟨"A", 3, 24⟩
  let b : Sawyer := ⟨"B", 2, 28⟩
  isFasterThan a b := by sorry

end a_faster_than_b_l3226_322698


namespace largest_non_sum_of_30multiple_and_composite_l3226_322699

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- The statement to be proved -/
theorem largest_non_sum_of_30multiple_and_composite :
  ∀ n : ℕ, n > 93 →
    ∃ (k : ℕ) (c : ℕ), k > 0 ∧ isComposite c ∧ n = 30 * k + c :=
by sorry

end largest_non_sum_of_30multiple_and_composite_l3226_322699


namespace kylie_picks_220_apples_l3226_322646

/-- The number of apples Kylie picks in the first hour -/
def first_hour_apples : ℕ := 66

/-- The number of apples Kylie picks in the second hour -/
def second_hour_apples : ℕ := 2 * first_hour_apples

/-- The number of apples Kylie picks in the third hour -/
def third_hour_apples : ℕ := first_hour_apples / 3

/-- The total number of apples Kylie picks over three hours -/
def total_apples : ℕ := first_hour_apples + second_hour_apples + third_hour_apples

/-- Theorem stating that the total number of apples Kylie picks is 220 -/
theorem kylie_picks_220_apples : total_apples = 220 := by
  sorry

end kylie_picks_220_apples_l3226_322646


namespace rectangular_garden_area_l3226_322624

/-- The area of a rectangular garden -/
def garden_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular garden with length 12 m and width 5 m is 60 square meters -/
theorem rectangular_garden_area :
  garden_area 12 5 = 60 := by
  sorry

end rectangular_garden_area_l3226_322624


namespace race_finish_difference_l3226_322654

/-- The time difference between two runners in a race -/
def time_difference (race_distance : ℕ) (speed1 speed2 : ℕ) : ℕ :=
  race_distance * speed2 - race_distance * speed1

/-- Theorem: In a 12-mile race, a runner with 7 min/mile speed finishes 24 minutes 
    after a runner with 5 min/mile speed -/
theorem race_finish_difference :
  time_difference 12 5 7 = 24 := by sorry

end race_finish_difference_l3226_322654


namespace inequality_solution_l3226_322611

theorem inequality_solution (x : ℝ) : 
  (x^2 + x^3 - x^4) / (x + x^2 - x^3) ≥ -1 ↔ 
  (x ∈ Set.Icc (-1) ((1 - Real.sqrt 5) / 2) ∪ 
   Set.Ioo ((1 - Real.sqrt 5) / 2) 0 ∪ 
   Set.Ioo 0 ((1 + Real.sqrt 5) / 2) ∪ 
   Set.Ioi ((1 + Real.sqrt 5) / 2)) := by
sorry

end inequality_solution_l3226_322611


namespace extra_hours_worked_l3226_322608

def hours_week1 : ℕ := 35
def hours_week2 : ℕ := 35
def hours_week3 : ℕ := 48
def hours_week4 : ℕ := 48

theorem extra_hours_worked : 
  (hours_week3 + hours_week4) - (hours_week1 + hours_week2) = 26 := by
  sorry

end extra_hours_worked_l3226_322608


namespace highest_probability_greater_than_2_l3226_322682

-- Define a fair dice
def FairDice := Finset (Fin 6)

-- Define the probability of an event on a fair dice
def probability (event : Finset (Fin 6)) (dice : FairDice) : ℚ :=
  (event.card : ℚ) / (dice.card : ℚ)

-- Define the events
def less_than_2 (dice : FairDice) : Finset (Fin 6) :=
  dice.filter (λ x => x < 2)

def greater_than_2 (dice : FairDice) : Finset (Fin 6) :=
  dice.filter (λ x => x > 2)

def even_number (dice : FairDice) : Finset (Fin 6) :=
  dice.filter (λ x => x % 2 = 0)

-- Theorem statement
theorem highest_probability_greater_than_2 (dice : FairDice) :
  probability (greater_than_2 dice) dice > probability (even_number dice) dice ∧
  probability (greater_than_2 dice) dice > probability (less_than_2 dice) dice :=
sorry

end highest_probability_greater_than_2_l3226_322682


namespace code_decryption_probability_l3226_322626

theorem code_decryption_probability :
  let p := 1 / 5  -- probability of success for each person
  let n := 3      -- number of people
  let prob_at_least_two := 
    Finset.sum (Finset.range (n - 1 + 1)) (fun k => 
      if k ≥ 2 then Nat.choose n k * p^k * (1 - p)^(n - k) else 0)
  prob_at_least_two = 13 / 125 := by
sorry

end code_decryption_probability_l3226_322626


namespace sum_of_fractions_geq_one_l3226_322681

theorem sum_of_fractions_geq_one (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end sum_of_fractions_geq_one_l3226_322681


namespace wendy_picture_upload_l3226_322677

/-- The number of pictures Wendy uploaded to Facebook -/
def total_pictures : ℕ := 79

/-- The number of pictures in the first album -/
def first_album_pictures : ℕ := 44

/-- The number of additional albums -/
def additional_albums : ℕ := 5

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 7

/-- Theorem stating that the total number of pictures is correct -/
theorem wendy_picture_upload :
  total_pictures = first_album_pictures + additional_albums * pictures_per_additional_album :=
by sorry

end wendy_picture_upload_l3226_322677


namespace solutions_equation1_solutions_equation2_l3226_322652

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 10*x + 16 = 0
def equation2 (x : ℝ) : Prop := 2*x*(x-1) = x-1

-- Theorem for the first equation
theorem solutions_equation1 : 
  ∀ x : ℝ, equation1 x ↔ (x = 2 ∨ x = 8) :=
by sorry

-- Theorem for the second equation
theorem solutions_equation2 : 
  ∀ x : ℝ, equation2 x ↔ (x = 1 ∨ x = 1/2) :=
by sorry

end solutions_equation1_solutions_equation2_l3226_322652


namespace unique_m_value_l3226_322645

def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3*m + 2}

theorem unique_m_value : ∃! m : ℝ, 2 ∈ A m ∧ m = 3 := by sorry

end unique_m_value_l3226_322645


namespace gecko_lizard_insect_ratio_l3226_322642

theorem gecko_lizard_insect_ratio :
  let num_geckos : ℕ := 5
  let insects_per_gecko : ℕ := 6
  let num_lizards : ℕ := 3
  let total_insects : ℕ := 66
  let geckos_total := num_geckos * insects_per_gecko
  let lizards_total := total_insects - geckos_total
  let insects_per_lizard := lizards_total / num_lizards
  insects_per_lizard / insects_per_gecko = 2 :=
by sorry

end gecko_lizard_insect_ratio_l3226_322642


namespace compute_expression_l3226_322678

theorem compute_expression : 5 + 7 * (2 - 9)^2 = 348 := by
  sorry

end compute_expression_l3226_322678


namespace f_n_has_real_root_l3226_322684

def f (x : ℝ) : ℝ := x^2 + 2007*x + 1

def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => f (f_n n x)

theorem f_n_has_real_root (n : ℕ+) : ∃ x : ℝ, f_n n x = 0 := by
  sorry

end f_n_has_real_root_l3226_322684


namespace xyz_equals_two_l3226_322609

theorem xyz_equals_two
  (a b c x y z : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (eq_a : a = (b + c) / (x - 3))
  (eq_b : b = (a + c) / (y - 3))
  (eq_c : c = (a + b) / (z - 3))
  (sum_xy_xz_yz : x * y + x * z + y * z = 7)
  (sum_x_y_z : x + y + z = 3) :
  x * y * z = 2 := by
sorry


end xyz_equals_two_l3226_322609


namespace circle_equation_l3226_322638

def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

def circle_passes_through_vertices (cx cy r : ℝ) : Prop :=
  (cx - 4)^2 + cy^2 = r^2 ∧
  cx^2 + (cy - 2)^2 = r^2 ∧
  cx^2 + (cy + 2)^2 = r^2

def center_on_negative_x_axis (cx cy : ℝ) : Prop :=
  cx < 0 ∧ cy = 0

theorem circle_equation (cx cy r : ℝ) :
  ellipse 4 0 ∧ ellipse 0 2 ∧ ellipse 0 (-2) ∧
  circle_passes_through_vertices cx cy r ∧
  center_on_negative_x_axis cx cy →
  cx = -3/2 ∧ cy = 0 ∧ r = 5/2 :=
sorry

end circle_equation_l3226_322638


namespace parallel_vectors_subtraction_l3226_322651

/-- Given vectors a and b where a is parallel to b, prove that 2a - b = (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, 4]
  (∃ (k : ℝ), a = k • b) →
  (2 • a - b) = ![4, -8] := by
sorry

end parallel_vectors_subtraction_l3226_322651


namespace arithmetic_sequence_lower_bound_l3226_322675

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property a₁² + a₂ₙ₊₁² = 1 -/
def SequenceProperty (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a 1 ^ 2 + a (2 * n + 1) ^ 2 = 1

theorem arithmetic_sequence_lower_bound
  (a : ℕ → ℝ) (h1 : ArithmeticSequence a) (h2 : SequenceProperty a) :
  ∀ n : ℕ, a (n + 1) ^ 2 + a (3 * n + 1) ^ 2 ≥ 2 := by
  sorry

end arithmetic_sequence_lower_bound_l3226_322675


namespace arithmetic_sequence_a8_l3226_322659

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_a8 (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 4 + a 6 = 8)
  (h_a2 : a 2 = 3) :
  a 8 = 5 := by
  sorry

end arithmetic_sequence_a8_l3226_322659


namespace complex_fraction_evaluation_l3226_322633

theorem complex_fraction_evaluation : 
  (0.125 / 0.25 + (1 + 9/16) / 2.5) / 
  ((10 - 22 / 2.3) * 0.46 + 1.6) + 
  (17/20 + 1.9) * 0.5 = 2 := by
  sorry

end complex_fraction_evaluation_l3226_322633


namespace scrabble_score_calculation_l3226_322661

/-- Scrabble game score calculation -/
theorem scrabble_score_calculation 
  (brenda_turn1 : ℕ) 
  (david_turn1 : ℕ) 
  (brenda_turn2 : ℕ) 
  (david_turn2 : ℕ) 
  (brenda_lead_before_turn3 : ℕ) 
  (brenda_turn3 : ℕ) 
  (david_turn3 : ℕ) 
  (h1 : brenda_turn1 = 18) 
  (h2 : david_turn1 = 10) 
  (h3 : brenda_turn2 = 25) 
  (h4 : david_turn2 = 35) 
  (h5 : brenda_lead_before_turn3 = 22) 
  (h6 : brenda_turn3 = 15) 
  (h7 : david_turn3 = 32) : 
  (david_turn1 + david_turn2 + david_turn3) - (brenda_turn1 + brenda_turn2 + brenda_turn3) = 19 := by
  sorry

end scrabble_score_calculation_l3226_322661


namespace care_package_weight_ratio_l3226_322629

/-- Represents the weight of the care package at different stages --/
structure CarePackage where
  initial_weight : ℝ
  after_brownies : ℝ
  before_gummies : ℝ
  final_weight : ℝ

/-- Theorem stating the ratio of final weight to weight before gummies is 2:1 --/
theorem care_package_weight_ratio (package : CarePackage) : 
  package.initial_weight = 2 →
  package.after_brownies = 3 * package.initial_weight →
  package.before_gummies = package.after_brownies + 2 →
  package.final_weight = 16 →
  package.final_weight / package.before_gummies = 2 := by
  sorry


end care_package_weight_ratio_l3226_322629


namespace sum_of_squares_l3226_322668

theorem sum_of_squares (a b c : ℝ) 
  (h_arithmetic : (a + b + c) / 3 = 10)
  (h_geometric : (a * b * c) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/a + 1/b + 1/c) = 4) :
  a^2 + b^2 + c^2 = 576 := by
sorry

end sum_of_squares_l3226_322668


namespace parabola_focus_l3226_322643

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = 4 * x^2 - 3

-- Define the focus of a parabola
def is_focus (x y : ℝ) (p : ℝ → ℝ → Prop) : Prop :=
  ∀ (px py : ℝ), p px py →
    (px - x)^2 + (py - y)^2 = (py - (y - 1/4))^2

-- Theorem statement
theorem parabola_focus :
  is_focus 0 (-47/16) parabola := by sorry

end parabola_focus_l3226_322643


namespace distance_walked_l3226_322670

/-- Represents the walking pace in miles per hour -/
def pace : ℝ := 4

/-- Represents the time walked in hours -/
def time : ℝ := 2

/-- Theorem stating that the distance walked is the product of pace and time -/
theorem distance_walked : pace * time = 8 := by sorry

end distance_walked_l3226_322670


namespace sarah_meal_combinations_l3226_322686

/-- Represents the number of options for each meal component -/
structure MealOptions where
  appetizers : Nat
  mainCourses : Nat
  drinks : Nat
  desserts : Nat

/-- Represents the constraint on drink options when fries are chosen -/
def drinkOptionsWithFries (options : MealOptions) : Nat :=
  options.drinks - 1

/-- Calculates the number of meal combinations -/
def calculateMealCombinations (options : MealOptions) : Nat :=
  let mealsWithFries := 1 * options.mainCourses * (drinkOptionsWithFries options) * options.desserts
  let mealsWithoutFries := (options.appetizers - 1) * options.mainCourses * options.drinks * options.desserts
  mealsWithFries + mealsWithoutFries

/-- The main theorem stating the number of distinct meals Sarah can buy -/
theorem sarah_meal_combinations (options : MealOptions) 
  (h1 : options.appetizers = 3)
  (h2 : options.mainCourses = 3)
  (h3 : options.drinks = 3)
  (h4 : options.desserts = 2) : 
  calculateMealCombinations options = 48 := by
  sorry

#eval calculateMealCombinations { appetizers := 3, mainCourses := 3, drinks := 3, desserts := 2 }

end sarah_meal_combinations_l3226_322686


namespace range_of_a_l3226_322636

def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

theorem range_of_a (a : ℝ) (h : A ∩ B a = B a) : a ≤ 0 ∨ a ≥ 3 := by
  sorry

end range_of_a_l3226_322636


namespace x_plus_y_value_l3226_322620

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.sin y = 2008)
  (eq2 : x + 2008 * Real.cos y = 2007)
  (h : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := by
  sorry

end x_plus_y_value_l3226_322620


namespace unique_solution_l3226_322665

theorem unique_solution : ∃! (a b c d : ℤ),
  (a^2 - b^2 - c^2 - d^2 = c - b - 2) ∧
  (2*a*b = a - d - 32) ∧
  (2*a*c = 28 - a - d) ∧
  (2*a*d = b + c + 31) ∧
  (a = 5) ∧ (b = -3) ∧ (c = 2) ∧ (d = 3) := by
sorry

end unique_solution_l3226_322665


namespace smallest_non_factor_product_l3226_322676

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_non_factor_product (a b : ℕ) : 
  a ≠ b →
  a > 0 →
  b > 0 →
  is_factor a 48 →
  is_factor b 48 →
  ¬ is_factor (a * b) 48 →
  (∀ x y : ℕ, x ≠ y → x > 0 → y > 0 → is_factor x 48 → is_factor y 48 → 
    ¬ is_factor (x * y) 48 → a * b ≤ x * y) →
  a * b = 32 := by sorry

end smallest_non_factor_product_l3226_322676


namespace triangle_area_l3226_322621

theorem triangle_area (a b c : ℝ) (h_a : a = 39) (h_b : b = 36) (h_c : c = 15) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 270 :=
by sorry

end triangle_area_l3226_322621


namespace product_of_sums_equal_difference_of_powers_l3226_322604

theorem product_of_sums_equal_difference_of_powers : 
  (2^1 + 3^1) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * 
  (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := by
  sorry

end product_of_sums_equal_difference_of_powers_l3226_322604


namespace stating_danny_bottle_caps_l3226_322674

/-- Represents the number of bottle caps Danny had initially. -/
def initial_caps : ℕ := 69

/-- Represents the number of bottle caps Danny threw away. -/
def thrown_away_caps : ℕ := 60

/-- Represents the number of new bottle caps Danny found. -/
def new_caps : ℕ := 58

/-- Represents the number of bottle caps Danny has now. -/
def current_caps : ℕ := 67

/-- 
Theorem stating that the initial number of bottle caps minus the thrown away caps,
plus the new caps found, equals the current number of caps.
-/
theorem danny_bottle_caps : 
  initial_caps - thrown_away_caps + new_caps = current_caps := by
  sorry

#check danny_bottle_caps

end stating_danny_bottle_caps_l3226_322674


namespace candle_scent_ratio_l3226_322660

/-- Represents the number of candles made for each scent -/
structure CandleCounts where
  coconut : ℕ
  lavender : ℕ
  almond : ℕ

/-- Represents the amount of scent used for each type -/
structure ScentAmounts where
  coconut : ℝ
  lavender : ℝ
  almond : ℝ

/-- The theorem stating the relationship between candle counts and scent amounts -/
theorem candle_scent_ratio 
  (counts : CandleCounts) 
  (amounts : ScentAmounts) 
  (h1 : counts.lavender = 2 * counts.coconut) 
  (h2 : counts.almond = 10) 
  (h3 : ∀ (s : ScentAmounts), s.coconut = s.lavender ∧ s.coconut = s.almond) : 
  amounts.coconut / amounts.almond = counts.coconut / counts.almond :=
sorry

end candle_scent_ratio_l3226_322660


namespace art_arrangement_probability_l3226_322679

/-- The probability of arranging art pieces with specific conditions -/
theorem art_arrangement_probability (total_pieces : ℕ) (dali_paintings : ℕ) : 
  total_pieces = 12 →
  dali_paintings = 4 →
  (7 : ℚ) / 1485 = (
    (total_pieces - dali_paintings)  -- non-Dali pieces for first position
    * (total_pieces - dali_paintings)  -- positions for Dali group after first piece
    * (Nat.factorial (total_pieces - dali_paintings + 1))  -- arrangements of remaining pieces
  ) / (Nat.factorial total_pieces) := by
  sorry

#check art_arrangement_probability

end art_arrangement_probability_l3226_322679


namespace tricycle_wheels_l3226_322600

theorem tricycle_wheels (num_bicycles : ℕ) (num_tricycles : ℕ) (total_wheels : ℕ) :
  num_bicycles = 24 →
  num_tricycles = 14 →
  total_wheels = 90 →
  ∃ (tricycle_wheels : ℕ),
    tricycle_wheels = 3 ∧
    total_wheels = num_bicycles * 2 + num_tricycles * tricycle_wheels :=
by
  sorry

end tricycle_wheels_l3226_322600


namespace students_in_sports_l3226_322672

theorem students_in_sports (total : ℕ) (basketball soccer baseball cricket : ℕ)
  (basketball_soccer basketball_baseball basketball_cricket : ℕ)
  (soccer_baseball cricket_soccer cricket_baseball : ℕ)
  (basketball_cricket_soccer : ℕ) (no_sport : ℕ)
  (h1 : total = 200)
  (h2 : basketball = 50)
  (h3 : soccer = 60)
  (h4 : baseball = 35)
  (h5 : cricket = 80)
  (h6 : basketball_soccer = 10)
  (h7 : basketball_baseball = 15)
  (h8 : basketball_cricket = 20)
  (h9 : soccer_baseball = 25)
  (h10 : cricket_soccer = 30)
  (h11 : cricket_baseball = 5)
  (h12 : basketball_cricket_soccer = 10)
  (h13 : no_sport = 30) :
  basketball + soccer + baseball + cricket -
  basketball_soccer - basketball_baseball - basketball_cricket -
  soccer_baseball - cricket_soccer - cricket_baseball +
  basketball_cricket_soccer = 130 := by
  sorry

end students_in_sports_l3226_322672
