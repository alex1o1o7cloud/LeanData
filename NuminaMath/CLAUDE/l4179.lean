import Mathlib

namespace NUMINAMATH_CALUDE_check_amount_problem_l4179_417924

theorem check_amount_problem :
  ∃ (x y : ℕ),
    10 ≤ x ∧ x ≤ 99 ∧
    10 ≤ y ∧ y ≤ 99 ∧
    100 * y + x - (100 * x + y) = 1782 ∧
    y = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_check_amount_problem_l4179_417924


namespace NUMINAMATH_CALUDE_contest_ranking_l4179_417940

theorem contest_ranking (A B C D : ℝ) 
  (sum_equal : A + B = C + D)
  (interchange : C + A > D + B)
  (bob_highest : B > A + D)
  (nonnegative : A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ D ≥ 0) :
  B > A ∧ A > C ∧ C > D := by
  sorry

end NUMINAMATH_CALUDE_contest_ranking_l4179_417940


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l4179_417946

theorem longest_side_of_triangle (x : ℝ) : 
  7 + (x + 4) + (2 * x + 1) = 36 → 
  max 7 (max (x + 4) (2 * x + 1)) = 17 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l4179_417946


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l4179_417942

theorem least_subtraction_for_divisibility :
  ∃ (x : ℕ), x = 7 ∧
  12 ∣ (652543 - x) ∧
  ∀ (y : ℕ), y < x → ¬(12 ∣ (652543 - y)) :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l4179_417942


namespace NUMINAMATH_CALUDE_decimal_addition_l4179_417956

theorem decimal_addition : (5.467 : ℝ) + 3.92 = 9.387 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l4179_417956


namespace NUMINAMATH_CALUDE_no_valid_a_l4179_417920

theorem no_valid_a : ¬∃ a : ℝ, ∀ x : ℝ, x^2 + a*x + a - 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_a_l4179_417920


namespace NUMINAMATH_CALUDE_cookies_difference_l4179_417925

/-- Given Paco's cookie situation, prove the difference between bought and eaten cookies --/
theorem cookies_difference (initial : ℕ) (eaten : ℕ) (bought : ℕ)
  (h1 : initial = 13)
  (h2 : eaten = 2)
  (h3 : bought = 36) :
  bought - eaten = 34 := by
  sorry

end NUMINAMATH_CALUDE_cookies_difference_l4179_417925


namespace NUMINAMATH_CALUDE_honey_bee_count_l4179_417962

/-- The number of honey bees that produce a given amount of honey in a fixed time period. -/
def num_honey_bees (total_honey : ℕ) (honey_per_bee : ℕ) : ℕ :=
  total_honey / honey_per_bee

/-- Theorem stating that the number of honey bees is 30 given the problem conditions. -/
theorem honey_bee_count : num_honey_bees 30 1 = 30 := by
  sorry

end NUMINAMATH_CALUDE_honey_bee_count_l4179_417962


namespace NUMINAMATH_CALUDE_total_cats_is_31_l4179_417985

/-- The number of cats owned by Jamie, Gordon, Hawkeye, and Natasha -/
def total_cats : ℕ :=
  let jamie_persian := 4
  let jamie_maine_coon := 2
  let gordon_persian := jamie_persian / 2
  let gordon_maine_coon := jamie_maine_coon + 1
  let hawkeye_persian := 0
  let hawkeye_maine_coon := gordon_maine_coon * 2
  let natasha_persian := 3
  let natasha_maine_coon := jamie_maine_coon + gordon_maine_coon + hawkeye_maine_coon
  jamie_persian + jamie_maine_coon +
  gordon_persian + gordon_maine_coon +
  hawkeye_persian + hawkeye_maine_coon +
  natasha_persian + natasha_maine_coon

theorem total_cats_is_31 : total_cats = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_is_31_l4179_417985


namespace NUMINAMATH_CALUDE_south_side_maximum_l4179_417916

/-- Represents the number of paths for each side of the mountain -/
structure MountainPaths where
  east : Nat
  west : Nat
  south : Nat
  north : Nat

/-- Calculates the number of ways to ascend and descend for a given side -/
def waysForSide (paths : MountainPaths) (side : Nat) : Nat :=
  side * (paths.east + paths.west + paths.south + paths.north - side)

/-- Theorem stating that the south side provides the maximum number of ways -/
theorem south_side_maximum (paths : MountainPaths) 
    (h1 : paths.east = 2)
    (h2 : paths.west = 3)
    (h3 : paths.south = 4)
    (h4 : paths.north = 1) :
  ∀ side, waysForSide paths paths.south ≥ waysForSide paths side :=
by
  sorry

#eval waysForSide { east := 2, west := 3, south := 4, north := 1 } 4

end NUMINAMATH_CALUDE_south_side_maximum_l4179_417916


namespace NUMINAMATH_CALUDE_steve_final_marbles_l4179_417903

theorem steve_final_marbles (steve_initial sam_initial sally_initial : ℕ) 
  (h1 : sam_initial = 2 * steve_initial)
  (h2 : sally_initial = sam_initial - 5)
  (h3 : sam_initial - 6 = 8) : 
  steve_initial + 3 = 10 := by
sorry

end NUMINAMATH_CALUDE_steve_final_marbles_l4179_417903


namespace NUMINAMATH_CALUDE_incorrect_equation_l4179_417994

/-- Represents a repeating decimal -/
structure RepeatingDecimal where
  R : ℕ  -- non-repeating part
  S : ℕ  -- repeating part
  m : ℕ  -- number of non-repeating digits
  t : ℕ  -- number of repeating digits

/-- The value of the repeating decimal as a real number -/
noncomputable def value (E : RepeatingDecimal) : ℝ :=
  (E.R : ℝ) / 10^E.m + (E.S : ℝ) / (10^E.m * (10^E.t - 1))

/-- The statement that the equation is false for any repeating decimal -/
theorem incorrect_equation (E : RepeatingDecimal) :
  10^E.m * (10^E.t - 1) * (value E) ≠ E.S * (E.R - 1) :=
sorry

end NUMINAMATH_CALUDE_incorrect_equation_l4179_417994


namespace NUMINAMATH_CALUDE_quadratic_factorization_l4179_417993

theorem quadratic_factorization (x : ℝ) :
  x^2 + 4*x - 5 = 0 ↔ (x + 2)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l4179_417993


namespace NUMINAMATH_CALUDE_waiter_earnings_proof_l4179_417941

/-- Calculates the waiter's earnings from tips given the total number of customers,
    number of non-tipping customers, and the tip amount from each tipping customer. -/
def waiterEarnings (totalCustomers nonTippingCustomers tipAmount : ℕ) : ℕ :=
  (totalCustomers - nonTippingCustomers) * tipAmount

/-- Proves that the waiter's earnings are $27 given the specific conditions -/
theorem waiter_earnings_proof :
  waiterEarnings 7 4 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_waiter_earnings_proof_l4179_417941


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l4179_417952

/-- Represents an isosceles triangle with given properties -/
structure IsoscelesTriangle where
  perimeter : ℝ
  side_ratio : ℝ
  leg_length : ℝ
  h_perimeter_positive : perimeter > 0
  h_ratio_positive : side_ratio > 0
  h_leg_length_positive : leg_length > 0
  h_perimeter_eq : perimeter = (1 + 2 * side_ratio) * leg_length / side_ratio

/-- Theorem stating the possible leg lengths of the isosceles triangle -/
theorem isosceles_triangle_leg_length 
  (triangle : IsoscelesTriangle) 
  (h_perimeter : triangle.perimeter = 70) 
  (h_ratio : triangle.side_ratio = 3) : 
  triangle.leg_length = 14 ∨ triangle.leg_length = 30 := by
  sorry

#check isosceles_triangle_leg_length

end NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l4179_417952


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l4179_417982

/-- The focus of the parabola y² = 4x has coordinates (1, 0) -/
theorem parabola_focus_coordinates :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4*x}
  ∃ (f : ℝ × ℝ), f ∈ parabola ∧ f = (1, 0) ∧ 
    (∀ (p : ℝ × ℝ), p ∈ parabola → (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.1 - 0)^2 + (p.2 - 0)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l4179_417982


namespace NUMINAMATH_CALUDE_infinite_binary_decimal_divisible_by_2017_l4179_417975

/-- A number composed only of digits 0 and 1 in decimal representation -/
def is_binary_decimal (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The set of numbers composed only of digits 0 and 1 in decimal representation -/
def binary_decimal_set : Set ℕ :=
  {n : ℕ | is_binary_decimal n}

/-- The theorem statement -/
theorem infinite_binary_decimal_divisible_by_2017 :
  ∃ S : Set ℕ, (∀ n ∈ S, is_binary_decimal n ∧ 2017 ∣ n) ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_infinite_binary_decimal_divisible_by_2017_l4179_417975


namespace NUMINAMATH_CALUDE_four_weavers_four_days_eight_mats_l4179_417926

/-- The rate at which mat-weavers work, in mats per weaver per day -/
def weaving_rate (mats : ℕ) (weavers : ℕ) (days : ℕ) : ℚ :=
  (mats : ℚ) / (weavers * days)

/-- The number of mats that can be woven given a number of weavers, days, and a weaving rate -/
def mats_woven (weavers : ℕ) (days : ℕ) (rate : ℚ) : ℚ :=
  (weavers : ℚ) * days * rate

theorem four_weavers_four_days_eight_mats 
  (h : weaving_rate 16 8 8 = weaving_rate 8 4 4) : 
  mats_woven 4 4 (weaving_rate 16 8 8) = 8 := by
  sorry

end NUMINAMATH_CALUDE_four_weavers_four_days_eight_mats_l4179_417926


namespace NUMINAMATH_CALUDE_wage_recovery_percentage_l4179_417990

theorem wage_recovery_percentage (original_wage : ℝ) (h : original_wage > 0) :
  let decreased_wage := 0.7 * original_wage
  let required_increase := (original_wage / decreased_wage - 1) * 100
  ∃ ε > 0, abs (required_increase - 42.86) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_wage_recovery_percentage_l4179_417990


namespace NUMINAMATH_CALUDE_exp_2pi_3i_in_second_quadrant_l4179_417981

-- Define Euler's formula
axiom euler_formula (x : ℝ) : Complex.exp (x * Complex.I) = Complex.mk (Real.cos x) (Real.sin x)

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem exp_2pi_3i_in_second_quadrant :
  second_quadrant (Complex.exp ((2 * Real.pi / 3) * Complex.I)) :=
sorry

end NUMINAMATH_CALUDE_exp_2pi_3i_in_second_quadrant_l4179_417981


namespace NUMINAMATH_CALUDE_men_in_room_l4179_417974

/-- Represents the number of people in a room -/
structure RoomPopulation where
  men : ℕ
  women : ℕ

/-- Calculates the final number of men in the room -/
def finalMenCount (initial : RoomPopulation) : ℕ :=
  initial.men + 2

/-- Theorem: Given the initial conditions and final number of women,
    prove that there are 14 men in the room -/
theorem men_in_room (initial : RoomPopulation) 
    (h1 : initial.men = 4 * initial.women / 5)  -- Initial ratio 4:5
    (h2 : 2 * (initial.women - 3) = 24)         -- Final women count after changes
    : finalMenCount initial = 14 := by
  sorry


end NUMINAMATH_CALUDE_men_in_room_l4179_417974


namespace NUMINAMATH_CALUDE_arctans_and_arcsins_sum_l4179_417970

theorem arctans_and_arcsins_sum : 
  Real.arctan (1/3) + Real.arctan (1/5) + Real.arcsin (1/Real.sqrt 50) + Real.arcsin (1/Real.sqrt 65) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_arctans_and_arcsins_sum_l4179_417970


namespace NUMINAMATH_CALUDE_collinear_points_m_equals_four_l4179_417964

-- Define the points
def A : ℝ × ℝ := (-2, 12)
def B : ℝ × ℝ := (1, 3)
def C : ℝ → ℝ × ℝ := λ m ↦ (m, -6)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - p.1) = (r.2 - p.2) * (q.1 - p.1)

-- Theorem statement
theorem collinear_points_m_equals_four :
  collinear A B (C 4) := by sorry

end NUMINAMATH_CALUDE_collinear_points_m_equals_four_l4179_417964


namespace NUMINAMATH_CALUDE_sqrt_7x_equals_14_l4179_417963

theorem sqrt_7x_equals_14 (x : ℝ) (h : x / 2 - 5 = 9) : Real.sqrt (7 * x) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_7x_equals_14_l4179_417963


namespace NUMINAMATH_CALUDE_jessie_points_l4179_417912

def total_points : ℕ := 311
def other_players_points : ℕ := 188
def num_equal_scorers : ℕ := 3

theorem jessie_points : 
  (total_points - other_players_points) / num_equal_scorers = 41 := by
  sorry

end NUMINAMATH_CALUDE_jessie_points_l4179_417912


namespace NUMINAMATH_CALUDE_least_b_value_l4179_417929

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The smallest prime factor of a positive integer -/
def smallest_prime_factor (n : ℕ+) : ℕ := sorry

theorem least_b_value (a b : ℕ+) 
  (ha_factors : num_factors a = 3)
  (hb_factors : num_factors b = a)
  (hb_div_a : a ∣ b)
  (ha_smallest_prime : smallest_prime_factor a = 3) :
  36 ≤ b :=
sorry

end NUMINAMATH_CALUDE_least_b_value_l4179_417929


namespace NUMINAMATH_CALUDE_stock_market_investment_l4179_417906

theorem stock_market_investment
  (initial_investment : ℝ)
  (first_year_increase : ℝ)
  (net_increase : ℝ)
  (h1 : first_year_increase = 0.8)
  (h2 : net_increase = 0.26)
  : ∃ (second_year_decrease : ℝ),
    second_year_decrease = 0.3 ∧
    (1 + first_year_increase) * (1 - second_year_decrease) = 1 + net_increase :=
by sorry

end NUMINAMATH_CALUDE_stock_market_investment_l4179_417906


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l4179_417960

theorem solution_set_reciprocal_inequality (x : ℝ) :
  {x : ℝ | 1 / x > 3} = Set.Ioo 0 (1 / 3) := by sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l4179_417960


namespace NUMINAMATH_CALUDE_peter_book_percentage_l4179_417989

theorem peter_book_percentage (total_books : ℕ) (brother_percentage : ℚ) (difference : ℕ) : 
  total_books = 20 →
  brother_percentage = 1/10 →
  difference = 6 →
  (↑(brother_percentage * ↑total_books + ↑difference) / ↑total_books : ℚ) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_peter_book_percentage_l4179_417989


namespace NUMINAMATH_CALUDE_modulus_of_z_l4179_417927

theorem modulus_of_z (z : ℂ) (h : z^2 = 3 - 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l4179_417927


namespace NUMINAMATH_CALUDE_particle_position_after_2020_minutes_l4179_417919

/-- Represents the position of a particle -/
structure Position :=
  (x : ℤ)
  (y : ℤ)

/-- Calculates the position of the particle after a given number of minutes -/
def particle_position (minutes : ℕ) : Position :=
  sorry

/-- The movement pattern of the particle as described in the problem -/
axiom movement_pattern : 
  (∀ m : ℕ, m > 1 → 
    ∃ n : ℕ, 
      (particle_position (m - 1)).x < (particle_position m).x ∧ 
      (particle_position (m - 1)).y < (particle_position m).y) ∧
  (∀ m : ℕ, m > 2 → 
    ∃ n : ℕ, 
      (particle_position (m - 1)).x > (particle_position m).x ∧ 
      (particle_position (m - 1)).y > (particle_position m).y)

/-- The particle starts at the origin -/
axiom start_at_origin : particle_position 0 = ⟨0, 0⟩

/-- After one minute, the particle is at (1,1) -/
axiom first_minute : particle_position 1 = ⟨1, 1⟩

/-- The theorem to be proved -/
theorem particle_position_after_2020_minutes : 
  particle_position 2020 = ⟨30, 40⟩ := by
  sorry

end NUMINAMATH_CALUDE_particle_position_after_2020_minutes_l4179_417919


namespace NUMINAMATH_CALUDE_harry_pencils_left_l4179_417976

/-- Calculates the number of pencils left with Harry given Anna's pencils and Harry's lost pencils. -/
def pencils_left_with_harry (anna_pencils : ℕ) (harry_lost_pencils : ℕ) : ℕ :=
  2 * anna_pencils - harry_lost_pencils

/-- Proves that Harry has 81 pencils left given the problem conditions. -/
theorem harry_pencils_left : pencils_left_with_harry 50 19 = 81 := by
  sorry

end NUMINAMATH_CALUDE_harry_pencils_left_l4179_417976


namespace NUMINAMATH_CALUDE_milk_quantity_proof_l4179_417983

/-- The initial quantity of milk in container A --/
def initial_quantity : ℝ := 1216

/-- The fraction of milk in container B compared to A --/
def b_fraction : ℝ := 0.375

/-- The amount transferred between containers --/
def transfer_amount : ℝ := 152

/-- Theorem stating the initial quantity of milk in container A --/
theorem milk_quantity_proof :
  ∃ (a b c : ℝ),
    a = initial_quantity ∧
    b = b_fraction * a ∧
    c = a - b ∧
    b + transfer_amount = c - transfer_amount :=
by sorry

end NUMINAMATH_CALUDE_milk_quantity_proof_l4179_417983


namespace NUMINAMATH_CALUDE_repeat_perfect_square_exists_l4179_417907

theorem repeat_perfect_square_exists : ∃ (n : ℕ+) (k : ℕ), 
  (n : ℤ) * (10^k + 1) = (m : ℤ)^2 ∧ 
  10^k ≤ n ∧ n < 10^(k+1) :=
sorry

end NUMINAMATH_CALUDE_repeat_perfect_square_exists_l4179_417907


namespace NUMINAMATH_CALUDE_partition_to_magic_square_l4179_417979

/-- Represents a 3x3 square of integers -/
def Square : Type := Fin 3 → Fin 3 → ℤ

/-- Checks if a square is a magic square -/
def isMagicSquare (s : Square) : Prop :=
  let rowSum (i : Fin 3) := (s i 0) + (s i 1) + (s i 2)
  let colSum (j : Fin 3) := (s 0 j) + (s 1 j) + (s 2 j)
  let diagSum1 := (s 0 0) + (s 1 1) + (s 2 2)
  let diagSum2 := (s 0 2) + (s 1 1) + (s 2 0)
  ∀ i j : Fin 3, rowSum i = colSum j ∧ rowSum i = diagSum1 ∧ rowSum i = diagSum2

/-- Represents a partition of numbers from 1 to 360 into 9 subsets -/
def Partition : Type := Fin 9 → List ℕ

/-- Checks if a partition is valid (consecutive integers, sum to 360) -/
def isValidPartition (p : Partition) : Prop :=
  (∀ i : Fin 9, List.Chain' (·+1=·) (p i)) ∧
  (List.sum (List.join (List.ofFn p)) = 360) ∧
  (∀ i : Fin 9, p i ≠ [])

/-- The sum of a subset in the partition -/
def subsetSum (p : Partition) (i : Fin 9) : ℤ :=
  List.sum (p i)

/-- Theorem: It is possible to arrange the sums of a valid partition into a magic square -/
theorem partition_to_magic_square :
  ∃ (p : Partition) (s : Square), isValidPartition p ∧ isMagicSquare s ∧
  ∀ i : Fin 9, ∃ j k : Fin 3, s j k = subsetSum p i :=
sorry

end NUMINAMATH_CALUDE_partition_to_magic_square_l4179_417979


namespace NUMINAMATH_CALUDE_geometric_sequence_transformation_l4179_417986

/-- Given a geometric sequence {a_n} with common ratio q (q ≠ 1),
    prove that the sequence {b_n} defined as b_n = a_{3n-2} + a_{3n-1} + a_{3n}
    is a geometric sequence with common ratio q^3. -/
theorem geometric_sequence_transformation (q : ℝ) (hq : q ≠ 1) (a : ℕ → ℝ) 
    (h_geom : ∀ n : ℕ, a (n + 1) = q * a n) :
  let b : ℕ → ℝ := λ n ↦ a (3 * n - 2) + a (3 * n - 1) + a (3 * n)
  ∀ n : ℕ, b (n + 1) = q^3 * b n := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_transformation_l4179_417986


namespace NUMINAMATH_CALUDE_field_length_is_28_l4179_417953

/-- Proves that the length of a rectangular field is 28 meters given specific conditions --/
theorem field_length_is_28 (l w : ℝ) (h1 : l = 2 * w) (h2 : (7 : ℝ)^2 = (1/8) * (l * w)) : l = 28 :=
by sorry

end NUMINAMATH_CALUDE_field_length_is_28_l4179_417953


namespace NUMINAMATH_CALUDE_probability_of_three_in_eight_elevenths_l4179_417954

def decimal_representation (n d : ℕ) : List ℕ :=
  sorry -- Implementation of decimal representation calculation

theorem probability_of_three_in_eight_elevenths (n d : ℕ) (h : n = 8 ∧ d = 11) :
  let rep := decimal_representation n d
  (rep.count 3) / rep.length = 0 :=
sorry

end NUMINAMATH_CALUDE_probability_of_three_in_eight_elevenths_l4179_417954


namespace NUMINAMATH_CALUDE_remainder_of_B_l4179_417973

theorem remainder_of_B (A : ℕ) : (9 * A + 13) % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_B_l4179_417973


namespace NUMINAMATH_CALUDE_base_of_exponent_l4179_417955

theorem base_of_exponent (a : ℝ) (x : ℝ) (h1 : a^(2*x + 2) = 16^(3*x - 1)) (h2 : x = 1) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_of_exponent_l4179_417955


namespace NUMINAMATH_CALUDE_coal_consumption_factory_coal_consumption_l4179_417910

/-- Given a factory that burns coal at a constant daily rate, calculate the total coal burned over a longer period. -/
theorem coal_consumption (initial_coal : ℝ) (initial_days : ℝ) (total_days : ℝ) :
  initial_coal > 0 → initial_days > 0 → total_days > initial_days →
  (initial_coal / initial_days) * total_days = 
    initial_coal * (total_days / initial_days) := by
  sorry

/-- Specific instance of coal consumption calculation -/
theorem factory_coal_consumption :
  let initial_coal : ℝ := 37.5
  let initial_days : ℝ := 5
  let total_days : ℝ := 13
  (initial_coal / initial_days) * total_days = 97.5 := by
  sorry

end NUMINAMATH_CALUDE_coal_consumption_factory_coal_consumption_l4179_417910


namespace NUMINAMATH_CALUDE_solve_widgets_problem_l4179_417949

def widgets_problem (initial_widgets : ℕ) (total_money : ℕ) (price_reduction : ℕ) : Prop :=
  let initial_price := total_money / initial_widgets
  let new_price := initial_price - price_reduction
  let new_widgets := total_money / new_price
  new_widgets = 8

theorem solve_widgets_problem :
  widgets_problem 6 48 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_widgets_problem_l4179_417949


namespace NUMINAMATH_CALUDE_dragons_volleyball_games_l4179_417900

theorem dragons_volleyball_games :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (initial_games * 55 / 100) →
    (initial_wins + 8) = ((initial_games + 12) * 60 / 100) →
    initial_games + 12 = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_dragons_volleyball_games_l4179_417900


namespace NUMINAMATH_CALUDE_unique_solution_2000_l4179_417937

-- Define the differential equation
def diff_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ x > 0, deriv y x = Real.log (y x / x)

-- Define the solution y(x) with initial condition
noncomputable def y : ℝ → ℝ :=
  sorry

-- Main theorem
theorem unique_solution_2000 :
  ∃! x : ℝ, x > 0 ∧ y x = 2000 :=
by
  sorry


end NUMINAMATH_CALUDE_unique_solution_2000_l4179_417937


namespace NUMINAMATH_CALUDE_at_least_one_perpendicular_l4179_417988

structure GeometricSpace where
  Plane : Type
  Line : Type
  Point : Type

variable {G : GeometricSpace}

-- Define the necessary relations
def perpendicular (α β : G.Plane) : Prop := sorry
def contains (α : G.Plane) (l : G.Line) : Prop := sorry
def perpendicular_lines (l₁ l₂ : G.Line) : Prop := sorry
def perpendicular_line_plane (l : G.Line) (α : G.Plane) : Prop := sorry

-- State the theorem
theorem at_least_one_perpendicular
  (α β : G.Plane) (n m : G.Line)
  (h1 : perpendicular α β)
  (h2 : contains α n)
  (h3 : contains β m)
  (h4 : perpendicular_lines m n) :
  perpendicular_line_plane n β ∨ perpendicular_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_at_least_one_perpendicular_l4179_417988


namespace NUMINAMATH_CALUDE_equation_equivalence_l4179_417972

theorem equation_equivalence (p q : ℝ) 
  (hp1 : p ≠ 0) (hp2 : p ≠ 5) (hq1 : q ≠ 0) (hq2 : q ≠ 7) :
  (3 / p + 4 / q = 1 / 3) ↔ (9 * q / (q - 12) = p) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l4179_417972


namespace NUMINAMATH_CALUDE_sandwich_jam_cost_l4179_417943

/-- Represents the cost of jam used in sandwiches given specific conditions -/
theorem sandwich_jam_cost :
  ∀ (N B J : ℕ),
  N > 1 →
  B > 0 →
  J > 0 →
  N * (4 * B + 5 * J) = 253 →
  (N * J * 5 : ℚ) / 100 = 1.65 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_jam_cost_l4179_417943


namespace NUMINAMATH_CALUDE_class_gpa_calculation_l4179_417980

/-- Calculates the overall GPA of a class given the number of students and their GPAs in three groups -/
def overall_gpa (total_students : ℕ) (group1_students : ℕ) (group1_gpa : ℚ) 
                (group2_students : ℕ) (group2_gpa : ℚ)
                (group3_students : ℕ) (group3_gpa : ℚ) : ℚ :=
  (group1_students * group1_gpa + group2_students * group2_gpa + group3_students * group3_gpa) / total_students

/-- Theorem stating that the overall GPA of the class is 1030/60 -/
theorem class_gpa_calculation :
  overall_gpa 60 20 15 15 17 25 19 = 1030 / 60 := by
  sorry

#eval overall_gpa 60 20 15 15 17 25 19

end NUMINAMATH_CALUDE_class_gpa_calculation_l4179_417980


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l4179_417930

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 3) : 1 + a - b = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l4179_417930


namespace NUMINAMATH_CALUDE_gcf_of_2836_and_8965_l4179_417971

theorem gcf_of_2836_and_8965 : Nat.gcd 2836 8965 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_2836_and_8965_l4179_417971


namespace NUMINAMATH_CALUDE_sum_of_radii_l4179_417928

/-- A circle with center C is tangent to positive x and y-axes and externally tangent to a circle centered at (5,0) with radius 2. -/
structure TangentCircle where
  r : ℝ
  center : ℝ × ℝ
  tangent_to_axes : center = (r, r)
  tangent_to_other_circle : (r - 5)^2 + r^2 = (r + 2)^2

/-- The sum of all possible radii of the circle with center C is 14. -/
theorem sum_of_radii (c : TangentCircle) : 
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ (c.r = r1 ∨ c.r = r2) ∧ r1 + r2 = 14 :=
sorry

end NUMINAMATH_CALUDE_sum_of_radii_l4179_417928


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l4179_417914

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 3)^2) - Real.sqrt ((x - 7)^2 + (y + 3)^2) = 4

/-- The positive slope of an asymptote of the hyperbola -/
def positive_asymptote_slope : ℝ := 0.75

/-- Theorem stating that the positive slope of an asymptote of the given hyperbola is 0.75 -/
theorem hyperbola_asymptote_slope :
  ∃ (x y : ℝ), hyperbola_eq x y ∧ positive_asymptote_slope = 0.75 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l4179_417914


namespace NUMINAMATH_CALUDE_floor_equation_solution_range_l4179_417987

theorem floor_equation_solution_range (a : ℝ) : 
  (∃ x : ℕ+, ⌊(x + a) / 3⌋ = 2) → a < 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_range_l4179_417987


namespace NUMINAMATH_CALUDE_scientific_notation_of_8500_l4179_417967

theorem scientific_notation_of_8500 : 
  ∃ (a : ℝ) (n : ℤ), 8500 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_8500_l4179_417967


namespace NUMINAMATH_CALUDE_cos_45_sin_15_minus_sin_45_cos_15_l4179_417996

theorem cos_45_sin_15_minus_sin_45_cos_15 :
  Real.cos (45 * π / 180) * Real.sin (15 * π / 180) - 
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_45_sin_15_minus_sin_45_cos_15_l4179_417996


namespace NUMINAMATH_CALUDE_well_depth_solution_l4179_417922

def well_depth_problem (d : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ),
    d = 20 * t₁^2 ∧  -- Stone's fall distance
    d = 1100 * t₂ ∧  -- Sound's travel distance
    t₁ + t₂ = 10 ∧   -- Total time
    d = 122500       -- Depth to prove

theorem well_depth_solution :
  ∃ d : ℝ, well_depth_problem d :=
sorry

end NUMINAMATH_CALUDE_well_depth_solution_l4179_417922


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l4179_417977

/-- Checks if three lengths can form a triangle according to the triangle inequality theorem -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_theorem :
  can_form_triangle 3 4 5 ∧
  ¬can_form_triangle 2 4 7 ∧
  ¬can_form_triangle 3 6 9 ∧
  ¬can_form_triangle 4 4 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l4179_417977


namespace NUMINAMATH_CALUDE_cube_surface_area_l4179_417947

/-- The surface area of a cube with volume 64 cubic cm is 96 square cm. -/
theorem cube_surface_area (cube_volume : ℝ) (h : cube_volume = 64) : 
  6 * (cube_volume ^ (1/3))^2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l4179_417947


namespace NUMINAMATH_CALUDE_cuboid_diagonal_l4179_417935

/-- 
Given a cuboid with edge lengths a, b, and c, where:
- ab = √2
- bc = √3
- ca = √6
The length of the diagonal is √6.
-/
theorem cuboid_diagonal (a b c : ℝ) 
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : c * a = Real.sqrt 6) : 
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_l4179_417935


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l4179_417921

theorem other_root_of_quadratic (a : ℝ) : 
  (1^2 + a*1 + 2 = 0) → ∃ b : ℝ, b ≠ 1 ∧ b^2 + a*b + 2 = 0 ∧ b = 2 :=
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l4179_417921


namespace NUMINAMATH_CALUDE_consecutive_integers_median_l4179_417992

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (median : ℕ) : 
  n = 81 → sum = 9^5 → sum = n * median → median = 729 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_median_l4179_417992


namespace NUMINAMATH_CALUDE_subtract_inequality_l4179_417915

theorem subtract_inequality {a b c : ℝ} (h : a > b) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtract_inequality_l4179_417915


namespace NUMINAMATH_CALUDE_triangle_third_side_range_l4179_417904

theorem triangle_third_side_range (a b x : ℕ) : 
  a = 7 → b = 10 → (∃ (s : ℕ), s = x ∧ 4 ≤ s ∧ s ≤ 16) ↔ 
  (a + b > x ∧ x + a > b ∧ x + b > a) := by sorry

end NUMINAMATH_CALUDE_triangle_third_side_range_l4179_417904


namespace NUMINAMATH_CALUDE_A_maximized_at_19_l4179_417969

def factorial (n : ℕ) : ℕ := Nat.factorial n

def A (n : ℕ+) : ℚ := (20^n.val + 11^n.val) / factorial n.val

theorem A_maximized_at_19 : ∀ n : ℕ+, A n ≤ A 19 := by sorry

end NUMINAMATH_CALUDE_A_maximized_at_19_l4179_417969


namespace NUMINAMATH_CALUDE_percentage_subtraction_l4179_417959

theorem percentage_subtraction (a : ℝ) (p : ℝ) (h : a - p * a = 0.94 * a) : p = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_percentage_subtraction_l4179_417959


namespace NUMINAMATH_CALUDE_power_subtraction_l4179_417948

theorem power_subtraction : (2 : ℕ) ^ 4 - (2 : ℕ) ^ 3 = (2 : ℕ) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_subtraction_l4179_417948


namespace NUMINAMATH_CALUDE_lime_bottom_implies_magenta_top_l4179_417901

-- Define the colors
inductive Color
| Purple
| Cyan
| Magenta
| Lime
| Silver
| Black

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  top : Face
  bottom : Face
  front : Face
  back : Face
  left : Face
  right : Face

-- Define the property that all faces have different colors
def has_unique_colors (c : Cube) : Prop :=
  c.top.color ≠ c.bottom.color ∧
  c.top.color ≠ c.front.color ∧
  c.top.color ≠ c.back.color ∧
  c.top.color ≠ c.left.color ∧
  c.top.color ≠ c.right.color ∧
  c.bottom.color ≠ c.front.color ∧
  c.bottom.color ≠ c.back.color ∧
  c.bottom.color ≠ c.left.color ∧
  c.bottom.color ≠ c.right.color ∧
  c.front.color ≠ c.back.color ∧
  c.front.color ≠ c.left.color ∧
  c.front.color ≠ c.right.color ∧
  c.back.color ≠ c.left.color ∧
  c.back.color ≠ c.right.color ∧
  c.left.color ≠ c.right.color

-- Theorem statement
theorem lime_bottom_implies_magenta_top (c : Cube) 
  (h1 : has_unique_colors c) 
  (h2 : c.bottom.color = Color.Lime) : 
  c.top.color = Color.Magenta :=
sorry

end NUMINAMATH_CALUDE_lime_bottom_implies_magenta_top_l4179_417901


namespace NUMINAMATH_CALUDE_red_balls_count_l4179_417913

theorem red_balls_count (total : ℕ) (p_black : ℚ) (p_at_least_one_white : ℚ) :
  total = 10 ∧ 
  p_black = 2/5 ∧ 
  p_at_least_one_white = 7/9 →
  ∃ (black white red : ℕ), 
    black + white + red = total ∧
    black = 4 ∧
    red = 1 ∧
    (black : ℚ) / total = p_black ∧
    1 - (Nat.choose (black + red) 2 : ℚ) / (Nat.choose total 2) = p_at_least_one_white :=
by
  sorry

#check red_balls_count

end NUMINAMATH_CALUDE_red_balls_count_l4179_417913


namespace NUMINAMATH_CALUDE_fraction_increase_l4179_417909

theorem fraction_increase (a b : ℝ) (h : 3 * a - 4 * b ≠ 0) :
  (2 * (3 * a) * (3 * b)) / (3 * (3 * a) - 4 * (3 * b)) = 3 * ((2 * a * b) / (3 * a - 4 * b)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_increase_l4179_417909


namespace NUMINAMATH_CALUDE_problem_solution_l4179_417957

def P : Set ℝ := {x | -2 ≤ x ∧ x ≤ 10}
def Q (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem problem_solution :
  (∀ x, x ∉ P ↔ (x < -2 ∨ x > 10)) ∧
  (∀ m, P ⊆ Q m ↔ m ≥ 9) ∧
  (∀ m, P ∩ Q m = Q m ↔ m ≤ 9) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l4179_417957


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_prime_square_difference_l4179_417911

theorem largest_prime_divisor_of_prime_square_difference (m n : ℕ) 
  (hm : Prime m) (hn : Prime n) (hmn : m ≠ n) :
  (∃ (p : ℕ) (hp : Prime p), p ∣ (m^2 - n^2)) ∧
  (∀ (q : ℕ) (hq : Prime q), q ∣ (m^2 - n^2) → q ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_prime_square_difference_l4179_417911


namespace NUMINAMATH_CALUDE_quadratic_inequality_relationship_l4179_417978

theorem quadratic_inequality_relationship (x : ℝ) :
  (∀ x, x^2 - 2*x < 0 → 0 < x ∧ x < 4) ∧
  (∃ x, 0 < x ∧ x < 4 ∧ ¬(x^2 - 2*x < 0)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_relationship_l4179_417978


namespace NUMINAMATH_CALUDE_cube_points_form_octahedron_l4179_417905

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A cube with edge length a -/
structure Cube (a : ℝ) where
  vertices : Fin 8 → Point3D

/-- An octahedron -/
structure Octahedron where
  vertices : Fin 6 → Point3D

/-- Function to select points on the edges of a cube -/
def selectPointsOnCube (c : Cube a) : Fin 6 → Point3D :=
  sorry

/-- Function to check if points form an octahedron -/
def isOctahedron (points : Fin 6 → Point3D) : Prop :=
  sorry

/-- Theorem stating that it's possible to select 6 points on the edges of a cube
    such that they form the vertices of an octahedron -/
theorem cube_points_form_octahedron (a : ℝ) (h : a > 0) :
  ∃ (c : Cube a), isOctahedron (selectPointsOnCube c) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_points_form_octahedron_l4179_417905


namespace NUMINAMATH_CALUDE_perpendicular_lines_l4179_417950

/-- Two lines are perpendicular if the sum of the products of their corresponding coefficients is zero -/
def perpendicular (a b c e f g : ℝ) : Prop := a * e + b * f = 0

/-- The line equation x + (m^2 - m)y = 4m - 1 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := x + (m^2 - m) * y = 4 * m - 1

/-- The line equation 2x - y - 5 = 0 -/
def line2 (x y : ℝ) : Prop := 2 * x - y - 5 = 0

theorem perpendicular_lines (m : ℝ) : 
  perpendicular 1 (m^2 - m) (1 - 4*m) 2 (-1) (-5) → m = -1 ∨ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l4179_417950


namespace NUMINAMATH_CALUDE_root_sum_of_quadratic_l4179_417999

theorem root_sum_of_quadratic : ∃ (C D : ℝ), 
  (∀ x : ℝ, 3 * x^2 - 9 * x + 6 = 0 ↔ (x = C ∨ x = D)) ∧ 
  C + D = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_of_quadratic_l4179_417999


namespace NUMINAMATH_CALUDE_taehyung_current_age_l4179_417961

/-- Taehyung's age this year -/
def taehyung_age : ℕ := 9

/-- Taehyung's uncle's age this year -/
def uncle_age : ℕ := taehyung_age + 17

/-- The sum of Taehyung's and his uncle's ages four years later -/
def sum_ages_later : ℕ := (taehyung_age + 4) + (uncle_age + 4)

theorem taehyung_current_age :
  taehyung_age = 9 ∧ uncle_age = taehyung_age + 17 ∧ sum_ages_later = 43 :=
by sorry

end NUMINAMATH_CALUDE_taehyung_current_age_l4179_417961


namespace NUMINAMATH_CALUDE_total_cash_realized_eq_17364_82065_l4179_417936

/-- Calculates the total cash realized in INR from selling four stocks -/
def total_cash_realized (stock1_value stock1_brokerage stock2_value stock2_brokerage : ℚ)
                        (stock3_value stock3_brokerage stock4_value stock4_brokerage : ℚ)
                        (usd_to_inr : ℚ) : ℚ :=
  let stock1_realized := stock1_value * (1 - stock1_brokerage / 100)
  let stock2_realized := stock2_value * (1 - stock2_brokerage / 100)
  let stock3_realized := stock3_value * (1 - stock3_brokerage / 100) * usd_to_inr
  let stock4_realized := stock4_value * (1 - stock4_brokerage / 100) * usd_to_inr
  stock1_realized + stock2_realized + stock3_realized + stock4_realized

/-- Theorem stating that the total cash realized is equal to 17364.82065 INR -/
theorem total_cash_realized_eq_17364_82065 :
  total_cash_realized 120.50 (1/4) 210.75 0.5 80.90 0.3 150.55 0.65 74 = 17364.82065 := by
  sorry

end NUMINAMATH_CALUDE_total_cash_realized_eq_17364_82065_l4179_417936


namespace NUMINAMATH_CALUDE_multiple_problem_l4179_417902

theorem multiple_problem (m : ℝ) : 38 + m * 43 = 124 ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_multiple_problem_l4179_417902


namespace NUMINAMATH_CALUDE_cos_four_theta_l4179_417951

theorem cos_four_theta (θ : Real) 
  (h : Real.exp (Real.log 2 * (-2 + 3 * Real.cos θ)) + 1 = Real.exp (Real.log 2 * (1/2 + 2 * Real.cos θ))) : 
  Real.cos (4 * θ) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_cos_four_theta_l4179_417951


namespace NUMINAMATH_CALUDE_A_union_complement_B_equals_one_three_l4179_417938

-- Define the universal set U
def U : Set Nat := {1, 2, 3}

-- Define set A
def A : Set Nat := {1}

-- Define set B
def B : Set Nat := {1, 2}

-- Theorem statement
theorem A_union_complement_B_equals_one_three :
  A ∪ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_A_union_complement_B_equals_one_three_l4179_417938


namespace NUMINAMATH_CALUDE_red_lucky_stars_count_l4179_417939

theorem red_lucky_stars_count 
  (blue_count : ℕ) 
  (yellow_count : ℕ) 
  (red_count : ℕ) 
  (total_count : ℕ) 
  (pick_probability : ℚ) :
  blue_count = 20 →
  yellow_count = 15 →
  total_count = blue_count + yellow_count + red_count →
  pick_probability = 1/2 →
  (red_count : ℚ) / (total_count : ℚ) = pick_probability →
  red_count = 35 := by
sorry

end NUMINAMATH_CALUDE_red_lucky_stars_count_l4179_417939


namespace NUMINAMATH_CALUDE_phi_value_for_even_shifted_sine_l4179_417932

/-- Given a function f and a real number φ, this theorem states that
    if f(x) = 2sin(x + φ) where 0 < φ < π/2, and g(x) = f(x + π/3) is an even function,
    then φ = π/6 -/
theorem phi_value_for_even_shifted_sine
  (f : ℝ → ℝ)
  (φ : ℝ)
  (h1 : 0 < φ)
  (h2 : φ < π / 2)
  (h3 : ∀ x, f x = 2 * Real.sin (x + φ))
  (g : ℝ → ℝ)
  (h4 : ∀ x, g x = f (x + π / 3))
  (h5 : ∀ x, g x = g (-x)) :
  φ = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_phi_value_for_even_shifted_sine_l4179_417932


namespace NUMINAMATH_CALUDE_yoongis_calculation_l4179_417917

theorem yoongis_calculation (x : ℝ) : x / 9 = 30 → x - 37 = 233 := by
  sorry

end NUMINAMATH_CALUDE_yoongis_calculation_l4179_417917


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l4179_417923

/-- Given a line passing through the points (1,2) and (4,20), 
    prove that the sum of its slope and y-intercept is 8. -/
theorem line_slope_intercept_sum : ∀ (m b : ℝ),
  (∀ x y : ℝ, y = m * x + b ↔ (x = 1 ∧ y = 2) ∨ (x = 4 ∧ y = 20)) →
  m + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l4179_417923


namespace NUMINAMATH_CALUDE_tan_neg_five_pi_sixths_l4179_417968

theorem tan_neg_five_pi_sixths : Real.tan (-5 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_five_pi_sixths_l4179_417968


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4179_417991

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + 2| < a) → a > 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4179_417991


namespace NUMINAMATH_CALUDE_turnip_bag_options_l4179_417998

def bag_weights : List Nat := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (turnip_weight : Nat) : Prop :=
  turnip_weight ∈ bag_weights ∧
  ∃ (onion_weights carrots_weights : List Nat),
    onion_weights ++ carrots_weights ++ [turnip_weight] = bag_weights ∧
    onion_weights.sum * 2 = carrots_weights.sum

theorem turnip_bag_options :
  ∀ w ∈ bag_weights, is_valid_turnip_weight w ↔ w = 13 ∨ w = 16 := by sorry

end NUMINAMATH_CALUDE_turnip_bag_options_l4179_417998


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l4179_417965

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) : (a + 2*b)^2 - 4*b*(a + b) = a^2 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) (h3 : x ≠ 1) :
  ((x^2 - 2*x) / (x^2 - 4*x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l4179_417965


namespace NUMINAMATH_CALUDE_angle_measure_proof_l4179_417944

theorem angle_measure_proof (x : ℝ) : 
  (180 - x = 4 * x + 7) → x = 173 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l4179_417944


namespace NUMINAMATH_CALUDE_midpoint_trajectory_equation_l4179_417945

/-- The equation of the trajectory of the midpoint of the line connecting a fixed point to any point on a circle -/
theorem midpoint_trajectory_equation (P : ℝ × ℝ) (r : ℝ) :
  P = (4, -2) →
  r = 2 →
  ∀ (x y : ℝ), (∃ (x₁ y₁ : ℝ), x₁^2 + y₁^2 = r^2 ∧ x = (x₁ + P.1) / 2 ∧ y = (y₁ + P.2) / 2) →
  (x - 2)^2 + (y + 1)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_equation_l4179_417945


namespace NUMINAMATH_CALUDE_equal_intercepts_condition_l4179_417934

/-- A line with equation ax + y - 2 + a = 0 has equal intercepts on both coordinate axes if and only if a = 2 or a = 1 -/
theorem equal_intercepts_condition (a : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ 
    (∀ (x y : ℝ), a * x + y - 2 + a = 0 ↔ (x = k ∧ y = 0) ∨ (x = 0 ∧ y = k))) ↔ 
  (a = 2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_equal_intercepts_condition_l4179_417934


namespace NUMINAMATH_CALUDE_factor_x12_minus_729_l4179_417908

theorem factor_x12_minus_729 (x : ℝ) :
  x^12 - 729 = (x^3 + 3) * (x - Real.rpow 3 (1/3)) * 
               (x^2 + x * Real.rpow 3 (1/3) + Real.rpow 3 (2/3)) * 
               (x^12 + 9*x^6 + 81) :=
by sorry

end NUMINAMATH_CALUDE_factor_x12_minus_729_l4179_417908


namespace NUMINAMATH_CALUDE_inequality_proof_l4179_417966

theorem inequality_proof (x₁ x₂ : ℝ) (h₁ : |x₁| ≤ 1) (h₂ : |x₂| ≤ 1) :
  Real.sqrt (1 - x₁^2) + Real.sqrt (1 - x₂^2) ≤ 2 * Real.sqrt (1 - ((x₁ + x₂) / 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4179_417966


namespace NUMINAMATH_CALUDE_bench_cost_l4179_417984

theorem bench_cost (bench_cost table_cost : ℕ) : 
  bench_cost + table_cost = 750 → 
  table_cost = 2 * bench_cost →
  bench_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_bench_cost_l4179_417984


namespace NUMINAMATH_CALUDE_gcd_factorial_8_factorial_6_squared_l4179_417933

theorem gcd_factorial_8_factorial_6_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_8_factorial_6_squared_l4179_417933


namespace NUMINAMATH_CALUDE_complex_number_solution_l4179_417918

theorem complex_number_solution (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 1 - I) 
  (h₂ : z₁ * z₂ = 1 + I) : 
  z₂ = I := by sorry

end NUMINAMATH_CALUDE_complex_number_solution_l4179_417918


namespace NUMINAMATH_CALUDE_no_valid_tetrahedron_labeling_l4179_417958

/-- Represents a labeling of a tetrahedron's vertices -/
def TetrahedronLabeling := Fin 4 → Fin 4

/-- Checks if a labeling uses each number exactly once -/
def is_valid_labeling (l : TetrahedronLabeling) : Prop :=
  ∀ i : Fin 4, ∃! j : Fin 4, l j = i

/-- Calculates the sum of labels on a face -/
def face_sum (l : TetrahedronLabeling) (face : Fin 4 → Fin 3) : ℕ :=
  (face 0).val + (face 1).val + (face 2).val

/-- Checks if all face sums are equal -/
def all_face_sums_equal (l : TetrahedronLabeling) (faces : Fin 4 → (Fin 4 → Fin 3)) : Prop :=
  ∀ i j : Fin 4, face_sum l (faces i) = face_sum l (faces j)

/-- The main theorem stating that no valid labeling exists -/
theorem no_valid_tetrahedron_labeling (faces : Fin 4 → (Fin 4 → Fin 3)) :
  ¬∃ l : TetrahedronLabeling, is_valid_labeling l ∧ all_face_sums_equal l faces :=
sorry

end NUMINAMATH_CALUDE_no_valid_tetrahedron_labeling_l4179_417958


namespace NUMINAMATH_CALUDE_difference_of_squares_divisible_by_18_l4179_417997

theorem difference_of_squares_divisible_by_18 (a b : ℤ) 
  (ha : Odd a) (hb : Odd b) : 
  ∃ k : ℤ, (3*a + 2)^2 - (3*b + 2)^2 = 18 * k := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_divisible_by_18_l4179_417997


namespace NUMINAMATH_CALUDE_lg_sqrt5_plus_half_lg20_equals_1_l4179_417995

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_sqrt5_plus_half_lg20_equals_1 : lg (Real.sqrt 5) + (1/2) * lg 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_sqrt5_plus_half_lg20_equals_1_l4179_417995


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l4179_417931

theorem number_puzzle_solution : ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l4179_417931
