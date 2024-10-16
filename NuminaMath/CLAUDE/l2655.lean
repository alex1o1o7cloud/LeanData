import Mathlib

namespace NUMINAMATH_CALUDE_girls_not_adjacent_arrangements_l2655_265547

theorem girls_not_adjacent_arrangements :
  let num_boys : ℕ := 4
  let num_girls : ℕ := 4
  let total_people : ℕ := num_boys + num_girls
  let num_spaces : ℕ := num_boys + 1
  
  (num_boys.factorial * num_spaces.factorial) = 2880 :=
by sorry

end NUMINAMATH_CALUDE_girls_not_adjacent_arrangements_l2655_265547


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2655_265516

/-- The polynomial x^4 - 3x^3 + mx + n -/
def f (m n x : ℂ) : ℂ := x^4 - 3*x^3 + m*x + n

/-- The polynomial x^2 - 2x + 4 -/
def g (x : ℂ) : ℂ := x^2 - 2*x + 4

theorem polynomial_divisibility (m n : ℂ) :
  (∀ x, g x = 0 → f m n x = 0) →
  g (1 + Complex.I * Real.sqrt 3) = 0 →
  g (1 - Complex.I * Real.sqrt 3) = 0 →
  m = 8 ∧ n = -24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2655_265516


namespace NUMINAMATH_CALUDE_first_digit_base_nine_of_2121122_base_three_l2655_265549

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def first_digit_base_nine (n : Nat) : Nat :=
  Nat.log 9 n

theorem first_digit_base_nine_of_2121122_base_three :
  let y : Nat := base_three_to_decimal [2, 2, 1, 1, 2, 1, 2]
  first_digit_base_nine y = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_base_nine_of_2121122_base_three_l2655_265549


namespace NUMINAMATH_CALUDE_product_of_logarithms_l2655_265568

theorem product_of_logarithms (c d : ℕ) (hc : c > 0) (hd : d > 0) :
  (Real.log d / Real.log c = 2) → (d - c = 630) → (c + d = 1260) := by
  sorry

end NUMINAMATH_CALUDE_product_of_logarithms_l2655_265568


namespace NUMINAMATH_CALUDE_candy_sampling_percentage_l2655_265545

theorem candy_sampling_percentage : 
  ∀ (total_customers : ℝ) (caught_percent : ℝ) (not_caught_percent : ℝ),
  caught_percent = 22 →
  not_caught_percent = 12 →
  ∃ (total_sampling_percent : ℝ),
    total_sampling_percent = caught_percent + (not_caught_percent / 100) * total_sampling_percent ∧
    total_sampling_percent = 25 := by
  sorry

end NUMINAMATH_CALUDE_candy_sampling_percentage_l2655_265545


namespace NUMINAMATH_CALUDE_circle_radius_is_sqrt_21_25_l2655_265571

-- Define the circle Ω
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points P and Q
def P : ℝ × ℝ := (9, 17)
def Q : ℝ × ℝ := (18, 15)

-- Define the line y = 2
def line_y_2 (x : ℝ) : ℝ := 2

-- Theorem statement
theorem circle_radius_is_sqrt_21_25 (Ω : Circle) :
  P ∈ {p : ℝ × ℝ | (p.1 - Ω.center.1)^2 + (p.2 - Ω.center.2)^2 = Ω.radius^2} →
  Q ∈ {p : ℝ × ℝ | (p.1 - Ω.center.1)^2 + (p.2 - Ω.center.2)^2 = Ω.radius^2} →
  (∃ x : ℝ, (x, line_y_2 x) ∈ {p : ℝ × ℝ | ∃ t : ℝ, p = (P.1 + t * (P.2 - Ω.center.2), P.2 - t * (P.1 - Ω.center.1))} ∩
                               {p : ℝ × ℝ | ∃ t : ℝ, p = (Q.1 + t * (Q.2 - Ω.center.2), Q.2 - t * (Q.1 - Ω.center.1))}) →
  Ω.radius = Real.sqrt 21.25 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_sqrt_21_25_l2655_265571


namespace NUMINAMATH_CALUDE_greater_number_proof_l2655_265505

theorem greater_number_proof (x y : ℝ) (h1 : x > y) (h2 : x > 0) (h3 : y > 0)
  (h4 : x * y = 2688) (h5 : (x + y) - (x - y) = 64) : x = 84 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l2655_265505


namespace NUMINAMATH_CALUDE_divisible_by_six_sum_powers_divisible_by_seven_l2655_265565

-- Part (a)
theorem divisible_by_six (n : ℤ) : 6 ∣ (n * (n + 1) * (n + 2)) := by
  sorry

-- Part (b)
theorem sum_powers_divisible_by_seven :
  7 ∣ (1^2015 + 2^2015 + 3^2015 + 4^2015 + 5^2015 + 6^2015) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_sum_powers_divisible_by_seven_l2655_265565


namespace NUMINAMATH_CALUDE_correct_guess_probability_l2655_265537

/-- The number of possible choices for the last digit -/
def last_digit_choices : ℕ := 4

/-- The number of possible choices for the second-to-last digit -/
def second_last_digit_choices : ℕ := 3

/-- The probability of correctly guessing the two-digit code -/
def guess_probability : ℚ := 1 / (last_digit_choices * second_last_digit_choices)

theorem correct_guess_probability : guess_probability = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_correct_guess_probability_l2655_265537


namespace NUMINAMATH_CALUDE_negative_sixty_four_two_thirds_power_l2655_265522

theorem negative_sixty_four_two_thirds_power : (-64 : ℝ) ^ (2/3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_two_thirds_power_l2655_265522


namespace NUMINAMATH_CALUDE_sara_wrapping_paper_l2655_265599

theorem sara_wrapping_paper (total_paper : ℚ) (num_presents : ℕ) (paper_per_present : ℚ) :
  total_paper = 1/2 →
  num_presents = 5 →
  total_paper = num_presents * paper_per_present →
  paper_per_present = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_sara_wrapping_paper_l2655_265599


namespace NUMINAMATH_CALUDE_min_coins_for_distribution_l2655_265578

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (friends * (friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_coins_for_distribution (friends : ℕ) (initial_coins : ℕ) 
  (h1 : friends = 15) (h2 : initial_coins = 70) :
  min_additional_coins friends initial_coins = 50 := by
  sorry

#eval min_additional_coins 15 70

end NUMINAMATH_CALUDE_min_coins_for_distribution_l2655_265578


namespace NUMINAMATH_CALUDE_net_salary_average_is_correct_l2655_265567

/-- Represents Sharon's salary structure and performance scenarios -/
structure SalaryStructure where
  initial_salary : ℝ
  exceptional_increase : ℝ
  good_increase : ℝ
  average_increase : ℝ
  exceptional_bonus : ℝ
  good_bonus : ℝ
  federal_tax : ℝ
  state_tax : ℝ
  healthcare_deduction : ℝ

/-- Calculates the net salary for average performance -/
def net_salary_average (s : SalaryStructure) : ℝ :=
  let increased_salary := s.initial_salary * (1 + s.average_increase)
  let tax_deduction := increased_salary * (s.federal_tax + s.state_tax)
  increased_salary - tax_deduction - s.healthcare_deduction

/-- Theorem stating that the net salary for average performance is 497.40 -/
theorem net_salary_average_is_correct (s : SalaryStructure) 
    (h1 : s.initial_salary = 560)
    (h2 : s.average_increase = 0.15)
    (h3 : s.federal_tax = 0.10)
    (h4 : s.state_tax = 0.05)
    (h5 : s.healthcare_deduction = 50) :
    net_salary_average s = 497.40 := by
  sorry

#eval net_salary_average { 
  initial_salary := 560,
  exceptional_increase := 0.25,
  good_increase := 0.20,
  average_increase := 0.15,
  exceptional_bonus := 0.05,
  good_bonus := 0.03,
  federal_tax := 0.10,
  state_tax := 0.05,
  healthcare_deduction := 50
}

end NUMINAMATH_CALUDE_net_salary_average_is_correct_l2655_265567


namespace NUMINAMATH_CALUDE_topology_check_l2655_265556

-- Define the set X
def X : Set Char := {'{', 'a', 'b', 'c', '}'}

-- Define the four sets v
def v1 : Set (Set Char) := {∅, {'a'}, {'c'}, {'a', 'b', 'c'}}
def v2 : Set (Set Char) := {∅, {'b'}, {'c'}, {'b', 'c'}, {'a', 'b', 'c'}}
def v3 : Set (Set Char) := {∅, {'a'}, {'a', 'b'}, {'a', 'c'}}
def v4 : Set (Set Char) := {∅, {'a', 'c'}, {'b', 'c'}, {'c'}, {'a', 'b', 'c'}}

-- Define the topology property
def is_topology (v : Set (Set Char)) : Prop :=
  X ∈ v ∧ ∅ ∈ v ∧
  (∀ (S : Set (Set Char)), S ⊆ v → ⋃₀ S ∈ v) ∧
  (∀ (S : Set (Set Char)), S ⊆ v → ⋂₀ S ∈ v)

-- Theorem statement
theorem topology_check :
  is_topology v2 ∧ is_topology v4 ∧ ¬is_topology v1 ∧ ¬is_topology v3 :=
sorry

end NUMINAMATH_CALUDE_topology_check_l2655_265556


namespace NUMINAMATH_CALUDE_min_cut_length_40x30_paper_10x5_rect_l2655_265550

/-- Represents a rectangular piece of paper -/
structure Paper where
  width : ℕ
  height : ℕ

/-- Represents a rectangle to be cut out -/
structure CutRectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum cut length required to extract a rectangle from a paper -/
def minCutLength (paper : Paper) (rect : CutRectangle) : ℕ :=
  sorry

/-- Theorem stating the minimum cut length for the given problem -/
theorem min_cut_length_40x30_paper_10x5_rect :
  let paper := Paper.mk 40 30
  let rect := CutRectangle.mk 10 5
  minCutLength paper rect = 40 := by sorry

end NUMINAMATH_CALUDE_min_cut_length_40x30_paper_10x5_rect_l2655_265550


namespace NUMINAMATH_CALUDE_floor_times_self_eq_72_l2655_265548

theorem floor_times_self_eq_72 (x : ℝ) :
  x > 0 ∧ ⌊x⌋ * x = 72 → x = 9 := by sorry

end NUMINAMATH_CALUDE_floor_times_self_eq_72_l2655_265548


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2655_265585

theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of S_n
  (∀ n m, a n - a m = (n - m) * (a 2 - a 1)) →  -- Definition of arithmetic sequence
  a 8 / a 7 = 13 / 5 →  -- Given condition
  S 15 / S 13 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2655_265585


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l2655_265525

theorem inscribed_cube_volume (large_cube_edge : ℝ) (small_cube_edge : ℝ) (small_cube_volume : ℝ) : 
  large_cube_edge = 12 →
  small_cube_edge * Real.sqrt 3 = large_cube_edge →
  small_cube_volume = small_cube_edge ^ 3 →
  small_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l2655_265525


namespace NUMINAMATH_CALUDE_freshman_groups_l2655_265555

theorem freshman_groups (total_freshmen : Nat) (group_decrease : Nat) :
  total_freshmen = 2376 →
  group_decrease = 9 →
  ∃ (initial_groups final_groups : Nat),
    initial_groups = final_groups + group_decrease ∧
    total_freshmen % initial_groups = 0 ∧
    total_freshmen % final_groups = 0 ∧
    total_freshmen / final_groups < 30 ∧
    final_groups = 99 := by
  sorry

end NUMINAMATH_CALUDE_freshman_groups_l2655_265555


namespace NUMINAMATH_CALUDE_product_fourth_minus_seven_l2655_265512

theorem product_fourth_minus_seven (a b c d : ℕ) (h₁ : a = 5) (h₂ : b = 9) (h₃ : c = 4) (h₄ : d = 7) :
  (a * b * c : ℚ) / 4 - d = 38 := by
  sorry

end NUMINAMATH_CALUDE_product_fourth_minus_seven_l2655_265512


namespace NUMINAMATH_CALUDE_inequality_proof_l2655_265575

/-- The function f(x) defined as |x-a| + |x-3| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 3|

/-- Theorem: Given the conditions, prove that m + 2n ≥ 2 -/
theorem inequality_proof (a m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_solution_set : Set.Icc 1 3 = {x | f a x ≤ 1 + |x - 3|})
  (h_a : 1/m + 1/(2*n) = a) : m + 2*n ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2655_265575


namespace NUMINAMATH_CALUDE_average_age_of_ten_students_l2655_265594

theorem average_age_of_ten_students
  (total_students : ℕ)
  (average_age_all : ℚ)
  (num_group1 : ℕ)
  (average_age_group1 : ℚ)
  (age_last_student : ℕ)
  (h1 : total_students = 25)
  (h2 : average_age_all = 25)
  (h3 : num_group1 = 14)
  (h4 : average_age_group1 = 28)
  (h5 : age_last_student = 13)
  : ∃ (average_age_group2 : ℚ),
    average_age_group2 = 22 ∧
    average_age_group2 * (total_students - num_group1 - 1) =
      total_students * average_age_all - num_group1 * average_age_group1 - age_last_student :=
by sorry

end NUMINAMATH_CALUDE_average_age_of_ten_students_l2655_265594


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2655_265598

/-- A convex decagon is a polygon with 10 sides -/
def ConvexDecagon : Type := Unit

/-- Number of sides in a convex decagon -/
def numSides : ℕ := 10

/-- Number of right angles in the given decagon -/
def numRightAngles : ℕ := 3

/-- The number of diagonals in a polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem decagon_diagonals (d : ConvexDecagon) : 
  numDiagonals numSides = 35 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2655_265598


namespace NUMINAMATH_CALUDE_sams_correct_percentage_l2655_265520

theorem sams_correct_percentage (y : ℕ) : 
  let total_problems := 8 * y
  let missed_problems := 3 * y
  let correct_problems := total_problems - missed_problems
  (correct_problems : ℚ) / (total_problems : ℚ) * 100 = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_sams_correct_percentage_l2655_265520


namespace NUMINAMATH_CALUDE_age_difference_approximation_l2655_265529

-- Define the age ratios and total age sum
def patrick_michael_monica_ratio : Rat := 3 / 5
def michael_monica_nola_ratio : Rat × Rat := (3 / 5, 5 / 7)
def monica_nola_olivia_ratio : Rat × Rat := (4 / 3, 3 / 2)
def total_age_sum : ℕ := 146

-- Define a function to calculate the age difference
def age_difference (patrick_michael_monica_ratio : Rat) 
                   (michael_monica_nola_ratio : Rat × Rat)
                   (monica_nola_olivia_ratio : Rat × Rat)
                   (total_age_sum : ℕ) : ℝ :=
  sorry

-- Theorem statement
theorem age_difference_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |age_difference patrick_michael_monica_ratio 
                  michael_monica_nola_ratio
                  monica_nola_olivia_ratio
                  total_age_sum - 6.412| < ε :=
sorry

end NUMINAMATH_CALUDE_age_difference_approximation_l2655_265529


namespace NUMINAMATH_CALUDE_at_least_one_positive_l2655_265542

theorem at_least_one_positive (x y z : ℝ) : 
  let a := x^2 - 2*y + π/2
  let b := y^2 - 2*z + π/4
  let c := z^2 - 2*x + π/4
  max a (max b c) > 0 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_positive_l2655_265542


namespace NUMINAMATH_CALUDE_fans_with_all_items_l2655_265596

/-- The maximum capacity of the stadium --/
def stadium_capacity : ℕ := 3000

/-- The interval at which t-shirts are given --/
def tshirt_interval : ℕ := 50

/-- The interval at which caps are given --/
def cap_interval : ℕ := 25

/-- The interval at which wristbands are given --/
def wristband_interval : ℕ := 60

/-- Theorem stating that the number of fans receiving all three items is 10 --/
theorem fans_with_all_items : 
  (stadium_capacity / (Nat.lcm tshirt_interval (Nat.lcm cap_interval wristband_interval))) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l2655_265596


namespace NUMINAMATH_CALUDE_apple_cost_l2655_265580

theorem apple_cost (initial_cost : ℝ) (initial_dozen : ℕ) (target_dozen : ℕ) : 
  initial_cost * (target_dozen / initial_dozen) = 54.60 :=
by
  sorry

#check apple_cost 39.00 5 7

end NUMINAMATH_CALUDE_apple_cost_l2655_265580


namespace NUMINAMATH_CALUDE_inequality_proof_l2655_265536

theorem inequality_proof (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_sum : 1/x + 1/y + 1/z = 2) : 8 * (x - 1) * (y - 1) * (z - 1) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2655_265536


namespace NUMINAMATH_CALUDE_negative_integer_solution_l2655_265500

theorem negative_integer_solution : ∃! (N : ℤ), N < 0 ∧ N + 2 * N^2 = 12 ∧ N = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_solution_l2655_265500


namespace NUMINAMATH_CALUDE_volume_formula_l2655_265591

/-- A pyramid with a rectangular base -/
structure Pyramid where
  /-- Length of side AB of the base -/
  ab : ℝ
  /-- Length of side AD of the base -/
  ad : ℝ
  /-- Angle AQB where Q is the apex -/
  θ : ℝ
  /-- Assertion that AB = 2 -/
  hab : ab = 2
  /-- Assertion that AD = 1 -/
  had : ad = 1
  /-- Assertion that Q is directly above the center of the base -/
  hcenter : True
  /-- Assertion that Q is equidistant from all vertices -/
  hequidistant : True

/-- The volume of the pyramid -/
noncomputable def volume (p : Pyramid) : ℝ :=
  (2/3) * Real.sqrt (Real.tan p.θ ^ 2 + 1/4)

/-- Theorem stating that the volume formula is correct -/
theorem volume_formula (p : Pyramid) : volume p = (2/3) * Real.sqrt (Real.tan p.θ ^ 2 + 1/4) := by
  sorry

end NUMINAMATH_CALUDE_volume_formula_l2655_265591


namespace NUMINAMATH_CALUDE_share_division_l2655_265519

theorem share_division (total : ℝ) (a b c : ℝ) 
  (h_total : total = 400)
  (h_sum : a + b + c = total)
  (h_a : a = (2/3) * (b + c))
  (h_b : b = (6/9) * (a + c)) :
  a = 160 := by sorry

end NUMINAMATH_CALUDE_share_division_l2655_265519


namespace NUMINAMATH_CALUDE_count_non_negative_rationals_l2655_265543

def rational_list : List ℚ := [-8, 0, -1.04, -(-3), 1/3, -|-2|]

theorem count_non_negative_rationals :
  (rational_list.filter (λ x => x ≥ 0)).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_non_negative_rationals_l2655_265543


namespace NUMINAMATH_CALUDE_geralds_apples_count_l2655_265572

/-- Given that Pam has 10 bags of apples, 1200 apples in total, and each of her bags
    contains 3 times the number of apples in each of Gerald's bags, prove that
    each of Gerald's bags contains 40 apples. -/
theorem geralds_apples_count (pam_bags : ℕ) (pam_total_apples : ℕ) (gerald_apples : ℕ) 
  (h1 : pam_bags = 10)
  (h2 : pam_total_apples = 1200)
  (h3 : pam_total_apples = pam_bags * (3 * gerald_apples)) :
  gerald_apples = 40 := by
  sorry

end NUMINAMATH_CALUDE_geralds_apples_count_l2655_265572


namespace NUMINAMATH_CALUDE_smaller_square_area_half_larger_l2655_265552

/-- A circle with an inscribed square and a smaller square -/
structure SquaresInCircle where
  /-- The radius of the circle -/
  R : ℝ
  /-- The side length of the larger square -/
  a : ℝ
  /-- The side length of the smaller square -/
  b : ℝ
  /-- The larger square is inscribed in the circle -/
  h1 : R = a * Real.sqrt 2 / 2
  /-- The smaller square has one side coinciding with a side of the larger square -/
  h2 : b ≤ a
  /-- The smaller square has two vertices on the circle -/
  h3 : R^2 = (a/2 - b/2)^2 + b^2
  /-- The side length of the larger square is 4 units -/
  h4 : a = 4

/-- The area of the smaller square is half the area of the larger square -/
theorem smaller_square_area_half_larger (sq : SquaresInCircle) : 
  sq.b^2 = sq.a^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_square_area_half_larger_l2655_265552


namespace NUMINAMATH_CALUDE_equation_solution_l2655_265589

theorem equation_solution :
  ∃ x : ℚ, (x - 30) / 3 = (5 - 3 * x) / 4 ∧ x = 135 / 13 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2655_265589


namespace NUMINAMATH_CALUDE_tunnel_safety_condition_l2655_265587

def height_limit : ℝ := 4.5

def can_pass_safely (h : ℝ) : Prop := h ≤ height_limit

theorem tunnel_safety_condition (h : ℝ) :
  can_pass_safely h ↔ h ≤ height_limit :=
sorry

end NUMINAMATH_CALUDE_tunnel_safety_condition_l2655_265587


namespace NUMINAMATH_CALUDE_min_area_of_B_l2655_265546

-- Define set A
def A : Set (ℝ × ℝ) := {p | |p.1 - 2| + |p.2 - 3| ≤ 1}

-- Define set B
def B (D E F : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + D * p.1 + E * p.2 + F ≤ 0}

-- State the theorem
theorem min_area_of_B (D E F : ℝ) (h1 : D^2 + E^2 - 4*F > 0) (h2 : A ⊆ B D E F) :
  ∃ (S : ℝ), S = 2 * Real.pi ∧ ∀ (S' : ℝ), (∃ (D' E' F' : ℝ), D'^2 + E'^2 - 4*F' > 0 ∧ A ⊆ B D' E' F' ∧ S' = Real.pi * ((D'^2 + E'^2) / 4 - F')) → S ≤ S' :=
sorry

end NUMINAMATH_CALUDE_min_area_of_B_l2655_265546


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2655_265544

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  second_quadrant (-5 : ℝ) (4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2655_265544


namespace NUMINAMATH_CALUDE_chalk_inventory_theorem_l2655_265539

-- Define the types of chalk
inductive ChalkType
  | Regular
  | Unusual
  | Excellent

-- Define the store's chalk inventory
structure ChalkInventory where
  regular : ℕ
  unusual : ℕ
  excellent : ℕ

def initial_ratio : Fin 3 → ℕ
  | 0 => 3  -- Regular
  | 1 => 4  -- Unusual
  | 2 => 6  -- Excellent

def new_ratio : Fin 3 → ℕ
  | 0 => 2  -- Regular
  | 1 => 5  -- Unusual
  | 2 => 8  -- Excellent

theorem chalk_inventory_theorem (initial : ChalkInventory) (final : ChalkInventory) :
  -- Initial ratio condition
  initial.regular * initial_ratio 1 = initial.unusual * initial_ratio 0 ∧
  initial.regular * initial_ratio 2 = initial.excellent * initial_ratio 0 ∧
  -- New ratio condition
  final.regular * new_ratio 1 = final.unusual * new_ratio 0 ∧
  final.regular * new_ratio 2 = final.excellent * new_ratio 0 ∧
  -- Excellent chalk increase condition
  final.excellent = initial.excellent * 180 / 100 ∧
  -- Regular chalk decrease condition
  initial.regular - final.regular ≤ 10 ∧
  -- Total initial packs
  initial.regular + initial.unusual + initial.excellent = 390 :=
by sorry

end NUMINAMATH_CALUDE_chalk_inventory_theorem_l2655_265539


namespace NUMINAMATH_CALUDE_first_complete_coverage_l2655_265570

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Function to check if all remainders modulo 12 have been covered -/
def allRemaindersCovered (n : ℕ) : Prop :=
  ∀ r : Fin 12, ∃ k ≤ n, triangular k % 12 = r

/-- The main theorem -/
theorem first_complete_coverage :
  (allRemaindersCovered 19 ∧ ∀ m < 19, ¬allRemaindersCovered m) :=
sorry

end NUMINAMATH_CALUDE_first_complete_coverage_l2655_265570


namespace NUMINAMATH_CALUDE_jack_flyers_l2655_265503

theorem jack_flyers (total : ℕ) (rose : ℕ) (left : ℕ) (h1 : total = 1236) (h2 : rose = 320) (h3 : left = 796) :
  total - (rose + left) = 120 := by
  sorry

end NUMINAMATH_CALUDE_jack_flyers_l2655_265503


namespace NUMINAMATH_CALUDE_eggs_produced_this_year_l2655_265502

/-- Calculates the total egg production for this year given last year's production and additional eggs produced. -/
def total_eggs_this_year (last_year_production additional_eggs : ℕ) : ℕ :=
  last_year_production + additional_eggs

/-- Theorem stating that the total eggs produced this year is 4636. -/
theorem eggs_produced_this_year : 
  total_eggs_this_year 1416 3220 = 4636 := by
  sorry

end NUMINAMATH_CALUDE_eggs_produced_this_year_l2655_265502


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l2655_265513

theorem scooter_gain_percent (initial_cost repair1 repair2 repair3 selling_price : ℚ) : 
  initial_cost = 800 →
  repair1 = 150 →
  repair2 = 75 →
  repair3 = 225 →
  selling_price = 1600 →
  let total_cost := initial_cost + repair1 + repair2 + repair3
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  gain_percent = 28 := by
sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l2655_265513


namespace NUMINAMATH_CALUDE_right_trapezoid_area_l2655_265595

/-- The area of a right trapezoid with specific base proportions -/
theorem right_trapezoid_area : ∀ (lower_base : ℝ),
  lower_base > 0 →
  let upper_base := (3 / 5) * lower_base
  let height := (lower_base - upper_base) / 2
  (lower_base - 8 = height) →
  (1 / 2) * (lower_base + upper_base) * height = 192 := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_area_l2655_265595


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2655_265515

open Set
open Function

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the derivative of f
noncomputable def f' : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_of_inequality 
  (hf_domain : ∀ x, x ∈ (Set.Ioi 0) → DifferentiableAt ℝ f x)
  (hf'_def : ∀ x, x ∈ (Set.Ioi 0) → HasDerivAt f (f' x) x)
  (hf'_condition : ∀ x, x ∈ (Set.Ioi 0) → x * f' x > f x) :
  {x : ℝ | (x - 1) * f (x + 1) > f (x^2 - 1)} = Ioo 1 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2655_265515


namespace NUMINAMATH_CALUDE_mono_increasing_minus_decreasing_mono_decreasing_minus_increasing_l2655_265511

-- Define monotonically increasing function
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define monotonically decreasing function
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Theorem for proposition ②
theorem mono_increasing_minus_decreasing
  (f g : ℝ → ℝ) (hf : MonoIncreasing f) (hg : MonoDecreasing g) :
  MonoIncreasing (fun x ↦ f x - g x) :=
sorry

-- Theorem for proposition ③
theorem mono_decreasing_minus_increasing
  (f g : ℝ → ℝ) (hf : MonoDecreasing f) (hg : MonoIncreasing g) :
  MonoDecreasing (fun x ↦ f x - g x) :=
sorry

end NUMINAMATH_CALUDE_mono_increasing_minus_decreasing_mono_decreasing_minus_increasing_l2655_265511


namespace NUMINAMATH_CALUDE_complex_modulus_example_l2655_265527

theorem complex_modulus_example : Complex.abs (-3 + (9/4)*Complex.I) = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l2655_265527


namespace NUMINAMATH_CALUDE_same_prime_factors_imply_power_of_two_l2655_265561

theorem same_prime_factors_imply_power_of_two (b m n : ℕ) 
  (hb : b > 1) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hmn : m ≠ n) 
  (h_same_factors : ∀ p : ℕ, Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) : 
  ∃ k : ℕ, b + 1 = 2^k :=
sorry

end NUMINAMATH_CALUDE_same_prime_factors_imply_power_of_two_l2655_265561


namespace NUMINAMATH_CALUDE_quadratic_expression_l2655_265559

theorem quadratic_expression (m : ℤ) : 
  (∃ (a b c : ℤ), a * m^2 + b * m + c = (m - 8) * (m + 3)) → 
  (∃ (a b c : ℤ), a * m^2 + b * m + c = m^2 - 5*m - 24) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_l2655_265559


namespace NUMINAMATH_CALUDE_trig_expression_equality_l2655_265593

theorem trig_expression_equality : 
  (1 + Real.cos (20 * π / 180)) / (2 * Real.sin (20 * π / 180)) - 
  Real.sin (10 * π / 180) * ((1 / Real.tan (5 * π / 180)) - Real.tan (5 * π / 180)) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l2655_265593


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l2655_265563

theorem parabola_point_coordinates (x y : ℝ) :
  y^2 = 4*x →                             -- P is on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 100 →                 -- Distance from P to focus (1, 0) is 10
  (x = 9 ∧ (y = 6 ∨ y = -6)) :=           -- Coordinates of P are (9, ±6)
by
  sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l2655_265563


namespace NUMINAMATH_CALUDE_correct_negation_of_existential_statement_l2655_265573

theorem correct_negation_of_existential_statement :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_correct_negation_of_existential_statement_l2655_265573


namespace NUMINAMATH_CALUDE_university_students_count_l2655_265582

/-- Proves that given the initial ratio of male to female students and the ratio after increasing female students, the total number of students after the increase is 4800 -/
theorem university_students_count (x y : ℕ) : 
  x = 3 * y ∧ 
  9 * (y + 400) = 4 * x → 
  x + y + 400 = 4800 :=
by sorry

end NUMINAMATH_CALUDE_university_students_count_l2655_265582


namespace NUMINAMATH_CALUDE_calculation_proof_l2655_265576

theorem calculation_proof : 1 - (1/2)⁻¹ * Real.sin (π/3) + |2^0 - Real.sqrt 3| = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2655_265576


namespace NUMINAMATH_CALUDE_rotation_equivalence_l2655_265560

/-- 
Given:
- A point P is rotated 750 degrees clockwise about point Q, resulting in point R.
- The same point P is rotated y degrees counterclockwise about point Q, also resulting in point R.
- y < 360

Prove that y = 330.
-/
theorem rotation_equivalence (y : ℝ) (h1 : y < 360) : 
  (750 % 360 : ℝ) + y = 360 → y = 330 := by
  sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l2655_265560


namespace NUMINAMATH_CALUDE_divisible_by_24_l2655_265504

theorem divisible_by_24 (n : ℕ+) : ∃ k : ℤ, (n : ℤ)^4 + 2*(n : ℤ)^3 + 11*(n : ℤ)^2 + 10*(n : ℤ) = 24*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l2655_265504


namespace NUMINAMATH_CALUDE_eliminate_denominators_l2655_265535

theorem eliminate_denominators (x : ℝ) :
  (1 / 2 * (x + 1) = 1 - 1 / 3 * x) →
  (3 * (x + 1) = 6 - 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l2655_265535


namespace NUMINAMATH_CALUDE_umar_age_l2655_265509

/-- Given the ages of Ali, Yusaf, and Umar, prove Umar's age -/
theorem umar_age (ali_age yusaf_age umar_age : ℕ) : 
  ali_age = 8 →
  ali_age = yusaf_age + 3 →
  umar_age = 2 * yusaf_age →
  umar_age = 10 := by
sorry

end NUMINAMATH_CALUDE_umar_age_l2655_265509


namespace NUMINAMATH_CALUDE_cube_root_over_sixth_root_of_eight_l2655_265521

theorem cube_root_over_sixth_root_of_eight (x : ℝ) :
  (8 ^ (1/3)) / (8 ^ (1/6)) = 8 ^ (1/6) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_over_sixth_root_of_eight_l2655_265521


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l2655_265501

def digit_sum (n : ℕ) : ℕ := sorry

def is_prime (n : ℕ) : Prop := sorry

theorem smallest_prime_with_digit_sum_23 :
  (∀ p : ℕ, is_prime p ∧ digit_sum p = 23 → p ≥ 599) ∧
  is_prime 599 ∧
  digit_sum 599 = 23 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l2655_265501


namespace NUMINAMATH_CALUDE_radical_simplification_l2655_265524

theorem radical_simplification (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (40 * q) * Real.sqrt (20 * q) * Real.sqrt (10 * q) = 40 * q * Real.sqrt (5 * q) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l2655_265524


namespace NUMINAMATH_CALUDE_same_solution_implies_a_equals_four_l2655_265583

theorem same_solution_implies_a_equals_four (a : ℝ) : 
  (∃ x : ℝ, 2 * x + 1 = 3 ∧ 2 - (a - x) / 3 = 1) → a = 4 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_equals_four_l2655_265583


namespace NUMINAMATH_CALUDE_evaluate_expression_l2655_265533

theorem evaluate_expression : -(16 / 4 * 7 - 50 + 5 * 7) = -13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2655_265533


namespace NUMINAMATH_CALUDE_tomatoes_picked_today_l2655_265592

/-- Represents the number of tomatoes in various states --/
structure TomatoCount where
  initial : ℕ
  pickedYesterday : ℕ
  leftAfterYesterday : ℕ

/-- Theorem: The number of tomatoes picked today is equal to the initial number
    minus the number left after yesterday's picking --/
theorem tomatoes_picked_today (t : TomatoCount)
  (h1 : t.initial = 160)
  (h2 : t.pickedYesterday = 56)
  (h3 : t.leftAfterYesterday = 104)
  : t.initial - t.leftAfterYesterday = 56 := by
  sorry


end NUMINAMATH_CALUDE_tomatoes_picked_today_l2655_265592


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_l2655_265590

/-- Time taken for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (total_length : ℝ) 
  (h1 : train_length = 130) 
  (h2 : train_speed_kmh = 45) 
  (h3 : total_length = 245) : 
  (total_length / (train_speed_kmh * 1000 / 3600)) = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_time_l2655_265590


namespace NUMINAMATH_CALUDE_square_dancing_problem_l2655_265554

/-- The number of female students in the first class that satisfies the square dancing conditions --/
def female_students_in_first_class : ℕ := by sorry

theorem square_dancing_problem :
  let males_class1 : ℕ := 17
  let males_class2 : ℕ := 14
  let females_class2 : ℕ := 18
  let males_class3 : ℕ := 15
  let females_class3 : ℕ := 17
  let total_males : ℕ := males_class1 + males_class2 + males_class3
  let total_females : ℕ := female_students_in_first_class + females_class2 + females_class3
  let unpartnered_students : ℕ := 2

  female_students_in_first_class = 9 ∧
  total_males = total_females + unpartnered_students := by sorry

end NUMINAMATH_CALUDE_square_dancing_problem_l2655_265554


namespace NUMINAMATH_CALUDE_linear_equation_equivalence_l2655_265574

theorem linear_equation_equivalence (x y : ℝ) (h : x + 3 * y = 3) : 
  (x = 3 - 3 * y) ∧ (y = (3 - x) / 3) := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_equivalence_l2655_265574


namespace NUMINAMATH_CALUDE_marcella_lost_shoes_l2655_265557

/-- Given the initial number of shoe pairs and the final number of matching pairs,
    calculate the number of individual shoes lost. -/
def shoes_lost (initial_pairs : ℕ) (final_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * final_pairs

/-- Theorem stating that Marcella lost 10 individual shoes. -/
theorem marcella_lost_shoes : shoes_lost 23 18 = 10 := by
  sorry

end NUMINAMATH_CALUDE_marcella_lost_shoes_l2655_265557


namespace NUMINAMATH_CALUDE_probability_of_event_B_l2655_265532

theorem probability_of_event_B 
  (P_A : ℝ) 
  (P_A_and_B : ℝ) 
  (P_A_or_B : ℝ) 
  (h1 : P_A = 0.4)
  (h2 : P_A_and_B = 0.25)
  (h3 : P_A_or_B = 0.6) :
  P_A + (P_A_or_B - P_A + P_A_and_B) - P_A_and_B = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_event_B_l2655_265532


namespace NUMINAMATH_CALUDE_triangle_max_area_l2655_265538

/-- Given a triangle ABC where the sum of two sides is 4 and one angle is 30°, 
    the maximum area of the triangle is 1 -/
theorem triangle_max_area (a b : ℝ) (C : ℝ) (h1 : a + b = 4) (h2 : C = 30 * π / 180) :
  ∀ S : ℝ, S = 1/2 * a * b * Real.sin C → S ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2655_265538


namespace NUMINAMATH_CALUDE_cubic_inches_in_cubic_foot_l2655_265569

-- Define the conversion factor
def inches_per_foot : ℕ := 12

-- Theorem statement
theorem cubic_inches_in_cubic_foot : 
  1 * (inches_per_foot ^ 3) = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inches_in_cubic_foot_l2655_265569


namespace NUMINAMATH_CALUDE_triangle_side_function_is_identity_l2655_265528

/-- A function satisfying the triangle side and perimeter conditions -/
def TriangleSideFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x > 0 → f x > 0) ∧ 
  (∀ x y z, x > 0 → y > 0 → z > 0 →
    (x + f y > f (f z) + f x ∧ 
     f (f y) + z > x + f y ∧
     f (f z) + f x > f (f y) + z)) ∧
  (∀ p, p > 0 → ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + f y + f (f y) + z + f (f z) + f x = p)

/-- The main theorem stating that the identity function is the only function
    satisfying the triangle side and perimeter conditions -/
theorem triangle_side_function_is_identity 
  (f : ℝ → ℝ) (hf : TriangleSideFunction f) : 
  ∀ x, x > 0 → f x = x :=
sorry

end NUMINAMATH_CALUDE_triangle_side_function_is_identity_l2655_265528


namespace NUMINAMATH_CALUDE_element_in_set_implies_a_values_l2655_265579

theorem element_in_set_implies_a_values (a : ℝ) : 
  -3 ∈ ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ) → a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_implies_a_values_l2655_265579


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt10_implies_a_plusminus2_l2655_265586

theorem complex_modulus_sqrt10_implies_a_plusminus2 (a : ℝ) : 
  Complex.abs ((a + Complex.I) * (1 - Complex.I)) = Real.sqrt 10 → 
  a = 2 ∨ a = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt10_implies_a_plusminus2_l2655_265586


namespace NUMINAMATH_CALUDE_stratified_sampling_group_c_l2655_265562

/-- Represents the number of cities selected from a group in a stratified sampling. -/
def citiesSelected (totalCities : ℕ) (sampleSize : ℕ) (groupSize : ℕ) : ℕ :=
  (sampleSize * groupSize) / totalCities

/-- Proves that in a stratified sampling of 6 cities from 24 total cities, 
    where 8 cities belong to group C, the number of cities selected from group C is 2. -/
theorem stratified_sampling_group_c : 
  citiesSelected 24 6 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_c_l2655_265562


namespace NUMINAMATH_CALUDE_four_integer_pairs_satisfy_equation_l2655_265566

theorem four_integer_pairs_satisfy_equation : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1 + p.2 = p.1 * p.2 - 1) ∧ 
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_integer_pairs_satisfy_equation_l2655_265566


namespace NUMINAMATH_CALUDE_inconsistent_farm_animals_l2655_265518

theorem inconsistent_farm_animals :
  ∀ (x y z g : ℕ),
  x = 2 * y →
  y = 310 →
  z = 180 →
  x + y + z + g = 900 →
  g < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_inconsistent_farm_animals_l2655_265518


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l2655_265510

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l2655_265510


namespace NUMINAMATH_CALUDE_base_prime_representation_441_l2655_265508

/-- Base prime representation of a natural number -/
def BasePrimeRepresentation (n : ℕ) : List ℕ := sorry

/-- The list of primes up to a given number -/
def PrimesUpTo (n : ℕ) : List ℕ := sorry

theorem base_prime_representation_441 :
  let n := 441
  let primes := PrimesUpTo 7
  BasePrimeRepresentation n = [0, 2, 2, 0] ∧ 
  n = 3^2 * 7^2 ∧
  primes = [2, 3, 5, 7] := by sorry

end NUMINAMATH_CALUDE_base_prime_representation_441_l2655_265508


namespace NUMINAMATH_CALUDE_relationship_abc_l2655_265558

theorem relationship_abc (a b c : ℝ) : 
  (∃ x y : ℝ, x + y = a ∧ x^2 + y^2 = b ∧ x^3 + y^3 = c) → 
  a^3 - 3*a*b + 2*c = 0 := by
sorry

end NUMINAMATH_CALUDE_relationship_abc_l2655_265558


namespace NUMINAMATH_CALUDE_candy_pack_cost_l2655_265540

theorem candy_pack_cost (cory_has : ℝ) (cory_needs : ℝ) (num_packs : ℕ) :
  cory_has = 20 →
  cory_needs = 78 →
  num_packs = 2 →
  (cory_has + cory_needs) / num_packs = 49 := by
  sorry

end NUMINAMATH_CALUDE_candy_pack_cost_l2655_265540


namespace NUMINAMATH_CALUDE_barbara_shopping_expense_l2655_265506

-- Define the quantities and prices
def tuna_packs : ℕ := 5
def tuna_price : ℚ := 2
def water_bottles : ℕ := 4
def water_price : ℚ := 3/2
def total_spent : ℚ := 56

-- Define the theorem
theorem barbara_shopping_expense :
  total_spent - (tuna_packs * tuna_price + water_bottles * water_price) = 40 := by
  sorry

end NUMINAMATH_CALUDE_barbara_shopping_expense_l2655_265506


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_twenty_thirds_l2655_265541

theorem sum_abcd_equals_negative_twenty_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 4 ∧ b + 4 = c + 6 ∧ c + 6 = d + 8 ∧ d + 8 = a + b + c + d + 10) : 
  a + b + c + d = -20/3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_twenty_thirds_l2655_265541


namespace NUMINAMATH_CALUDE_guppy_count_theorem_l2655_265526

/-- Calculates the total number of guppies given initial count and two batches of baby guppies -/
def total_guppies (initial : ℕ) (first_batch : ℕ) (second_batch : ℕ) : ℕ :=
  initial + first_batch + second_batch

/-- Converts dozens to individual count -/
def dozens_to_count (dozens : ℕ) : ℕ :=
  dozens * 12

theorem guppy_count_theorem (initial : ℕ) (first_batch_dozens : ℕ) (second_batch : ℕ) 
  (h1 : initial = 7)
  (h2 : first_batch_dozens = 3)
  (h3 : second_batch = 9) :
  total_guppies initial (dozens_to_count first_batch_dozens) second_batch = 52 := by
  sorry

end NUMINAMATH_CALUDE_guppy_count_theorem_l2655_265526


namespace NUMINAMATH_CALUDE_david_crunches_count_l2655_265581

/-- The number of crunches Zachary did -/
def zachary_crunches : ℕ := 62

/-- The difference in crunches between Zachary and David -/
def crunch_difference : ℕ := 17

/-- The number of crunches David did -/
def david_crunches : ℕ := zachary_crunches - crunch_difference

theorem david_crunches_count : david_crunches = 45 := by
  sorry

end NUMINAMATH_CALUDE_david_crunches_count_l2655_265581


namespace NUMINAMATH_CALUDE_system_solution_exists_l2655_265523

theorem system_solution_exists (m : ℝ) : 
  m ≠ 3 → ∃ (x y : ℝ), y = m * x + 6 ∧ y = (2 * m - 3) * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_l2655_265523


namespace NUMINAMATH_CALUDE_integral_quarter_circle_area_l2655_265588

theorem integral_quarter_circle_area (r : ℝ) (h : r > 0) :
  ∫ x in (0)..(r), Real.sqrt (r^2 - x^2) = (π * r^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_quarter_circle_area_l2655_265588


namespace NUMINAMATH_CALUDE_unknown_number_solution_l2655_265534

theorem unknown_number_solution :
  ∃! y : ℝ, (0.47 * 1442 - 0.36 * y) + 65 = 5 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_solution_l2655_265534


namespace NUMINAMATH_CALUDE_isosceles_triangle_l2655_265577

theorem isosceles_triangle (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  c = 2 * a * Real.cos B → -- Given condition
  A = B                    -- Conclusion: triangle is isosceles
  := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l2655_265577


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l2655_265597

/-- A quadratic function f(x) = x^2 + bx + c with real constants b and c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_roots_properties (b c x₁ x₂ : ℝ) 
  (hroot₁ : f b c x₁ = x₁)
  (hroot₂ : f b c x₂ = x₂)
  (hx₁_pos : x₁ > 0)
  (hx₂_x₁ : x₂ - x₁ > 1) :
  (b^2 > 2*(b + 2*c)) ∧ 
  (∀ t : ℝ, 0 < t → t < x₁ → f b c t > x₁) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l2655_265597


namespace NUMINAMATH_CALUDE_divide_decimals_l2655_265584

theorem divide_decimals : (0.08 : ℚ) / (0.002 : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_divide_decimals_l2655_265584


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2655_265514

/-- The number of positive single-digit integers A for which x^2 - (2A + 1)x + 3A = 0 has positive integer solutions is 1. -/
theorem unique_quadratic_solution : 
  ∃! (A : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 
  ∃ (x : ℕ), x > 0 ∧ x^2 - (2 * A + 1) * x + 3 * A = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2655_265514


namespace NUMINAMATH_CALUDE_xy_bounds_l2655_265517

/-- Given a system of equations x + y = a and x^2 + y^2 = -a^2 + 2,
    prove that the product xy is bounded by -1 ≤ xy ≤ 1/3 -/
theorem xy_bounds (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  -1 ≤ x * y ∧ x * y ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_xy_bounds_l2655_265517


namespace NUMINAMATH_CALUDE_sin_2x_minus_pi_6_equals_cos_2x_minus_2pi_3_l2655_265531

theorem sin_2x_minus_pi_6_equals_cos_2x_minus_2pi_3 (x : ℝ) : 
  Real.sin (2 * x - π / 6) = Real.cos (2 * (x - π / 3)) := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_minus_pi_6_equals_cos_2x_minus_2pi_3_l2655_265531


namespace NUMINAMATH_CALUDE_percentage_boys_playing_soccer_l2655_265564

theorem percentage_boys_playing_soccer 
  (total_students : ℕ) 
  (boys : ℕ) 
  (playing_soccer : ℕ) 
  (girls_not_playing : ℕ) 
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : playing_soccer = 250)
  (h4 : girls_not_playing = 89)
  : (boys - (total_students - boys - girls_not_playing)) / playing_soccer * 100 = 86 := by
  sorry

end NUMINAMATH_CALUDE_percentage_boys_playing_soccer_l2655_265564


namespace NUMINAMATH_CALUDE_parallel_case_perpendicular_case_l2655_265507

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 8)

-- Define vector CD
def CD (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem for parallel case
theorem parallel_case : 
  (∃ k : ℝ, AB = k • CD 1) → 1 = 1 := by sorry

-- Theorem for perpendicular case
theorem perpendicular_case :
  (AB.1 * (CD (-9)).1 + AB.2 * (CD (-9)).2 = 0) → -9 = -9 := by sorry

end NUMINAMATH_CALUDE_parallel_case_perpendicular_case_l2655_265507


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2655_265530

theorem function_inequality_implies_a_bound 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ → ℝ) 
  (h : ∀ x₁ ∈ Set.Icc 0 1, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g a x₂)
  (hf : ∀ x, f x = x - 1 / (x + 1))
  (hg : ∀ a x, g a x = x^2 - 2*a*x + 4) :
  a ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2655_265530


namespace NUMINAMATH_CALUDE_circular_track_circumference_l2655_265551

/-- The circumference of a circular track given two cyclists' speeds and meeting time -/
theorem circular_track_circumference (speed1 speed2 : ℝ) (time : ℝ) (h1 : speed1 = 7)
    (h2 : speed2 = 8) (h3 : time = 42) :
    speed1 * time + speed2 * time = 630 := by
  sorry

end NUMINAMATH_CALUDE_circular_track_circumference_l2655_265551


namespace NUMINAMATH_CALUDE_universal_rook_program_exists_l2655_265553

/-- Represents a position on the 8x8 chessboard --/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a command for moving the rook --/
inductive Command
  | RIGHT
  | LEFT
  | UP
  | DOWN

/-- Represents a maze configuration on the 8x8 chessboard --/
def Maze := Set (Position × Position)

/-- Represents a program as a finite sequence of commands --/
def Program := List Command

/-- Function to determine if a square is accessible from a given position in a maze --/
def isAccessible (maze : Maze) (start finish : Position) : Prop := sorry

/-- Function to determine if a program visits all accessible squares from a given start position --/
def visitsAllAccessible (maze : Maze) (start : Position) (program : Program) : Prop := sorry

/-- The main theorem stating that there exists a program that works for all mazes and start positions --/
theorem universal_rook_program_exists :
  ∃ (program : Program),
    ∀ (maze : Maze) (start : Position),
      visitsAllAccessible maze start program := by sorry

end NUMINAMATH_CALUDE_universal_rook_program_exists_l2655_265553
