import Mathlib

namespace NUMINAMATH_CALUDE_max_non_intersecting_diagonals_correct_l2671_267127

/-- The maximum number of non-intersecting diagonals in a convex n-gon -/
def max_non_intersecting_diagonals (n : ℕ) : ℕ := n - 3

/-- Theorem: The maximum number of non-intersecting diagonals in a convex n-gon is n - 3 -/
theorem max_non_intersecting_diagonals_correct (n : ℕ) (h : n ≥ 3) :
  max_non_intersecting_diagonals n = n - 3 :=
by sorry

end NUMINAMATH_CALUDE_max_non_intersecting_diagonals_correct_l2671_267127


namespace NUMINAMATH_CALUDE_inequality_proof_l2671_267198

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 9*y + 3*z) * (x + 4*y + 2*z) * (2*x + 12*y + 9*z) ≥ 1029 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2671_267198


namespace NUMINAMATH_CALUDE_logarithmic_identity_l2671_267163

theorem logarithmic_identity (a b : ℝ) (h1 : a^2 + b^2 = 7*a*b) (h2 : a*b ≠ 0) :
  Real.log (|a + b| / 3) = (1/2) * (Real.log |a| + Real.log |b|) := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_identity_l2671_267163


namespace NUMINAMATH_CALUDE_total_flour_used_l2671_267139

-- Define the ratios
def cake_ratio : Fin 3 → ℕ
  | 0 => 3  -- flour
  | 1 => 2  -- butter
  | 2 => 1  -- sugar
  | _ => 0

def cream_ratio : Fin 2 → ℕ
  | 0 => 2  -- butter
  | 1 => 3  -- sugar
  | _ => 0

def cookie_ratio : Fin 3 → ℕ
  | 0 => 5  -- flour
  | 1 => 3  -- butter
  | 2 => 2  -- sugar
  | _ => 0

-- Define the additional flour
def additional_flour : ℕ := 300

-- Theorem statement
theorem total_flour_used (x y : ℕ) :
  (3 * x + additional_flour) / (2 * x + 2 * y) = 5 / 3 →
  (2 * x + 2 * y) / (x + 3 * y) = 3 / 2 →
  3 * x + additional_flour = 1200 :=
by sorry

end NUMINAMATH_CALUDE_total_flour_used_l2671_267139


namespace NUMINAMATH_CALUDE_largest_2023_digit_prime_squared_minus_one_div_30_l2671_267132

/-- p is the largest prime with 2023 digits -/
def p : Nat := sorry

/-- p^2 - 1 is divisible by 30 -/
theorem largest_2023_digit_prime_squared_minus_one_div_30 : 
  30 ∣ (p^2 - 1) := by sorry

end NUMINAMATH_CALUDE_largest_2023_digit_prime_squared_minus_one_div_30_l2671_267132


namespace NUMINAMATH_CALUDE_seven_distinct_reverse_numbers_l2671_267131

def is_reverse_after_adding_18 (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ n + 18 = (n % 10) * 10 + (n / 10)

theorem seven_distinct_reverse_numbers :
  ∃ (S : Finset ℕ), S.card = 7 ∧ ∀ n ∈ S, is_reverse_after_adding_18 n ∧
    ∀ m ∈ S, m ≠ n → m ≠ n := by
  sorry

end NUMINAMATH_CALUDE_seven_distinct_reverse_numbers_l2671_267131


namespace NUMINAMATH_CALUDE_find_m_l2671_267138

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem find_m : ∃ m : ℕ, m * factorial m + 2 * factorial m = 5040 ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l2671_267138


namespace NUMINAMATH_CALUDE_melissa_family_theorem_l2671_267166

/-- The number of Melissa's daughters and granddaughters who have no daughters -/
def num_without_daughters (total_descendants : ℕ) (num_daughters : ℕ) (daughters_with_children : ℕ) (granddaughters_per_daughter : ℕ) : ℕ :=
  (num_daughters - daughters_with_children) + (daughters_with_children * granddaughters_per_daughter)

theorem melissa_family_theorem :
  let total_descendants := 50
  let num_daughters := 10
  let daughters_with_children := num_daughters / 2
  let granddaughters_per_daughter := 4
  num_without_daughters total_descendants num_daughters daughters_with_children granddaughters_per_daughter = 45 := by
sorry

end NUMINAMATH_CALUDE_melissa_family_theorem_l2671_267166


namespace NUMINAMATH_CALUDE_units_digit_of_first_four_composites_product_l2671_267118

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

theorem units_digit_of_first_four_composites_product :
  (product_of_list first_four_composites) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_first_four_composites_product_l2671_267118


namespace NUMINAMATH_CALUDE_five_digit_sum_l2671_267155

def sum_of_digits (x : ℕ) : ℕ := 1 + 3 + 4 + 6 + x

def number_of_permutations : ℕ := 120  -- This is A₅⁵

theorem five_digit_sum (x : ℕ) :
  sum_of_digits x * number_of_permutations = 2640 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_sum_l2671_267155


namespace NUMINAMATH_CALUDE_initial_number_proof_l2671_267167

theorem initial_number_proof (x : ℝ) : x - 70 = 70 + 40 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l2671_267167


namespace NUMINAMATH_CALUDE_probability_exactly_one_instrument_l2671_267160

theorem probability_exactly_one_instrument (total : ℕ) (at_least_one_fraction : ℚ) (two_or_more : ℕ) :
  total = 800 →
  at_least_one_fraction = 2 / 5 →
  two_or_more = 96 →
  (((at_least_one_fraction * total) - two_or_more) / total : ℚ) = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_probability_exactly_one_instrument_l2671_267160


namespace NUMINAMATH_CALUDE_smallest_n_with_9_and_terminating_l2671_267179

def has_digit_9 (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ d = 9 ∧ ∃ k m : ℕ, n = k * 10 + d + m * 100

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ m k : ℕ, n = 2^m * 5^k

theorem smallest_n_with_9_and_terminating : 
  (∀ n : ℕ, n > 0 ∧ n < 4096 → ¬(is_terminating_decimal n ∧ has_digit_9 n)) ∧ 
  (is_terminating_decimal 4096 ∧ has_digit_9 4096) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_9_and_terminating_l2671_267179


namespace NUMINAMATH_CALUDE_additional_water_needed_l2671_267162

/-- Represents the capacity of a tank in liters -/
def TankCapacity : ℝ := 1000

/-- Represents the volume of water in the first tank in liters -/
def FirstTankVolume : ℝ := 300

/-- Represents the volume of water in the second tank in liters -/
def SecondTankVolume : ℝ := 450

/-- Represents the percentage of the second tank that is filled -/
def SecondTankPercentage : ℝ := 0.45

theorem additional_water_needed : 
  let remaining_first := TankCapacity - FirstTankVolume
  let remaining_second := TankCapacity - SecondTankVolume
  remaining_first + remaining_second = 1250 := by sorry

end NUMINAMATH_CALUDE_additional_water_needed_l2671_267162


namespace NUMINAMATH_CALUDE_smallest_valid_n_l2671_267120

def is_valid_n (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  (10 * (n % 10) + n / 10 - 5 = 2 * n)

theorem smallest_valid_n :
  is_valid_n 13 ∧ ∀ m, is_valid_n m → m ≥ 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l2671_267120


namespace NUMINAMATH_CALUDE_system_solution_l2671_267124

theorem system_solution :
  ∃ (x y : ℝ), 3 * x + 2 * y = 19 ∧ 2 * x - y = 1 ∧ x = 3 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2671_267124


namespace NUMINAMATH_CALUDE_point_on_graph_l2671_267150

theorem point_on_graph (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a * x - 1
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_point_on_graph_l2671_267150


namespace NUMINAMATH_CALUDE_oscar_voting_theorem_l2671_267147

/-- Represents a vote for an actor and an actress -/
structure Vote where
  actor : ℕ
  actress : ℕ

/-- The problem statement -/
theorem oscar_voting_theorem 
  (votes : Finset Vote) 
  (vote_count : votes.card = 3366)
  (unique_counts : ∀ n : ℕ, 1 ≤ n → n ≤ 100 → 
    (∃ a : ℕ, (votes.filter (λ v => v.actor = a)).card = n) ∨ 
    (∃ b : ℕ, (votes.filter (λ v => v.actress = b)).card = n)) :
  ∃ v₁ v₂ : Vote, v₁ ∈ votes ∧ v₂ ∈ votes ∧ v₁ ≠ v₂ ∧ v₁.actor = v₂.actor ∧ v₁.actress = v₂.actress :=
by
  sorry

end NUMINAMATH_CALUDE_oscar_voting_theorem_l2671_267147


namespace NUMINAMATH_CALUDE_gcd_bound_from_lcm_l2671_267148

theorem gcd_bound_from_lcm (a b : ℕ) : 
  a ≥ 1000000 ∧ a < 10000000 ∧ 
  b ≥ 1000000 ∧ b < 10000000 ∧ 
  Nat.lcm a b ≥ 100000000000 ∧ Nat.lcm a b < 1000000000000 →
  Nat.gcd a b < 1000 := by
sorry

end NUMINAMATH_CALUDE_gcd_bound_from_lcm_l2671_267148


namespace NUMINAMATH_CALUDE_f_2_eq_0_f_positive_solution_set_l2671_267185

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - (3 - a)*x + 2*(1 - a)

-- Theorem for f(2) = 0
theorem f_2_eq_0 (a : ℝ) : f a 2 = 0 := by sorry

-- Define the solution set for f(x) > 0
def solution_set (a : ℝ) : Set ℝ :=
  if a < -1 then {x | x < 2 ∨ x > 1 - a}
  else if a = -1 then {x | x < 2 ∨ x > 2}
  else {x | x < 1 - a ∨ x > 2}

-- Theorem for the solution set of f(x) > 0
theorem f_positive_solution_set (a : ℝ) :
  {x : ℝ | f a x > 0} = solution_set a := by sorry

end NUMINAMATH_CALUDE_f_2_eq_0_f_positive_solution_set_l2671_267185


namespace NUMINAMATH_CALUDE_slower_train_time_l2671_267119

/-- Represents a train traveling between two stations -/
structure Train where
  speed : ℝ
  remainingDistance : ℝ

/-- The problem setup -/
def trainProblem (fasterTrain slowerTrain : Train) : Prop :=
  fasterTrain.speed = 3 * slowerTrain.speed ∧
  fasterTrain.remainingDistance = slowerTrain.remainingDistance ∧
  fasterTrain.remainingDistance = 4 * fasterTrain.speed

/-- The theorem to prove -/
theorem slower_train_time
    (fasterTrain slowerTrain : Train)
    (h : trainProblem fasterTrain slowerTrain) :
    slowerTrain.remainingDistance / slowerTrain.speed = 12 := by
  sorry

#check slower_train_time

end NUMINAMATH_CALUDE_slower_train_time_l2671_267119


namespace NUMINAMATH_CALUDE_min_quadrilateral_area_l2671_267168

-- Define the curve E
def curve_E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the tangent circle
def tangent_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the tangent line
def tangent_line (x : ℝ) : Prop := x = -2

-- Define the point F
def point_F : ℝ × ℝ := (1, 0)

-- Define the quadrilateral area function
def quadrilateral_area (a b c d : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem min_quadrilateral_area :
  ∀ (a b c d : ℝ × ℝ),
    (∃ (m : ℝ), m ≠ 0 ∧
      curve_E a.1 a.2 ∧ curve_E b.1 b.2 ∧ curve_E c.1 c.2 ∧ curve_E d.1 d.2 ∧
      (a.1 - point_F.1) * (c.1 - point_F.1) + (a.2 - point_F.2) * (c.2 - point_F.2) = 0 ∧
      (b.1 - point_F.1) * (d.1 - point_F.1) + (b.2 - point_F.2) * (d.2 - point_F.2) = 0) →
    quadrilateral_area a b c d ≥ 32 :=
sorry

end NUMINAMATH_CALUDE_min_quadrilateral_area_l2671_267168


namespace NUMINAMATH_CALUDE_starters_count_l2671_267101

/-- The number of ways to select 7 starters from a team of 16 players,
    including a set of twins, with the condition that at least one but
    no more than two twins must be included. -/
def select_starters (total_players : ℕ) (num_twins : ℕ) (num_starters : ℕ) : ℕ :=
  let non_twin_players := total_players - num_twins
  let one_twin := num_twins * Nat.choose non_twin_players (num_starters - 1)
  let both_twins := Nat.choose non_twin_players (num_starters - num_twins)
  one_twin + both_twins

theorem starters_count :
  select_starters 16 2 7 = 8008 := by
  sorry

end NUMINAMATH_CALUDE_starters_count_l2671_267101


namespace NUMINAMATH_CALUDE_triangle_kite_property_l2671_267112

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))
-- Define points D, H, M, N
variable (D H M N : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def is_acute_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_angle_bisector (A D B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_altitude (A H B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def on_circle (M B D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_kite (A M H N : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- State the theorem
theorem triangle_kite_property 
  (h_acute : is_acute_triangle A B C)
  (h_bisector : is_angle_bisector A D B C)
  (h_altitude : is_altitude A H B C)
  (h_circle_M : on_circle M B D)
  (h_circle_N : on_circle N C D) :
  is_kite A M H N :=
sorry

end NUMINAMATH_CALUDE_triangle_kite_property_l2671_267112


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2671_267106

theorem division_remainder_problem (x y : ℤ) (r : ℕ) 
  (h1 : x > 0)
  (h2 : x = 10 * y + r)
  (h3 : 0 ≤ r ∧ r < 10)
  (h4 : 2 * x = 7 * (3 * y) + 1)
  (h5 : 11 * y - x = 2) :
  r = 3 := by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2671_267106


namespace NUMINAMATH_CALUDE_q_profit_share_l2671_267122

/-- Calculates the share of profit for a partner in a business partnership --/
def calculateProfitShare (investmentP investmentQ totalProfit : ℕ) : ℕ :=
  let totalInvestment := investmentP + investmentQ
  let shareQ := (investmentQ * totalProfit) / totalInvestment
  shareQ

/-- Theorem stating that Q's share of the profit is 7200 given the specified investments and total profit --/
theorem q_profit_share :
  calculateProfitShare 54000 36000 18000 = 7200 := by
  sorry

#eval calculateProfitShare 54000 36000 18000

end NUMINAMATH_CALUDE_q_profit_share_l2671_267122


namespace NUMINAMATH_CALUDE_inheritance_distribution_correct_l2671_267171

/-- Represents the distribution of an inheritance among three sons and a hospital. -/
structure InheritanceDistribution where
  eldest : ℕ
  middle : ℕ
  youngest : ℕ
  hospital : ℕ

/-- Checks if the given distribution satisfies the inheritance conditions. -/
def satisfies_conditions (d : InheritanceDistribution) : Prop :=
  -- Total inheritance is $1320
  d.eldest + d.middle + d.youngest + d.hospital = 1320 ∧
  -- If hospital's portion went to eldest son
  d.eldest + d.hospital = d.middle + d.youngest ∧
  -- If hospital's portion went to middle son
  d.middle + d.hospital = 2 * (d.eldest + d.youngest) ∧
  -- If hospital's portion went to youngest son
  d.youngest + d.hospital = 3 * (d.eldest + d.middle)

/-- The theorem stating that the given distribution satisfies all conditions. -/
theorem inheritance_distribution_correct : 
  satisfies_conditions ⟨55, 275, 385, 605⟩ := by
  sorry

end NUMINAMATH_CALUDE_inheritance_distribution_correct_l2671_267171


namespace NUMINAMATH_CALUDE_sum_of_digits_power_of_two_l2671_267143

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Main theorem -/
theorem sum_of_digits_power_of_two : sum_of_digits (sum_of_digits (sum_of_digits (2^2006))) = 4 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_digits_power_of_two_l2671_267143


namespace NUMINAMATH_CALUDE_park_length_l2671_267157

/-- The length of a rectangular park given its perimeter and breadth -/
theorem park_length (perimeter breadth : ℝ) (h1 : perimeter = 1000) (h2 : breadth = 200) :
  2 * (perimeter / 2 - breadth) = 300 := by
  sorry

end NUMINAMATH_CALUDE_park_length_l2671_267157


namespace NUMINAMATH_CALUDE_ad_arrangement_count_l2671_267180

/-- The number of ways to arrange n items, taking r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of ways to arrange 6 advertisements (4 commercial and 2 public service) 
    where the 2 public service ads cannot be consecutive -/
def ad_arrangements : ℕ :=
  permutations 4 4 * permutations 5 2

theorem ad_arrangement_count : 
  ad_arrangements = permutations 4 4 * permutations 5 2 := by
  sorry

end NUMINAMATH_CALUDE_ad_arrangement_count_l2671_267180


namespace NUMINAMATH_CALUDE_equation_equivalence_l2671_267103

theorem equation_equivalence (x : ℝ) : 2 * (x + 1) = x + 7 ↔ x = 5 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2671_267103


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2671_267146

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x+2) * (4 : ℝ)^(2*x+3) = (8 : ℝ)^(3*x+4) * (2 : ℝ)^2 ∧ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2671_267146


namespace NUMINAMATH_CALUDE_combined_price_is_3105_l2671_267184

/-- Calculate the selling price of an item given its cost and profit percentage -/
def selling_price (cost : ℕ) (profit_percentage : ℕ) : ℕ :=
  cost + cost * profit_percentage / 100

/-- Combined selling price of three items -/
def combined_selling_price (cost_A cost_B cost_C : ℕ) (profit_A profit_B profit_C : ℕ) : ℕ :=
  selling_price cost_A profit_A + selling_price cost_B profit_B + selling_price cost_C profit_C

/-- Theorem stating the combined selling price of the three items -/
theorem combined_price_is_3105 :
  combined_selling_price 500 800 1200 25 30 20 = 3105 := by
  sorry


end NUMINAMATH_CALUDE_combined_price_is_3105_l2671_267184


namespace NUMINAMATH_CALUDE_smallest_zack_students_correct_l2671_267170

/-- Represents the number of students in a group for each tutor -/
structure TutorGroup where
  zack : Nat
  karen : Nat
  julie : Nat

/-- Represents the ratio of students for each tutor -/
structure TutorRatio where
  zack : Nat
  karen : Nat
  julie : Nat

/-- The smallest number of students Zack can have given the conditions -/
def smallestZackStudents (g : TutorGroup) (r : TutorRatio) : Nat :=
  630

theorem smallest_zack_students_correct (g : TutorGroup) (r : TutorRatio) :
  g.zack = 14 →
  g.karen = 10 →
  g.julie = 15 →
  r.zack = 3 →
  r.karen = 2 →
  r.julie = 5 →
  smallestZackStudents g r = 630 ∧
  smallestZackStudents g r % g.zack = 0 ∧
  (smallestZackStudents g r / r.zack * r.karen) % g.karen = 0 ∧
  (smallestZackStudents g r / r.zack * r.julie) % g.julie = 0 ∧
  ∀ n : Nat, n < smallestZackStudents g r →
    (n % g.zack = 0 ∧ (n / r.zack * r.karen) % g.karen = 0 ∧ (n / r.zack * r.julie) % g.julie = 0) →
    False :=
by
  sorry

#check smallest_zack_students_correct

end NUMINAMATH_CALUDE_smallest_zack_students_correct_l2671_267170


namespace NUMINAMATH_CALUDE_combination_minus_permutation_l2671_267183

-- Define combination
def combination (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define permutation
def permutation (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial (n - k))

-- Theorem statement
theorem combination_minus_permutation : combination 7 4 - permutation 5 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_combination_minus_permutation_l2671_267183


namespace NUMINAMATH_CALUDE_coefficient_of_x_l2671_267137

/-- The coefficient of x in the simplified expression 5(2x - 3) + 7(10 - 3x^2 + 2x) - 9(4x - 2) is -12 -/
theorem coefficient_of_x (x : ℝ) : 
  let expr := 5*(2*x - 3) + 7*(10 - 3*x^2 + 2*x) - 9*(4*x - 2)
  ∃ (a b c : ℝ), expr = a*x^2 + (-12)*x + b + c := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l2671_267137


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_two_sqrt_31_l2671_267161

theorem sqrt_sum_equals_two_sqrt_31 :
  Real.sqrt (24 - 10 * Real.sqrt 5) + Real.sqrt (24 + 10 * Real.sqrt 5) = 2 * Real.sqrt 31 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_two_sqrt_31_l2671_267161


namespace NUMINAMATH_CALUDE_max_value_of_4x_plus_3y_l2671_267192

theorem max_value_of_4x_plus_3y (x y : ℝ) : 
  x^2 + y^2 = 10*x + 8*y + 10 → (4*x + 3*y ≤ 70) ∧ ∃ x y, x^2 + y^2 = 10*x + 8*y + 10 ∧ 4*x + 3*y = 70 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_4x_plus_3y_l2671_267192


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2671_267175

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (1 + a) * x + a > 0}
  (a > 1 → S = {x : ℝ | x > a ∨ x < 1}) ∧
  (a = 1 → S = {x : ℝ | x ≠ 1}) ∧
  (a < 1 → S = {x : ℝ | x > 1 ∨ x < a}) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2671_267175


namespace NUMINAMATH_CALUDE_chord_length_circle_line_l2671_267136

/-- The length of the chord intersected by a line on a circle -/
theorem chord_length_circle_line (x y : ℝ) : 
  let circle := fun (x y : ℝ) => (x - 2)^2 + y^2 = 4
  let line := fun (x y : ℝ) => 4*x - 3*y - 3 = 0
  let center := (2, 0)
  let radius := 2
  let d := |4*2 - 3*0 - 3| / Real.sqrt (4^2 + (-3)^2)
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle x₁ y₁ ∧ circle x₂ y₂ ∧ 
    line x₁ y₁ ∧ line x₂ y₂ ∧
    ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 * (radius^2 - d^2) :=
by
  sorry

#check chord_length_circle_line

end NUMINAMATH_CALUDE_chord_length_circle_line_l2671_267136


namespace NUMINAMATH_CALUDE_stating_binary_arithmetic_equality_l2671_267182

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Represents the binary number 1101₂ -/
def b1101 : List Bool := [true, true, false, true]

/-- Represents the binary number 111₂ -/
def b111 : List Bool := [true, true, true]

/-- Represents the binary number 101₂ -/
def b101 : List Bool := [true, false, true]

/-- Represents the binary number 1001₂ -/
def b1001 : List Bool := [true, false, false, true]

/-- Represents the binary number 11₂ -/
def b11 : List Bool := [true, true]

/-- Represents the binary number 10101₂ (the expected result) -/
def b10101 : List Bool := [true, false, true, false, true]

/-- 
Theorem stating that the binary arithmetic operation 
1101₂ + 111₂ - 101₂ + 1001₂ - 11₂ equals 10101₂
-/
theorem binary_arithmetic_equality : 
  binary_to_nat b1101 + binary_to_nat b111 - binary_to_nat b101 + 
  binary_to_nat b1001 - binary_to_nat b11 = binary_to_nat b10101 := by
  sorry

end NUMINAMATH_CALUDE_stating_binary_arithmetic_equality_l2671_267182


namespace NUMINAMATH_CALUDE_locus_is_hyperbola_l2671_267149

/-- The locus of points equidistant from two fixed points is a hyperbola -/
theorem locus_is_hyperbola (P : ℝ × ℝ) :
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (4, 0)
  let dist (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  dist P F - dist P O = 1 →
  ∃ (a b : ℝ), (P.1 / a)^2 - (P.2 / b)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_locus_is_hyperbola_l2671_267149


namespace NUMINAMATH_CALUDE_sum_of_incircle_areas_l2671_267111

/-- Given a triangle ABC with side lengths a, b, c and inradius r, 
    the sum of the areas of its incircle and the incircles of the three smaller triangles 
    formed by tangent lines to the incircle parallel to the sides of ABC 
    is equal to (7πr²)/4. -/
theorem sum_of_incircle_areas (a b c r : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) :
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  r = K / s →
  (π * r^2) + 3 * (π * (r/2)^2) = (7 * π * r^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_incircle_areas_l2671_267111


namespace NUMINAMATH_CALUDE_part_one_part_two_l2671_267104

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 2}

-- Theorem for part 1
theorem part_one : (Set.univ \ A) ∩ B 1 = {x | 1 < x ∧ x < 2} := by sorry

-- Theorem for part 2
theorem part_two : ∀ a : ℝ, (Set.univ \ A) ∩ B a = ∅ ↔ a ≤ -1 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2671_267104


namespace NUMINAMATH_CALUDE_emily_subtraction_l2671_267110

theorem emily_subtraction : 50^2 - 49^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_emily_subtraction_l2671_267110


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_360_l2671_267178

theorem largest_divisor_of_n_squared_div_360 (n : ℕ+) (h : 360 ∣ n^2) :
  ∃ (t : ℕ), t = 60 ∧ t ∣ n ∧ ∀ (k : ℕ), k ∣ n → k ≤ t :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_360_l2671_267178


namespace NUMINAMATH_CALUDE_not_necessarily_equal_numbers_l2671_267102

theorem not_necessarily_equal_numbers : ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a + b^2 + c^2 = b + a^2 + c^2) ∧
  (a + b^2 + c^2 = c + a^2 + b^2) ∧
  ¬(a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_equal_numbers_l2671_267102


namespace NUMINAMATH_CALUDE_special_function_properties_l2671_267189

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (4 - x) = 0) ∧ (∀ x, f (x + 2) - f (x - 2) = 0)

/-- Theorem stating the properties of the special function -/
theorem special_function_properties (f : ℝ → ℝ) (h : SpecialFunction f) :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 4) = f x) := by
  sorry


end NUMINAMATH_CALUDE_special_function_properties_l2671_267189


namespace NUMINAMATH_CALUDE_monotonic_sine_range_l2671_267128

/-- The function f(x) = 2sin(ωx) is monotonically increasing on [-π/3, π/4] iff 0 < ω ≤ 12/7 -/
theorem monotonic_sine_range (ω : ℝ) :
  ω > 0 →
  (∀ x ∈ Set.Icc (-π/3) (π/4), Monotone (fun x => 2 * Real.sin (ω * x))) ↔
  ω ≤ 12/7 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_sine_range_l2671_267128


namespace NUMINAMATH_CALUDE_set_equality_condition_l2671_267108

-- Define set A
def A : Set ℝ := {x | (x + 1)^2 * (2 - x) / (4 + x) ≥ 0 ∧ x ≠ -4}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 2*a + 1) ≤ 0}

-- Theorem statement
theorem set_equality_condition (a : ℝ) : 
  A ∪ B a = A ↔ -3/2 < a ∧ a ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_set_equality_condition_l2671_267108


namespace NUMINAMATH_CALUDE_target_hit_probability_l2671_267172

theorem target_hit_probability (hit_rate_A hit_rate_B : ℚ) 
  (h1 : hit_rate_A = 4/5)
  (h2 : hit_rate_B = 5/6) :
  1 - (1 - hit_rate_A) * (1 - hit_rate_B) = 29/30 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l2671_267172


namespace NUMINAMATH_CALUDE_line_translation_upwards_l2671_267194

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically --/
def translateLine (l : Line) (c : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + c }

/-- The equation of a line in slope-intercept form --/
def lineEquation (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem line_translation_upwards 
  (original : Line) 
  (c : ℝ) 
  (h : c > 0) : 
  ∀ x y : ℝ, lineEquation original x y ↔ lineEquation (translateLine original c) x (y + c) :=
by sorry

end NUMINAMATH_CALUDE_line_translation_upwards_l2671_267194


namespace NUMINAMATH_CALUDE_max_value_inequality_l2671_267121

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x^2 + 1/y^2) * (x^2 + 1/y^2 - 100) + (y^2 + 1/x^2) * (y^2 + 1/x^2 - 100) ≤ -5000 := by
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2671_267121


namespace NUMINAMATH_CALUDE_race_course_length_race_course_length_proof_l2671_267159

/-- Given two runners A and B, where A runs 4 times as fast as B and gives B a 63-meter head start,
    the length of the race course that allows both runners to finish at the same time is 84 meters. -/
theorem race_course_length : ℝ → ℝ → Prop :=
  fun (speed_B : ℝ) (course_length : ℝ) =>
    speed_B > 0 →
    course_length > 63 →
    course_length / (4 * speed_B) = (course_length - 63) / speed_B →
    course_length = 84

/-- Proof of the race_course_length theorem -/
theorem race_course_length_proof : ∃ (speed_B : ℝ) (course_length : ℝ),
  race_course_length speed_B course_length :=
by
  sorry

end NUMINAMATH_CALUDE_race_course_length_race_course_length_proof_l2671_267159


namespace NUMINAMATH_CALUDE_tan_X_equals_four_l2671_267153

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ) where
  -- Angle Y is 90°
  right_angle : Y = 90
  -- Length of side YZ
  yz_length : Z - Y = 4
  -- Length of side XZ
  xz_length : Z - X = Real.sqrt 17

-- Theorem statement
theorem tan_X_equals_four {X Y Z : ℝ} (t : Triangle X Y Z) : Real.tan X = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_X_equals_four_l2671_267153


namespace NUMINAMATH_CALUDE_race_time_differences_l2671_267134

def race_distance : ℝ := 10
def john_speed : ℝ := 15
def alice_time : ℝ := 48
def bob_time : ℝ := 52
def charlie_time : ℝ := 55

theorem race_time_differences :
  let john_time := race_distance / john_speed * 60
  (alice_time - john_time = 8) ∧
  (bob_time - john_time = 12) ∧
  (charlie_time - john_time = 15) := by
  sorry

end NUMINAMATH_CALUDE_race_time_differences_l2671_267134


namespace NUMINAMATH_CALUDE_max_q_minus_r_for_1027_l2671_267165

theorem max_q_minus_r_for_1027 :
  ∃ (q r : ℕ+), 1027 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 1027 = 23 * q' + r' → q' - r' ≤ q - r ∧ q - r = 29 := by
sorry

end NUMINAMATH_CALUDE_max_q_minus_r_for_1027_l2671_267165


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l2671_267177

def f (x : ℝ) := x^3 - 3*x - 3

theorem f_has_root_in_interval :
  ∃ (x : ℝ), x ∈ Set.Ioo 2 3 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_root_in_interval_l2671_267177


namespace NUMINAMATH_CALUDE_arithmetic_sequence_minimum_l2671_267142

theorem arithmetic_sequence_minimum (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (a 2 * a 12).sqrt = 4 →  -- Geometric mean of a_2 and a_12 is 4
  (∃ r : ℝ, ∀ n, a (n + 1) = a n + r) →  -- Arithmetic sequence
  (∃ m : ℝ, ∀ r : ℝ, 2 * a 5 + 8 * a 9 ≥ m) →  -- Minimum exists
  a 3 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_minimum_l2671_267142


namespace NUMINAMATH_CALUDE_water_one_eighth_after_three_pourings_l2671_267164

def water_remaining (n : ℕ) : ℚ :=
  (1 : ℚ) / 2^n

theorem water_one_eighth_after_three_pourings :
  water_remaining 3 = (1 : ℚ) / 8 := by
  sorry

#check water_one_eighth_after_three_pourings

end NUMINAMATH_CALUDE_water_one_eighth_after_three_pourings_l2671_267164


namespace NUMINAMATH_CALUDE_quadratic_real_root_l2671_267190

theorem quadratic_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l2671_267190


namespace NUMINAMATH_CALUDE_farmer_field_area_l2671_267100

/-- Represents the farmer's field ploughing problem -/
def FarmerField (initial_productivity : ℝ) (productivity_increase : ℝ) (days_saved : ℕ) : Prop :=
  ∃ (total_days : ℕ) (field_area : ℝ),
    field_area = initial_productivity * total_days ∧
    field_area = (2 * initial_productivity) + 
      ((total_days - days_saved - 2) * (initial_productivity * (1 + productivity_increase))) ∧
    field_area = 1440

/-- Theorem stating that the field area is 1440 hectares given the problem conditions -/
theorem farmer_field_area :
  FarmerField 120 0.25 2 :=
sorry

end NUMINAMATH_CALUDE_farmer_field_area_l2671_267100


namespace NUMINAMATH_CALUDE_car_travel_distance_l2671_267154

/-- Proves that a car can travel 500 miles before refilling given specific journey conditions. -/
theorem car_travel_distance (fuel_cost : ℝ) (journey_distance : ℝ) (food_ratio : ℝ) (total_spent : ℝ)
  (h1 : fuel_cost = 45)
  (h2 : journey_distance = 2000)
  (h3 : food_ratio = 3/5)
  (h4 : total_spent = 288) :
  journey_distance / (total_spent / ((1 + food_ratio) * fuel_cost)) = 500 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l2671_267154


namespace NUMINAMATH_CALUDE_f_positive_m_range_l2671_267193

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for the solution set of f(x) > 0
theorem f_positive (x : ℝ) : f x > 0 ↔ x < -1/3 ∨ x > 3 := by sorry

-- Theorem for the range of m
theorem m_range (m : ℝ) : 
  (∃ x₀ : ℝ, f x₀ + 2*m^2 < 4*m) ↔ -1/2 < m ∧ m < 5/2 := by sorry

end NUMINAMATH_CALUDE_f_positive_m_range_l2671_267193


namespace NUMINAMATH_CALUDE_girls_trying_out_l2671_267116

theorem girls_trying_out (girls : ℕ) (boys : ℕ) (called_back : ℕ) (didnt_make_cut : ℕ) :
  boys = 32 →
  called_back = 10 →
  didnt_make_cut = 39 →
  girls + boys = called_back + didnt_make_cut →
  girls = 17 := by
  sorry

end NUMINAMATH_CALUDE_girls_trying_out_l2671_267116


namespace NUMINAMATH_CALUDE_total_oranges_l2671_267156

/-- Given 3.0 children and 1.333333333 oranges per child, prove that the total number of oranges is 4. -/
theorem total_oranges (num_children : ℝ) (oranges_per_child : ℝ) 
  (h1 : num_children = 3.0) 
  (h2 : oranges_per_child = 1.333333333) : 
  num_children * oranges_per_child = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_l2671_267156


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_f_l2671_267196

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) := x * Real.log x

-- State the theorem
theorem monotonic_decreasing_interval_f :
  ∀ x : ℝ, x > 0 → (StrictMonoOn f (Set.Ioo 0 (Real.exp (-1)))) ∧
  (∀ y : ℝ, y > Real.exp (-1) → ¬ StrictMonoOn f (Set.Ioo 0 y)) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_f_l2671_267196


namespace NUMINAMATH_CALUDE_product_of_solutions_l2671_267140

theorem product_of_solutions (x₁ x₂ : ℝ) : 
  (|5 * x₁ - 1| + 4 = 54) → 
  (|5 * x₂ - 1| + 4 = 54) → 
  x₁ ≠ x₂ →
  x₁ * x₂ = -99.96 := by
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2671_267140


namespace NUMINAMATH_CALUDE_kamals_math_marks_l2671_267135

def english_marks : ℕ := 96
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def average_marks : ℕ := 79
def total_subjects : ℕ := 5

theorem kamals_math_marks :
  let total_marks := average_marks * total_subjects
  let known_marks := english_marks + physics_marks + chemistry_marks + biology_marks
  let math_marks := total_marks - known_marks
  math_marks = 65 := by sorry

end NUMINAMATH_CALUDE_kamals_math_marks_l2671_267135


namespace NUMINAMATH_CALUDE_paige_homework_problem_l2671_267114

/-- The number of problems Paige has left to do for homework -/
def problems_left (math science history language_arts finished_at_school unfinished_math : ℕ) : ℕ :=
  math + science + history + language_arts - finished_at_school + unfinished_math

theorem paige_homework_problem :
  problems_left 43 12 10 5 44 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_paige_homework_problem_l2671_267114


namespace NUMINAMATH_CALUDE_ship_journey_distance_l2671_267130

/-- The total distance traveled by a ship in three days -/
def ship_total_distance (first_day_distance : ℝ) : ℝ :=
  let second_day_distance := 3 * first_day_distance
  let third_day_distance := second_day_distance + 110
  first_day_distance + second_day_distance + third_day_distance

/-- Theorem stating the total distance traveled by the ship -/
theorem ship_journey_distance : ship_total_distance 100 = 810 := by
  sorry

end NUMINAMATH_CALUDE_ship_journey_distance_l2671_267130


namespace NUMINAMATH_CALUDE_fraction_pair_sum_equality_l2671_267123

theorem fraction_pair_sum_equality (n : ℕ) (h : n > 2009) :
  ∃ (a b c d : ℕ), a ≤ n ∧ b ≤ n ∧ c ≤ n ∧ d ≤ n ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (1 : ℚ) / (n + 1 - a) + (1 : ℚ) / (n + 1 - b) =
  (1 : ℚ) / (n + 1 - c) + (1 : ℚ) / (n + 1 - d) :=
by sorry

end NUMINAMATH_CALUDE_fraction_pair_sum_equality_l2671_267123


namespace NUMINAMATH_CALUDE_erased_numbers_sum_l2671_267151

/-- Represents a sequence of consecutive odd numbers -/
def OddSequence : ℕ → ℕ := λ n => 2 * n - 1

/-- Sum of the first n odd numbers -/
def SumOfOddNumbers (n : ℕ) : ℕ := n * n

theorem erased_numbers_sum (first_segment_sum second_segment_sum : ℕ) 
  (h1 : first_segment_sum = 961) 
  (h2 : second_segment_sum = 1001) : 
  ∃ (k1 k2 : ℕ), 
    k1 < k2 ∧ 
    SumOfOddNumbers (k1 - 1) = first_segment_sum ∧
    SumOfOddNumbers (k2 - 1) - SumOfOddNumbers k1 = second_segment_sum ∧
    OddSequence k1 + OddSequence k2 = 154 := by
  sorry

end NUMINAMATH_CALUDE_erased_numbers_sum_l2671_267151


namespace NUMINAMATH_CALUDE_distance_between_5th_and_29th_red_light_l2671_267115

/-- Represents the color of a light in the sequence -/
inductive Color
  | Red
  | Blue
  | Green

/-- Defines the repeating pattern of lights -/
def pattern : List Color := [Color.Red, Color.Red, Color.Red, Color.Blue, Color.Blue, Color.Green, Color.Green]

/-- The distance between each light in inches -/
def light_distance : ℕ := 8

/-- Calculates the position of the nth red light in the sequence -/
def red_light_position (n : ℕ) : ℕ :=
  (n - 1) / 3 * 7 + (n - 1) % 3 + 1

/-- Calculates the distance between two positions in the sequence -/
def distance_between (pos1 pos2 : ℕ) : ℕ :=
  (pos2 - pos1) * light_distance

/-- Converts a distance in inches to feet -/
def inches_to_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem distance_between_5th_and_29th_red_light :
  inches_to_feet (distance_between (red_light_position 5) (red_light_position 29)) = 37 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_5th_and_29th_red_light_l2671_267115


namespace NUMINAMATH_CALUDE_min_area_enclosed_l2671_267191

/-- The function f(x) = 3 - x^2 --/
def f (x : ℝ) : ℝ := 3 - x^2

/-- A point on the graph of f --/
structure PointOnGraph where
  x : ℝ
  y : ℝ
  on_graph : y = f x

/-- The area enclosed by tangents and x-axis --/
def enclosed_area (A B : PointOnGraph) : ℝ :=
  sorry -- Definition of the area calculation

/-- Theorem: Minimum area enclosed by tangents and x-axis --/
theorem min_area_enclosed (A B : PointOnGraph) 
    (h_opposite : A.x * B.x < 0) : -- A and B are on opposite sides of y-axis
  ∃ (min_area : ℝ), min_area = 8 ∧ ∀ (P Q : PointOnGraph), 
    P.x * Q.x < 0 → enclosed_area P Q ≥ min_area := by
  sorry

end NUMINAMATH_CALUDE_min_area_enclosed_l2671_267191


namespace NUMINAMATH_CALUDE_tangent_and_below_and_two_zeros_l2671_267173

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := Real.log x - a * x + 1

def tangent_line (x y : ℝ) : Prop := (1 - a) * x - y = 0

def g (x : ℝ) : ℝ := 1/2 * a * x^2 - (f a x + a * x)

theorem tangent_and_below_and_two_zeros :
  (∀ y, tangent_line a 1 y ↔ y = f a 1) ∧
  (∀ x > 0, x ≠ 1 → f a x < (1 - a) * x) ∧
  (∃ x₁ x₂, x₁ < x₂ ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧ ∀ x, x ≠ x₁ → x ≠ x₂ → g a x ≠ 0) ↔
  0 < a ∧ a < Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_and_below_and_two_zeros_l2671_267173


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2671_267188

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 47) 
  (eq2 : 7 * a + 2 * b = 54) : 
  a + b = -103 / 31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2671_267188


namespace NUMINAMATH_CALUDE_y_axis_intersection_l2671_267152

/-- The quadratic function f(x) = 3x^2 - 4x + 5 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 5

/-- The y-axis intersection point of f(x) is (0, 5) -/
theorem y_axis_intersection :
  f 0 = 5 :=
by sorry

end NUMINAMATH_CALUDE_y_axis_intersection_l2671_267152


namespace NUMINAMATH_CALUDE_z_squared_minus_norm_squared_l2671_267113

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- Theorem statement
theorem z_squared_minus_norm_squared :
  z^2 - Complex.abs z^2 = 2 * Complex.I - 2 := by
  sorry

end NUMINAMATH_CALUDE_z_squared_minus_norm_squared_l2671_267113


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2671_267107

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2671_267107


namespace NUMINAMATH_CALUDE_five_solutions_for_f_f_x_eq_8_l2671_267181

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -2 then x^2 - 1 else x + 4

theorem five_solutions_for_f_f_x_eq_8 :
  ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x : ℝ, x ∈ s ↔ f (f x) = 8 :=
sorry

end NUMINAMATH_CALUDE_five_solutions_for_f_f_x_eq_8_l2671_267181


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_200_l2671_267144

theorem closest_integer_to_cube_root_200 : 
  ∀ n : ℤ, |n - (200 : ℝ)^(1/3)| ≥ |6 - (200 : ℝ)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_200_l2671_267144


namespace NUMINAMATH_CALUDE_problem_solution_l2671_267129

theorem problem_solution : (2010^2 - 2010) / 2010 = 2009 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2671_267129


namespace NUMINAMATH_CALUDE_simplify_radicals_l2671_267117

theorem simplify_radicals : 
  (Real.sqrt 440 / Real.sqrt 55) - (Real.sqrt 210 / Real.sqrt 70) = 2 * Real.sqrt 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l2671_267117


namespace NUMINAMATH_CALUDE_second_year_increase_is_twenty_percent_l2671_267126

/-- Calculates the percentage increase in the second year given initial population,
    first year increase, and final population after two years. -/
def second_year_increase (initial_population : ℕ) (first_year_increase : ℚ) (final_population : ℕ) : ℚ :=
  let after_first_year := initial_population * (1 + first_year_increase)
  let second_year_factor := final_population / after_first_year
  (second_year_factor - 1) * 100

/-- Theorem stating that given the problem conditions, the second year increase is 20%. -/
theorem second_year_increase_is_twenty_percent :
  second_year_increase 1000 (10 / 100) 1320 = 20 := by
  sorry

#eval second_year_increase 1000 (10 / 100) 1320

end NUMINAMATH_CALUDE_second_year_increase_is_twenty_percent_l2671_267126


namespace NUMINAMATH_CALUDE_cos_angle_F₁PF₂_l2671_267158

-- Define the ellipse and hyperbola
def is_on_ellipse (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1
def is_on_hyperbola (x y : ℝ) : Prop := x^2/3 - y^2 = 1

-- Define the common foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the common point P
structure CommonPoint where
  x : ℝ
  y : ℝ
  on_ellipse : is_on_ellipse x y
  on_hyperbola : is_on_hyperbola x y

-- Theorem statement
theorem cos_angle_F₁PF₂ (P : CommonPoint) : 
  let PF₁ := (F₁.1 - P.x, F₁.2 - P.y)
  let PF₂ := (F₂.1 - P.x, F₂.2 - P.y)
  let dot_product := PF₁.1 * PF₂.1 + PF₁.2 * PF₂.2
  let magnitude_PF₁ := Real.sqrt (PF₁.1^2 + PF₁.2^2)
  let magnitude_PF₂ := Real.sqrt (PF₂.1^2 + PF₂.2^2)
  dot_product / (magnitude_PF₁ * magnitude_PF₂) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_cos_angle_F₁PF₂_l2671_267158


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l2671_267174

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def has_no_repeated_prime_factors (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → (n % (p * p) ≠ 0)

def is_valid_triple (x y : ℕ) : Prop :=
  x < 10 ∧ y < 10 ∧ x ≠ y ∧ is_prime (10 * x + y)

theorem largest_three_digit_product :
  ∃ m x y : ℕ,
    m = x * y * (10 * x + y) ∧
    is_valid_triple x y ∧
    has_no_repeated_prime_factors m ∧
    m < 1000 ∧
    (∀ m' x' y' : ℕ,
      m' = x' * y' * (10 * x' + y') →
      is_valid_triple x' y' →
      has_no_repeated_prime_factors m' →
      m' < 1000 →
      m' ≤ m) ∧
    m = 777 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l2671_267174


namespace NUMINAMATH_CALUDE_dogsled_race_speed_l2671_267176

/-- Given two teams racing on a 300-mile course, where one team finishes 3 hours faster
    and has an average speed 5 mph greater than the other, prove that the slower team's
    average speed is 20 mph. -/
theorem dogsled_race_speed (course_length : ℝ) (time_difference : ℝ) (speed_difference : ℝ)
  (h1 : course_length = 300)
  (h2 : time_difference = 3)
  (h3 : speed_difference = 5) :
  let speed_B := (course_length / (course_length / (20 + speed_difference) + time_difference))
  speed_B = 20 := by sorry

end NUMINAMATH_CALUDE_dogsled_race_speed_l2671_267176


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2671_267105

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - i) / (3 + 4 * i) = 2 / 5 - 11 / 25 * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2671_267105


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2671_267187

/-- The number of games in a chess tournament where each player plays every other player twice -/
def num_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 12 players, where every player plays twice with each opponent, 
    the total number of games played is 264. -/
theorem chess_tournament_games : num_games 12 * 2 = 264 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2671_267187


namespace NUMINAMATH_CALUDE_arthur_walked_four_point_five_miles_l2671_267197

/-- The distance Arthur walked in miles -/
def arthur_distance (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 4.5 miles -/
theorem arthur_walked_four_point_five_miles :
  arthur_distance 8 10 (1/4) = 4.5 := by
sorry

end NUMINAMATH_CALUDE_arthur_walked_four_point_five_miles_l2671_267197


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2671_267169

-- Define the sets P and Q
def P : Set ℝ := {-1, 0, Real.sqrt 2}
def Q : Set ℝ := {y | ∃ θ : ℝ, y = Real.sin θ}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2671_267169


namespace NUMINAMATH_CALUDE_line_ellipse_no_intersection_l2671_267186

/-- Given a line y = 2x + b and an ellipse x^2/4 + y^2 = 1,
    if the line has no point in common with the ellipse,
    then b < -2√2 or b > 2√2 -/
theorem line_ellipse_no_intersection (b : ℝ) : 
  (∀ x y : ℝ, y = 2*x + b → x^2/4 + y^2 ≠ 1) → 
  (b < -2 * Real.sqrt 2 ∨ b > 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_line_ellipse_no_intersection_l2671_267186


namespace NUMINAMATH_CALUDE_sum_equation_proof_l2671_267195

theorem sum_equation_proof (N : ℕ) : 
  985 + 987 + 989 + 991 + 993 + 995 + 997 + 999 = 8000 - N → N = 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_proof_l2671_267195


namespace NUMINAMATH_CALUDE_product_xy_equals_four_l2671_267109

-- Define variables
variable (a b x y : ℕ)

-- State the theorem
theorem product_xy_equals_four
  (h1 : x = a)
  (h2 : y = b)
  (h3 : a + a = b * a)
  (h4 : y = a)
  (h5 : a * a = a + a)
  (h6 : b = 3) :
  x * y = 4 := by
sorry

end NUMINAMATH_CALUDE_product_xy_equals_four_l2671_267109


namespace NUMINAMATH_CALUDE_square_perimeter_when_area_equals_diagonal_l2671_267145

theorem square_perimeter_when_area_equals_diagonal : 
  ∀ s : ℝ, s > 0 → s^2 = s * Real.sqrt 2 → 4 * s = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_when_area_equals_diagonal_l2671_267145


namespace NUMINAMATH_CALUDE_mary_picked_nine_lemons_l2671_267125

/-- The number of lemons picked by Sally -/
def sally_lemons : ℕ := 7

/-- The total number of lemons picked by Sally and Mary -/
def total_lemons : ℕ := 16

/-- The number of lemons picked by Mary -/
def mary_lemons : ℕ := total_lemons - sally_lemons

theorem mary_picked_nine_lemons : mary_lemons = 9 := by
  sorry

end NUMINAMATH_CALUDE_mary_picked_nine_lemons_l2671_267125


namespace NUMINAMATH_CALUDE_problem_solution_l2671_267141

theorem problem_solution (a b : ℚ) 
  (h1 : 7 * a + 3 * b = 0) 
  (h2 : b - 4 = a) : 
  9 * b = 126 / 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2671_267141


namespace NUMINAMATH_CALUDE_employee_share_l2671_267133

def total_profit : ℝ := 50
def num_employees : ℕ := 9
def self_percentage : ℝ := 0.1

theorem employee_share : 
  (total_profit - self_percentage * total_profit) / num_employees = 5 := by
sorry

end NUMINAMATH_CALUDE_employee_share_l2671_267133


namespace NUMINAMATH_CALUDE_bisection_method_accuracy_l2671_267199

theorem bisection_method_accuracy (f : ℝ → ℝ) (x₀ : ℝ) :
  ContinuousOn f (Set.Ioi 0) →
  Irrational x₀ →
  x₀ ∈ Set.Ioo 2 3 →
  f x₀ = 0 →
  ∃ (a b : ℝ), a < x₀ ∧ x₀ < b ∧ b - a ≤ 1 / 2^9 ∧ b - a > 1 / 2^8 := by
  sorry

end NUMINAMATH_CALUDE_bisection_method_accuracy_l2671_267199
