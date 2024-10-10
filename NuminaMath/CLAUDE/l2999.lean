import Mathlib

namespace only_C_not_like_terms_l2999_299919

-- Define what it means for two terms to be like terms
def are_like_terms (term1 term2 : ℕ → ℕ → ℝ) : Prop :=
  ∀ a b, ∃ k, term1 a b = k * term2 a b ∨ term2 a b = k * term1 a b

-- Define the terms from the problem
def term_A1 (_ _ : ℕ) : ℝ := -2
def term_A2 (_ _ : ℕ) : ℝ := 12

def term_B1 (a b : ℕ) : ℝ := -2 * a^2 * b
def term_B2 (a b : ℕ) : ℝ := a^2 * b

def term_C1 (m _ : ℕ) : ℝ := 2 * m
def term_C2 (_ n : ℕ) : ℝ := 2 * n

def term_D1 (x y : ℕ) : ℝ := -1 * x^2 * y^2
def term_D2 (x y : ℕ) : ℝ := 12 * x^2 * y^2

-- Theorem stating that only C is not like terms
theorem only_C_not_like_terms :
  are_like_terms term_A1 term_A2 ∧
  are_like_terms term_B1 term_B2 ∧
  ¬(are_like_terms term_C1 term_C2) ∧
  are_like_terms term_D1 term_D2 :=
sorry

end only_C_not_like_terms_l2999_299919


namespace actual_distance_traveled_l2999_299989

theorem actual_distance_traveled (speed : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  speed = 10 →
  faster_speed = 20 →
  extra_distance = 40 →
  (∃ (time : ℝ), speed * time = faster_speed * time - extra_distance) →
  speed * (extra_distance / (faster_speed - speed)) = 40 :=
by sorry

end actual_distance_traveled_l2999_299989


namespace add_4500_seconds_to_10_45_00_l2999_299972

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The starting time 10:45:00 -/
def startTime : Time :=
  { hours := 10, minutes := 45, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 4500

/-- The expected end time 12:00:00 -/
def endTime : Time :=
  { hours := 12, minutes := 0, seconds := 0 }

theorem add_4500_seconds_to_10_45_00 :
  addSeconds startTime secondsToAdd = endTime := by
  sorry

end add_4500_seconds_to_10_45_00_l2999_299972


namespace shooting_competition_stability_l2999_299924

/-- Represents a participant in the shooting competition -/
structure Participant where
  name : String
  variance : ℝ

/-- Defines when a participant has more stable performance -/
def more_stable (p1 p2 : Participant) : Prop :=
  p1.variance < p2.variance

theorem shooting_competition_stability :
  let A : Participant := ⟨"A", 3⟩
  let B : Participant := ⟨"B", 1.2⟩
  more_stable B A := by
  sorry

end shooting_competition_stability_l2999_299924


namespace triangle_property_l2999_299936

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  a = 3 →
  b = 2 * Real.sqrt 6 →
  B = 2 * A →
  0 < A →
  A < π →
  0 < B →
  B < π →
  0 < C →
  C < π →
  A + B + C = π →
  a = 2 * (Real.sin (A / 2)) * (Real.sin (C / 2)) / Real.sin B →
  b = 2 * (Real.sin (B / 2)) * (Real.sin (C / 2)) / Real.sin A →
  c = 2 * (Real.sin (A / 2)) * (Real.sin (B / 2)) / Real.sin C →
  Real.cos A = Real.sqrt 6 / 3 ∧ c = 5 := by
  sorry

end triangle_property_l2999_299936


namespace workers_wage_increase_l2999_299960

theorem workers_wage_increase (original_wage new_wage : ℝ) : 
  (original_wage * 1.5 = new_wage) → 
  (new_wage = 51) → 
  (original_wage = 34) := by
sorry

end workers_wage_increase_l2999_299960


namespace system_solution_l2999_299962

theorem system_solution :
  ∃ (x y : ℚ), 
    (4 * x - 6 * y = -2) ∧ 
    (5 * x + 3 * y = 13/2) ∧ 
    (x = 7/22) ∧ 
    (y = 6/11) := by
  sorry

end system_solution_l2999_299962


namespace milk_water_ratio_problem_l2999_299917

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- The total volume of a mixture -/
def Mixture.volume (m : Mixture) : ℚ := m.milk + m.water

/-- The ratio of milk to water in a mixture -/
def Mixture.ratio (m : Mixture) : ℚ := m.milk / m.water

theorem milk_water_ratio_problem (m1 m2 : Mixture) :
  m1.volume = m2.volume →
  m1.ratio = 7/2 →
  (Mixture.mk (m1.milk + m2.milk) (m1.water + m2.water)).ratio = 5 →
  m2.ratio = 8 := by
  sorry

end milk_water_ratio_problem_l2999_299917


namespace problem_solution_l2999_299911

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.sin y = 2008)
  (h2 : x + 2008 * Real.cos y = 2007)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := by
  sorry

end problem_solution_l2999_299911


namespace beetle_distance_theorem_l2999_299997

def beetle_crawl (start : ℤ) (stop1 : ℤ) (stop2 : ℤ) : ℕ :=
  (Int.natAbs (stop1 - start)) + (Int.natAbs (stop2 - stop1))

theorem beetle_distance_theorem :
  beetle_crawl 3 (-5) 7 = 20 := by
  sorry

end beetle_distance_theorem_l2999_299997


namespace product_equals_square_l2999_299920

theorem product_equals_square : 10 * 9.99 * 0.999 * 100 = (99.9 : ℝ)^2 := by
  sorry

end product_equals_square_l2999_299920


namespace adlai_chickens_l2999_299953

def num_dogs : ℕ := 2
def total_legs : ℕ := 10
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

theorem adlai_chickens :
  (total_legs - num_dogs * legs_per_dog) / legs_per_chicken = 1 := by
  sorry

end adlai_chickens_l2999_299953


namespace olympic_medals_l2999_299970

theorem olympic_medals (total gold silver bronze : ℕ) : 
  total = 89 → 
  gold + silver = 4 * bronze - 6 → 
  gold + silver + bronze = total → 
  bronze = 19 := by
sorry

end olympic_medals_l2999_299970


namespace smallest_positive_b_squared_l2999_299969

/-- Definition of circle u₁ -/
def u₁ (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 20*y - 75 = 0

/-- Definition of circle u₂ -/
def u₂ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 20*y + 175 = 0

/-- A circle is externally tangent to u₂ and internally tangent to u₁ -/
def is_tangent_circle (x y t : ℝ) : Prop :=
  t + 7 = Real.sqrt ((x - 4)^2 + (y - 10)^2) ∧
  11 - t = Real.sqrt ((x + 4)^2 + (y - 10)^2)

/-- The center of the tangent circle lies on the line y = bx -/
def center_on_line (x y b : ℝ) : Prop := y = b * x

/-- Main theorem: The smallest positive b satisfying the conditions has b² = 5/16 -/
theorem smallest_positive_b_squared (b : ℝ) :
  (∃ x y t, u₁ x y ∧ u₂ x y ∧ is_tangent_circle x y t ∧ center_on_line x y b) →
  (∀ b' : ℝ, 0 < b' → b' < b →
    ¬∃ x y t, u₁ x y ∧ u₂ x y ∧ is_tangent_circle x y t ∧ center_on_line x y b') →
  b^2 = 5/16 := by sorry

end smallest_positive_b_squared_l2999_299969


namespace not_prime_two_pow_plus_one_l2999_299901

theorem not_prime_two_pow_plus_one (n : ℕ) (d : ℕ) (h_odd : Odd d) (h_div : d ∣ n) :
  ¬ Nat.Prime (2^n + 1) := by
  sorry

end not_prime_two_pow_plus_one_l2999_299901


namespace range_of_a_for_C_subset_B_l2999_299987

def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem range_of_a_for_C_subset_B :
  {a : ℝ | C a ⊆ B} = {a : ℝ | 2 ≤ a ∧ a ≤ 8} := by sorry

end range_of_a_for_C_subset_B_l2999_299987


namespace nancy_files_distribution_l2999_299977

/-- Given the initial number of files, number of deleted files, and number of folders,
    calculate the number of files in each folder after distribution. -/
def filesPerFolder (initialFiles deletedFiles numFolders : ℕ) : ℕ :=
  (initialFiles - deletedFiles) / numFolders

/-- Prove that given 80 initial files, after deleting 31 files and distributing
    the remaining files equally into 7 folders, each folder contains 7 files. -/
theorem nancy_files_distribution :
  filesPerFolder 80 31 7 = 7 := by
  sorry

end nancy_files_distribution_l2999_299977


namespace certain_number_problem_l2999_299909

theorem certain_number_problem (x y : ℝ) : 
  0.12 / x * y = 12 ∧ x = 0.1 → y = 10 := by
  sorry

end certain_number_problem_l2999_299909


namespace banana_orange_equivalence_l2999_299967

/-- The value of one banana in terms of oranges -/
def banana_value : ℚ := 1

/-- The number of bananas that are worth as much as 12 oranges -/
def bananas_worth_12_oranges : ℚ := 12

/-- The fraction of 16 bananas that are worth as much as 12 oranges -/
def fraction_of_16_bananas : ℚ := 3/4

/-- The number of bananas we're considering in the question -/
def question_bananas : ℚ := 9

/-- The fraction of question_bananas we're considering -/
def fraction_of_question_bananas : ℚ := 2/3

theorem banana_orange_equivalence :
  fraction_of_16_bananas * 16 = bananas_worth_12_oranges →
  fraction_of_question_bananas * question_bananas * banana_value = 6 := by
  sorry

end banana_orange_equivalence_l2999_299967


namespace existence_of_nondivisible_power_l2999_299944

theorem existence_of_nondivisible_power (a b c : ℕ+) (h : Nat.gcd a b.val = 1 ∧ Nat.gcd (Nat.gcd a b.val) c.val = 1) :
  ∃ n : ℕ+, ∀ k : ℕ+, ¬(2^n.val ∣ a^k.val + b^k.val + c^k.val) :=
sorry

end existence_of_nondivisible_power_l2999_299944


namespace min_value_expression_min_value_attained_l2999_299986

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (3 * z) / (x + 2 * y) + (5 * x) / (2 * y + 3 * z) + (2 * y) / (3 * x + z) ≥ (3 : ℝ) / 4 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (3 * z) / (x + 2 * y) + (5 * x) / (2 * y + 3 * z) + (2 * y) / (3 * x + z) < (3 : ℝ) / 4 + ε :=
by sorry

end min_value_expression_min_value_attained_l2999_299986


namespace investment_sum_l2999_299988

/-- Proves that if a sum P invested at 18% p.a. for two years yields Rs. 504 more interest
    than if invested at 12% p.a. for the same period, then P = 4200. -/
theorem investment_sum (P : ℚ) : 
  (P * 18 * 2 / 100) - (P * 12 * 2 / 100) = 504 → P = 4200 := by
  sorry

end investment_sum_l2999_299988


namespace minimum_days_to_plant_trees_l2999_299925

def trees_planted (n : ℕ) : ℕ := 2^n - 1

theorem minimum_days_to_plant_trees : 
  (∀ k : ℕ, k < 7 → trees_planted k < 100) ∧ 
  trees_planted 7 ≥ 100 := by
sorry

end minimum_days_to_plant_trees_l2999_299925


namespace other_root_of_quadratic_l2999_299968

theorem other_root_of_quadratic (m : ℝ) : 
  (2 : ℝ)^2 + m * 2 - 6 = 0 → (-3 : ℝ)^2 + m * (-3) - 6 = 0 := by
  sorry

end other_root_of_quadratic_l2999_299968


namespace k_less_than_one_necessary_not_sufficient_l2999_299981

/-- For x in the open interval (0, π/2), "k < 1" is a necessary but not sufficient condition for k*sin(x)*cos(x) < x. -/
theorem k_less_than_one_necessary_not_sufficient :
  ∀ k : ℝ, (∃ x : ℝ, 0 < x ∧ x < π/2 ∧ k * Real.sin x * Real.cos x < x) →
  (k < 1 ∧ ∃ k' ≥ 1, ∃ x : ℝ, 0 < x ∧ x < π/2 ∧ k' * Real.sin x * Real.cos x < x) :=
by sorry

end k_less_than_one_necessary_not_sufficient_l2999_299981


namespace special_integer_count_l2999_299940

/-- Count of positive integers less than 100,000 with at most two different digits -/
def count_special_integers : ℕ :=
  let single_digit_count := 45
  let two_digit_count_no_zero := 1872
  let two_digit_count_with_zero := 234
  single_digit_count + two_digit_count_no_zero + two_digit_count_with_zero

/-- The count of positive integers less than 100,000 with at most two different digits is 2151 -/
theorem special_integer_count : count_special_integers = 2151 := by
  sorry

end special_integer_count_l2999_299940


namespace equidistant_points_on_line_l2999_299900

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 4 * x - 3 * y = 12

-- Define the equidistant condition
def equidistant (x y : ℝ) : Prop := abs x = abs y

-- Define quadrants
def quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0
def quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0
def quadrant_III (x y : ℝ) : Prop := x < 0 ∧ y < 0
def quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem equidistant_points_on_line :
  (∃ x y : ℝ, line_equation x y ∧ equidistant x y ∧ quadrant_I x y) ∧
  (∃ x y : ℝ, line_equation x y ∧ equidistant x y ∧ quadrant_IV x y) ∧
  (¬∃ x y : ℝ, line_equation x y ∧ equidistant x y ∧ quadrant_II x y) ∧
  (¬∃ x y : ℝ, line_equation x y ∧ equidistant x y ∧ quadrant_III x y) :=
sorry

end equidistant_points_on_line_l2999_299900


namespace total_dogs_l2999_299980

/-- Represents the properties of dogs in a kennel -/
structure Kennel where
  longFurred : Nat
  brown : Nat
  longFurredBrown : Nat
  neitherLongFurredNorBrown : Nat

/-- Theorem stating the total number of dogs in the kennel -/
theorem total_dogs (k : Kennel) 
  (h1 : k.longFurred = 26)
  (h2 : k.brown = 22)
  (h3 : k.longFurredBrown = 11)
  (h4 : k.neitherLongFurredNorBrown = 8) :
  k.longFurred + k.brown - k.longFurredBrown + k.neitherLongFurredNorBrown = 45 := by
  sorry

#check total_dogs

end total_dogs_l2999_299980


namespace hayley_sticker_distribution_l2999_299905

theorem hayley_sticker_distribution (total_stickers : ℕ) (num_friends : ℕ) 
  (h1 : total_stickers = 72) 
  (h2 : num_friends = 9) 
  (h3 : total_stickers % num_friends = 0) : 
  total_stickers / num_friends = 8 := by
sorry

end hayley_sticker_distribution_l2999_299905


namespace stratified_sampling_sample_size_l2999_299956

theorem stratified_sampling_sample_size 
  (high_school_students : ℕ) 
  (junior_high_students : ℕ) 
  (high_school_sample : ℕ) 
  (h1 : high_school_students = 3500)
  (h2 : junior_high_students = 1500)
  (h3 : high_school_sample = 70) :
  let total_students := high_school_students + junior_high_students
  let sample_proportion := high_school_sample / high_school_students
  let total_sample_size := total_students * sample_proportion
  total_sample_size = 100 := by sorry

end stratified_sampling_sample_size_l2999_299956


namespace cylinder_radius_proof_l2999_299999

theorem cylinder_radius_proof (r : ℝ) : 
  (r > 0) →                            -- r is positive (radius)
  (2 > 0) →                            -- original height is positive
  (π * (r + 6)^2 * 2 = π * r^2 * 8) →  -- volumes are equal when increased
  r = 6 := by
sorry

end cylinder_radius_proof_l2999_299999


namespace quadratic_inequality_solution_l2999_299961

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) → a = -2 :=
by sorry

end quadratic_inequality_solution_l2999_299961


namespace expected_collectors_is_120_l2999_299995

-- Define the number of customers
def num_customers : ℕ := 3000

-- Define the probability of a customer collecting a prize
def prob_collect : ℝ := 0.04

-- Define the expected number of prize collectors
def expected_collectors : ℝ := num_customers * prob_collect

-- Theorem statement
theorem expected_collectors_is_120 : expected_collectors = 120 := by
  sorry

end expected_collectors_is_120_l2999_299995


namespace cube_sum_square_not_prime_product_l2999_299922

theorem cube_sum_square_not_prime_product (a b : ℕ+) (h : ∃ (u : ℕ), (a.val ^ 3 + b.val ^ 3 : ℕ) = u ^ 2) :
  ¬∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ a.val + b.val = p * q :=
by sorry

end cube_sum_square_not_prime_product_l2999_299922


namespace ratio_x_to_y_l2999_299908

theorem ratio_x_to_y (x y : ℝ) (h : 0.1 * x = 0.2 * y) : x / y = 2 := by
  sorry

end ratio_x_to_y_l2999_299908


namespace division_equality_l2999_299929

theorem division_equality (h : 29.94 / 1.45 = 17.3) : 2994 / 14.5 = 173 := by
  sorry

end division_equality_l2999_299929


namespace A_div_B_eq_37_l2999_299933

-- Define the series A
def A : ℝ := sorry

-- Define the series B
def B : ℝ := sorry

-- Theorem stating the relationship between A and B
theorem A_div_B_eq_37 : A / B = 37 := by sorry

end A_div_B_eq_37_l2999_299933


namespace arithmetic_geometric_sequence_l2999_299993

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  (∀ n : ℕ, a (n + 1) = a n + d) →
  (∃ r : ℝ, r ≠ 0 ∧ a 5 = r * a 1 ∧ a 17 = r * a 5) →
  (∃ r : ℝ, r ≠ 0 ∧ a 5 = r * a 1 ∧ a 17 = r * a 5 ∧ r = 3) :=
by sorry

end arithmetic_geometric_sequence_l2999_299993


namespace intersection_subset_iff_m_eq_two_l2999_299918

/-- Sets A, B, and C as defined in the problem -/
def A : Set ℝ := {x | -4 < x ∧ x < 2}
def B : Set ℝ := {x | x > 1 ∨ x < -5}
def C (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m}

/-- Theorem stating that A ∩ B ⊆ C(m) if and only if m = 2 -/
theorem intersection_subset_iff_m_eq_two :
  ∀ m : ℝ, (A ∩ B) ⊆ C m ↔ m = 2 := by sorry

end intersection_subset_iff_m_eq_two_l2999_299918


namespace fair_coin_heads_then_tails_l2999_299954

/-- The probability of getting heads on a single flip of a fair coin. -/
def prob_heads : ℚ := 1/2

/-- The probability of getting tails on a single flip of a fair coin. -/
def prob_tails : ℚ := 1/2

/-- The probability of getting heads on the first flip and tails on the second flip
    of a fair coin. -/
def prob_heads_then_tails : ℚ := prob_heads * prob_tails

theorem fair_coin_heads_then_tails :
  prob_heads_then_tails = 1/4 := by sorry

end fair_coin_heads_then_tails_l2999_299954


namespace solution_set_theorem_range_of_a_theorem_l2999_299910

/-- Function f(x) = |x - 1| + |x + 2| -/
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

/-- Function g(x) = |x + 1| - |x - a| + a -/
def g (a x : ℝ) : ℝ := |x + 1| - |x - a| + a

/-- The solution set of f(x) + g(x) < 6 when a = 1 is (-4, 1) -/
theorem solution_set_theorem :
  {x : ℝ | f x + g 1 x < 6} = Set.Ioo (-4) 1 := by sorry

/-- For any real numbers x₁ and x₂, f(x₁) ≥ g(x₂) if and only if a ∈ (-∞, 1] -/
theorem range_of_a_theorem :
  ∀ (a : ℝ), (∀ (x₁ x₂ : ℝ), f x₁ ≥ g a x₂) ↔ a ∈ Set.Iic 1 := by sorry

end solution_set_theorem_range_of_a_theorem_l2999_299910


namespace pacos_countertop_marble_weight_l2999_299903

theorem pacos_countertop_marble_weight : 
  let weights : List ℝ := [0.33, 0.33, 0.08, 0.25, 0.02, 0.12, 0.15]
  weights.sum = 1.28 := by
  sorry

end pacos_countertop_marble_weight_l2999_299903


namespace jellybean_problem_l2999_299974

theorem jellybean_problem (J : ℕ) : 
  J - 15 + 5 - 4 = 23 → J = 33 := by
  sorry

end jellybean_problem_l2999_299974


namespace passenger_ticket_probability_l2999_299902

/-- The probability of a passenger getting a ticket at three counters -/
theorem passenger_ticket_probability
  (p₁ p₂ p₃ p₄ p₅ p₆ : ℝ)
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1)
  (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1)
  (h₃ : 0 ≤ p₃ ∧ p₃ ≤ 1)
  (h₄ : 0 ≤ p₄ ∧ p₄ ≤ 1)
  (h₅ : 0 ≤ p₅ ∧ p₅ ≤ 1)
  (h₆ : 0 ≤ p₆ ∧ p₆ ≤ 1)
  (h_sum : p₁ + p₂ + p₃ = 1) :
  let prob_get_ticket := p₁ * (1 - p₄) + p₂ * (1 - p₅) + p₃ * (1 - p₆)
  0 ≤ prob_get_ticket ∧ prob_get_ticket ≤ 1 :=
by sorry

end passenger_ticket_probability_l2999_299902


namespace travelers_checks_denomination_l2999_299942

theorem travelers_checks_denomination 
  (total_checks : ℕ) 
  (total_worth : ℚ) 
  (spent_checks : ℕ) 
  (remaining_average : ℚ) 
  (h1 : total_checks = 30)
  (h2 : total_worth = 1800)
  (h3 : spent_checks = 6)
  (h4 : remaining_average = 62.5)
  (h5 : (total_checks - spent_checks : ℚ) * remaining_average + spent_checks * x = total_worth) :
  x = 50 := by
  sorry

end travelers_checks_denomination_l2999_299942


namespace system_solution_l2999_299943

theorem system_solution (c₁ c₂ c₃ : ℝ) :
  let x₁ := -2 * c₁ - c₂ + 2
  let x₂ := c₁ + 1
  let x₃ := c₂ + 3
  let x₄ := 2 * c₂ + 2 * c₃ - 2
  let x₅ := c₃ + 1
  (x₁ + 2 * x₂ - x₃ + x₄ - 2 * x₅ = -3) ∧
  (x₁ + 2 * x₂ + 3 * x₃ - x₄ + 2 * x₅ = 17) ∧
  (2 * x₁ + 4 * x₂ + 2 * x₃ = 14) :=
by sorry

end system_solution_l2999_299943


namespace inverse_of_B_cubed_l2999_299990

def B_inverse : Matrix (Fin 2) (Fin 2) ℝ :=
  !![3, 4;
    -2, -3]

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = B_inverse) : 
  (B^3)⁻¹ = B_inverse := by
  sorry

end inverse_of_B_cubed_l2999_299990


namespace race_car_count_l2999_299947

theorem race_car_count (p_x p_y p_z p_combined : ℚ) (h1 : p_x = 1 / 7)
    (h2 : p_y = 1 / 3) (h3 : p_z = 1 / 5)
    (h4 : p_combined = p_x + p_y + p_z)
    (h5 : p_combined = 71 / 105) : ∃ n : ℕ, n = 105 ∧ p_x = 1 / n := by
  sorry

end race_car_count_l2999_299947


namespace line_length_difference_l2999_299979

theorem line_length_difference : 
  let white_line : ℝ := 7.67
  let blue_line : ℝ := 3.33
  white_line - blue_line = 4.34 := by
sorry

end line_length_difference_l2999_299979


namespace smallest_positive_integer_2010m_44000n_l2999_299945

theorem smallest_positive_integer_2010m_44000n : 
  (∃ (k : ℕ), k > 0 ∧ ∃ (m n : ℤ), k = 2010 * m + 44000 * n) ∧ 
  (∀ (k : ℕ), k > 0 → (∃ (m n : ℤ), k = 2010 * m + 44000 * n) → k ≥ 10) ∧
  (∃ (m n : ℤ), 10 = 2010 * m + 44000 * n) :=
by sorry

end smallest_positive_integer_2010m_44000n_l2999_299945


namespace total_cost_is_240000_l2999_299991

/-- The total cost of three necklaces and a set of earrings -/
def total_cost (necklace_price : ℕ) : ℕ :=
  3 * necklace_price + 3 * necklace_price

/-- Proof that the total cost is $240,000 -/
theorem total_cost_is_240000 :
  total_cost 40000 = 240000 := by
  sorry

#eval total_cost 40000

end total_cost_is_240000_l2999_299991


namespace generating_function_value_l2999_299994

/-- The generating function of two linear functions -/
def generating_function (m n x : ℝ) : ℝ := m * (x + 1) + n * (2 * x)

/-- Theorem: The generating function equals 2 when x = 1 and m + n = 1 -/
theorem generating_function_value : 
  ∀ m n : ℝ, m + n = 1 → generating_function m n 1 = 2 := by
  sorry

end generating_function_value_l2999_299994


namespace staircase_perimeter_l2999_299941

/-- A staircase-shaped region with right angles -/
structure StaircaseRegion where
  /-- Number of congruent sides -/
  num_sides : ℕ
  /-- Length of each congruent side -/
  side_length : ℝ
  /-- Area of the region -/
  area : ℝ

/-- Calculates the perimeter of a StaircaseRegion -/
def perimeter (s : StaircaseRegion) : ℝ :=
  sorry

theorem staircase_perimeter (s : StaircaseRegion) 
  (h1 : s.num_sides = 12)
  (h2 : s.side_length = 1)
  (h3 : s.area = 89) :
  perimeter s = 43 := by
  sorry

end staircase_perimeter_l2999_299941


namespace point_on_y_axis_l2999_299975

/-- A point lies on the y-axis if and only if its x-coordinate is 0 -/
def lies_on_y_axis (x y : ℝ) : Prop := x = 0

/-- The theorem states that if the point (a+1, a-1) lies on the y-axis, then a = -1 -/
theorem point_on_y_axis (a : ℝ) : lies_on_y_axis (a + 1) (a - 1) → a = -1 := by
  sorry

end point_on_y_axis_l2999_299975


namespace sum_even_positive_lt_100_l2999_299966

/-- The sum of all even, positive integers less than 100 is 2450 -/
theorem sum_even_positive_lt_100 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ 0 < n) (Finset.range 100)).sum id = 2450 := by
  sorry

end sum_even_positive_lt_100_l2999_299966


namespace nth_monomial_formula_l2999_299927

/-- Represents the coefficient of the nth monomial in the sequence -/
def coefficient (n : ℕ) : ℕ := 3 * n + 2

/-- Represents the exponent of 'a' in the nth monomial of the sequence -/
def exponent (n : ℕ) : ℕ := n

/-- Represents the nth monomial in the sequence as a function of 'a' -/
def nthMonomial (n : ℕ) (a : ℝ) : ℝ := (coefficient n : ℝ) * a ^ (exponent n)

/-- The sequence of monomials follows the pattern 5a, 8a^2, 11a^3, 14a^4, ... -/
axiom sequence_pattern (n : ℕ) (a : ℝ) : 
  n ≥ 1 → nthMonomial n a = (3 * n + 2 : ℝ) * a ^ n

/-- Theorem: The nth monomial in the sequence is equal to (3n+2)a^n -/
theorem nth_monomial_formula (n : ℕ) (a : ℝ) : 
  n ≥ 1 → nthMonomial n a = (3 * n + 2 : ℝ) * a ^ n := by
  sorry

end nth_monomial_formula_l2999_299927


namespace lucy_sold_29_packs_l2999_299984

/-- The number of packs of cookies sold by Robyn -/
def robyn_packs : ℕ := 47

/-- The total number of packs of cookies sold by Robyn and Lucy -/
def total_packs : ℕ := 76

/-- The number of packs of cookies sold by Lucy -/
def lucy_packs : ℕ := total_packs - robyn_packs

theorem lucy_sold_29_packs : lucy_packs = 29 := by
  sorry

end lucy_sold_29_packs_l2999_299984


namespace cylinder_radius_calculation_l2999_299939

theorem cylinder_radius_calculation (shadow_length : ℝ) (flagpole_height : ℝ) (flagpole_shadow : ℝ) 
  (h1 : shadow_length = 12)
  (h2 : flagpole_height = 1.5)
  (h3 : flagpole_shadow = 3)
  (h4 : flagpole_shadow > 0) -- To avoid division by zero
  : ∃ (radius : ℝ), radius = shadow_length * (flagpole_height / flagpole_shadow) ∧ radius = 6 := by
  sorry

end cylinder_radius_calculation_l2999_299939


namespace simplify_expression_l2999_299934

theorem simplify_expression (x : ℝ) : 
  3*x + 6*x + 9*x + 12*x + 15*x + 18 + 24 = 45*x + 42 := by
  sorry

end simplify_expression_l2999_299934


namespace total_selling_price_proof_l2999_299998

def calculate_selling_price (cost : ℝ) (profit_percentage : ℝ) : ℝ :=
  cost * (1 + profit_percentage)

theorem total_selling_price_proof :
  let cost_prices : List ℝ := [280, 350, 500]
  let profit_percentages : List ℝ := [0.30, 0.45, 0.25]
  let selling_prices := List.zipWith calculate_selling_price cost_prices profit_percentages
  List.sum selling_prices = 1496.50 := by
sorry

end total_selling_price_proof_l2999_299998


namespace greatest_NPM_value_l2999_299971

theorem greatest_NPM_value : ∀ M N P : ℕ,
  (M ≥ 1 ∧ M ≤ 9) →  -- M is a one-digit integer
  (N ≥ 1 ∧ N ≤ 9) →  -- N is a one-digit integer
  (P ≥ 0 ∧ P ≤ 9) →  -- P is a one-digit integer
  (10 * M + M) * M = 100 * N + 10 * P + M →  -- MM * M = NPM
  100 * N + 10 * P + M ≤ 396 :=
by sorry

end greatest_NPM_value_l2999_299971


namespace batsman_average_after_17th_inning_l2999_299948

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (bp : BatsmanPerformance) (runsInLastInning : ℕ) : ℚ :=
  (bp.totalRuns + runsInLastInning) / (bp.innings + 1)

/-- Theorem: If a batsman's average increases by 2 after scoring 50 in the 17th inning, 
    then the new average is 18 -/
theorem batsman_average_after_17th_inning 
  (bp : BatsmanPerformance)
  (h1 : bp.innings = 16)
  (h2 : newAverage bp 50 = bp.average + 2)
  : newAverage bp 50 = 18 := by
  sorry


end batsman_average_after_17th_inning_l2999_299948


namespace parabola_directrix_l2999_299930

/-- Given a parabola y² = 2px and a point M(1, m) on the parabola,
    with the distance from M to its focus being 5,
    prove that the equation of the directrix is x = -4 -/
theorem parabola_directrix (p : ℝ) (m : ℝ) : 
  m^2 = 2*p  -- M(1, m) is on the parabola y² = 2px
  → (1 - p/2)^2 + m^2 = 25  -- Distance from M to focus is 5
  → -p/2 = -4  -- Equation of directrix is x = -p/2
  := by sorry

end parabola_directrix_l2999_299930


namespace distance_AB_is_300_l2999_299958

/-- The distance between points A and B in meters -/
def distance_AB : ℝ := 300

/-- The speed ratio of Person A to Person B -/
def speed_ratio : ℝ := 2

/-- The distance Person B is from point B when Person A reaches B -/
def distance_B_to_B : ℝ := 100

/-- The distance from B where Person A and B meet when A returns -/
def meeting_distance : ℝ := 60

/-- Theorem stating the distance between A and B is 300 meters -/
theorem distance_AB_is_300 :
  distance_AB = 300 ∧
  speed_ratio = 2 ∧
  distance_B_to_B = 100 ∧
  meeting_distance = 60 →
  distance_AB = 300 := by
  sorry

#check distance_AB_is_300

end distance_AB_is_300_l2999_299958


namespace doughnut_cost_theorem_l2999_299921

/-- Calculate the total cost of doughnuts for a class --/
theorem doughnut_cost_theorem (total_students : ℕ) 
  (chocolate_students : ℕ) (glazed_students : ℕ) (maple_students : ℕ) (strawberry_students : ℕ)
  (chocolate_cost : ℚ) (glazed_cost : ℚ) (maple_cost : ℚ) (strawberry_cost : ℚ) :
  total_students = 25 →
  chocolate_students = 10 →
  glazed_students = 8 →
  maple_students = 5 →
  strawberry_students = 2 →
  chocolate_cost = 2 →
  glazed_cost = 1 →
  maple_cost = (3/2) →
  strawberry_cost = (5/2) →
  (chocolate_students : ℚ) * chocolate_cost + 
  (glazed_students : ℚ) * glazed_cost + 
  (maple_students : ℚ) * maple_cost + 
  (strawberry_students : ℚ) * strawberry_cost = (81/2) := by
  sorry

#eval (81/2 : ℚ)

end doughnut_cost_theorem_l2999_299921


namespace triangle_rectangle_ratio_and_f_properties_l2999_299965

/-- Triangle ABC with base AB and height 1 -/
structure Triangle :=
  (base : ℝ)
  (height : ℝ)
  (height_eq_one : height = 1)

/-- Rectangle PQRS with width PQ and height 1 -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)
  (height_eq_one : height = 1)

/-- The function f(x) representing the height of rectangle PQNM -/
def f (x : ℝ) : ℝ := 2 * x - x^2

theorem triangle_rectangle_ratio_and_f_properties 
  (triangle : Triangle) (rectangle : Rectangle) :
  (triangle.base / rectangle.width = 2) ∧
  (triangle.base * triangle.height / 2 = rectangle.width * rectangle.height) ∧
  (f (1/2) = 3/4) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 
    f x = 2 * x - x^2) := by
  sorry

end triangle_rectangle_ratio_and_f_properties_l2999_299965


namespace least_positive_angle_theorem_l2999_299931

theorem least_positive_angle_theorem : ∃ θ : ℝ,
  θ > 0 ∧
  θ ≤ 90 ∧
  Real.cos (10 * π / 180) = Real.sin (30 * π / 180) + Real.sin (θ * π / 180) ∧
  ∀ φ : ℝ, φ > 0 ∧ φ < θ →
    Real.cos (10 * π / 180) ≠ Real.sin (30 * π / 180) + Real.sin (φ * π / 180) ∧
  θ = 80 := by
  sorry

end least_positive_angle_theorem_l2999_299931


namespace max_value_of_f_l2999_299996

/-- Given a function f(x) = (x^2 - 4)(x - a) where a is a real number and f'(1) = 0,
    the maximum value of f(x) on the interval [-2, 2] is 50/27. -/
theorem max_value_of_f (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = (x^2 - 4) * (x - a)) 
    (h2 : deriv f 1 = 0) : 
    ∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y ≤ f x ∧ f x = 50/27 := by
  sorry

end max_value_of_f_l2999_299996


namespace complex_fraction_simplification_l2999_299928

theorem complex_fraction_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^(4/3))^(3/(2*5)) / (a^4)^(3/5) / ((a * (a^2 * b)^(1/3))^(1/2))^4 * ((a * b^(1/2))^(1/4))^6 = 1 / (a^2 * b)^(1/12) :=
by sorry

end complex_fraction_simplification_l2999_299928


namespace max_value_implies_m_l2999_299951

/-- The function f(x) = -x³ + 3x² + 9x + m has a maximum value of 20 on the interval [-2, 2] -/
def f (x m : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + m

/-- The maximum value of f(x) on the interval [-2, 2] is 20 -/
def has_max_20 (m : ℝ) : Prop :=
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 2 ∧
  f x m = 20 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-2) 2 → f y m ≤ 20

theorem max_value_implies_m (m : ℝ) :
  has_max_20 m → m = -2 :=
by sorry

end max_value_implies_m_l2999_299951


namespace absolute_value_difference_l2999_299932

theorem absolute_value_difference (x p : ℝ) (h1 : |x - 5| = p) (h2 : x > 5) : x - p = 5 := by
  sorry

end absolute_value_difference_l2999_299932


namespace angle_terminal_side_l2999_299937

/-- Two angles have the same terminal side if they differ by a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α - β = k * 360

/-- The expression for angles with the same terminal side as -463° -/
def angle_expression (k : ℤ) : ℝ := k * 360 + 257

theorem angle_terminal_side :
  ∀ k : ℤ, same_terminal_side (angle_expression k) (-463) :=
by
  sorry

end angle_terminal_side_l2999_299937


namespace candy_distribution_l2999_299973

theorem candy_distribution (total_candy : ℕ) (num_students : ℕ) (pieces_per_student : ℕ) : 
  total_candy = 344 →
  num_students = 43 →
  total_candy = num_students * pieces_per_student →
  pieces_per_student = 8 := by
  sorry

end candy_distribution_l2999_299973


namespace triangle_problem_l2999_299913

theorem triangle_problem (A B C : Real) (a b c : Real) :
  C = π / 3 →
  b = 8 →
  (1 / 2) * a * b * Real.sin C = 10 * Real.sqrt 3 →
  c = 7 ∧ Real.cos (B - C) = 13 / 14 := by
  sorry

end triangle_problem_l2999_299913


namespace expression_factorization_l2999_299992

theorem expression_factorization (x : ℝ) : 
  (12 * x^6 + 30 * x^4 - 6) - (2 * x^6 - 4 * x^4 - 6) = 2 * x^4 * (5 * x^2 + 17) := by
  sorry

end expression_factorization_l2999_299992


namespace binary_multiplication_theorem_l2999_299983

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec toBits (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBits (m / 2)
  toBits n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [false, true, true]        -- 110₂
  let product := [false, true, true, true, true, false, true]  -- 1011110₂
  binaryToNat a * binaryToNat b = binaryToNat product :=
by sorry

end binary_multiplication_theorem_l2999_299983


namespace A_neither_sufficient_nor_necessary_l2999_299906

-- Define propositions A and B
def proposition_A (a b : ℝ) : Prop := a + b ≠ 4
def proposition_B (a b : ℝ) : Prop := a ≠ 1 ∧ b ≠ 3

-- Theorem stating that A is neither sufficient nor necessary for B
theorem A_neither_sufficient_nor_necessary :
  ¬(∀ a b : ℝ, proposition_A a b → proposition_B a b) ∧
  ¬(∀ a b : ℝ, proposition_B a b → proposition_A a b) :=
sorry

end A_neither_sufficient_nor_necessary_l2999_299906


namespace option_c_is_linear_system_l2999_299985

-- Define what a linear equation in two variables is
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

-- Define a system of two equations
def is_system_of_two_equations (f g : ℝ → ℝ → ℝ) : Prop :=
  true  -- This is always true as we're given two equations

-- Define the specific equations from Option C
def eq1 (x y : ℝ) : ℝ := x + y - 5
def eq2 (x y : ℝ) : ℝ := 3 * x - 4 * y - 12

-- Theorem stating that eq1 and eq2 form a system of two linear equations
theorem option_c_is_linear_system :
  is_linear_equation eq1 ∧ is_linear_equation eq2 ∧ is_system_of_two_equations eq1 eq2 :=
sorry

end option_c_is_linear_system_l2999_299985


namespace triangle_side_length_l2999_299938

/-- In a triangle DEF, given angle E, side DE, and side DF, prove the length of EF --/
theorem triangle_side_length (E D F : ℝ) (hE : E = 45 * π / 180) 
  (hDE : D = 100) (hDF : F = 100 * Real.sqrt 2) : 
  ∃ (EF : ℝ), abs (EF - Real.sqrt (10000 + 5176.4)) < 0.001 := by
  sorry

end triangle_side_length_l2999_299938


namespace xiaohua_school_time_l2999_299904

-- Define a custom type for time
structure SchoolTime where
  hours : ℕ
  minutes : ℕ
  is_pm : Bool

-- Define a function to calculate the time difference in minutes
def time_diff (t1 t2 : SchoolTime) : ℕ :=
  let total_minutes1 := t1.hours * 60 + t1.minutes + (if t1.is_pm then 12 * 60 else 0)
  let total_minutes2 := t2.hours * 60 + t2.minutes + (if t2.is_pm then 12 * 60 else 0)
  total_minutes2 - total_minutes1

-- Define Xiaohua's schedule
def morning_arrival : SchoolTime := ⟨7, 20, false⟩
def morning_departure : SchoolTime := ⟨11, 45, false⟩
def afternoon_arrival : SchoolTime := ⟨1, 50, true⟩
def afternoon_departure : SchoolTime := ⟨5, 15, true⟩

-- Theorem statement
theorem xiaohua_school_time :
  time_diff morning_arrival morning_departure +
  time_diff afternoon_arrival afternoon_departure = 7 * 60 + 50 := by
  sorry

end xiaohua_school_time_l2999_299904


namespace min_m_for_perfect_fourth_power_min_m_value_exact_min_m_l2999_299964

theorem min_m_for_perfect_fourth_power (m n : ℕ+) (h : 24 * m = n ^ 4) : 
  ∀ k : ℕ+, 24 * k = (some_nat : ℕ) ^ 4 → m ≤ k := by
  sorry

theorem min_m_value (m n : ℕ+) (h : 24 * m = n ^ 4) : m ≥ 54 := by
  sorry

theorem exact_min_m (m n : ℕ+) (h : 24 * m = n ^ 4) : 
  (∃ k : ℕ+, 24 * 54 = k ^ 4) ∧ m ≥ 54 := by
  sorry

end min_m_for_perfect_fourth_power_min_m_value_exact_min_m_l2999_299964


namespace two_piggy_banks_value_l2999_299923

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "dime" => 10
  | _ => 0

/-- Represents the number of coins in a piggy bank -/
def coins_in_bank (coin : String) : ℕ :=
  match coin with
  | "penny" => 100
  | "dime" => 50
  | _ => 0

/-- Calculates the total value in cents for a single piggy bank -/
def piggy_bank_value : ℕ :=
  coin_value "penny" * coins_in_bank "penny" +
  coin_value "dime" * coins_in_bank "dime"

/-- Calculates the total value in dollars for two piggy banks -/
def total_value : ℚ :=
  (2 * piggy_bank_value : ℚ) / 100

theorem two_piggy_banks_value : total_value = 12 := by
  sorry

end two_piggy_banks_value_l2999_299923


namespace problem_statement_l2999_299907

theorem problem_statement (m : ℤ) : 
  2^2000 - 3 * 2^1998 + 5 * 2^1996 - 2^1995 = m * 2^1995 → m = 17 := by
  sorry

end problem_statement_l2999_299907


namespace liberty_middle_school_math_competition_l2999_299914

theorem liberty_middle_school_math_competition (sixth_graders seventh_graders : ℕ) : 
  (3 * sixth_graders = 7 * seventh_graders) →
  (sixth_graders + seventh_graders = 140) →
  sixth_graders = 61 := by
  sorry

end liberty_middle_school_math_competition_l2999_299914


namespace zeta_power_sum_l2999_299976

theorem zeta_power_sum (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 20) :
  ζ₁^5 + ζ₂^5 + ζ₃^5 = 54 := by
  sorry

end zeta_power_sum_l2999_299976


namespace complex_number_quadrant_l2999_299955

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (1 + Complex.I) ∧ z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_number_quadrant_l2999_299955


namespace percent_relation_l2999_299915

theorem percent_relation (x y z : ℝ) 
  (hxy : x = 1.20 * y) 
  (hyz : y = 0.30 * z) : 
  x = 0.36 * z := by
sorry

end percent_relation_l2999_299915


namespace sqrt_sum_inequality_l2999_299963

theorem sqrt_sum_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  2 * Real.sqrt (a + b + c + d) ≥ Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d := by
  sorry

end sqrt_sum_inequality_l2999_299963


namespace quadratic_decreasing_parameter_range_l2999_299935

/-- Given a quadratic function f(x) = -x^2 - 2ax - 3 that is decreasing on the interval (-2, +∞),
    prove that the parameter a is in the range [2, +∞). -/
theorem quadratic_decreasing_parameter_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = -x^2 - 2*a*x - 3) 
  (h2 : ∀ x y, x > -2 → y > x → f y < f x) : 
  a ∈ Set.Ici 2 := by
sorry

end quadratic_decreasing_parameter_range_l2999_299935


namespace one_fourth_of_7_2_l2999_299982

theorem one_fourth_of_7_2 : 
  (7.2 : ℚ) / 4 = 9 / 5 := by sorry

end one_fourth_of_7_2_l2999_299982


namespace angle_supplement_complement_relation_l2999_299916

theorem angle_supplement_complement_relation (x : ℝ) : 
  (180 - x = 2 * (90 - x) + 40) → x = 40 := by
  sorry

end angle_supplement_complement_relation_l2999_299916


namespace expand_and_simplify_l2999_299952

theorem expand_and_simplify (x : ℝ) : 
  (x + 2)^2 + x * (3 - x) = 7 * x + 4 := by sorry

end expand_and_simplify_l2999_299952


namespace tree_height_proof_l2999_299957

/-- The initial height of a tree that grows 0.5 feet per year for 6 years and is 1/6 taller
    at the end of the 6th year compared to the 4th year. -/
def initial_tree_height : ℝ :=
  let growth_rate : ℝ := 0.5
  let years : ℕ := 6
  let h : ℝ := 4  -- Initial height to be proved
  h

theorem tree_height_proof (h : ℝ) (growth_rate : ℝ) (years : ℕ) 
    (h_growth : growth_rate = 0.5)
    (h_years : years = 6)
    (h_ratio : h + years * growth_rate = (h + 4 * growth_rate) * (1 + 1/6)) :
  h = initial_tree_height :=
sorry

#check tree_height_proof

end tree_height_proof_l2999_299957


namespace polygon_interior_angles_l2999_299959

theorem polygon_interior_angles (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 1260 → (n - 2) * 180 = sum_angles → n = 9 := by
  sorry

end polygon_interior_angles_l2999_299959


namespace fraction_to_decimal_l2999_299912

theorem fraction_to_decimal : (125 : ℚ) / 144 = 0.78125 := by
  sorry

end fraction_to_decimal_l2999_299912


namespace sum_of_squares_and_squared_sum_l2999_299950

theorem sum_of_squares_and_squared_sum : (5 + 9 - 3)^2 + (5^2 + 9^2 + 3^2) = 236 := by
  sorry

end sum_of_squares_and_squared_sum_l2999_299950


namespace helen_cookies_theorem_l2999_299978

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 31

/-- The number of cookies Helen baked the day before yesterday -/
def cookies_day_before_yesterday : ℕ := 419

/-- The total number of cookies Helen baked till last night -/
def total_cookies_till_last_night : ℕ := cookies_yesterday + cookies_day_before_yesterday

theorem helen_cookies_theorem : total_cookies_till_last_night = 450 := by
  sorry

end helen_cookies_theorem_l2999_299978


namespace combined_tax_rate_l2999_299926

/-- Given two individuals with different tax rates and incomes, calculate their combined tax rate -/
theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.40) 
  (h2 : mindy_rate = 0.25) 
  (h3 : income_ratio = 4) : 
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.28 := by
  sorry

end combined_tax_rate_l2999_299926


namespace oldest_child_age_l2999_299946

theorem oldest_child_age (a b c d : ℕ) : 
  a = 6 ∧ b = 9 ∧ c = 12 ∧ (a + b + c + d : ℚ) / 4 = 9 → d = 9 := by
  sorry

end oldest_child_age_l2999_299946


namespace hyperbola_asymptotes_l2999_299949

/-- The asymptotes of the hyperbola (y²/16) - (x²/9) = 1 are y = ±(4/3)x -/
theorem hyperbola_asymptotes :
  let hyperbola := (fun (x y : ℝ) => (y^2 / 16) - (x^2 / 9) = 1)
  ∃ (m : ℝ), m > 0 ∧
    (∀ (x y : ℝ), hyperbola x y → (y = m*x ∨ y = -m*x)) ∧
    m = 4/3 :=
sorry

end hyperbola_asymptotes_l2999_299949
