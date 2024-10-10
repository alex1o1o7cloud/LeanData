import Mathlib

namespace imaginary_unit_sixth_power_l1708_170892

theorem imaginary_unit_sixth_power (i : ℂ) (hi : i * i = -1) : i^6 = -1 := by
  sorry

end imaginary_unit_sixth_power_l1708_170892


namespace intersection_theorem_l1708_170815

/-- A permutation of {1, ..., n} is a bijective function from {1, ..., n} to itself. -/
def Permutation (n : ℕ) := {f : Fin n → Fin n // Function.Bijective f}

/-- Two permutations intersect if they have the same value at some position. -/
def intersect {n : ℕ} (p q : Permutation n) : Prop :=
  ∃ k : Fin n, p.val k = q.val k

/-- There exists a set of 1006 permutations of {1, ..., 2010} such that 
    any permutation of {1, ..., 2010} intersects with at least one of them. -/
theorem intersection_theorem : 
  ∃ (S : Finset (Permutation 2010)), S.card = 1006 ∧ 
    ∀ p : Permutation 2010, ∃ q ∈ S, intersect p q := by
  sorry

end intersection_theorem_l1708_170815


namespace fixed_point_exponential_function_l1708_170804

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) + 1
  f (-1) = 2 := by
  sorry

end fixed_point_exponential_function_l1708_170804


namespace complex_addition_complex_division_complex_multiplication_division_vector_operations_vector_dot_product_all_parts_combined_all_parts_combined_proof_l1708_170881

-- (1) (3+2i)+(\sqrt{3}-2)i
theorem complex_addition : ℂ → Prop :=
  fun z ↦ (3 + 2*Complex.I) + (Real.sqrt 3 - 2)*Complex.I = 3 + Real.sqrt 3 * Complex.I

-- (2) (9+2i)/(2+i)
theorem complex_division : ℂ → Prop :=
  fun z ↦ (9 + 2*Complex.I) / (2 + Complex.I) = 4 - Complex.I

-- (3) ((-1+i)(2+i))/(i^3)
theorem complex_multiplication_division : ℂ → Prop :=
  fun z ↦ ((-1 + Complex.I) * (2 + Complex.I)) / (Complex.I^3) = -1 - 3*Complex.I

-- (4) Given vectors a⃗=(-1,2) and b⃗=(2,1), calculate 2a⃗+3b⃗ and a⃗•b⃗
theorem vector_operations (a b : ℝ × ℝ) : Prop :=
  let a := (-1, 2)
  let b := (2, 1)
  (2 • a + 3 • b = (4, 7)) ∧ (a.1 * b.1 + a.2 * b.2 = 0)

-- (5) Given vectors a⃗ and b⃗ satisfy |a⃗|=1 and a⃗•b⃗=-1, calculate a⃗•(2a⃗-b⃗)
theorem vector_dot_product (a b : ℝ × ℝ) : Prop :=
  (a.1^2 + a.2^2 = 1) →
  (a.1 * b.1 + a.2 * b.2 = -1) →
  a.1 * (2*a.1 - b.1) + a.2 * (2*a.2 - b.2) = 3

-- Proofs are omitted
theorem all_parts_combined : Prop :=
  complex_addition 0 ∧
  complex_division 0 ∧
  complex_multiplication_division 0 ∧
  vector_operations (0, 0) (0, 0) ∧
  vector_dot_product (0, 0) (0, 0)

-- Add sorry to skip the proof
theorem all_parts_combined_proof : all_parts_combined := by sorry

end complex_addition_complex_division_complex_multiplication_division_vector_operations_vector_dot_product_all_parts_combined_all_parts_combined_proof_l1708_170881


namespace distance_to_plane_l1708_170884

/-- The distance from a point to a plane in 3D space -/
def distance_point_to_plane (P : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_to_plane :
  let P : ℝ × ℝ × ℝ := (-1, 3, 2)
  let n : ℝ × ℝ × ℝ := (2, -2, 1)
  distance_point_to_plane P n = 2 := by
  sorry

end distance_to_plane_l1708_170884


namespace quadratic_equation_k_value_l1708_170827

theorem quadratic_equation_k_value (k : ℝ) : 
  (∃ (r₁ r₂ : ℝ), r₁ > r₂ ∧ 
    2 * r₁^2 + 5 * r₁ = k ∧
    2 * r₂^2 + 5 * r₂ = k ∧
    r₁ - r₂ = 5.5) →
  k = -28.875 := by
sorry

end quadratic_equation_k_value_l1708_170827


namespace bucket_fills_theorem_l1708_170854

/-- Calculates the number of times a bucket is filled to reach the top of a bathtub. -/
def bucket_fills_to_top (bucket_capacity : ℕ) (buckets_removed : ℕ) (weekly_usage : ℕ) (days_per_week : ℕ) : ℕ :=
  let daily_usage := weekly_usage / days_per_week
  let removed_water := buckets_removed * bucket_capacity
  let full_tub_water := daily_usage + removed_water
  full_tub_water / bucket_capacity

/-- Theorem stating that under given conditions, the bucket is filled 14 times to reach the top. -/
theorem bucket_fills_theorem :
  bucket_fills_to_top 120 3 9240 7 = 14 := by
  sorry

end bucket_fills_theorem_l1708_170854


namespace election_result_l1708_170868

theorem election_result (total_votes : ℕ) (majority : ℕ) (winning_percentage : ℚ) : 
  total_votes = 800 →
  majority = 320 →
  winning_percentage = 70 →
  (winning_percentage / 100) * total_votes - ((100 - winning_percentage) / 100) * total_votes = majority :=
by sorry

end election_result_l1708_170868


namespace quadratic_inequality_solution_l1708_170846

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x, x^2 + a*x - 3 ≤ 0 ↔ -1 ≤ x ∧ x ≤ 3) → a = -2 := by
  sorry

end quadratic_inequality_solution_l1708_170846


namespace variance_of_linear_transform_l1708_170863

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- A linear transformation of a random variable -/
structure LinearTransform (X : BinomialRV) where
  a : ℝ
  b : ℝ
  Y : ℝ := a * X.n + b

theorem variance_of_linear_transform (X : BinomialRV) (Y : LinearTransform X) :
  X.n = 5 ∧ X.p = 1/4 ∧ Y.a = 4 ∧ Y.b = -3 →
  Y.a^2 * variance X = 15 :=
sorry

end variance_of_linear_transform_l1708_170863


namespace fair_spending_theorem_l1708_170812

/-- Calculates the remaining amount after spending at the fair -/
def remaining_amount (initial : ℕ) (snacks : ℕ) (rides_multiplier : ℕ) (games : ℕ) : ℕ :=
  initial - (snacks + rides_multiplier * snacks + games)

/-- Theorem stating that the remaining amount is 10 dollars -/
theorem fair_spending_theorem (initial : ℕ) (snacks : ℕ) (rides_multiplier : ℕ) (games : ℕ)
  (h1 : initial = 80)
  (h2 : snacks = 15)
  (h3 : rides_multiplier = 3)
  (h4 : games = 10) :
  remaining_amount initial snacks rides_multiplier games = 10 := by
  sorry

end fair_spending_theorem_l1708_170812


namespace cylinder_volume_from_sheet_l1708_170859

/-- The volume of a cylinder formed by a rectangular sheet as its lateral surface -/
theorem cylinder_volume_from_sheet (length width : ℝ) (h : length = 12 ∧ width = 8) :
  ∃ (volume : ℝ), (volume = 192 / Real.pi ∨ volume = 288 / Real.pi) ∧
  ∃ (radius height : ℝ), 
    (2 * Real.pi * radius = width ∧ height = length ∧ volume = Real.pi * radius^2 * height) ∨
    (2 * Real.pi * radius = length ∧ height = width ∧ volume = Real.pi * radius^2 * height) :=
by sorry

end cylinder_volume_from_sheet_l1708_170859


namespace modular_congruence_solution_l1708_170893

theorem modular_congruence_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n < 25 ∧ 72542 ≡ n [ZMOD 25] ∧ n = 17 := by
  sorry

end modular_congruence_solution_l1708_170893


namespace segment_length_l1708_170857

/-- A rectangle with side lengths 4 and 6, divided into four equal parts by two segments emanating from one vertex -/
structure DividedRectangle where
  /-- The length of the rectangle -/
  length : ℝ
  /-- The width of the rectangle -/
  width : ℝ
  /-- The length of the first dividing segment -/
  segment1 : ℝ
  /-- The length of the second dividing segment -/
  segment2 : ℝ
  /-- The rectangle has side lengths 4 and 6 -/
  dim_constraint : length = 4 ∧ width = 6
  /-- The two segments divide the rectangle into four equal parts -/
  division_constraint : ∃ (a b c d : ℝ), a + b + c + d = length * width ∧ 
                        a = b ∧ b = c ∧ c = d

/-- The theorem stating that one of the dividing segments has length √18.25 -/
theorem segment_length (r : DividedRectangle) : r.segment1 = Real.sqrt 18.25 ∨ r.segment2 = Real.sqrt 18.25 := by
  sorry

end segment_length_l1708_170857


namespace correct_num_children_l1708_170843

/-- The number of pencils each child has -/
def pencils_per_child : ℕ := 2

/-- The total number of pencils -/
def total_pencils : ℕ := 16

/-- The number of children -/
def num_children : ℕ := total_pencils / pencils_per_child

theorem correct_num_children : num_children = 8 := by
  sorry

end correct_num_children_l1708_170843


namespace twelfth_term_is_12_l1708_170855

/-- An arithmetic sequence with a₂ = -8 and common difference d = 2 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  let a₁ := -10  -- Derived from a₂ = -8 and d = 2
  a₁ + (n - 1) * 2

/-- The 12th term of the arithmetic sequence is 12 -/
theorem twelfth_term_is_12 : arithmetic_sequence 12 = 12 := by
  sorry

#eval arithmetic_sequence 12  -- For verification

end twelfth_term_is_12_l1708_170855


namespace sin_cos_power_six_sum_one_l1708_170847

theorem sin_cos_power_six_sum_one (α : Real) (h : Real.sin α + Real.cos α = 1) :
  Real.sin α ^ 6 + Real.cos α ^ 6 = 1 := by
  sorry

end sin_cos_power_six_sum_one_l1708_170847


namespace largest_constant_inequality_l1708_170810

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C*(x + y + z)) ↔ C ≤ 2 / Real.sqrt 3 :=
by sorry

end largest_constant_inequality_l1708_170810


namespace boat_speed_problem_l1708_170830

/-- Proves that given a boat traveling 45 miles upstream in 5 hours and having a speed of 12 mph in still water, the speed of the current is 3 mph. -/
theorem boat_speed_problem (distance : ℝ) (time : ℝ) (still_water_speed : ℝ) 
  (h1 : distance = 45) 
  (h2 : time = 5) 
  (h3 : still_water_speed = 12) : 
  still_water_speed - (distance / time) = 3 := by
  sorry

#check boat_speed_problem

end boat_speed_problem_l1708_170830


namespace saturday_price_of_200_dollar_coat_l1708_170887

/-- Calculates the Saturday price of a coat at Ajax Outlet Store -/
def saturday_price (original_price : ℝ) : ℝ :=
  let regular_discount_rate : ℝ := 0.6
  let saturday_discount_rate : ℝ := 0.3
  let price_after_regular_discount := original_price * (1 - regular_discount_rate)
  price_after_regular_discount * (1 - saturday_discount_rate)

/-- Theorem stating that the Saturday price of a $200 coat is $56 -/
theorem saturday_price_of_200_dollar_coat :
  saturday_price 200 = 56 := by
  sorry

end saturday_price_of_200_dollar_coat_l1708_170887


namespace smallest_perfect_cube_multiplier_l1708_170835

def y : ℕ := 2^3 * 3^4 * 4^3 * 5^4 * 6^6 * 7^7 * 8^8 * 9^9

theorem smallest_perfect_cube_multiplier (n : ℕ) :
  (∀ m : ℕ, m < 29400 → ¬ ∃ k : ℕ, m * y = k^3) ∧
  ∃ k : ℕ, 29400 * y = k^3 :=
sorry

end smallest_perfect_cube_multiplier_l1708_170835


namespace roots_of_equation_l1708_170869

theorem roots_of_equation (x : ℝ) : 
  (x = 0 ∨ x = -3) ↔ -x * (x + 3) = x * (x + 3) :=
by sorry

end roots_of_equation_l1708_170869


namespace f_negative_before_root_l1708_170823

-- Define the function f(x) = 2^x + log_2(x)
noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x / Real.log 2

-- State the theorem
theorem f_negative_before_root (a : ℝ) (h1 : f a = 0) (x : ℝ) (h2 : 0 < x) (h3 : x < a) :
  f x < 0 := by
  sorry

end f_negative_before_root_l1708_170823


namespace thirtieth_day_production_l1708_170811

/-- Represents the daily cloth production in feet -/
def cloth_sequence (n : ℕ) : ℚ :=
  5 + (n - 1) * ((390 - 30 * 5) / (30 * 29 / 2))

/-- The sum of the cloth_sequence for the first 30 days -/
def total_cloth : ℚ := 390

/-- The theorem states that the 30th term of the cloth_sequence is 21 -/
theorem thirtieth_day_production : cloth_sequence 30 = 21 := by sorry

end thirtieth_day_production_l1708_170811


namespace tangent_slope_tan_at_pi_over_four_l1708_170819

theorem tangent_slope_tan_at_pi_over_four :
  let f : ℝ → ℝ := λ x ↦ Real.tan x
  let x₀ : ℝ := π / 4
  (deriv f) x₀ = 2 :=
by sorry

end tangent_slope_tan_at_pi_over_four_l1708_170819


namespace proposition_equivalence_l1708_170813

theorem proposition_equivalence (x : ℝ) :
  (x^2 + 3*x - 4 = 0 → x = -4 ∨ x = 1) ↔ (x ≠ -4 ∧ x ≠ 1 → x^2 + 3*x - 4 ≠ 0) :=
by sorry

end proposition_equivalence_l1708_170813


namespace max_salary_basketball_team_l1708_170816

/-- Represents the maximum possible salary for a single player in a basketball team. -/
def maxSalary (numPlayers : ℕ) (minSalary : ℕ) (totalSalaryCap : ℕ) : ℕ :=
  totalSalaryCap - (numPlayers - 1) * minSalary

/-- Theorem stating the maximum possible salary for a single player
    given the team composition and salary constraints. -/
theorem max_salary_basketball_team :
  maxSalary 12 20000 500000 = 280000 := by
  sorry

#eval maxSalary 12 20000 500000

end max_salary_basketball_team_l1708_170816


namespace major_axis_length_l1708_170831

/-- Represents an ellipse formed by the intersection of a plane and a right circular cylinder. -/
structure IntersectionEllipse where
  cylinder_radius : ℝ
  major_axis : ℝ
  minor_axis : ℝ

/-- The theorem stating the length of the major axis given the conditions. -/
theorem major_axis_length 
  (e : IntersectionEllipse) 
  (h1 : e.cylinder_radius = 2)
  (h2 : e.minor_axis = 2 * e.cylinder_radius)
  (h3 : e.major_axis = e.minor_axis * (1 + 0.75)) :
  e.major_axis = 7 := by
  sorry


end major_axis_length_l1708_170831


namespace equal_product_sequence_characterization_l1708_170882

def is_equal_product_sequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 2 → a n * a (n - 1) = k

theorem equal_product_sequence_characterization (a : ℕ → ℝ) :
  is_equal_product_sequence a ↔
    ∃ k : ℝ, ∀ n : ℕ, n ≥ 2 → a n * a (n - 1) = k :=
by sorry

end equal_product_sequence_characterization_l1708_170882


namespace triangle_area_angle_l1708_170873

theorem triangle_area_angle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (a^2 + b^2 - c^2) / 4
  S = (1/2) * a * b * Real.sin (π/4) →
  ∃ A B C : ℝ,
    A + B + C = π ∧
    a = BC ∧ b = AC ∧ c = AB ∧
    C = π/4 :=
sorry

end triangle_area_angle_l1708_170873


namespace repeating_decimal_sum_l1708_170838

theorem repeating_decimal_sum (c d : ℕ) : 
  (c < 10 ∧ d < 10) →  -- Ensuring c and d are single digits
  (5 : ℚ) / 13 = (c * 10 + d : ℚ) / 99 →  -- Representing the repeating decimal
  c + d = 11 := by
  sorry

end repeating_decimal_sum_l1708_170838


namespace clock_partition_exists_l1708_170806

/-- A partition of the set {1, 2, ..., 12} into three subsets -/
structure ClockPartition where
  part1 : Finset Nat
  part2 : Finset Nat
  part3 : Finset Nat
  partition_complete : part1 ∪ part2 ∪ part3 = Finset.range 12
  partition_disjoint1 : Disjoint part1 part2
  partition_disjoint2 : Disjoint part1 part3
  partition_disjoint3 : Disjoint part2 part3

/-- The theorem stating that there exists a partition of the clock numbers
    into three parts with equal sums -/
theorem clock_partition_exists : ∃ (p : ClockPartition),
  (p.part1.sum id = p.part2.sum id) ∧ (p.part2.sum id = p.part3.sum id) :=
sorry

end clock_partition_exists_l1708_170806


namespace pentagon_reconstruction_l1708_170801

-- Define the pentagon and extended points
variable (A B C D E A' B' C' D' E' : ℝ × ℝ)

-- Define the conditions
axiom ext_A : A' = A + (A - B)
axiom ext_B : B' = B + (B - C)
axiom ext_C : C' = C + (C - D)
axiom ext_D : D' = D + (D - E)
axiom ext_E : E' = E + (E - A)

-- Define the theorem
theorem pentagon_reconstruction :
  A = (1/31 : ℝ) • A' + (5/31 : ℝ) • B' + (10/31 : ℝ) • C' + (15/31 : ℝ) • D' + (1/31 : ℝ) • E' := by
  sorry

end pentagon_reconstruction_l1708_170801


namespace remainder_2023_times_7_div_45_l1708_170809

theorem remainder_2023_times_7_div_45 : (2023 * 7) % 45 = 31 := by
  sorry

end remainder_2023_times_7_div_45_l1708_170809


namespace pentomino_circumscribing_rectangle_ratio_l1708_170860

/-- A pentomino is a planar geometric figure formed by joining five equal squares edge to edge. -/
structure Pentomino where
  -- Add necessary fields to represent a pentomino
  -- This is a placeholder and may need to be expanded based on specific requirements

/-- A rectangle that circumscribes a pentomino. -/
structure CircumscribingRectangle (p : Pentomino) where
  width : ℝ
  height : ℝ
  -- Add necessary fields to represent the relationship between the pentomino and the rectangle
  -- This is a placeholder and may need to be expanded based on specific requirements

/-- The theorem stating that for any pentomino inscribed in a rectangle, 
    the ratio of the shorter side to the longer side of the rectangle is 1:2. -/
theorem pentomino_circumscribing_rectangle_ratio (p : Pentomino) 
  (r : CircumscribingRectangle p) : 
  min r.width r.height / max r.width r.height = 1 / 2 := by
  sorry

end pentomino_circumscribing_rectangle_ratio_l1708_170860


namespace bhanu_house_rent_expenditure_l1708_170879

/-- Calculates Bhanu's expenditure on house rent based on his spending pattern -/
theorem bhanu_house_rent_expenditure (total_income : ℝ) 
  (h1 : 0.30 * total_income = 300) 
  (h2 : total_income > 0) : 
  0.14 * (total_income - 0.30 * total_income) = 98 := by
  sorry

end bhanu_house_rent_expenditure_l1708_170879


namespace sqrt_fourth_power_equals_256_l1708_170824

theorem sqrt_fourth_power_equals_256 (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end sqrt_fourth_power_equals_256_l1708_170824


namespace infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l1708_170817

def is_lovely (n : ℕ+) : Prop :=
  ∃ (k : ℕ+) (d : Fin k → ℕ+),
    n = (Finset.range k).prod (λ i => d i) ∧
    ∀ i : Fin k, (d i)^2 ∣ (n + d i)

theorem infinitely_many_lovely_numbers :
  ∀ N : ℕ, ∃ n : ℕ+, n > N ∧ is_lovely n :=
sorry

theorem no_lovely_square_greater_than_one :
  ¬∃ m : ℕ+, m > 1 ∧ is_lovely (m^2) :=
sorry

end infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l1708_170817


namespace sufficient_condition_for_ellipse_l1708_170842

/-- The equation of a potential ellipse -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / m + y^2 / (2*m - 1) = 1

/-- Condition for the equation to represent an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  m > 0 ∧ 2*m - 1 > 0 ∧ m ≠ 2*m - 1

/-- Theorem stating that m > 1 is a sufficient but not necessary condition for the equation to represent an ellipse -/
theorem sufficient_condition_for_ellipse :
  ∀ m : ℝ, m > 1 → is_ellipse m ∧ ∃ m₀ : ℝ, m₀ ≤ 1 ∧ is_ellipse m₀ :=
sorry

end sufficient_condition_for_ellipse_l1708_170842


namespace grasshopper_position_l1708_170802

/-- Represents the points on the circle -/
inductive Point : Type
| one : Point
| two : Point
| three : Point
| four : Point
| five : Point
| six : Point
| seven : Point

/-- Determines if a point is odd-numbered -/
def is_odd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true
  | Point.six => false
  | Point.seven => true

/-- Represents a single jump of the grasshopper -/
def jump (p : Point) : Point :=
  match p with
  | Point.one => Point.seven
  | Point.two => Point.seven
  | Point.three => Point.two
  | Point.four => Point.two
  | Point.five => Point.four
  | Point.six => Point.four
  | Point.seven => Point.six

/-- Represents multiple jumps of the grasshopper -/
def multi_jump (p : Point) (n : Nat) : Point :=
  match n with
  | 0 => p
  | Nat.succ m => jump (multi_jump p m)

theorem grasshopper_position : multi_jump Point.seven 2011 = Point.two := by
  sorry

end grasshopper_position_l1708_170802


namespace blood_donation_selection_count_l1708_170867

def male_teachers : ℕ := 3
def female_teachers : ℕ := 6
def total_teachers : ℕ := male_teachers + female_teachers
def selection_size : ℕ := 5

theorem blood_donation_selection_count :
  (Nat.choose total_teachers selection_size) - (Nat.choose female_teachers selection_size) = 120 := by
  sorry

end blood_donation_selection_count_l1708_170867


namespace discount_composition_l1708_170805

theorem discount_composition (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.3
  let second_discount := 0.4
  let price_after_first := original_price * (1 - first_discount)
  let price_after_second := price_after_first * (1 - second_discount)
  let total_discount := 1 - (price_after_second / original_price)
  total_discount = 0.58 := by
sorry

end discount_composition_l1708_170805


namespace f_comp_three_roots_l1708_170877

/-- A quadratic function f(x) = x^2 + 4x + c -/
def f (c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + 4*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) : ℝ → ℝ := fun x ↦ f c (f c x)

/-- Predicate to check if a function has exactly 3 distinct real roots -/
def has_exactly_three_distinct_real_roots (g : ℝ → ℝ) : Prop :=
  ∃ (x y z : ℝ), (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
    (g x = 0 ∧ g y = 0 ∧ g z = 0) ∧
    (∀ w : ℝ, g w = 0 → w = x ∨ w = y ∨ w = z)

/-- Theorem stating that f(f(x)) has exactly 3 distinct real roots iff c = 1 -/
theorem f_comp_three_roots :
  ∀ c : ℝ, has_exactly_three_distinct_real_roots (f_comp c) ↔ c = 1 :=
sorry

end f_comp_three_roots_l1708_170877


namespace complex_equation_solutions_l1708_170890

theorem complex_equation_solutions (c p q r s : ℂ) : 
  (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) →
  (∀ z : ℂ, (z - p) * (z - q) * (z - r) * (z - s) = 
             (z - c*p) * (z - c*q) * (z - c*r) * (z - c*s)) →
  (∃ (solutions : Finset ℂ), solutions.card = 4 ∧ c ∈ solutions ∧
    ∀ x ∈ solutions, x^4 = 1) :=
by sorry

end complex_equation_solutions_l1708_170890


namespace unique_sequence_solution_l1708_170895

/-- Represents a solution to the sequence problem -/
structure SequenceSolution where
  n : ℕ
  q : ℚ
  d : ℚ

/-- Checks if a given solution satisfies all conditions of the problem -/
def is_valid_solution (sol : SequenceSolution) : Prop :=
  sol.n > 1 ∧
  1 + (sol.n - 1) * sol.d = 81 ∧
  1 * sol.q^(sol.n - 1) = 81 ∧
  sol.q / sol.d = 0.15

/-- The unique solution to the sequence problem -/
def unique_solution : SequenceSolution :=
  { n := 5, q := 3, d := 20 }

/-- Theorem stating that the unique_solution is the only valid solution -/
theorem unique_sequence_solution :
  is_valid_solution unique_solution ∧
  ∀ (sol : SequenceSolution), is_valid_solution sol → sol = unique_solution :=
sorry

end unique_sequence_solution_l1708_170895


namespace mom_bought_51_shirts_l1708_170837

/-- The number of t-shirts in a package -/
def shirts_per_package : ℕ := 3

/-- The number of packages if t-shirts were purchased in packages -/
def num_packages : ℕ := 17

/-- The total number of t-shirts Mom bought -/
def total_shirts : ℕ := shirts_per_package * num_packages

theorem mom_bought_51_shirts : total_shirts = 51 := by
  sorry

end mom_bought_51_shirts_l1708_170837


namespace sum_of_segments_constant_l1708_170821

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- Checks if a point is inside a triangle -/
def isInterior (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Calculates the length of a segment from a vertex to the intersection
    of a parallel line through a point with the opposite side -/
def segmentLength (t : Triangle) (p : Point) (v : Point) : ℝ :=
  sorry

/-- Main theorem -/
theorem sum_of_segments_constant (t : Triangle) (p : Point) :
  isEquilateral t → isInterior p t →
  segmentLength t p t.A + segmentLength t p t.B + segmentLength t p t.C =
  distance t.A t.B :=
by sorry

end sum_of_segments_constant_l1708_170821


namespace last_digit_of_sum_l1708_170852

/-- Given a = 25 and b = -3, the last digit of a^1999 + b^2002 is 4 -/
theorem last_digit_of_sum (a b : ℤ) : a = 25 ∧ b = -3 → (a^1999 + b^2002) % 10 = 4 := by
  sorry

end last_digit_of_sum_l1708_170852


namespace marble_collection_l1708_170896

theorem marble_collection (total : ℕ) (friend_total : ℕ) : 
  (40 : ℚ) / 100 * total + (20 : ℚ) / 100 * total + (40 : ℚ) / 100 * total = total →
  (40 : ℚ) / 100 * friend_total = 2 →
  friend_total = 5 :=
by
  sorry

end marble_collection_l1708_170896


namespace jeff_pickup_cost_l1708_170834

/-- The cost of last year's costume in dollars -/
def last_year_cost : ℝ := 250

/-- The percentage increase in cost compared to last year -/
def cost_increase_percent : ℝ := 0.4

/-- The deposit percentage -/
def deposit_percent : ℝ := 0.1

/-- The total cost of this year's costume -/
def total_cost : ℝ := last_year_cost * (1 + cost_increase_percent)

/-- The amount of the deposit -/
def deposit : ℝ := total_cost * deposit_percent

/-- The amount Jeff paid when picking up the costume -/
def pickup_cost : ℝ := total_cost - deposit

theorem jeff_pickup_cost : pickup_cost = 315 := by
  sorry

end jeff_pickup_cost_l1708_170834


namespace jason_gave_nine_cards_l1708_170894

/-- The number of Pokemon cards Jason gave to his friends -/
def cards_given_to_friends (initial_cards : ℕ) (remaining_cards : ℕ) : ℕ :=
  initial_cards - remaining_cards

/-- Theorem: Jason gave 9 Pokemon cards to his friends -/
theorem jason_gave_nine_cards : cards_given_to_friends 13 4 = 9 := by
  sorry

end jason_gave_nine_cards_l1708_170894


namespace sqrt_neg_five_squared_l1708_170883

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end sqrt_neg_five_squared_l1708_170883


namespace simplify_fraction_l1708_170849

theorem simplify_fraction (a : ℝ) (h : a = 5) : 15 * a^4 / (75 * a^3) = 1 := by
  sorry

end simplify_fraction_l1708_170849


namespace ratio_of_percentages_l1708_170897

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.3 * P) 
  (hN : N = 0.6 * (2 * P)) : 
  M / N = 1 / 10 := by
  sorry

end ratio_of_percentages_l1708_170897


namespace project_completion_time_l1708_170825

theorem project_completion_time
  (days_A : ℝ)
  (days_B : ℝ)
  (break_days : ℝ)
  (h1 : days_A = 18)
  (h2 : days_B = 15)
  (h3 : break_days = 4) :
  let efficiency_A := 1 / days_A
  let efficiency_B := 1 / days_B
  let combined_efficiency := efficiency_A + efficiency_B
  let work_during_break := efficiency_B * break_days
  (1 - work_during_break) / combined_efficiency + break_days = 10 :=
by sorry

end project_completion_time_l1708_170825


namespace unique_perpendicular_line_l1708_170814

/-- A plane in Euclidean geometry -/
structure EuclideanPlane :=
  (points : Type*)
  (lines : Type*)
  (on_line : points → lines → Prop)

/-- Definition of perpendicular lines in a plane -/
def perpendicular (p : EuclideanPlane) (l1 l2 : p.lines) : Prop :=
  sorry

/-- Statement: In a plane, given a line and a point not on the line,
    there exists a unique line passing through the point
    that is perpendicular to the given line -/
theorem unique_perpendicular_line
  (p : EuclideanPlane) (l : p.lines) (pt : p.points)
  (h : ¬ p.on_line pt l) :
  ∃! l' : p.lines, p.on_line pt l' ∧ perpendicular p l l' :=
sorry

end unique_perpendicular_line_l1708_170814


namespace computer_price_difference_l1708_170803

/-- The price difference between two stores selling the same computer with different prices and discounts -/
theorem computer_price_difference (price1 : ℝ) (discount1 : ℝ) (price2 : ℝ) (discount2 : ℝ) 
  (h1 : price1 = 950) (h2 : discount1 = 0.06) (h3 : price2 = 920) (h4 : discount2 = 0.05) :
  abs (price1 * (1 - discount1) - price2 * (1 - discount2)) = 19 :=
by sorry

end computer_price_difference_l1708_170803


namespace quadrilateral_area_is_half_unit_l1708_170888

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the intersection point of two lines -/
def intersectionPoint (l1 l2 : Line) : Point :=
  sorry

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ :=
  sorry

/-- The main theorem stating that the area of the quadrilateral is 0.5 square units -/
theorem quadrilateral_area_is_half_unit : 
  let l1 : Line := { a := 3, b := 4, c := -12 }
  let l2 : Line := { a := 6, b := -4, c := -12 }
  let l3 : Line := { a := 1, b := 0, c := -3 }
  let l4 : Line := { a := 0, b := 1, c := -1 }
  let p1 := intersectionPoint l1 l2
  let p2 := intersectionPoint l1 l3
  let p3 := intersectionPoint l2 l3
  let p4 := intersectionPoint l1 l4
  quadrilateralArea p1 p2 p3 p4 = 0.5 := by
  sorry

end quadrilateral_area_is_half_unit_l1708_170888


namespace divisibility_theorem_l1708_170866

theorem divisibility_theorem (m n : ℕ+) (h : 5 ∣ (2^n.val + 3^m.val)) :
  5 ∣ (2^m.val + 3^n.val) := by
  sorry

end divisibility_theorem_l1708_170866


namespace star_calculation_l1708_170800

/-- The custom operation ⋆ defined as x ⋆ y = (x² + y²)(x - y) -/
def star (x y : ℝ) : ℝ := (x^2 + y^2) * (x - y)

/-- Theorem stating that 2 ⋆ (3 ⋆ 4) = 16983 -/
theorem star_calculation : star 2 (star 3 4) = 16983 := by
  sorry

end star_calculation_l1708_170800


namespace sum_reciprocals_l1708_170865

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -2) (hb : b ≠ -2) (hc : c ≠ -2) (hd : d ≠ -2)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω) :
  (1 / (a + 2)) + (1 / (b + 2)) + (1 / (c + 2)) + (1 / (d + 2)) = 3 := by
  sorry

end sum_reciprocals_l1708_170865


namespace outfits_count_l1708_170886

/-- The number of different outfits that can be created with given clothing items. -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (blazers : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (blazers + 1)

/-- Theorem stating the number of outfits with specific clothing items. -/
theorem outfits_count : number_of_outfits 5 4 5 2 = 360 := by
  sorry

end outfits_count_l1708_170886


namespace class_mean_calculation_l1708_170851

theorem class_mean_calculation (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_mean : ℚ) (group2_mean : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 32 →
  group2_students = 8 →
  group1_mean = 68 / 100 →
  group2_mean = 82 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 708 / 1000 :=
by sorry

end class_mean_calculation_l1708_170851


namespace triangle_trigonometric_identities_l1708_170885

theorem triangle_trigonometric_identities 
  (a b c : ℝ) (α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_sum_angles : α + β + γ = π)
  (h_law_of_sines : a / Real.sin α = b / Real.sin β ∧ b / Real.sin β = c / Real.sin γ) :
  (a + b) / c = Real.cos ((α - β) / 2) / Real.sin (γ / 2) ∧
  (a - b) / c = Real.sin ((α - β) / 2) / Real.cos (γ / 2) := by
  sorry

end triangle_trigonometric_identities_l1708_170885


namespace fabric_cost_theorem_l1708_170861

/-- Represents the cost in livres, sous, and deniers -/
structure Cost :=
  (livres : ℕ)
  (sous : ℕ)
  (deniers : ℚ)

/-- Converts a Cost to deniers -/
def cost_to_deniers (c : Cost) : ℚ :=
  c.livres * 20 * 12 + c.sous * 12 + c.deniers

/-- Converts deniers to a Cost -/
def deniers_to_cost (d : ℚ) : Cost :=
  let total_sous := d / 12
  let livres := (total_sous / 20).floor
  let remaining_sous := total_sous - livres * 20
  { livres := livres.toNat,
    sous := remaining_sous.floor.toNat,
    deniers := d - (livres * 20 * 12 + remaining_sous.floor * 12) }

def ell_cost : Cost := { livres := 42, sous := 17, deniers := 11 }

def fabric_length : ℚ := 15 + 13 / 16

theorem fabric_cost_theorem :
  deniers_to_cost (cost_to_deniers ell_cost * fabric_length) =
  { livres := 682, sous := 15, deniers := 9 + 11 / 16 } := by
  sorry

end fabric_cost_theorem_l1708_170861


namespace coefficient_x4_in_product_l1708_170876

/-- The coefficient of x^4 in the expansion of (2x^3 + 5x^2 - 3x)(3x^3 - 8x^2 + 6x - 9) is -37 -/
theorem coefficient_x4_in_product : 
  let p₁ : Polynomial ℤ := 2 * X^3 + 5 * X^2 - 3 * X
  let p₂ : Polynomial ℤ := 3 * X^3 - 8 * X^2 + 6 * X - 9
  (p₁ * p₂).coeff 4 = -37 := by
sorry

end coefficient_x4_in_product_l1708_170876


namespace multiples_of_seven_l1708_170880

theorem multiples_of_seven (a b : ℕ) (q : Finset ℕ) : 
  (∃ k₁ k₂ : ℕ, a = 14 * k₁ ∧ b = 14 * k₂) →  -- a and b are multiples of 14
  (∀ x ∈ q, a ≤ x ∧ x ≤ b) →  -- q is the set of consecutive integers between a and b, inclusive
  (∀ x ∈ q, x + 1 ∈ q ∨ x = b) →  -- q contains consecutive integers
  (q.filter (λ x => x % 14 = 0)).card = 14 →  -- q contains 14 multiples of 14
  (q.filter (λ x => x % 7 = 0)).card = 27 :=  -- The number of multiples of 7 in q is 27
by sorry

end multiples_of_seven_l1708_170880


namespace no_double_reverse_number_l1708_170848

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ+) : ℕ+ :=
  sorry

/-- Theorem: There does not exist a positive integer N such that 
    when its digits are reversed, the resulting number is exactly twice N -/
theorem no_double_reverse_number : ¬ ∃ (N : ℕ+), reverseDigits N = 2 * N := by
  sorry

end no_double_reverse_number_l1708_170848


namespace probability_not_adjacent_l1708_170871

/-- The number of chairs in the row -/
def n : ℕ := 12

/-- The probability that Mary and James don't sit next to each other -/
def prob_not_adjacent : ℚ := 5/6

/-- The theorem stating the probability of Mary and James not sitting next to each other -/
theorem probability_not_adjacent :
  (1 - (n - 1 : ℚ) / (n.choose 2 : ℚ)) = prob_not_adjacent :=
sorry

end probability_not_adjacent_l1708_170871


namespace count_valid_permutations_l1708_170808

def alphabet : List Char := ['a', 'b', 'c', 'd', 'e']

def is_adjacent (c1 c2 : Char) : Bool :=
  let idx1 := alphabet.indexOf c1
  let idx2 := alphabet.indexOf c2
  (idx1 + 1 = idx2) || (idx2 + 1 = idx1)

def is_valid_permutation (perm : List Char) : Bool :=
  List.zip perm (List.tail perm) |>.all (fun (c1, c2) => !is_adjacent c1 c2)

def valid_permutations : List (List Char) :=
  List.permutations alphabet |>.filter is_valid_permutation

theorem count_valid_permutations : valid_permutations.length = 8 := by
  sorry

end count_valid_permutations_l1708_170808


namespace initial_distance_proof_l1708_170889

/-- The initial distance between Tim and Élan -/
def initial_distance : ℝ := 30

/-- Tim's initial speed in mph -/
def tim_speed : ℝ := 10

/-- Élan's initial speed in mph -/
def elan_speed : ℝ := 5

/-- The distance Tim travels until meeting Élan -/
def tim_distance : ℝ := 20

/-- The time it takes for Tim and Élan to meet -/
def meeting_time : ℝ := 1.5

theorem initial_distance_proof :
  initial_distance = 
    tim_speed * 1 + 
    elan_speed * 1 + 
    (tim_speed * 2) * (meeting_time - 1) + 
    (elan_speed * 2) * (meeting_time - 1) :=
sorry

end initial_distance_proof_l1708_170889


namespace ratio_of_sums_l1708_170845

/-- Represents an arithmetic progression --/
structure ArithmeticProgression where
  firstTerm : ℕ
  difference : ℕ
  length : ℕ

/-- Calculates the sum of an arithmetic progression --/
def sumOfArithmeticProgression (ap : ArithmeticProgression) : ℕ :=
  ap.length * (2 * ap.firstTerm + (ap.length - 1) * ap.difference) / 2

/-- Generates a list of arithmetic progressions for the first group --/
def firstGroup : List ArithmeticProgression :=
  List.range 15
    |> List.map (fun i => ArithmeticProgression.mk (i + 1) (2 * (i + 1)) 10)

/-- Generates a list of arithmetic progressions for the second group --/
def secondGroup : List ArithmeticProgression :=
  List.range 15
    |> List.map (fun i => ArithmeticProgression.mk (i + 1) (2 * i + 1) 10)

/-- Calculates the sum of all elements in a group of arithmetic progressions --/
def sumOfGroup (group : List ArithmeticProgression) : ℕ :=
  group.map sumOfArithmeticProgression |> List.sum

theorem ratio_of_sums : 
  (sumOfGroup firstGroup : ℚ) / (sumOfGroup secondGroup : ℚ) = 160 / 151 := by
  sorry

end ratio_of_sums_l1708_170845


namespace cube_root_27_times_fourth_root_81_times_sixth_root_64_l1708_170872

theorem cube_root_27_times_fourth_root_81_times_sixth_root_64 :
  ∃ (a b c : ℝ), a^3 = 27 ∧ b^4 = 81 ∧ c^6 = 64 ∧ a * b * c = 18 := by
  sorry

end cube_root_27_times_fourth_root_81_times_sixth_root_64_l1708_170872


namespace alex_max_correct_answers_l1708_170829

/-- Represents a math contest with multiple-choice questions. -/
structure MathContest where
  total_questions : ℕ
  correct_points : ℤ
  blank_points : ℤ
  incorrect_points : ℤ

/-- Represents a student's performance in the math contest. -/
structure StudentPerformance where
  contest : MathContest
  total_score : ℤ

/-- Calculates the maximum number of correct answers for a given student performance. -/
def max_correct_answers (perf : StudentPerformance) : ℕ :=
  sorry

/-- The theorem stating the maximum number of correct answers for Alex's performance. -/
theorem alex_max_correct_answers :
  let contest : MathContest := {
    total_questions := 80,
    correct_points := 5,
    blank_points := 0,
    incorrect_points := -2
  }
  let performance : StudentPerformance := {
    contest := contest,
    total_score := 150
  }
  max_correct_answers performance = 44 := by
  sorry

end alex_max_correct_answers_l1708_170829


namespace height_difference_calculation_l1708_170874

/-- The combined height difference between an uncle and his two relatives -/
def combined_height_difference (uncle_height james_initial_ratio growth_spurt younger_sibling_height : ℝ) : ℝ :=
  let james_new_height := uncle_height * james_initial_ratio + growth_spurt
  let diff_uncle_james := uncle_height - james_new_height
  let diff_uncle_younger := uncle_height - younger_sibling_height
  diff_uncle_james + diff_uncle_younger

/-- Theorem stating the combined height difference given specific measurements -/
theorem height_difference_calculation :
  combined_height_difference 72 (2/3) 10 38 = 48 := by
  sorry

end height_difference_calculation_l1708_170874


namespace most_stable_scores_l1708_170841

theorem most_stable_scores (S_A S_B S_C : ℝ) 
  (h1 : S_A = 38) (h2 : S_B = 10) (h3 : S_C = 26) :
  S_B < S_A ∧ S_B < S_C := by
  sorry

end most_stable_scores_l1708_170841


namespace mark_change_factor_l1708_170820

/-- Given a class of students, prove that if their marks are changed by a factor
    that doubles the average, then this factor must be 2. -/
theorem mark_change_factor
  (n : ℕ)                    -- number of students
  (initial_avg : ℝ)          -- initial average mark
  (final_avg : ℝ)            -- final average mark
  (h_n : n = 30)             -- there are 30 students
  (h_initial : initial_avg = 45)  -- initial average is 45
  (h_final : final_avg = 90)      -- final average is 90
  : (final_avg / initial_avg : ℝ) = 2 := by
  sorry

end mark_change_factor_l1708_170820


namespace volume_removed_percent_l1708_170840

def box_length : ℝ := 15
def box_width : ℝ := 10
def box_height : ℝ := 8
def cube_side : ℝ := 3
def num_corners : ℕ := 8

def box_volume : ℝ := box_length * box_width * box_height
def removed_cube_volume : ℝ := cube_side ^ 3
def total_removed_volume : ℝ := num_corners * removed_cube_volume

theorem volume_removed_percent :
  (total_removed_volume / box_volume) * 100 = 18 := by sorry

end volume_removed_percent_l1708_170840


namespace equation_solution_l1708_170899

theorem equation_solution :
  ∀ x : ℚ, (25 - 7 : ℚ) = 5/2 + x → x = 31/2 := by
  sorry

end equation_solution_l1708_170899


namespace solution_range_l1708_170818

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ (2 * x + m) / (x - 1) = 1) → 
  (m ≤ -1 ∧ m ≠ -2) :=
by sorry

end solution_range_l1708_170818


namespace f_derivative_at_zero_l1708_170822

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.log (1 + 2 * x^2 + x^3)) / x else 0

theorem f_derivative_at_zero : 
  deriv f 0 = 2 := by
  sorry

end f_derivative_at_zero_l1708_170822


namespace grocer_bananas_theorem_l1708_170826

/-- Represents the number of pounds of bananas purchased by the grocer -/
def bananas_purchased : ℝ := 96

/-- Represents the purchase price in dollars per 3 pounds of bananas -/
def purchase_price : ℝ := 0.50

/-- Represents the selling price in dollars per 4 pounds of bananas -/
def selling_price : ℝ := 1.00

/-- Represents the total profit in dollars -/
def total_profit : ℝ := 8.00

/-- Theorem stating that the number of pounds of bananas purchased by the grocer is 96 -/
theorem grocer_bananas_theorem :
  bananas_purchased = 96 ∧
  (selling_price / 4 - purchase_price / 3) * bananas_purchased = total_profit :=
sorry

end grocer_bananas_theorem_l1708_170826


namespace bookshop_inventory_l1708_170828

/-- Bookshop inventory problem -/
theorem bookshop_inventory (initial_books : ℕ) (saturday_instore : ℕ) (saturday_online : ℕ) 
  (sunday_instore : ℕ) (shipment : ℕ) (final_books : ℕ) 
  (h1 : initial_books = 743)
  (h2 : saturday_instore = 37)
  (h3 : saturday_online = 128)
  (h4 : sunday_instore = 2 * saturday_instore)
  (h5 : shipment = 160)
  (h6 : final_books = 502) :
  ∃ (sunday_online : ℕ), 
    final_books = initial_books - (saturday_instore + saturday_online + sunday_instore + sunday_online) + shipment ∧ 
    sunday_online = saturday_online + 34 :=
by sorry

end bookshop_inventory_l1708_170828


namespace simple_interest_rate_example_l1708_170836

/-- Calculates the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

theorem simple_interest_rate_example :
  simple_interest_rate 750 900 10 = 2 := by
  sorry

end simple_interest_rate_example_l1708_170836


namespace max_n_value_l1708_170875

theorem max_n_value (A B : ℤ) (h : A * B = 54) : 
  ∃ (n : ℤ), n = 3 * B + A ∧ ∀ (m : ℤ), m = 3 * B + A → m ≤ n :=
sorry

end max_n_value_l1708_170875


namespace square_side_length_l1708_170898

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 49) (h2 : side^2 = area) : side = 7 := by
  sorry

end square_side_length_l1708_170898


namespace bubble_radius_l1708_170833

/-- The radius of a sphere with volume equal to the sum of volumes of a hemisphere and a cylinder --/
theorem bubble_radius (hemisphere_radius cylinder_radius cylinder_height : ℝ) 
  (hr : hemisphere_radius = 5)
  (hcr : cylinder_radius = 2)
  (hch : cylinder_height = hemisphere_radius) : 
  ∃ R : ℝ, R^3 = 77.5 ∧ 
  (4/3 * Real.pi * R^3 = 2/3 * Real.pi * hemisphere_radius^3 + Real.pi * cylinder_radius^2 * cylinder_height) :=
by sorry

end bubble_radius_l1708_170833


namespace a_gt_abs_b_sufficient_not_necessary_l1708_170856

theorem a_gt_abs_b_sufficient_not_necessary :
  (∃ a b : ℝ, a > |b| ∧ a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > |b|)) :=
by sorry

end a_gt_abs_b_sufficient_not_necessary_l1708_170856


namespace food_bank_remaining_lyanna_food_bank_remaining_l1708_170853

/-- Given a food bank with donations over two weeks and a distribution in the third week,
    calculate the remaining food. -/
theorem food_bank_remaining (first_week : ℝ) (second_week_multiplier : ℝ) (distribution_percentage : ℝ) : ℝ :=
  let second_week := first_week * second_week_multiplier
  let total_donated := first_week + second_week
  let distributed := total_donated * distribution_percentage
  let remaining := total_donated - distributed
  remaining

/-- The amount of food remaining in Lyanna's food bank after two weeks of donations
    and a distribution in the third week. -/
theorem lyanna_food_bank_remaining : food_bank_remaining 40 2 0.7 = 36 := by
  sorry

end food_bank_remaining_lyanna_food_bank_remaining_l1708_170853


namespace discount_percentage_calculation_l1708_170850

/-- Calculates the discount percentage on half of the bricks given the total number of bricks,
    full price per brick, and total amount spent. -/
theorem discount_percentage_calculation
  (total_bricks : ℕ)
  (full_price_per_brick : ℚ)
  (total_spent : ℚ)
  (h1 : total_bricks = 1000)
  (h2 : full_price_per_brick = 1/2)
  (h3 : total_spent = 375) :
  let half_bricks := total_bricks / 2
  let full_price_half := half_bricks * full_price_per_brick
  let discounted_price := total_spent - full_price_half
  let discount_amount := full_price_half - discounted_price
  let discount_percentage := (discount_amount / full_price_half) * 100
  discount_percentage = 50 := by sorry

end discount_percentage_calculation_l1708_170850


namespace amusement_park_admission_difference_l1708_170807

theorem amusement_park_admission_difference :
  let students : ℕ := 194
  let adults : ℕ := 235
  let free_admission : ℕ := 68
  let total_visitors : ℕ := students + adults
  let paid_admission : ℕ := total_visitors - free_admission
  paid_admission - free_admission = 293 :=
by
  sorry

end amusement_park_admission_difference_l1708_170807


namespace polynomial_division_remainder_l1708_170862

def polynomial (x : ℝ) : ℝ := 8*x^4 - 10*x^3 + 7*x^2 - 5*x - 30

def divisor (x : ℝ) : ℝ := 2*x - 4

theorem polynomial_division_remainder :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 36 := by
  sorry

end polynomial_division_remainder_l1708_170862


namespace mary_breeding_balls_l1708_170864

theorem mary_breeding_balls (snakes_per_ball : ℕ) (additional_pairs : ℕ) (total_snakes : ℕ) 
  (h1 : snakes_per_ball = 8)
  (h2 : additional_pairs = 6)
  (h3 : total_snakes = 36) :
  ∃ (num_balls : ℕ), 
    num_balls * snakes_per_ball + additional_pairs * 2 = total_snakes ∧ 
    num_balls = 3 := by
  sorry

end mary_breeding_balls_l1708_170864


namespace white_sox_wins_l1708_170891

theorem white_sox_wins (total_games : ℕ) (games_lost : ℕ) (win_difference : ℕ) : 
  total_games = 162 →
  games_lost = 63 →
  win_difference = 36 →
  total_games = games_lost + (games_lost + win_difference) →
  games_lost + win_difference = 99 := by
sorry

end white_sox_wins_l1708_170891


namespace special_triangle_property_l1708_170832

noncomputable section

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the angles of the triangle
def angle_A (t : Triangle) : ℝ := Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))
def angle_B (t : Triangle) : ℝ := Real.arccos ((t.c^2 + t.a^2 - t.b^2) / (2 * t.c * t.a))
def angle_C (t : Triangle) : ℝ := Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

-- The main theorem
theorem special_triangle_property (t : Triangle) 
  (h : t.b * (t.a + t.b) * (t.b + t.c) = t.a^3 + t.b * (t.a^2 + t.c^2) + t.c^3) :
  1 / (Real.sqrt (angle_A t) + Real.sqrt (angle_B t)) + 
  1 / (Real.sqrt (angle_B t) + Real.sqrt (angle_C t)) = 
  2 / (Real.sqrt (angle_C t) + Real.sqrt (angle_A t)) :=
sorry

end

end special_triangle_property_l1708_170832


namespace probability_at_least_three_white_balls_l1708_170839

theorem probability_at_least_three_white_balls 
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (black_balls : ℕ) 
  (drawn_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 8)
  (h3 : black_balls = 7)
  (h4 : drawn_balls = 5) :
  let favorable_outcomes := Nat.choose white_balls 3 * Nat.choose black_balls 2 +
                            Nat.choose white_balls 4 * Nat.choose black_balls 1 +
                            Nat.choose white_balls 5 * Nat.choose black_balls 0
  let total_outcomes := Nat.choose total_balls drawn_balls
  (favorable_outcomes : ℚ) / total_outcomes = 1722 / 3003 := by
sorry

end probability_at_least_three_white_balls_l1708_170839


namespace journey_time_calculation_l1708_170844

/-- Given a journey with the following conditions:
    - Total distance is 224 km
    - Journey is divided into two equal halves
    - First half is traveled at 21 km/hr
    - Second half is traveled at 24 km/hr
    The total time taken to complete the journey is 10 hours. -/
theorem journey_time_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_speed : ℝ) :
  total_distance = 224 →
  first_half_speed = 21 →
  second_half_speed = 24 →
  (total_distance / 2 / first_half_speed) + (total_distance / 2 / second_half_speed) = 10 := by
sorry

end journey_time_calculation_l1708_170844


namespace remainder_double_n_l1708_170878

theorem remainder_double_n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end remainder_double_n_l1708_170878


namespace cubic_polynomials_with_specific_roots_and_difference_l1708_170870

/-- Two monic cubic polynomials with specific roots and a constant difference -/
theorem cubic_polynomials_with_specific_roots_and_difference (f g : ℝ → ℝ) (r : ℝ) :
  (∀ x, f x = (x - (r + 1)) * (x - (r + 7)) * (x - (3 * r + 8))) →  -- f is monic cubic with roots r+1, r+7, and 3r+8
  (∀ x, g x = (x - (r + 3)) * (x - (r + 9)) * (x - (3 * r + 12))) →  -- g is monic cubic with roots r+3, r+9, and 3r+12
  (∀ x, f x - g x = r) →  -- constant difference between f and g
  r = 32 := by
sorry

end cubic_polynomials_with_specific_roots_and_difference_l1708_170870


namespace lily_account_balance_l1708_170858

def initial_amount : ℕ := 55
def shirt_cost : ℕ := 7

theorem lily_account_balance :
  initial_amount - (shirt_cost + 3 * shirt_cost) = 27 :=
by sorry

end lily_account_balance_l1708_170858
