import Mathlib

namespace money_sharing_problem_l2128_212853

theorem money_sharing_problem (amanda_share ben_share carlos_share total : ℕ) : 
  amanda_share = 30 ∧ 
  ben_share = 2 * amanda_share + 10 ∧
  amanda_share + ben_share + carlos_share = total ∧
  3 * ben_share = 4 * amanda_share ∧
  3 * carlos_share = 9 * amanda_share →
  total = 190 := by sorry

end money_sharing_problem_l2128_212853


namespace abs_equation_solution_l2128_212820

theorem abs_equation_solution :
  ∃! x : ℝ, |x + 4| = 3 - x :=
by
  -- Proof goes here
  sorry

end abs_equation_solution_l2128_212820


namespace divided_number_problem_l2128_212848

theorem divided_number_problem (x y : ℝ) : 
  x > y ∧ y = 11 ∧ 7 * x + 5 * y = 146 → x + y = 24 := by
  sorry

end divided_number_problem_l2128_212848


namespace line_passes_through_point_l2128_212811

/-- The line mx + y - m = 0 passes through the point (1, 0) for all real m. -/
theorem line_passes_through_point :
  ∀ (m : ℝ), m * 1 + 0 - m = 0 := by sorry

end line_passes_through_point_l2128_212811


namespace max_value_of_g_l2128_212899

def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

theorem max_value_of_g :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.sqrt 2 ∧
  g x = 25/8 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ Real.sqrt 2 → g y ≤ g x :=
by sorry

end max_value_of_g_l2128_212899


namespace gcf_of_90_and_126_l2128_212887

theorem gcf_of_90_and_126 : Nat.gcd 90 126 = 18 := by
  sorry

end gcf_of_90_and_126_l2128_212887


namespace geometric_sequences_operations_l2128_212856

/-- A sequence is geometric if the ratio of consecutive terms is constant and non-zero -/
def IsGeometricSequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, s (n + 1) = r * s n

theorem geometric_sequences_operations
  (a b : ℕ → ℝ)
  (ha : IsGeometricSequence a)
  (hb : IsGeometricSequence b) :
  IsGeometricSequence (fun n ↦ a n * b n) ∧
  IsGeometricSequence (fun n ↦ a n / b n) ∧
  ¬ (∀ a b : ℕ → ℝ, IsGeometricSequence a → IsGeometricSequence b → IsGeometricSequence (fun n ↦ a n + b n)) ∧
  ¬ (∀ a b : ℕ → ℝ, IsGeometricSequence a → IsGeometricSequence b → IsGeometricSequence (fun n ↦ a n - b n)) :=
by sorry


end geometric_sequences_operations_l2128_212856


namespace f_properties_l2128_212894

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - x) * Real.exp x - 1

theorem f_properties :
  ∃ (a : ℝ),
    (∀ x ≠ 0, f a x / x < 1) ∧
    (∀ x : ℝ, f 1 x ≤ 0) ∧
    (f 1 0 = 0) :=
sorry

end f_properties_l2128_212894


namespace sum_of_popsicle_sticks_l2128_212876

/-- The number of popsicle sticks Gino has -/
def gino_sticks : ℕ := 63

/-- The number of popsicle sticks I have -/
def my_sticks : ℕ := 50

/-- The sum of Gino's and my popsicle sticks -/
def total_sticks : ℕ := gino_sticks + my_sticks

theorem sum_of_popsicle_sticks : total_sticks = 113 := by sorry

end sum_of_popsicle_sticks_l2128_212876


namespace hyperbola_condition_l2128_212816

-- Define the equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 1) - y^2 / (k + 2) = 1

-- Define the condition
def condition (k : ℝ) : Prop := 0 < k ∧ k < 1

-- Theorem statement
theorem hyperbola_condition :
  ¬(∀ k : ℝ, is_hyperbola k ↔ condition k) :=
sorry

end hyperbola_condition_l2128_212816


namespace rectangle_shorter_side_l2128_212872

theorem rectangle_shorter_side
  (area : ℝ)
  (perimeter : ℝ)
  (h_area : area = 117)
  (h_perimeter : perimeter = 44)
  : ∃ (short_side long_side : ℝ),
    short_side * long_side = area ∧
    2 * (short_side + long_side) = perimeter ∧
    short_side = 9 ∧
    short_side ≤ long_side :=
by
  sorry

end rectangle_shorter_side_l2128_212872


namespace max_value_sine_cosine_l2128_212837

theorem max_value_sine_cosine (x : Real) : 
  0 ≤ x → x < 2 * Real.pi → 
  ∃ (max_x : Real), max_x = 5 * Real.pi / 6 ∧
    ∀ y : Real, 0 ≤ y → y < 2 * Real.pi → 
      Real.sin x - Real.sqrt 3 * Real.cos x ≤ Real.sin max_x - Real.sqrt 3 * Real.cos max_x :=
by sorry

end max_value_sine_cosine_l2128_212837


namespace arccos_cos_three_pi_half_l2128_212828

theorem arccos_cos_three_pi_half : Real.arccos (Real.cos (3 * π / 2)) = π / 2 := by
  sorry

end arccos_cos_three_pi_half_l2128_212828


namespace store_exit_ways_l2128_212857

/-- The number of different oreo flavors --/
def oreo_flavors : ℕ := 8

/-- The number of different milk types --/
def milk_types : ℕ := 4

/-- The total number of items Charlie can choose from --/
def charlie_choices : ℕ := oreo_flavors + milk_types

/-- The total number of products they leave with --/
def total_products : ℕ := 5

/-- Function to calculate the number of ways Delta can choose n oreos --/
def delta_choices (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then oreo_flavors
  else if n = 2 then (Nat.choose oreo_flavors 2) + oreo_flavors
  else if n = 3 then (Nat.choose oreo_flavors 3) + oreo_flavors * (oreo_flavors - 1) + oreo_flavors
  else if n = 4 then (Nat.choose oreo_flavors 4) + (Nat.choose oreo_flavors 2) * (Nat.choose (oreo_flavors - 2) 2) / 2 + oreo_flavors * (oreo_flavors - 1) + oreo_flavors
  else (Nat.choose oreo_flavors 5) + (Nat.choose oreo_flavors 2) * (Nat.choose (oreo_flavors - 2) 3) + oreo_flavors * (Nat.choose (oreo_flavors - 1) 2) + oreo_flavors

/-- The total number of ways Charlie and Delta could have left the store --/
def total_ways : ℕ :=
  (Nat.choose charlie_choices total_products) +
  (Nat.choose charlie_choices 4) * (delta_choices 1) +
  (Nat.choose charlie_choices 3) * (delta_choices 2) +
  (Nat.choose charlie_choices 2) * (delta_choices 3) +
  (Nat.choose charlie_choices 1) * (delta_choices 4) +
  (delta_choices 5)

theorem store_exit_ways : total_ways = 25512 := by
  sorry

end store_exit_ways_l2128_212857


namespace solve_equation_l2128_212818

theorem solve_equation : ∃ y : ℝ, (3 * y - 15) / 7 = 18 ∧ y = 47 := by sorry

end solve_equation_l2128_212818


namespace functional_equation_solution_l2128_212824

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y = f (x - y)) →
  ((∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1)) :=
by sorry

end functional_equation_solution_l2128_212824


namespace trig_expression_equals_sqrt_two_l2128_212823

theorem trig_expression_equals_sqrt_two :
  (Real.cos (-585 * π / 180)) / (Real.tan (495 * π / 180) + Real.sin (-690 * π / 180)) = Real.sqrt 2 := by
  sorry

end trig_expression_equals_sqrt_two_l2128_212823


namespace added_amount_l2128_212810

theorem added_amount (n : ℝ) (x : ℝ) (h1 : n = 12) (h2 : n / 2 + x = 11) : x = 5 := by
  sorry

end added_amount_l2128_212810


namespace johns_age_l2128_212802

theorem johns_age (john_age father_age : ℕ) : 
  john_age + father_age = 77 →
  father_age = 2 * john_age + 32 →
  john_age = 15 := by
sorry

end johns_age_l2128_212802


namespace quadratic_expression_value_l2128_212886

theorem quadratic_expression_value (a : ℝ) (h : a^2 + 4*a - 5 = 0) : 3*a^2 + 12*a = 15 := by
  sorry

end quadratic_expression_value_l2128_212886


namespace smallest_b_is_85_l2128_212830

/-- A pair of integers that multiply to give 1764 -/
def ValidPair : Type := { p : ℤ × ℤ // p.1 * p.2 = 1764 }

/-- Predicate to check if a number is a perfect square -/
def IsPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

/-- The sum of a valid pair -/
def PairSum (p : ValidPair) : ℤ := p.val.1 + p.val.2

theorem smallest_b_is_85 :
  (∃ (b : ℕ), 
    (∃ (p : ValidPair), PairSum p = b) ∧ 
    (∃ (p : ValidPair), IsPerfectSquare p.val.1 ∨ IsPerfectSquare p.val.2) ∧
    (∀ (b' : ℕ), b' < b → 
      (∀ (p : ValidPair), PairSum p ≠ b' ∨ 
        (¬ IsPerfectSquare p.val.1 ∧ ¬ IsPerfectSquare p.val.2)))) ∧
  (∀ (b : ℕ), 
    ((∃ (p : ValidPair), PairSum p = b) ∧ 
     (∃ (p : ValidPair), IsPerfectSquare p.val.1 ∨ IsPerfectSquare p.val.2) ∧
     (∀ (b' : ℕ), b' < b → 
       (∀ (p : ValidPair), PairSum p ≠ b' ∨ 
         (¬ IsPerfectSquare p.val.1 ∧ ¬ IsPerfectSquare p.val.2))))
    → b = 85) :=
sorry

end smallest_b_is_85_l2128_212830


namespace f_composition_eq_one_fourth_l2128_212865

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

theorem f_composition_eq_one_fourth :
  f (f (1/9)) = 1/4 := by sorry

end f_composition_eq_one_fourth_l2128_212865


namespace two_numbers_solution_l2128_212851

theorem two_numbers_solution : ∃ (x y : ℝ), 
  (2/3 : ℝ) * x + 2 * y = 20 ∧ 
  (1/4 : ℝ) * x - y = 2 ∧ 
  x = 144/7 ∧ 
  y = 22/7 := by
  sorry

end two_numbers_solution_l2128_212851


namespace paul_running_time_l2128_212833

/-- Given that Paul watches movies while running on a treadmill, prove that it takes him 12 minutes to run one mile. -/
theorem paul_running_time (num_movies : ℕ) (avg_movie_length : ℝ) (total_miles : ℝ) :
  num_movies = 2 →
  avg_movie_length = 1.5 →
  total_miles = 15 →
  (num_movies * avg_movie_length * 60) / total_miles = 12 := by
  sorry

end paul_running_time_l2128_212833


namespace money_sharing_l2128_212871

theorem money_sharing (total : ℕ) (amanda ben carlos : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 24 →
  2 * ben = 3 * amanda →
  8 * amanda = 3 * carlos →
  total = 156 := by
sorry

end money_sharing_l2128_212871


namespace triangle_abc_cosine_sine_l2128_212893

theorem triangle_abc_cosine_sine (A B C : ℝ) (cosC_half : ℝ) (BC AC : ℝ) :
  cosC_half = Real.sqrt 5 / 5 →
  BC = 1 →
  AC = 5 →
  (Real.cos C = -3/5 ∧ Real.sin A = Real.sqrt 2 / 10) :=
by sorry

end triangle_abc_cosine_sine_l2128_212893


namespace twelve_sided_die_expected_value_l2128_212804

-- Define the number of sides on the die
def n : ℕ := 12

-- Define the expected value function for a fair die with n sides
def expected_value (n : ℕ) : ℚ :=
  (↑n + 1) / 2

-- Theorem statement
theorem twelve_sided_die_expected_value :
  expected_value n = 13/2 := by
  sorry

end twelve_sided_die_expected_value_l2128_212804


namespace exists_angle_leq_90_degrees_l2128_212814

-- Define a type for rays in space
def Ray : Type := ℝ → ℝ × ℝ × ℝ

-- Define a function to calculate the angle between two rays
def angle_between_rays (r1 r2 : Ray) : ℝ := sorry

-- State the theorem
theorem exists_angle_leq_90_degrees (rays : Fin 5 → Ray) 
  (h_distinct : ∀ i j, i ≠ j → rays i ≠ rays j) : 
  ∃ i j, i ≠ j ∧ angle_between_rays (rays i) (rays j) ≤ 90 := by sorry

end exists_angle_leq_90_degrees_l2128_212814


namespace correct_calculation_l2128_212825

theorem correct_calculation (m : ℝ) : 2 * m^3 * 3 * m^2 = 6 * m^5 := by
  sorry

end correct_calculation_l2128_212825


namespace painters_work_days_l2128_212852

theorem painters_work_days (painters_initial : ℕ) (painters_new : ℕ) (days_initial : ℚ) : 
  painters_initial = 5 → 
  painters_new = 4 → 
  days_initial = 3/2 → 
  ∃ (days_new : ℚ), days_new = 15/8 ∧ 
    painters_initial * days_initial = painters_new * days_new :=
by sorry

end painters_work_days_l2128_212852


namespace jim_purchase_total_l2128_212815

/-- Calculate the total amount Jim paid for lamps and bulbs --/
theorem jim_purchase_total : 
  let lamp_cost : ℚ := 7
  let bulb_cost : ℚ := lamp_cost - 4
  let lamp_quantity : ℕ := 2
  let bulb_quantity : ℕ := 6
  let tax_rate : ℚ := 5 / 100
  let bulb_discount : ℚ := 10 / 100
  let total_lamp_cost : ℚ := lamp_cost * lamp_quantity
  let total_bulb_cost : ℚ := bulb_cost * bulb_quantity
  let discounted_bulb_cost : ℚ := total_bulb_cost * (1 - bulb_discount)
  let subtotal : ℚ := total_lamp_cost + discounted_bulb_cost
  let tax_amount : ℚ := subtotal * tax_rate
  let total_cost : ℚ := subtotal + tax_amount
  total_cost = 3171 / 100 := by sorry

end jim_purchase_total_l2128_212815


namespace complement_of_A_range_of_m_for_subset_range_of_m_for_disjoint_l2128_212844

-- Define the sets A and B
def A : Set ℝ := {x | -x^2 - 3*x > 0}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- Theorem for the complement of A
theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x | x ≤ -3 ∨ x ≥ 0} := by sorry

-- Theorem for the range of m when A is a subset of B
theorem range_of_m_for_subset : 
  ∀ m : ℝ, A ⊆ B m → m ≥ 0 := by sorry

-- Theorem for the range of m when A and B are disjoint
theorem range_of_m_for_disjoint : 
  ∀ m : ℝ, A ∩ B m = ∅ → m ≤ -3 := by sorry

end complement_of_A_range_of_m_for_subset_range_of_m_for_disjoint_l2128_212844


namespace contrapositive_truth_l2128_212881

theorem contrapositive_truth (p q : Prop) : 
  (q → p) → (¬p → ¬q) := by sorry

end contrapositive_truth_l2128_212881


namespace friendly_sequences_exist_l2128_212868

/-- Definition of a friendly pair of sequences -/
def is_friendly_pair (a b : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0 ∧ b n > 0) ∧
  (∀ k : ℕ, ∃! (i j : ℕ), a i * b j = k)

/-- Theorem stating the existence of friendly sequences -/
theorem friendly_sequences_exist : ∃ (a b : ℕ → ℕ), is_friendly_pair a b :=
sorry

end friendly_sequences_exist_l2128_212868


namespace weight_lifting_competition_l2128_212809

theorem weight_lifting_competition (total_weight first_lift : ℕ) 
  (h1 : total_weight = 1800)
  (h2 : first_lift = 700) :
  2 * first_lift - (total_weight - first_lift) = 300 := by
  sorry

end weight_lifting_competition_l2128_212809


namespace connie_blue_markers_l2128_212812

/-- Given the total number of markers and the number of red markers,
    calculate the number of blue markers. -/
def blue_markers (total : ℕ) (red : ℕ) : ℕ := total - red

/-- Theorem stating that Connie has 64 blue markers -/
theorem connie_blue_markers :
  let total_markers : ℕ := 105
  let red_markers : ℕ := 41
  blue_markers total_markers red_markers = 64 := by
  sorry

end connie_blue_markers_l2128_212812


namespace smallest_integer_of_three_l2128_212879

theorem smallest_integer_of_three (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 100 →
  2 * b = 3 * a →
  2 * c = 5 * a →
  a = 20 := by sorry

end smallest_integer_of_three_l2128_212879


namespace investment_percentage_proof_l2128_212800

/-- Proves that given the investment conditions, the percentage at which $3,500 was invested is 4% --/
theorem investment_percentage_proof (total_investment : ℝ) (investment1 : ℝ) (investment2 : ℝ) 
  (rate1 : ℝ) (rate3 : ℝ) (desired_income : ℝ) (x : ℝ) :
  total_investment = 10000 →
  investment1 = 4000 →
  investment2 = 3500 →
  rate1 = 0.05 →
  rate3 = 0.064 →
  desired_income = 500 →
  investment1 * rate1 + investment2 * (x / 100) + (total_investment - investment1 - investment2) * rate3 = desired_income →
  x = 4 := by
sorry

end investment_percentage_proof_l2128_212800


namespace limit_of_a_l2128_212889

def a (n : ℕ) : ℚ := (3 * n - 1) / (5 * n + 1)

theorem limit_of_a : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 3/5| < ε :=
sorry

end limit_of_a_l2128_212889


namespace triangle_properties_l2128_212885

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A + t.B = 3 * t.C ∧
  2 * Real.sin (t.A - t.C) = Real.sin t.B ∧
  t.AB = 5

-- Define the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  Real.sin t.A = 3 * (10 ^ (1/2 : ℝ)) / 10 ∧
  ∃ (height : ℝ), height = 6 ∧ 
    height * t.AB / 2 = Real.sin t.C * (Real.sin t.A * t.AB / Real.sin t.C) * (Real.sin t.B * t.AB / Real.sin t.C) / 2 :=
sorry

end triangle_properties_l2128_212885


namespace remainder_problem_l2128_212803

theorem remainder_problem : (7 * 7^10 + 1^10) % 11 = 8 := by
  sorry

end remainder_problem_l2128_212803


namespace average_stream_speed_theorem_l2128_212883

/-- Represents the swimming scenario with given parameters. -/
structure SwimmingScenario where
  swimmer_speed : ℝ  -- Speed of the swimmer in still water (km/h)
  upstream_time_ratio : ℝ  -- Ratio of upstream time to downstream time
  stream_speed_increase : ℝ  -- Increase in stream speed per 100 meters (km/h)
  upstream_distance : ℝ  -- Total upstream distance (meters)

/-- Calculates the average stream speed over the given distance. -/
def average_stream_speed (scenario : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating the average stream speed for the given scenario. -/
theorem average_stream_speed_theorem (scenario : SwimmingScenario) 
  (h1 : scenario.swimmer_speed = 1.5)
  (h2 : scenario.upstream_time_ratio = 2)
  (h3 : scenario.stream_speed_increase = 0.2)
  (h4 : scenario.upstream_distance = 500) :
  average_stream_speed scenario = 0.7 :=
sorry

end average_stream_speed_theorem_l2128_212883


namespace nested_expression_evaluation_l2128_212817

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) = 161 := by
  sorry

end nested_expression_evaluation_l2128_212817


namespace barbara_paper_problem_l2128_212805

/-- The number of sheets in a bundle -/
def sheets_per_bundle : ℕ := 2

/-- The number of sheets in a heap -/
def sheets_per_heap : ℕ := 20

/-- The number of bundles Barbara found -/
def num_bundles : ℕ := 3

/-- The number of bunches Barbara found -/
def num_bunches : ℕ := 2

/-- The number of heaps Barbara found -/
def num_heaps : ℕ := 5

/-- The total number of sheets Barbara removed -/
def total_sheets : ℕ := 114

/-- The number of sheets in a bunch -/
def sheets_per_bunch : ℕ := 4

theorem barbara_paper_problem :
  sheets_per_bunch * num_bunches + sheets_per_bundle * num_bundles + sheets_per_heap * num_heaps = total_sheets :=
by sorry

end barbara_paper_problem_l2128_212805


namespace trigonometric_expression_equals_one_l2128_212834

open Real

theorem trigonometric_expression_equals_one : 
  (sin (15 * π / 180) * cos (15 * π / 180) + cos (165 * π / 180) * cos (105 * π / 180)) /
  (sin (19 * π / 180) * cos (11 * π / 180) + cos (161 * π / 180) * cos (101 * π / 180)) = 1 := by
  sorry

end trigonometric_expression_equals_one_l2128_212834


namespace sqrt_six_diamond_sqrt_six_l2128_212821

-- Define the operation ¤
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- Theorem statement
theorem sqrt_six_diamond_sqrt_six : diamond (Real.sqrt 6) (Real.sqrt 6) = 24 := by
  sorry

end sqrt_six_diamond_sqrt_six_l2128_212821


namespace fraction_calculation_l2128_212892

theorem fraction_calculation (N : ℝ) (h : 0.4 * N = 240) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 20 := by
  sorry

end fraction_calculation_l2128_212892


namespace slope_angle_of_intersecting_line_l2128_212870

/-- The slope angle of a line intersecting a circle -/
theorem slope_angle_of_intersecting_line (α : Real) : 
  (∃ (A B : ℝ × ℝ), 
    (∀ t : ℝ, (1 + t * Real.cos α, t * Real.sin α) ∈ {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 4}) →
    A ∈ {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 4} →
    B ∈ {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 4} →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14) →
  α = π/4 ∨ α = 3*π/4 := by
sorry

end slope_angle_of_intersecting_line_l2128_212870


namespace linear_system_solution_l2128_212835

theorem linear_system_solution (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) 
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (given : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
  sorry

end linear_system_solution_l2128_212835


namespace word_arrangements_count_l2128_212847

def word_length : ℕ := 12
def repeated_letter_1_count : ℕ := 3
def repeated_letter_2_count : ℕ := 2
def repeated_letter_3_count : ℕ := 2
def unique_letters_count : ℕ := 5

def arrangements_count : ℕ := 19958400

theorem word_arrangements_count :
  (word_length.factorial) / 
  (repeated_letter_1_count.factorial * 
   repeated_letter_2_count.factorial * 
   repeated_letter_3_count.factorial) = arrangements_count := by
  sorry

end word_arrangements_count_l2128_212847


namespace parabola_hyperbola_equations_l2128_212806

/-- Given a parabola and a hyperbola with specific properties, prove their equations -/
theorem parabola_hyperbola_equations :
  ∀ (a b : ℝ) (P : ℝ × ℝ),
    a > 0 → b > 0 →
    P = (3/2, Real.sqrt 6) →
    -- Parabola vertex at origin
    -- Directrix of parabola passes through a focus of hyperbola
    -- Directrix perpendicular to line connecting foci of hyperbola
    -- Parabola and hyperbola intersect at P
    ∃ (p : ℝ),
      -- Parabola equation
      (λ (x y : ℝ) => y^2 = 2*p*x) P.1 P.2 ∧
      -- Hyperbola equation
      (λ (x y : ℝ) => x^2/a^2 - y^2/b^2 = 1) P.1 P.2 →
      -- Prove the specific equations
      (λ (x y : ℝ) => y^2 = 4*x) = (λ (x y : ℝ) => y^2 = 2*p*x) ∧
      (λ (x y : ℝ) => 4*x^2 - 4/3*y^2 = 1) = (λ (x y : ℝ) => x^2/a^2 - y^2/b^2 = 1) := by
  sorry

end parabola_hyperbola_equations_l2128_212806


namespace greatest_prime_factor_of_341_l2128_212866

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l2128_212866


namespace library_visitors_equation_l2128_212867

/-- Represents the equation for library visitors over three months -/
theorem library_visitors_equation 
  (initial_visitors : ℕ) 
  (growth_rate : ℝ) 
  (total_visitors : ℕ) :
  initial_visitors + 
  initial_visitors * (1 + growth_rate) + 
  initial_visitors * (1 + growth_rate)^2 = total_visitors ↔ 
  initial_visitors = 600 ∧ 
  growth_rate > 0 ∧ 
  total_visitors = 2850 :=
by sorry

end library_visitors_equation_l2128_212867


namespace milk_problem_l2128_212827

theorem milk_problem (initial_milk : ℚ) (rachel_fraction : ℚ) (jack_fraction : ℚ) :
  initial_milk = 3/4 →
  rachel_fraction = 5/8 →
  jack_fraction = 1/2 →
  (initial_milk - rachel_fraction * initial_milk) * jack_fraction = 9/64 := by
  sorry

end milk_problem_l2128_212827


namespace parrot_guinea_pig_ownership_l2128_212884

theorem parrot_guinea_pig_ownership (total : ℕ) (parrot : ℕ) (guinea_pig : ℕ) :
  total = 48 →
  parrot = 30 →
  guinea_pig = 35 →
  ∃ (both : ℕ), both = 17 ∧ total = parrot + guinea_pig - both :=
by
  sorry

end parrot_guinea_pig_ownership_l2128_212884


namespace binary_10111_equals_23_l2128_212863

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_10111_equals_23 : 
  binary_to_decimal [true, true, true, false, true] = 23 := by
  sorry

end binary_10111_equals_23_l2128_212863


namespace nancy_keeps_ten_l2128_212843

def nancy_chips : ℕ := 22
def brother_chips : ℕ := 7
def sister_chips : ℕ := 5

theorem nancy_keeps_ten : 
  nancy_chips - (brother_chips + sister_chips) = 10 := by
  sorry

end nancy_keeps_ten_l2128_212843


namespace decimal_to_percentage_l2128_212877

theorem decimal_to_percentage (x : ℝ) (h : x = 0.005) : x * 100 = 0.5 := by
  sorry

end decimal_to_percentage_l2128_212877


namespace complex_square_roots_l2128_212855

theorem complex_square_roots (z : ℂ) : z^2 = -45 - 54*I ↔ z = 3 - 9*I ∨ z = -3 + 9*I := by
  sorry

end complex_square_roots_l2128_212855


namespace vector_decomposition_l2128_212831

def x : ℝ × ℝ × ℝ := (-5, -5, 5)
def p : ℝ × ℝ × ℝ := (-2, 0, 1)
def q : ℝ × ℝ × ℝ := (1, 3, -1)
def r : ℝ × ℝ × ℝ := (0, 4, 1)

theorem vector_decomposition :
  x = p + (-3 : ℝ) • q + r := by sorry

end vector_decomposition_l2128_212831


namespace brochure_distribution_l2128_212840

theorem brochure_distribution (total_brochures : ℕ) (num_boxes : ℕ) 
  (h1 : total_brochures = 5000) 
  (h2 : num_boxes = 5) : 
  (total_brochures / num_boxes : ℚ) / total_brochures = 1 / 5 := by
  sorry

end brochure_distribution_l2128_212840


namespace tuesday_calls_l2128_212845

/-- Represents the number of calls answered by Jean for each day of the work week -/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total number of calls answered in a week -/
def totalCalls (w : WeekCalls) : ℕ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday

/-- Calculates the average number of calls per day -/
def averageCalls (w : WeekCalls) : ℚ :=
  totalCalls w / 5

theorem tuesday_calls (w : WeekCalls) 
  (h1 : w.monday = 35)
  (h2 : w.wednesday = 27)
  (h3 : w.thursday = 61)
  (h4 : w.friday = 31)
  (h5 : averageCalls w = 40) :
  w.tuesday = 46 := by
  sorry

#check tuesday_calls

end tuesday_calls_l2128_212845


namespace lamps_with_burnt_bulbs_l2128_212869

/-- Given a set of lamps with some burnt-out bulbs, proves the number of bulbs per lamp -/
theorem lamps_with_burnt_bulbs 
  (total_lamps : ℕ) 
  (burnt_fraction : ℚ) 
  (burnt_per_lamp : ℕ) 
  (working_bulbs : ℕ) : 
  total_lamps = 20 → 
  burnt_fraction = 1/4 → 
  burnt_per_lamp = 2 → 
  working_bulbs = 130 → 
  (total_lamps * (burnt_fraction * burnt_per_lamp + (1 - burnt_fraction) * working_bulbs / total_lamps)) / total_lamps = 7 := by
sorry

end lamps_with_burnt_bulbs_l2128_212869


namespace condition_equivalence_l2128_212862

-- Define the sets A, B, and C
def A : Set ℝ := {x | x - 2 > 0}
def B : Set ℝ := {x | x < 0}
def C : Set ℝ := {x | x * (x - 2) > 0}

-- State the theorem
theorem condition_equivalence : ∀ x : ℝ, x ∈ A ∪ B ↔ x ∈ C := by
  sorry

end condition_equivalence_l2128_212862


namespace bug_pentagon_probability_l2128_212849

/-- Probability of the bug being at the starting vertex after n moves -/
def Q (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | 1 => 0
  | 2 => 1/2
  | n+1 => 1/2 * (1 - Q n)

/-- The probability of returning to the starting vertex on the 12th move in a regular pentagon -/
theorem bug_pentagon_probability : Q 12 = 341/1024 := by
  sorry

end bug_pentagon_probability_l2128_212849


namespace quadrilateral_area_l2128_212846

/-- The area of a quadrilateral with vertices A(1, 3), B(1, 1), C(5, 6), and D(4, 3) is 8.5 square units. -/
theorem quadrilateral_area : 
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (5, 6)
  let D : ℝ × ℝ := (4, 3)
  let area := abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2 +
              abs (A.1 * (C.2 - D.2) + C.1 * (D.2 - A.2) + D.1 * (A.2 - C.2)) / 2
  area = 8.5 := by
  sorry

end quadrilateral_area_l2128_212846


namespace winning_probability_is_five_eighths_l2128_212854

/-- Represents the color of a ball in the lottery bag -/
inductive BallColor
  | Red
  | Yellow
  | White
  | Black

/-- Represents the lottery bag -/
structure LotteryBag where
  total_balls : ℕ
  red_balls : ℕ
  yellow_balls : ℕ
  black_balls : ℕ
  white_balls : ℕ
  h_total : total_balls = red_balls + yellow_balls + black_balls + white_balls

/-- Calculates the probability of winning in the lottery -/
def winning_probability (bag : LotteryBag) : ℚ :=
  (bag.red_balls + bag.yellow_balls + bag.white_balls : ℚ) / bag.total_balls

/-- The lottery bag configuration -/
def lottery_bag : LotteryBag := {
  total_balls := 24
  red_balls := 3
  yellow_balls := 6
  black_balls := 9
  white_balls := 6
  h_total := by rfl
}

/-- Theorem: The probability of winning in the given lottery bag is 5/8 -/
theorem winning_probability_is_five_eighths :
  winning_probability lottery_bag = 5/8 := by
  sorry

end winning_probability_is_five_eighths_l2128_212854


namespace fraction_evaluation_l2128_212861

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by sorry

end fraction_evaluation_l2128_212861


namespace age_problem_l2128_212875

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 12 → 
  b = 4 := by
sorry

end age_problem_l2128_212875


namespace range_of_m_l2128_212836

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬ is_periodic f q

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  is_odd f →
  smallest_positive_period f 3 →
  f 1 > -2 →
  f 2 = m^2 - m →
  m ∈ Set.Ioo (-1 : ℝ) 2 :=
by sorry

end range_of_m_l2128_212836


namespace quadratic_roots_relation_l2128_212897

/-- Given two quadratic equations, where the roots of the first are three less than the roots of the second, 
    this theorem proves that the constant term of the first equation is -14.5 -/
theorem quadratic_roots_relation (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 ↔ ∃ y, 2*y^2 - 11*y - 14 = 0 ∧ x = y - 3) →
  c = -14.5 := by
sorry

end quadratic_roots_relation_l2128_212897


namespace cookies_per_bag_l2128_212841

/-- Given 26 bags with an equal number of cookies and 52 cookies in total,
    prove that each bag contains 2 cookies. -/
theorem cookies_per_bag :
  ∀ (num_bags : ℕ) (total_cookies : ℕ) (cookies_per_bag : ℕ),
    num_bags = 26 →
    total_cookies = 52 →
    num_bags * cookies_per_bag = total_cookies →
    cookies_per_bag = 2 :=
by
  sorry

end cookies_per_bag_l2128_212841


namespace angle_at_point_l2128_212842

theorem angle_at_point (x : ℝ) : 
  (x + x + 160 = 360) → x = 100 := by sorry

end angle_at_point_l2128_212842


namespace cafeteria_students_l2128_212859

theorem cafeteria_students (total : ℕ) (no_lunch : ℕ) (cafeteria : ℕ) : 
  total = 60 → 
  no_lunch = 20 → 
  total = cafeteria + 3 * cafeteria + no_lunch → 
  cafeteria = 10 := by
sorry

end cafeteria_students_l2128_212859


namespace processing_box_function_l2128_212890

-- Define the types of boxes in a flowchart
inductive FlowchartBox
  | Processing
  | Decision
  | Terminal
  | InputOutput

-- Define the functions of boxes in a flowchart
def boxFunction : FlowchartBox → String
  | FlowchartBox.Processing => "assignment and calculation"
  | FlowchartBox.Decision => "determine execution direction"
  | FlowchartBox.Terminal => "start and end of algorithm"
  | FlowchartBox.InputOutput => "handle data input and output"

-- Theorem statement
theorem processing_box_function :
  boxFunction FlowchartBox.Processing = "assignment and calculation" := by
  sorry

end processing_box_function_l2128_212890


namespace haley_trees_l2128_212860

theorem haley_trees (initial_trees : ℕ) (dead_trees : ℕ) (final_trees : ℕ) 
  (h1 : initial_trees = 9)
  (h2 : dead_trees = 4)
  (h3 : final_trees = 10) :
  final_trees - (initial_trees - dead_trees) = 5 := by
sorry

end haley_trees_l2128_212860


namespace tangent_line_curve_n_value_l2128_212801

/-- Given a line and a curve that are tangent at a point, prove the value of n. -/
theorem tangent_line_curve_n_value :
  ∀ (k m n : ℝ),
  (∀ x, k * x + 1 = x^3 + m * x + n → x = 1 ∧ k * x + 1 = 3) →
  (∀ x, (3 * x^2 + m) * (x - 1) + (1^3 + m * 1 + n) = k * x + 1) →
  n = 3 := by
sorry

end tangent_line_curve_n_value_l2128_212801


namespace ellipse_area_ratio_range_l2128_212832

/-- An ellipse with given properties --/
structure Ellipse where
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  passesThrough : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- A line intersecting the ellipse --/
structure IntersectingLine where
  passingThrough : ℝ × ℝ
  intersectionPoints : (ℝ × ℝ) × (ℝ × ℝ)

/-- The ratio of triangle areas --/
def areaRatio (e : Ellipse) (l : IntersectingLine) : ℝ := sorry

theorem ellipse_area_ratio_range 
  (e : Ellipse) 
  (l : IntersectingLine) 
  (h1 : e.foci = ((-Real.sqrt 3, 0), (Real.sqrt 3, 0)))
  (h2 : e.passesThrough = (1, Real.sqrt 3 / 2))
  (h3 : e.equation = fun x y ↦ x^2 / 4 + y^2 = 1)
  (h4 : l.passingThrough = (0, 2))
  (h5 : ∃ (M N : ℝ × ℝ), l.intersectionPoints = (M, N) ∧ 
        e.equation M.1 M.2 ∧ e.equation N.1 N.2 ∧ 
        (∃ t : ℝ, 0 < t ∧ t < 1 ∧ M = (t * l.passingThrough.1 + (1 - t) * N.1, 
                                       t * l.passingThrough.2 + (1 - t) * N.2))) :
  1/3 < areaRatio e l ∧ areaRatio e l < 1 := by sorry

end ellipse_area_ratio_range_l2128_212832


namespace sum_of_triangle_ops_equals_21_l2128_212898

-- Define the triangle operation
def triangle_op (a b c : ℕ) : ℕ := a + b + c

-- Define the two triangles
def triangle1 : (ℕ × ℕ × ℕ) := (2, 4, 3)
def triangle2 : (ℕ × ℕ × ℕ) := (1, 6, 5)

-- Theorem statement
theorem sum_of_triangle_ops_equals_21 :
  triangle_op triangle1.1 triangle1.2.1 triangle1.2.2 +
  triangle_op triangle2.1 triangle2.2.1 triangle2.2.2 = 21 := by
  sorry

end sum_of_triangle_ops_equals_21_l2128_212898


namespace bookcase_weight_theorem_l2128_212822

/-- Represents the weight of the bookcase and items -/
def BookcaseWeightProblem : Prop :=
  let bookcaseLimit : ℕ := 80
  let hardcoverCount : ℕ := 70
  let hardcoverWeight : ℚ := 1/2
  let textbookCount : ℕ := 30
  let textbookWeight : ℕ := 2
  let knickknackCount : ℕ := 3
  let knickknackWeight : ℕ := 6
  let totalWeight : ℚ := 
    hardcoverCount * hardcoverWeight + 
    textbookCount * textbookWeight + 
    knickknackCount * knickknackWeight
  totalWeight - bookcaseLimit = 33

theorem bookcase_weight_theorem : BookcaseWeightProblem := by
  sorry

end bookcase_weight_theorem_l2128_212822


namespace rhombus_area_l2128_212878

/-- The area of a rhombus with side length 4 cm and an angle of 45 degrees between adjacent sides is 8√2 square centimeters. -/
theorem rhombus_area (side_length : ℝ) (angle : ℝ) :
  side_length = 4 →
  angle = π / 4 →
  let area := side_length * side_length * Real.sin angle
  area = 8 * Real.sqrt 2 := by
  sorry

end rhombus_area_l2128_212878


namespace leifeng_pagoda_height_l2128_212880

/-- The height of the Leifeng Pagoda problem -/
theorem leifeng_pagoda_height 
  (AC : ℝ) 
  (α β : ℝ) 
  (h1 : AC = 62 * Real.sqrt 2)
  (h2 : α = 45 * π / 180)
  (h3 : β = 15 * π / 180) :
  ∃ BC : ℝ, BC = 62 :=
sorry

end leifeng_pagoda_height_l2128_212880


namespace square_plus_one_to_zero_is_one_l2128_212838

theorem square_plus_one_to_zero_is_one (m : ℝ) : (m^2 + 1)^0 = 1 := by
  sorry

end square_plus_one_to_zero_is_one_l2128_212838


namespace tangent_line_equation_l2128_212826

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (1, f 1)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (x = tangent_point.1 ∧ y = tangent_point.2) ∨
    (y - tangent_point.2 = m * (x - tangent_point.1)) ↔
    (2 * x - y - 1 = 0) :=
sorry

end tangent_line_equation_l2128_212826


namespace line_vector_to_slope_intercept_l2128_212829

/-- Given a line in vector form, prove its equivalence to slope-intercept form -/
theorem line_vector_to_slope_intercept :
  let vector_line : ℝ × ℝ → Prop :=
    λ p => (3 : ℝ) * (p.1 + 2) + (7 : ℝ) * (p.2 - 8) = 0
  let slope_intercept_line : ℝ × ℝ → Prop :=
    λ p => p.2 = (-3/7 : ℝ) * p.1 + 50/7
  ∀ p : ℝ × ℝ, vector_line p ↔ slope_intercept_line p :=
by sorry

end line_vector_to_slope_intercept_l2128_212829


namespace systematic_sampling_elimination_l2128_212819

/-- The number of individuals randomly eliminated in a systematic sampling -/
def individuals_eliminated (population : ℕ) (sample_size : ℕ) : ℕ :=
  population % sample_size

/-- Theorem: The number of individuals randomly eliminated in a systematic sampling
    of 50 students from a population of 1252 is equal to 2 -/
theorem systematic_sampling_elimination :
  individuals_eliminated 1252 50 = 2 := by
  sorry

end systematic_sampling_elimination_l2128_212819


namespace min_sum_abc_l2128_212873

theorem min_sum_abc (a b c : ℕ+) : 
  a * b * c = 2310 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ c = p * q) →
  (∀ x y z : ℕ+, x * y * z = 2310 → 
    (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ z = p * q) →
    a + b + c ≤ x + y + z) →
  a + b + c = 88 :=
by sorry

end min_sum_abc_l2128_212873


namespace f_max_at_a_l2128_212808

/-- The function f(x) = x^3 - 12x -/
def f (x : ℝ) : ℝ := x^3 - 12*x

/-- The maximum value point of f(x) -/
def a : ℝ := -2

theorem f_max_at_a : IsLocalMax f a := by sorry

end f_max_at_a_l2128_212808


namespace courtyard_width_is_14_l2128_212850

/-- Represents the dimensions of a paving stone -/
structure PavingStone where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a rectangular courtyard -/
structure Courtyard where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular shape -/
def area (length width : ℝ) : ℝ := length * width

/-- Theorem: The width of the courtyard is 14 meters -/
theorem courtyard_width_is_14 (stone : PavingStone) (yard : Courtyard) 
    (h1 : stone.length = 3)
    (h2 : stone.width = 2)
    (h3 : yard.length = 60)
    (h4 : area yard.length yard.width = 140 * area stone.length stone.width) :
  yard.width = 14 := by
  sorry

#check courtyard_width_is_14

end courtyard_width_is_14_l2128_212850


namespace dress_discount_price_l2128_212807

/-- The final price of a dress after applying a discount -/
def final_price (original_price discount_percentage : ℚ) : ℚ :=
  original_price * (1 - discount_percentage / 100)

/-- Theorem stating that a dress originally priced at $350 with a 60% discount costs $140 -/
theorem dress_discount_price : final_price 350 60 = 140 := by
  sorry

end dress_discount_price_l2128_212807


namespace round_repeating_decimal_to_thousandth_l2128_212882

/-- Represents a repeating decimal where the whole number part is 67 and the repeating part is 836 -/
def repeating_decimal : ℚ := 67 + 836 / 999

/-- Rounding function to the nearest thousandth -/
def round_to_thousandth (x : ℚ) : ℚ := 
  (⌊x * 1000 + 1/2⌋ : ℚ) / 1000

theorem round_repeating_decimal_to_thousandth :
  round_to_thousandth repeating_decimal = 67837 / 1000 := by sorry

end round_repeating_decimal_to_thousandth_l2128_212882


namespace apple_cost_is_twelve_l2128_212839

/-- The cost of an apple given the total money, number of apples, and number of kids -/
def apple_cost (total_money : ℕ) (num_apples : ℕ) (num_kids : ℕ) : ℚ :=
  (total_money : ℚ) / (num_apples : ℚ)

/-- Theorem stating that the cost of each apple is 12 dollars -/
theorem apple_cost_is_twelve :
  apple_cost 360 30 6 = 12 := by
  sorry

end apple_cost_is_twelve_l2128_212839


namespace quadrilateral_weighted_centers_l2128_212858

-- Define a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a function to calculate the ratio of distances
def distanceRatio (P Q R : Point) : ℝ :=
  sorry

-- Define the weighted center
def weightedCenter (P Q : Point) (m₁ m₂ : ℝ) : Point :=
  sorry

-- Main theorem
theorem quadrilateral_weighted_centers 
  (quad : Quadrilateral) (P Q R S : Point) :
  (∃ (m₁ m₂ m₃ m₄ : ℝ), 
    P = weightedCenter quad.A quad.B m₁ m₂ ∧
    Q = weightedCenter quad.B quad.C m₂ m₃ ∧
    R = weightedCenter quad.C quad.D m₃ m₄ ∧
    S = weightedCenter quad.D quad.A m₄ m₁) ↔
  distanceRatio quad.A P quad.B *
  distanceRatio quad.B Q quad.C *
  distanceRatio quad.C R quad.D *
  distanceRatio quad.D S quad.A = 1 :=
sorry

end quadrilateral_weighted_centers_l2128_212858


namespace direction_vector_form_l2128_212864

/-- Given a line passing through two points, prove that its direction vector
    has a specific form. -/
theorem direction_vector_form (p1 p2 : ℝ × ℝ) (c : ℝ) : 
  p1 = (-6, 1) →
  p2 = (-1, 5) →
  (p2.1 - p1.1, p2.2 - p1.2) = (5, c) →
  c = 4 := by
  sorry

end direction_vector_form_l2128_212864


namespace cone_base_circumference_l2128_212874

/-- The circumference of the base of a right circular cone with given volume and height -/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (π : ℝ) :
  V = 36 * π →
  h = 3 →
  π > 0 →
  (2 * π * (3 * V / (π * h))^(1/2) : ℝ) = 12 * π := by sorry

end cone_base_circumference_l2128_212874


namespace power_product_rule_l2128_212891

theorem power_product_rule (a : ℝ) : (a * a^3)^2 = a^8 := by
  sorry

end power_product_rule_l2128_212891


namespace factorial_sum_square_solutions_l2128_212895

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

def sum_factorials (n : ℕ) : ℕ := (Finset.range n).sum (λ i => factorial (i + 1))

theorem factorial_sum_square_solutions :
  ∀ n m : ℕ, sum_factorials n = m^2 ↔ (n = 1 ∧ m = 1) ∨ (n = 3 ∧ m = 3) := by
  sorry

end factorial_sum_square_solutions_l2128_212895


namespace symmetric_point_coordinates_l2128_212896

/-- Given a point P in the second quadrant with absolute x-coordinate 5 and absolute y-coordinate 7,
    the point symmetric to P with respect to the origin has coordinates (5, -7). -/
theorem symmetric_point_coordinates :
  ∀ (x y : ℝ),
    x < 0 →  -- Point is in the second quadrant (x is negative)
    y > 0 →  -- Point is in the second quadrant (y is positive)
    |x| = 5 →
    |y| = 7 →
    (- x, - y) = (5, -7) :=
by sorry

end symmetric_point_coordinates_l2128_212896


namespace sqrt_pattern_l2128_212813

theorem sqrt_pattern (n : ℕ) (h : n ≥ 1) : 
  Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2)) := by
  sorry

end sqrt_pattern_l2128_212813


namespace max_regions_five_lines_l2128_212888

/-- The maximum number of regions a rectangle can be divided into by n line segments -/
def maxRegions (n : ℕ) : ℕ :=
  if n = 0 then 1 else maxRegions (n - 1) + n

/-- Theorem: The maximum number of regions a rectangle can be divided into by 5 line segments is 16 -/
theorem max_regions_five_lines :
  maxRegions 5 = 16 := by sorry

end max_regions_five_lines_l2128_212888
