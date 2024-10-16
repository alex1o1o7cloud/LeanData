import Mathlib

namespace NUMINAMATH_CALUDE_proportional_from_equality_l1632_163252

/-- Two real numbers are directly proportional if their ratio is constant -/
def DirectlyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x = k * y

/-- Given x/3 = y/4, prove that x and y are directly proportional -/
theorem proportional_from_equality (x y : ℝ) (h : x / 3 = y / 4) :
  DirectlyProportional x y := by
  sorry

end NUMINAMATH_CALUDE_proportional_from_equality_l1632_163252


namespace NUMINAMATH_CALUDE_all_stars_arrangement_l1632_163204

/-- The number of ways to arrange All-Stars from different teams in a row -/
def arrange_all_stars (cubs : ℕ) (red_sox : ℕ) (yankees : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial cubs) * (Nat.factorial red_sox) * (Nat.factorial yankees)

/-- Theorem stating that there are 6912 ways to arrange 10 All-Stars with the given conditions -/
theorem all_stars_arrangement :
  arrange_all_stars 4 4 2 = 6912 := by
  sorry

end NUMINAMATH_CALUDE_all_stars_arrangement_l1632_163204


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1632_163218

theorem complex_equation_solution (z : ℂ) : 
  Complex.I * z = 1 - 2 * Complex.I → z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1632_163218


namespace NUMINAMATH_CALUDE_complex_addition_l1632_163240

theorem complex_addition (z₁ z₂ : ℂ) (h₁ : z₁ = 1 + I) (h₂ : z₂ = 2 + 3*I) :
  z₁ + z₂ = 3 + 4*I := by sorry

end NUMINAMATH_CALUDE_complex_addition_l1632_163240


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1632_163238

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1632_163238


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l1632_163216

/-- Represents different sampling methods -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a community with different income groups -/
structure Community where
  total_households : ℕ
  high_income : ℕ
  middle_income : ℕ
  low_income : ℕ
  sample_size : ℕ

/-- Represents a group of volleyball players -/
structure VolleyballTeam where
  total_players : ℕ
  players_to_select : ℕ

/-- Determines the most appropriate sampling method for a community survey -/
def best_community_sampling_method (c : Community) : SamplingMethod :=
  sorry

/-- Determines the most appropriate sampling method for a volleyball team survey -/
def best_volleyball_sampling_method (v : VolleyballTeam) : SamplingMethod :=
  sorry

/-- Theorem stating the most appropriate sampling methods for the given scenarios -/
theorem appropriate_sampling_methods 
  (community : Community)
  (volleyball_team : VolleyballTeam)
  (h_community : community = { 
    total_households := 400,
    high_income := 120,
    middle_income := 180,
    low_income := 100,
    sample_size := 100
  })
  (h_volleyball : volleyball_team = {
    total_players := 12,
    players_to_select := 3
  }) :
  best_community_sampling_method community = SamplingMethod.StratifiedSampling ∧
  best_volleyball_sampling_method volleyball_team = SamplingMethod.SimpleRandomSampling :=
sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l1632_163216


namespace NUMINAMATH_CALUDE_cost_system_correct_l1632_163249

/-- Represents the cost of seedlings in yuan -/
def CostSystem (x y : ℝ) : Prop :=
  (4 * x + 3 * y = 180) ∧ (x - y = 10)

/-- The cost system correctly represents the seedling pricing scenario -/
theorem cost_system_correct (x y : ℝ) :
  (4 * x + 3 * y = 180) →
  (y = x - 10) →
  CostSystem x y :=
by sorry

end NUMINAMATH_CALUDE_cost_system_correct_l1632_163249


namespace NUMINAMATH_CALUDE_volume_ratio_is_correct_l1632_163212

/-- The ratio of the volume of a cube with edge length 9 inches to the volume of a cube with edge length 2 feet -/
def volume_ratio : ℚ :=
  let inch_per_foot : ℚ := 12
  let edge1 : ℚ := 9  -- 9 inches
  let edge2 : ℚ := 2 * inch_per_foot  -- 2 feet in inches
  (edge1 / edge2) ^ 3

theorem volume_ratio_is_correct : volume_ratio = 27 / 512 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_is_correct_l1632_163212


namespace NUMINAMATH_CALUDE_inequality_proof_l1632_163219

theorem inequality_proof (x y z w : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0) 
  (h_sum_squares : x^2 + y^2 + z^2 + w^2 = 1) : 
  x^2 * y * z * w + x * y^2 * z * w + x * y * z^2 * w + x * y * z * w^2 ≤ 1/8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1632_163219


namespace NUMINAMATH_CALUDE_line_up_five_people_l1632_163296

theorem line_up_five_people (people : Finset Char) : 
  people.card = 5 → Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_line_up_five_people_l1632_163296


namespace NUMINAMATH_CALUDE_f_eight_equals_zero_l1632_163225

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_two_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

theorem f_eight_equals_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_sym : has_period_two_symmetry f) :
  f 8 = 0 :=
sorry

end NUMINAMATH_CALUDE_f_eight_equals_zero_l1632_163225


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_l1632_163262

theorem smallest_number_with_remainder (n : ℤ) : 
  (n % 5 = 2) ∧ 
  ((n + 1) % 5 = 2) ∧ 
  ((n + 2) % 5 = 2) ∧ 
  (n + (n + 1) + (n + 2) = 336) →
  n = 107 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_l1632_163262


namespace NUMINAMATH_CALUDE_log_property_l1632_163261

theorem log_property (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.log x) (h2 : f (a * b) = 1) :
  f (a ^ 2) + f (b ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_property_l1632_163261


namespace NUMINAMATH_CALUDE_second_box_capacity_l1632_163226

/-- Represents the dimensions and capacity of a box -/
structure Box where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℝ

/-- The first box with given dimensions and capacity -/
def box1 : Box :=
  { height := 4
  , width := 5
  , length := 10
  , capacity := 200 }

/-- The second box with dimensions relative to the first box -/
def box2 : Box :=
  { height := 3 * box1.height
  , width := 4 * box1.width
  , length := box1.length
  , capacity := 0 }  -- We'll prove this

/-- The volume of a box -/
def volume (b : Box) : ℝ := b.height * b.width * b.length

/-- The theorem to prove -/
theorem second_box_capacity : box2.capacity = 2400 := by
  sorry

end NUMINAMATH_CALUDE_second_box_capacity_l1632_163226


namespace NUMINAMATH_CALUDE_smallest_square_partition_l1632_163233

/-- Represents a square partition of a larger square -/
structure SquarePartition where
  side_length : ℕ
  partitions : List ℕ
  partition_count : partitions.length = 15
  all_integer : ∀ n ∈ partitions, n > 0
  sum_areas : (partitions.map (λ x => x * x)).sum = side_length * side_length
  unit_squares : (partitions.filter (λ x => x = 1)).length ≥ 12

/-- The smallest square that satisfies the partition conditions has side length 5 -/
theorem smallest_square_partition :
  ∀ sp : SquarePartition, sp.side_length ≥ 5 ∧
  ∃ sp' : SquarePartition, sp'.side_length = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_partition_l1632_163233


namespace NUMINAMATH_CALUDE_refrigerator_price_l1632_163248

/-- The price Ramesh paid for the refrigerator -/
def price_paid (labelled_price : ℝ) : ℝ :=
  0.80 * labelled_price + 125 + 250

/-- The theorem stating the price Ramesh paid for the refrigerator -/
theorem refrigerator_price :
  ∃ (labelled_price : ℝ),
    (1.10 * labelled_price = 21725) ∧
    (price_paid labelled_price = 16175) :=
by
  sorry

end NUMINAMATH_CALUDE_refrigerator_price_l1632_163248


namespace NUMINAMATH_CALUDE_x_range_l1632_163266

theorem x_range (x : Real) 
  (h1 : -Real.pi/2 ≤ x ∧ x ≤ 3*Real.pi/2) 
  (h2 : Real.sqrt (1 + Real.sin (2*x)) = Real.sin x + Real.cos x) : 
  -Real.pi/4 ≤ x ∧ x ≤ 3*Real.pi/4 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1632_163266


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1632_163299

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ ∃ r : ℝ, r > 0 ∧ ∀ k : ℕ, a (k + 1) = r * a k

-- Define the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  (a 1)^2 - 10*(a 1) + 16 = 0 →
  (a 19)^2 - 10*(a 19) + 16 = 0 →
  a 8 * a 10 * a 12 = 64 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1632_163299


namespace NUMINAMATH_CALUDE_calculation_proof_l1632_163290

theorem calculation_proof : (1 / 6 : ℚ) * (-6 : ℚ) / (-1/6 : ℚ) * (6 : ℚ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1632_163290


namespace NUMINAMATH_CALUDE_unshaded_area_equilateral_triangle_l1632_163223

/-- The area of the unshaded region inside an equilateral triangle, 
    whose side is the diameter of a semi-circle with radius 1. -/
theorem unshaded_area_equilateral_triangle (r : ℝ) : 
  r = 1 → 
  ∃ (A : ℝ), A = Real.sqrt 3 - π / 6 ∧ 
  A = (3 * Real.sqrt 3 / 4) * (2 * r)^2 - π * r^2 / 6 :=
by sorry

end NUMINAMATH_CALUDE_unshaded_area_equilateral_triangle_l1632_163223


namespace NUMINAMATH_CALUDE_p_squared_plus_26_composite_l1632_163279

theorem p_squared_plus_26_composite (p : Nat) (hp : Prime p) : 
  ∃ (a b : Nat), a > 1 ∧ b > 1 ∧ p^2 + 26 = a * b :=
sorry

end NUMINAMATH_CALUDE_p_squared_plus_26_composite_l1632_163279


namespace NUMINAMATH_CALUDE_gcd_bound_from_lcm_l1632_163231

theorem gcd_bound_from_lcm (a b : ℕ) : 
  (10^6 ≤ a ∧ a < 10^7) →
  (10^6 ≤ b ∧ b < 10^7) →
  (10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) →
  Nat.gcd a b < 1000 := by sorry

end NUMINAMATH_CALUDE_gcd_bound_from_lcm_l1632_163231


namespace NUMINAMATH_CALUDE_g_equals_inverse_at_three_point_five_l1632_163232

def g (x : ℝ) : ℝ := 3 * x - 7

theorem g_equals_inverse_at_three_point_five :
  g (3.5) = (Function.invFun g) (3.5) := by sorry

end NUMINAMATH_CALUDE_g_equals_inverse_at_three_point_five_l1632_163232


namespace NUMINAMATH_CALUDE_sum_of_two_and_four_l1632_163294

theorem sum_of_two_and_four : 2 + 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_and_four_l1632_163294


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_over_four_l1632_163235

theorem tan_alpha_plus_pi_over_four (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_over_four_l1632_163235


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_ten_l1632_163201

theorem last_digit_of_one_over_three_to_ten (n : ℕ) : 
  (1 : ℚ) / (3^10 : ℚ) * 10^n % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_ten_l1632_163201


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2023rd_term_l1632_163277

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_2023rd_term 
  (p q : ℝ) 
  (h1 : arithmetic_sequence p 6 2 = p + 6)
  (h2 : arithmetic_sequence p 6 3 = 4*p - q)
  (h3 : arithmetic_sequence p 6 4 = 4*p + q) :
  arithmetic_sequence p 6 2023 = 12137 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2023rd_term_l1632_163277


namespace NUMINAMATH_CALUDE_fraction_problem_l1632_163283

theorem fraction_problem (n : ℕ) : 
  (n : ℚ) / (4 * n - 4) = 1 / 2 → n = 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1632_163283


namespace NUMINAMATH_CALUDE_slower_train_speed_l1632_163206

/-- Proves that the speed of the slower train is 36 km/hr given the conditions of the problem --/
theorem slower_train_speed (faster_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) :
  faster_speed = 46 →
  passing_time = 36 →
  train_length = 50 →
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * passing_time * (1000 / 3600) = 2 * train_length :=
by
  sorry


end NUMINAMATH_CALUDE_slower_train_speed_l1632_163206


namespace NUMINAMATH_CALUDE_cylinder_cone_hemisphere_volume_l1632_163298

/-- Given a cylinder with volume 72π cm³, prove that the combined volume of a cone 
    with the same height as the cylinder and a hemisphere with the same radius 
    as the cylinder is equal to 72π cm³. -/
theorem cylinder_cone_hemisphere_volume 
  (r : ℝ) 
  (h : ℝ) 
  (cylinder_volume : π * r^2 * h = 72 * π) : 
  (1/3) * π * r^2 * h + (2/3) * π * r^3 = 72 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cone_hemisphere_volume_l1632_163298


namespace NUMINAMATH_CALUDE_jonathan_book_purchase_l1632_163293

-- Define the costs of the books and Jonathan's savings
def dictionary_cost : ℕ := 11
def dinosaur_book_cost : ℕ := 19
def cookbook_cost : ℕ := 7
def savings : ℕ := 8

-- Define the total cost of the books
def total_cost : ℕ := dictionary_cost + dinosaur_book_cost + cookbook_cost

-- Define the amount Jonathan needs
def amount_needed : ℕ := total_cost - savings

-- Theorem statement
theorem jonathan_book_purchase :
  amount_needed = 29 :=
by sorry

end NUMINAMATH_CALUDE_jonathan_book_purchase_l1632_163293


namespace NUMINAMATH_CALUDE_arctan_sum_theorem_l1632_163203

theorem arctan_sum_theorem (a b c : ℝ) 
  (h : Real.arctan a + Real.arctan b + Real.arctan c + π / 2 = 0) : 
  (a * b + b * c + c * a = 1) ∧ (a + b + c < a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_theorem_l1632_163203


namespace NUMINAMATH_CALUDE_number_count_l1632_163243

theorem number_count (avg_all : ℝ) (avg_pair1 : ℝ) (avg_pair2 : ℝ) (avg_pair3 : ℝ) 
  (h1 : avg_all = 4.60)
  (h2 : avg_pair1 = 3.4)
  (h3 : avg_pair2 = 3.8)
  (h4 : avg_pair3 = 6.6) :
  ∃ n : ℕ, n = 6 ∧ n * avg_all = 2 * (avg_pair1 + avg_pair2 + avg_pair3) := by
  sorry

end NUMINAMATH_CALUDE_number_count_l1632_163243


namespace NUMINAMATH_CALUDE_boy_initial_height_l1632_163220

/-- Represents the growth rates and heights of a tree and a boy -/
structure GrowthProblem where
  initialTreeHeight : ℝ
  finalTreeHeight : ℝ
  finalBoyHeight : ℝ
  treeGrowthRate : ℝ
  boyGrowthRate : ℝ

/-- Theorem stating the boy's initial height given the growth problem parameters -/
theorem boy_initial_height (p : GrowthProblem)
  (h1 : p.initialTreeHeight = 16)
  (h2 : p.finalTreeHeight = 40)
  (h3 : p.finalBoyHeight = 36)
  (h4 : p.treeGrowthRate = 2 * p.boyGrowthRate) :
  p.finalBoyHeight - (p.finalTreeHeight - p.initialTreeHeight) / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_boy_initial_height_l1632_163220


namespace NUMINAMATH_CALUDE_jerrys_books_count_l1632_163245

theorem jerrys_books_count :
  let initial_action_figures : ℕ := 3
  let added_action_figures : ℕ := 2
  let total_action_figures := initial_action_figures + added_action_figures
  let books_count := total_action_figures + 2
  books_count = 7 := by sorry

end NUMINAMATH_CALUDE_jerrys_books_count_l1632_163245


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1632_163291

/-- Given two lines in the 2D plane represented by their equations:
    ax + by + c = 0 and dx + ey + f = 0,
    this function returns true if the lines are perpendicular. -/
def are_perpendicular (a b c d e f : ℝ) : Prop :=
  a * d + b * e = 0

/-- Given a line ax + by + c = 0 and a point (x₀, y₀),
    this function returns true if the point lies on the line. -/
def point_on_line (a b c x₀ y₀ : ℝ) : Prop :=
  a * x₀ + b * y₀ + c = 0

theorem perpendicular_line_through_point :
  are_perpendicular 4 (-3) 2 3 4 1 ∧
  point_on_line 4 (-3) 2 1 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1632_163291


namespace NUMINAMATH_CALUDE_mock_exam_participants_l1632_163236

/-- The number of students who took a mock exam -/
def total_students : ℕ := 400

/-- The number of girls who took the exam -/
def num_girls : ℕ := 100

/-- The proportion of boys who cleared the cut off -/
def boys_cleared_ratio : ℚ := 3/5

/-- The proportion of girls who cleared the cut off -/
def girls_cleared_ratio : ℚ := 4/5

/-- The total proportion of students who qualified -/
def total_qualified_ratio : ℚ := 13/20

theorem mock_exam_participants :
  ∃ (num_boys : ℕ),
    (boys_cleared_ratio * num_boys + girls_cleared_ratio * num_girls : ℚ) = 
    total_qualified_ratio * (num_boys + num_girls) ∧
    total_students = num_boys + num_girls :=
by sorry

end NUMINAMATH_CALUDE_mock_exam_participants_l1632_163236


namespace NUMINAMATH_CALUDE_sequence_inequality_l1632_163228

theorem sequence_inequality (n : ℕ) (a : ℕ → ℚ) (h1 : a 0 = 1/2) 
  (h2 : ∀ k, k < n → a (k + 1) = a k + (1/n) * (a k)^2) :
  1 - 1/n < a n ∧ a n < 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1632_163228


namespace NUMINAMATH_CALUDE_replaced_crew_member_weight_l1632_163258

/-- Given a crew of 10 oarsmen, if replacing one member with a new member weighing 71 kg
    increases the average weight by 1.8 kg, then the replaced member weighed 53 kg. -/
theorem replaced_crew_member_weight
  (n : ℕ)
  (new_weight : ℝ)
  (avg_increase : ℝ)
  (h_crew_size : n = 10)
  (h_new_weight : new_weight = 71)
  (h_avg_increase : avg_increase = 1.8) :
  let old_total := n * (avg_increase + (new_weight - 53) / n)
  let new_total := n * (avg_increase + new_weight / n)
  new_total - old_total = n * avg_increase :=
by sorry

end NUMINAMATH_CALUDE_replaced_crew_member_weight_l1632_163258


namespace NUMINAMATH_CALUDE_no_real_roots_x_squared_plus_three_l1632_163256

theorem no_real_roots_x_squared_plus_three : 
  ∀ x : ℝ, x^2 + 3 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_x_squared_plus_three_l1632_163256


namespace NUMINAMATH_CALUDE_bales_in_barn_after_addition_l1632_163273

/-- The number of bales in the barn after addition -/
def bales_after_addition (initial_bales : ℕ) (added_bales : ℕ) : ℕ :=
  initial_bales + added_bales

/-- Theorem stating that the number of bales after Benny's addition is 82 -/
theorem bales_in_barn_after_addition :
  bales_after_addition 47 35 = 82 := by
  sorry

end NUMINAMATH_CALUDE_bales_in_barn_after_addition_l1632_163273


namespace NUMINAMATH_CALUDE_possible_no_snorers_in_sample_l1632_163221

-- Define the types for our problem
def Person : Type := Unit
def HasHeartDisease (p : Person) : Prop := sorry
def Snores (p : Person) : Prop := sorry

-- Define correlation and confidence
def Correlation (A B : Person → Prop) : Prop := sorry
def ConfidenceLevel : ℝ := sorry

-- State the theorem
theorem possible_no_snorers_in_sample 
  (corr : Correlation HasHeartDisease Snores)
  (conf : ConfidenceLevel > 0.99)
  : ∃ (sample : Finset Person), 
    (Finset.card sample = 100) ∧ 
    (∀ p ∈ sample, HasHeartDisease p) ∧
    (∀ p ∈ sample, ¬Snores p) :=
sorry

end NUMINAMATH_CALUDE_possible_no_snorers_in_sample_l1632_163221


namespace NUMINAMATH_CALUDE_g_composition_value_l1632_163264

def g (y : ℝ) : ℝ := y^3 - 3*y + 1

theorem g_composition_value : g (g (g (-1))) = 6803 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_value_l1632_163264


namespace NUMINAMATH_CALUDE_sqrt_8_times_sqrt_50_l1632_163267

theorem sqrt_8_times_sqrt_50 : Real.sqrt 8 * Real.sqrt 50 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_8_times_sqrt_50_l1632_163267


namespace NUMINAMATH_CALUDE_table_function_proof_l1632_163274

def f (x : ℝ) : ℝ := x^2 - x + 2

theorem table_function_proof :
  (f 2 = 3) ∧ (f 3 = 8) ∧ (f 4 = 15) ∧ (f 5 = 24) ∧ (f 6 = 35) := by
  sorry

end NUMINAMATH_CALUDE_table_function_proof_l1632_163274


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1632_163241

-- Define the inequality function
def f (x : ℝ) : ℝ := -x^2 - x + 2

-- Define the solution set
def solution_set : Set ℝ := {x | -2 < x ∧ x < 1}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x > 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1632_163241


namespace NUMINAMATH_CALUDE_factorization_2y_squared_minus_8_l1632_163297

theorem factorization_2y_squared_minus_8 (y : ℝ) : 2 * y^2 - 8 = 2 * (y + 2) * (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2y_squared_minus_8_l1632_163297


namespace NUMINAMATH_CALUDE_problem_proof_l1632_163271

theorem problem_proof : (-8: ℝ) ^ (1/3) + π^0 + Real.log 4 + Real.log 25 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l1632_163271


namespace NUMINAMATH_CALUDE_fir_tree_count_l1632_163253

/-- Represents the four children in the problem -/
inductive Child
| Anya
| Borya
| Vera
| Gena

/-- Represents the gender of a child -/
inductive Gender
| Boy
| Girl

/-- Returns the gender of a child -/
def childGender (c : Child) : Gender :=
  match c with
  | Child.Anya => Gender.Girl
  | Child.Borya => Gender.Boy
  | Child.Vera => Gender.Girl
  | Child.Gena => Gender.Boy

/-- Represents a statement made by a child -/
def Statement := ℕ → Prop

/-- Returns the statement made by each child -/
def childStatement (c : Child) : Statement :=
  match c with
  | Child.Anya => λ n => n = 15
  | Child.Borya => λ n => n % 11 = 0
  | Child.Vera => λ n => n < 25
  | Child.Gena => λ n => n % 22 = 0

theorem fir_tree_count :
  ∃ (n : ℕ) (truthTellers : Finset Child),
    n = 11 ∧
    truthTellers.card = 2 ∧
    (∃ (boy girl : Child), boy ∈ truthTellers ∧ girl ∈ truthTellers ∧
      childGender boy = Gender.Boy ∧ childGender girl = Gender.Girl) ∧
    (∀ c ∈ truthTellers, childStatement c n) ∧
    (∀ c ∉ truthTellers, ¬(childStatement c n)) :=
  sorry

end NUMINAMATH_CALUDE_fir_tree_count_l1632_163253


namespace NUMINAMATH_CALUDE_seminar_attendees_l1632_163278

theorem seminar_attendees (total : ℕ) (a : ℕ) (h1 : total = 185) (h2 : a = 30) : 
  total - (a + 2*a + (a + 10) + ((a + 10) - 5)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_seminar_attendees_l1632_163278


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1632_163229

/-- A geometric sequence with positive terms and a specific condition -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n) ∧
  (2 * a 1 + a 2 = a 3)

/-- The common ratio of the geometric sequence is 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (h : GeometricSequence a) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n, a (n + 1) = q * a n := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1632_163229


namespace NUMINAMATH_CALUDE_website_earnings_theorem_l1632_163210

/-- Calculates the earnings for a website over a week given the following conditions:
  - The website gets a fixed number of visitors per day for the first 6 days
  - On the 7th day, it gets twice as many visitors as the previous 6 days combined
  - There is a fixed earning per visit -/
def websiteEarnings (dailyVisitors : ℕ) (earningsPerVisit : ℚ) : ℚ :=
  let firstSixDaysVisits : ℕ := 6 * dailyVisitors
  let seventhDayVisits : ℕ := 2 * firstSixDaysVisits
  let totalVisits : ℕ := firstSixDaysVisits + seventhDayVisits
  (totalVisits : ℚ) * earningsPerVisit

/-- Theorem stating that under the given conditions, the website earnings for the week are $18 -/
theorem website_earnings_theorem :
  websiteEarnings 100 (1 / 100) = 18 := by
  sorry


end NUMINAMATH_CALUDE_website_earnings_theorem_l1632_163210


namespace NUMINAMATH_CALUDE_anne_twice_sister_height_l1632_163286

/-- Represents the heights of Anne, her sister, and Bella -/
structure Heights where
  anne : ℝ
  sister : ℝ
  bella : ℝ

/-- The conditions of the problem -/
def HeightConditions (h : Heights) : Prop :=
  ∃ (n : ℝ),
    h.anne = n * h.sister ∧
    h.bella = 3 * h.anne ∧
    h.anne = 80 ∧
    h.bella - h.sister = 200

/-- The theorem stating that under the given conditions, 
    Anne's height is twice her sister's height -/
theorem anne_twice_sister_height (h : Heights) 
  (hc : HeightConditions h) : h.anne = 2 * h.sister := by
  sorry

end NUMINAMATH_CALUDE_anne_twice_sister_height_l1632_163286


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_l1632_163211

/-- The number of elements in the nth row of Pascal's Triangle -/
def pascal_row_elements (n : ℕ) : ℕ := n + 1

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of elements in the first 30 rows of Pascal's Triangle -/
def total_elements : ℕ := sum_first_n 30

theorem pascal_triangle_30_rows :
  total_elements = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_l1632_163211


namespace NUMINAMATH_CALUDE_larger_number_problem_l1632_163285

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : max x y = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1632_163285


namespace NUMINAMATH_CALUDE_trig_power_sum_l1632_163269

theorem trig_power_sum (x : Real) 
  (h : Real.sin x ^ 10 + Real.cos x ^ 10 = 11/36) : 
  Real.sin x ^ 14 + Real.cos x ^ 14 = 41/216 := by
  sorry

end NUMINAMATH_CALUDE_trig_power_sum_l1632_163269


namespace NUMINAMATH_CALUDE_min_tablets_to_extract_l1632_163202

/-- Represents the number of tablets of each medicine type in the box -/
def tablets_per_type : ℕ := 10

/-- Represents the minimum number of tablets of each type we want to guarantee -/
def min_tablets_per_type : ℕ := 2

/-- Theorem: The minimum number of tablets to extract to guarantee at least two of each type -/
theorem min_tablets_to_extract :
  tablets_per_type + min_tablets_per_type = 12 := by sorry

end NUMINAMATH_CALUDE_min_tablets_to_extract_l1632_163202


namespace NUMINAMATH_CALUDE_angle_C_value_l1632_163259

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a = 5 ∧ 
  t.b + t.c = 2 * t.a ∧ 
  3 * Real.sin t.A = 5 * Real.sin t.B

-- Theorem statement
theorem angle_C_value (t : Triangle) (h : satisfiesConditions t) : t.C = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_value_l1632_163259


namespace NUMINAMATH_CALUDE_first_digit_of_y_in_base_9_l1632_163250

def base_3_num : List Nat := [1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1]

def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

def y : Nat := to_base_10 base_3_num 3

def first_digit_base_9 (n : Nat) : Nat :=
  if n = 0 then 0 else
  let log_9 := Nat.log n 9
  (n / (9 ^ log_9)) % 9

theorem first_digit_of_y_in_base_9 :
  first_digit_base_9 y = 4 := by sorry

end NUMINAMATH_CALUDE_first_digit_of_y_in_base_9_l1632_163250


namespace NUMINAMATH_CALUDE_cubic_function_value_l1632_163288

/-- Given a cubic function f(x) = ax³ + bx + 3 where f(-3) = 10, prove that f(3) = 27a + 3b + 3 -/
theorem cubic_function_value (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * x + 3)
  (h2 : f (-3) = 10) :
  f 3 = 27 * a + 3 * b + 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_value_l1632_163288


namespace NUMINAMATH_CALUDE_max_internet_days_l1632_163260

/-- Represents the tiered pricing structure for internet service -/
def daily_rate (day : ℕ) : ℚ :=
  if day ≤ 3 then 1/2
  else if day ≤ 7 then 7/10
  else 9/10

/-- Calculates the additional fee for every 5 days -/
def additional_fee (day : ℕ) : ℚ :=
  if day % 5 = 0 then 1 else 0

/-- Calculates the total cost for a given number of days -/
def total_cost (days : ℕ) : ℚ :=
  (Finset.range days).sum (λ d => daily_rate (d + 1) + additional_fee (d + 1))

/-- Theorem stating that 8 is the maximum number of days of internet connection -/
theorem max_internet_days : 
  ∀ n : ℕ, n ≤ 8 → total_cost n ≤ 7 ∧ 
  (n < 8 → total_cost (n + 1) ≤ 7) ∧
  (total_cost 9 > 7) :=
sorry

end NUMINAMATH_CALUDE_max_internet_days_l1632_163260


namespace NUMINAMATH_CALUDE_tan_fec_value_l1632_163222

/-- Square ABCD with inscribed isosceles triangle AEF -/
structure SquareWithTriangle where
  /-- Side length of the square -/
  a : ℝ
  /-- Point E on side BC -/
  e : ℝ × ℝ
  /-- Point F on side CD -/
  f : ℝ × ℝ
  /-- ABCD is a square -/
  square_abcd : e.1 ≤ a ∧ e.1 ≥ 0 ∧ f.2 ≤ a ∧ f.2 ≥ 0
  /-- E is on BC -/
  e_on_bc : e.2 = 0
  /-- F is on CD -/
  f_on_cd : f.1 = a
  /-- AEF is isosceles with AE = EF -/
  isosceles_aef : (0 - e.1)^2 + e.2^2 = (a - f.1)^2 + (f.2 - 0)^2
  /-- tan(∠AEF) = 2 -/
  tan_aef : (f.2 - 0) / (f.1 - e.1) = 2

/-- The tangent of angle FEC in the described configuration is 3 - √5 -/
theorem tan_fec_value (st : SquareWithTriangle) : 
  (st.a - st.e.1) / (st.f.2 - 0) = 3 - Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_tan_fec_value_l1632_163222


namespace NUMINAMATH_CALUDE_correct_ranking_count_l1632_163292

/-- Represents a team in the tournament -/
inductive Team : Type
| E : Team
| F : Team
| G : Team
| H : Team

/-- Represents the outcome of a match -/
inductive MatchOutcome : Type
| Win : Team → MatchOutcome
| Draw : MatchOutcome

/-- Represents the final ranking of teams -/
def Ranking := List Team

/-- The structure of the tournament -/
structure Tournament :=
  (saturdayMatch1 : MatchOutcome)
  (saturdayMatch2 : MatchOutcome)
  (sundayMatch1Winner : Team)
  (sundayMatch2Winner : Team)

/-- Function to calculate the number of possible rankings -/
def countPossibleRankings : ℕ :=
  -- Implementation details omitted
  256

/-- Theorem stating that the number of possible rankings is 256 -/
theorem correct_ranking_count :
  countPossibleRankings = 256 := by sorry


end NUMINAMATH_CALUDE_correct_ranking_count_l1632_163292


namespace NUMINAMATH_CALUDE_integer_square_four_l1632_163217

theorem integer_square_four (x : ℝ) (y : ℤ) 
  (eq1 : 4 * x + y = 34)
  (eq2 : 2 * x - y = 20) : 
  y = -2 ∧ y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_integer_square_four_l1632_163217


namespace NUMINAMATH_CALUDE_total_time_for_ten_pictures_l1632_163246

/-- The total time spent on drawing and coloring pictures -/
def total_time (num_pictures : ℕ) (draw_time : ℝ) (color_time_reduction : ℝ) : ℝ :=
  let color_time := draw_time * (1 - color_time_reduction)
  num_pictures * (draw_time + color_time)

/-- Theorem: The total time spent on 10 pictures is 34 hours -/
theorem total_time_for_ten_pictures :
  total_time 10 2 0.3 = 34 := by
  sorry

#eval total_time 10 2 0.3

end NUMINAMATH_CALUDE_total_time_for_ten_pictures_l1632_163246


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l1632_163257

theorem pasta_preference_ratio : 
  ∀ (total spaghetti manicotti : ℕ),
    total = 800 →
    spaghetti = 320 →
    manicotti = 160 →
    (spaghetti : ℚ) / (manicotti : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l1632_163257


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1632_163213

theorem expression_simplification_and_evaluation :
  let x : ℝ := 2
  let expr := (1 / (x - 3)) / (1 / (x^2 - 9)) - (x / (x + 1)) * ((x^2 + x) / x^2)
  expr = 4 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1632_163213


namespace NUMINAMATH_CALUDE_trig_sum_problem_l1632_163207

theorem trig_sum_problem (α : Real) 
  (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin α * Real.cos α = -1/2) :
  1 / (1 + Real.sin α) + 1 / (1 + Real.cos α) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_problem_l1632_163207


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l1632_163280

theorem complex_fraction_sum (a b : ℝ) : 
  (2 + 3 * Complex.I) / Complex.I = Complex.mk a b → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l1632_163280


namespace NUMINAMATH_CALUDE_article_price_proof_l1632_163254

/-- The normal price of an article before discounts -/
def normal_price : ℝ := 150

/-- The final price after discounts -/
def final_price : ℝ := 108

/-- The first discount rate -/
def discount1 : ℝ := 0.1

/-- The second discount rate -/
def discount2 : ℝ := 0.2

theorem article_price_proof :
  normal_price * (1 - discount1) * (1 - discount2) = final_price := by
  sorry

end NUMINAMATH_CALUDE_article_price_proof_l1632_163254


namespace NUMINAMATH_CALUDE_parking_space_area_l1632_163242

/-- A rectangular parking space with three painted sides -/
structure ParkingSpace where
  length : ℝ
  width : ℝ
  painted_sides_sum : ℝ
  unpainted_side : ℝ
  is_rectangular : length > 0 ∧ width > 0
  three_sides_painted : painted_sides_sum = 2 * width + length
  unpainted_is_length : unpainted_side = length

/-- The area of a parking space is equal to its length multiplied by its width -/
def area (p : ParkingSpace) : ℝ := p.length * p.width

/-- Theorem: If a rectangular parking space has an unpainted side of 9 feet
    and the sum of the painted sides is 37 feet, then its area is 126 square feet -/
theorem parking_space_area 
  (p : ParkingSpace) 
  (h1 : p.unpainted_side = 9) 
  (h2 : p.painted_sides_sum = 37) : 
  area p = 126 := by
  sorry

end NUMINAMATH_CALUDE_parking_space_area_l1632_163242


namespace NUMINAMATH_CALUDE_inequality_proof_l1632_163247

theorem inequality_proof (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 / (y - 1)) + (y^2 / (x - 1)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1632_163247


namespace NUMINAMATH_CALUDE_car_speed_l1632_163255

-- Define the problem parameters
def gallons_per_40_miles : ℝ := 1
def tank_capacity : ℝ := 12
def travel_time : ℝ := 5
def fuel_used_fraction : ℝ := 0.4166666666666667

-- Define the theorem
theorem car_speed (speed : ℝ) : 
  (gallons_per_40_miles * speed * travel_time / 40 = fuel_used_fraction * tank_capacity) →
  speed = 40 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_l1632_163255


namespace NUMINAMATH_CALUDE_arithmetic_operations_l1632_163227

theorem arithmetic_operations :
  ((-3) + 5 - (-3) = 5) ∧
  ((-1/3 - 3/4 + 5/6) * (-24) = 6) ∧
  (1 - 1/9 * (-1/2 - 2^2) = 3/2) ∧
  ((-1)^2023 * (18 - (-2) * 3) / (15 - 3^3) = 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l1632_163227


namespace NUMINAMATH_CALUDE_complement_of_M_l1632_163208

-- Define the set M
def M : Set ℝ := {x : ℝ | x * (x - 3) > 0}

-- State the theorem
theorem complement_of_M : 
  (Set.univ : Set ℝ) \ M = Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l1632_163208


namespace NUMINAMATH_CALUDE_white_stamp_price_is_20_cents_l1632_163287

/-- The price of a white stamp that satisfies the given conditions -/
def white_stamp_price : ℚ :=
  let red_stamps : ℕ := 30
  let white_stamps : ℕ := 80
  let red_stamp_price : ℚ := 1/2
  let sales_difference : ℚ := 1
  (sales_difference + red_stamps * red_stamp_price) / white_stamps

/-- Theorem stating that the white stamp price is 20 cents -/
theorem white_stamp_price_is_20_cents :
  white_stamp_price = 1/5 := by sorry

end NUMINAMATH_CALUDE_white_stamp_price_is_20_cents_l1632_163287


namespace NUMINAMATH_CALUDE_total_rainfall_2005_l1632_163282

def rainfall_2005 (initial_rainfall : ℝ) (yearly_increase : ℝ) : ℝ :=
  12 * (initial_rainfall + 2 * yearly_increase)

theorem total_rainfall_2005 (initial_rainfall yearly_increase : ℝ) 
  (h1 : initial_rainfall = 30)
  (h2 : yearly_increase = 3) :
  rainfall_2005 initial_rainfall yearly_increase = 432 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_2005_l1632_163282


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1632_163237

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_sets :
  is_pythagorean_triple 8 15 17 ∧
  is_pythagorean_triple 7 24 25 ∧
  is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 2 3 4 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l1632_163237


namespace NUMINAMATH_CALUDE_cubic_three_roots_m_range_l1632_163244

/-- The cubic polynomial f(x) = x³ - 6x² + 9x -/
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x

/-- Theorem stating the range of m for which x³ - 6x² + 9x + m = 0 has exactly three distinct real roots -/
theorem cubic_three_roots_m_range :
  ∀ m : ℝ, (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    f r₁ + m = 0 ∧ f r₂ + m = 0 ∧ f r₃ + m = 0) ↔ 
  -4 < m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_roots_m_range_l1632_163244


namespace NUMINAMATH_CALUDE_equation_solution_l1632_163200

theorem equation_solution : ∃! y : ℝ, (128 : ℝ) ^ (y + 1) / (8 : ℝ) ^ (y + 1) = (64 : ℝ) ^ (3 * y - 2) ∧ y = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1632_163200


namespace NUMINAMATH_CALUDE_ages_sum_l1632_163215

theorem ages_sum (j l : ℝ) : 
  j = l + 8 ∧ 
  j + 5 = 3 * (l - 6) → 
  j + l = 39 := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l1632_163215


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_degree_of_monomial_l1632_163268

-- Define the monomial
def monomial : ℚ × (ℕ × ℕ) := (-4/3, (2, 1))

-- Theorem for the coefficient
theorem coefficient_of_monomial :
  (monomial.fst : ℚ) = -4/3 := by sorry

-- Theorem for the degree
theorem degree_of_monomial :
  (monomial.snd.fst + monomial.snd.snd : ℕ) = 3 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_degree_of_monomial_l1632_163268


namespace NUMINAMATH_CALUDE_sine_in_triangle_l1632_163275

theorem sine_in_triangle (a b : ℝ) (A B : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : Real.sin A = 3/5) :
  Real.sin B = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sine_in_triangle_l1632_163275


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l1632_163276

theorem x_range_for_inequality (x : ℝ) :
  (∀ m ∈ Set.Icc (1/2 : ℝ) 3, x^2 + m*x + 4 > 2*m + 4*x) →
  x > 2 ∨ x < -1 :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l1632_163276


namespace NUMINAMATH_CALUDE_opposite_numbers_expression_l1632_163263

theorem opposite_numbers_expression (a b : ℝ) (h1 : a + b = 0) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  (a + b - 1) * (a / b + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_expression_l1632_163263


namespace NUMINAMATH_CALUDE_frog_climbs_out_l1632_163270

def well_depth : ℕ := 19
def day_climb : ℕ := 3
def night_slide : ℕ := 2

def days_to_climb (depth : ℕ) (day_climb : ℕ) (night_slide : ℕ) : ℕ :=
  (depth - day_climb) / (day_climb - night_slide) + 1

theorem frog_climbs_out : days_to_climb well_depth day_climb night_slide = 17 := by
  sorry

end NUMINAMATH_CALUDE_frog_climbs_out_l1632_163270


namespace NUMINAMATH_CALUDE_function_not_in_first_quadrant_l1632_163251

/-- The function f(x) = (1/2)^x + m does not pass through the first quadrant if and only if m ≤ -1 -/
theorem function_not_in_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, x > 0 → (1/2)^x + m ≤ 0) ↔ m ≤ -1 := by
sorry

end NUMINAMATH_CALUDE_function_not_in_first_quadrant_l1632_163251


namespace NUMINAMATH_CALUDE_catch_difference_l1632_163281

theorem catch_difference (joe_catches derek_catches tammy_catches : ℕ) : 
  joe_catches = 23 →
  derek_catches = 2 * joe_catches - 4 →
  tammy_catches = 30 →
  tammy_catches > derek_catches / 3 →
  tammy_catches - derek_catches / 3 = 16 := by
sorry

end NUMINAMATH_CALUDE_catch_difference_l1632_163281


namespace NUMINAMATH_CALUDE_function_properties_l1632_163214

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2*a - x) = f x

theorem function_properties (f : ℝ → ℝ) :
  (∀ x, f (x + 2) = f (x - 2)) →
  (∀ x, f (4 - x) = f x) →
  (is_periodic f 4 ∧ is_symmetric_about f 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1632_163214


namespace NUMINAMATH_CALUDE_benjamin_car_insurance_expenditure_l1632_163272

/-- The annual expenditure on car insurance, given the total expenditure over a decade -/
def annual_expenditure (total_expenditure : ℕ) (years : ℕ) : ℕ :=
  total_expenditure / years

/-- Theorem stating that the annual expenditure is 3000 dollars given the conditions -/
theorem benjamin_car_insurance_expenditure :
  annual_expenditure 30000 10 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_benjamin_car_insurance_expenditure_l1632_163272


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l1632_163205

theorem two_digit_number_difference (a b : ℕ) : 
  a ≥ 1 → a ≤ 9 → b ≤ 9 → (10 * a + b) - (10 * b + a) = 45 → a - b = 5 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l1632_163205


namespace NUMINAMATH_CALUDE_strawberry_yogurt_probability_l1632_163230

def prob_strawberry_yogurt (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem strawberry_yogurt_probability :
  let n₁ := 3
  let n₂ := 3
  let p₁ := (1 : ℚ) / 2
  let p₂ := (3 : ℚ) / 4
  let total_days := n₁ + n₂
  let success_days := 4
  (total_days.choose success_days : ℚ) *
    (prob_strawberry_yogurt n₁ 2 p₁ * prob_strawberry_yogurt n₂ 2 p₂ +
     prob_strawberry_yogurt n₁ 3 p₁ * prob_strawberry_yogurt n₂ 1 p₂) =
  1485 / 64 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_yogurt_probability_l1632_163230


namespace NUMINAMATH_CALUDE_dispatch_plans_count_l1632_163295

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of students --/
def total_students : ℕ := 6

/-- The number of students participating in the campaign --/
def participating_students : ℕ := 4

/-- The number of students participating on Sunday --/
def sunday_students : ℕ := 2

/-- The number of students participating on Friday --/
def friday_students : ℕ := 1

/-- The number of students participating on Saturday --/
def saturday_students : ℕ := 1

theorem dispatch_plans_count :
  (choose total_students sunday_students) *
  (choose (total_students - sunday_students) friday_students) *
  (choose (total_students - sunday_students - friday_students) saturday_students) = 180 := by
  sorry

end NUMINAMATH_CALUDE_dispatch_plans_count_l1632_163295


namespace NUMINAMATH_CALUDE_expand_expression_l1632_163284

theorem expand_expression (x y : ℝ) : (-x + 2*y) * (-x - 2*y) = x^2 - 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1632_163284


namespace NUMINAMATH_CALUDE_bug_path_tiles_l1632_163209

-- Define the garden dimensions
def width : ℕ := 12
def length : ℕ := 18

-- Define the function to calculate the number of tiles visited
def tilesVisited (w l : ℕ) : ℕ := w + l - Nat.gcd w l

-- Theorem statement
theorem bug_path_tiles :
  tilesVisited width length = 24 :=
sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l1632_163209


namespace NUMINAMATH_CALUDE_pokemon_cards_cost_l1632_163289

/-- The cost of a pack of Pokemon cards -/
def pokemon_cost (football_pack_cost baseball_deck_cost total_cost : ℚ) : ℚ :=
  total_cost - (2 * football_pack_cost + baseball_deck_cost)

/-- Theorem: The cost of the Pokemon cards is $4.01 -/
theorem pokemon_cards_cost : 
  pokemon_cost 2.73 8.95 18.42 = 4.01 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_cost_l1632_163289


namespace NUMINAMATH_CALUDE_overlapping_squares_areas_l1632_163234

/-- Represents the side lengths of three overlapping squares -/
structure SquareSides where
  largest : ℝ
  middle : ℝ
  smallest : ℝ

/-- Represents the areas of three overlapping squares -/
structure SquareAreas where
  largest : ℝ
  middle : ℝ
  smallest : ℝ

/-- Calculates the areas of three overlapping squares given their side lengths -/
def calculateAreas (sides : SquareSides) : SquareAreas :=
  { largest := sides.largest ^ 2,
    middle := sides.middle ^ 2,
    smallest := sides.smallest ^ 2 }

/-- Theorem stating the areas of three overlapping squares given specific conditions -/
theorem overlapping_squares_areas :
  ∀ (sides : SquareSides),
    sides.largest = sides.middle + 1 →
    sides.largest = sides.smallest + 2 →
    (sides.largest - 1) * (sides.middle - 1) = 100 →
    (sides.middle - 1) * (sides.smallest - 1) = 64 →
    calculateAreas sides = { largest := 361, middle := 324, smallest := 289 } := by
  sorry


end NUMINAMATH_CALUDE_overlapping_squares_areas_l1632_163234


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l1632_163239

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0) →
  (q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0) →
  (r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0) →
  (s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l1632_163239


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1632_163224

theorem inequality_solution_set (x : ℝ) :
  (|2*x - 3| < 1) ↔ (1 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1632_163224


namespace NUMINAMATH_CALUDE_final_game_score_l1632_163265

/-- Represents the points scored by each player in the basketball game -/
structure TeamPoints where
  bailey : ℕ
  michiko : ℕ
  akiko : ℕ
  chandra : ℕ

/-- Calculates the total points scored by the team -/
def total_points (t : TeamPoints) : ℕ :=
  t.bailey + t.michiko + t.akiko + t.chandra

/-- Theorem stating the total points scored by the team under given conditions -/
theorem final_game_score (t : TeamPoints) 
  (h1 : t.bailey = 14)
  (h2 : t.michiko = t.bailey / 2)
  (h3 : t.akiko = t.michiko + 4)
  (h4 : t.chandra = 2 * t.akiko) :
  total_points t = 54 := by
  sorry

end NUMINAMATH_CALUDE_final_game_score_l1632_163265
