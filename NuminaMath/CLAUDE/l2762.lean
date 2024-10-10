import Mathlib

namespace unique_solution_cubic_system_l2762_276216

theorem unique_solution_cubic_system :
  ∃! (x y z : ℝ), x^3 = 2*y - 1 ∧ y^3 = 2*z - 1 ∧ z^3 = 2*x - 1 :=
by
  -- The proof goes here
  sorry

end unique_solution_cubic_system_l2762_276216


namespace andrews_cheese_pops_l2762_276214

theorem andrews_cheese_pops (hotdogs chicken_nuggets total : ℕ) 
  (hotdogs_count : hotdogs = 30)
  (chicken_nuggets_count : chicken_nuggets = 40)
  (total_count : total = 90)
  (sum_equation : hotdogs + chicken_nuggets + (total - hotdogs - chicken_nuggets) = total) :
  total - hotdogs - chicken_nuggets = 20 := by
  sorry

end andrews_cheese_pops_l2762_276214


namespace eggs_per_group_l2762_276200

theorem eggs_per_group (total_eggs : ℕ) (num_groups : ℕ) (eggs_per_group : ℕ) : 
  total_eggs = 8 → num_groups = 4 → eggs_per_group = total_eggs / num_groups → eggs_per_group = 2 := by
  sorry

end eggs_per_group_l2762_276200


namespace sum_of_alan_and_bob_ages_l2762_276241

-- Define the set of possible ages
def Ages : Set ℕ := {3, 8, 12, 14}

-- Define the cousins' ages as natural numbers
variables (alan_age bob_age carl_age dan_age : ℕ)

-- Define the conditions
def conditions (alan_age bob_age carl_age dan_age : ℕ) : Prop :=
  alan_age ∈ Ages ∧ bob_age ∈ Ages ∧ carl_age ∈ Ages ∧ dan_age ∈ Ages ∧
  alan_age ≠ bob_age ∧ alan_age ≠ carl_age ∧ alan_age ≠ dan_age ∧
  bob_age ≠ carl_age ∧ bob_age ≠ dan_age ∧ carl_age ≠ dan_age ∧
  alan_age < carl_age ∧
  (alan_age + dan_age) % 5 = 0 ∧
  (carl_age + dan_age) % 5 = 0

-- Theorem statement
theorem sum_of_alan_and_bob_ages 
  (h : conditions alan_age bob_age carl_age dan_age) :
  alan_age + bob_age = 17 := by
  sorry

end sum_of_alan_and_bob_ages_l2762_276241


namespace cinnamon_swirls_distribution_l2762_276266

theorem cinnamon_swirls_distribution (total_pieces : ℕ) (num_people : ℕ) (pieces_per_person : ℕ) : 
  total_pieces = 12 → num_people = 3 → total_pieces = num_people * pieces_per_person → pieces_per_person = 4 := by
  sorry

end cinnamon_swirls_distribution_l2762_276266


namespace linear_function_properties_l2762_276297

-- Define the linear function
def f (x : ℝ) : ℝ := -2 * x + 1

-- Define the properties to be proven
theorem linear_function_properties :
  (∀ x y : ℝ, f x - f y = -2 * (x - y)) ∧  -- Slope is -2
  (f 0 = 1) ∧                              -- y-intercept is (0, 1)
  (∃ x y z : ℝ, f x > 0 ∧ x > 0 ∧          -- Passes through first quadrant
               f y < 0 ∧ y > 0 ∧           -- Passes through second quadrant
               f z < 0 ∧ z < 0) ∧          -- Passes through fourth quadrant
  (∀ x y : ℝ, x < y → f x > f y)           -- Slope is negative
  := by sorry

end linear_function_properties_l2762_276297


namespace third_number_proof_l2762_276233

theorem third_number_proof (a b c : ℕ) (h1 : a = 794) (h2 : b = 858) (h3 : c = 922) : 
  (∃ (k l m : ℕ), a = 64 * k + 22 ∧ b = 64 * l + 22 ∧ c = 64 * m + 22) ∧ 
  (∀ x : ℕ, b < x ∧ x < c → ¬(∃ n : ℕ, x = 64 * n + 22)) := by
  sorry

end third_number_proof_l2762_276233


namespace complex_ratio_theorem_l2762_276237

/-- Given complex numbers z₁, z₂, z₃ that satisfy certain conditions,
    prove that z₁z₂/z₃ = -5. -/
theorem complex_ratio_theorem (z₁ z₂ z₃ : ℂ)
  (h1 : Complex.abs z₁ = Complex.abs z₂)
  (h2 : Complex.abs z₁ = Real.sqrt 3 * Complex.abs z₃)
  (h3 : z₁ + z₃ = z₂) :
  z₁ * z₂ / z₃ = -5 := by
  sorry

end complex_ratio_theorem_l2762_276237


namespace sqrt_sum_equality_l2762_276268

theorem sqrt_sum_equality : 
  Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) + 1 = 8 * Real.sqrt 2 + 1 := by
  sorry

end sqrt_sum_equality_l2762_276268


namespace line_segment_param_sum_squares_l2762_276230

/-- Given a line segment from (-3, 9) to (4, 10) parameterized by x = at + b and y = ct + d,
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (-3, 9), prove that a^2 + b^2 + c^2 + d^2 = 140 -/
theorem line_segment_param_sum_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  b = -3 →
  d = 9 →
  a + b = 4 →
  c + d = 10 →
  a^2 + b^2 + c^2 + d^2 = 140 :=
by sorry

end line_segment_param_sum_squares_l2762_276230


namespace smallest_integer_for_inequality_l2762_276212

theorem smallest_integer_for_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x y z : ℝ, (a*x^2 + b*y^2 + c*z^2)^2 ≤ 3*(a^2*x^4 + b^2*y^4 + c^2*z^4)) ∧
  (∀ n : ℕ, n < 3 → ∃ x y z : ℝ, (a*x^2 + b*y^2 + c*z^2)^2 > n*(a^2*x^4 + b^2*y^4 + c^2*z^4)) :=
sorry

end smallest_integer_for_inequality_l2762_276212


namespace fraction_inequality_solution_l2762_276261

open Set

theorem fraction_inequality_solution (x : ℝ) :
  (x - 5) / ((x - 3)^2) < 0 ↔ x ∈ Iio 3 ∪ Ioo 3 5 :=
by sorry

end fraction_inequality_solution_l2762_276261


namespace quadratic_root_zero_l2762_276280

/-- Given a quadratic equation (k-1)x^2 + 6x + k^2 - k = 0 with a root of 0, prove that k = 0 -/
theorem quadratic_root_zero (k : ℝ) : 
  (∃ x, (k - 1) * x^2 + 6 * x + k^2 - k = 0) ∧ 
  ((k - 1) * 0^2 + 6 * 0 + k^2 - k = 0) → 
  k = 0 :=
sorry

end quadratic_root_zero_l2762_276280


namespace sqrt_three_cubed_l2762_276206

theorem sqrt_three_cubed : Real.sqrt 3 ^ 3 = 3 * Real.sqrt 3 := by
  sorry

end sqrt_three_cubed_l2762_276206


namespace ratio_of_divisor_sums_l2762_276235

def M : ℕ := 24 * 36 * 49 * 125

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 62 := by sorry

end ratio_of_divisor_sums_l2762_276235


namespace range_of_p_l2762_276284

def h (x : ℝ) : ℝ := 4 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_of_p :
  ∀ y ∈ Set.range p, -1 ≤ y ∧ y ≤ 1023 ∧
  ∀ z, -1 ≤ z ∧ z ≤ 1023 → ∃ x, -1 ≤ x ∧ x ≤ 3 ∧ p x = z :=
sorry

end range_of_p_l2762_276284


namespace line_slope_product_l2762_276209

/-- Given two lines L₁ and L₂ with equations y = mx and y = nx respectively,
    where L₁ makes twice as large of an angle with the horizontal as L₂,
    L₁ has 3 times the slope of L₂, and L₁ is not horizontal,
    then mn = 1. -/
theorem line_slope_product (m n : ℝ) (hm : m ≠ 0) :
  (∃ θ : ℝ, m = Real.tan (2 * θ) ∧ n = Real.tan θ) →
  m = 3 * n →
  m * n = 1 := by
  sorry

end line_slope_product_l2762_276209


namespace prob_same_color_proof_l2762_276262

def total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2

def prob_same_color : ℚ := 1 / 3

theorem prob_same_color_proof :
  let prob_red := red_balls / total_balls * (red_balls - 1) / (total_balls - 1)
  let prob_white := white_balls / total_balls * (white_balls - 1) / (total_balls - 1)
  prob_red + prob_white = prob_same_color :=
sorry

end prob_same_color_proof_l2762_276262


namespace polygon_internal_external_angles_equal_l2762_276240

theorem polygon_internal_external_angles_equal (n : ℕ) : 
  (n : ℝ) ≥ 3 → ((n - 2) * 180 = 360) → n = 4 := by sorry

end polygon_internal_external_angles_equal_l2762_276240


namespace train_speed_problem_l2762_276298

/-- Proves that given the conditions of the train problem, the average speed of Train B is 43 miles per hour. -/
theorem train_speed_problem (initial_gap : ℝ) (train_a_speed : ℝ) (overtake_time : ℝ) (final_gap : ℝ) :
  initial_gap = 13 →
  train_a_speed = 37 →
  overtake_time = 5 →
  final_gap = 17 →
  (initial_gap + train_a_speed * overtake_time + final_gap) / overtake_time = 43 :=
by sorry

end train_speed_problem_l2762_276298


namespace floor_sqrt_eight_count_l2762_276226

theorem floor_sqrt_eight_count :
  (Finset.filter (fun x : ℕ => ⌊Real.sqrt x⌋ = 8) (Finset.range 81)).card = 17 :=
sorry

end floor_sqrt_eight_count_l2762_276226


namespace infinitely_many_square_sum_averages_l2762_276265

theorem infinitely_many_square_sum_averages :
  ∀ k : ℕ, ∃ n > k, ∃ m : ℕ, ((n + 1) * (2 * n + 1)) / 6 = m^2 := by
  sorry

end infinitely_many_square_sum_averages_l2762_276265


namespace solution_system_equations_l2762_276276

theorem solution_system_equations (A : ℤ) (hA : A ≠ 0) :
  ∀ x y z : ℤ,
    x + y^2 + z^3 = A ∧
    (1 : ℚ) / x + (1 : ℚ) / y^2 + (1 : ℚ) / z^3 = (1 : ℚ) / A ∧
    x * y^2 * z^3 = A^2 →
    ∃ k : ℤ, A = -k^12 ∧
      ((x = -k^12 ∧ (y = k^3 ∨ y = -k^3) ∧ z = -k^2) ∨
       (x = -k^3 ∧ (y = k^3 ∨ y = -k^3) ∧ z = -k^4)) :=
by sorry

end solution_system_equations_l2762_276276


namespace smallest_value_z_plus_i_l2762_276252

theorem smallest_value_z_plus_i (z : ℂ) (h : Complex.abs (z^2 + 4) = Complex.abs (z * (z + 2*I))) :
  ∃ (min_val : ℝ), min_val = 1 ∧ ∀ (w : ℂ), Complex.abs (w^2 + 4) = Complex.abs (w * (w + 2*I)) →
    Complex.abs (w + I) ≥ min_val :=
by sorry

end smallest_value_z_plus_i_l2762_276252


namespace labourer_income_is_78_l2762_276275

/-- Represents the financial situation of a labourer over a 10-month period. -/
structure LabourerFinances where
  monthly_income : ℝ
  initial_debt : ℝ
  first_period_months : ℕ := 6
  second_period_months : ℕ := 4
  first_period_monthly_expense : ℝ := 85
  second_period_monthly_expense : ℝ := 60
  final_savings : ℝ := 30

/-- The labourer's financial situation satisfies the given conditions. -/
def satisfies_conditions (f : LabourerFinances) : Prop :=
  f.first_period_months * f.monthly_income - f.initial_debt = 
    f.first_period_months * f.first_period_monthly_expense ∧
  f.second_period_months * f.monthly_income = 
    f.second_period_months * f.second_period_monthly_expense + f.initial_debt + f.final_savings

/-- The labourer's monthly income is 78 given the conditions. -/
theorem labourer_income_is_78 (f : LabourerFinances) 
  (h : satisfies_conditions f) : f.monthly_income = 78 := by
  sorry

end labourer_income_is_78_l2762_276275


namespace zoo_animals_l2762_276288

theorem zoo_animals (b r m : ℕ) : 
  b + r + m = 300 →
  2 * b + 3 * r + 4 * m = 798 →
  r = 102 :=
by sorry

end zoo_animals_l2762_276288


namespace twelve_chairs_adjacent_subsets_l2762_276277

/-- The number of subsets containing at least three adjacent chairs 
    when n chairs are arranged in a circle. -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs arranged in a circle, 
    the number of subsets containing at least three adjacent chairs is 2040. -/
theorem twelve_chairs_adjacent_subsets : 
  subsets_with_adjacent_chairs 12 = 2040 := by sorry

end twelve_chairs_adjacent_subsets_l2762_276277


namespace triangle_side_length_l2762_276291

theorem triangle_side_length (a b c : ℝ) (A B C : Real) :
  b = 2 * Real.sqrt 3 →
  a = 2 →
  B = π / 3 →  -- 60° in radians
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos B →
  c = 4 := by
  sorry

end triangle_side_length_l2762_276291


namespace max_min_difference_c_l2762_276213

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 6) 
  (sum_sq_eq : a^2 + b^2 + c^2 = 18) : 
  (∃ (a₁ b₁ : ℝ), a₁ + b₁ + 6 = 6 ∧ a₁^2 + b₁^2 + 6^2 = 18) ∧ 
  (∃ (a₂ b₂ : ℝ), a₂ + b₂ + (-2) = 6 ∧ a₂^2 + b₂^2 + (-2)^2 = 18) ∧
  (∀ (a₃ b₃ c₃ : ℝ), a₃ + b₃ + c₃ = 6 → a₃^2 + b₃^2 + c₃^2 = 18 → c₃ ≤ 6 ∧ c₃ ≥ -2) ∧
  (6 - (-2) = 8) := by
  sorry

end max_min_difference_c_l2762_276213


namespace correct_sums_l2762_276257

theorem correct_sums (total : ℕ) (wrong_ratio : ℕ) (correct : ℕ) : 
  total = 36 → 
  wrong_ratio = 2 → 
  total = correct + wrong_ratio * correct → 
  correct = 12 := by sorry

end correct_sums_l2762_276257


namespace daily_servings_sold_l2762_276287

theorem daily_servings_sold (cost profit_A profit_B revenue total_profit : ℚ)
  (h1 : cost = 14)
  (h2 : profit_A = 20)
  (h3 : profit_B = 18)
  (h4 : revenue = 1120)
  (h5 : total_profit = 280) :
  ∃ (x y : ℚ), x + y = 60 ∧ 
    profit_A * x + profit_B * y = revenue ∧
    (profit_A - cost) * x + (profit_B - cost) * y = total_profit :=
by sorry

end daily_servings_sold_l2762_276287


namespace vehicle_inspection_is_systematic_l2762_276296

/-- Represents a vehicle's license plate -/
structure LicensePlate where
  number : Nat

/-- Represents a sampling method -/
inductive SamplingMethod
  | Systematic
  | Other

/-- The criterion for selecting a vehicle based on its license plate -/
def selectionCriterion (plate : LicensePlate) : Bool :=
  plate.number % 10 = 5

/-- The sampling method used in the vehicle inspection process -/
def vehicleInspectionSampling : SamplingMethod :=
  SamplingMethod.Systematic

/-- Theorem stating that the vehicle inspection sampling method is systematic sampling -/
theorem vehicle_inspection_is_systematic :
  vehicleInspectionSampling = SamplingMethod.Systematic :=
sorry

end vehicle_inspection_is_systematic_l2762_276296


namespace solve_age_problem_l2762_276219

def age_problem (rona_age : ℕ) : Prop :=
  let rachel_age := 2 * rona_age
  let collete_age := rona_age / 2
  let tommy_age := collete_age + rona_age
  rachel_age + rona_age + collete_age + tommy_age = 40

theorem solve_age_problem : age_problem 8 := by
  sorry

end solve_age_problem_l2762_276219


namespace proposition_q_false_l2762_276283

open Real

theorem proposition_q_false (p q : Prop) 
  (hp : ¬ (∃ x : ℝ, (1/10)^(x-3) ≤ cos 2))
  (hpq : ¬((¬p) ∧ q)) : ¬q := by
  sorry

end proposition_q_false_l2762_276283


namespace onion_rings_cost_l2762_276282

/-- Proves that the cost of onion rings is $2 given the costs of other items and payment details --/
theorem onion_rings_cost (hamburger_cost smoothie_cost total_paid change : ℕ) :
  hamburger_cost = 4 →
  smoothie_cost = 3 →
  total_paid = 20 →
  change = 11 →
  total_paid - change - hamburger_cost - smoothie_cost = 2 := by
  sorry

end onion_rings_cost_l2762_276282


namespace pyramid_angles_theorem_l2762_276255

/-- Represents the angles formed by the lateral faces of a pyramid with its square base -/
structure PyramidAngles where
  α : Real
  β : Real
  γ : Real
  δ : Real

/-- Theorem: Given a pyramid with a square base, if the angles formed by the lateral faces 
    with the base are in the ratio 1:2:4:2, then these angles are π/6, π/3, 2π/3, and π/3. -/
theorem pyramid_angles_theorem (angles : PyramidAngles) : 
  (angles.α : Real) / (angles.β : Real) = 1 / 2 ∧
  (angles.α : Real) / (angles.γ : Real) = 1 / 4 ∧
  (angles.α : Real) / (angles.δ : Real) = 1 / 2 ∧
  angles.α + angles.β + angles.γ + angles.δ = 2 * Real.pi →
  angles.α = Real.pi / 6 ∧
  angles.β = Real.pi / 3 ∧
  angles.γ = 2 * Real.pi / 3 ∧
  angles.δ = Real.pi / 3 :=
by sorry

end pyramid_angles_theorem_l2762_276255


namespace tic_tac_toe_losses_l2762_276205

theorem tic_tac_toe_losses (total_games wins draws : ℕ) (h1 : total_games = 14) (h2 : wins = 2) (h3 : draws = 10) :
  total_games = wins + (total_games - wins - draws) + draws :=
by sorry

#check tic_tac_toe_losses

end tic_tac_toe_losses_l2762_276205


namespace inequality_solution_set_l2762_276253

theorem inequality_solution_set :
  ∀ x : ℝ, (x / 4 - 1 ≤ 3 + x ∧ 3 + x < 1 - 3 * (2 + x)) ↔ x ∈ Set.Icc (-16/3) (-2) :=
by sorry

end inequality_solution_set_l2762_276253


namespace greatest_root_of_g_l2762_276231

def g (x : ℝ) : ℝ := 10 * x^4 - 16 * x^2 + 6

theorem greatest_root_of_g :
  ∃ (r : ℝ), g r = 0 ∧ r = 1 ∧ ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end greatest_root_of_g_l2762_276231


namespace hyperbola_equation_l2762_276271

/-- Definition of a hyperbola with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  focus : ℝ × ℝ
  intersection_line : ℝ → ℝ
  midpoint_x : ℝ

/-- Theorem stating the equation of the hyperbola with given properties -/
theorem hyperbola_equation (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_focus : h.focus = (Real.sqrt 7, 0))
  (h_line : h.intersection_line = fun x ↦ x - 1)
  (h_midpoint : h.midpoint_x = -2/3) :
  ∃ (x y : ℝ), x^2/2 - y^2/5 = 1 :=
sorry

end hyperbola_equation_l2762_276271


namespace park_area_is_1500000_l2762_276293

/-- Represents the scale of the map in miles per inch -/
def scale : ℝ := 250

/-- Represents the length of the park on the map in inches -/
def map_length : ℝ := 6

/-- Represents the width of the park on the map in inches -/
def map_width : ℝ := 4

/-- Calculates the actual area of the park in square miles -/
def park_area : ℝ := (map_length * scale) * (map_width * scale)

/-- Theorem stating that the actual area of the park is 1500000 square miles -/
theorem park_area_is_1500000 : park_area = 1500000 := by
  sorry

end park_area_is_1500000_l2762_276293


namespace decimal_sum_to_fraction_l2762_276254

theorem decimal_sum_to_fraction : 
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 733 / 12500 := by
  sorry

end decimal_sum_to_fraction_l2762_276254


namespace function_inequality_l2762_276260

open Set

-- Define the interval [a, b]
variable (a b : ℝ) (hab : a < b)

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State that f and g are differentiable on [a, b]
variable (hf : DifferentiableOn ℝ f (Icc a b))
variable (hg : DifferentiableOn ℝ g (Icc a b))

-- State that f'(x) < g'(x) for all x in [a, b]
variable (h_deriv : ∀ x ∈ Icc a b, deriv f x < deriv g x)

-- State the theorem
theorem function_inequality (x : ℝ) (hx : a < x ∧ x < b) :
  f x + g a < g x + f a :=
sorry

end function_inequality_l2762_276260


namespace base_3_to_decimal_l2762_276239

/-- Converts a list of digits in base k to its decimal representation -/
def to_decimal (digits : List Nat) (k : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * k^i) 0

/-- The base-3 representation of a number -/
def base_3_number : List Nat := [1, 0, 2]

/-- Theorem stating that the base-3 number (102)₃ is equal to 11 in decimal -/
theorem base_3_to_decimal :
  to_decimal base_3_number 3 = 11 := by sorry

end base_3_to_decimal_l2762_276239


namespace new_plan_cost_l2762_276270

def old_plan_cost : ℝ := 150
def increase_percentage : ℝ := 0.3

theorem new_plan_cost : 
  old_plan_cost * (1 + increase_percentage) = 195 := by sorry

end new_plan_cost_l2762_276270


namespace range_of_a_l2762_276294

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end range_of_a_l2762_276294


namespace rook_placement_count_l2762_276229

def chessboard_size : ℕ := 8
def num_rooks : ℕ := 6

theorem rook_placement_count :
  (Nat.choose chessboard_size num_rooks) * (Nat.factorial chessboard_size / Nat.factorial (chessboard_size - num_rooks)) = 564480 :=
by sorry

end rook_placement_count_l2762_276229


namespace estimate_pi_l2762_276201

theorem estimate_pi (n : ℕ) (m : ℕ) (h1 : n = 200) (h2 : m = 56) :
  let estimate := 4 * (m / n + 1 / 2)
  estimate = 78 / 25 := by
  sorry

end estimate_pi_l2762_276201


namespace dihedral_angle_line_relationship_l2762_276245

/-- A dihedral angle with edge l and planes α and β -/
structure DihedralAngle where
  l : Line
  α : Plane
  β : Plane

/-- A right dihedral angle -/
def is_right_dihedral (d : DihedralAngle) : Prop := sorry

/-- A line a in plane α -/
def line_in_plane_α (d : DihedralAngle) (a : Line) : Prop := sorry

/-- A line b in plane β -/
def line_in_plane_β (d : DihedralAngle) (b : Line) : Prop := sorry

/-- Line not perpendicular to edge l -/
def not_perp_to_edge (d : DihedralAngle) (m : Line) : Prop := sorry

/-- Two lines are parallel -/
def are_parallel (m n : Line) : Prop := sorry

/-- Two lines are perpendicular -/
def are_perpendicular (m n : Line) : Prop := sorry

theorem dihedral_angle_line_relationship (d : DihedralAngle) (a b : Line) 
  (h_right : is_right_dihedral d)
  (h_a_in_α : line_in_plane_α d a)
  (h_b_in_β : line_in_plane_β d b)
  (h_a_not_perp : not_perp_to_edge d a)
  (h_b_not_perp : not_perp_to_edge d b) :
  (∃ (a' b' : Line), line_in_plane_α d a' ∧ line_in_plane_β d b' ∧ 
    not_perp_to_edge d a' ∧ not_perp_to_edge d b' ∧ are_parallel a' b') ∧ 
  (∀ (a' b' : Line), line_in_plane_α d a' → line_in_plane_β d b' → 
    not_perp_to_edge d a' → not_perp_to_edge d b' → ¬ are_perpendicular a' b') :=
sorry

end dihedral_angle_line_relationship_l2762_276245


namespace complex_fraction_simplification_l2762_276274

theorem complex_fraction_simplification :
  (Complex.I + 1) / (1 - Complex.I) = Complex.I :=
by sorry

end complex_fraction_simplification_l2762_276274


namespace limit_exists_and_equals_20_21_l2762_276227

/-- Sum of exponents of 71 and 97 in the prime factorization of n -/
def s (n : ℕ+) : ℕ :=
  sorry

/-- Function f(n) = (-1)^(s(n)) -/
def f (n : ℕ+) : ℤ :=
  (-1) ^ (s n)

/-- Sum of f(x) from x = 1 to n -/
def S (n : ℕ+) : ℤ :=
  (Finset.range n).sum (fun x => f ⟨x + 1, Nat.succ_pos x⟩)

/-- The main theorem: limit of S(n)/n exists and equals 20/21 -/
theorem limit_exists_and_equals_20_21 :
    ∃ (L : ℚ), L = 20 / 21 ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N,
      |((S n : ℚ) / n) - L| < ε :=
  sorry

end limit_exists_and_equals_20_21_l2762_276227


namespace f_surjective_and_unique_l2762_276238

def f (x y : ℕ) : ℕ := (x + y - 1) * (x + y - 2) / 2 + y

theorem f_surjective_and_unique :
  ∀ n : ℕ, ∃! (x y : ℕ), f x y = n :=
by sorry

end f_surjective_and_unique_l2762_276238


namespace equation_solution_l2762_276204

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, (3 * x₁^2 - 6 * x₁ = -1) ∧ 
              (3 * x₂^2 - 6 * x₂ = -1) ∧ 
              (x₁ = 1 + Real.sqrt 6 / 3) ∧ 
              (x₂ = 1 - Real.sqrt 6 / 3) :=
by sorry

end equation_solution_l2762_276204


namespace one_diagonal_polygon_has_four_edges_edges_equal_vertices_one_diagonal_polygon_four_edges_l2762_276273

/-- A polygon is a shape with straight sides and angles. -/
structure Polygon where
  vertices : ℕ
  vertices_positive : vertices > 0

/-- A diagonal in a polygon is a line segment that connects two non-adjacent vertices. -/
def diagonals_from_vertex (p : Polygon) : ℕ := p.vertices - 3

/-- A polygon where only one diagonal can be drawn from a single vertex has 4 edges. -/
theorem one_diagonal_polygon_has_four_edges (p : Polygon) 
  (h : diagonals_from_vertex p = 1) : p.vertices = 4 := by
  sorry

/-- The number of edges in a polygon is equal to its number of vertices. -/
theorem edges_equal_vertices (p : Polygon) : 
  (number_of_edges : ℕ) → number_of_edges = p.vertices := by
  sorry

/-- A polygon where only one diagonal can be drawn from a single vertex has 4 edges. -/
theorem one_diagonal_polygon_four_edges (p : Polygon) 
  (h : diagonals_from_vertex p = 1) : (number_of_edges : ℕ) → number_of_edges = 4 := by
  sorry

end one_diagonal_polygon_has_four_edges_edges_equal_vertices_one_diagonal_polygon_four_edges_l2762_276273


namespace fifteenSidedFigureArea_l2762_276243

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon defined by a list of vertices -/
def Polygon := List Point

/-- The 15-sided figure described in the problem -/
def fifteenSidedFigure : Polygon := [
  ⟨1, 2⟩, ⟨2, 2⟩, ⟨2, 3⟩, ⟨3, 4⟩, ⟨4, 4⟩, ⟨5, 5⟩, ⟨6, 5⟩, ⟨7, 4⟩,
  ⟨6, 3⟩, ⟨6, 2⟩, ⟨5, 1⟩, ⟨4, 1⟩, ⟨3, 1⟩, ⟨2, 1⟩, ⟨1, 2⟩
]

/-- Calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℝ :=
  sorry

/-- Theorem stating that the area of the 15-sided figure is 15 cm² -/
theorem fifteenSidedFigureArea :
  calculateArea fifteenSidedFigure = 15 := by sorry

end fifteenSidedFigureArea_l2762_276243


namespace max_k_inequality_l2762_276248

theorem max_k_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (k : ℝ), k > 0 ∧ 
  (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    Real.sqrt (x^2 + k*y^2) + Real.sqrt (y^2 + k*x^2) ≥ x + y + (k-1) * Real.sqrt (x*y)) ∧
  (∀ (k' : ℝ), k' > k → 
    ∃ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
      Real.sqrt (x^2 + k'*y^2) + Real.sqrt (y^2 + k'*x^2) < x + y + (k'-1) * Real.sqrt (x*y)) ∧
  k = 3 :=
sorry

end max_k_inequality_l2762_276248


namespace instantaneous_velocity_at_3_l2762_276202

-- Define the motion equation
def s (t : ℝ) : ℝ := t^3 + t^2 - 1

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 3 * t^2 + 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_3 : v 3 = 33 := by
  sorry

end instantaneous_velocity_at_3_l2762_276202


namespace bushes_needed_for_perfume_l2762_276223

/-- The number of rose petals needed to make an ounce of perfume -/
def petals_per_ounce : ℕ := 320

/-- The number of petals produced by each rose -/
def petals_per_rose : ℕ := 8

/-- The number of roses per bush -/
def roses_per_bush : ℕ := 12

/-- The number of bottles of perfume to be made -/
def num_bottles : ℕ := 20

/-- The number of ounces in each bottle of perfume -/
def ounces_per_bottle : ℕ := 12

/-- The theorem stating the number of bushes needed to make the required perfume -/
theorem bushes_needed_for_perfume : 
  (petals_per_ounce * num_bottles * ounces_per_bottle) / (petals_per_rose * roses_per_bush) = 800 := by
  sorry

end bushes_needed_for_perfume_l2762_276223


namespace perfect_square_problem_l2762_276225

theorem perfect_square_problem (n : ℕ+) :
  ∃ k : ℕ, (n : ℤ)^2 + 19*(n : ℤ) + 48 = k^2 → n = 33 := by
  sorry

end perfect_square_problem_l2762_276225


namespace two_numbers_sum_l2762_276220

theorem two_numbers_sum (x y : ℝ) 
  (sum_eq : x + y = 5)
  (diff_eq : x - y = 10)
  (square_diff_eq : x^2 - y^2 = 50) : 
  x + y = 5 := by
sorry

end two_numbers_sum_l2762_276220


namespace inequality_solution_set_l2762_276258

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (a + 1)*x + a > 0}
  if a < 1 then S = {x : ℝ | x < a ∨ x > 1}
  else if a = 1 then S = {x : ℝ | x ≠ 1}
  else S = {x : ℝ | x < 1 ∨ x > a} := by
sorry

end inequality_solution_set_l2762_276258


namespace count_bases_with_final_digit_one_l2762_276215

/-- The number of bases between 2 and 12 (inclusive) where 625 in base 10 has a final digit of 1 -/
def count_bases : ℕ := 7

/-- The set of bases between 2 and 12 (inclusive) where 625 in base 10 has a final digit of 1 -/
def valid_bases : Finset ℕ := {2, 3, 4, 6, 8, 9, 12}

theorem count_bases_with_final_digit_one :
  (Finset.range 11).filter (fun b => 625 % (b + 2) = 1) = valid_bases ∧
  valid_bases.card = count_bases :=
sorry

end count_bases_with_final_digit_one_l2762_276215


namespace line_angle_problem_l2762_276259

theorem line_angle_problem (a : ℝ) : 
  let line1 := {(x, y) : ℝ × ℝ | a * x - y + 3 = 0}
  let line2 := {(x, y) : ℝ × ℝ | x - 2 * y + 4 = 0}
  let angle := Real.arccos (Real.sqrt 5 / 5)
  (∃ (θ : ℝ), θ = angle ∧ 
    θ = Real.arccos ((1 + a * (1/2)) / Real.sqrt ((1 + a^2) * (1 + (1/2)^2))))
  → a = -3/4 := by
  sorry

end line_angle_problem_l2762_276259


namespace moving_point_on_line_segment_l2762_276289

/-- Two fixed points in a plane -/
structure FixedPoints where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  distance : dist F₁ F₂ = 16

/-- A moving point M satisfying the condition |MF₁| + |MF₂| = 16 -/
def MovingPoint (fp : FixedPoints) (M : ℝ × ℝ) : Prop :=
  dist M fp.F₁ + dist M fp.F₂ = 16

/-- The theorem stating that any moving point M lies on the line segment F₁F₂ -/
theorem moving_point_on_line_segment (fp : FixedPoints) (M : ℝ × ℝ) 
    (h : MovingPoint fp M) : 
    ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • fp.F₁ + t • fp.F₂ :=
  sorry

end moving_point_on_line_segment_l2762_276289


namespace solve_equation_l2762_276224

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.03) : x = 0.3 := by
  sorry

end solve_equation_l2762_276224


namespace noah_holidays_l2762_276295

/-- The number of holidays Noah takes per month -/
def holidays_per_month : ℕ := 3

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total number of holidays Noah takes in a year -/
def total_holidays : ℕ := holidays_per_month * months_in_year

theorem noah_holidays : total_holidays = 36 := by
  sorry

end noah_holidays_l2762_276295


namespace x_value_l2762_276218

theorem x_value (y : ℝ) (h1 : 2 * x - y = 14) (h2 : y = 2) : x = 8 := by
  sorry

end x_value_l2762_276218


namespace largest_decimal_number_l2762_276208

theorem largest_decimal_number : 
  let a := 0.989
  let b := 0.9098
  let c := 0.9899
  let d := 0.9009
  let e := 0.9809
  c > a ∧ c > b ∧ c > d ∧ c > e := by sorry

end largest_decimal_number_l2762_276208


namespace quadrilateral_cosine_sum_l2762_276249

theorem quadrilateral_cosine_sum (α β γ δ : Real) :
  (α + β + γ + δ = 2 * Real.pi) →
  (Real.cos α + Real.cos β + Real.cos γ + Real.cos δ = 0) →
  (α + β = Real.pi) ∨ (α + γ = Real.pi) ∨ (α + δ = Real.pi) :=
by sorry

end quadrilateral_cosine_sum_l2762_276249


namespace y_sixth_power_root_l2762_276285

theorem y_sixth_power_root (y : ℝ) (hy : y > 0) (h : Real.sin (Real.arctan y) = y^3) :
  ∃ (z : ℝ), z > 0 ∧ z^3 + z^2 - 1 = 0 ∧ y^6 = z := by
  sorry

end y_sixth_power_root_l2762_276285


namespace night_day_crew_ratio_l2762_276247

theorem night_day_crew_ratio (D N : ℝ) (h1 : D > 0) (h2 : N > 0) : 
  (D / (D + 3/4 * N) = 0.64) → (N / D = 3/4) := by
  sorry

end night_day_crew_ratio_l2762_276247


namespace rational_includes_positive_and_negative_l2762_276232

-- Define rational numbers
def RationalNumber : Type := ℚ

-- Define positive and negative rational numbers
def PositiveRational (q : ℚ) : Prop := q > 0
def NegativeRational (q : ℚ) : Prop := q < 0

-- State the theorem
theorem rational_includes_positive_and_negative :
  (∃ q : ℚ, PositiveRational q) ∧ (∃ q : ℚ, NegativeRational q) :=
sorry

end rational_includes_positive_and_negative_l2762_276232


namespace game_result_l2762_276210

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 3 = 0 then 6
  else if n % 2 = 0 then 2
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2]
def betty_rolls : List ℕ := [6, 3, 3, 2]

def total_points (rolls : List ℕ) : ℕ :=
  (rolls.map f).sum

theorem game_result : 
  (total_points allie_rolls) * (total_points betty_rolls) = 32 := by
  sorry

end game_result_l2762_276210


namespace smallest_fraction_proof_l2762_276279

def is_natural_number (q : ℚ) : Prop := ∃ (n : ℕ), q = n

theorem smallest_fraction_proof (f : ℚ) : 
  (f ≥ 42/5) →
  (is_natural_number (f / (21/25))) →
  (is_natural_number (f / (14/15))) →
  (∀ g : ℚ, g < f → ¬(is_natural_number (g / (21/25)) ∧ is_natural_number (g / (14/15)))) →
  f = 42/5 :=
by sorry

end smallest_fraction_proof_l2762_276279


namespace circle_radius_is_three_l2762_276299

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y - 7 = 0

-- State the theorem
theorem circle_radius_is_three :
  ∃ (h k r : ℝ), r = 3 ∧ ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end circle_radius_is_three_l2762_276299


namespace cookie_difference_l2762_276267

theorem cookie_difference (paul_cookies : ℕ) (total_cookies : ℕ) (paula_cookies : ℕ) : 
  paul_cookies = 45 → 
  total_cookies = 87 → 
  paula_cookies < paul_cookies →
  paul_cookies + paula_cookies = total_cookies →
  paul_cookies - paula_cookies = 3 := by
sorry

end cookie_difference_l2762_276267


namespace parabola_focus_directrix_distance_l2762_276207

/-- Given a parabola y = 2x^2, the distance from its focus to its directrix is 1/2 -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y = 2 * x^2 →
  ∃ (focus_x focus_y directrix_y : ℝ),
    (focus_x = 0 ∧ focus_y = 1/4) ∧
    directrix_y = -1/4 ∧
    focus_y - directrix_y = 1/2 :=
by sorry

end parabola_focus_directrix_distance_l2762_276207


namespace runner_problem_l2762_276272

/-- Proves that for a 40-mile run where the speed is halved halfway through,
    and the second half takes 12 hours longer than the first half,
    the time to complete the second half is 24 hours. -/
theorem runner_problem (v : ℝ) (h1 : v > 0) : 
  (40 / v = 20 / v + 12) → (40 / (v / 2) = 24) :=
by sorry

end runner_problem_l2762_276272


namespace final_K_value_l2762_276290

/-- Represents the state of the program at each iteration -/
structure ProgramState :=
  (S : ℕ)
  (K : ℕ)

/-- Defines a single iteration of the loop -/
def iterate (state : ProgramState) : ProgramState :=
  { S := state.S^2 + 1,
    K := state.K + 1 }

/-- Defines the condition for continuing the loop -/
def loopCondition (state : ProgramState) : Prop :=
  state.S < 100

/-- Theorem: The final value of K is 4 -/
theorem final_K_value :
  ∃ (n : ℕ), ∃ (finalState : ProgramState),
    (finalState.K = 4) ∧
    (¬loopCondition finalState) ∧
    (finalState = (iterate^[n] ⟨1, 1⟩)) :=
  sorry

end final_K_value_l2762_276290


namespace arithmetic_sequence_common_difference_l2762_276221

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) -- Sequence of integers indexed by natural numbers
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- Arithmetic sequence condition
  (h_a1 : a 1 = -1) -- First term condition
  (h_a4 : a 4 = 8) -- Fourth term condition
  : ∃ d : ℤ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d := by
  sorry

end arithmetic_sequence_common_difference_l2762_276221


namespace intersection_equality_implies_a_range_l2762_276251

def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem intersection_equality_implies_a_range (a : ℝ) :
  A ∩ B a = B a → a = 1 ∨ a ≤ -1 := by sorry

end intersection_equality_implies_a_range_l2762_276251


namespace radical_simplification_l2762_276246

theorem radical_simplification (y : ℝ) (h : y > 0) :
  Real.sqrt (50 * y) * Real.sqrt (5 * y) * Real.sqrt (45 * y) = 15 * y * Real.sqrt (10 * y) :=
by sorry

end radical_simplification_l2762_276246


namespace tangent_range_l2762_276234

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The equation of the tangent line passing through (a, f(a)) and (2, t) --/
def tangent_equation (a t : ℝ) : Prop :=
  t - (f a) = (f' a) * (2 - a)

/-- The condition for three distinct tangent lines --/
def three_tangents (t : ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    tangent_equation a t ∧ tangent_equation b t ∧ tangent_equation c t

/-- Theorem: If a point (2, t) can be used to draw three tangent lines to y = f(x),
    then t is in the open interval (-6, 2) --/
theorem tangent_range :
  ∀ t : ℝ, three_tangents t → -6 < t ∧ t < 2 := by sorry

end tangent_range_l2762_276234


namespace contrapositive_equivalence_l2762_276256

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 < x ∧ x < 1) ↔ ((x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) :=
by sorry

end contrapositive_equivalence_l2762_276256


namespace geometric_sequence_sum_l2762_276236

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 18 = -6) →
  (a 2 * a 18 = 4) →
  a 4 * a 16 + a 10 = 6 :=
by
  sorry

end geometric_sequence_sum_l2762_276236


namespace tuesday_temperature_l2762_276203

theorem tuesday_temperature
  (temp_tues wed thur fri : ℝ)
  (avg_tues_wed_thur : (temp_tues + wed + thur) / 3 = 52)
  (avg_wed_thur_fri : (wed + thur + fri) / 3 = 54)
  (fri_temp : fri = 53) :
  temp_tues = 47 := by
sorry

end tuesday_temperature_l2762_276203


namespace lcm_gcd_equation_solutions_l2762_276264

def solution_pairs : List (Nat × Nat) := [(3, 6), (4, 6), (4, 4), (6, 4), (6, 3)]

theorem lcm_gcd_equation_solutions :
  ∀ a b : Nat,
    a > 0 ∧ b > 0 →
    (Nat.lcm a b + Nat.gcd a b + a + b = a * b) ↔ (a, b) ∈ solution_pairs := by
  sorry

end lcm_gcd_equation_solutions_l2762_276264


namespace min_sales_to_break_even_l2762_276211

-- Define the given conditions
def current_salary : ℕ := 90000
def new_base_salary : ℕ := 45000
def commission_rate : ℚ := 15 / 100
def sale_value : ℕ := 1500

-- Define the function to calculate the total earnings in the new job
def new_job_earnings (num_sales : ℕ) : ℚ :=
  new_base_salary + (num_sales * sale_value * commission_rate)

-- Theorem statement
theorem min_sales_to_break_even :
  ∃ n : ℕ, (∀ m : ℕ, m < n → new_job_earnings m < current_salary) ∧
           new_job_earnings n ≥ current_salary ∧
           n = 200 := by
  sorry


end min_sales_to_break_even_l2762_276211


namespace relay_for_life_total_miles_l2762_276269

/-- Calculates the total miles walked in a relay event -/
def total_miles_walked (john_speed bob_speed alice_speed : ℝ) 
                       (john_time bob_time alice_time : ℝ) : ℝ :=
  john_speed * john_time + alice_speed * alice_time + bob_speed * bob_time

/-- The combined total miles walked by John, Alice, and Bob during the Relay for Life event -/
theorem relay_for_life_total_miles : 
  total_miles_walked 3.5 4 2.8 4 6 8 = 62.8 := by
  sorry

end relay_for_life_total_miles_l2762_276269


namespace cookie_making_time_l2762_276278

/-- Proves that the time to make dough and cool cookies is equal to the total time minus the sum of baking time and icing hardening times. -/
theorem cookie_making_time (total_time baking_time white_icing_time chocolate_icing_time : ℕ)
  (h1 : total_time = 120)
  (h2 : baking_time = 15)
  (h3 : white_icing_time = 30)
  (h4 : chocolate_icing_time = 30) :
  total_time - (baking_time + white_icing_time + chocolate_icing_time) = 45 :=
by sorry

end cookie_making_time_l2762_276278


namespace tuesday_income_l2762_276222

/-- Calculates Lauren's income from her social media channel --/
def laurens_income (commercial_rate : ℚ) (subscription_rate : ℚ) (commercials_viewed : ℕ) (new_subscribers : ℕ) : ℚ :=
  commercial_rate * commercials_viewed + subscription_rate * new_subscribers

/-- Proves that Lauren's income on Tuesday is $77.00 --/
theorem tuesday_income : 
  laurens_income (1/2) 1 100 27 = 77 := by
  sorry

end tuesday_income_l2762_276222


namespace trapezoid_solutions_l2762_276263

def is_trapezoid_solution (b₁ b₂ : ℕ) : Prop :=
  b₁ % 8 = 0 ∧ b₂ % 8 = 0 ∧ (b₁ + b₂) * 50 / 2 = 1400 ∧ b₁ > 0 ∧ b₂ > 0

theorem trapezoid_solutions :
  ∃! (solutions : List (ℕ × ℕ)), solutions.length = 3 ∧
    ∀ (b₁ b₂ : ℕ), (b₁, b₂) ∈ solutions ↔ is_trapezoid_solution b₁ b₂ :=
sorry

end trapezoid_solutions_l2762_276263


namespace hyperbola_focus_k_value_l2762_276292

/-- Theorem: For a hyperbola with equation 8kx^2 - ky^2 = 8 and one focus at (0, -3), the value of k is -1. -/
theorem hyperbola_focus_k_value (k : ℝ) : 
  (∀ x y : ℝ, 8 * k * x^2 - k * y^2 = 8) → -- hyperbola equation
  (∃ x : ℝ, (x, -3) ∈ {(x, y) | x^2 / (8 / k) + y^2 / (8 / k + 1) = 1}) → -- focus at (0, -3)
  k = -1 :=
by sorry

end hyperbola_focus_k_value_l2762_276292


namespace arithmetic_calculations_l2762_276244

theorem arithmetic_calculations :
  ((1 : ℝ) - 12 + (-6) - (-28) = 10) ∧
  ((2 : ℝ) - 3^2 + (7/8 - 1) * (-2)^2 = -9.5) := by
  sorry

end arithmetic_calculations_l2762_276244


namespace smallest_n_for_given_mean_l2762_276281

theorem smallest_n_for_given_mean : ∃ (n : ℕ) (m : ℕ),
  n > 0 ∧
  m ∈ Finset.range (n + 1) ∧
  (Finset.sum (Finset.range (n + 1) \ {m}) id) / ((n : ℚ) - 1) = 439 / 13 ∧
  ∀ (k : ℕ) (j : ℕ), k > 0 ∧ k < n →
    j ∈ Finset.range (k + 1) →
    (Finset.sum (Finset.range (k + 1) \ {j}) id) / ((k : ℚ) - 1) ≠ 439 / 13 ∧
  n = 68 ∧
  m = 45 :=
sorry

end smallest_n_for_given_mean_l2762_276281


namespace initial_children_on_bus_prove_initial_children_on_bus_l2762_276217

theorem initial_children_on_bus : ℕ → Prop :=
  fun initial_children =>
    ∀ (added_children total_children : ℕ),
      added_children = 7 →
      total_children = 25 →
      initial_children + added_children = total_children →
      initial_children = 18

-- Proof
theorem prove_initial_children_on_bus :
  ∃ (initial_children : ℕ), initial_children_on_bus initial_children :=
by
  sorry

end initial_children_on_bus_prove_initial_children_on_bus_l2762_276217


namespace board_length_l2762_276250

-- Define the lengths of the two pieces
def shorter_piece : ℝ := 2
def longer_piece : ℝ := 2 * shorter_piece

-- Define the total length of the board
def total_length : ℝ := shorter_piece + longer_piece

-- Theorem to prove
theorem board_length : total_length = 6 := by
  sorry

end board_length_l2762_276250


namespace meeting_time_prove_meeting_time_l2762_276242

/-- The time it takes for Petya and Vasya to meet under the given conditions -/
theorem meeting_time : ℝ → ℝ → ℝ → Prop :=
  fun (x : ℝ) (v_g : ℝ) (t : ℝ) =>
    x > 0 ∧ v_g > 0 ∧  -- Positive distance and speed
    x = 3 * v_g ∧  -- Petya reaches the bridge in 1 hour
    t = 1 + (2 * x - 2 * v_g) / (2 * v_g) ∧  -- Total time calculation
    t = 2  -- The meeting time is 2 hours

/-- Proof of the meeting time theorem -/
theorem prove_meeting_time : ∃ (x v_g : ℝ), meeting_time x v_g 2 := by
  sorry


end meeting_time_prove_meeting_time_l2762_276242


namespace twentieth_fisherman_catch_l2762_276228

theorem twentieth_fisherman_catch (total_fishermen : Nat) (total_fish : Nat) 
  (each_fish : Nat) (n : Nat) (h1 : total_fishermen = 20) 
  (h2 : total_fish = 10000) (h3 : each_fish = 400) (h4 : n = 19) : 
  total_fish - n * each_fish = 2400 := by
  sorry

#check twentieth_fisherman_catch

end twentieth_fisherman_catch_l2762_276228


namespace number_of_divisors_of_60_l2762_276286

theorem number_of_divisors_of_60 : Nat.card {d : Nat | d > 0 ∧ 60 % d = 0} = 12 := by
  sorry

end number_of_divisors_of_60_l2762_276286
