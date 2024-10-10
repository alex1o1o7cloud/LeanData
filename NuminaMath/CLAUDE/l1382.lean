import Mathlib

namespace product_sum_of_three_numbers_l1382_138213

theorem product_sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 252) 
  (h2 : a + b + c = 22) : 
  a*b + b*c + c*a = 116 := by
sorry

end product_sum_of_three_numbers_l1382_138213


namespace max_distance_for_A_l1382_138263

/-- Represents a member of the expedition team -/
structure Member where
  name : String
  supplies : Nat

/-- Represents the expedition team -/
structure Team where
  members : List Member
  daily_distance : Nat

/-- Calculates the maximum distance a member can travel -/
def max_distance (team : Team) : Nat :=
  sorry

/-- Main theorem: The maximum distance A can travel is 900 kilometers -/
theorem max_distance_for_A (team : Team) :
  team.members.length = 3 ∧
  team.members.all (λ m => m.supplies = 36) ∧
  team.daily_distance = 30 →
  max_distance team = 900 :=
sorry

end max_distance_for_A_l1382_138263


namespace stream_speed_l1382_138200

/-- Given a boat traveling a round trip with known parameters, prove the speed of the stream -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) : 
  boat_speed = 16 → 
  distance = 7560 → 
  total_time = 960 → 
  ∃ (stream_speed : ℝ), 
    stream_speed = 2 ∧ 
    distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = total_time :=
by sorry

end stream_speed_l1382_138200


namespace abs_neg_2022_l1382_138223

theorem abs_neg_2022 : |(-2022 : ℤ)| = 2022 := by
  sorry

end abs_neg_2022_l1382_138223


namespace shaded_square_covering_all_columns_l1382_138252

def shaded_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => shaded_sequence n + (2 * (n + 1) - 1)

def column_position (n : ℕ) : ℕ :=
  (shaded_sequence n - 1) % 10 + 1

def all_columns_covered (n : ℕ) : Prop :=
  ∀ k : ℕ, k ∈ Finset.range 10 → ∃ i : ℕ, i ≤ n ∧ column_position i = k + 1

theorem shaded_square_covering_all_columns :
  all_columns_covered 20 ∧ ∀ m : ℕ, m < 20 → ¬ all_columns_covered m :=
sorry

end shaded_square_covering_all_columns_l1382_138252


namespace sine_matrix_determinant_zero_l1382_138273

theorem sine_matrix_determinant_zero :
  let A : Matrix (Fin 3) (Fin 3) ℝ := λ i j =>
    match i, j with
    | 0, 0 => Real.sin 3
    | 0, 1 => Real.sin 4
    | 0, 2 => Real.sin 5
    | 1, 0 => Real.sin 6
    | 1, 1 => Real.sin 7
    | 1, 2 => Real.sin 8
    | 2, 0 => Real.sin 9
    | 2, 1 => Real.sin 10
    | 2, 2 => Real.sin 11
  Matrix.det A = 0 := by
  sorry

-- Sine angle addition formula
axiom sine_angle_addition (x y : ℝ) :
  Real.sin (x + y) = Real.sin x * Real.cos y + Real.cos x * Real.sin y

end sine_matrix_determinant_zero_l1382_138273


namespace problem_solution_l1382_138236

theorem problem_solution (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12) 
  (h2 : a^2 + b^2 + c^2 = a*b + b*c + c*a) : 
  a + b^2 + c^3 = 14 := by
  sorry

end problem_solution_l1382_138236


namespace power_of_two_in_factorial_eight_l1382_138202

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem power_of_two_in_factorial_eight :
  ∀ i k m p : ℕ,
  i > 0 → k > 0 → m > 0 → p > 0 →
  factorial 8 = 2^i * 3^k * 5^m * 7^p →
  i + k + m + p = 11 →
  i = 7 :=
by
  sorry

end power_of_two_in_factorial_eight_l1382_138202


namespace fair_expenses_correct_l1382_138238

/-- Calculates the total amount spent at a fair given the following conditions:
  - Entrance fee for persons under 18: $5
  - Entrance fee for persons 18 and older: 20% more than $5
  - Cost per ride: $0.50
  - One adult (Joe) and two children (6-year-old twin brothers)
  - Each person took 3 rides
-/
def fairExpenses (childEntranceFee adultEntranceFeeIncrease ridePrice : ℚ) 
                 (numChildren numAdults numRidesPerPerson : ℕ) : ℚ :=
  let childrenEntranceFees := childEntranceFee * numChildren
  let adultEntranceFee := childEntranceFee * (1 + adultEntranceFeeIncrease)
  let adultEntranceFees := adultEntranceFee * numAdults
  let totalEntranceFees := childrenEntranceFees + adultEntranceFees
  let totalRideCost := ridePrice * numRidesPerPerson * (numChildren + numAdults)
  totalEntranceFees + totalRideCost

/-- Theorem stating that the total amount spent at the fair under the given conditions is $20.50 -/
theorem fair_expenses_correct : 
  fairExpenses 5 0.2 0.5 2 1 3 = 41/2 := by
  sorry

end fair_expenses_correct_l1382_138238


namespace product_equals_one_l1382_138257

theorem product_equals_one (x₁ x₂ x₃ : ℝ) 
  (h_nonneg₁ : x₁ ≥ 0) (h_nonneg₂ : x₂ ≥ 0) (h_nonneg₃ : x₃ ≥ 0)
  (h_sum : x₁ + x₂ + x₃ = 1) :
  (x₁ + 3*x₂ + 5*x₃) * (x₁ + x₂/3 + x₃/5) = 1 := by
  sorry

end product_equals_one_l1382_138257


namespace max_x_for_perfect_square_l1382_138291

theorem max_x_for_perfect_square : 
  ∀ x : ℕ, x > 1972 → ¬(∃ y : ℕ, 4^27 + 4^1000 + 4^x = y^2) ∧ 
  ∃ y : ℕ, 4^27 + 4^1000 + 4^1972 = y^2 :=
by sorry

end max_x_for_perfect_square_l1382_138291


namespace two_alarms_parallel_reliability_l1382_138210

/-- The reliability of a single alarm -/
def single_alarm_reliability : ℝ := 0.90

/-- The reliability of two independent alarms connected in parallel -/
def parallel_reliability (p : ℝ) : ℝ := 1 - (1 - p) * (1 - p)

theorem two_alarms_parallel_reliability :
  parallel_reliability single_alarm_reliability = 0.99 := by
  sorry

end two_alarms_parallel_reliability_l1382_138210


namespace four_sharp_40_l1382_138277

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem four_sharp_40 : sharp (sharp (sharp (sharp 40))) = 9.536 := by
  sorry

end four_sharp_40_l1382_138277


namespace contrapositive_equivalence_l1382_138207

theorem contrapositive_equivalence (a b : ℝ) :
  (ab = 0 → a = 0 ∨ b = 0) ↔ (a ≠ 0 ∧ b ≠ 0 → ab ≠ 0) :=
by sorry

end contrapositive_equivalence_l1382_138207


namespace complex_magnitude_problem_l1382_138229

theorem complex_magnitude_problem (z : ℂ) (h : z * (2 - 4*I) = 1 + 3*I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_magnitude_problem_l1382_138229


namespace correct_sample_size_l1382_138275

-- Define the population size
def population_size : ℕ := 5000

-- Define the number of sampled students
def sampled_students : ℕ := 450

-- Define what sample size means in this context
def sample_size (n : ℕ) : Prop := n = sampled_students

-- Theorem stating that the sample size is 450
theorem correct_sample_size : sample_size 450 := by sorry

end correct_sample_size_l1382_138275


namespace anthony_pencils_l1382_138295

theorem anthony_pencils (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 245 → received = 758 → total = initial + received → total = 1003 := by
  sorry

end anthony_pencils_l1382_138295


namespace hearty_beads_count_l1382_138206

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := 3

/-- The number of packages of red beads Hearty bought -/
def red_packages : ℕ := 5

/-- The number of beads in each red package -/
def beads_per_red_package : ℕ := 40

/-- The number of beads in each blue package is twice the number in each red package -/
def beads_per_blue_package : ℕ := 2 * beads_per_red_package

/-- The total number of beads Hearty has -/
def total_beads : ℕ := blue_packages * beads_per_blue_package + red_packages * beads_per_red_package

theorem hearty_beads_count : total_beads = 440 := by
  sorry

end hearty_beads_count_l1382_138206


namespace second_team_cups_l1382_138269

def total_required : ℕ := 280
def first_team : ℕ := 90
def third_team : ℕ := 70

theorem second_team_cups : total_required - first_team - third_team = 120 := by
  sorry

end second_team_cups_l1382_138269


namespace unattainable_value_l1382_138204

theorem unattainable_value (x : ℝ) (y : ℝ) (h : x ≠ -4/3) :
  y = (2 - x) / (3 * x + 4) → y ≠ -1/3 :=
by sorry

end unattainable_value_l1382_138204


namespace inequality_proof_l1382_138259

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 2) * (y^2 + 2) * (z^2 + 2) ≥ 9 * (x*y + y*z + z*x) := by
  sorry

end inequality_proof_l1382_138259


namespace roots_of_quadratic_equation_l1382_138250

theorem roots_of_quadratic_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x
  (f 0 = 0) ∧ (f 3 = 0) ∧ (∀ x : ℝ, f x = 0 → x = 0 ∨ x = 3) := by
  sorry

end roots_of_quadratic_equation_l1382_138250


namespace quadratic_equation_transformation_l1382_138290

theorem quadratic_equation_transformation :
  ∀ x : ℝ, (2 * x^2 = -3 * x + 1) ↔ (2 * x^2 + 3 * x - 1 = 0) :=
by sorry

end quadratic_equation_transformation_l1382_138290


namespace other_divisor_proof_l1382_138268

theorem other_divisor_proof (x : ℕ) (h : x > 0) : 
  (261 % 37 = 2 ∧ 261 % x = 2) → x = 7 := by
  sorry

end other_divisor_proof_l1382_138268


namespace passes_through_point1_passes_through_point2_unique_line_l1382_138216

/-- A line passing through two points (-1, 0) and (0, 2) -/
def line (x y : ℝ) : Prop := y = 2 * x + 2

/-- The line passes through the point (-1, 0) -/
theorem passes_through_point1 : line (-1) 0 := by sorry

/-- The line passes through the point (0, 2) -/
theorem passes_through_point2 : line 0 2 := by sorry

/-- The equation y = 2x + 2 represents the unique line passing through (-1, 0) and (0, 2) -/
theorem unique_line : ∀ (x y : ℝ), (y = 2 * x + 2) ↔ line x y := by sorry

end passes_through_point1_passes_through_point2_unique_line_l1382_138216


namespace square_fence_perimeter_is_77_and_third_l1382_138293

/-- The outer perimeter of a square fence with given specifications -/
def squareFencePerimeter (totalPosts : ℕ) (postWidth : ℚ) (gapWidth : ℕ) : ℚ :=
  let postsPerSide : ℕ := totalPosts / 4 + 1
  let gapsPerSide : ℕ := postsPerSide - 1
  let sideLength : ℚ := gapsPerSide * gapWidth + postsPerSide * postWidth
  4 * sideLength

/-- Theorem stating the perimeter of the square fence with given specifications -/
theorem square_fence_perimeter_is_77_and_third :
  squareFencePerimeter 16 (1/3) 6 = 77 + 1/3 := by
  sorry

end square_fence_perimeter_is_77_and_third_l1382_138293


namespace sum_of_first_10_odd_numbers_l1382_138225

def sum_of_odd_numbers (n : ℕ) : ℕ := n^2

theorem sum_of_first_10_odd_numbers : sum_of_odd_numbers 10 = 100 := by
  sorry

end sum_of_first_10_odd_numbers_l1382_138225


namespace geometric_relations_l1382_138289

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Axioms
axiom different_lines {m n : Line} : m ≠ n
axiom different_planes {α β : Plane} : α ≠ β

-- Theorem
theorem geometric_relations 
  (m n : Line) (α β : Plane) :
  (perpendicular_plane m α ∧ perpendicular m n → 
    parallel_plane n α ∨ contains α n) ∧
  (parallel_planes α β ∧ perpendicular_plane n α ∧ parallel_plane m β → 
    perpendicular m n) ∧
  (parallel_plane m α ∧ perpendicular_plane n β ∧ perpendicular m n → 
    ¬(perpendicular_planes α β)) ∧
  (parallel_plane m α ∧ perpendicular_plane n β ∧ parallel m n → 
    perpendicular_planes α β) :=
by sorry


end geometric_relations_l1382_138289


namespace original_number_is_two_l1382_138265

theorem original_number_is_two :
  ∃ (x : ℕ), 
    (∃ (y : ℕ), 
      (∀ (z : ℕ), z < y → ¬∃ (w : ℕ), x * z = w^3) ∧ 
      (∃ (w : ℕ), x * y = w^3) ∧
      x * y = 4 * x) →
    x = 2 := by
  sorry

end original_number_is_two_l1382_138265


namespace total_cement_is_15_1_l1382_138217

/-- The amount of cement used for Lexi's street in tons -/
def lexis_street_cement : ℝ := 10

/-- The amount of cement used for Tess's street in tons -/
def tesss_street_cement : ℝ := 5.1

/-- The total amount of cement used by Roadster's Paving Company in tons -/
def total_cement : ℝ := lexis_street_cement + tesss_street_cement

/-- Theorem stating that the total cement used is 15.1 tons -/
theorem total_cement_is_15_1 : total_cement = 15.1 := by
  sorry

end total_cement_is_15_1_l1382_138217


namespace absolute_value_equation_solution_l1382_138233

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2*x + 1| - |x - 5| = 6) ↔ (x = -12 ∨ x = 10/3) :=
by sorry

end absolute_value_equation_solution_l1382_138233


namespace incorrect_operation_l1382_138288

theorem incorrect_operation : 
  (∀ a : ℝ, (-a)^4 = a^4) ∧ 
  (∀ a : ℝ, -a + 3*a = 2*a) ∧ 
  (¬ ∀ a : ℝ, (2*a^2)^3 = 6*a^5) ∧ 
  (∀ a : ℝ, a^6 / a^2 = a^4) := by sorry

end incorrect_operation_l1382_138288


namespace expand_and_simplify_l1382_138232

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 4) + 6 = x^2 + x - 6 := by
  sorry

end expand_and_simplify_l1382_138232


namespace polynomial_evaluation_l1382_138264

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 2*x - 8 = 0) :
  x^3 - 2*x^2 - 8*x + 4 = 4 := by
  sorry

end polynomial_evaluation_l1382_138264


namespace matrix_multiplication_example_l1382_138248

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![2, 0; 5, -3]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![8, -2; 1, 1]
  A * B = !![16, -4; 37, -13] := by
  sorry

end matrix_multiplication_example_l1382_138248


namespace orange_buckets_total_l1382_138224

theorem orange_buckets_total (bucket1 bucket2 bucket3 : ℕ) : 
  bucket1 = 22 →
  bucket2 = bucket1 + 17 →
  bucket3 = bucket2 - 11 →
  bucket1 + bucket2 + bucket3 = 89 := by
sorry

end orange_buckets_total_l1382_138224


namespace fixed_point_exponential_function_l1382_138203

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2)
  f (-2) = 1 := by sorry

end fixed_point_exponential_function_l1382_138203


namespace max_apartment_size_l1382_138245

/-- Given:
  * The rental rate in Greenview is $1.20 per square foot.
  * Max's monthly budget for rent is $720.
  Prove that the largest apartment size Max can afford is 600 square feet. -/
theorem max_apartment_size (rental_rate : ℝ) (max_budget : ℝ) (max_size : ℝ) : 
  rental_rate = 1.20 →
  max_budget = 720 →
  max_size * rental_rate = max_budget →
  max_size = 600 := by
  sorry

#check max_apartment_size

end max_apartment_size_l1382_138245


namespace subcommittee_formation_ways_l1382_138246

def senate_committee_ways (total_republicans : ℕ) (total_democrats : ℕ) 
                          (subcommittee_republicans : ℕ) (subcommittee_democrats : ℕ) : ℕ :=
  Nat.choose total_republicans subcommittee_republicans * 
  Nat.choose total_democrats subcommittee_democrats

theorem subcommittee_formation_ways : 
  senate_committee_ways 10 8 4 3 = 11760 := by
  sorry

end subcommittee_formation_ways_l1382_138246


namespace banana_apple_sales_l1382_138208

/-- Proves that if the revenue from selling apples and bananas with reversed prices
    is $1 more than the revenue with original prices, then the number of bananas
    sold is 10 more than the number of apples sold. -/
theorem banana_apple_sales
  (apple_price : ℚ)
  (banana_price : ℚ)
  (apple_count : ℕ)
  (banana_count : ℕ)
  (h1 : apple_price = 0.5)
  (h2 : banana_price = 0.4)
  (h3 : banana_price * apple_count + apple_price * banana_count =
        apple_price * apple_count + banana_price * banana_count + 1) :
  banana_count = apple_count + 10 := by
sorry

end banana_apple_sales_l1382_138208


namespace sum_200_consecutive_integers_l1382_138256

theorem sum_200_consecutive_integers (n : ℕ) : 
  (n = 2000200000 ∨ n = 3000300000 ∨ n = 4000400000 ∨ n = 5000500000 ∨ n = 6000600000) →
  ¬∃ k : ℕ, n = (200 * (k + 100)) + 10050 := by
  sorry

end sum_200_consecutive_integers_l1382_138256


namespace crocodile_count_correct_l1382_138280

/-- Represents the number of crocodiles in the pond -/
def num_crocodiles : ℕ := 10

/-- Represents the number of frogs in the pond -/
def num_frogs : ℕ := 20

/-- Represents the number of eyes each animal (frog or crocodile) has -/
def eyes_per_animal : ℕ := 2

/-- Represents the total number of animal eyes in the pond -/
def total_eyes : ℕ := 60

/-- Theorem stating that the number of crocodiles is correct given the conditions -/
theorem crocodile_count_correct :
  num_crocodiles * eyes_per_animal + num_frogs * eyes_per_animal = total_eyes :=
sorry

end crocodile_count_correct_l1382_138280


namespace article_percentage_loss_l1382_138231

theorem article_percentage_loss 
  (selling_price : ℝ) 
  (selling_price_with_gain : ℝ) 
  (gain_percentage : ℝ) :
  selling_price = 136 →
  selling_price_with_gain = 192 →
  gain_percentage = 20 →
  let cost_price := selling_price_with_gain / (1 + gain_percentage / 100)
  let loss := cost_price - selling_price
  let percentage_loss := (loss / cost_price) * 100
  percentage_loss = 15 := by
sorry

end article_percentage_loss_l1382_138231


namespace program_flowchart_components_l1382_138211

-- Define a program flowchart
structure ProgramFlowchart where
  is_diagram : Bool
  represents_algorithm : Bool
  uses_specified_shapes : Bool
  uses_directional_lines : Bool
  uses_textual_explanations : Bool

-- Define the components of a program flowchart
structure FlowchartComponents where
  has_operation_boxes : Bool
  has_flow_lines_with_arrows : Bool
  has_textual_explanations : Bool

-- Theorem statement
theorem program_flowchart_components 
  (pf : ProgramFlowchart) 
  (h1 : pf.is_diagram = true)
  (h2 : pf.represents_algorithm = true)
  (h3 : pf.uses_specified_shapes = true)
  (h4 : pf.uses_directional_lines = true)
  (h5 : pf.uses_textual_explanations = true) :
  ∃ (fc : FlowchartComponents), 
    fc.has_operation_boxes = true ∧ 
    fc.has_flow_lines_with_arrows = true ∧ 
    fc.has_textual_explanations = true :=
  sorry

end program_flowchart_components_l1382_138211


namespace infinite_square_free_triples_l1382_138234

/-- A positive integer is square-free if it's not divisible by any perfect square greater than 1 -/
def IsSquareFree (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 1 → k * k ∣ n → k = 1

/-- The set of positive integers n for which n, n+1, and n+2 are all square-free -/
def SquareFreeTriples : Set ℕ :=
  {n : ℕ | n > 0 ∧ IsSquareFree n ∧ IsSquareFree (n + 1) ∧ IsSquareFree (n + 2)}

/-- The set of positive integers n for which n, n+1, and n+2 are all square-free is infinite -/
theorem infinite_square_free_triples : Set.Infinite SquareFreeTriples :=
sorry

end infinite_square_free_triples_l1382_138234


namespace tangent_line_equations_l1382_138274

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a - 2)*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a - 2)

-- Theorem statement
theorem tangent_line_equations (a : ℝ) 
  (h1 : ∀ x, f' a x = f' a (-x)) -- f' is an even function
  : (∃ x₀ y₀, x₀ ≠ 1 ∧ f a x₀ = y₀ ∧ 
    (y₀ - (-2)) / (x₀ - 1) = f' a x₀ ∧
    (2 * x + y = 0 ∨ 19 * x - 4 * y - 27 = 0)) :=
sorry

end tangent_line_equations_l1382_138274


namespace train_length_l1382_138281

/-- Calculates the length of a train given its speed and the time and distance it takes to cross a bridge -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * (1000 / 3600) →
  bridge_length = 150 →
  crossing_time = 20 →
  (train_speed * crossing_time) - bridge_length = 250 := by
  sorry

end train_length_l1382_138281


namespace rachel_hourly_wage_l1382_138284

/-- Rachel's earnings as a waitress in a coffee shop -/
def rachel_earnings (people_served : ℕ) (tip_per_person : ℚ) (total_earnings : ℚ) : Prop :=
  let total_tips : ℚ := people_served * tip_per_person
  let hourly_wage_without_tips : ℚ := total_earnings - total_tips
  hourly_wage_without_tips = 12

theorem rachel_hourly_wage :
  rachel_earnings 20 (25/20) 37 := by
  sorry

end rachel_hourly_wage_l1382_138284


namespace milk_price_calculation_l1382_138285

/-- Proves that given the initial volume of milk, volume of water added, and final price of the mixture,
    the original price of milk per litre can be calculated. -/
theorem milk_price_calculation (initial_milk_volume : ℝ) (water_added : ℝ) (final_mixture_price : ℝ) :
  initial_milk_volume = 60 →
  water_added = 15 →
  final_mixture_price = 32 / 3 →
  ∃ (original_milk_price : ℝ), original_milk_price = 800 / 60 := by
  sorry

end milk_price_calculation_l1382_138285


namespace ball_probabilities_l1382_138283

theorem ball_probabilities (total_balls : ℕ) (red_prob black_prob white_prob green_prob : ℚ)
  (h_total : total_balls = 12)
  (h_red : red_prob = 5 / 12)
  (h_black : black_prob = 1 / 3)
  (h_white : white_prob = 1 / 6)
  (h_green : green_prob = 1 / 12)
  (h_sum : red_prob + black_prob + white_prob + green_prob = 1) :
  (red_prob + black_prob = 3 / 4) ∧ (red_prob + black_prob + white_prob = 11 / 12) := by
  sorry

end ball_probabilities_l1382_138283


namespace probability_two_non_red_marbles_l1382_138258

/-- Given a bag of marbles, calculate the probability of drawing two non-red marbles in succession with replacement after the first draw. -/
theorem probability_two_non_red_marbles 
  (total_marbles : ℕ) 
  (red_marbles : ℕ) 
  (h1 : total_marbles = 84) 
  (h2 : red_marbles = 12) :
  (total_marbles - red_marbles : ℚ) / total_marbles * 
  ((total_marbles - red_marbles : ℚ) / total_marbles) = 36/49 := by
  sorry

end probability_two_non_red_marbles_l1382_138258


namespace chucks_team_lead_l1382_138254

/-- Represents a team in the basketball match -/
inductive Team
| ChucksTeam
| YellowTeam

/-- Represents a quarter in the basketball match -/
inductive Quarter
| First
| Second
| Third
| Fourth

/-- Calculates the score for a given team in a given quarter -/
def quarterScore (team : Team) (quarter : Quarter) : ℤ :=
  match team, quarter with
  | Team.ChucksTeam, Quarter.First => 23
  | Team.ChucksTeam, Quarter.Second => 18
  | Team.ChucksTeam, Quarter.Third => 19
  | Team.ChucksTeam, Quarter.Fourth => 17
  | Team.YellowTeam, Quarter.First => 24
  | Team.YellowTeam, Quarter.Second => 19
  | Team.YellowTeam, Quarter.Third => 14
  | Team.YellowTeam, Quarter.Fourth => 16

/-- Points gained from technical fouls -/
def technicalFoulPoints (team : Team) : ℤ :=
  match team with
  | Team.ChucksTeam => 3
  | Team.YellowTeam => 2

/-- Calculates the total score for a team -/
def totalScore (team : Team) : ℤ :=
  quarterScore team Quarter.First +
  quarterScore team Quarter.Second +
  quarterScore team Quarter.Third +
  quarterScore team Quarter.Fourth +
  technicalFoulPoints team

/-- The main theorem stating Chuck's Team's lead -/
theorem chucks_team_lead :
  totalScore Team.ChucksTeam - totalScore Team.YellowTeam = 5 := by
  sorry


end chucks_team_lead_l1382_138254


namespace bus_car_ratio_l1382_138240

theorem bus_car_ratio (num_cars : ℕ) (num_buses : ℕ) : 
  num_cars = 85 →
  num_buses = num_cars - 80 →
  (num_buses : ℚ) / (num_cars : ℚ) = 1 / 17 := by
  sorry

end bus_car_ratio_l1382_138240


namespace seed_germination_problem_l1382_138282

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  (0.30 * x + 0.50 * 200) / (x + 200) = 0.35714285714285715 → 
  x = 500 := by
sorry

end seed_germination_problem_l1382_138282


namespace album_jumps_l1382_138228

/-- Calculates the total number of jumps a person can make while listening to an album. -/
theorem album_jumps (jumps_per_second : ℕ) (song_length : ℚ) (num_songs : ℕ) :
  jumps_per_second = 1 →
  song_length = 3.5 →
  num_songs = 10 →
  (jumps_per_second * 60 : ℚ) * (song_length * num_songs) = 2100 := by
  sorry

end album_jumps_l1382_138228


namespace nested_fraction_evaluation_l1382_138298

theorem nested_fraction_evaluation :
  2 + (3 / (4 + (5 / (6 + 7/8)))) = 137/52 := by
  sorry

end nested_fraction_evaluation_l1382_138298


namespace parabola_directrix_l1382_138241

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = 4 * x^2

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop := y = -1/16

/-- Theorem: The directrix of the parabola y = 4x^2 is y = -1/16 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_directrix : ℝ, directrix_equation y_directrix :=
sorry

end parabola_directrix_l1382_138241


namespace fair_lines_theorem_l1382_138201

/-- Represents the number of people in the bumper cars line -/
def bumper_cars_line (initial : ℕ) (left : ℕ) (joined : ℕ) : ℕ :=
  initial - left + joined

/-- Represents the total number of people in both lines -/
def total_people (bumper_cars : ℕ) (roller_coaster : ℕ) : ℕ :=
  bumper_cars + roller_coaster

theorem fair_lines_theorem (x y Z : ℕ) (h1 : Z = bumper_cars_line 25 x y) 
  (h2 : Z ≥ x) : total_people Z 15 = 40 - x + y := by
  sorry

#check fair_lines_theorem

end fair_lines_theorem_l1382_138201


namespace exists_triangle_altitudes_form_triangle_but_not_bisectors_l1382_138278

/-- A triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The altitude triangle formed by the altitudes of the original triangle. -/
def AltitudeTriangle (t : Triangle) : Triangle := sorry

/-- The angle bisectors of a triangle. -/
def AngleBisectors (t : Triangle) : Fin 3 → ℝ := sorry

/-- Predicate to check if three lengths can form a triangle. -/
def CanFormTriangle (l₁ l₂ l₃ : ℝ) : Prop :=
  l₁ + l₂ > l₃ ∧ l₂ + l₃ > l₁ ∧ l₃ + l₁ > l₂

theorem exists_triangle_altitudes_form_triangle_but_not_bisectors :
  ∃ t : Triangle,
    CanFormTriangle (AltitudeTriangle t).a (AltitudeTriangle t).b (AltitudeTriangle t).c ∧
    ¬CanFormTriangle (AngleBisectors (AltitudeTriangle t) 0)
                     (AngleBisectors (AltitudeTriangle t) 1)
                     (AngleBisectors (AltitudeTriangle t) 2) :=
sorry

end exists_triangle_altitudes_form_triangle_but_not_bisectors_l1382_138278


namespace intersection_point_integer_coordinates_l1382_138270

theorem intersection_point_integer_coordinates (m : ℕ+) : 
  (∃ x y : ℤ, 17 * x + 7 * y = 1000 ∧ y = m * x + 2) ↔ m = 68 :=
sorry

end intersection_point_integer_coordinates_l1382_138270


namespace min_value_product_min_value_product_achieved_l1382_138227

theorem min_value_product (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by
  sorry

theorem min_value_product_achieved (x : ℝ) : 
  ∃ y : ℝ, (15 - y) * (13 - y) * (15 + y) * (13 + y) = -784 :=
by
  sorry

end min_value_product_min_value_product_achieved_l1382_138227


namespace quadratic_inequalities_l1382_138262

theorem quadratic_inequalities :
  (∀ x : ℝ, 2 * x^2 + x + 1 > 0) ∧
  (∃ a b : ℝ, (∀ x : ℝ, a * x^2 + b * x + 2 > 0 ↔ -1/2 < x ∧ x < 2) ∧ a + b = 1) := by
  sorry

end quadratic_inequalities_l1382_138262


namespace dance_theorem_l1382_138230

/-- Represents a dance function with boys and girls -/
structure DanceFunction where
  boys : ℕ
  girls : ℕ
  first_boy_dances : ℕ
  last_boy_dances_all : Prop

/-- The relationship between boys and girls in the dance function -/
def dance_relationship (df : DanceFunction) : Prop :=
  df.boys = df.girls - df.first_boy_dances + 1

theorem dance_theorem (df : DanceFunction) 
  (h1 : df.first_boy_dances = 6)
  (h2 : df.last_boy_dances_all)
  : df.boys = df.girls - 5 := by
  sorry

end dance_theorem_l1382_138230


namespace sum_of_square_areas_l1382_138220

theorem sum_of_square_areas (side1 side2 : ℝ) (h1 : side1 = 11) (h2 : side2 = 5) :
  side1 * side1 + side2 * side2 = 146 := by
  sorry

end sum_of_square_areas_l1382_138220


namespace white_balls_count_l1382_138219

theorem white_balls_count (total green blue yellow white : ℕ) : 
  total = green + blue + yellow + white →
  4 * green = total →
  8 * blue = total →
  12 * yellow = total →
  blue = 6 →
  white = 26 := by
  sorry

end white_balls_count_l1382_138219


namespace penultimate_digit_of_quotient_l1382_138253

theorem penultimate_digit_of_quotient : ∃ k : ℕ, 
  (4^1994 + 7^1994) / 10 = k * 10 + 1 := by
  sorry

end penultimate_digit_of_quotient_l1382_138253


namespace range_of_b_l1382_138226

/-- A region in the xy-plane defined by y ≤ 3x + b -/
def region (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≤ 3 * p.1 + b}

/-- The theorem stating the range of b given the conditions -/
theorem range_of_b :
  ∀ b : ℝ,
  (¬ ((3, 4) ∈ region b) ∧ ((4, 4) ∈ region b)) ↔
  (-8 ≤ b ∧ b < -5) :=
by sorry

end range_of_b_l1382_138226


namespace periodic_function_l1382_138249

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x| ≤ 1)
  (h2 : ∀ x, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f 1 :=
sorry

end periodic_function_l1382_138249


namespace simplify_expression_l1382_138279

theorem simplify_expression (x : ℝ) : (2*x)^4 + (3*x)*(x^3) = 19*x^4 := by
  sorry

end simplify_expression_l1382_138279


namespace cube_side_length_l1382_138255

theorem cube_side_length (s₂ : ℝ) : 
  s₂ > 0 →
  (6 * s₂^2) / (6 * 1^2) = 36 →
  s₂ = 6 := by
sorry

end cube_side_length_l1382_138255


namespace copper_percentage_in_first_alloy_l1382_138222

/-- The percentage of copper in the first alloy -/
def first_alloy_copper_percentage : ℝ := 25

/-- The percentage of copper in the second alloy -/
def second_alloy_copper_percentage : ℝ := 50

/-- The weight of the first alloy used -/
def first_alloy_weight : ℝ := 200

/-- The weight of the second alloy used -/
def second_alloy_weight : ℝ := 800

/-- The total weight of the final alloy -/
def total_weight : ℝ := 1000

/-- The percentage of copper in the final alloy -/
def final_alloy_copper_percentage : ℝ := 45

theorem copper_percentage_in_first_alloy :
  (first_alloy_weight * first_alloy_copper_percentage / 100 +
   second_alloy_weight * second_alloy_copper_percentage / 100) / total_weight * 100 =
  final_alloy_copper_percentage :=
by sorry

end copper_percentage_in_first_alloy_l1382_138222


namespace min_ballots_proof_l1382_138237

/-- Represents the number of candidates for each position -/
def candidates : List Nat := [3, 4, 5]

/-- Represents the requirement that each candidate must appear under each number
    an equal number of times -/
def equal_appearance (ballots : Nat) : Prop :=
  ∀ n ∈ candidates, ballots % n = 0

/-- The minimum number of different ballots required -/
def min_ballots : Nat := 5

/-- Theorem stating that the minimum number of ballots satisfying the equal appearance
    requirement is 5 -/
theorem min_ballots_proof :
  (∀ k : Nat, k < min_ballots → ¬(equal_appearance k)) ∧
  (equal_appearance min_ballots) :=
sorry

end min_ballots_proof_l1382_138237


namespace prob_no_consecutive_heads_10_is_9_64_l1382_138251

/-- The number of coin tosses -/
def n : ℕ := 10

/-- The probability of getting heads on a single toss -/
def p : ℚ := 1/2

/-- The number of ways to arrange k heads in n tosses without consecutive heads -/
def non_consecutive_heads (n k : ℕ) : ℕ := Nat.choose (n - k + 1) k

/-- The total number of favorable outcomes -/
def total_favorable_outcomes (n : ℕ) : ℕ :=
  (List.range (n/2 + 1)).map (non_consecutive_heads n) |>.sum

/-- The probability of not having consecutive heads in n fair coin tosses -/
def prob_no_consecutive_heads (n : ℕ) : ℚ :=
  (total_favorable_outcomes n : ℚ) / 2^n

/-- The main theorem -/
theorem prob_no_consecutive_heads_10_is_9_64 :
  prob_no_consecutive_heads n = 9/64 := by sorry

end prob_no_consecutive_heads_10_is_9_64_l1382_138251


namespace exists_k_l1382_138244

/-- A game configuration with two players and blank squares. -/
structure GameConfig where
  s₁ : ℕ  -- Steps for player 1
  s₂ : ℕ  -- Steps for player 2
  board_size : ℕ  -- Total number of squares on the board

/-- Winning probability for a player given a game configuration and number of blank squares. -/
def winning_probability (config : GameConfig) (player : ℕ) (num_blanks : ℕ) : ℝ :=
  sorry

/-- The statement that proves the existence of k satisfying the given conditions. -/
theorem exists_k (config : GameConfig) : ∃ k : ℕ,
  (∀ n < k, winning_probability config 1 n > 1/2) ∧
  (∃ board_config : List ℕ, 
    board_config.length = k ∧ 
    winning_probability config 2 k > 1/2) :=
by
  -- Assume s₁ = 3 and s₂ = 2
  have h1 : config.s₁ = 3 := by sorry
  have h2 : config.s₂ = 2 := by sorry

  -- Prove that k = 3 satisfies the conditions
  use 3
  sorry


end exists_k_l1382_138244


namespace buckets_taken_away_is_three_l1382_138242

/-- Calculates the number of buckets taken away to reach the bath level -/
def buckets_taken_away (bucket_capacity : ℕ) (buckets_to_fill : ℕ) (weekly_usage : ℕ) (baths_per_week : ℕ) : ℕ :=
  let full_tub := bucket_capacity * buckets_to_fill
  let bath_level := weekly_usage / baths_per_week
  let difference := full_tub - bath_level
  difference / bucket_capacity

/-- Proves that the number of buckets taken away is 3 given the problem conditions -/
theorem buckets_taken_away_is_three :
  buckets_taken_away 120 14 9240 7 = 3 := by
  sorry

end buckets_taken_away_is_three_l1382_138242


namespace childrens_cookbook_cost_l1382_138221

theorem childrens_cookbook_cost (dictionary_cost dinosaur_book_cost savings needed_more total_cost : ℕ) :
  dictionary_cost = 11 →
  dinosaur_book_cost = 19 →
  savings = 8 →
  needed_more = 29 →
  total_cost = savings + needed_more →
  total_cost - (dictionary_cost + dinosaur_book_cost) = 7 := by
  sorry

end childrens_cookbook_cost_l1382_138221


namespace computer_price_reduction_l1382_138276

/-- Given a computer with original price x, after reducing it by m yuan and then by 20%,
    resulting in a final price of n yuan, prove that the original price x is equal to (5/4)n + m. -/
theorem computer_price_reduction (x m n : ℝ) (h : (x - m) * (1 - 0.2) = n) :
  x = (5/4) * n + m := by
  sorry

end computer_price_reduction_l1382_138276


namespace school_teachers_count_l1382_138287

theorem school_teachers_count (total : ℕ) (sample_size : ℕ) (sampled_students : ℕ) : 
  total = 2400 →
  sample_size = 160 →
  sampled_students = 150 →
  (total : ℚ) / sample_size = 15 →
  total - (sampled_students * ((total : ℚ) / sample_size).floor) = 150 :=
by sorry

end school_teachers_count_l1382_138287


namespace investment_principal_calculation_l1382_138297

/-- Proves that given a monthly interest payment of $234 and a simple annual interest rate of 9%,
    the principal amount of the investment is $31,200. -/
theorem investment_principal_calculation (monthly_interest : ℝ) (annual_rate : ℝ) :
  monthly_interest = 234 →
  annual_rate = 0.09 →
  (monthly_interest * 12) / annual_rate = 31200 := by
  sorry

end investment_principal_calculation_l1382_138297


namespace train_speed_problem_l1382_138209

theorem train_speed_problem (x : ℝ) (h : x > 0) :
  let total_distance := 3 * x
  let first_distance := x
  let second_distance := 2 * x
  let second_speed := 20
  let average_speed := 26
  let time_first := first_distance / V
  let time_second := second_distance / second_speed
  let total_time := time_first + time_second
  average_speed = total_distance / total_time →
  V = 65 := by
sorry

end train_speed_problem_l1382_138209


namespace andrew_work_hours_l1382_138205

/-- Calculates the total hours worked given the number of days and hours per day -/
def total_hours (days : ℕ) (hours_per_day : ℝ) : ℝ :=
  days * hours_per_day

/-- Proves that Andrew worked for 7.5 hours given the conditions -/
theorem andrew_work_hours :
  let days : ℕ := 3
  let hours_per_day : ℝ := 2.5
  total_hours days hours_per_day = 7.5 := by
sorry

end andrew_work_hours_l1382_138205


namespace root_implies_ab_leq_one_l1382_138243

theorem root_implies_ab_leq_one (a b : ℝ) : 
  ((a + b + a) * (a + b + b) = 9) → ab ≤ 1 := by
  sorry

end root_implies_ab_leq_one_l1382_138243


namespace number_equation_solution_l1382_138267

theorem number_equation_solution : ∃ x : ℝ, (0.75 * x + 2 = 8) ∧ (x = 8) := by
  sorry

end number_equation_solution_l1382_138267


namespace smallest_sum_of_leftmost_three_digits_l1382_138286

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def contains_zero (n : ℕ) : Prop := ∃ (a c : ℕ), n = 100 * a + c ∧ a < 10 ∧ c < 100

def all_digits_different (x y : ℕ) : Prop :=
  ∀ (d : ℕ), d < 10 → (
    (∃ (i : ℕ), i < 3 ∧ (x / 10^i) % 10 = d) ↔
    ¬(∃ (j : ℕ), j < 3 ∧ (y / 10^j) % 10 = d)
  )

def sum_of_leftmost_three_digits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10)

theorem smallest_sum_of_leftmost_three_digits
  (x y : ℕ)
  (hx : is_three_digit x)
  (hy : is_three_digit y)
  (hx0 : contains_zero x)
  (hdiff : all_digits_different x y)
  (hsum : 1000 ≤ x + y ∧ x + y ≤ 9999) :
  ∀ (z : ℕ), is_three_digit z → contains_zero z → all_digits_different z (x + y - z) →
    sum_of_leftmost_three_digits (x + y) ≤ sum_of_leftmost_three_digits (z + (x + y - z)) :=
by sorry

end smallest_sum_of_leftmost_three_digits_l1382_138286


namespace count_squares_below_line_l1382_138294

/-- The number of 1x1 squares in the first quadrant with interiors lying entirely below the line 7x + 268y = 1876 -/
def squares_below_line : ℕ := 801

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 7 * x + 268 * y = 1876

theorem count_squares_below_line :
  squares_below_line = 801 :=
sorry

end count_squares_below_line_l1382_138294


namespace equation_solution_l1382_138266

theorem equation_solution :
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 :=
by sorry

end equation_solution_l1382_138266


namespace joint_completion_time_l1382_138214

/-- Given two people A and B who can complete a task in x and y hours respectively,
    the time it takes for them to complete the task together is xy/(x+y) hours. -/
theorem joint_completion_time (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x⁻¹ + y⁻¹)⁻¹ = x * y / (x + y) :=
by sorry

end joint_completion_time_l1382_138214


namespace abc_inequality_l1382_138272

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : a + b + c ≥ 3 := by
  sorry

end abc_inequality_l1382_138272


namespace simplify_fraction_l1382_138299

theorem simplify_fraction : (5^3 + 5^5) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l1382_138299


namespace x_power_twelve_equals_one_l1382_138235

theorem x_power_twelve_equals_one (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^12 = 1 := by
  sorry

end x_power_twelve_equals_one_l1382_138235


namespace mark_chocolates_proof_l1382_138212

/-- The number of chocolates Mark started with --/
def initial_chocolates : ℕ := 104

/-- The number of chocolates Mark's sister took --/
def sister_chocolates : ℕ → Prop := λ x => 5 ≤ x ∧ x ≤ 10

theorem mark_chocolates_proof :
  ∃ (sister_took : ℕ),
    sister_chocolates sister_took ∧
    (initial_chocolates / 4 : ℚ) * 3 / 3 * 2 - 40 - sister_took = 4 ∧
    initial_chocolates % 4 = 0 ∧
    initial_chocolates % 2 = 0 :=
sorry

end mark_chocolates_proof_l1382_138212


namespace unfollows_calculation_correct_l1382_138292

/-- Calculates the number of unfollows for an Instagram influencer over a year -/
def calculate_unfollows (initial_followers : ℕ) (daily_new_followers : ℕ) (final_followers : ℕ) : ℕ :=
  let potential_followers := initial_followers + daily_new_followers * 365
  potential_followers - final_followers

/-- Theorem: The number of unfollows is correct given the problem conditions -/
theorem unfollows_calculation_correct :
  calculate_unfollows 100000 1000 445000 = 20000 := by
  sorry

end unfollows_calculation_correct_l1382_138292


namespace second_drawn_number_l1382_138247

def systematicSampling (totalStudents : ℕ) (sampleSize : ℕ) (firstDrawn : ℕ) : ℕ → ℕ :=
  fun n => firstDrawn + (totalStudents / sampleSize) * (n - 1)

theorem second_drawn_number
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (firstDrawn : ℕ)
  (h1 : totalStudents = 500)
  (h2 : sampleSize = 50)
  (h3 : firstDrawn = 3) :
  systematicSampling totalStudents sampleSize firstDrawn 2 = 13 := by
  sorry

end second_drawn_number_l1382_138247


namespace gold_silver_coin_values_l1382_138271

theorem gold_silver_coin_values :
  ∃! n : ℕ, n > 0 ∧ 
  (∃ S : Finset ℕ, S.card = n ∧
    ∀ x ∈ S, x > 0 ∧
    ∃ y : ℕ, y > 0 ∧ y < 100 ∧
    (100 + x) * (100 - y) = 10000) :=
by
  -- The proof goes here
  sorry

end gold_silver_coin_values_l1382_138271


namespace jessica_watermelons_l1382_138215

/-- Given that Jessica grew some watermelons and 30 carrots,
    rabbits ate 27 watermelons, and Jessica has 8 watermelons left,
    prove that Jessica originally grew 35 watermelons. -/
theorem jessica_watermelons :
  ∀ (original_watermelons : ℕ) (carrots : ℕ),
    carrots = 30 →
    original_watermelons - 27 = 8 →
    original_watermelons = 35 := by
  sorry

end jessica_watermelons_l1382_138215


namespace egg_weight_probability_l1382_138239

theorem egg_weight_probability (p_less_than_30 : ℝ) (p_between_30_and_40 : ℝ) 
  (h1 : p_less_than_30 = 0.3)
  (h2 : p_between_30_and_40 = 0.5) :
  1 - p_less_than_30 = 0.7 := by
  sorry

end egg_weight_probability_l1382_138239


namespace max_homework_time_l1382_138296

def homework_time (biology_time : ℕ) : ℕ :=
  let history_time := 2 * biology_time
  let geography_time := 3 * history_time
  biology_time + history_time + geography_time

theorem max_homework_time :
  homework_time 20 = 180 := by
  sorry

end max_homework_time_l1382_138296


namespace solution_set_implies_m_equals_one_l1382_138261

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 3*m*x - 4

-- State the theorem
theorem solution_set_implies_m_equals_one :
  (∀ x : ℝ, f m x < 0 ↔ -4 < x ∧ x < 1) → m = 1 := by
  sorry

end solution_set_implies_m_equals_one_l1382_138261


namespace greatest_number_of_sets_l1382_138260

theorem greatest_number_of_sets (t_shirts : ℕ) (buttons : ℕ) : 
  t_shirts = 4 → buttons = 20 → 
  (∃ (sets : ℕ), sets > 0 ∧ 
    t_shirts % sets = 0 ∧ 
    buttons % sets = 0 ∧
    ∀ (k : ℕ), k > 0 ∧ t_shirts % k = 0 ∧ buttons % k = 0 → k ≤ sets) →
  Nat.gcd t_shirts buttons = 4 :=
by sorry

end greatest_number_of_sets_l1382_138260


namespace car_trip_speed_l1382_138218

theorem car_trip_speed (initial_speed initial_time total_speed total_time : ℝ) 
  (h1 : initial_speed = 45)
  (h2 : initial_time = 4)
  (h3 : total_speed = 65)
  (h4 : total_time = 12) :
  let remaining_time := total_time - initial_time
  let initial_distance := initial_speed * initial_time
  let total_distance := total_speed * total_time
  let remaining_distance := total_distance - initial_distance
  remaining_distance / remaining_time = 75 := by sorry

end car_trip_speed_l1382_138218
