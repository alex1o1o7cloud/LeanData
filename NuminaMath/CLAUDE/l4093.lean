import Mathlib

namespace score_well_defined_and_nonnegative_l4093_409393

theorem score_well_defined_and_nonnegative (N C : ℕ+) 
  (h1 : N ≤ 20) (h2 : C ≥ 1) : 
  ∃ (score : ℕ), score = ⌊(N : ℝ) / (C : ℝ)⌋ ∧ score ≥ 0 := by
  sorry

end score_well_defined_and_nonnegative_l4093_409393


namespace gordon_jamie_maine_coon_difference_total_cats_verification_l4093_409379

/-- The number of Persian cats Jamie owns -/
def jamie_persians : ℕ := 4

/-- The number of Maine Coons Jamie owns -/
def jamie_maine_coons : ℕ := 2

/-- The number of Persian cats Gordon owns -/
def gordon_persians : ℕ := jamie_persians / 2

/-- The total number of cats -/
def total_cats : ℕ := 13

/-- The number of Maine Coons Gordon owns -/
def gordon_maine_coons : ℕ := 3

theorem gordon_jamie_maine_coon_difference :
  gordon_maine_coons - jamie_maine_coons = 1 :=
by
  sorry

/-- The number of Maine Coons Hawkeye owns -/
def hawkeye_maine_coons : ℕ := gordon_maine_coons - 1

/-- Verification that the total number of cats is correct -/
theorem total_cats_verification :
  jamie_persians + jamie_maine_coons + gordon_persians + gordon_maine_coons +
  hawkeye_maine_coons = total_cats :=
by
  sorry

end gordon_jamie_maine_coon_difference_total_cats_verification_l4093_409379


namespace steve_sleeping_time_l4093_409308

theorem steve_sleeping_time (total_hours : ℝ) (school_fraction : ℝ) (assignment_fraction : ℝ) (family_hours : ℝ)
  (h1 : total_hours = 24)
  (h2 : school_fraction = 1 / 6)
  (h3 : assignment_fraction = 1 / 12)
  (h4 : family_hours = 10) :
  (total_hours - (school_fraction * total_hours + assignment_fraction * total_hours + family_hours)) / total_hours = 1 / 3 := by
  sorry

end steve_sleeping_time_l4093_409308


namespace family_size_theorem_l4093_409375

def family_size_problem (fathers_side : ℕ) (total : ℕ) : Prop :=
  let mothers_side := total - fathers_side
  let difference := mothers_side - fathers_side
  let percentage := (difference : ℚ) / fathers_side * 100
  fathers_side = 10 ∧ total = 23 → percentage = 30

theorem family_size_theorem :
  family_size_problem 10 23 := by
  sorry

end family_size_theorem_l4093_409375


namespace betsy_games_won_l4093_409317

theorem betsy_games_won (betsy helen susan : ℕ) 
  (helen_games : helen = 2 * betsy)
  (susan_games : susan = 3 * betsy)
  (total_games : betsy + helen + susan = 30) :
  betsy = 5 := by
sorry

end betsy_games_won_l4093_409317


namespace tangent_line_through_origin_l4093_409357

/-- The tangent line to y = e^x passing through the origin -/
theorem tangent_line_through_origin :
  ∃! (a b : ℝ), 
    b = Real.exp a ∧ 
    0 = b - (Real.exp a) * a ∧ 
    a = 1 ∧ 
    b = Real.exp 1 ∧
    Real.exp a = Real.exp 1 := by
  sorry

end tangent_line_through_origin_l4093_409357


namespace profit_achieved_min_disks_optimal_l4093_409359

/-- The number of disks Maria buys for $6 -/
def buy_rate : ℕ := 5

/-- The price Maria pays for buy_rate disks -/
def buy_price : ℚ := 6

/-- The number of disks Maria sells for $7 -/
def sell_rate : ℕ := 4

/-- The price Maria receives for sell_rate disks -/
def sell_price : ℚ := 7

/-- The target profit Maria wants to achieve -/
def target_profit : ℚ := 120

/-- The minimum number of disks Maria must sell to make the target profit -/
def min_disks_to_sell : ℕ := 219

theorem profit_achieved (n : ℕ) : 
  n ≥ min_disks_to_sell → 
  n * (sell_price / sell_rate - buy_price / buy_rate) ≥ target_profit :=
by sorry

theorem min_disks_optimal : 
  ∀ m : ℕ, m < min_disks_to_sell → 
  m * (sell_price / sell_rate - buy_price / buy_rate) < target_profit :=
by sorry

end profit_achieved_min_disks_optimal_l4093_409359


namespace reunion_handshakes_l4093_409339

/-- The number of handshakes when n boys each shake hands once with every other boy -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a reunion of 11 boys where each boy shakes hands exactly once with each of the others, 
    the total number of handshakes is 55 -/
theorem reunion_handshakes : handshakes 11 = 55 := by
  sorry

end reunion_handshakes_l4093_409339


namespace complex_equations_solutions_l4093_409381

theorem complex_equations_solutions :
  let x₁ : ℚ := -7/5
  let y₁ : ℚ := 5
  let x₂ : ℚ := 5
  let y₂ : ℚ := -1
  (3 * y₁ : ℂ) + (5 * x₁ * I) = 15 - 7 * I ∧
  (2 * x₂ + 3 * y₂ : ℂ) + ((x₂ - y₂) * I) = 7 + 6 * I :=
by sorry

end complex_equations_solutions_l4093_409381


namespace second_pipe_fill_time_l4093_409371

theorem second_pipe_fill_time (pipe1_time pipe2_time outlet_time all_pipes_time : ℝ) 
  (h1 : pipe1_time = 18)
  (h2 : outlet_time = 45)
  (h3 : all_pipes_time = 0.08333333333333333)
  (h4 : 1 / pipe1_time + 1 / pipe2_time - 1 / outlet_time = 1 / all_pipes_time) :
  pipe2_time = 20 := by sorry

end second_pipe_fill_time_l4093_409371


namespace extension_point_coordinates_l4093_409325

/-- Given two points P₁ and P₂ in ℝ², and a point P on the extension line of P₁P₂ 
    such that |⃗P₁P| = 2|⃗PP₂|, prove that P has coordinates (-2, 11). -/
theorem extension_point_coordinates (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (2, -1) → 
  P₂ = (0, 5) → 
  (∃ t : ℝ, P = P₁ + t • (P₂ - P₁)) → 
  ‖P - P₁‖ = 2 * ‖P₂ - P‖ → 
  P = (-2, 11) := by
  sorry

end extension_point_coordinates_l4093_409325


namespace wire_length_ratio_l4093_409346

/-- The ratio of wire lengths for equivalent volume cubes -/
theorem wire_length_ratio (large_cube_edge : ℝ) (small_cube_edge : ℝ) : 
  large_cube_edge = 8 →
  small_cube_edge = 2 →
  (12 * large_cube_edge) / (12 * small_cube_edge * (large_cube_edge / small_cube_edge)^3) = 1/16 := by
  sorry

#check wire_length_ratio

end wire_length_ratio_l4093_409346


namespace initial_boys_count_l4093_409341

/-- Given an initial group of boys with an average weight of 102 kg,
    adding a new person weighing 40 kg reduces the average by 2 kg.
    This function calculates the initial number of boys. -/
def initial_number_of_boys : ℕ :=
  let initial_avg : ℚ := 102
  let new_person_weight : ℚ := 40
  let avg_decrease : ℚ := 2
  let n : ℕ := 30  -- The number we want to prove
  n

/-- Theorem stating that the initial number of boys is 30 -/
theorem initial_boys_count :
  let n := initial_number_of_boys
  let initial_avg : ℚ := 102
  let new_person_weight : ℚ := 40
  let avg_decrease : ℚ := 2
  (n : ℚ) * initial_avg + new_person_weight = (n + 1) * (initial_avg - avg_decrease) :=
by sorry

end initial_boys_count_l4093_409341


namespace original_workers_count_l4093_409326

/-- Represents the number of days required to complete the work. -/
def original_days : ℕ := 12

/-- Represents the number of days saved after additional workers joined. -/
def days_saved : ℕ := 4

/-- Represents the number of additional workers who joined. -/
def additional_workers : ℕ := 5

/-- Represents the number of additional workers working at twice the original rate. -/
def double_rate_workers : ℕ := 3

/-- Represents the number of additional workers working at the original rate. -/
def normal_rate_workers : ℕ := 2

/-- Theorem stating that the original number of workers is 16. -/
theorem original_workers_count : ℕ := by
  sorry

#check original_workers_count

end original_workers_count_l4093_409326


namespace complex_equation_imag_part_l4093_409344

theorem complex_equation_imag_part : 
  ∃ (z : ℂ), (3 - 4*I) * z = Complex.abs (4 + 3*I) ∧ z.im = 4/5 := by sorry

end complex_equation_imag_part_l4093_409344


namespace esteban_exercise_time_l4093_409318

/-- Proves that Esteban exercised for 10 minutes each day given the conditions. -/
theorem esteban_exercise_time :
  -- Natasha's exercise time per day in minutes
  let natasha_daily := 30
  -- Number of days Natasha exercised
  let natasha_days := 7
  -- Number of days Esteban exercised
  let esteban_days := 9
  -- Total exercise time for both in hours
  let total_hours := 5
  -- Calculate Esteban's daily exercise time in minutes
  let esteban_daily := 
    (total_hours * 60 - natasha_daily * natasha_days) / esteban_days
  -- Prove that Esteban's daily exercise time is 10 minutes
  esteban_daily = 10 := by
  sorry

end esteban_exercise_time_l4093_409318


namespace specific_function_value_l4093_409334

def is_odd_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = f x)

def agrees_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc 0 1 → f x = x

theorem specific_function_value
  (f : ℝ → ℝ)
  (h_odd_periodic : is_odd_periodic f)
  (h_unit : agrees_on_unit_interval f) :
  f 2011.5 = -0.5 := by
sorry

end specific_function_value_l4093_409334


namespace multiple_of_17_binary_properties_l4093_409351

/-- A function that returns the number of 1's in the binary representation of a natural number -/
def count_ones (n : ℕ) : ℕ := sorry

/-- A function that returns the number of 0's in the binary representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem multiple_of_17_binary_properties (n : ℕ) 
  (h1 : n % 17 = 0) 
  (h2 : count_ones n = 3) : 
  (count_zeros n ≥ 6) ∧ 
  (count_zeros n = 7 → Even n) := by
  sorry

end multiple_of_17_binary_properties_l4093_409351


namespace polynomial_division_theorem_l4093_409358

theorem polynomial_division_theorem (x : ℝ) : 
  (9*x^3 + 32*x^2 + 89*x + 271)*(x - 3) + 801 = 9*x^4 + 5*x^3 - 7*x^2 + 4*x - 12 := by
  sorry

end polynomial_division_theorem_l4093_409358


namespace inequality_proof_l4093_409388

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
sorry

end inequality_proof_l4093_409388


namespace factorization_proof_l4093_409321

theorem factorization_proof : 989 * 1001 * 1007 + 320 = 991 * 997 * 1009 := by
  sorry

end factorization_proof_l4093_409321


namespace max_tan_b_in_triangle_l4093_409362

/-- Given a triangle ABC with AB = 25 and BC = 15, the maximum value of tan B is 4/3 -/
theorem max_tan_b_in_triangle (A B C : ℝ × ℝ) :
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B = 25 →
  d B C = 15 →
  ∀ C' : ℝ × ℝ, d A C' ≥ d A C → d B C' = 15 →
  Real.tan (Real.arccos ((d A B)^2 + (d B C)^2 - (d A C)^2) / (2 * d A B * d B C)) ≤ 4/3 :=
by sorry

end max_tan_b_in_triangle_l4093_409362


namespace bakers_cakes_l4093_409373

theorem bakers_cakes (initial_cakes : ℕ) : 
  initial_cakes - 105 + 170 = 186 → initial_cakes = 121 := by
  sorry

end bakers_cakes_l4093_409373


namespace town_population_l4093_409354

theorem town_population (pet_owners_percentage : Real) 
  (dog_owners_fraction : Real) (cat_owners : ℕ) :
  pet_owners_percentage = 0.6 →
  dog_owners_fraction = 0.5 →
  cat_owners = 30 →
  (cat_owners : Real) / (1 - dog_owners_fraction) / pet_owners_percentage = 100 :=
by sorry

end town_population_l4093_409354


namespace bridge_length_l4093_409394

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 225 := by
  sorry

end bridge_length_l4093_409394


namespace multiplication_equation_solution_l4093_409342

theorem multiplication_equation_solution : 
  ∃ x : ℕ, 18396 * x = 183868020 ∧ x = 9990 := by
  sorry

end multiplication_equation_solution_l4093_409342


namespace prob_less_than_130_l4093_409323

-- Define the normal distribution
def normal_distribution (μ σ : ℝ) : ℝ → ℝ := sorry

-- Define the cumulative distribution function (CDF) for the normal distribution
def normal_cdf (μ σ : ℝ) : ℝ → ℝ := sorry

-- Define the probability of a score being within μ ± kσ
def prob_within_k_sigma (k : ℝ) : ℝ := sorry

-- Theorem to prove
theorem prob_less_than_130 :
  let μ : ℝ := 110
  let σ : ℝ := 20
  ∃ ε > 0, |normal_cdf μ σ 130 - 0.97725| < ε :=
sorry

end prob_less_than_130_l4093_409323


namespace sum_of_coordinates_A_l4093_409312

/-- Given three points A, B, and C in the plane satisfying certain conditions,
    prove that the sum of the coordinates of A is 16. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (C.2 - A.2) / (B.2 - A.2) = 1/3 →
  (B.1 - C.1) / (B.1 - A.1) = 2/3 →
  (B.2 - C.2) / (B.2 - A.2) = 2/3 →
  B = (2, 5) →
  C = (5, 8) →
  A.1 + A.2 = 16 := by
sorry


end sum_of_coordinates_A_l4093_409312


namespace system_two_solutions_l4093_409300

open Real

-- Define the system of equations
def equation1 (a x y : ℝ) : Prop :=
  arcsin ((a + y) / 2) = arcsin ((x + 3) / 3)

def equation2 (b x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x + 6*y = b

-- Define the condition for exactly two solutions
def hasTwoSolutions (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    equation1 a x₁ y₁ ∧ equation1 a x₂ y₂ ∧
    equation2 b x₁ y₁ ∧ equation2 b x₂ y₂ ∧
    ∀ (x y : ℝ), equation1 a x y ∧ equation2 b x y → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)

-- Theorem statement
theorem system_two_solutions (a : ℝ) :
  (∃ b, hasTwoSolutions a b) ↔ -7/2 < a ∧ a < 19/2 :=
sorry

end system_two_solutions_l4093_409300


namespace sqrt_sum_equal_l4093_409305

theorem sqrt_sum_equal : Real.sqrt 12 + Real.sqrt 27 = 5 * Real.sqrt 3 := by
  sorry

end sqrt_sum_equal_l4093_409305


namespace gcd_459_357_f_neg_four_l4093_409307

-- Part 1: GCD of 459 and 357
theorem gcd_459_357 : Int.gcd 459 357 = 51 := by sorry

-- Part 2: Polynomial evaluation
def f (x : ℤ) : ℤ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

theorem f_neg_four : f (-4) = 3392 := by sorry

end gcd_459_357_f_neg_four_l4093_409307


namespace greatest_three_digit_divisible_by_4_and_9_l4093_409380

theorem greatest_three_digit_divisible_by_4_and_9 : 
  ∃ n : ℕ, n = 972 ∧ 
  n < 1000 ∧ 
  n ≥ 100 ∧
  n % 4 = 0 ∧ 
  n % 9 = 0 ∧
  ∀ m : ℕ, m < 1000 → m ≥ 100 → m % 4 = 0 → m % 9 = 0 → m ≤ n :=
by sorry

end greatest_three_digit_divisible_by_4_and_9_l4093_409380


namespace percent_of_percent_l4093_409398

theorem percent_of_percent (a b : ℝ) (ha : a = 20) (hb : b = 25) :
  (a / 100) * (b / 100) * 100 = 5 := by
  sorry

end percent_of_percent_l4093_409398


namespace intersection_implies_a_greater_than_one_l4093_409304

-- Define the sets A and B
def A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a}

def B (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = b^p.1 + 1}

-- State the theorem
theorem intersection_implies_a_greater_than_one 
  (a b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) :
  (A a ∩ B b).Nonempty → a > 1 := by
  sorry


end intersection_implies_a_greater_than_one_l4093_409304


namespace m_range_l4093_409387

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem m_range (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 2) →
  m ∈ Set.Icc 1 2 :=
by sorry

end m_range_l4093_409387


namespace dealership_anticipation_l4093_409302

/-- Given a ratio of SUVs to trucks and an expected number of SUVs,
    calculate the anticipated number of trucks -/
def anticipatedTrucks (suvRatio truckRatio expectedSUVs : ℕ) : ℕ :=
  (expectedSUVs * truckRatio) / suvRatio

/-- Theorem: Given the ratio of SUVs to trucks is 3:5,
    if 45 SUVs are expected to be sold,
    then 75 trucks are anticipated to be sold -/
theorem dealership_anticipation :
  anticipatedTrucks 3 5 45 = 75 := by
  sorry

end dealership_anticipation_l4093_409302


namespace domain_of_sqrt_2cos_minus_1_l4093_409396

/-- The domain of f(x) = √(2cos(x) - 1) -/
theorem domain_of_sqrt_2cos_minus_1 (x : ℝ) : 
  (∃ f : ℝ → ℝ, f x = Real.sqrt (2 * Real.cos x - 1)) ↔ 
  (∃ k : ℤ, 2 * k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 3) :=
sorry

end domain_of_sqrt_2cos_minus_1_l4093_409396


namespace line_intersections_l4093_409306

/-- The line equation y = -2x + 4 -/
def line_equation (x y : ℝ) : Prop := y = -2 * x + 4

/-- The point (x, y) lies on the x-axis -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The point (x, y) lies on the y-axis -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The line y = -2x + 4 intersects the x-axis at (2, 0) and the y-axis at (0, 4) -/
theorem line_intersections :
  (∃ (x y : ℝ), line_equation x y ∧ on_x_axis x y ∧ x = 2 ∧ y = 0) ∧
  (∃ (x y : ℝ), line_equation x y ∧ on_y_axis x y ∧ x = 0 ∧ y = 4) :=
sorry

end line_intersections_l4093_409306


namespace arun_weight_average_l4093_409330

theorem arun_weight_average (w : ℝ) 
  (h1 : 64 < w ∧ w < 72)
  (h2 : 60 < w ∧ w < 70)
  (h3 : w ≤ 67) : 
  (64 + 67) / 2 = 65.5 := by sorry

end arun_weight_average_l4093_409330


namespace sufficient_not_necessary_condition_l4093_409364

/-- A quadratic equation x^2 + x + a = 0 has one positive root and one negative root -/
def has_one_positive_one_negative_root (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + x + a = 0 ∧ y^2 + y + a = 0

/-- The condition a < -1 is sufficient but not necessary for x^2 + x + a = 0 
    to have one positive and one negative root -/
theorem sufficient_not_necessary_condition :
  (∀ a : ℝ, a < -1 → has_one_positive_one_negative_root a) ∧
  (∃ a : ℝ, -1 ≤ a ∧ a < 0 ∧ has_one_positive_one_negative_root a) :=
sorry

end sufficient_not_necessary_condition_l4093_409364


namespace constant_b_value_l4093_409356

theorem constant_b_value (a b c : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 4 * x + 5/2) * (a * x^2 + b * x + c) = 
    6 * x^4 - 17 * x^3 + 11 * x^2 - 7/2 * x + 5/3) →
  b = -3 := by
sorry

end constant_b_value_l4093_409356


namespace intersection_of_M_and_N_l4093_409386

def M : Set ℝ := {x | x - 1 = 0}
def N : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem intersection_of_M_and_N : M ∩ N = {1} := by sorry

end intersection_of_M_and_N_l4093_409386


namespace touching_circle_radius_l4093_409360

/-- A circle touching two semicircles and a line segment --/
structure TouchingCircle where
  /-- Radius of the larger semicircle -/
  R : ℝ
  /-- Radius of the smaller semicircle -/
  r : ℝ
  /-- Radius of the touching circle -/
  x : ℝ
  /-- The smaller semicircle's diameter is half of the larger one -/
  h1 : r = R / 2
  /-- The touching circle is tangent to both semicircles and the line segment -/
  h2 : x > 0 ∧ x < r

/-- The radius of the touching circle is 8 when the larger semicircle has diameter 36 -/
theorem touching_circle_radius (c : TouchingCircle) (h : c.R = 18) : c.x = 8 := by
  sorry

end touching_circle_radius_l4093_409360


namespace sin_cos_difference_65_35_l4093_409384

theorem sin_cos_difference_65_35 :
  Real.sin (65 * π / 180) * Real.cos (35 * π / 180) - 
  Real.cos (65 * π / 180) * Real.sin (35 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_difference_65_35_l4093_409384


namespace sun_division_problem_l4093_409340

/-- Proves that the total amount is 156 rupees given the conditions of the problem -/
theorem sun_division_problem (x y z : ℝ) : 
  (∀ (r : ℝ), r > 0 → y = 0.45 * r ∧ z = 0.5 * r) →  -- For each rupee x gets, y gets 45 paisa and z gets 50 paisa
  y = 36 →  -- y's share is Rs. 36
  x + y + z = 156 := by  -- The total amount is Rs. 156
sorry

end sun_division_problem_l4093_409340


namespace arithmetic_sequence_terms_l4093_409352

theorem arithmetic_sequence_terms (a₁ l d : ℤ) (h₁ : a₁ = 165) (h₂ : l = 30) (h₃ : d = -5) :
  ∃ n : ℕ, n = 28 ∧ l = a₁ + (n - 1) * d :=
sorry

end arithmetic_sequence_terms_l4093_409352


namespace pie_remainder_l4093_409336

theorem pie_remainder (carlos_portion jessica_portion : Real) : 
  carlos_portion = 0.6 →
  jessica_portion = 0.25 * (1 - carlos_portion) →
  1 - carlos_portion - jessica_portion = 0.3 := by
sorry

end pie_remainder_l4093_409336


namespace monkey_climb_time_25m_l4093_409327

/-- Represents the time taken for a monkey to climb a greased pole -/
def monkey_climb_time (pole_height : ℕ) (ascend_rate : ℕ) (slip_rate : ℕ) : ℕ :=
  let full_cycles := (pole_height - ascend_rate) / (ascend_rate - slip_rate)
  full_cycles * 2 + 1

/-- Theorem stating that it takes 45 minutes for the monkey to climb the pole -/
theorem monkey_climb_time_25m :
  monkey_climb_time 25 3 2 = 45 := by sorry

end monkey_climb_time_25m_l4093_409327


namespace marble_probability_difference_l4093_409376

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1101

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1101

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def P_s : ℚ := (Nat.choose red_marbles 2 + Nat.choose black_marbles 2) / Nat.choose total_marbles 2

/-- The probability of drawing two marbles of different colors -/
def P_d : ℚ := (red_marbles * black_marbles) / Nat.choose total_marbles 2

/-- The theorem stating that the absolute difference between P_s and P_d is 1/2201 -/
theorem marble_probability_difference : |P_s - P_d| = 1 / 2201 := by sorry

end marble_probability_difference_l4093_409376


namespace floor_equality_iff_in_interval_l4093_409367

theorem floor_equality_iff_in_interval (x : ℝ) : 
  ⌊⌊3*x⌋ - 1/3⌋ = ⌊x + 3⌋ ↔ 5/3 ≤ x ∧ x < 3 :=
sorry

end floor_equality_iff_in_interval_l4093_409367


namespace south_american_stamps_cost_l4093_409333

/-- Represents a country in Maria's stamp collection. -/
inductive Country
| Brazil
| Peru
| France
| Spain

/-- Represents a decade in which stamps were issued. -/
inductive Decade
| Fifties
| Sixties
| Nineties

/-- The cost of a stamp in cents for a given country. -/
def stampCost (c : Country) : ℕ :=
  match c with
  | Country.Brazil => 7
  | Country.Peru => 5
  | Country.France => 7
  | Country.Spain => 6

/-- Whether a country is in South America. -/
def isSouthAmerican (c : Country) : Bool :=
  match c with
  | Country.Brazil => true
  | Country.Peru => true
  | _ => false

/-- The number of stamps Maria has for a given country and decade. -/
def stampCount (c : Country) (d : Decade) : ℕ :=
  match c, d with
  | Country.Brazil, Decade.Fifties => 6
  | Country.Brazil, Decade.Sixties => 9
  | Country.Peru, Decade.Fifties => 8
  | Country.Peru, Decade.Sixties => 6
  | _, _ => 0

/-- The total cost of stamps for a given country and decade, in cents. -/
def decadeCost (c : Country) (d : Decade) : ℕ :=
  stampCost c * stampCount c d

/-- The theorem stating the total cost of South American stamps issued before the 90s. -/
theorem south_american_stamps_cost :
  (decadeCost Country.Brazil Decade.Fifties +
   decadeCost Country.Brazil Decade.Sixties +
   decadeCost Country.Peru Decade.Fifties +
   decadeCost Country.Peru Decade.Sixties) = 175 := by
  sorry


end south_american_stamps_cost_l4093_409333


namespace min_value_a_plus_b_l4093_409349

theorem min_value_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.log x - a * Real.exp x - b + 1 ≤ 0) →
  ∃ m : ℝ, m = 0 ∧ (∀ a' b' : ℝ, (∀ x : ℝ, x > 0 → Real.log x - a' * Real.exp x - b' + 1 ≤ 0) → a' + b' ≥ m) :=
by sorry

end min_value_a_plus_b_l4093_409349


namespace kays_aerobics_time_l4093_409372

/-- Given Kay's weekly exercise routine, calculate the time spent on aerobics. -/
theorem kays_aerobics_time (total_time : ℕ) (aerobics_ratio : ℕ) (weight_ratio : ℕ) 
  (h1 : total_time = 250)
  (h2 : aerobics_ratio = 3)
  (h3 : weight_ratio = 2) :
  (aerobics_ratio * total_time) / (aerobics_ratio + weight_ratio) = 150 := by
  sorry

#check kays_aerobics_time

end kays_aerobics_time_l4093_409372


namespace tulips_to_add_l4093_409361

def tulip_to_daisy_ratio : ℚ := 3 / 4
def initial_daisies : ℕ := 32
def added_daisies : ℕ := 24

theorem tulips_to_add (tulips_added : ℕ) : 
  (tulip_to_daisy_ratio * (initial_daisies + added_daisies : ℚ)).num = 
  (tulip_to_daisy_ratio * initial_daisies).num + tulips_added → 
  tulips_added = 18 :=
by sorry

end tulips_to_add_l4093_409361


namespace inequality_equivalence_l4093_409353

def f (x : ℝ) : ℝ := 5 * x + 3

theorem inequality_equivalence (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x + 2| < b → |f x + 7| < a) ↔ b ≤ a / 5 := by
  sorry

end inequality_equivalence_l4093_409353


namespace algebraic_expression_value_l4093_409389

theorem algebraic_expression_value (x : ℝ) : 
  2 * x^2 + 3 * x + 7 = 8 → 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end algebraic_expression_value_l4093_409389


namespace unique_number_theorem_l4093_409331

/-- A function that checks if a number n can be expressed as 2a + xb 
    where a and b are positive integers --/
def isExpressible (n x : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ n = 2 * a + x * b

/-- The main theorem stating that 5 is the unique number satisfying the condition --/
theorem unique_number_theorem :
  ∃! (x : ℕ), x > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = 8 ∧ 
    (∀ n ∈ S, n < 15 ∧ isExpressible n x) ∧
    (∀ n < 15, isExpressible n x → n ∈ S)) ∧
  x = 5 := by
  sorry


end unique_number_theorem_l4093_409331


namespace george_first_half_correct_l4093_409324

def trivia_game (first_half_correct : ℕ) (second_half_correct : ℕ) (points_per_question : ℕ) (total_score : ℕ) : Prop :=
  first_half_correct * points_per_question + second_half_correct * points_per_question = total_score

theorem george_first_half_correct :
  ∃ (x : ℕ), trivia_game x 4 3 30 ∧ x = 6 := by sorry

end george_first_half_correct_l4093_409324


namespace intersecting_lines_k_value_l4093_409382

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersecting_lines_k_value 
  (x y : ℝ) -- The coordinates of the intersection point
  (h1 : y = 3 * x + 6) -- First line equation
  (h2 : y = -4 * x - 20) -- Second line equation
  (h3 : y = 2 * x + k) -- Third line equation
  : k = 16 / 7 := by
  sorry

end intersecting_lines_k_value_l4093_409382


namespace solve_food_bank_problem_l4093_409370

def food_bank_problem (first_week_donation : ℝ) (second_week_multiplier : ℝ) (remaining_food : ℝ) : Prop :=
  let total_donation := first_week_donation + (second_week_multiplier * first_week_donation)
  let food_given_out := total_donation - remaining_food
  let percentage_given_out := (food_given_out / total_donation) * 100
  percentage_given_out = 70

theorem solve_food_bank_problem :
  food_bank_problem 40 2 36 := by
  sorry

end solve_food_bank_problem_l4093_409370


namespace no_valid_n_for_ap_l4093_409365

theorem no_valid_n_for_ap : ¬∃ (n : ℕ), n > 1 ∧ 
  180 % n = 0 ∧ 
  ∃ (k : ℕ), k^2 = (180 / n : ℚ) - (3/2 : ℚ) * n + (3/2 : ℚ) := by
  sorry

end no_valid_n_for_ap_l4093_409365


namespace lines_perpendicular_iff_m_eq_one_l4093_409390

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line x - y = 0 -/
def slope1 : ℝ := 1

/-- The slope of the line x + my = 0 -/
def slope2 (m : ℝ) : ℝ := -m

/-- Theorem: The lines x - y = 0 and x + my = 0 are perpendicular if and only if m = 1 -/
theorem lines_perpendicular_iff_m_eq_one (m : ℝ) :
  perpendicular slope1 (slope2 m) ↔ m = 1 := by
  sorry

end lines_perpendicular_iff_m_eq_one_l4093_409390


namespace sixth_root_of_594823321_l4093_409309

theorem sixth_root_of_594823321 : (594823321 : ℝ) ^ (1/6 : ℝ) = 51 := by
  sorry

end sixth_root_of_594823321_l4093_409309


namespace probability_at_least_one_first_class_l4093_409355

theorem probability_at_least_one_first_class (total : Nat) (first_class : Nat) (second_class : Nat) (selected : Nat) :
  total = 6 →
  first_class = 4 →
  second_class = 2 →
  selected = 2 →
  (1 : ℚ) - (Nat.choose second_class selected : ℚ) / (Nat.choose total selected : ℚ) = 14 / 15 := by
  sorry

end probability_at_least_one_first_class_l4093_409355


namespace set_operations_l4093_409369

def U : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}
def A : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem set_operations :
  (A ∩ B = {x | 0 < x ∧ x ≤ 1}) ∧
  (B ∪ (U \ A) = {x | (-5 ≤ x ∧ x ≤ 1) ∨ (3 < x ∧ x ≤ 5)}) := by
  sorry

end set_operations_l4093_409369


namespace negation_of_implication_l4093_409301

theorem negation_of_implication (x : ℝ) :
  ¬(x > 2015 → x > 0) ↔ (x ≤ 2015 → x ≤ 0) :=
by sorry

end negation_of_implication_l4093_409301


namespace max_valid_coloring_size_l4093_409399

/-- A type representing the color of a square on the board -/
inductive Color
| Black
| White

/-- A function type representing a coloring of an n × n board -/
def BoardColoring (n : ℕ) := Fin n → Fin n → Color

/-- Predicate to check if a board coloring satisfies the condition -/
def ValidColoring (n : ℕ) (coloring : BoardColoring n) : Prop :=
  ∀ (r1 r2 c1 c2 : Fin n), r1 ≠ r2 → c1 ≠ c2 → 
    (coloring r1 c1 = coloring r1 c2 → coloring r2 c1 ≠ coloring r2 c2) ∧
    (coloring r1 c1 = coloring r2 c1 → coloring r1 c2 ≠ coloring r2 c2)

/-- Theorem stating that 4 is the maximum value of n for which a valid coloring exists -/
theorem max_valid_coloring_size :
  (∃ (coloring : BoardColoring 4), ValidColoring 4 coloring) ∧
  (∀ n : ℕ, n > 4 → ¬∃ (coloring : BoardColoring n), ValidColoring n coloring) :=
sorry

end max_valid_coloring_size_l4093_409399


namespace necessary_but_not_sufficient_l4093_409350

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x > 5 → x > 3) ∧ 
  (∃ x : ℝ, x > 3 ∧ ¬(x > 5)) :=
by sorry

end necessary_but_not_sufficient_l4093_409350


namespace smallest_T_value_l4093_409329

theorem smallest_T_value : ∃ (m : ℕ), 
  (∀ k : ℕ, k < m → 8 * k < 2400) ∧ 
  8 * m ≥ 2400 ∧
  9 * m - 2400 = 300 := by
  sorry

end smallest_T_value_l4093_409329


namespace bakers_remaining_cakes_l4093_409337

theorem bakers_remaining_cakes 
  (initial_cakes : ℝ) 
  (bought_cakes : ℝ) 
  (h1 : initial_cakes = 397.5) 
  (h2 : bought_cakes = 289) : 
  initial_cakes - bought_cakes = 108.5 := by
sorry

end bakers_remaining_cakes_l4093_409337


namespace crickets_found_later_l4093_409363

theorem crickets_found_later (initial : ℝ) (final : ℕ) : initial = 7.0 → final = 18 → final - initial = 11 := by
  sorry

end crickets_found_later_l4093_409363


namespace trays_needed_to_replace_ice_l4093_409395

def ice_cubes_in_glass : ℕ := 8
def ice_cubes_in_pitcher : ℕ := 2 * ice_cubes_in_glass
def spaces_per_tray : ℕ := 12

theorem trays_needed_to_replace_ice : 
  (ice_cubes_in_glass + ice_cubes_in_pitcher) / spaces_per_tray = 2 := by
  sorry

end trays_needed_to_replace_ice_l4093_409395


namespace closest_result_is_180_l4093_409343

theorem closest_result_is_180 (options : List ℝ := [160, 180, 190, 200, 240]) : 
  let result := (0.000345 * 7650000) / 15
  options.argmin (λ x => |x - result|) = some 180 := by
  sorry

end closest_result_is_180_l4093_409343


namespace vector_magnitude_problem_l4093_409338

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  a = (2, 1) →
  a • b = 10 →
  ‖a + b‖ = 5 * Real.sqrt 2 →
  ‖b‖ = 5 := by
  sorry

end vector_magnitude_problem_l4093_409338


namespace sin_squared_alpha_plus_pi_fourth_l4093_409368

theorem sin_squared_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/4)) 
  (h2 : Real.cos (2*α) = 4/5) : 
  Real.sin (α + π/4)^2 = 4/5 := by sorry

end sin_squared_alpha_plus_pi_fourth_l4093_409368


namespace negative_plus_square_not_always_positive_l4093_409316

theorem negative_plus_square_not_always_positive : 
  ∃ x : ℝ, x < 0 ∧ x + x^2 ≤ 0 := by
  sorry

end negative_plus_square_not_always_positive_l4093_409316


namespace sum4_is_27857_l4093_409320

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  sum3 : a + a*r + a*r^2 = 13
  sum5 : a + a*r + a*r^2 + a*r^3 + a*r^4 = 121

/-- The sum of the first 4 terms of the geometric sequence -/
def sum4 (seq : GeometricSequence) : ℝ :=
  seq.a + seq.a * seq.r + seq.a * seq.r^2 + seq.a * seq.r^3

/-- Theorem stating that the sum of the first 4 terms is 27.857 -/
theorem sum4_is_27857 (seq : GeometricSequence) : sum4 seq = 27.857 := by
  sorry

end sum4_is_27857_l4093_409320


namespace max_digit_sum_for_special_fraction_l4093_409392

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the decimal 0.abc where a, b, c are digits -/
def DecimalABC (a b c : Digit) : ℚ := (a.val * 100 + b.val * 10 + c.val : ℕ) / 1000

/-- The theorem statement -/
theorem max_digit_sum_for_special_fraction :
  ∃ (a b c : Digit) (y : ℕ+),
    DecimalABC a b c = (1 : ℚ) / y ∧
    y ≤ 12 ∧
    ∀ (a' b' c' : Digit) (y' : ℕ+),
      DecimalABC a' b' c' = (1 : ℚ) / y' →
      y' ≤ 12 →
      a.val + b.val + c.val ≥ a'.val + b'.val + c'.val ∧
      a.val + b.val + c.val = 8 :=
sorry

end max_digit_sum_for_special_fraction_l4093_409392


namespace regular_polygon_144_degrees_has_10_sides_l4093_409335

/-- A regular polygon with interior angles of 144 degrees has 10 sides -/
theorem regular_polygon_144_degrees_has_10_sides :
  ∀ n : ℕ,
  n > 2 →
  (n - 2 : ℝ) * 180 / n = 144 →
  n = 10 :=
by
  sorry

end regular_polygon_144_degrees_has_10_sides_l4093_409335


namespace certain_yellow_ball_pick_l4093_409311

theorem certain_yellow_ball_pick (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) (m : ℕ) : 
  total_balls = 8 →
  red_balls = 3 →
  yellow_balls = 5 →
  total_balls = red_balls + yellow_balls →
  m ≤ red_balls →
  yellow_balls = total_balls - m →
  m = 3 := by
  sorry

end certain_yellow_ball_pick_l4093_409311


namespace successive_discounts_l4093_409345

theorem successive_discounts (P d1 d2 : ℝ) (h1 : 0 ≤ d1 ∧ d1 < 1) (h2 : 0 ≤ d2 ∧ d2 < 1) :
  let final_price := P * (1 - d1) * (1 - d2)
  let percentage := (final_price / P) * 100
  percentage = (1 - d1) * (1 - d2) * 100 :=
by sorry

end successive_discounts_l4093_409345


namespace negative_three_squared_l4093_409391

theorem negative_three_squared : (-3 : ℤ)^2 = 9 := by
  sorry

end negative_three_squared_l4093_409391


namespace new_person_weight_is_68_l4093_409314

/-- The weight of the new person given the conditions of the problem -/
def new_person_weight (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 68 kg -/
theorem new_person_weight_is_68 :
  new_person_weight 6 3.5 47 = 68 := by
  sorry

end new_person_weight_is_68_l4093_409314


namespace complement_of_A_l4093_409378

def U : Set ℕ := {x | 0 ≤ x ∧ x < 10}

def A : Set ℕ := {2, 4, 6, 8}

theorem complement_of_A : U \ A = {1, 3, 5, 7, 9} := by sorry

end complement_of_A_l4093_409378


namespace fair_rides_l4093_409322

theorem fair_rides (total_tickets : ℕ) (spent_tickets : ℕ) (ride_cost : ℕ) : 
  total_tickets = 79 → spent_tickets = 23 → ride_cost = 7 → 
  (total_tickets - spent_tickets) / ride_cost = 8 := by
  sorry

end fair_rides_l4093_409322


namespace nonagon_arithmetic_mean_property_l4093_409310

/-- Represents a vertex of the nonagon with its assigned number -/
structure Vertex where
  index : Fin 9
  value : Nat

/-- Checks if three vertices form an equilateral triangle in a regular nonagon -/
def isEquilateralTriangle (v1 v2 v3 : Vertex) : Prop :=
  (v2.index - v1.index) % 3 = 0 ∧ (v3.index - v2.index) % 3 = 0 ∧ (v1.index - v3.index) % 3 = 0

/-- Checks if one number is the arithmetic mean of the other two -/
def isArithmeticMean (a b c : Nat) : Prop :=
  2 * b = a + c

/-- The arrangement of numbers on the nonagon -/
def arrangement : List Vertex :=
  List.map (fun i => ⟨i, 2016 + i⟩) (List.range 9)

/-- The main theorem to prove -/
theorem nonagon_arithmetic_mean_property :
  ∀ v1 v2 v3 : Vertex,
    v1 ∈ arrangement →
    v2 ∈ arrangement →
    v3 ∈ arrangement →
    isEquilateralTriangle v1 v2 v3 →
    isArithmeticMean v1.value v2.value v3.value ∨
    isArithmeticMean v2.value v3.value v1.value ∨
    isArithmeticMean v3.value v1.value v2.value :=
  sorry

end nonagon_arithmetic_mean_property_l4093_409310


namespace evaluate_complex_expression_l4093_409377

theorem evaluate_complex_expression :
  ∃ (x : ℝ), x > 0 ∧ x^2 = (297 - 99*Real.sqrt 5 + 108*Real.sqrt 6 - 36*Real.sqrt 30) / 64 ∧
  x = (3*(Real.sqrt 3 + Real.sqrt 8)) / (4*Real.sqrt (3 + Real.sqrt 5)) :=
by sorry

end evaluate_complex_expression_l4093_409377


namespace no_positive_lower_bound_l4093_409303

/-- The number of positive integers not containing the digit 9 that are less than or equal to n -/
def f (n : ℕ+) : ℕ := sorry

/-- For any positive real number p, there exists a positive integer n such that f(n)/n < p -/
theorem no_positive_lower_bound :
  ∀ p : ℝ, p > 0 → ∃ n : ℕ+, (f n : ℝ) / n < p :=
sorry

end no_positive_lower_bound_l4093_409303


namespace marts_income_percentage_l4093_409385

theorem marts_income_percentage (juan tim mart : ℝ) 
  (h1 : tim = juan * (1 - 0.4))
  (h2 : mart = tim * (1 + 0.3)) :
  mart / juan = 0.78 := by
sorry

end marts_income_percentage_l4093_409385


namespace garden_playground_area_equality_l4093_409383

theorem garden_playground_area_equality (garden_width garden_length playground_width : ℝ) :
  garden_width = 8 →
  2 * (garden_width + garden_length) = 64 →
  garden_width * garden_length = 16 * playground_width :=
by
  sorry

end garden_playground_area_equality_l4093_409383


namespace max_value_of_expression_l4093_409348

theorem max_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * b * c + a + c = b) :
  2 / (a^2 + 1) - 2 / (b^2 + 1) + 3 / (c^2 + 1) ≤ 26 / 5 := by
  sorry

end max_value_of_expression_l4093_409348


namespace cylinder_dihedral_angle_l4093_409328

-- Define the cylinder and its properties
structure Cylinder where
  A : Point
  A₁ : Point
  B : Point
  B₁ : Point
  C : Point
  α : Real  -- dihedral angle
  β : Real  -- ∠CAB
  γ : Real  -- ∠CA₁B

-- Define the theorem
theorem cylinder_dihedral_angle (cyl : Cylinder) :
  cyl.α = Real.arcsin (Real.cos cyl.β / Real.cos cyl.γ) := by
  sorry

end cylinder_dihedral_angle_l4093_409328


namespace no_line_bisected_by_P_intersects_hyperbola_l4093_409347

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The hyperbola equation -/
def isOnHyperbola (p : Point) : Prop :=
  p.x^2 / 9 - p.y^2 / 4 = 1

/-- Check if a point is on a line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- Check if a point bisects a line segment -/
def isMidpoint (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  p.x = (p1.x + p2.x) / 2 ∧ p.y = (p1.y + p2.y) / 2

/-- The main theorem -/
theorem no_line_bisected_by_P_intersects_hyperbola :
  ¬ ∃ (l : Line) (p1 p2 : Point),
    p1 ≠ p2 ∧
    isOnHyperbola p1 ∧
    isOnHyperbola p2 ∧
    isOnLine p1 l ∧
    isOnLine p2 l ∧
    isMidpoint ⟨2, 1⟩ p1 p2 :=
  sorry

end no_line_bisected_by_P_intersects_hyperbola_l4093_409347


namespace pure_imaginary_condition_l4093_409397

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ (k : ℝ), m^2 + m - 2 + (m^2 - 1) * I = k * I) → m = -2 :=
by
  sorry

end pure_imaginary_condition_l4093_409397


namespace intersection_of_A_and_B_l4093_409366

def A : Set ℝ := {x : ℝ | x^2 ≤ 1}
def B : Set ℝ := {x : ℝ | x ≠ 0 ∧ 2/x ≥ 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l4093_409366


namespace sum_of_digits_special_product_l4093_409332

/-- Sum of digits function -/
def sum_of_digits (x : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem sum_of_digits_special_product (m n : ℕ) (d : ℕ) :
  m > 0 → n > 0 → d > 0 → d ≤ n → d = (Nat.digits 10 m).length →
  sum_of_digits ((10^n - 1) * m) = 9 * n := by sorry

end sum_of_digits_special_product_l4093_409332


namespace unbounded_digit_sum_ratio_l4093_409313

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- State the theorem
theorem unbounded_digit_sum_ratio :
  ∀ c : ℝ, c > 0 → ∃ n : ℕ, (sum_of_digits n : ℝ) / (sum_of_digits (n^2)) > c :=
sorry

end unbounded_digit_sum_ratio_l4093_409313


namespace greatest_sum_consecutive_integers_l4093_409319

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m * (m + 1) < 500 → n + (n + 1) ≥ m + (m + 1)) → 
  n + (n + 1) = 43 :=
sorry

end greatest_sum_consecutive_integers_l4093_409319


namespace converse_of_square_inequality_l4093_409315

theorem converse_of_square_inequality :
  (∀ a b : ℝ, a > b → a^2 > b^2) →
  (∀ a b : ℝ, a^2 > b^2 → a > b) :=
by sorry

end converse_of_square_inequality_l4093_409315


namespace not_divisible_by_59_l4093_409374

theorem not_divisible_by_59 (x y : ℕ) 
  (hx : ¬ 59 ∣ x) 
  (hy : ¬ 59 ∣ y) 
  (h_sum : 59 ∣ (3 * x + 28 * y)) : 
  ¬ 59 ∣ (5 * x + 16 * y) := by
  sorry

end not_divisible_by_59_l4093_409374
