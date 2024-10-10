import Mathlib

namespace min_value_of_expression_l2661_266107

theorem min_value_of_expression (x : ℝ) :
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2027 ≥ 2026 ∧
  ∃ y : ℝ, (y + 1) * (y + 2) * (y + 3) * (y + 4) + 2027 = 2026 :=
by sorry

end min_value_of_expression_l2661_266107


namespace line_intercepts_sum_l2661_266172

theorem line_intercepts_sum (c : ℚ) : 
  (∃ x y : ℚ, 4 * x + 7 * y + 3 * c = 0 ∧ x + y = 11) → c = -308/33 := by
  sorry

end line_intercepts_sum_l2661_266172


namespace tan_alpha_plus_20_l2661_266130

theorem tan_alpha_plus_20 (α : ℝ) (h : Real.tan (α + 80 * π / 180) = 4 * Real.sin (420 * π / 180)) :
  Real.tan (α + 20 * π / 180) = Real.sqrt 3 / 7 := by
  sorry

end tan_alpha_plus_20_l2661_266130


namespace no_quadratic_transform_l2661_266115

/-- A polynomial function of degree 2 or less -/
def QuadraticPolynomial (a b c : ℚ) : ℚ → ℚ := λ x => a * x^2 + b * x + c

/-- Theorem stating that no quadratic polynomial can transform (1,4,7) to (1,10,7) -/
theorem no_quadratic_transform :
  ¬ ∃ (a b c : ℚ), 
    (QuadraticPolynomial a b c 1 = 1) ∧ 
    (QuadraticPolynomial a b c 4 = 10) ∧ 
    (QuadraticPolynomial a b c 7 = 7) := by
  sorry

end no_quadratic_transform_l2661_266115


namespace chris_balls_l2661_266164

/-- The number of golf balls in a dozen -/
def balls_per_dozen : ℕ := 12

/-- The number of dozens Dan buys -/
def dan_dozens : ℕ := 5

/-- The number of dozens Gus buys -/
def gus_dozens : ℕ := 2

/-- The total number of golf balls purchased -/
def total_balls : ℕ := 132

/-- Theorem: Chris buys 48 golf balls -/
theorem chris_balls : 
  total_balls - (dan_dozens * balls_per_dozen + gus_dozens * balls_per_dozen) = 48 := by
  sorry

end chris_balls_l2661_266164


namespace classroom_size_theorem_l2661_266126

/-- Represents the number of students in a classroom -/
def classroom_size (boys : ℕ) (girls : ℕ) : ℕ := boys + girls

/-- Represents the ratio of boys to girls -/
def ratio_boys_girls (boys : ℕ) (girls : ℕ) : Prop := 3 * girls = 5 * boys

theorem classroom_size_theorem (boys girls : ℕ) :
  ratio_boys_girls boys girls →
  girls = boys + 4 →
  classroom_size boys girls = 16 := by
sorry

end classroom_size_theorem_l2661_266126


namespace cubic_root_cubes_l2661_266148

-- Define the polynomials h(x) and p(x)
def h (x : ℝ) : ℝ := x^3 - x^2 - 4*x + 4
def p (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem cubic_root_cubes (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ h x = 0 ∧ h y = 0 ∧ h z = 0) →
  (∀ s : ℝ, h s = 0 → p a b c (s^3) = 0) →
  a = 12 ∧ b = -13 ∧ c = -64 :=
sorry

end cubic_root_cubes_l2661_266148


namespace irrationality_of_sqrt_five_l2661_266197

theorem irrationality_of_sqrt_five :
  ¬ (∃ (q : ℚ), q * q = 5) ∧
  (∃ (a : ℚ), a * a = 4) ∧
  (∃ (b : ℚ), b * b = 9) ∧
  (∃ (c : ℚ), c * c = 16) :=
sorry

end irrationality_of_sqrt_five_l2661_266197


namespace smallest_common_multiple_proof_l2661_266151

/-- The smallest number divisible by 3, 15, and 9 -/
def smallest_common_multiple : ℕ := 45

/-- Gabe's group size -/
def gabe_group : ℕ := 3

/-- Steven's group size -/
def steven_group : ℕ := 15

/-- Maya's group size -/
def maya_group : ℕ := 9

theorem smallest_common_multiple_proof :
  (smallest_common_multiple % gabe_group = 0) ∧
  (smallest_common_multiple % steven_group = 0) ∧
  (smallest_common_multiple % maya_group = 0) ∧
  (∀ n : ℕ, n < smallest_common_multiple →
    ¬((n % gabe_group = 0) ∧ (n % steven_group = 0) ∧ (n % maya_group = 0))) :=
by sorry

end smallest_common_multiple_proof_l2661_266151


namespace distance_to_school_l2661_266174

/-- The distance to school given the travel conditions -/
theorem distance_to_school (total_time : ℝ) (speed_to_school : ℝ) (speed_from_school : ℝ) 
  (h1 : total_time = 1)
  (h2 : speed_to_school = 5)
  (h3 : speed_from_school = 21) :
  ∃ d : ℝ, d = 105 / 26 ∧ d / speed_to_school + d / speed_from_school = total_time := by
  sorry

end distance_to_school_l2661_266174


namespace max_value_of_expression_l2661_266158

theorem max_value_of_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_condition : x + y + z = 1) :
  x + y^2 + z^3 ≤ 1 ∧ ∃ (x' y' z' : ℝ), x' + y'^2 + z'^3 = 1 ∧ 
    x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 1 := by
  sorry

end max_value_of_expression_l2661_266158


namespace businessmen_drink_count_l2661_266170

theorem businessmen_drink_count (total : ℕ) (coffee : ℕ) (tea : ℕ) (coffee_and_tea : ℕ) 
  (juice : ℕ) (juice_and_tea_not_coffee : ℕ) 
  (h1 : total = 35)
  (h2 : coffee = 18)
  (h3 : tea = 15)
  (h4 : coffee_and_tea = 7)
  (h5 : juice = 6)
  (h6 : juice_and_tea_not_coffee = 3) : 
  total - ((coffee + tea - coffee_and_tea) + (juice - juice_and_tea_not_coffee)) = 6 := by
  sorry

end businessmen_drink_count_l2661_266170


namespace bill_calculation_l2661_266198

def original_bill : ℝ := 500
def late_charge_rate : ℝ := 0.02

def final_bill : ℝ :=
  original_bill * (1 + late_charge_rate) * (1 + late_charge_rate) * (1 + late_charge_rate)

theorem bill_calculation :
  final_bill = 530.604 := by sorry

end bill_calculation_l2661_266198


namespace population_ratio_l2661_266110

/-- Represents the population of a city -/
structure CityPopulation where
  value : ℕ

/-- The relationship between populations of different cities -/
structure PopulationRelationship where
  cityA : CityPopulation
  cityB : CityPopulation
  cityC : CityPopulation
  cityD : CityPopulation
  cityE : CityPopulation
  cityF : CityPopulation
  A_to_B : cityA.value = 5 * cityB.value
  B_to_C : cityB.value = 3 * cityC.value
  C_to_D : cityC.value = 8 * cityD.value
  D_to_E : cityD.value = 2 * cityE.value
  E_to_F : cityE.value = 6 * cityF.value

/-- Theorem stating the ratio of population of City A to City F -/
theorem population_ratio (r : PopulationRelationship) : 
  r.cityA.value = 1440 * r.cityF.value := by
  sorry

end population_ratio_l2661_266110


namespace square_ends_with_three_identical_nonzero_digits_l2661_266155

theorem square_ends_with_three_identical_nonzero_digits : 
  ∃ n : ℤ, ∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧ n^2 % 1000 = d * 100 + d * 10 + d :=
sorry

end square_ends_with_three_identical_nonzero_digits_l2661_266155


namespace hexagon_area_ratio_l2661_266156

-- Define the regular hexagon
def RegularHexagon (a : ℝ) : Set (ℝ × ℝ) := sorry

-- Define points on the sides of the hexagon
def PointOnSide (hexagon : Set (ℝ × ℝ)) (side : Set (ℝ × ℝ)) : (ℝ × ℝ) := sorry

-- Define parallel lines with specific spacing ratio
def ParallelLinesWithRatio (l1 l2 l3 l4 : Set (ℝ × ℝ)) (ratio : ℝ × ℝ × ℝ) : Prop := sorry

-- Define area of a polygon
def AreaOfPolygon (polygon : Set (ℝ × ℝ)) : ℝ := sorry

theorem hexagon_area_ratio 
  (a : ℝ) 
  (ABCDEF : Set (ℝ × ℝ))
  (G H I J : ℝ × ℝ)
  (BC CD EF FA : Set (ℝ × ℝ)) :
  ABCDEF = RegularHexagon a →
  G = PointOnSide ABCDEF BC →
  H = PointOnSide ABCDEF CD →
  I = PointOnSide ABCDEF EF →
  J = PointOnSide ABCDEF FA →
  ParallelLinesWithRatio AB GJ IH ED (1, 2, 1) →
  (AreaOfPolygon {A, G, I, H, J, F} / AreaOfPolygon ABCDEF) = 2/3 := by
  sorry

end hexagon_area_ratio_l2661_266156


namespace jenna_photo_groups_l2661_266191

theorem jenna_photo_groups (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → Nat.choose n k = 35 := by
  sorry

end jenna_photo_groups_l2661_266191


namespace trigonometric_identities_l2661_266150

theorem trigonometric_identities :
  (∀ n : ℤ,
    (Real.sin (4 * Real.pi / 3) * Real.cos (25 * Real.pi / 6) * Real.tan (5 * Real.pi / 4) = -3/4) ∧
    (Real.sin ((2 * n + 1) * Real.pi - 2 * Real.pi / 3) = Real.sqrt 3 / 2)) :=
by sorry

end trigonometric_identities_l2661_266150


namespace zeros_when_b_neg_one_inequality_condition_max_value_on_interval_l2661_266193

-- Define the function f
def f (a b x : ℝ) : ℝ := x * |x - a| + b * x

-- Theorem 1
theorem zeros_when_b_neg_one (a : ℝ) :
  (∃! (z₁ z₂ : ℝ), z₁ ≠ z₂ ∧ f a (-1) z₁ = 0 ∧ f a (-1) z₂ = 0) ↔ (a = 1 ∨ a = -1) :=
sorry

-- Theorem 2
theorem inequality_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f a 1 x / x ≤ 2 * Real.sqrt (x + 1)) ↔ 
  (0 ≤ a ∧ a ≤ 2 * Real.sqrt 2) :=
sorry

-- Define the piecewise function g
noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ 4 * Real.sqrt 3 - 5 then 6 - 2*a
  else if a < 3 then (a + 1)^2 / 4
  else 2*a - 2

-- Theorem 3
theorem max_value_on_interval (a : ℝ) (h : a > 0) :
  (∃ (m : ℝ), ∀ x ∈ Set.Icc 0 2, f a 1 x ≤ m ∧ ∃ y ∈ Set.Icc 0 2, f a 1 y = m) ∧
  (∀ (m : ℝ), (∀ x ∈ Set.Icc 0 2, f a 1 x ≤ m ∧ ∃ y ∈ Set.Icc 0 2, f a 1 y = m) → m = g a) :=
sorry

end zeros_when_b_neg_one_inequality_condition_max_value_on_interval_l2661_266193


namespace power_of_128_over_7_l2661_266133

theorem power_of_128_over_7 : (128 : ℝ) ^ (3/7) = 8 := by sorry

end power_of_128_over_7_l2661_266133


namespace correct_swap_l2661_266135

-- Define the initial values
def a : Int := 2
def b : Int := -6

-- Define the swap operation
def swap (x y : Int) : (Int × Int) :=
  let c := x
  let new_x := y
  let new_y := c
  (new_x, new_y)

-- Theorem statement
theorem correct_swap :
  swap a b = (-6, 2) := by
  sorry

end correct_swap_l2661_266135


namespace tangent_line_condition_l2661_266146

-- Define the condition for a line being tangent to a circle
def is_tangent (k : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + 2 ∧ x^2 + y^2 = 1 ∧
  ∀ x' y' : ℝ, y' = k * x' + 2 → x'^2 + y'^2 ≥ 1

-- State the theorem
theorem tangent_line_condition :
  (∀ k : ℝ, ¬(k = Real.sqrt 3) → ¬(is_tangent k)) ∧
  ¬(∀ k : ℝ, ¬(is_tangent k) → ¬(k = Real.sqrt 3)) :=
by sorry

end tangent_line_condition_l2661_266146


namespace stratified_sample_size_l2661_266142

/-- Given a stratified sample with ratio 2:3:5 for products A:B:C, 
    prove that if 16 type A products are sampled, the total sample size is 80 -/
theorem stratified_sample_size 
  (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ) 
  (h_ratio : ratio_A = 2 ∧ ratio_B = 3 ∧ ratio_C = 5) 
  (sample_A : ℕ) (h_sample_A : sample_A = 16) : 
  let total_ratio := ratio_A + ratio_B + ratio_C
  let sample_size := (sample_A * total_ratio) / ratio_A
  sample_size = 80 := by sorry

end stratified_sample_size_l2661_266142


namespace arithmetic_sequence_remainder_l2661_266100

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The number of terms in the sequence -/
def sequence_length (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

theorem arithmetic_sequence_remainder (a₁ d aₙ : ℕ) (h₁ : a₁ = 2) (h₂ : d = 6) (h₃ : aₙ = 278) :
  (arithmetic_sum a₁ d (sequence_length a₁ d aₙ)) % 6 = 4 := by
  sorry

end arithmetic_sequence_remainder_l2661_266100


namespace work_fraction_is_half_l2661_266137

/-- Represents the highway construction project -/
structure HighwayProject where
  initialWorkers : ℕ
  totalLength : ℝ
  initialDuration : ℕ
  initialDailyHours : ℕ
  completedDays : ℕ
  additionalWorkers : ℕ
  newDailyHours : ℕ

/-- Calculates the total man-hours for a given number of workers, days, and daily hours -/
def manHours (workers : ℕ) (days : ℕ) (hours : ℕ) : ℕ :=
  workers * days * hours

/-- Theorem stating that the fraction of work completed is 1/2 -/
theorem work_fraction_is_half (project : HighwayProject) 
  (h1 : project.initialWorkers = 100)
  (h2 : project.totalLength = 2)
  (h3 : project.initialDuration = 50)
  (h4 : project.initialDailyHours = 8)
  (h5 : project.completedDays = 25)
  (h6 : project.additionalWorkers = 60)
  (h7 : project.newDailyHours = 10)
  (h8 : manHours (project.initialWorkers + project.additionalWorkers) 
              (project.initialDuration - project.completedDays) 
              project.newDailyHours = 
        manHours project.initialWorkers project.initialDuration project.initialDailyHours) :
  (manHours project.initialWorkers project.completedDays project.initialDailyHours : ℝ) / 
  (manHours project.initialWorkers project.initialDuration project.initialDailyHours) = 1/2 := by
  sorry

end work_fraction_is_half_l2661_266137


namespace angle_in_quadrant_four_l2661_266190

/-- If cos(π - α) < 0 and tan(α) < 0, then α is in Quadrant IV -/
theorem angle_in_quadrant_four (α : Real) 
  (h1 : Real.cos (Real.pi - α) < 0) 
  (h2 : Real.tan α < 0) : 
  0 < α ∧ α < Real.pi/2 ∧ Real.sin α < 0 ∧ Real.cos α > 0 := by
  sorry

end angle_in_quadrant_four_l2661_266190


namespace sum_of_squares_progression_l2661_266129

/-- Given two infinite geometric progressions with common ratio q where |q| < 1,
    differing only in the sign of their common ratios, and with sums S₁ and S₂ respectively,
    the sum of the infinite geometric progression formed from the squares of the terms
    of either progression is equal to S₁ * S₂. -/
theorem sum_of_squares_progression (q : ℝ) (S₁ S₂ : ℝ) (h : |q| < 1) :
  let b₁ : ℝ := S₁ * (1 - q)
  ∑' n, (b₁ * q ^ n) ^ 2 = S₁ * S₂ :=
by sorry

end sum_of_squares_progression_l2661_266129


namespace cot_sixty_degrees_l2661_266143

theorem cot_sixty_degrees : Real.cos (π / 3) / Real.sin (π / 3) = Real.sqrt 3 / 3 := by
  sorry

end cot_sixty_degrees_l2661_266143


namespace sum_squared_equals_sixteen_l2661_266134

theorem sum_squared_equals_sixteen (a b : ℝ) (h : a + b = 4) : a^2 + 2*a*b + b^2 = 16 := by
  sorry

end sum_squared_equals_sixteen_l2661_266134


namespace multiply_by_twenty_l2661_266183

theorem multiply_by_twenty (x : ℝ) (h : 10 * x = 40) : 20 * x = 80 := by
  sorry

end multiply_by_twenty_l2661_266183


namespace jack_and_toddlers_time_l2661_266101

/-- The time it takes for Jack and his toddlers to get ready -/
def total_time (jack_shoe_time : ℕ) (toddler_extra_time : ℕ) (num_toddlers : ℕ) : ℕ :=
  jack_shoe_time + num_toddlers * (jack_shoe_time + toddler_extra_time)

/-- Theorem: The total time for Jack and his toddlers to get ready is 18 minutes -/
theorem jack_and_toddlers_time : total_time 4 3 2 = 18 := by
  sorry

end jack_and_toddlers_time_l2661_266101


namespace odd_function_log_value_l2661_266105

theorem odd_function_log_value (f : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = Real.log x / Real.log 2) →  -- f(x) = log₂(x) for x > 0
  f (-2) = -1 := by
sorry

end odd_function_log_value_l2661_266105


namespace quadratic_inequality_solution_set_l2661_266139

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - 3 * x - 2 > 0} = {x : ℝ | x < -1/2 ∨ x > 2} := by sorry

end quadratic_inequality_solution_set_l2661_266139


namespace equation_I_consecutive_odd_equation_I_not_prime_equation_II_not_consecutive_odd_equation_II_multiple_of_5_equation_II_consecutive_int_l2661_266178

-- Define the necessary types and functions
def ConsecutiveOdd (x y z : ℕ) : Prop := y = x + 2 ∧ z = y + 2
def ConsecutiveInt (x y z w : ℕ) : Prop := y = x + 1 ∧ z = y + 1 ∧ w = z + 1
def MultipleOf5 (n : ℕ) : Prop := ∃ k, n = 5 * k

-- Theorem statements
theorem equation_I_consecutive_odd :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ ConsecutiveOdd x y z ∧ x + y + z = 45 := by sorry

theorem equation_I_not_prime :
  ¬ ∃ x y z : ℕ, x.Prime ∧ y.Prime ∧ z.Prime ∧ x + y + z = 45 := by sorry

theorem equation_II_not_consecutive_odd :
  ¬ ∃ x y z w : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    ConsecutiveOdd x y z ∧ w = z + 2 ∧ x + y + z + w = 50 := by sorry

theorem equation_II_multiple_of_5 :
  ∃ x y z w : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    MultipleOf5 x ∧ MultipleOf5 y ∧ MultipleOf5 z ∧ MultipleOf5 w ∧
    x + y + z + w = 50 := by sorry

theorem equation_II_consecutive_int :
  ∃ x y z w : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    ConsecutiveInt x y z w ∧ x + y + z + w = 50 := by sorry

end equation_I_consecutive_odd_equation_I_not_prime_equation_II_not_consecutive_odd_equation_II_multiple_of_5_equation_II_consecutive_int_l2661_266178


namespace diesel_in_container_l2661_266181

/-- Represents the ratio of diesel to water in the final mixture -/
def diesel_water_ratio : ℚ := 3 / 5

/-- Amount of petrol in the container -/
def petrol_amount : ℚ := 4

/-- Amount of water added to the container -/
def water_added : ℚ := 2.666666666666667

/-- Calculates the amount of diesel in the container -/
def diesel_amount (ratio : ℚ) (petrol : ℚ) (water : ℚ) : ℚ :=
  ratio * (petrol + water)

theorem diesel_in_container :
  diesel_amount diesel_water_ratio petrol_amount water_added = 4 := by
  sorry

end diesel_in_container_l2661_266181


namespace jana_height_l2661_266153

/-- Given the heights of Jana, Kelly, and Jess, prove Jana's height -/
theorem jana_height (jana_height kelly_height jess_height : ℕ) 
  (h1 : jana_height = kelly_height + 5)
  (h2 : kelly_height = jess_height - 3)
  (h3 : jess_height = 72) : 
  jana_height = 74 := by
  sorry

end jana_height_l2661_266153


namespace tunnel_length_l2661_266196

/-- Given a train and a tunnel, calculate the length of the tunnel. -/
theorem tunnel_length
  (train_length : ℝ)
  (exit_time : ℝ)
  (train_speed : ℝ)
  (h1 : train_length = 2)
  (h2 : exit_time = 4)
  (h3 : train_speed = 120) :
  let distance_traveled := train_speed / 60 * exit_time
  let tunnel_length := distance_traveled - train_length
  tunnel_length = 6 := by
  sorry

end tunnel_length_l2661_266196


namespace circle_tangency_l2661_266114

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle C₂ with equation x² + y² - 6x - 8y + F = 0 -/
def C₂ (F : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 - 8*p.2 + F = 0}

/-- Two circles are internally tangent if they intersect at exactly one point
    and one circle is completely inside the other -/
def internally_tangent (C₁ C₂ : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ C₁ ∧ p ∈ C₂ ∧ (∀ q, q ∈ C₁ ∩ C₂ → q = p) ∧
  (∀ r, r ∈ C₁ → r ∈ C₂ ∨ r = p)

/-- Theorem: If C₁ is internally tangent to C₂, then F = -11 -/
theorem circle_tangency (F : ℝ) :
  internally_tangent C₁ (C₂ F) → F = -11 := by
  sorry

end circle_tangency_l2661_266114


namespace negative_seven_plus_three_l2661_266117

theorem negative_seven_plus_three : (-7 : ℤ) + 3 = -4 := by
  sorry

end negative_seven_plus_three_l2661_266117


namespace benny_baseball_gear_expense_l2661_266127

/-- The amount Benny spent on baseball gear -/
def amount_spent (initial_amount remaining_amount : ℕ) : ℕ :=
  initial_amount - remaining_amount

/-- Theorem: Benny spent $47 on baseball gear -/
theorem benny_baseball_gear_expense :
  amount_spent 79 32 = 47 := by
  sorry

end benny_baseball_gear_expense_l2661_266127


namespace min_value_of_expression_l2661_266120

theorem min_value_of_expression (c d : ℤ) (h : c > d) :
  (c + 2*d) / (c - d) + (c - d) / (c + 2*d) ≥ 2 ∧
  ∃ (c' d' : ℤ), c' > d' ∧ (c' + 2*d') / (c' - d') + (c' - d') / (c' + 2*d') = 2 :=
sorry

end min_value_of_expression_l2661_266120


namespace car_fuel_efficiency_l2661_266112

theorem car_fuel_efficiency (x : ℝ) : x = 40 :=
  by
  have h1 : x > 0 := sorry
  have h2 : (4 / x + 4 / 20) = (8 / x) * 1.50000000000000014 := sorry
  sorry

end car_fuel_efficiency_l2661_266112


namespace max_value_f_on_interval_l2661_266147

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- State the theorem
theorem max_value_f_on_interval : 
  ∃ (c : ℝ), c ∈ Set.Icc (-3 : ℝ) 0 ∧ 
  ∀ (x : ℝ), x ∈ Set.Icc (-3 : ℝ) 0 → f x ≤ f c ∧ f c = 3 := by
  sorry

end max_value_f_on_interval_l2661_266147


namespace square_with_external_triangle_l2661_266199

/-- Given a square ABCD with side length s and an equilateral triangle CDE
    constructed externally on side CD, the ratio of AE to AB is 1 + √3/2 -/
theorem square_with_external_triangle (s : ℝ) (s_pos : s > 0) :
  let AB := s
  let AD := s
  let CD := s
  let CE := s
  let DE := s
  let CDE_altitude := s * Real.sqrt 3 / 2
  let AE := AD + CDE_altitude
  AE / AB = 1 + Real.sqrt 3 / 2 := by sorry

end square_with_external_triangle_l2661_266199


namespace divisible_by_9_when_repeated_thrice_repeat_2013_thrice_divisible_by_9_l2661_266179

/-- Represents the number 2013 repeated n times -/
def repeat_2013 (n : ℕ) : ℕ :=
  2013 * (10 ^ (4 * n) - 1) / 9

/-- The sum of digits of 2013 -/
def sum_of_digits_2013 : ℕ := 2 + 0 + 1 + 3

theorem divisible_by_9_when_repeated_thrice :
  ∃ k : ℕ, repeat_2013 3 = 9 * k :=
sorry

/-- The resulting number when 2013 is repeated 3 times is divisible by 9 -/
theorem repeat_2013_thrice_divisible_by_9 :
  9 ∣ repeat_2013 3 :=
sorry

end divisible_by_9_when_repeated_thrice_repeat_2013_thrice_divisible_by_9_l2661_266179


namespace floor_length_l2661_266184

/-- Represents the properties of a rectangular floor -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  paintCost : ℝ
  paintRate : ℝ

/-- The length of the floor is 200% more than its breadth -/
def lengthCondition (floor : RectangularFloor) : Prop :=
  floor.length = 3 * floor.breadth

/-- The cost to paint the floor is Rs. 300 -/
def costCondition (floor : RectangularFloor) : Prop :=
  floor.paintCost = 300

/-- The painting rate is Rs. 5 per sq m -/
def rateCondition (floor : RectangularFloor) : Prop :=
  floor.paintRate = 5

/-- Theorem stating the length of the floor -/
theorem floor_length (floor : RectangularFloor) 
  (h1 : lengthCondition floor) 
  (h2 : costCondition floor) 
  (h3 : rateCondition floor) : 
  floor.length = 6 * Real.sqrt 5 := by
  sorry

end floor_length_l2661_266184


namespace hyperbola_parameter_sum_l2661_266159

/-- The hyperbola defined by two foci and the difference of distances to these foci. -/
structure Hyperbola where
  f₁ : ℝ × ℝ
  f₂ : ℝ × ℝ
  diff : ℝ

/-- The standard form of a hyperbola equation. -/
structure HyperbolaEquation where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Theorem stating the relationship between the hyperbola's parameters and its equation. -/
theorem hyperbola_parameter_sum (H : Hyperbola) (E : HyperbolaEquation) : 
  H.f₁ = (-3, 1 - Real.sqrt 5 / 4) →
  H.f₂ = (-3, 1 + Real.sqrt 5 / 4) →
  H.diff = 1 →
  E.a > 0 →
  E.b > 0 →
  (∀ (x y : ℝ), (y - E.k)^2 / E.a^2 - (x - E.h)^2 / E.b^2 = 1 ↔ 
    |((x - H.f₁.1)^2 + (y - H.f₁.2)^2).sqrt - ((x - H.f₂.1)^2 + (y - H.f₂.2)^2).sqrt| = H.diff) →
  E.h + E.k + E.a + E.b = -5/4 := by
sorry

end hyperbola_parameter_sum_l2661_266159


namespace fifteen_factorial_base_eight_zeroes_l2661_266144

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- 15! ends with 3 zeroes when written in base 8 -/
theorem fifteen_factorial_base_eight_zeroes :
  trailingZeroes 15 8 = 3 := by
  sorry

end fifteen_factorial_base_eight_zeroes_l2661_266144


namespace a_union_b_iff_c_l2661_266195

-- Define sets A, B, and C
def A : Set ℝ := {x | x - 2 > 0}
def B : Set ℝ := {x | x < 0}
def C : Set ℝ := {x | x * (x - 2) > 0}

-- Theorem statement
theorem a_union_b_iff_c : ∀ x : ℝ, x ∈ A ∪ B ↔ x ∈ C := by sorry

end a_union_b_iff_c_l2661_266195


namespace carl_stamps_l2661_266180

/-- Given that Kevin has 57 stamps and Carl has 32 more stamps than Kevin, 
    prove that Carl has 89 stamps. -/
theorem carl_stamps (kevin_stamps : ℕ) (carl_extra_stamps : ℕ) : 
  kevin_stamps = 57 → 
  carl_extra_stamps = 32 → 
  kevin_stamps + carl_extra_stamps = 89 := by
sorry

end carl_stamps_l2661_266180


namespace product_xyz_w_l2661_266176

theorem product_xyz_w (x y z w : ℚ) 
  (eq1 : 3 * x + 4 * y = 60)
  (eq2 : 6 * x - 4 * y = 12)
  (eq3 : 2 * x - 3 * z = 38)
  (eq4 : x + y + z = w) :
  x * y * z * w = -5104 := by
  sorry

end product_xyz_w_l2661_266176


namespace sin_600_degrees_l2661_266123

theorem sin_600_degrees : Real.sin (600 * π / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end sin_600_degrees_l2661_266123


namespace money_problem_l2661_266121

theorem money_problem (M : ℚ) : 
  (3/4 * (2/3 * (2/3 * M + 10) + 20) = M) → M = 30 := by
  sorry

end money_problem_l2661_266121


namespace second_number_value_second_number_proof_l2661_266166

theorem second_number_value : ℝ → Prop :=
  fun second_number =>
    let first_number : ℝ := 40
    (0.65 * first_number = 0.05 * second_number + 23) →
    second_number = 60

-- Proof
theorem second_number_proof : ∃ (x : ℝ), second_number_value x :=
  sorry

end second_number_value_second_number_proof_l2661_266166


namespace parabola_c_value_l2661_266168

/-- Represents a parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value : 
  ∀ p : Parabola, 
  p.x_coord 3 = 5 → -- vertex at (5, 3)
  p.x_coord 1 = 7 → -- passes through (7, 1)
  p.c = 19/2 := by
sorry

end parabola_c_value_l2661_266168


namespace jerry_to_ivan_ratio_l2661_266194

def ivan_dice : ℕ := 20
def total_dice : ℕ := 60

theorem jerry_to_ivan_ratio : 
  (total_dice - ivan_dice) / ivan_dice = 2 := by
  sorry

end jerry_to_ivan_ratio_l2661_266194


namespace root_product_cubic_l2661_266116

theorem root_product_cubic (p q r : ℝ) : 
  (3 * p^3 - 9 * p^2 + 5 * p - 15 = 0) ∧ 
  (3 * q^3 - 9 * q^2 + 5 * q - 15 = 0) ∧ 
  (3 * r^3 - 9 * r^2 + 5 * r - 15 = 0) →
  p * q * r = 5 := by
sorry

end root_product_cubic_l2661_266116


namespace wilson_sledding_l2661_266106

/-- The number of times Wilson sleds down a tall hill -/
def tall_hill_slides : ℕ := 4

/-- The number of small hills -/
def small_hills : ℕ := 3

/-- The total number of times Wilson sled down all hills -/
def total_slides : ℕ := 14

/-- The number of tall hills Wilson sled down -/
def tall_hills : ℕ := 2

theorem wilson_sledding :
  tall_hills * tall_hill_slides + small_hills * (tall_hill_slides / 2) = total_slides :=
by sorry

end wilson_sledding_l2661_266106


namespace typing_service_problem_l2661_266186

/-- Typing service problem -/
theorem typing_service_problem 
  (total_pages : ℕ) 
  (pages_revised_twice : ℕ) 
  (cost_first_typing : ℕ) 
  (cost_per_revision : ℕ) 
  (total_cost : ℕ) 
  (h1 : total_pages = 200)
  (h2 : pages_revised_twice = 20)
  (h3 : cost_first_typing = 5)
  (h4 : cost_per_revision = 3)
  (h5 : total_cost = 1360) :
  ∃ (pages_revised_once : ℕ),
    pages_revised_once = 80 ∧
    total_cost = 
      total_pages * cost_first_typing + 
      pages_revised_once * cost_per_revision + 
      pages_revised_twice * cost_per_revision * 2 :=
by sorry

end typing_service_problem_l2661_266186


namespace tile_arrangements_count_l2661_266165

/-- Represents the number of ways to arrange four tiles in a row using three colors. -/
def tileArrangements : ℕ := 36

/-- The number of positions in the row of tiles. -/
def numPositions : ℕ := 4

/-- The number of available colors. -/
def numColors : ℕ := 3

/-- The number of tiles of the same color that must be used. -/
def sameColorTiles : ℕ := 2

/-- Theorem stating that the number of tile arrangements is 36. -/
theorem tile_arrangements_count :
  (numColors * (Nat.choose numPositions sameColorTiles * Nat.factorial (numPositions - sameColorTiles))) = tileArrangements :=
by sorry

end tile_arrangements_count_l2661_266165


namespace max_value_on_circle_l2661_266171

theorem max_value_on_circle : 
  ∀ (x y : ℝ), (x - 2)^2 + (y - 2)^2 = 2 → x + 2*y ≤ 6 + Real.sqrt 10 := by
  sorry

end max_value_on_circle_l2661_266171


namespace right_triangle_inequality_l2661_266149

theorem right_triangle_inequality (a b c : ℝ) (h : c^2 = a^2 + b^2) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b ≤ c * Real.sqrt 2 ∧ (a + b = c * Real.sqrt 2 ↔ a = b) := by
  sorry

end right_triangle_inequality_l2661_266149


namespace units_digit_of_k_squared_plus_two_to_k_l2661_266102

def n : ℕ := 4016

def k : ℕ := n^2 + 2^n

theorem units_digit_of_k_squared_plus_two_to_k (n : ℕ) (k : ℕ) :
  n = 4016 →
  k = n^2 + 2^n →
  (k^2 + 2^k) % 10 = 7 := by sorry

end units_digit_of_k_squared_plus_two_to_k_l2661_266102


namespace max_girls_in_ballet_l2661_266125

/-- Represents the number of boys participating in the ballet -/
def num_boys : ℕ := 5

/-- Represents the distance requirement between girls and boys -/
def distance : ℕ := 5

/-- Represents the number of boys required at the specified distance from each girl -/
def boys_per_girl : ℕ := 2

/-- Calculates the maximum number of girls that can participate in the ballet -/
def max_girls : ℕ := (num_boys.choose boys_per_girl) * 2

/-- Theorem stating the maximum number of girls that can participate in the ballet -/
theorem max_girls_in_ballet : max_girls = 20 := by
  sorry

end max_girls_in_ballet_l2661_266125


namespace quadratic_roots_sum_minus_product_l2661_266111

theorem quadratic_roots_sum_minus_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → 
  x₂^2 - 3*x₂ + 2 = 0 → 
  x₁ + x₂ - x₁*x₂ = 1 := by
sorry

end quadratic_roots_sum_minus_product_l2661_266111


namespace roots_are_irrational_l2661_266138

theorem roots_are_irrational (k : ℝ) : 
  (∃ x y : ℝ, x * y = 10 ∧ x^2 - 4*k*x + 3*k^2 + 1 = 0 ∧ y^2 - 4*k*y + 3*k^2 + 1 = 0) →
  (∃ x y : ℝ, x * y = 10 ∧ x^2 - 4*k*x + 3*k^2 + 1 = 0 ∧ y^2 - 4*k*y + 3*k^2 + 1 = 0 ∧ 
   ¬(∃ m n : ℤ, x = ↑m / ↑n) ∧ ¬(∃ m n : ℤ, y = ↑m / ↑n)) :=
by sorry

end roots_are_irrational_l2661_266138


namespace nested_bracket_evaluation_l2661_266167

-- Define the operation [a, b, c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- Define the main theorem
theorem nested_bracket_evaluation :
  let x := bracket (2^4) (2^3) (2^5)
  let y := bracket (3^2) 3 (3^2 + 1)
  let z := bracket (5^2) 5 (5^2 + 1)
  bracket x y z = 169/100 := by sorry

end nested_bracket_evaluation_l2661_266167


namespace equation_solution_l2661_266175

theorem equation_solution (x : ℝ) :
  x ≠ -4 →
  -x^2 = (4*x + 2) / (x + 4) →
  x = -2 ∨ x = -1 := by
sorry

end equation_solution_l2661_266175


namespace late_start_time_l2661_266182

-- Define the usual time to reach the office
def usual_time : ℝ := 60

-- Define the slower speed factor
def slower_speed_factor : ℝ := 0.75

-- Define the late arrival time
def late_arrival : ℝ := 50

-- Theorem statement
theorem late_start_time (actual_journey_time : ℝ) :
  actual_journey_time = usual_time / slower_speed_factor + late_arrival →
  actual_journey_time - (usual_time / slower_speed_factor) = 30 := by
  sorry

end late_start_time_l2661_266182


namespace shifted_parabola_vertex_l2661_266113

/-- Given a parabola y = -2x^2 + 1 shifted 1 unit left and 3 units up, its vertex is at (-1, 4) -/
theorem shifted_parabola_vertex (x y : ℝ) :
  let f : ℝ → ℝ := λ x ↦ -2 * x^2 + 1
  let g : ℝ → ℝ := λ x ↦ f (x + 1) + 3
  g x = y ∧ ∀ t, g t ≤ y → (x = -1 ∧ y = 4) :=
by sorry

end shifted_parabola_vertex_l2661_266113


namespace pen_drawing_probabilities_l2661_266136

/-- Represents a box of pens with different classes -/
structure PenBox where
  total : Nat
  firstClass : Nat
  secondClass : Nat
  thirdClass : Nat

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem about probabilities when drawing pens from a box -/
theorem pen_drawing_probabilities (box : PenBox)
  (h1 : box.total = 6)
  (h2 : box.firstClass = 3)
  (h3 : box.secondClass = 2)
  (h4 : box.thirdClass = 1)
  (h5 : box.total = box.firstClass + box.secondClass + box.thirdClass) :
  let totalCombinations := choose box.total 2
  let exactlyOneFirstClass := box.firstClass * (box.secondClass + box.thirdClass)
  let noThirdClass := choose (box.firstClass + box.secondClass) 2
  (exactlyOneFirstClass : Rat) / totalCombinations = 3 / 5 ∧
  (noThirdClass : Rat) / totalCombinations = 2 / 3 := by
  sorry

end pen_drawing_probabilities_l2661_266136


namespace sms_genuine_iff_criteria_sms_scam_iff_not_genuine_l2661_266160

/-- Represents an SMS message -/
structure SMS where
  sender : Nat
  content : String

/-- Represents a bank -/
structure Bank where
  name : String
  officialSMSNumber : Nat
  customerServiceNumber : Nat

/-- Predicate to check if an SMS is genuine -/
def is_genuine_sms (s : SMS) (b : Bank) : Prop :=
  s.sender = b.officialSMSNumber ∧
  ∃ (confirmation : Bool), 
    (confirmation = true) ∧ 
    (∃ (response : String), response = "Confirmed")

/-- Theorem: An SMS is genuine if and only if it meets the specified criteria -/
theorem sms_genuine_iff_criteria (s : SMS) (b : Bank) :
  is_genuine_sms s b ↔ 
  (s.sender = b.officialSMSNumber ∧ 
   ∃ (confirmation : Bool), 
     (confirmation = true) ∧ 
     (∃ (response : String), response = "Confirmed")) :=
by sorry

/-- Theorem: An SMS is a scam if and only if it doesn't meet the criteria for being genuine -/
theorem sms_scam_iff_not_genuine (s : SMS) (b : Bank) :
  ¬(is_genuine_sms s b) ↔ 
  (s.sender ≠ b.officialSMSNumber ∨ 
   ∀ (confirmation : Bool), 
     (confirmation = false) ∨ 
     (∀ (response : String), response ≠ "Confirmed")) :=
by sorry

end sms_genuine_iff_criteria_sms_scam_iff_not_genuine_l2661_266160


namespace fifth_number_pascal_proof_l2661_266173

/-- The fifth number in the row of Pascal's triangle that starts with 1 and 15 -/
def fifth_number_pascal : ℕ := 1365

/-- The row of Pascal's triangle we're interested in -/
def pascal_row : List ℕ := [1, 15]

/-- Theorem stating that the fifth number in the specified row of Pascal's triangle is 1365 -/
theorem fifth_number_pascal_proof : 
  ∀ (row : List ℕ), row = pascal_row → 
  (List.nthLe row 4 sorry : ℕ) = fifth_number_pascal := by
  sorry

end fifth_number_pascal_proof_l2661_266173


namespace gain_percent_calculation_l2661_266177

/-- Prove that if the cost price of 75 articles equals the selling price of 56.25 articles,
    then the gain percent is 33.33%. -/
theorem gain_percent_calculation (C S : ℝ) (h : 75 * C = 56.25 * S) :
  (S - C) / C * 100 = 100 / 3 := by
  sorry

end gain_percent_calculation_l2661_266177


namespace map_scale_calculation_l2661_266140

/-- Given a map where 15 cm represents 90 km, prove that 20 cm represents 120 km. -/
theorem map_scale_calculation (map_cm : ℝ) (real_km : ℝ) (h : map_cm = 15 ∧ real_km = 90) :
  (20 * real_km) / map_cm = 120 :=
by sorry

end map_scale_calculation_l2661_266140


namespace cos_squared_pi_sixth_plus_half_alpha_l2661_266103

theorem cos_squared_pi_sixth_plus_half_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (π / 6 + α / 2) ^ 2 = 2 / 3 := by
sorry

end cos_squared_pi_sixth_plus_half_alpha_l2661_266103


namespace square_root_product_equals_28_l2661_266141

theorem square_root_product_equals_28 : 
  Real.sqrt (49 * Real.sqrt 25 * Real.sqrt 64) = 28 := by
  sorry

end square_root_product_equals_28_l2661_266141


namespace triangular_weight_is_60_l2661_266118

/-- Given a set of weights with specific balancing conditions, prove that the triangular weight is 60 grams. -/
theorem triangular_weight_is_60 :
  ∀ (round_weight triangular_weight : ℝ),
  (round_weight + triangular_weight = 3 * round_weight) →
  (4 * round_weight + triangular_weight = triangular_weight + round_weight + 90) →
  triangular_weight = 60 := by
  sorry

end triangular_weight_is_60_l2661_266118


namespace ball_probability_l2661_266187

theorem ball_probability (n : ℕ) : 
  (1 : ℕ) + (1 : ℕ) + n > 0 →
  (n : ℚ) / ((1 : ℚ) + (1 : ℚ) + (n : ℚ)) = (1 : ℚ) / (2 : ℚ) →
  n = 2 := by
sorry

end ball_probability_l2661_266187


namespace complex_fraction_calculation_l2661_266162

theorem complex_fraction_calculation : 
  (7 + 4/25 + 8.6) / ((4 + 5/7 - 0.005 * 900) / (6/7)) = 63.04 := by
  sorry

end complex_fraction_calculation_l2661_266162


namespace complex_modulus_example_l2661_266109

theorem complex_modulus_example : Complex.abs (7/8 + 3*I) = 25/8 := by
  sorry

end complex_modulus_example_l2661_266109


namespace tangent_condition_l2661_266128

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x + y + 1 = 0

/-- The circle equation -/
def circle_equation (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 2

/-- The line is tangent to the circle -/
def is_tangent (a b : ℝ) : Prop := ∃ x y : ℝ, line_equation x y ∧ circle_equation x y a b ∧
  ∀ x' y' : ℝ, line_equation x' y' → circle_equation x' y' a b → (x', y') = (x, y)

theorem tangent_condition (a b : ℝ) :
  (a + b = 1 → is_tangent a b) ∧
  (∃ a' b' : ℝ, is_tangent a' b' ∧ a' + b' ≠ 1) :=
sorry

end tangent_condition_l2661_266128


namespace can_collection_ratio_l2661_266192

theorem can_collection_ratio : 
  ∀ (solomon juwan levi : ℕ),
  solomon = 3 * juwan →
  solomon = 66 →
  solomon + juwan + levi = 99 →
  levi * 2 = juwan :=
by
  sorry

end can_collection_ratio_l2661_266192


namespace min_value_geometric_sequence_l2661_266169

theorem min_value_geometric_sequence (a₁ a₂ a₃ : ℝ) :
  a₁ = 1 →
  (∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) →
  (∀ b₂ b₃ : ℝ, (∃ s : ℝ, b₂ = a₁ * s ∧ b₃ = b₂ * s) → 4 * a₂ + 5 * a₃ ≤ 4 * b₂ + 5 * b₃) →
  4 * a₂ + 5 * a₃ = -4/5 :=
by sorry

end min_value_geometric_sequence_l2661_266169


namespace consecutive_integers_product_255_l2661_266185

theorem consecutive_integers_product_255 (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 255) :
  x + (x + 1) = 31 := by
sorry

end consecutive_integers_product_255_l2661_266185


namespace inequality_not_hold_l2661_266119

theorem inequality_not_hold (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(1 / (a - b) > 1 / a) := by
sorry

end inequality_not_hold_l2661_266119


namespace karlson_candy_theorem_l2661_266145

/-- The number of initial ones on the board -/
def initial_ones : ℕ := 39

/-- The number of minutes the process continues -/
def total_minutes : ℕ := 39

/-- The maximum number of candies Karlson could have eaten -/
def max_candies : ℕ := initial_ones.choose 2

theorem karlson_candy_theorem : 
  max_candies = (initial_ones * (initial_ones - 1)) / 2 := by
  sorry

#eval max_candies

end karlson_candy_theorem_l2661_266145


namespace cycle_gain_percent_l2661_266161

/-- The gain percent when selling a cycle -/
theorem cycle_gain_percent (cost_price selling_price : ℚ) :
  cost_price = 900 →
  selling_price = 1080 →
  (selling_price - cost_price) / cost_price * 100 = 20 := by
sorry

end cycle_gain_percent_l2661_266161


namespace triangle_circumradius_l2661_266104

/-- Given a triangle with sides a and b, area S, and the median to the third side
    less than half of that side, prove that the radius of the circumcircle is 8 / √15 -/
theorem triangle_circumradius (a b : ℝ) (S : ℝ) (h_a : a = 2) (h_b : b = 3)
  (h_S : S = (3 * Real.sqrt 15) / 4)
  (h_median : ∃ (m : ℝ), m < (a + b) / 4 ∧ m^2 = (2 * (a^2 + b^2) - ((a + b) / 2)^2) / 4) :
  ∃ (R : ℝ), R = 8 / Real.sqrt 15 ∧ R * 2 * S = a * b * (Real.sqrt ((a + b + (a + b)) * (-a + b + (a + b)) * (a - b + (a + b)) * (a + b - (a + b))) / (4 * (a + b))) := by
  sorry

end triangle_circumradius_l2661_266104


namespace line_parallel_to_parallel_planes_l2661_266157

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relationship between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relationship between planes
variable (parallel_plane : Plane → Plane → Prop)

-- Define the "within" relationship between lines and planes
variable (within : Line → Plane → Prop)

-- Theorem statement
theorem line_parallel_to_parallel_planes 
  (b : Line) (α β : Plane) 
  (h1 : parallel_line_plane b α) 
  (h2 : parallel_plane α β) : 
  parallel_line_plane b β ∨ within b β := by
  sorry

end line_parallel_to_parallel_planes_l2661_266157


namespace circle_equation_l2661_266163

theorem circle_equation (x y k : ℝ) : 
  (∃ h c : ℝ, ∀ x y, x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x - h)^2 + (y - c)^2 = 10^2) ↔ 
  k = 35 := by
sorry

end circle_equation_l2661_266163


namespace percentage_returned_is_65_percent_l2661_266124

/-- Represents the library's special collection --/
structure SpecialCollection where
  initial_count : ℕ
  final_count : ℕ
  loaned_out : ℕ

/-- Calculates the percentage of loaned books returned --/
def percentage_returned (sc : SpecialCollection) : ℚ :=
  (sc.loaned_out - (sc.initial_count - sc.final_count)) / sc.loaned_out * 100

/-- Theorem stating that the percentage of loaned books returned is 65% --/
theorem percentage_returned_is_65_percent (sc : SpecialCollection) 
  (h1 : sc.initial_count = 150)
  (h2 : sc.final_count = 122)
  (h3 : sc.loaned_out = 80) : 
  percentage_returned sc = 65 := by
  sorry

#eval percentage_returned { initial_count := 150, final_count := 122, loaned_out := 80 }

end percentage_returned_is_65_percent_l2661_266124


namespace task_assignment_count_l2661_266131

/-- Represents the number of people who can work as both English translators and software designers -/
def both_jobs : ℕ := 1

/-- Represents the total number of people -/
def total_people : ℕ := 8

/-- Represents the number of people who can work as English translators -/
def english_translators : ℕ := 5

/-- Represents the number of people who can work as software designers -/
def software_designers : ℕ := 4

/-- Represents the number of people to be selected for the task -/
def selected_people : ℕ := 5

/-- Represents the number of people to be assigned as English translators -/
def assigned_translators : ℕ := 3

/-- Represents the number of people to be assigned as software designers -/
def assigned_designers : ℕ := 2

/-- Theorem stating that the number of ways to assign tasks is 42 -/
theorem task_assignment_count : 
  (Nat.choose (english_translators - both_jobs) assigned_translators * 
   Nat.choose (software_designers - both_jobs) assigned_designers) +
  (Nat.choose (english_translators - both_jobs) (assigned_translators - 1) * 
   Nat.choose software_designers assigned_designers) +
  (Nat.choose english_translators assigned_translators * 
   Nat.choose (software_designers - both_jobs) (assigned_designers - 1)) = 42 :=
by sorry

end task_assignment_count_l2661_266131


namespace walnut_trees_after_planting_l2661_266122

/-- The number of walnut trees in the park after planting -/
def total_trees (current_trees newly_planted_trees : ℕ) : ℕ :=
  current_trees + newly_planted_trees

/-- Theorem: The total number of walnut trees after planting is the sum of current trees and newly planted trees -/
theorem walnut_trees_after_planting 
  (current_trees : ℕ) 
  (newly_planted_trees : ℕ) :
  total_trees current_trees newly_planted_trees = current_trees + newly_planted_trees :=
by sorry

/-- Given information about the walnut trees in the park -/
def current_walnut_trees : ℕ := 22
def new_walnut_trees : ℕ := 55

/-- The total number of walnut trees after planting -/
def final_walnut_trees : ℕ := total_trees current_walnut_trees new_walnut_trees

#eval final_walnut_trees

end walnut_trees_after_planting_l2661_266122


namespace a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq_l2661_266108

theorem a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq :
  (∀ a b : ℝ, a = b → a^2 = b^2) ∧
  (∃ a b : ℝ, a^2 = b^2 ∧ a ≠ b) := by
  sorry

end a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq_l2661_266108


namespace correct_banana_distribution_l2661_266189

def banana_distribution (total dawn lydia donna emily : ℚ) : Prop :=
  total = 550.5 ∧
  dawn = lydia + 93 ∧
  lydia = 80.25 ∧
  donna = emily / 2 ∧
  dawn + lydia + donna + emily = total

theorem correct_banana_distribution :
  ∃ (dawn lydia donna emily : ℚ),
    banana_distribution total dawn lydia donna emily ∧
    dawn = 173.25 ∧
    lydia = 80.25 ∧
    donna = 99 ∧
    emily = 198 := by
  sorry

end correct_banana_distribution_l2661_266189


namespace emerald_count_l2661_266154

/-- Represents the count of gemstones in a box -/
def GemCount := Nat

/-- Represents a box of gemstones -/
structure Box where
  count : GemCount

/-- Represents the collection of all boxes -/
structure JewelryBox where
  boxes : List Box
  diamond_boxes : List Box
  ruby_boxes : List Box
  emerald_boxes : List Box

/-- The total count of gemstones in a list of boxes -/
def total_gems (boxes : List Box) : Nat :=
  boxes.map (λ b => b.count) |>.sum

theorem emerald_count (jb : JewelryBox) 
  (h1 : jb.boxes.length = 6)
  (h2 : jb.diamond_boxes.length = 2)
  (h3 : jb.ruby_boxes.length = 2)
  (h4 : jb.emerald_boxes.length = 2)
  (h5 : jb.boxes = jb.diamond_boxes ++ jb.ruby_boxes ++ jb.emerald_boxes)
  (h6 : total_gems jb.ruby_boxes = total_gems jb.diamond_boxes + 15)
  (h7 : total_gems jb.boxes = 39) :
  total_gems jb.emerald_boxes = 12 := by
  sorry

end emerald_count_l2661_266154


namespace nicholas_bottle_caps_l2661_266132

theorem nicholas_bottle_caps :
  ∀ (initial : ℕ),
  initial + 85 = 93 →
  initial = 8 :=
by
  sorry

end nicholas_bottle_caps_l2661_266132


namespace equation_solution_l2661_266188

theorem equation_solution : ∃ x : ℚ, 50 + 5 * x / (180 / 3) = 51 ∧ x = 12 := by
  sorry

end equation_solution_l2661_266188


namespace water_displacement_l2661_266152

theorem water_displacement (tank_length tank_width : ℝ) 
  (water_level_rise : ℝ) (num_men : ℕ) :
  tank_length = 40 ∧ 
  tank_width = 20 ∧ 
  water_level_rise = 0.25 ∧ 
  num_men = 50 → 
  (tank_length * tank_width * water_level_rise) / num_men = 4 := by
  sorry

end water_displacement_l2661_266152
