import Mathlib

namespace right_triangle_area_l86_8613

theorem right_triangle_area (base hypotenuse : ℝ) (h_right_angle : base > 0 ∧ hypotenuse > 0 ∧ hypotenuse > base) :
  base = 12 → hypotenuse = 13 → 
  ∃ height : ℝ, height > 0 ∧ height ^ 2 + base ^ 2 = hypotenuse ^ 2 ∧ 
  (1 / 2) * base * height = 30 :=
by sorry

end right_triangle_area_l86_8613


namespace last_box_weight_l86_8612

theorem last_box_weight (box1_weight box2_weight total_weight : ℕ) : 
  box1_weight = 2 → 
  box2_weight = 11 → 
  total_weight = 18 → 
  ∃ last_box_weight : ℕ, last_box_weight = total_weight - (box1_weight + box2_weight) ∧ 
                           last_box_weight = 5 := by
  sorry

end last_box_weight_l86_8612


namespace class_average_weight_l86_8600

/-- The average weight of a group of children given their total weight and count -/
def average_weight (total_weight : ℚ) (count : ℕ) : ℚ :=
  total_weight / count

/-- The total weight of a group of children given their average weight and count -/
def total_weight (avg_weight : ℚ) (count : ℕ) : ℚ :=
  avg_weight * count

theorem class_average_weight 
  (boys_count : ℕ) 
  (girls_count : ℕ) 
  (boys_avg_weight : ℚ) 
  (girls_avg_weight : ℚ) 
  (h1 : boys_count = 8)
  (h2 : girls_count = 6)
  (h3 : boys_avg_weight = 140)
  (h4 : girls_avg_weight = 130) :
  average_weight 
    (total_weight boys_avg_weight boys_count + total_weight girls_avg_weight girls_count) 
    (boys_count + girls_count) = 135 := by
  sorry

end class_average_weight_l86_8600


namespace min_sum_with_product_constraint_l86_8641

theorem min_sum_with_product_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : a * b = a + b + 3) : 
  6 ≤ a + b ∧ ∀ (x y : ℝ), 0 < x → 0 < y → x * y = x + y + 3 → a + b ≤ x + y := by
  sorry

end min_sum_with_product_constraint_l86_8641


namespace special_gp_ratio_is_one_l86_8624

/-- A geometric progression with positive terms where any term is the product of the next two -/
structure SpecialGP where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  term_product : ∀ n : ℕ, a * r^n = (a * r^(n+1)) * (a * r^(n+2))

/-- The common ratio of a SpecialGP is 1 -/
theorem special_gp_ratio_is_one (gp : SpecialGP) : gp.r = 1 := by
  sorry

end special_gp_ratio_is_one_l86_8624


namespace arithmetic_mean_greater_than_harmonic_mean_l86_8631

theorem arithmetic_mean_greater_than_harmonic_mean 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (a + b) / 2 > 2 * a * b / (a + b) := by
  sorry

end arithmetic_mean_greater_than_harmonic_mean_l86_8631


namespace bottles_left_l86_8646

theorem bottles_left (initial_bottles drunk_bottles : ℕ) :
  initial_bottles = 17 →
  drunk_bottles = 3 →
  initial_bottles - drunk_bottles = 14 :=
by sorry

end bottles_left_l86_8646


namespace beavers_swimming_l86_8658

theorem beavers_swimming (initial_beavers : ℕ) (remaining_beavers : ℕ) : 
  initial_beavers = 2 → remaining_beavers = 1 → initial_beavers - remaining_beavers = 1 := by
  sorry

end beavers_swimming_l86_8658


namespace smallest_angle_solution_l86_8656

theorem smallest_angle_solution (θ : Real) : 
  (θ > 0) → 
  (θ < 360) → 
  (Real.cos (θ * π / 180) = Real.sin (70 * π / 180) + Real.cos (50 * π / 180) - Real.sin (20 * π / 180) - Real.cos (10 * π / 180)) → 
  (∀ φ, 0 < φ ∧ φ < θ → Real.cos (φ * π / 180) ≠ Real.sin (70 * π / 180) + Real.cos (50 * π / 180) - Real.sin (20 * π / 180) - Real.cos (10 * π / 180)) → 
  θ = 50 := by
sorry

end smallest_angle_solution_l86_8656


namespace paint_time_per_room_l86_8651

theorem paint_time_per_room 
  (total_rooms : ℕ) 
  (painted_rooms : ℕ) 
  (remaining_time : ℕ) 
  (h1 : total_rooms = 11) 
  (h2 : painted_rooms = 2) 
  (h3 : remaining_time = 63) : 
  remaining_time / (total_rooms - painted_rooms) = 7 := by
  sorry

end paint_time_per_room_l86_8651


namespace amount_left_after_pool_l86_8639

-- Define the given conditions
def total_earned : ℝ := 30
def cost_per_person : ℝ := 2.5
def number_of_people : ℕ := 10

-- Define the theorem
theorem amount_left_after_pool : 
  total_earned - (cost_per_person * number_of_people) = 5 := by
  sorry

end amount_left_after_pool_l86_8639


namespace base3_20112_equals_176_l86_8616

/-- Converts a base-3 digit to its base-10 equivalent --/
def base3ToBase10Digit (d : Nat) : Nat :=
  if d < 3 then d else 0

/-- Converts a base-3 number represented as a list of digits to its base-10 equivalent --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + base3ToBase10Digit d * 3^i) 0

theorem base3_20112_equals_176 :
  base3ToBase10 [2, 1, 1, 0, 2] = 176 := by
  sorry

end base3_20112_equals_176_l86_8616


namespace expression_factorization_l86_8608

theorem expression_factorization (x : ℝ) :
  (12 * x^4 - 27 * x^2 + 9) - (-3 * x^4 - 9 * x^2 + 6) = 3 * (5 * x^2 - 1) * (x^2 - 1) := by
  sorry

end expression_factorization_l86_8608


namespace solve_for_m_l86_8648

/-- Given that the solution set of mx + 2 > 0 is {x | x < 2}, prove that m = -1 -/
theorem solve_for_m (m : ℝ) 
  (h : ∀ x, mx + 2 > 0 ↔ x < 2) : 
  m = -1 := by
  sorry

end solve_for_m_l86_8648


namespace p_sufficient_not_necessary_for_q_l86_8667

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (0 < x ∧ x < 5) → (-5 < x - 2 ∧ x - 2 < 5)) ∧
  (∃ x : ℝ, (-5 < x - 2 ∧ x - 2 < 5) ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end p_sufficient_not_necessary_for_q_l86_8667


namespace grace_september_earnings_l86_8694

/-- Calculates Grace's earnings for landscaping in September --/
theorem grace_september_earnings :
  let mowing_rate : ℕ := 6
  let weeding_rate : ℕ := 11
  let mulching_rate : ℕ := 9
  let mowing_hours : ℕ := 63
  let weeding_hours : ℕ := 9
  let mulching_hours : ℕ := 10
  let total_earnings : ℕ := 
    mowing_rate * mowing_hours + 
    weeding_rate * weeding_hours + 
    mulching_rate * mulching_hours
  total_earnings = 567 := by
sorry

end grace_september_earnings_l86_8694


namespace cos_12_cos_18_minus_sin_12_sin_18_l86_8636

theorem cos_12_cos_18_minus_sin_12_sin_18 :
  Real.cos (12 * π / 180) * Real.cos (18 * π / 180) - 
  Real.sin (12 * π / 180) * Real.sin (18 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end cos_12_cos_18_minus_sin_12_sin_18_l86_8636


namespace triangle_area_l86_8676

/-- Given a triangle ABC with side lengths a, b, c, prove that its area is √3/4 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  b = 1 →
  b * Real.cos C = c * Real.cos B →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4 := by
  sorry

end triangle_area_l86_8676


namespace jenny_jellybeans_proof_l86_8629

/-- The original number of jellybeans in Jenny's jar -/
def original_jellybeans : ℝ := 85

/-- The fraction of jellybeans remaining after each day -/
def daily_remaining_fraction : ℝ := 0.7

/-- The number of days Jenny eats jellybeans -/
def days : ℕ := 3

/-- The number of jellybeans remaining after 'days' days -/
def remaining_jellybeans : ℝ := 29.16

/-- Theorem stating that the original number of jellybeans is correct -/
theorem jenny_jellybeans_proof :
  original_jellybeans * daily_remaining_fraction ^ days = remaining_jellybeans := by
  sorry

#eval original_jellybeans -- Should output 85

end jenny_jellybeans_proof_l86_8629


namespace sum_of_mean_and_median_l86_8627

def number_set : List ℕ := [1, 2, 3, 0, 1]

def median (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

theorem sum_of_mean_and_median :
  median number_set + mean number_set = 12/5 := by sorry

end sum_of_mean_and_median_l86_8627


namespace jace_travel_distance_l86_8680

/-- Calculates the total distance traveled given a constant speed and two driving periods -/
def total_distance (speed : ℝ) (time1 : ℝ) (time2 : ℝ) : ℝ :=
  speed * (time1 + time2)

/-- Theorem stating that given the specified conditions, the total distance traveled is 780 miles -/
theorem jace_travel_distance :
  let speed : ℝ := 60
  let time1 : ℝ := 4
  let time2 : ℝ := 9
  total_distance speed time1 time2 = 780 := by
  sorry

end jace_travel_distance_l86_8680


namespace power_seven_137_mod_nine_l86_8623

theorem power_seven_137_mod_nine : 7^137 % 9 = 7 := by
  sorry

end power_seven_137_mod_nine_l86_8623


namespace b_plus_3b_squared_positive_l86_8604

theorem b_plus_3b_squared_positive (b : ℝ) (h1 : -0.5 < b) (h2 : b < 0) : 
  b + 3 * b^2 > 0 := by
  sorry

end b_plus_3b_squared_positive_l86_8604


namespace square_property_necessary_not_sufficient_l86_8611

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The property a_{n+1}^2 = a_n * a_{n+2} for all n -/
def HasSquareProperty (a : Sequence) : Prop :=
  ∀ n : ℕ, (a (n + 1))^2 = a n * a (n + 2)

/-- Definition of a geometric sequence -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem: HasSquareProperty is necessary but not sufficient for IsGeometric -/
theorem square_property_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → HasSquareProperty a) ∧
  ¬(∀ a : Sequence, HasSquareProperty a → IsGeometric a) := by
  sorry


end square_property_necessary_not_sufficient_l86_8611


namespace cupcakes_left_l86_8654

/-- The number of cupcakes in a dozen -/
def dozen : ℕ := 12

/-- The number of cupcakes Dani brings -/
def cupcakes_brought : ℕ := 2 * dozen + dozen / 2

/-- The total number of students -/
def total_students : ℕ := 27

/-- The number of teachers -/
def teachers : ℕ := 1

/-- The number of teacher's aids -/
def teacher_aids : ℕ := 1

/-- The number of students who called in sick -/
def sick_students : ℕ := 3

/-- Theorem: The number of cupcakes left after distribution -/
theorem cupcakes_left : 
  cupcakes_brought - (total_students - sick_students + teachers + teacher_aids) = 4 := by
  sorry

end cupcakes_left_l86_8654


namespace probability_one_ball_in_last_box_l86_8661

theorem probability_one_ball_in_last_box (n : ℕ) (h : n = 100) :
  let p := 1 / n
  (n : ℝ) * p * (1 - p)^(n - 1) = ((n - 1 : ℝ) / n)^(n - 1) :=
by sorry

end probability_one_ball_in_last_box_l86_8661


namespace trapezoid_area_is_72_l86_8633

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- One of the base angles in radians -/
  baseAngle : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The trapezoid is circumscribed around a circle -/
  isCircumscribed : True

/-- Calculate the area of the isosceles trapezoid -/
def areaOfTrapezoid (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem trapezoid_area_is_72 (t : IsoscelesTrapezoid) 
  (h1 : t.longerBase = 20)
  (h2 : t.baseAngle = Real.arcsin 0.6) :
  areaOfTrapezoid t = 72 := by
  sorry

end trapezoid_area_is_72_l86_8633


namespace rectangular_fence_length_l86_8647

/-- A rectangular fence with a perimeter of 30 meters and a length that is twice its width has a length of 10 meters. -/
theorem rectangular_fence_length (width : ℝ) (length : ℝ) : 
  width > 0 → 
  length > 0 → 
  length = 2 * width → 
  2 * length + 2 * width = 30 → 
  length = 10 := by
sorry

end rectangular_fence_length_l86_8647


namespace square_perimeter_ratio_l86_8652

theorem square_perimeter_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_area_ratio : a^2 / b^2 = 16 / 25) :
  a / b = 4 / 5 := by sorry

end square_perimeter_ratio_l86_8652


namespace square_root_sum_l86_8657

theorem square_root_sum (x y : ℝ) : (x + 2)^2 + Real.sqrt (y - 18) = 0 → Real.sqrt (x + y) = 4 := by
  sorry

end square_root_sum_l86_8657


namespace subset_implies_a_values_l86_8605

def A : Set ℝ := {3, 5}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem subset_implies_a_values (a : ℝ) (h : B a ⊆ A) : a = 0 ∨ a = 1/3 ∨ a = 1/5 := by
  sorry

end subset_implies_a_values_l86_8605


namespace arithmetic_series_sum_l86_8607

theorem arithmetic_series_sum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = -300 →
  aₙ = 309 →
  d = 3 →
  n = (aₙ - a₁) / d + 1 →
  (n : ℤ) * (a₁ + aₙ) / 2 = 918 :=
by sorry

end arithmetic_series_sum_l86_8607


namespace arithmetic_sequence_eighth_term_l86_8688

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_second : a 2 = 4)
  (h_sixth : a 6 = 2) :
  a 8 = 1 := by
  sorry

end arithmetic_sequence_eighth_term_l86_8688


namespace sum_and_product_of_primes_l86_8668

theorem sum_and_product_of_primes :
  ∀ p q : ℕ, Prime p → Prime q → p + q = 85 → p * q = 166 := by
sorry

end sum_and_product_of_primes_l86_8668


namespace line_circle_intersection_slope_range_l86_8672

/-- Given a line intersecting a circle, prove the range of its slope. -/
theorem line_circle_intersection_slope_range :
  ∀ (a : ℝ),
  (∃ (x y : ℝ), x + a * y + 2 = 0 ∧ x^2 + y^2 + 2*x - 2*y + 1 = 0) →
  a ≤ 0 :=
by sorry

end line_circle_intersection_slope_range_l86_8672


namespace stanley_walk_distance_l86_8663

theorem stanley_walk_distance (run_distance walk_distance : ℝ) :
  run_distance = 0.4 →
  run_distance = walk_distance + 0.2 →
  walk_distance = 0.2 := by
sorry

end stanley_walk_distance_l86_8663


namespace no_primes_divisible_by_42_l86_8662

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define the factors of 42
def factorsOf42 : List ℕ := [2, 3, 7]

-- Theorem statement
theorem no_primes_divisible_by_42 : 
  ∀ p : ℕ, isPrime p → ¬(42 ∣ p) :=
sorry

end no_primes_divisible_by_42_l86_8662


namespace intersecting_circles_properties_l86_8681

/-- Two circles intersecting at two distinct points -/
structure IntersectingCircles where
  r : ℝ
  a : ℝ
  b : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  r_pos : r > 0
  on_C₁_A : x₁^2 + y₁^2 = r^2
  on_C₁_B : x₂^2 + y₂^2 = r^2
  on_C₂_A : (x₁ - a)^2 + (y₁ - b)^2 = r^2
  on_C₂_B : (x₂ - a)^2 + (y₂ - b)^2 = r^2
  distinct : (x₁, y₁) ≠ (x₂, y₂)

/-- Properties of intersecting circles -/
theorem intersecting_circles_properties (c : IntersectingCircles) :
  (c.a * (c.x₁ - c.x₂) + c.b * (c.y₁ - c.y₂) = 0) ∧
  (2 * c.a * c.x₁ + 2 * c.b * c.y₁ = c.a^2 + c.b^2) ∧
  (c.x₁ + c.x₂ = c.a ∧ c.y₁ + c.y₂ = c.b) := by
  sorry

end intersecting_circles_properties_l86_8681


namespace blue_tickets_per_red_l86_8671

/-- The number of yellow tickets needed to win a Bible -/
def yellow_tickets_needed : ℕ := 10

/-- The number of red tickets needed for one yellow ticket -/
def red_per_yellow : ℕ := 10

/-- The number of yellow tickets Tom has -/
def tom_yellow : ℕ := 8

/-- The number of red tickets Tom has -/
def tom_red : ℕ := 3

/-- The number of blue tickets Tom has -/
def tom_blue : ℕ := 7

/-- The number of additional blue tickets Tom needs to win a Bible -/
def additional_blue_needed : ℕ := 163

/-- The number of blue tickets required to obtain one red ticket -/
def blue_per_red : ℕ := 10

theorem blue_tickets_per_red : 
  yellow_tickets_needed = 10 ∧ 
  red_per_yellow = 10 ∧ 
  tom_yellow = 8 ∧ 
  tom_red = 3 ∧ 
  tom_blue = 7 ∧ 
  additional_blue_needed = 163 → 
  blue_per_red = 10 := by sorry

end blue_tickets_per_red_l86_8671


namespace sin_decreasing_interval_l86_8643

/-- The function f(x) = sin(π/6 - x) is strictly decreasing on the interval [0, 2π/3] -/
theorem sin_decreasing_interval (x : ℝ) :
  x ∈ Set.Icc 0 (2 * Real.pi / 3) →
  StrictMonoOn (fun x => Real.sin (Real.pi / 6 - x)) (Set.Icc 0 (2 * Real.pi / 3)) := by
  sorry

end sin_decreasing_interval_l86_8643


namespace sqrt_equation_solution_l86_8625

theorem sqrt_equation_solution :
  ∃ (x : ℝ), x = (1225 : ℝ) / 36 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 :=
by sorry

end sqrt_equation_solution_l86_8625


namespace pin_combinations_l86_8669

/-- The number of distinct digits in the PIN -/
def n : ℕ := 4

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem: The number of permutations of 4 distinct objects is 24 -/
theorem pin_combinations : permutations n = 24 := by
  sorry

end pin_combinations_l86_8669


namespace model_M_completion_time_l86_8690

/-- The time (in minutes) it takes for a model M computer to complete the task -/
def model_M_time : ℝ := 24

/-- The time (in minutes) it takes for a model N computer to complete the task -/
def model_N_time : ℝ := 12

/-- The number of each model of computer used -/
def num_computers : ℕ := 8

/-- The time (in minutes) it takes for the combined computers to complete the task -/
def combined_time : ℝ := 1

theorem model_M_completion_time :
  (num_computers : ℝ) / model_M_time + (num_computers : ℝ) / model_N_time = 1 / combined_time :=
sorry

end model_M_completion_time_l86_8690


namespace base_five_digits_of_1297_l86_8687

theorem base_five_digits_of_1297 : ∃ n : ℕ, n > 0 ∧ 5^(n-1) ≤ 1297 ∧ 1297 < 5^n ∧ n = 5 := by
  sorry

end base_five_digits_of_1297_l86_8687


namespace contract_schemes_count_l86_8630

def projects : ℕ := 6
def company_a_projects : ℕ := 3
def company_b_projects : ℕ := 2
def company_c_projects : ℕ := 1

theorem contract_schemes_count :
  (Nat.choose projects company_a_projects) *
  (Nat.choose (projects - company_a_projects) company_b_projects) *
  (Nat.choose (projects - company_a_projects - company_b_projects) company_c_projects) = 60 := by
  sorry

end contract_schemes_count_l86_8630


namespace hardware_store_earnings_l86_8682

/-- Represents the sales data for a single item -/
structure ItemSales where
  quantity : Nat
  price : Nat
  discount_percent : Nat
  returns : Nat

/-- Calculates the total earnings from the hardware store sales -/
def calculate_earnings (sales_data : List ItemSales) : Nat :=
  sales_data.foldl (fun acc item =>
    let gross_sales := item.quantity * item.price
    let discount := gross_sales * item.discount_percent / 100
    let returns := item.returns * item.price
    acc + gross_sales - discount - returns
  ) 0

/-- Theorem stating that the total earnings of the hardware store are $11740 -/
theorem hardware_store_earnings : 
  let sales_data : List ItemSales := [
    { quantity := 10, price := 600, discount_percent := 10, returns := 0 },  -- Graphics cards
    { quantity := 14, price := 80,  discount_percent := 0,  returns := 0 },  -- Hard drives
    { quantity := 8,  price := 200, discount_percent := 0,  returns := 2 },  -- CPUs
    { quantity := 4,  price := 60,  discount_percent := 0,  returns := 0 },  -- RAM
    { quantity := 12, price := 90,  discount_percent := 0,  returns := 0 },  -- Power supply units
    { quantity := 6,  price := 250, discount_percent := 0,  returns := 0 },  -- Monitors
    { quantity := 18, price := 40,  discount_percent := 0,  returns := 0 },  -- Keyboards
    { quantity := 24, price := 20,  discount_percent := 0,  returns := 0 }   -- Mice
  ]
  calculate_earnings sales_data = 11740 := by
  sorry


end hardware_store_earnings_l86_8682


namespace find_n_l86_8697

theorem find_n : ∃ n : ℕ, 2^n = 2 * 16^2 * 4^3 ∧ n = 15 := by sorry

end find_n_l86_8697


namespace smallest_d_for_injective_g_l86_8617

def g (x : ℝ) : ℝ := (x - 3)^2 - 7

theorem smallest_d_for_injective_g :
  ∀ d : ℝ, (∀ x y, x ≥ d → y ≥ d → g x = g y → x = y) ↔ d ≥ 3 :=
sorry

end smallest_d_for_injective_g_l86_8617


namespace product_expansion_l86_8693

theorem product_expansion (a b c : ℝ) :
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c := by
  sorry

end product_expansion_l86_8693


namespace no_natural_squares_l86_8696

theorem no_natural_squares (x y : ℕ) : ¬(∃ a b : ℕ, x^2 + y = a^2 ∧ y^2 + x = b^2) := by
  sorry

end no_natural_squares_l86_8696


namespace original_savings_calculation_l86_8634

theorem original_savings_calculation (savings : ℝ) : 
  (5/6 : ℝ) * savings + 500 = savings → savings = 3000 := by
  sorry

end original_savings_calculation_l86_8634


namespace circles_intersect_l86_8620

/-- The equation of circle C₁ -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y - 2 = 0

/-- The equation of circle C₂ -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The circles C₁ and C₂ intersect -/
theorem circles_intersect : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y :=
  sorry

end circles_intersect_l86_8620


namespace third_year_percentage_l86_8683

theorem third_year_percentage
  (total : ℝ)
  (third_year : ℝ)
  (second_year : ℝ)
  (h1 : second_year = 0.1 * total)
  (h2 : second_year / (total - third_year) = 1 / 7)
  : third_year = 0.3 * total :=
by sorry

end third_year_percentage_l86_8683


namespace fifteen_hundredth_day_is_wednesday_l86_8691

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek : Type
  | Monday : DayOfWeek
  | Tuesday : DayOfWeek
  | Wednesday : DayOfWeek
  | Thursday : DayOfWeek
  | Friday : DayOfWeek
  | Saturday : DayOfWeek
  | Sunday : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to get the day of the week after n days -/
def dayAfter (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | Nat.succ m => nextDay (dayAfter start m)

theorem fifteen_hundredth_day_is_wednesday :
  dayAfter DayOfWeek.Monday 1499 = DayOfWeek.Wednesday :=
by
  sorry


end fifteen_hundredth_day_is_wednesday_l86_8691


namespace smallest_marble_count_l86_8684

theorem smallest_marble_count : ∃ (M : ℕ), 
  M > 1 ∧
  M % 5 = 1 ∧
  M % 7 = 1 ∧
  M % 11 = 1 ∧
  M % 4 = 2 ∧
  (∀ (N : ℕ), N > 1 ∧ N % 5 = 1 ∧ N % 7 = 1 ∧ N % 11 = 1 ∧ N % 4 = 2 → M ≤ N) ∧
  M = 386 := by
  sorry

end smallest_marble_count_l86_8684


namespace unique_intersection_main_theorem_l86_8678

/-- The curve C generated by rotating P(t, √(2)t^2 - 2t) by 45° anticlockwise -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + 2*p.1*p.2 + p.2^2 - p.1 - 3*p.2 = 0}

/-- The line y = -1/8 -/
def L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -1/8}

/-- The intersection of C and L is a singleton -/
theorem unique_intersection : (C ∩ L).Finite ∧ (C ∩ L).Nonempty := by
  sorry

/-- The main theorem stating that y = -1/8 intersects C at exactly one point -/
theorem main_theorem : ∃! p : ℝ × ℝ, p ∈ C ∧ p ∈ L := by
  sorry

end unique_intersection_main_theorem_l86_8678


namespace liam_strawberry_candies_l86_8686

theorem liam_strawberry_candies :
  ∀ (s g : ℕ),
  s = 3 * g →                     -- Initial condition
  s - 15 = 4 * (g - 15) →         -- Condition after giving away candies
  s = 135 :=                      -- Conclusion to prove
by
  sorry  -- Proof omitted

end liam_strawberry_candies_l86_8686


namespace parabola_x_intercepts_l86_8603

theorem parabola_x_intercepts :
  ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 3 ∧ y = 0 :=
sorry

end parabola_x_intercepts_l86_8603


namespace tangent_problem_l86_8618

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  (1 + Real.tan α) / (1 - Real.tan α) = 3 / 22 := by
  sorry

end tangent_problem_l86_8618


namespace tom_bonus_percentage_l86_8645

theorem tom_bonus_percentage :
  let points_per_enemy : ℕ := 10
  let enemies_killed : ℕ := 150
  let total_score : ℕ := 2250
  let score_without_bonus : ℕ := points_per_enemy * enemies_killed
  let bonus : ℕ := total_score - score_without_bonus
  let bonus_percentage : ℚ := (bonus : ℚ) / (score_without_bonus : ℚ) * 100
  bonus_percentage = 50 := by
sorry

end tom_bonus_percentage_l86_8645


namespace eight_possible_pairs_l86_8674

/-- Represents a seating arrangement at a round table -/
structure RoundTable :=
  (people : Finset (Fin 5))
  (girls : Finset (Fin 5))
  (boys : Finset (Fin 5))
  (all_seated : people = Finset.univ)
  (girls_boys_partition : girls ∪ boys = people ∧ girls ∩ boys = ∅)

/-- The number of people sitting next to at least one girl -/
def g (table : RoundTable) : ℕ := sorry

/-- The number of people sitting next to at least one boy -/
def b (table : RoundTable) : ℕ := sorry

/-- The set of all possible (g,b) pairs for a given round table -/
def possible_pairs (table : RoundTable) : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 = g table ∧ p.2 = b table) (Finset.product (Finset.range 6) (Finset.range 6))

/-- The theorem stating that there are exactly 8 possible (g,b) pairs -/
theorem eight_possible_pairs :
  ∀ table : RoundTable, Finset.card (possible_pairs table) = 8 := sorry

end eight_possible_pairs_l86_8674


namespace fibonacci_rectangle_division_l86_8660

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- A rectangle that can be divided into squares -/
structure DivisibleRectangle where
  width : ℕ
  height : ℕ
  num_squares : ℕ
  max_identical_squares : ℕ

/-- Proposition: For every natural number n, there exists a rectangle with 
    dimensions Fn × Fn+1 that can be divided into exactly n squares, 
    with no more than two squares of the same size -/
theorem fibonacci_rectangle_division (n : ℕ) : 
  ∃ (rect : DivisibleRectangle), 
    rect.width = fib n ∧ 
    rect.height = fib (n + 1) ∧ 
    rect.num_squares = n ∧ 
    rect.max_identical_squares ≤ 2 := by
  sorry

end fibonacci_rectangle_division_l86_8660


namespace kelly_vacation_days_at_sisters_house_l86_8664

/-- Represents Kelly's vacation schedule --/
structure VacationSchedule where
  totalDays : ℕ
  planeTravelDays : ℕ
  grandparentsDays : ℕ
  trainTravelDays : ℕ
  brotherDays : ℕ
  carToSisterDays : ℕ
  busToSisterDays : ℕ
  timeZoneExtraDays : ℕ
  busBackDays : ℕ
  carBackDays : ℕ

/-- Calculates the number of days Kelly spent at her sister's house --/
def daysAtSistersHouse (schedule : VacationSchedule) : ℕ :=
  schedule.totalDays -
  (schedule.planeTravelDays +
   schedule.grandparentsDays +
   schedule.trainTravelDays +
   schedule.brotherDays +
   schedule.carToSisterDays +
   schedule.busToSisterDays +
   schedule.timeZoneExtraDays +
   schedule.busBackDays +
   schedule.carBackDays)

/-- Theorem stating that Kelly spent 3 days at her sister's house --/
theorem kelly_vacation_days_at_sisters_house :
  ∀ (schedule : VacationSchedule),
    schedule.totalDays = 21 ∧
    schedule.planeTravelDays = 2 ∧
    schedule.grandparentsDays = 5 ∧
    schedule.trainTravelDays = 1 ∧
    schedule.brotherDays = 5 ∧
    schedule.carToSisterDays = 1 ∧
    schedule.busToSisterDays = 1 ∧
    schedule.timeZoneExtraDays = 1 ∧
    schedule.busBackDays = 1 ∧
    schedule.carBackDays = 1 →
    daysAtSistersHouse schedule = 3 := by
  sorry

end kelly_vacation_days_at_sisters_house_l86_8664


namespace total_cost_theorem_l86_8695

def original_price : ℝ := 10
def child_discount : ℝ := 0.3
def senior_discount : ℝ := 0.1
def handling_fee : ℝ := 5
def num_child_tickets : ℕ := 2
def num_senior_tickets : ℕ := 2

def senior_ticket_price : ℝ := 14

theorem total_cost_theorem :
  let child_ticket_price := (1 - child_discount) * original_price + handling_fee
  let total_cost := num_child_tickets * child_ticket_price + num_senior_tickets * senior_ticket_price
  total_cost = 52 := by sorry

end total_cost_theorem_l86_8695


namespace student_lecture_choices_l86_8635

/-- The number of different choices when n students can each independently
    choose one of m lectures to attend -/
def number_of_choices (n m : ℕ) : ℕ := m^n

/-- Theorem: Given 5 students and 3 lectures, where each student can independently
    choose one lecture to attend, the total number of different possible choices is 3^5 -/
theorem student_lecture_choices :
  number_of_choices 5 3 = 3^5 := by
  sorry

end student_lecture_choices_l86_8635


namespace divisibility_implies_r_value_l86_8670

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := 10 * x^3 - 5 * x^2 - 52 * x + 56

/-- Divisibility condition -/
def is_divisible_by_square (r : ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - r)^2 * q x

theorem divisibility_implies_r_value :
  ∀ r : ℝ, is_divisible_by_square r → r = 4/3 :=
by sorry

end divisibility_implies_r_value_l86_8670


namespace min_value_of_function_l86_8632

open Real

theorem min_value_of_function (θ : ℝ) (h1 : sin θ ≠ 0) (h2 : cos θ ≠ 0) :
  ∃ (min_val : ℝ), min_val = 5 * sqrt 5 ∧ 
  ∀ θ', sin θ' ≠ 0 → cos θ' ≠ 0 → 1 / sin θ' + 8 / cos θ' ≥ min_val :=
by sorry

end min_value_of_function_l86_8632


namespace f_derivative_at_pi_third_l86_8626

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin x

theorem f_derivative_at_pi_third : 
  (deriv f) (π / 3) = 0 := by sorry

end f_derivative_at_pi_third_l86_8626


namespace order_of_numbers_l86_8659

theorem order_of_numbers (x y z : ℝ) (h1 : 0.9 < x) (h2 : x < 1) 
  (h3 : y = x^x) (h4 : z = x^(x^x)) : x < z ∧ z < y := by
  sorry

end order_of_numbers_l86_8659


namespace sequence_general_term_l86_8673

def S (n : ℕ+) : ℤ := n.val^2 - 2

def a : ℕ+ → ℤ
  | ⟨1, _⟩ => -1
  | ⟨n+2, _⟩ => 2*(n+2) - 1

theorem sequence_general_term (n : ℕ+) : 
  a n = if n = 1 then -1 else S n - S (n-1) := by sorry

end sequence_general_term_l86_8673


namespace problem_proof_l86_8649

theorem problem_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + x*y - 3 = 0) :
  (0 < x*y ∧ x*y ≤ 1) ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b + a*b - 3 = 0 → x + 2*y ≤ a + 2*b) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b + a*b - 3 = 0 ∧ a + 2*b = 4*Real.sqrt 2 - 3) :=
by sorry

end problem_proof_l86_8649


namespace fraction_sum_product_equality_l86_8677

theorem fraction_sum_product_equality : (1 : ℚ) / 2 + ((1 : ℚ) / 2 * (1 : ℚ) / 2) = (3 : ℚ) / 4 := by
  sorry

end fraction_sum_product_equality_l86_8677


namespace nested_fraction_equality_l86_8619

theorem nested_fraction_equality : 
  (((((3 + 2)⁻¹ + 2)⁻¹ + 1)⁻¹ + 2)⁻¹ + 1 : ℚ) = 59 / 43 := by
  sorry

end nested_fraction_equality_l86_8619


namespace all_expressions_correct_l86_8642

theorem all_expressions_correct (x y : ℝ) (h : x / y = 5 / 6) :
  (x + 2*y) / y = 17 / 6 ∧
  (2*x) / (3*y) = 5 / 9 ∧
  (y - x) / (2*y) = 1 / 12 ∧
  (x + y) / (2*y) = 11 / 12 ∧
  x / (y + x) = 5 / 11 := by
  sorry

end all_expressions_correct_l86_8642


namespace division_simplification_l86_8653

theorem division_simplification (a b : ℝ) (h : a ≠ 0) :
  (-4 * a^2 + 12 * a^3 * b) / (-4 * a^2) = 1 - 3 * a * b :=
by sorry

end division_simplification_l86_8653


namespace floor_abs_negative_real_l86_8621

theorem floor_abs_negative_real : ⌊|(-58.7 : ℝ)|⌋ = 58 := by sorry

end floor_abs_negative_real_l86_8621


namespace linear_dependence_iff_k_eq_8_l86_8640

def vector1 : ℝ × ℝ × ℝ := (1, 4, -1)
def vector2 (k : ℝ) : ℝ × ℝ × ℝ := (2, k, 3)

def is_linearly_dependent (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ 
    c1 • v1 + c2 • v2 = (0, 0, 0)

theorem linear_dependence_iff_k_eq_8 :
  ∀ k : ℝ, is_linearly_dependent vector1 (vector2 k) ↔ k = 8 := by
  sorry

end linear_dependence_iff_k_eq_8_l86_8640


namespace smallest_three_digit_divisible_by_5_8_2_l86_8638

theorem smallest_three_digit_divisible_by_5_8_2 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 5 ∣ n ∧ 8 ∣ n ∧ 2 ∣ n → n ≥ 120 :=
by sorry

end smallest_three_digit_divisible_by_5_8_2_l86_8638


namespace cos_identity_l86_8622

theorem cos_identity : Real.cos (70 * π / 180) + 8 * Real.cos (20 * π / 180) * Real.cos (40 * π / 180) * Real.cos (80 * π / 180) = 2 * (Real.cos (35 * π / 180))^2 := by
  sorry

end cos_identity_l86_8622


namespace arrangements_not_adjacent_l86_8665

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n distinct objects, treating two objects as a single unit -/
def arrangements_with_pair (n : ℕ) : ℕ := 2 * factorial (n - 1)

/-- The number of ways to arrange 4 distinct people such that two specific people are not next to each other -/
theorem arrangements_not_adjacent : 
  factorial 4 - arrangements_with_pair 4 = 12 := by sorry

end arrangements_not_adjacent_l86_8665


namespace certain_triangle_angle_sum_l86_8637

/-- A triangle is a shape with three sides -/
structure Triangle where
  sides : Fin 3 → ℝ
  positive : ∀ i, sides i > 0

/-- The sum of interior angles of a triangle is 180° -/
axiom triangle_angle_sum (t : Triangle) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180

/-- For any triangle, the sum of its interior angles is always 180° -/
theorem certain_triangle_angle_sum (t : Triangle) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180 := by
  sorry

end certain_triangle_angle_sum_l86_8637


namespace birds_and_storks_l86_8655

/-- Given a fence with birds and storks, prove that the initial number of birds
    is equal to the initial number of storks plus 3. -/
theorem birds_and_storks (initial_birds initial_storks : ℕ) : 
  initial_storks = 3 → 
  (initial_birds + initial_storks + 2 = initial_birds + 1) → 
  initial_birds = initial_storks + 3 := by
sorry

end birds_and_storks_l86_8655


namespace lines_coplanar_conditions_l86_8685

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  -- This is a placeholder and should be replaced with actual representation
  dummy : Unit

-- Define what it means for three lines to be coplanar
def coplanar (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Add the actual definition here

-- Define the condition of three lines intersecting pairwise but not sharing a common point
def intersect_pairwise_no_common_point (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Add the actual definition here

-- Define the condition of two lines being parallel and the third intersecting both
def two_parallel_one_intersecting (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Add the actual definition here

-- The main theorem
theorem lines_coplanar_conditions (l1 l2 l3 : Line3D) :
  (intersect_pairwise_no_common_point l1 l2 l3 ∨ two_parallel_one_intersecting l1 l2 l3) →
  coplanar l1 l2 l3 := by
  sorry


end lines_coplanar_conditions_l86_8685


namespace exactly_two_red_prob_l86_8699

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def num_draws : ℕ := 4
def num_red_draws : ℕ := 2

def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

theorem exactly_two_red_prob : 
  (Nat.choose num_draws num_red_draws : ℚ) * prob_red ^ num_red_draws * prob_white ^ (num_draws - num_red_draws) = 3456 / 10000 := by
  sorry

end exactly_two_red_prob_l86_8699


namespace square_area_problem_l86_8602

theorem square_area_problem (s : ℝ) : 
  (2 / 5 : ℝ) * s * 10 = 140 → s^2 = 1225 := by
  sorry

end square_area_problem_l86_8602


namespace log_difference_sqrt_l86_8675

theorem log_difference_sqrt (x : ℝ) : 
  x = Real.sqrt (Real.log 8 / Real.log 4 - Real.log 16 / Real.log 8) → x = 1 / Real.sqrt 6 := by
  sorry

end log_difference_sqrt_l86_8675


namespace top_square_is_one_l86_8615

/-- Represents a 4x4 grid of squares --/
def Grid := Fin 4 → Fin 4 → ℕ

/-- Initial configuration of the grid --/
def initial_grid : Grid :=
  λ i j => 4 * i.val + j.val + 1

/-- Fold right half over left half --/
def fold_right_left (g : Grid) : Grid :=
  λ i j => g i (Fin.ofNat (3 - j.val))

/-- Fold left half over right half --/
def fold_left_right (g : Grid) : Grid :=
  λ i j => g i j

/-- Fold top half over bottom half --/
def fold_top_bottom (g : Grid) : Grid :=
  λ i j => g (Fin.ofNat (3 - i.val)) j

/-- Fold bottom half over top half --/
def fold_bottom_top (g : Grid) : Grid :=
  λ i j => g i j

/-- Perform all folds in sequence --/
def perform_folds (g : Grid) : Grid :=
  fold_bottom_top ∘ fold_top_bottom ∘ fold_left_right ∘ fold_right_left $ g

theorem top_square_is_one :
  (perform_folds initial_grid) 0 0 = 1 := by sorry

end top_square_is_one_l86_8615


namespace fish_count_l86_8679

/-- The number of fish per white duck -/
def fish_per_white_duck : ℕ := 5

/-- The number of fish per black duck -/
def fish_per_black_duck : ℕ := 10

/-- The number of fish per multicolor duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- The number of white ducks -/
def white_ducks : ℕ := 3

/-- The number of black ducks -/
def black_ducks : ℕ := 7

/-- The number of multicolor ducks -/
def multicolor_ducks : ℕ := 6

/-- The total number of fish in the lake -/
def total_fish : ℕ := fish_per_white_duck * white_ducks + 
                      fish_per_black_duck * black_ducks + 
                      fish_per_multicolor_duck * multicolor_ducks

theorem fish_count : total_fish = 157 := by
  sorry

end fish_count_l86_8679


namespace total_distance_is_twenty_l86_8692

/-- Represents the travel time per mile for each day -/
def travel_time (day : Nat) : Nat :=
  10 + 6 * (day - 1)

/-- Represents the distance traveled on each day -/
def distance (day : Nat) : Nat :=
  60 / travel_time day

/-- The total distance traveled over 5 days -/
def total_distance : Nat :=
  (List.range 5).map (fun i => distance (i + 1)) |>.sum

/-- Theorem stating that the total distance traveled is 20 miles -/
theorem total_distance_is_twenty : total_distance = 20 := by
  sorry

#eval total_distance

end total_distance_is_twenty_l86_8692


namespace parallel_transitivity_l86_8628

-- Define the type for planes
def Plane : Type := Unit

-- Define the parallel relation between planes
def parallel (p q : Plane) : Prop := sorry

-- State the theorem
theorem parallel_transitivity (α β γ : Plane) 
  (h1 : α ≠ β) (h2 : β ≠ γ) (h3 : α ≠ γ)
  (h4 : parallel α β) (h5 : parallel β γ) : 
  parallel α γ := by sorry

end parallel_transitivity_l86_8628


namespace complex_sum_problem_l86_8666

theorem complex_sum_problem (p r s t u : ℝ) : 
  (∃ q : ℝ, q = 4 ∧ 
   t = -p - r ∧ 
   Complex.I * (q + s + u) = Complex.I * 3) →
  s + u = -1 := by
sorry

end complex_sum_problem_l86_8666


namespace max_valid_triples_l86_8610

/-- A function that checks if four positive integers can be arranged in a circle
    with all neighbors being coprime -/
def can_arrange_coprime (a₁ a₂ a₃ a₄ : ℕ+) : Prop :=
  (Nat.gcd a₁.val a₂.val = 1 ∧ Nat.gcd a₂.val a₃.val = 1 ∧ Nat.gcd a₃.val a₄.val = 1 ∧ Nat.gcd a₄.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₂.val = 1 ∧ Nat.gcd a₂.val a₄.val = 1 ∧ Nat.gcd a₄.val a₃.val = 1 ∧ Nat.gcd a₃.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₃.val = 1 ∧ Nat.gcd a₃.val a₂.val = 1 ∧ Nat.gcd a₂.val a₄.val = 1 ∧ Nat.gcd a₄.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₃.val = 1 ∧ Nat.gcd a₃.val a₄.val = 1 ∧ Nat.gcd a₄.val a₂.val = 1 ∧ Nat.gcd a₂.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₄.val = 1 ∧ Nat.gcd a₄.val a₂.val = 1 ∧ Nat.gcd a₂.val a₃.val = 1 ∧ Nat.gcd a₃.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₄.val = 1 ∧ Nat.gcd a₄.val a₃.val = 1 ∧ Nat.gcd a₃.val a₂.val = 1 ∧ Nat.gcd a₂.val a₁.val = 1)

/-- A function that counts the number of valid triples (i,j,k) where (gcd(aᵢ,a_j))² | a_k -/
def count_valid_triples (a₁ a₂ a₃ a₄ : ℕ+) : ℕ :=
  let check (i j k : ℕ+) : Bool :=
    i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ (Nat.gcd i.val j.val)^2 ∣ k.val
  (if check a₁ a₂ a₃ then 1 else 0) +
  (if check a₁ a₂ a₄ then 1 else 0) +
  (if check a₁ a₃ a₂ then 1 else 0) +
  (if check a₁ a₃ a₄ then 1 else 0) +
  (if check a₁ a₄ a₂ then 1 else 0) +
  (if check a₁ a₄ a₃ then 1 else 0) +
  (if check a₂ a₃ a₁ then 1 else 0) +
  (if check a₂ a₃ a₄ then 1 else 0) +
  (if check a₂ a₄ a₁ then 1 else 0) +
  (if check a₂ a₄ a₃ then 1 else 0) +
  (if check a₃ a₄ a₁ then 1 else 0) +
  (if check a₃ a₄ a₂ then 1 else 0)

theorem max_valid_triples (a₁ a₂ a₃ a₄ : ℕ+) :
  ¬(can_arrange_coprime a₁ a₂ a₃ a₄) → count_valid_triples a₁ a₂ a₃ a₄ ≤ 8 :=
by sorry

end max_valid_triples_l86_8610


namespace oliver_final_balance_l86_8606

def olivers_money (initial_amount savings chores_earnings frisbee_cost puzzle_cost stickers_cost
                   movie_ticket_cost snack_cost birthday_gift : ℤ) : ℤ :=
  initial_amount + savings + chores_earnings - frisbee_cost - puzzle_cost - stickers_cost -
  movie_ticket_cost - snack_cost + birthday_gift

theorem oliver_final_balance :
  olivers_money 9 5 6 4 3 2 7 3 8 = 9 := by
  sorry

end oliver_final_balance_l86_8606


namespace coeff_x4_is_zero_l86_8698

/-- The coefficient of x^4 in the expansion of (x+2)(x-1)^5 -/
def coeff_x4 (x : ℝ) : ℝ :=
  let expansion := (x + 2) * (x - 1)^5
  sorry

theorem coeff_x4_is_zero :
  coeff_x4 x = 0 := by sorry

end coeff_x4_is_zero_l86_8698


namespace decimal_8543_to_base7_l86_8601

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: decimal_to_base7 (n / 7)

def base7_to_decimal (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 7 * acc) 0

theorem decimal_8543_to_base7 :
  decimal_to_base7 8543 = [3, 2, 6, 3, 3] ∧
  base7_to_decimal [3, 2, 6, 3, 3] = 8543 := by
  sorry

end decimal_8543_to_base7_l86_8601


namespace geometric_sequence_sum_l86_8689

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 1 * a 3 + 2 * a 2 * a 4 + a 2 * a 6 = 9 →
  a 2 + a 4 = 3 := by
sorry

end geometric_sequence_sum_l86_8689


namespace two_in_A_implies_a_eq_two_A_eq_B_implies_a_eq_one_A_intersection_B_eq_A_implies_a_eq_one_or_four_l86_8644

-- Define set A
def A (a : ℝ) : Set ℝ := {x | x^2 + 4*a = (a + 4)*x}

-- Define set B
def B : Set ℝ := {x | x^2 + 4 = 5*x}

-- Theorem 1
theorem two_in_A_implies_a_eq_two :
  ∀ a : ℝ, 2 ∈ A a → a = 2 := by sorry

-- Theorem 2
theorem A_eq_B_implies_a_eq_one :
  ∀ a : ℝ, A a = B → a = 1 := by sorry

-- Theorem 3
theorem A_intersection_B_eq_A_implies_a_eq_one_or_four :
  ∀ a : ℝ, A a ∩ B = A a → a = 1 ∨ a = 4 := by sorry

end two_in_A_implies_a_eq_two_A_eq_B_implies_a_eq_one_A_intersection_B_eq_A_implies_a_eq_one_or_four_l86_8644


namespace hyperbola_properties_l86_8609

/-- Given a hyperbola with standard equation x²/36 - y²/64 = 1, 
    prove its asymptote equations and eccentricity. -/
theorem hyperbola_properties :
  let a : ℝ := 6
  let b : ℝ := 8
  let c : ℝ := (a^2 + b^2).sqrt
  let asymptote (x : ℝ) : ℝ := (b / a) * x
  let eccentricity : ℝ := c / a
  (∀ x y : ℝ, x^2 / 36 - y^2 / 64 = 1 → 
    (y = asymptote x ∨ y = -asymptote x) ∧ eccentricity = 5/3) :=
by sorry

end hyperbola_properties_l86_8609


namespace workshop_workers_l86_8614

theorem workshop_workers (total_average : ℕ) (technician_average : ℕ) (other_average : ℕ) 
  (technician_count : ℕ) :
  total_average = 8000 →
  technician_average = 16000 →
  other_average = 6000 →
  technician_count = 7 →
  ∃ (total_workers : ℕ), 
    total_workers * total_average = 
      technician_count * technician_average + (total_workers - technician_count) * other_average ∧
    total_workers = 35 :=
by sorry

end workshop_workers_l86_8614


namespace inequality_system_solution_l86_8650

theorem inequality_system_solution (x : ℝ) :
  x - 2 ≤ 0 ∧ (x - 1) / 2 < x → -1 < x ∧ x ≤ 2 := by
  sorry

end inequality_system_solution_l86_8650
