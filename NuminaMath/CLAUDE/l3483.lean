import Mathlib

namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l3483_348311

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 160 ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 160 := by sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l3483_348311


namespace NUMINAMATH_CALUDE_simple_interest_from_compound_l3483_348380

/-- Given an initial investment, interest rate, time period, and compound interest earned,
    calculate the simple interest earned. -/
theorem simple_interest_from_compound (principal : ℝ) (rate : ℝ) (time : ℝ) (compound_interest : ℝ)
  (h1 : principal = 6000)
  (h2 : time = 2)
  (h3 : principal * (1 + rate)^time - principal = compound_interest)
  (h4 : compound_interest = 615) :
  principal * rate * time = 1200 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_from_compound_l3483_348380


namespace NUMINAMATH_CALUDE_sqrt_360000_l3483_348321

theorem sqrt_360000 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_l3483_348321


namespace NUMINAMATH_CALUDE_distribute_6_4_l3483_348304

/-- The number of ways to distribute n identical objects among k classes,
    with each class receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- Theorem stating that there are 10 ways to distribute 6 spots among 4 classes,
    with each class receiving at least one spot. -/
theorem distribute_6_4 : distribute 6 4 = 10 := by sorry

end NUMINAMATH_CALUDE_distribute_6_4_l3483_348304


namespace NUMINAMATH_CALUDE_additional_interest_rate_proof_l3483_348333

/-- Proves that given specific investment conditions, the additional interest rate must be 8% --/
theorem additional_interest_rate_proof (initial_investment : ℝ) (initial_rate : ℝ) 
  (total_rate : ℝ) (additional_investment : ℝ) : 
  initial_investment = 2400 →
  initial_rate = 0.04 →
  total_rate = 0.06 →
  additional_investment = 2400 →
  (initial_investment * initial_rate + additional_investment * 0.08) / 
    (initial_investment + additional_investment) = total_rate :=
by sorry

end NUMINAMATH_CALUDE_additional_interest_rate_proof_l3483_348333


namespace NUMINAMATH_CALUDE_min_distance_between_ships_l3483_348331

/-- The minimum distance between two ships given specific conditions -/
theorem min_distance_between_ships 
  (d : ℝ) -- Initial distance between ships
  (k : ℝ) -- Speed ratio v₁/v₂
  (h₁ : k > 0) -- Speed ratio is positive
  (h₂ : k < 1) -- Speed ratio is less than 1
  : ∃ (min_dist : ℝ), min_dist = d * Real.sqrt (1 - k^2) :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_ships_l3483_348331


namespace NUMINAMATH_CALUDE_max_stamps_for_50_dollars_max_stamps_is_maximum_l3483_348354

/-- The maximum number of stamps that can be purchased with a given budget and stamp price. -/
def maxStamps (budget : ℕ) (stampPrice : ℕ) : ℕ :=
  (budget / stampPrice : ℕ)

/-- Theorem stating the maximum number of stamps that can be purchased with $50 when each stamp costs 45 cents. -/
theorem max_stamps_for_50_dollars : maxStamps 5000 45 = 111 := by
  sorry

/-- Proof that the calculated maximum is indeed the largest possible number of stamps. -/
theorem max_stamps_is_maximum (budget : ℕ) (stampPrice : ℕ) :
  ∀ n : ℕ, n * stampPrice ≤ budget → n ≤ maxStamps budget stampPrice := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_for_50_dollars_max_stamps_is_maximum_l3483_348354


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_product_squares_l3483_348397

theorem consecutive_odd_numbers_sum_product_squares : 
  ∃ (a : ℤ), 
    let sequence := List.range 25 |>.map (λ i => a + 2*i - 24)
    ∃ (s p : ℤ), 
      (sequence.sum = s^2) ∧ 
      (sequence.prod = p^2) ∧
      (∀ n ∈ sequence, n % 2 = 1 ∨ n % 2 = -1) := by
  sorry

#check consecutive_odd_numbers_sum_product_squares

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_product_squares_l3483_348397


namespace NUMINAMATH_CALUDE_arrangements_count_is_correct_l3483_348338

/-- The number of arrangements of 4 boys and 3 girls in a row,
    where exactly two girls are standing next to each other. -/
def arrangements_count : ℕ := 2880

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 3

/-- Theorem stating that the number of arrangements of 4 boys and 3 girls in a row,
    where exactly two girls are standing next to each other, is equal to 2880. -/
theorem arrangements_count_is_correct :
  arrangements_count = num_girls * (num_girls - 1) / 2 * 
    (num_boys * (num_boys - 1) * (num_boys - 2) * (num_boys - 3)) *
    ((num_boys + 1) * num_boys) :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_is_correct_l3483_348338


namespace NUMINAMATH_CALUDE_final_average_is_23_l3483_348390

/-- Represents a cricketer's scoring data -/
structure CricketerData where
  inningsCount : ℕ
  scoreLastInning : ℕ
  averageIncrease : ℕ

/-- Calculates the final average score given the cricketer's data -/
def finalAverageScore (data : CricketerData) : ℕ :=
  data.averageIncrease + (data.scoreLastInning - data.averageIncrease * data.inningsCount) / (data.inningsCount - 1)

/-- Theorem stating that for the given conditions, the final average score is 23 -/
theorem final_average_is_23 (data : CricketerData) 
  (h1 : data.inningsCount = 19)
  (h2 : data.scoreLastInning = 95)
  (h3 : data.averageIncrease = 4) : 
  finalAverageScore data = 23 := by
  sorry

end NUMINAMATH_CALUDE_final_average_is_23_l3483_348390


namespace NUMINAMATH_CALUDE_triangle_perimeter_theorem_l3483_348368

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the perimeter function
def perimeter (t : Triangle) : ℝ := sorry

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define the ray function
def ray (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define the intersection function
def intersect (s₁ s₂ : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

theorem triangle_perimeter_theorem (ABC : Triangle) (X Y M : ℝ × ℝ) :
  perimeter ABC = 4 →
  X ∈ ray ABC.A ABC.B →
  Y ∈ ray ABC.A ABC.C →
  distance ABC.A X = 1 →
  distance ABC.A Y = 1 →
  M ∈ intersect (Set.Icc ABC.B ABC.C) (Set.Icc X Y) →
  (perimeter ⟨ABC.A, ABC.B, M⟩ = 2 ∨ perimeter ⟨ABC.A, ABC.C, M⟩ = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_theorem_l3483_348368


namespace NUMINAMATH_CALUDE_trig_identities_l3483_348317

theorem trig_identities (α : Real) (h : Real.tan α = 3) : 
  ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) ∧
  (1 / (Real.sin α ^ 2 - Real.sin α * Real.cos α - 2 * Real.cos α ^ 2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l3483_348317


namespace NUMINAMATH_CALUDE_cycling_trip_distances_l3483_348377

-- Define the total route distance
def total_distance : ℝ := 120

-- Define the distances traveled each day
def day1_distance : ℝ := 36
def day2_distance : ℝ := 40
def day3_distance : ℝ := 44

-- Theorem statement
theorem cycling_trip_distances :
  -- Day 1 condition
  day1_distance = total_distance / 3 - 4 ∧
  -- Day 2 condition
  day2_distance = (total_distance - day1_distance) / 2 - 2 ∧
  -- Day 3 condition
  day3_distance = (total_distance - day1_distance - day2_distance) * 10 / 11 + 4 ∧
  -- Total distance is the sum of all days
  total_distance = day1_distance + day2_distance + day3_distance :=
by sorry


end NUMINAMATH_CALUDE_cycling_trip_distances_l3483_348377


namespace NUMINAMATH_CALUDE_period_length_proof_l3483_348396

/-- Calculates the length of each period given the number of students, presentation time per student, and number of periods. -/
def period_length (num_students : ℕ) (presentation_time : ℕ) (num_periods : ℕ) : ℕ :=
  (num_students * presentation_time) / num_periods

/-- Proves that given 32 students, 5 minutes per presentation, and 4 periods, the length of each period is 40 minutes. -/
theorem period_length_proof :
  period_length 32 5 4 = 40 := by
  sorry

#eval period_length 32 5 4

end NUMINAMATH_CALUDE_period_length_proof_l3483_348396


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3483_348392

/-- Given a cylinder with volume 72π cm³ and height twice its radius,
    prove that a cone with the same height and radius has a volume of 24π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  (π * r^2 * h = 72 * π) →   -- Cylinder volume condition
  (h = 2 * r) →              -- Height-radius relation condition
  ((1/3) * π * r^2 * h = 24 * π) -- Cone volume to prove
  :=
by
  sorry


end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3483_348392


namespace NUMINAMATH_CALUDE_cupcake_production_difference_l3483_348362

def cupcake_difference (betty_rate : ℕ) (dora_rate : ℕ) (total_time : ℕ) (break_time : ℕ) : ℕ :=
  (dora_rate * total_time) - (betty_rate * (total_time - break_time))

theorem cupcake_production_difference :
  cupcake_difference 10 8 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_production_difference_l3483_348362


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3483_348342

theorem trig_identity_proof : 
  Real.sin (17 * π / 180) * Real.cos (43 * π / 180) + 
  Real.sin (73 * π / 180) * Real.sin (43 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3483_348342


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l3483_348386

/-- The orthocenter of a triangle ABC --/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates --/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (4, 1, 1)
  let C : ℝ × ℝ × ℝ := (1, 5, 6)
  orthocenter A B C = (-79/3, 91/3, 41/3) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l3483_348386


namespace NUMINAMATH_CALUDE_bianca_carrots_l3483_348340

def carrot_problem (initial : ℕ) (thrown_out : ℕ) (additional : ℕ) : ℕ :=
  initial - thrown_out + additional

theorem bianca_carrots : carrot_problem 23 10 47 = 60 := by
  sorry

end NUMINAMATH_CALUDE_bianca_carrots_l3483_348340


namespace NUMINAMATH_CALUDE_store_customers_l3483_348336

/-- Proves that the number of customers is 1000 given the specified conditions --/
theorem store_customers (return_rate : ℝ) (book_price : ℝ) (final_sales : ℝ) :
  return_rate = 0.37 →
  book_price = 15 →
  final_sales = 9450 →
  (1 - return_rate) * book_price * (final_sales / ((1 - return_rate) * book_price)) = 1000 := by
sorry

#eval (1 - 0.37) * 15 * (9450 / ((1 - 0.37) * 15)) -- Should output 1000.0

end NUMINAMATH_CALUDE_store_customers_l3483_348336


namespace NUMINAMATH_CALUDE_yoque_monthly_payment_l3483_348319

/-- Calculates the monthly payment for a loan with interest -/
def monthly_payment (principal : ℚ) (months : ℕ) (interest_rate : ℚ) : ℚ :=
  (principal * (1 + interest_rate)) / months

/-- Proves that the monthly payment is $15 given the problem conditions -/
theorem yoque_monthly_payment :
  let principal : ℚ := 150
  let months : ℕ := 11
  let interest_rate : ℚ := 1/10
  monthly_payment principal months interest_rate = 15 := by
sorry

end NUMINAMATH_CALUDE_yoque_monthly_payment_l3483_348319


namespace NUMINAMATH_CALUDE_sum_of_roots_is_nine_l3483_348346

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the symmetry property of f
def is_symmetric_about_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (3 + x) = f (3 - x)

-- Define a property for f having exactly three distinct real roots
def has_three_distinct_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ x y z : ℝ, (f x = 0 ∧ f y = 0 ∧ f z = 0) ∧ 
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
  (∀ w : ℝ, f w = 0 → w = x ∨ w = y ∨ w = z)

-- Theorem statement
theorem sum_of_roots_is_nine (f : ℝ → ℝ) 
  (h1 : is_symmetric_about_3 f) 
  (h2 : has_three_distinct_real_roots f) : 
  ∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0 ∧ x + y + z = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_nine_l3483_348346


namespace NUMINAMATH_CALUDE_tiffany_treasures_l3483_348361

theorem tiffany_treasures (points_per_treasure : ℕ) (first_level_treasures : ℕ) (total_score : ℕ) :
  points_per_treasure = 6 →
  first_level_treasures = 3 →
  total_score = 48 →
  (total_score - points_per_treasure * first_level_treasures) / points_per_treasure = 5 :=
by sorry

end NUMINAMATH_CALUDE_tiffany_treasures_l3483_348361


namespace NUMINAMATH_CALUDE_boys_dropped_out_l3483_348382

/-- Proves the number of boys who dropped out from a school, given initial counts and final total -/
theorem boys_dropped_out (initial_boys initial_girls girls_dropped final_total : ℕ) : 
  initial_boys = 14 →
  initial_girls = 10 →
  girls_dropped = 3 →
  final_total = 17 →
  initial_boys - (final_total - (initial_girls - girls_dropped)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_boys_dropped_out_l3483_348382


namespace NUMINAMATH_CALUDE_sum_of_four_digit_numbers_l3483_348352

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A four-digit number formed from the given digits -/
structure FourDigitNumber where
  d₁ : Nat
  d₂ : Nat
  d₃ : Nat
  d₄ : Nat
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  distinct : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄

/-- The value of a four-digit number -/
def value (n : FourDigitNumber) : Nat :=
  1000 * n.d₁ + 100 * n.d₂ + 10 * n.d₃ + n.d₄

/-- The set of all valid four-digit numbers -/
def allFourDigitNumbers : Finset FourDigitNumber :=
  sorry

/-- The theorem to be proved -/
theorem sum_of_four_digit_numbers :
  (allFourDigitNumbers.sum value) = 399960 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_numbers_l3483_348352


namespace NUMINAMATH_CALUDE_correct_number_of_hens_l3483_348355

/-- Given a total number of animals and feet, calculate the number of hens -/
def number_of_hens (total_animals : ℕ) (total_feet : ℕ) : ℕ :=
  2 * total_animals - total_feet / 2

theorem correct_number_of_hens :
  let total_animals := 46
  let total_feet := 140
  number_of_hens total_animals total_feet = 22 := by
  sorry

#eval number_of_hens 46 140

end NUMINAMATH_CALUDE_correct_number_of_hens_l3483_348355


namespace NUMINAMATH_CALUDE_base7_146_equals_83_l3483_348370

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base7_146_equals_83 :
  base7ToBase10 [6, 4, 1] = 83 := by sorry

end NUMINAMATH_CALUDE_base7_146_equals_83_l3483_348370


namespace NUMINAMATH_CALUDE_center_coordinates_sum_l3483_348314

/-- Given a circle with diameter endpoints (4, -7) and (-8, 5), 
    the sum of the coordinates of the center of the circle is -3. -/
theorem center_coordinates_sum (c : ℝ × ℝ) : 
  (∃ (r : ℝ), (c.1 - 4)^2 + (c.2 + 7)^2 = r^2 ∧ (c.1 + 8)^2 + (c.2 - 5)^2 = r^2) →
  c.1 + c.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_center_coordinates_sum_l3483_348314


namespace NUMINAMATH_CALUDE_sum_of_complex_sequence_l3483_348325

theorem sum_of_complex_sequence : 
  let n : ℕ := 150
  let a₀ : ℤ := -74
  let b₀ : ℤ := 30
  let d : ℤ := 1
  let sum : ℂ := (↑n / 2 : ℚ) * ↑(2 * a₀ + (n - 1) * d) + 
                 (↑n / 2 : ℚ) * ↑(2 * b₀ + (n - 1) * d) * Complex.I
  sum = 75 + 15675 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_sum_of_complex_sequence_l3483_348325


namespace NUMINAMATH_CALUDE_num_small_orders_l3483_348353

def large_order_weight : ℕ := 200
def small_order_weight : ℕ := 50
def total_weight_used : ℕ := 800
def num_large_orders : ℕ := 3

theorem num_small_orders : 
  (total_weight_used - num_large_orders * large_order_weight) / small_order_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_num_small_orders_l3483_348353


namespace NUMINAMATH_CALUDE_school_A_percentage_l3483_348371

theorem school_A_percentage (total : ℕ) (science_percent : ℚ) (non_science : ℕ) :
  total = 300 →
  science_percent = 30 / 100 →
  non_science = 42 →
  ∃ (school_A_percent : ℚ),
    school_A_percent = 20 / 100 ∧
    non_science = (1 - science_percent) * (school_A_percent * total) :=
by sorry

end NUMINAMATH_CALUDE_school_A_percentage_l3483_348371


namespace NUMINAMATH_CALUDE_fraction_increase_possible_l3483_348341

theorem fraction_increase_possible : ∃ (a b : ℕ+), (a + 1 : ℚ) / (b + 100) > (a : ℚ) / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_increase_possible_l3483_348341


namespace NUMINAMATH_CALUDE_south_five_is_negative_five_l3483_348335

/-- Represents the direction of movement -/
inductive Direction
| North
| South

/-- Represents a movement with magnitude and direction -/
structure Movement where
  magnitude : ℕ
  direction : Direction

/-- Function to convert a movement to its signed representation -/
def movementToSigned (m : Movement) : ℤ :=
  match m.direction with
  | Direction.North => m.magnitude
  | Direction.South => -m.magnitude

theorem south_five_is_negative_five :
  let southFive : Movement := ⟨5, Direction.South⟩
  movementToSigned southFive = -5 := by sorry

end NUMINAMATH_CALUDE_south_five_is_negative_five_l3483_348335


namespace NUMINAMATH_CALUDE_exam_grade_logic_l3483_348369

theorem exam_grade_logic 
  (student : Type) 
  (received_A : student → Prop)
  (all_mc_correct : student → Prop)
  (problem_solving_90_percent : student → Prop)
  (h : ∀ s : student, (all_mc_correct s ∨ problem_solving_90_percent s) → received_A s) :
  ∀ s : student, ¬(received_A s) → (¬(all_mc_correct s) ∧ ¬(problem_solving_90_percent s)) :=
by sorry

end NUMINAMATH_CALUDE_exam_grade_logic_l3483_348369


namespace NUMINAMATH_CALUDE_c_share_value_l3483_348358

/-- Proves that given the conditions, c's share is 398.75 -/
theorem c_share_value (total : ℚ) (a b c d : ℚ) : 
  total = 1500 →
  5/2 * a = 7/3 * b →
  5/2 * a = 2 * c →
  5/2 * a = 11/6 * d →
  a + b + c + d = total →
  c = 398.75 := by
sorry

end NUMINAMATH_CALUDE_c_share_value_l3483_348358


namespace NUMINAMATH_CALUDE_tangent_line_at_one_symmetry_condition_extreme_values_condition_l3483_348302

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

-- State the theorems
theorem tangent_line_at_one (a : ℝ) :
  a = -1 → ∃ m b, ∀ x, f a x = m * (x - 1) + b ∧ m = -Real.log 2 ∧ b = 0 := by sorry

theorem symmetry_condition (a : ℝ) :
  (∀ x > 0, f a (1/x) = f a (1/(-2 * x))) ↔ a = 1/2 := by sorry

theorem extreme_values_condition (a : ℝ) :
  (∃ x > 0, ∀ y > 0, f a x ≥ f a y ∨ f a x ≤ f a y) ↔ 0 < a ∧ a < 1/2 := by sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_one_symmetry_condition_extreme_values_condition_l3483_348302


namespace NUMINAMATH_CALUDE_round_trip_distance_bike_ride_distance_l3483_348360

/-- Calculates the total distance traveled in a round trip given speeds and total time -/
theorem round_trip_distance (speed_out speed_back total_time : ℝ) 
  (h1 : speed_out > 0) 
  (h2 : speed_back > 0) 
  (h3 : total_time > 0) : ℝ :=
  let one_way_distance := (speed_out * speed_back * total_time) / (speed_out + speed_back)
  2 * one_way_distance

/-- Proves that for the given speeds and time, the total distance is 144 miles -/
theorem bike_ride_distance : 
  round_trip_distance 24 18 7 (by norm_num) (by norm_num) (by norm_num) = 144 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_distance_bike_ride_distance_l3483_348360


namespace NUMINAMATH_CALUDE_unknown_number_proof_l3483_348343

theorem unknown_number_proof (n : ℕ) 
  (h1 : Nat.lcm 24 n = 168) 
  (h2 : Nat.gcd 24 n = 4) : 
  n = 28 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l3483_348343


namespace NUMINAMATH_CALUDE_sarah_bowled_160_l3483_348337

def sarahs_score (gregs_score : ℕ) : ℕ := gregs_score + 60

theorem sarah_bowled_160 (gregs_score : ℕ) :
  sarahs_score gregs_score = 160 ∧ 
  (sarahs_score gregs_score + gregs_score) / 2 = 130 :=
by sorry

end NUMINAMATH_CALUDE_sarah_bowled_160_l3483_348337


namespace NUMINAMATH_CALUDE_two_numbers_theorem_l3483_348315

theorem two_numbers_theorem :
  ∃ (x y : ℕ),
    x + y = 1244 ∧
    10 * x + 3 = (y - 2) / 10 ∧
    x = 12 ∧
    y = 1232 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_theorem_l3483_348315


namespace NUMINAMATH_CALUDE_zachary_cans_l3483_348310

def can_sequence (n : ℕ) : ℕ := 4 + 5 * (n - 1)

theorem zachary_cans : can_sequence 7 = 34 := by
  sorry

end NUMINAMATH_CALUDE_zachary_cans_l3483_348310


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3483_348357

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3483_348357


namespace NUMINAMATH_CALUDE_product_of_sums_equal_difference_of_powers_l3483_348391

theorem product_of_sums_equal_difference_of_powers : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equal_difference_of_powers_l3483_348391


namespace NUMINAMATH_CALUDE_equation_solutions_l3483_348305

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  log (x^2 + 1) - 2 * log (x + 3) + log 2 = 0

-- State the theorem
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = -1 ∨ x = 7) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3483_348305


namespace NUMINAMATH_CALUDE_triangle_count_is_83_l3483_348376

/-- Represents a rectangle divided into triangles -/
structure DividedRectangle where
  width : ℕ
  height : ℕ
  horizontal_divisions : ℕ
  vertical_divisions : ℕ

/-- Counts the number of triangles in the divided rectangle -/
def count_triangles (r : DividedRectangle) : ℕ :=
  sorry

/-- The specific rectangle from the problem -/
def problem_rectangle : DividedRectangle :=
  { width := 4
  , height := 5
  , horizontal_divisions := 4
  , vertical_divisions := 5 }

theorem triangle_count_is_83 : 
  count_triangles problem_rectangle = 83 :=
sorry

end NUMINAMATH_CALUDE_triangle_count_is_83_l3483_348376


namespace NUMINAMATH_CALUDE_difference_between_decimal_and_fraction_l3483_348388

theorem difference_between_decimal_and_fraction : 0.127 - (1 / 8 : ℚ) = 0.002 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_decimal_and_fraction_l3483_348388


namespace NUMINAMATH_CALUDE_tree_planting_multiple_l3483_348379

/-- The number of trees planted by 4th graders -/
def trees_4th : ℕ := 30

/-- The number of trees planted by 5th graders -/
def trees_5th : ℕ := 2 * trees_4th

/-- The total number of trees planted by all grades -/
def total_trees : ℕ := 240

/-- The multiple of 5th graders' trees compared to 6th graders' trees -/
def m : ℕ := 3

/-- Theorem stating that m is the correct multiple -/
theorem tree_planting_multiple :
  m * trees_5th - 30 = total_trees - trees_4th - trees_5th := by
  sorry

#check tree_planting_multiple

end NUMINAMATH_CALUDE_tree_planting_multiple_l3483_348379


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3483_348351

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -1 ∧ 
  (x₁^2 - 2*x₁ - 3 = 0) ∧ (x₂^2 - 2*x₂ - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3483_348351


namespace NUMINAMATH_CALUDE_count_with_zero_eq_952_l3483_348320

/-- A function that checks if a positive integer contains the digit 0 in its base-ten representation -/
def containsZero (n : ℕ+) : Bool :=
  sorry

/-- The count of positive integers less than or equal to 2500 that contain the digit 0 -/
def countWithZero : ℕ :=
  sorry

/-- Theorem stating that the count of positive integers less than or equal to 2500 
    containing the digit 0 is 952 -/
theorem count_with_zero_eq_952 : countWithZero = 952 := by
  sorry

end NUMINAMATH_CALUDE_count_with_zero_eq_952_l3483_348320


namespace NUMINAMATH_CALUDE_negation_equivalence_l3483_348374

def exactly_one_even (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 = 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 = 0)

def at_least_two_even_or_all_odd (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 = 0) ∨
  (a % 2 = 0 ∧ c % 2 = 0) ∨
  (b % 2 = 0 ∧ c % 2 = 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0)

theorem negation_equivalence (a b c : ℕ) :
  ¬(exactly_one_even a b c) ↔ at_least_two_even_or_all_odd a b c :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3483_348374


namespace NUMINAMATH_CALUDE_jills_shopping_breakdown_l3483_348323

/-- Represents the shopping breakdown and tax calculation for Jill's purchase --/
theorem jills_shopping_breakdown (T : ℝ) (x : ℝ) 
  (h1 : T > 0) -- Total amount spent is positive
  (h2 : x ≥ 0 ∧ x ≤ 1) -- Percentage spent on other items is between 0 and 100%
  (h3 : 0.5 + 0.2 + x = 1) -- Total percentage spent is 100%
  (h4 : 0.02 * T + 0.1 * x * T = 0.05 * T) -- Tax equation
  : x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_jills_shopping_breakdown_l3483_348323


namespace NUMINAMATH_CALUDE_pen_selling_problem_l3483_348322

/-- Proves that given the conditions of the pen selling problem, the initial number of pens purchased is 30 -/
theorem pen_selling_problem (n : ℕ) (P : ℝ) (h1 : P > 0) :
  (∃ (S : ℝ), S > 0 ∧ 20 * S = P ∧ n * (2/3 * S) = P) →
  n = 30 := by
  sorry

end NUMINAMATH_CALUDE_pen_selling_problem_l3483_348322


namespace NUMINAMATH_CALUDE_blue_folder_stickers_l3483_348395

/-- The number of stickers on each sheet in the blue folder -/
def blue_stickers_per_sheet : ℕ :=
  let total_stickers : ℕ := 60
  let sheets_per_folder : ℕ := 10
  let red_stickers_per_sheet : ℕ := 3
  let green_stickers_per_sheet : ℕ := 2
  let red_total := sheets_per_folder * red_stickers_per_sheet
  let green_total := sheets_per_folder * green_stickers_per_sheet
  let blue_total := total_stickers - red_total - green_total
  blue_total / sheets_per_folder

theorem blue_folder_stickers :
  blue_stickers_per_sheet = 1 := by
  sorry

end NUMINAMATH_CALUDE_blue_folder_stickers_l3483_348395


namespace NUMINAMATH_CALUDE_max_time_digit_sum_l3483_348300

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours ≤ 23
  minutes_valid : minutes ≤ 59

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def timeDigitSum (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum possible sum of digits for any Time24 -/
def maxTimeDigitSum : Nat := 24

theorem max_time_digit_sum :
  ∀ t : Time24, timeDigitSum t ≤ maxTimeDigitSum :=
by sorry

end NUMINAMATH_CALUDE_max_time_digit_sum_l3483_348300


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l3483_348347

-- Define the population
structure Population where
  grades : List String
  students : List String

-- Define the sampling methods
inductive SamplingMethod
  | DrawingLots
  | RandomNumber
  | Stratified
  | Systematic

-- Define the survey requirements
structure SurveyRequirements where
  proportional_sampling : Bool
  multiple_grades : Bool

-- Define a function to determine the most appropriate sampling method
def most_appropriate_method (pop : Population) (req : SurveyRequirements) : SamplingMethod :=
  sorry

-- Theorem stating that stratified sampling is most appropriate
-- for a population with multiple grades and proportional sampling requirement
theorem stratified_sampling_most_appropriate 
  (pop : Population) 
  (req : SurveyRequirements) :
  pop.grades.length > 1 → 
  req.proportional_sampling = true → 
  req.multiple_grades = true → 
  most_appropriate_method pop req = SamplingMethod.Stratified :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l3483_348347


namespace NUMINAMATH_CALUDE_product_evaluation_l3483_348339

theorem product_evaluation (n : ℕ) (h : n = 4) : n * (n + 1) * (n + 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3483_348339


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3483_348318

theorem quadratic_roots_sum (α β : ℝ) : 
  α^2 + 2*α - 2005 = 0 → β^2 + 2*β - 2005 = 0 → α^2 + 3*α + β = 2003 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3483_348318


namespace NUMINAMATH_CALUDE_amoebas_after_two_weeks_l3483_348309

/-- The number of amoebas in the tank on a given day -/
def amoebas (day : ℕ) : ℕ :=
  if day ≤ 7 then
    2^day
  else
    2^7 * 3^(day - 7)

/-- Theorem stating the number of amoebas after 14 days -/
theorem amoebas_after_two_weeks : amoebas 14 = 279936 := by
  sorry

end NUMINAMATH_CALUDE_amoebas_after_two_weeks_l3483_348309


namespace NUMINAMATH_CALUDE_chessboard_tiling_exists_l3483_348385

/-- Represents a chessboard of size 2^n x 2^n -/
structure Chessboard (n : ℕ) where
  size : Fin (2^n) × Fin (2^n)

/-- Represents an L-shaped triomino -/
inductive Triomino
  | L : Triomino

/-- Represents a tiling of a chessboard -/
def Tiling (n : ℕ) := Chessboard n → Option Triomino

/-- States that a tiling is valid for a chessboard with one square removed -/
def is_valid_tiling (n : ℕ) (t : Tiling n) (removed : Fin (2^n) × Fin (2^n)) : Prop :=
  ∀ (pos : Fin (2^n) × Fin (2^n)), pos ≠ removed → t ⟨pos⟩ = some Triomino.L

/-- Theorem: For any 2^n x 2^n chessboard with one square removed, 
    there exists a valid tiling using L-shaped triominoes -/
theorem chessboard_tiling_exists (n : ℕ) (removed : Fin (2^n) × Fin (2^n)) :
  ∃ (t : Tiling n), is_valid_tiling n t removed := by
  sorry

end NUMINAMATH_CALUDE_chessboard_tiling_exists_l3483_348385


namespace NUMINAMATH_CALUDE_ascending_order_proof_l3483_348381

def base_16_to_decimal (n : ℕ) : ℕ := n

def base_7_to_decimal (n : ℕ) : ℕ := n

def base_4_to_decimal (n : ℕ) : ℕ := n

theorem ascending_order_proof (a b c : ℕ) 
  (ha : a = base_16_to_decimal 0x12)
  (hb : b = base_7_to_decimal 25)
  (hc : c = base_4_to_decimal 33) :
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_proof_l3483_348381


namespace NUMINAMATH_CALUDE_disk_color_difference_l3483_348327

/-- Given a bag of disks with a specific color ratio and total count, 
    calculate the difference between green and blue disks. -/
theorem disk_color_difference 
  (total_disks : ℕ) 
  (blue_ratio yellow_ratio green_ratio red_ratio : ℕ) 
  (h_total : total_disks = 132)
  (h_ratio : blue_ratio + yellow_ratio + green_ratio + red_ratio = 22)
  (h_blue : blue_ratio = 3)
  (h_yellow : yellow_ratio = 7)
  (h_green : green_ratio = 8)
  (h_red : red_ratio = 4) :
  green_ratio * (total_disks / (blue_ratio + yellow_ratio + green_ratio + red_ratio)) -
  blue_ratio * (total_disks / (blue_ratio + yellow_ratio + green_ratio + red_ratio)) = 30 :=
by sorry

end NUMINAMATH_CALUDE_disk_color_difference_l3483_348327


namespace NUMINAMATH_CALUDE_democrat_ratio_l3483_348329

theorem democrat_ratio (total_participants male_participants female_participants male_democrats female_democrats : ℕ) :
  total_participants = 720 ∧
  female_participants = 240 ∧
  male_participants = 480 ∧
  female_democrats = 120 ∧
  2 * female_democrats = female_participants ∧
  3 * (male_democrats + female_democrats) = total_participants ∧
  male_participants + female_participants = total_participants →
  4 * male_democrats = male_participants :=
by
  sorry

end NUMINAMATH_CALUDE_democrat_ratio_l3483_348329


namespace NUMINAMATH_CALUDE_correct_ranking_l3483_348348

-- Define the team members
inductive TeamMember
| David
| Emma
| Frank

-- Define the experience relation
def has_more_experience (a b : TeamMember) : Prop := sorry

-- Define the most experienced member
def is_most_experienced (m : TeamMember) : Prop :=
  ∀ x : TeamMember, x ≠ m → has_more_experience m x

-- Define the statements
def statement_I : Prop := has_more_experience TeamMember.Frank TeamMember.Emma
def statement_II : Prop := has_more_experience TeamMember.David TeamMember.Frank
def statement_III : Prop := is_most_experienced TeamMember.Frank

-- Define the condition that exactly one statement is true
def exactly_one_true : Prop :=
  (statement_I ∧ ¬statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ ¬statement_II ∧ statement_III)

-- The theorem to prove
theorem correct_ranking (h : exactly_one_true) :
  has_more_experience TeamMember.David TeamMember.Emma ∧
  has_more_experience TeamMember.Emma TeamMember.Frank :=
sorry

end NUMINAMATH_CALUDE_correct_ranking_l3483_348348


namespace NUMINAMATH_CALUDE_thirtieth_number_in_base12_l3483_348312

/-- Converts a decimal number to its base 12 representation --/
def toBase12 (n : ℕ) : List ℕ :=
  if n < 12 then [n]
  else (n % 12) :: toBase12 (n / 12)

/-- Interprets a list of digits as a number in base 12 --/
def fromBase12 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 12 * acc) 0

theorem thirtieth_number_in_base12 :
  toBase12 30 = [6, 2] ∧ fromBase12 [6, 2] = 30 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_number_in_base12_l3483_348312


namespace NUMINAMATH_CALUDE_step_height_calculation_step_height_proof_l3483_348364

theorem step_height_calculation (num_flights : ℕ) (flight_height : ℕ) (total_steps : ℕ) (inches_per_foot : ℕ) : ℕ :=
  let total_height_feet := num_flights * flight_height
  let total_height_inches := total_height_feet * inches_per_foot
  total_height_inches / total_steps

theorem step_height_proof :
  step_height_calculation 9 10 60 12 = 18 := by
  sorry

end NUMINAMATH_CALUDE_step_height_calculation_step_height_proof_l3483_348364


namespace NUMINAMATH_CALUDE_odell_kershaw_passing_count_l3483_348328

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- radius of the lane in meters
  direction : ℝ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def passingCount (runner1 runner2 : Runner) (duration : ℝ) : ℕ :=
  sorry

theorem odell_kershaw_passing_count :
  let odell : Runner := ⟨260, 55, 1⟩
  let kershaw : Runner := ⟨280, 65, -1⟩
  passingCount odell kershaw 45 = 64 :=
sorry

end NUMINAMATH_CALUDE_odell_kershaw_passing_count_l3483_348328


namespace NUMINAMATH_CALUDE_derivative_difference_bound_l3483_348363

variable (f : ℝ → ℝ) (M : ℝ)

theorem derivative_difference_bound
  (h_diff : Differentiable ℝ f)
  (h_pos : M > 0)
  (h_bound : ∀ x t : ℝ, |f (x + t) - 2 * f x + f (x - t)| ≤ M * t^2) :
  ∀ x t : ℝ, |deriv f (x + t) - deriv f x| ≤ M * |t| :=
by sorry

end NUMINAMATH_CALUDE_derivative_difference_bound_l3483_348363


namespace NUMINAMATH_CALUDE_greatest_non_sum_of_composites_l3483_348307

def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬ Nat.Prime n

def isSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, isComposite a ∧ isComposite b ∧ a + b = n

theorem greatest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → isSumOfTwoComposites n) ∧
  ¬ isSumOfTwoComposites 11 := by sorry

end NUMINAMATH_CALUDE_greatest_non_sum_of_composites_l3483_348307


namespace NUMINAMATH_CALUDE_final_clay_pieces_l3483_348345

/-- Represents the number of pieces of clay of each color --/
structure ClayPieces where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Represents the operations performed on the clay pieces --/
def divide_non_red (pieces : ClayPieces) : ClayPieces :=
  { red := pieces.red,
    blue := pieces.blue * 2,
    yellow := pieces.yellow * 2 }

def divide_non_yellow (pieces : ClayPieces) : ClayPieces :=
  { red := pieces.red * 2,
    blue := pieces.blue * 2,
    yellow := pieces.yellow }

/-- The main theorem to prove --/
theorem final_clay_pieces :
  let initial_pieces := ClayPieces.mk 4 3 5
  let after_first_operation := divide_non_red initial_pieces
  let final_pieces := divide_non_yellow after_first_operation
  final_pieces.red + final_pieces.blue + final_pieces.yellow = 30 := by
  sorry


end NUMINAMATH_CALUDE_final_clay_pieces_l3483_348345


namespace NUMINAMATH_CALUDE_line_parameterization_l3483_348373

def is_valid_parameterization (p : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), p = (a, 3 * a - 4, b) ∧ v = (1/3, 1, 1)

theorem line_parameterization 
  (p : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) :
  is_valid_parameterization p v ↔
    (∃ (t : ℝ), 
      let (x, y, z) := p + t • v
      y = 3 * x - 4 ∧ z = t) :=
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3483_348373


namespace NUMINAMATH_CALUDE_inequality_proof_l3483_348387

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) : 
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3483_348387


namespace NUMINAMATH_CALUDE_problem_solution_l3483_348334

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -4 ∨ x > -2}
def C (m : ℝ) : Set ℝ := {x | 3 - 2*m ≤ x ∧ x ≤ 2 + m}
def D : Set ℝ := {y | y < -6 ∨ y > -5}

theorem problem_solution (m : ℝ) :
  (∀ x, x ∈ A ∩ B → x ∈ C m) →
  (B ∪ C m = Set.univ ∧ C m ⊆ D) →
  m ≥ 5/2 ∧ 7/2 ≤ m ∧ m < 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3483_348334


namespace NUMINAMATH_CALUDE_rectangular_floor_tiles_l3483_348330

theorem rectangular_floor_tiles (width : ℕ) (length : ℕ) (diagonal_tiles : ℕ) :
  (2 * width = 3 * length) →  -- length-to-width ratio is 3:2
  (diagonal_tiles * diagonal_tiles = 13 * width * width) →  -- diagonal covers whole number of tiles
  (2 * diagonal_tiles - 1 = 45) →  -- total tiles on both diagonals is 45
  (width * length = 245) :=  -- total tiles covering the floor
by sorry

end NUMINAMATH_CALUDE_rectangular_floor_tiles_l3483_348330


namespace NUMINAMATH_CALUDE_town_population_l3483_348344

theorem town_population (P : ℝ) : 
  (P * (1 - 0.2)^2 = 12800) → P = 20000 := by
  sorry

end NUMINAMATH_CALUDE_town_population_l3483_348344


namespace NUMINAMATH_CALUDE_train_length_l3483_348372

/-- The length of a train given its speed and time to pass an observer -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 144 → time_s = 10 → speed_kmh * (1000 / 3600) * time_s = 400 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3483_348372


namespace NUMINAMATH_CALUDE_distribution_methods_l3483_348384

/-- Represents the number of ways to distribute books to students -/
def distribute_books (novels : ℕ) (picture_books : ℕ) (students : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of distribution methods -/
theorem distribution_methods :
  distribute_books 2 2 3 = 12 :=
by sorry

end NUMINAMATH_CALUDE_distribution_methods_l3483_348384


namespace NUMINAMATH_CALUDE_solve_equation_for_x_l3483_348356

theorem solve_equation_for_x : ∃ x : ℚ, (3 * x / 7) - 2 = 12 ∧ x = 98 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_for_x_l3483_348356


namespace NUMINAMATH_CALUDE_condition_relationship_l3483_348308

theorem condition_relationship (x : ℝ) :
  (∀ x, (x + 2) * (x - 1) < 0 → x < 1) ∧
  (∃ x, x < 1 ∧ ¬((x + 2) * (x - 1) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l3483_348308


namespace NUMINAMATH_CALUDE_decimal_point_problem_l3483_348326

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 30 / 100 := by
sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l3483_348326


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l3483_348324

theorem min_value_and_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + Real.sqrt 2 * b + Real.sqrt 3 * c = 2 * Real.sqrt 3) :
  (∃ m : ℝ, 
    (∀ a' b' c' : ℝ, a' + Real.sqrt 2 * b' + Real.sqrt 3 * c' = 2 * Real.sqrt 3 → 
      a'^2 + b'^2 + c'^2 ≥ m) ∧ 
    (a^2 + b^2 + c^2 = m) ∧
    m = 2) ∧
  (∃ p q : ℝ, ∀ x : ℝ, (|x - 3| ≥ 2 ↔ x^2 + p*x + q ≥ 0) ∧ p = -6) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l3483_348324


namespace NUMINAMATH_CALUDE_fold_and_punch_theorem_l3483_348367

/-- Represents a rectangular piece of paper -/
structure Paper :=
  (width : ℕ)
  (height : ℕ)

/-- Represents the state of the paper after folding and punching -/
inductive FoldedPaper
  | Unfolded (p : Paper)
  | FoldedOnce (p : Paper)
  | FoldedTwice (p : Paper)
  | FoldedThrice (p : Paper)
  | Punched (p : Paper)

/-- Folds the paper from bottom to top -/
def foldBottomToTop (p : Paper) : FoldedPaper :=
  FoldedPaper.FoldedOnce p

/-- Folds the paper from right to left -/
def foldRightToLeft (p : FoldedPaper) : FoldedPaper :=
  match p with
  | FoldedPaper.FoldedOnce p => FoldedPaper.FoldedTwice p
  | _ => p

/-- Folds the paper from top to bottom -/
def foldTopToBottom (p : FoldedPaper) : FoldedPaper :=
  match p with
  | FoldedPaper.FoldedTwice p => FoldedPaper.FoldedThrice p
  | _ => p

/-- Punches a hole in the center of the folded paper -/
def punchHole (p : FoldedPaper) : FoldedPaper :=
  match p with
  | FoldedPaper.FoldedThrice p => FoldedPaper.Punched p
  | _ => p

/-- Counts the number of holes in the unfolded paper -/
def countHoles (p : FoldedPaper) : ℕ :=
  match p with
  | FoldedPaper.Punched _ => 8
  | _ => 0

/-- Theorem stating that folding a rectangular paper three times and punching a hole results in 8 holes when unfolded -/
theorem fold_and_punch_theorem (p : Paper) :
  countHoles (punchHole (foldTopToBottom (foldRightToLeft (foldBottomToTop p)))) = 8 := by
  sorry


end NUMINAMATH_CALUDE_fold_and_punch_theorem_l3483_348367


namespace NUMINAMATH_CALUDE_m_less_than_one_necessary_l3483_348316

/-- A function f(x) = x + mx + m has a root. -/
def has_root (m : ℝ) : Prop :=
  ∃ x : ℝ, x + m * x + m = 0

/-- "m < 1" is a necessary condition for f(x) = x + mx + m to have a root. -/
theorem m_less_than_one_necessary (m : ℝ) :
  has_root m → m < 1 := by sorry

end NUMINAMATH_CALUDE_m_less_than_one_necessary_l3483_348316


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_two_l3483_348375

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_two :
  lg 2 * lg 2 + lg 2 * lg 5 + lg 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_two_l3483_348375


namespace NUMINAMATH_CALUDE_dictionary_correct_and_complete_l3483_348301

-- Define the types for words and sentences
def Word : Type := String
def Sentence : Type := List Word

-- Define the type for a dictionary
def Dictionary : Type := List (Word × Word)

-- Define the Russian sentences
def russian_sentences : List Sentence := [
  ["Мышка", "ночью", "пошла", "гулять"],
  ["Кошка", "ночью", "видит", "мышка"],
  ["Мышку", "кошка", "пошла", "поймать"]
]

-- Define the Am-Yam sentences
def amyam_sentences : List Sentence := [
  ["ту", "ам", "ям", "му"],
  ["ля", "ам", "бу", "ту"],
  ["ту", "ля", "ям", "ям"]
]

-- Define the correct dictionary fragment
def correct_dictionary : Dictionary := [
  ("гулять", "му"),
  ("видит", "бу"),
  ("поймать", "ям"),
  ("мышка", "ту"),
  ("ночью", "ам"),
  ("пошла", "ям"),
  ("кошка", "ля")
]

-- Function to create dictionary from sentence pairs
def create_dictionary (russian : List Sentence) (amyam : List Sentence) : Dictionary :=
  sorry

-- Theorem statement
theorem dictionary_correct_and_complete 
  (russian : List Sentence := russian_sentences)
  (amyam : List Sentence := amyam_sentences)
  (correct : Dictionary := correct_dictionary) :
  create_dictionary russian amyam = correct :=
sorry

end NUMINAMATH_CALUDE_dictionary_correct_and_complete_l3483_348301


namespace NUMINAMATH_CALUDE_f_properties_l3483_348306

noncomputable def f (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

noncomputable def g (x : ℝ) : ℝ := f x - abs x

theorem f_properties :
  (∀ y ∈ Set.range f, 0 < y ∧ y < 4) ∧
  (∀ x, f x + f (-x) = 4) ∧
  (∃! a b, a < b ∧ g a = 0 ∧ g b = 0 ∧ ∀ x, x ≠ a ∧ x ≠ b → g x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3483_348306


namespace NUMINAMATH_CALUDE_cookies_needed_l3483_348383

/-- Given 6.0 people who each should receive 24.0 cookies, prove that the total number of cookies needed is 144.0. -/
theorem cookies_needed (people : Float) (cookies_per_person : Float) (h1 : people = 6.0) (h2 : cookies_per_person = 24.0) :
  people * cookies_per_person = 144.0 := by
  sorry

end NUMINAMATH_CALUDE_cookies_needed_l3483_348383


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l3483_348399

/-- The cost ratio of a muffin to a banana given Susie and Calvin's purchases -/
theorem muffin_banana_cost_ratio :
  let muffin_cost : ℚ := muffin_cost
  let banana_cost : ℚ := banana_cost
  (6 * muffin_cost + 4 * banana_cost) * 3 = 3 * muffin_cost + 24 * banana_cost →
  muffin_cost / banana_cost = 4 / 5 := by
sorry


end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l3483_348399


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l3483_348350

/-- The area of a square with adjacent points (2,1) and (3,4) on a Cartesian coordinate plane is 10. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (2, 1)
  let p2 : ℝ × ℝ := (3, 4)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 10 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l3483_348350


namespace NUMINAMATH_CALUDE_five_solutions_l3483_348389

/-- The number of positive integer solutions to 2x + y = 11 -/
def solution_count : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 + p.2 = 11 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 11) (Finset.range 11))).card

/-- Theorem stating that there are exactly 5 positive integer solutions to 2x + y = 11 -/
theorem five_solutions : solution_count = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_solutions_l3483_348389


namespace NUMINAMATH_CALUDE_books_count_l3483_348332

/-- The total number of books owned by six friends -/
def total_books (sandy benny tim rachel alex jordan : ℕ) : ℕ :=
  sandy + benny + tim + rachel + alex + jordan

/-- Theorem stating the total number of books owned by the six friends -/
theorem books_count :
  ∃ (sandy benny tim rachel alex jordan : ℕ),
    sandy = 10 ∧
    benny = 24 ∧
    tim = 33 ∧
    rachel = 2 * benny ∧
    alex = tim / 2 - 3 ∧
    jordan = sandy + benny ∧
    total_books sandy benny tim rachel alex jordan = 162 :=
by
  sorry

end NUMINAMATH_CALUDE_books_count_l3483_348332


namespace NUMINAMATH_CALUDE_service_provider_assignment_l3483_348398

theorem service_provider_assignment (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) = 303600 :=
by sorry

end NUMINAMATH_CALUDE_service_provider_assignment_l3483_348398


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3483_348313

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r

/-- Theorem: In a geometric sequence {aₙ}, if a₁a₃a₅ = 8, then a₃ = 2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : geometric_sequence a) 
    (h_prod : a 1 * a 3 * a 5 = 8) : 
  a 3 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3483_348313


namespace NUMINAMATH_CALUDE_transaction_difference_l3483_348359

theorem transaction_difference (mabel_transactions : ℕ) 
  (anthony_transactions : ℕ) (cal_transactions : ℕ) (jade_transactions : ℕ) :
  mabel_transactions = 90 →
  anthony_transactions = mabel_transactions + mabel_transactions / 10 →
  cal_transactions = anthony_transactions * 2 / 3 →
  jade_transactions = 85 →
  jade_transactions - cal_transactions = 19 := by
sorry

end NUMINAMATH_CALUDE_transaction_difference_l3483_348359


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l3483_348303

theorem sqrt_x_minus_one_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l3483_348303


namespace NUMINAMATH_CALUDE_point_not_in_region_l3483_348394

theorem point_not_in_region (m : ℝ) : 
  (1 : ℝ) - (m^2 - 2*m + 4)*(1 : ℝ) + 6 ≤ 0 ↔ m ≤ -1 ∨ m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_region_l3483_348394


namespace NUMINAMATH_CALUDE_quadratic_points_range_l3483_348378

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * a * x + 3

-- Define the theorem
theorem quadratic_points_range (a m y₁ y₂ : ℝ) :
  a > 0 →
  y₁ < y₂ →
  f a (m - 1) = y₁ →
  f a m = y₂ →
  m > -3/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_points_range_l3483_348378


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3483_348365

theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ a b : ℝ, (a ≠ 5 ∧ b ≠ -5) → a + b ≠ 0) ∧
  ¬(∀ a b : ℝ, a + b ≠ 0 → (a ≠ 5 ∧ b ≠ -5)) :=
by sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3483_348365


namespace NUMINAMATH_CALUDE_investment_interest_theorem_l3483_348366

/-- Calculates the total interest paid in an 18-month investment contract with specified conditions -/
def totalInterest (initialInvestment : ℝ) : ℝ :=
  let interestRate1 := 0.02
  let interestRate2 := 0.03
  let interestRate3 := 0.04
  
  let interest1 := initialInvestment * interestRate1
  let newInvestment1 := initialInvestment + interest1
  
  let interest2 := newInvestment1 * interestRate2
  let newInvestment2 := newInvestment1 + interest2
  
  let interest3 := newInvestment2 * interestRate3
  
  interest1 + interest2 + interest3

/-- Theorem stating that the total interest paid in the given investment scenario is $926.24 -/
theorem investment_interest_theorem :
  totalInterest 10000 = 926.24 := by sorry

end NUMINAMATH_CALUDE_investment_interest_theorem_l3483_348366


namespace NUMINAMATH_CALUDE_tagged_ratio_is_two_fiftieths_l3483_348393

/-- Represents the fish population and tagging experiment in a pond -/
structure FishExperiment where
  initial_tagged : ℕ
  second_catch : ℕ
  tagged_in_second : ℕ
  total_fish : ℕ

/-- The ratio of tagged fish to total fish in the second catch -/
def tagged_ratio (e : FishExperiment) : ℚ :=
  e.tagged_in_second / e.second_catch

/-- The given experiment data -/
def pond_experiment : FishExperiment :=
  { initial_tagged := 70
  , second_catch := 50
  , tagged_in_second := 2
  , total_fish := 1750 }

/-- Theorem stating that the ratio of tagged fish in the second catch is 2/50 -/
theorem tagged_ratio_is_two_fiftieths :
  tagged_ratio pond_experiment = 2 / 50 := by
  sorry

end NUMINAMATH_CALUDE_tagged_ratio_is_two_fiftieths_l3483_348393


namespace NUMINAMATH_CALUDE_log_inequality_l3483_348349

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x^2) < x^2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3483_348349
