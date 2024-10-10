import Mathlib

namespace min_b_minus_a_l619_61963

open Real

noncomputable def f (x : ℝ) : ℝ := log x - 1 / x

noncomputable def g (a b x : ℝ) : ℝ := -a * x + b

def is_tangent_line (f g : ℝ → ℝ) : Prop :=
  ∃ x₀, (∀ x, g x = f x₀ + (deriv f x₀) * (x - x₀))

theorem min_b_minus_a (a b : ℝ) :
  (∀ x, x > 0 → f x = f x) →
  is_tangent_line f (g a b) →
  b - a ≥ -1 ∧ ∃ a₀ b₀, b₀ - a₀ = -1 :=
sorry

end min_b_minus_a_l619_61963


namespace sufficient_condition_inequality_l619_61949

theorem sufficient_condition_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  b / a + a / b ≥ 2 := by
  sorry

end sufficient_condition_inequality_l619_61949


namespace f_range_on_interval_l619_61924

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * cos x - 2 * sin x ^ 2

theorem f_range_on_interval :
  ∀ x ∈ Set.Icc (π / 2) (3 * π / 2),
  ∃ y ∈ Set.Icc (-5/2) (-2), f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-5/2) (-2) :=
by sorry

end f_range_on_interval_l619_61924


namespace janet_shampoo_duration_l619_61914

/-- Calculates the number of days Janet's shampoo will last -/
def shampoo_duration (rose_shampoo : Rat) (jasmine_shampoo : Rat) (usage_per_day : Rat) : Nat :=
  Nat.floor ((rose_shampoo + jasmine_shampoo) / usage_per_day)

/-- Theorem: Janet's shampoo will last for 7 days -/
theorem janet_shampoo_duration :
  shampoo_duration (1/3) (1/4) (1/12) = 7 := by
  sorry

end janet_shampoo_duration_l619_61914


namespace mira_jogging_hours_l619_61911

/-- Mira's jogging problem -/
theorem mira_jogging_hours :
  ∀ (h : ℝ),
  (h > 0) →  -- Ensure positive jogging time
  (5 * h * 5 = 50) →  -- Total distance covered in 5 days
  h = 2 := by
sorry

end mira_jogging_hours_l619_61911


namespace triangle_division_theorem_l619_61913

/-- Represents a triangle with sides a, b, c and angles α, β, γ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  sum_angles : α + β + γ = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- Represents the two parts of a triangle after division along a median -/
structure TriangleParts where
  part1 : Triangle
  part2 : Triangle

theorem triangle_division_theorem (t1 t2 t3 : Triangle) 
  (h_identical : t1 = t2 ∧ t2 = t3) :
  ∃ (p1 p2 p3 : TriangleParts) (result : Triangle),
    (p1.part1.a = t1.a ∧ p1.part1.b = t1.b) ∧
    (p2.part1.a = t2.a ∧ p2.part1.b = t2.b) ∧
    (p3.part1.a = t3.a ∧ p3.part1.b = t3.b) ∧
    (p1.part1.α + p2.part1.α + p3.part1.α = 2 * π) ∧
    (result.a = t1.a ∧ result.b = t1.b ∧ result.c = t1.c) :=
by sorry

end triangle_division_theorem_l619_61913


namespace periodic_odd_function_property_l619_61941

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_property (f : ℝ → ℝ) (a : ℝ) 
    (h_periodic : is_periodic f 3)
    (h_odd : is_odd f)
    (h_f1 : f 1 > 1)
    (h_f2 : f 2 = a) :
  a < -1 := by
  sorry

end periodic_odd_function_property_l619_61941


namespace girls_who_bought_balloons_l619_61920

def initial_balloons : ℕ := 3 * 12
def boys_bought : ℕ := 3
def remaining_balloons : ℕ := 21

theorem girls_who_bought_balloons :
  initial_balloons - remaining_balloons - boys_bought = 12 :=
by sorry

end girls_who_bought_balloons_l619_61920


namespace anna_apple_count_l619_61997

/-- The number of apples Anna ate on Tuesday -/
def tuesday_apples : ℕ := 4

/-- The number of apples Anna ate on Wednesday -/
def wednesday_apples : ℕ := 2 * tuesday_apples

/-- The number of apples Anna ate on Thursday -/
def thursday_apples : ℕ := tuesday_apples / 2

/-- The total number of apples Anna ate over the three days -/
def total_apples : ℕ := tuesday_apples + wednesday_apples + thursday_apples

theorem anna_apple_count : total_apples = 14 := by
  sorry

end anna_apple_count_l619_61997


namespace floor_problem_l619_61903

theorem floor_problem (x : ℝ) : ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := by
  sorry

end floor_problem_l619_61903


namespace cubic_root_sum_l619_61961

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a*b/c + b*c/a + c*a/b = 49/6 := by
sorry

end cubic_root_sum_l619_61961


namespace muffin_banana_price_ratio_l619_61996

/-- The price ratio of a muffin to a banana is 2 -/
theorem muffin_banana_price_ratio :
  ∀ (m b S : ℝ),
  (3 * m + 5 * b = S) →
  (5 * m + 7 * b = 3 * S) →
  m = 2 * b :=
by sorry

end muffin_banana_price_ratio_l619_61996


namespace min_both_beethoven_chopin_survey_result_l619_61977

theorem min_both_beethoven_chopin 
  (total : ℕ) 
  (likes_beethoven : ℕ) 
  (likes_chopin : ℕ) 
  (h1 : total = 200)
  (h2 : likes_beethoven = 160)
  (h3 : likes_chopin = 150)
  : ℕ := by
  
  -- Define the minimum number who like both
  let min_both := likes_beethoven + likes_chopin - total
  
  -- Prove that min_both is the minimum number who like both Beethoven and Chopin
  sorry

-- State the theorem
theorem survey_result : min_both_beethoven_chopin 200 160 150 rfl rfl rfl = 110 := by
  sorry

end min_both_beethoven_chopin_survey_result_l619_61977


namespace ball_final_position_l619_61980

/-- Represents the possible final positions of the ball -/
inductive FinalPosition
  | B
  | A
  | C

/-- Determines the final position of the ball based on the parity of m and n -/
def finalBallPosition (m n : ℕ) : FinalPosition :=
  if m % 2 = 1 ∧ n % 2 = 1 then FinalPosition.B
  else if m % 2 = 0 ∧ n % 2 = 1 then FinalPosition.A
  else FinalPosition.C

/-- Theorem stating the final position of the ball -/
theorem ball_final_position (m n : ℕ) :
  (m > 0 ∧ n > 0) →
  (finalBallPosition m n = FinalPosition.B ↔ m % 2 = 1 ∧ n % 2 = 1) ∧
  (finalBallPosition m n = FinalPosition.A ↔ m % 2 = 0 ∧ n % 2 = 1) ∧
  (finalBallPosition m n = FinalPosition.C ↔ m % 2 = 1 ∧ n % 2 = 0) :=
by sorry

end ball_final_position_l619_61980


namespace function_ratio_bounds_l619_61932

open Real

theorem function_ratio_bounds (f : ℝ → ℝ) (hf : ∀ x > 0, f x > 0)
  (hf' : ∀ x > 0, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x) :
  8/27 < f 2 / f 3 ∧ f 2 / f 3 < 4/9 := by
  sorry

end function_ratio_bounds_l619_61932


namespace four_numbers_perfect_square_product_l619_61972

/-- A set of positive integers where all prime divisors are smaller than 30 -/
def SmallPrimeDivisorSet : Type := {s : Finset ℕ+ // ∀ n ∈ s, ∀ p : ℕ, Prime p → p ∣ n → p < 30}

theorem four_numbers_perfect_square_product (A : SmallPrimeDivisorSet) (h : A.val.card = 2016) :
  ∃ a b c d : ℕ+, a ∈ A.val ∧ b ∈ A.val ∧ c ∈ A.val ∧ d ∈ A.val ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ k : ℕ+, (a * b * c * d : ℕ) = k ^ 2 :=
sorry

end four_numbers_perfect_square_product_l619_61972


namespace min_perimeter_rectangle_is_square_l619_61981

/-- For a rectangle with area S and sides a and b, the perimeter is minimized when it's a square -/
theorem min_perimeter_rectangle_is_square (S : ℝ) (h : S > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = S ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x * y = S →
  2 * (a + b) ≤ 2 * (x + y) ∧
  (2 * (a + b) = 2 * (x + y) → a = b) :=
by sorry


end min_perimeter_rectangle_is_square_l619_61981


namespace hilt_current_rocks_l619_61989

/-- The number of rocks Mrs. Hilt needs to complete the border -/
def total_rocks_needed : ℕ := 125

/-- The number of additional rocks Mrs. Hilt needs -/
def additional_rocks_needed : ℕ := 61

/-- The number of rocks Mrs. Hilt currently has -/
def current_rocks : ℕ := total_rocks_needed - additional_rocks_needed

theorem hilt_current_rocks :
  current_rocks = 64 :=
sorry

end hilt_current_rocks_l619_61989


namespace problem_solution_l619_61908

theorem problem_solution (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1) 
  (h3 : Real.rpow a (2 * Real.log a / Real.log 3) = 81 * Real.sqrt 3) : 
  1 / a^2 + Real.log a / Real.log 9 = 105/4 := by
sorry

end problem_solution_l619_61908


namespace dorothy_doughnut_price_l619_61939

/-- Given Dorothy's doughnut business scenario, prove the selling price per doughnut. -/
theorem dorothy_doughnut_price 
  (ingredient_cost : ℚ) 
  (num_doughnuts : ℕ) 
  (profit : ℚ) 
  (h1 : ingredient_cost = 53)
  (h2 : num_doughnuts = 25)
  (h3 : profit = 22) :
  (ingredient_cost + profit) / num_doughnuts = 3 := by
  sorry

#eval (53 + 22) / 25

end dorothy_doughnut_price_l619_61939


namespace square_not_end_two_odd_digits_l619_61969

theorem square_not_end_two_odd_digits (n : ℕ) : 
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ n^2 % 100 = 10 * d₁ + d₂ → (d₁ % 2 = 0 ∨ d₂ % 2 = 0) :=
sorry

end square_not_end_two_odd_digits_l619_61969


namespace projection_a_onto_b_l619_61967

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (-2, 4)

theorem projection_a_onto_b :
  let proj := (a.1 * b.1 + a.2 * b.2) / Real.sqrt ((b.1 ^ 2 + b.2 ^ 2))
  proj = Real.sqrt 5 := by sorry

end projection_a_onto_b_l619_61967


namespace two_digit_integer_problem_l619_61904

theorem two_digit_integer_problem :
  ∀ m n : ℕ,
    m ≥ 10 ∧ m ≤ 99 →  -- m is a 2-digit positive integer
    n ≥ 10 ∧ n ≤ 99 →  -- n is a 2-digit positive integer
    n % 25 = 0 →  -- n is a multiple of 25
    (m + n) / 2 = m + n / 100 →  -- average equals decimal representation
    max m n = 50 :=
by sorry

end two_digit_integer_problem_l619_61904


namespace largest_common_divisor_660_483_l619_61950

theorem largest_common_divisor_660_483 : Nat.gcd 660 483 = 3 := by
  sorry

end largest_common_divisor_660_483_l619_61950


namespace cleaning_event_children_count_l619_61936

theorem cleaning_event_children_count (total_members : ℕ) 
  (adult_men_percentage : ℚ) (h1 : total_members = 2000) 
  (h2 : adult_men_percentage = 30 / 100) : 
  total_members - (adult_men_percentage * total_members).num - 
  (2 * (adult_men_percentage * total_members).num) = 200 := by
  sorry

end cleaning_event_children_count_l619_61936


namespace eggs_in_two_boxes_l619_61944

def eggs_per_box : ℕ := 3
def number_of_boxes : ℕ := 2

theorem eggs_in_two_boxes :
  eggs_per_box * number_of_boxes = 6 := by sorry

end eggs_in_two_boxes_l619_61944


namespace arithmetic_sequence_common_difference_l619_61905

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_property : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2

/-- Theorem: If 2S3 - 3S2 = 15 for an arithmetic sequence, then its common difference is 5 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 - 3 * seq.S 2 = 15) : 
  seq.d = 5 := by
sorry

end arithmetic_sequence_common_difference_l619_61905


namespace base_10_to_base_7_l619_61942

theorem base_10_to_base_7 (n : ℕ) (h : n = 3589) :
  ∃ (a b c d e : ℕ),
    n = a * 7^4 + b * 7^3 + c * 7^2 + d * 7^1 + e * 7^0 ∧
    a = 1 ∧ b = 3 ∧ c = 3 ∧ d = 1 ∧ e = 5 :=
by sorry

end base_10_to_base_7_l619_61942


namespace Q_one_smallest_l619_61900

def Q (x : ℝ) : ℝ := x^4 - 2*x^3 - 3*x^2 + 6*x - 5

theorem Q_one_smallest : 
  let q1 := Q 1
  let prod_zeros := -5
  let sum_coeff := 1 + (-2) + (-3) + 6 + (-5)
  q1 ≤ prod_zeros ∧ q1 ≤ sum_coeff :=
by sorry

end Q_one_smallest_l619_61900


namespace arithmetic_calculation_l619_61934

theorem arithmetic_calculation : 2^3 + 2 * 5 - 3 + 6 = 21 := by
  sorry

end arithmetic_calculation_l619_61934


namespace smallest_odd_with_three_prime_factors_l619_61984

-- Define a function to check if a number has exactly three distinct prime factors
def has_three_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  n = p * q * r

-- State the theorem
theorem smallest_odd_with_three_prime_factors :
  (∀ m : ℕ, m < 105 → m % 2 = 1 → ¬(has_three_distinct_prime_factors m)) ∧
  (105 % 2 = 1) ∧
  (has_three_distinct_prime_factors 105) :=
sorry

end smallest_odd_with_three_prime_factors_l619_61984


namespace bales_in_barn_after_addition_l619_61994

/-- The number of bales in the barn after addition -/
def bales_after_addition (initial_bales : ℕ) (added_bales : ℕ) : ℕ :=
  initial_bales + added_bales

/-- Theorem stating that the number of bales after Benny's addition is 82 -/
theorem bales_in_barn_after_addition :
  bales_after_addition 47 35 = 82 := by
  sorry

end bales_in_barn_after_addition_l619_61994


namespace intersection_of_A_and_B_l619_61971

def set_A : Set ℝ := {x | x^2 - 4 > 0}
def set_B : Set ℝ := {x | x + 2 < 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x : ℝ | x < -2} := by sorry

end intersection_of_A_and_B_l619_61971


namespace perpendicular_line_through_point_l619_61901

/-- Given two lines in the form ax + by + c = 0, this function returns true if they are perpendicular --/
def are_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

/-- Given a line in the form ax + by + c = 0 and a point (x, y), this function returns true if the point lies on the line --/
def point_on_line (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

theorem perpendicular_line_through_point :
  are_perpendicular 1 (-2) 2 1 ∧
  point_on_line 1 (-2) 3 1 2 :=
by
  sorry

end perpendicular_line_through_point_l619_61901


namespace fraction_simplification_l619_61935

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  1 / (x - 1) - 2 / (x^2 - 1) = 1 / (x + 1) :=
by sorry

end fraction_simplification_l619_61935


namespace squared_sum_of_x_and_y_l619_61959

theorem squared_sum_of_x_and_y (x y : ℝ) 
  (h : (2*x^2 + 2*y^2 + 3)*(2*x^2 + 2*y^2 - 3) = 27) : 
  x^2 + y^2 = 3 := by
sorry

end squared_sum_of_x_and_y_l619_61959


namespace circles_in_rectangle_l619_61964

theorem circles_in_rectangle (targetSum : ℝ) (h : targetSum = 1962) :
  ∃ (α : ℝ), 0 < α ∧ α < 1 / 3925 ∧
  ∀ (rectangle : Set (ℝ × ℝ)),
    (∃ a b, rectangle = {(x, y) | 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b} ∧ a * b = 1) →
    ∃ (n m : ℕ),
      (n : ℝ) * (m : ℝ) * (α / 2) > targetSum :=
by sorry

end circles_in_rectangle_l619_61964


namespace unique_valid_denomination_l619_61951

def is_valid_denomination (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 120 → ∃ (a b c : ℕ), k = 7 * a + n * b + (n + 2) * c

def is_greatest_unformable (n : ℕ) : Prop :=
  ¬∃ (a b c : ℕ), 120 = 7 * a + n * b + (n + 2) * c

theorem unique_valid_denomination :
  ∃! n : ℕ, n > 0 ∧ is_valid_denomination n ∧ is_greatest_unformable n :=
sorry

end unique_valid_denomination_l619_61951


namespace orthogonal_vectors_l619_61952

/-- Given vectors a and b in ℝ², prove that the value of t that makes (a - b) perpendicular to (a - t*b) is -11/30 -/
theorem orthogonal_vectors (a b : ℝ × ℝ) (h1 : a = (-3, 1)) (h2 : b = (2, 5)) :
  ∃ t : ℝ, t = -11/30 ∧ (a.1 - b.1, a.2 - b.2) • (a.1 - t * b.1, a.2 - t * b.2) = 0 :=
by sorry

end orthogonal_vectors_l619_61952


namespace parabola_kite_sum_l619_61987

/-- Two parabolas intersecting coordinate axes to form a kite -/
def parabola_kite (a b : ℝ) : Prop :=
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    -- Parabola equations
    (∀ x, a * x^2 - 4 = 6 - b * x^2 → x = x₁ ∨ x = x₂) ∧
    (∀ y, y = a * 0^2 - 4 → y = y₁) ∧
    (∀ y, y = 6 - b * 0^2 → y = y₂) ∧
    -- Four distinct intersection points
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    -- Kite area
    (1/2) * (x₂ - x₁) * (y₂ - y₁) = 18

/-- Theorem: If two parabolas form a kite with area 18, then a + b = 125/36 -/
theorem parabola_kite_sum (a b : ℝ) :
  parabola_kite a b → a + b = 125/36 := by
  sorry

end parabola_kite_sum_l619_61987


namespace root_product_l619_61923

theorem root_product (r b c : ℝ) : 
  r^2 = r + 1 → r^6 = b*r + c → b*c = 40 := by
  sorry

end root_product_l619_61923


namespace unique_sequence_l619_61983

/-- A strictly increasing sequence of natural numbers -/
def StrictlyIncreasingSeq (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- The property that a₂ = 2 -/
def SecondTermIsTwo (a : ℕ → ℕ) : Prop :=
  a 2 = 2

/-- The property that aₙₘ = aₙ * aₘ for any natural numbers n and m -/
def MultiplicativeProperty (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, a (n * m) = a n * a m

/-- The theorem stating that the only sequence satisfying all conditions is aₙ = n -/
theorem unique_sequence :
  ∀ a : ℕ → ℕ,
    StrictlyIncreasingSeq a →
    SecondTermIsTwo a →
    MultiplicativeProperty a →
    ∀ n : ℕ, a n = n :=
by sorry

end unique_sequence_l619_61983


namespace least_prime_factor_of_5_4_minus_5_3_l619_61993

theorem least_prime_factor_of_5_4_minus_5_3 :
  Nat.minFac (5^4 - 5^3) = 2 := by
  sorry

end least_prime_factor_of_5_4_minus_5_3_l619_61993


namespace teachers_health_survey_l619_61928

theorem teachers_health_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : high_bp = 90)
  (h3 : heart_trouble = 50)
  (h4 : both = 30) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 80 / 3 := by
  sorry

end teachers_health_survey_l619_61928


namespace equation_one_solution_equation_two_solution_l619_61915

-- Equation 1
theorem equation_one_solution :
  ∃ x : ℝ, (x ≠ 0 ∧ x ≠ 1) ∧ (9 / x = 8 / (x - 1)) → x = 9 :=
sorry

-- Equation 2
theorem equation_two_solution :
  ∃ x : ℝ, (x ≠ 2) ∧ (1 / (x - 2) - 3 = (x - 1) / (2 - x)) → x = 3 :=
sorry

end equation_one_solution_equation_two_solution_l619_61915


namespace circles_tangent_m_value_l619_61998

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y m : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + m = 0

-- Define the tangency condition
def are_tangent (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), C1 x y ∧ C2 x y ∧
  ∀ (x' y' : ℝ), (C1 x' y' ∧ C2 x' y') → (x' = x ∧ y' = y)

-- Theorem statement
theorem circles_tangent_m_value :
  are_tangent circle_C1 (circle_C2 · · 9) :=
sorry

end circles_tangent_m_value_l619_61998


namespace unique_natural_number_with_specific_divisor_differences_l619_61985

theorem unique_natural_number_with_specific_divisor_differences :
  ∃! n : ℕ,
    (∃ d₁ d₂ : ℕ, d₁ ∣ n ∧ d₂ ∣ n ∧ d₁ < d₂ ∧ ∀ d : ℕ, d ∣ n → d = d₁ ∨ d ≥ d₂) ∧
    (d₂ - d₁ = 4) ∧
    (∃ d₃ d₄ : ℕ, d₃ ∣ n ∧ d₄ ∣ n ∧ d₃ < d₄ ∧ ∀ d : ℕ, d ∣ n → d ≤ d₃ ∨ d = d₄) ∧
    (d₄ - d₃ = 308) ∧
    n = 385 :=
by
  sorry

end unique_natural_number_with_specific_divisor_differences_l619_61985


namespace complex_equation_solution_l619_61999

theorem complex_equation_solution (z : ℂ) : 
  Complex.I * z = 1 - 2 * Complex.I → z = 2 - Complex.I := by
  sorry

end complex_equation_solution_l619_61999


namespace largest_n_digit_divisible_by_61_l619_61931

theorem largest_n_digit_divisible_by_61 (n : ℕ+) :
  ∃ (k : ℕ), k = (10^n.val - 1) - ((10^n.val - 1) % 61) ∧ 
  k % 61 = 0 ∧
  k ≤ 10^n.val - 1 ∧
  ∀ m : ℕ, m % 61 = 0 → m ≤ 10^n.val - 1 → m ≤ k :=
by sorry

end largest_n_digit_divisible_by_61_l619_61931


namespace basketball_shots_l619_61907

theorem basketball_shots (total_points : ℕ) (three_point_shots : ℕ) : 
  total_points = 26 → 
  three_point_shots = 4 → 
  ∃ (two_point_shots : ℕ), 
    total_points = 3 * three_point_shots + 2 * two_point_shots ∧
    three_point_shots + two_point_shots = 11 :=
by sorry

end basketball_shots_l619_61907


namespace infinite_nonzero_digit_sum_equality_l619_61992

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a natural number contains zero in its digits -/
def contains_zero (n : ℕ) : Prop := sorry

theorem infinite_nonzero_digit_sum_equality :
  ∀ k : ℕ, ∃ f : ℕ → ℕ,
    (∀ n : ℕ, ¬contains_zero (f n)) ∧
    (∀ n : ℕ, sum_of_digits (f n) = sum_of_digits (k * f n)) ∧
    (∀ n : ℕ, f n < f (n + 1)) :=
by sorry

end infinite_nonzero_digit_sum_equality_l619_61992


namespace brians_net_commission_l619_61953

def house_price_1 : ℝ := 157000
def house_price_2 : ℝ := 499000
def house_price_3 : ℝ := 125000
def house_price_4 : ℝ := 275000
def house_price_5 : ℝ := 350000

def commission_rate_1 : ℝ := 0.025
def commission_rate_2 : ℝ := 0.018
def commission_rate_3 : ℝ := 0.02
def commission_rate_4 : ℝ := 0.022
def commission_rate_5 : ℝ := 0.023

def administrative_fee : ℝ := 500

def total_commission : ℝ := 
  house_price_1 * commission_rate_1 +
  house_price_2 * commission_rate_2 +
  house_price_3 * commission_rate_3 +
  house_price_4 * commission_rate_4 +
  house_price_5 * commission_rate_5

def net_commission : ℝ := total_commission - administrative_fee

theorem brians_net_commission : 
  net_commission = 29007 := by sorry

end brians_net_commission_l619_61953


namespace inverse_sum_equality_l619_61955

-- Define the function g and its inverse
variable (g : ℝ → ℝ)
variable (g_inv : ℝ → ℝ)

-- Define the given conditions
axiom g_inverse : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g
axiom g_4 : g 4 = 6
axiom g_6 : g 6 = 3
axiom g_7 : g 7 = 4

-- State the theorem
theorem inverse_sum_equality :
  g_inv (g_inv 4 + g_inv 6) = g_inv 11 :=
sorry

end inverse_sum_equality_l619_61955


namespace distance_between_points_l619_61937

/-- The distance between two points (3,2,0) and (7,6,0) in 3D space is 4√2. -/
theorem distance_between_points : Real.sqrt 32 = 4 * Real.sqrt 2 := by sorry

end distance_between_points_l619_61937


namespace even_increasing_inequality_l619_61906

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

theorem even_increasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) (h_incr : increasing_on_pos f) : 
  f 1 < f (-2) ∧ f (-2) < f 3 := by
  sorry

end even_increasing_inequality_l619_61906


namespace bennys_kids_l619_61919

/-- Prove that Benny has 18 kids given the conditions of the problem -/
theorem bennys_kids : ℕ :=
  let total_money : ℕ := 360
  let apple_cost : ℕ := 4
  let apples_per_kid : ℕ := 5
  let num_kids : ℕ := 18
  have h1 : total_money ≥ apple_cost * apples_per_kid * num_kids := by sorry
  have h2 : apples_per_kid > 0 := by sorry
  num_kids

/- Proof omitted -/

end bennys_kids_l619_61919


namespace consecutive_product_not_power_l619_61910

theorem consecutive_product_not_power (m k n : ℕ) (hn : n > 1) :
  m * (m + 1) ≠ k^n := by
  sorry

end consecutive_product_not_power_l619_61910


namespace animals_equal_humps_l619_61974

/-- Represents the number of animals of each type in the herd -/
structure Herd where
  horses : ℕ
  oneHumpCamels : ℕ
  twoHumpCamels : ℕ

/-- Calculates the total number of humps in the herd -/
def totalHumps (h : Herd) : ℕ :=
  h.oneHumpCamels + 2 * h.twoHumpCamels

/-- Calculates the total number of animals in the herd -/
def totalAnimals (h : Herd) : ℕ :=
  h.horses + h.oneHumpCamels + h.twoHumpCamels

/-- Theorem stating that under the given conditions, the total number of animals equals the total number of humps -/
theorem animals_equal_humps (h : Herd) 
    (hump_count : totalHumps h = 200) 
    (equal_horses_twohumps : h.horses = h.twoHumpCamels) : 
  totalAnimals h = 200 := by
  sorry


end animals_equal_humps_l619_61974


namespace ice_cream_sandwiches_l619_61947

theorem ice_cream_sandwiches (nieces : ℕ) (sandwiches_per_niece : ℕ) 
  (h1 : nieces = 11) (h2 : sandwiches_per_niece = 13) : 
  nieces * sandwiches_per_niece = 143 := by
  sorry

end ice_cream_sandwiches_l619_61947


namespace solve_system_of_equations_l619_61991

theorem solve_system_of_equations :
  ∃ (x y : ℝ), (x / 6) * 12 = 11 ∧ 4 * (x - y) + 5 = 11 ∧ x = 5.5 ∧ y = 4 := by
  sorry

end solve_system_of_equations_l619_61991


namespace bird_speed_indeterminate_l619_61916

/-- A structure representing the problem scenario -/
structure ScenarioData where
  train_speed : ℝ
  bird_distance : ℝ

/-- A function that attempts to calculate the bird's speed -/
def calculate_bird_speed (data : ScenarioData) : Option ℝ :=
  none

/-- Theorem stating that the bird's speed cannot be uniquely determined -/
theorem bird_speed_indeterminate (data : ScenarioData) 
  (h1 : data.train_speed = 60)
  (h2 : data.bird_distance = 120) :
  ∀ (s : ℝ), s > 0 → ∃ (t : ℝ), t > 0 ∧ s * t = data.bird_distance :=
sorry

#check bird_speed_indeterminate

end bird_speed_indeterminate_l619_61916


namespace mary_shop_visits_mary_shop_visits_proof_l619_61909

def shirt_cost : ℝ := 13.04
def jacket_cost : ℝ := 12.27
def total_cost : ℝ := 25.31

theorem mary_shop_visits : ℕ :=
  2

theorem mary_shop_visits_proof :
  (shirt_cost + jacket_cost = total_cost) →
  (∀ (shop : ℕ), shop ≤ mary_shop_visits → 
    (shop = 1 → shirt_cost > 0) ∧ 
    (shop = 2 → jacket_cost > 0)) →
  mary_shop_visits = 2 :=
by
  sorry

end mary_shop_visits_mary_shop_visits_proof_l619_61909


namespace point_in_fourth_quadrant_implies_m_greater_than_two_l619_61986

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point P as a function of m -/
def P (m : ℝ) : Point2D :=
  { x := m - 1, y := 2 - m }

theorem point_in_fourth_quadrant_implies_m_greater_than_two :
  ∀ m : ℝ, isInFourthQuadrant (P m) → m > 2 := by
  sorry

end point_in_fourth_quadrant_implies_m_greater_than_two_l619_61986


namespace custom_op_identity_l619_61945

/-- Custom operation ⊗ defined as k ⊗ l = k^2 - l^2 -/
def custom_op (k l : ℝ) : ℝ := k^2 - l^2

/-- Theorem stating that k ⊗ (k ⊗ k) = k^2 -/
theorem custom_op_identity (k : ℝ) : custom_op k (custom_op k k) = k^2 := by
  sorry

end custom_op_identity_l619_61945


namespace square_39_equals_square_40_minus_79_l619_61965

theorem square_39_equals_square_40_minus_79 : (39 : ℤ)^2 = (40 : ℤ)^2 - 79 := by
  sorry

end square_39_equals_square_40_minus_79_l619_61965


namespace corner_sum_possibilities_l619_61926

/-- Represents the color of a cell on the board -/
inductive CellColor
| Gold
| Silver

/-- Represents the board configuration -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (colorAt : Nat → Nat → CellColor)
  (numberAt : Nat → Nat → Nat)

/-- Defines a valid board configuration -/
def validBoard (b : Board) : Prop :=
  b.rows = 2016 ∧ b.cols = 2017 ∧
  (∀ i j, b.numberAt i j = 0 ∨ b.numberAt i j = 1) ∧
  (∀ i j, b.colorAt i j ≠ b.colorAt i (j+1)) ∧
  (∀ i j, b.colorAt i j ≠ b.colorAt (i+1) j) ∧
  (∀ i j, b.colorAt i j = CellColor.Gold →
    (b.numberAt i j + b.numberAt (i+1) j + b.numberAt i (j+1) + b.numberAt (i+1) (j+1)) % 2 = 0) ∧
  (∀ i j, b.colorAt i j = CellColor.Silver →
    (b.numberAt i j + b.numberAt (i+1) j + b.numberAt i (j+1) + b.numberAt (i+1) (j+1)) % 2 = 1)

/-- The theorem to be proved -/
theorem corner_sum_possibilities (b : Board) (h : validBoard b) :
  let cornerSum := b.numberAt 0 0 + b.numberAt 0 (b.cols-1) + b.numberAt (b.rows-1) 0 + b.numberAt (b.rows-1) (b.cols-1)
  cornerSum = 0 ∨ cornerSum = 2 ∨ cornerSum = 4 :=
sorry

end corner_sum_possibilities_l619_61926


namespace brothers_age_difference_l619_61973

/-- Bush and Matt are brothers with an age difference --/
def age_difference (bush_age : ℕ) (matt_future_age : ℕ) (years_to_future : ℕ) : ℕ :=
  (matt_future_age - years_to_future) - bush_age

/-- Theorem stating the age difference between Matt and Bush --/
theorem brothers_age_difference :
  age_difference 12 25 10 = 3 := by
  sorry

end brothers_age_difference_l619_61973


namespace inequality_proof_l619_61979

theorem inequality_proof (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 / (y - 1)) + (y^2 / (x - 1)) ≥ 8 := by
  sorry

end inequality_proof_l619_61979


namespace max_value_x_cubed_minus_y_cubed_l619_61922

theorem max_value_x_cubed_minus_y_cubed (x y : ℝ) (h : x^2 + y^2 = x + y) :
  ∃ (M : ℝ), M = 1 ∧ x^3 - y^3 ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = x₀ + y₀ ∧ x₀^3 - y₀^3 = M :=
sorry

end max_value_x_cubed_minus_y_cubed_l619_61922


namespace poster_ratio_l619_61927

theorem poster_ratio (total : ℕ) (small_fraction : ℚ) (large : ℕ) : 
  total = 50 → 
  small_fraction = 2 / 5 → 
  large = 5 → 
  (total - (small_fraction * total).num - large) / total = 1 / 2 := by
sorry

end poster_ratio_l619_61927


namespace angle_C_value_l619_61966

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

end angle_C_value_l619_61966


namespace floor_of_4_7_l619_61954

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end floor_of_4_7_l619_61954


namespace multiply_725143_by_999999_l619_61918

theorem multiply_725143_by_999999 : 725143 * 999999 = 725142274857 := by
  sorry

end multiply_725143_by_999999_l619_61918


namespace max_value_of_function_l619_61943

theorem max_value_of_function (x : ℝ) (h : x < -1) :
  ∃ (M : ℝ), M = -3 ∧ ∀ y, y = x + 1 / (x + 1) → y ≤ M :=
by sorry

end max_value_of_function_l619_61943


namespace line_perp_parallel_planes_l619_61948

-- Define the types for lines and planes
variable (L : Type) [LinearOrderedField L]
variable (P : Type) [AddCommGroup P] [Module L P]

-- Define the perpendicular and parallel relations
variable (perpLine : L → P → Prop)  -- Line perpendicular to plane
variable (perpPlane : P → P → Prop)  -- Plane perpendicular to plane
variable (parallel : P → P → Prop)  -- Plane parallel to plane

-- State the theorem
theorem line_perp_parallel_planes 
  (l : L) (α β : P) 
  (h1 : perpLine l β) 
  (h2 : parallel α β) : 
  perpLine l α :=
sorry

end line_perp_parallel_planes_l619_61948


namespace average_games_per_month_l619_61968

def total_games : ℕ := 323
def season_months : ℕ := 19

theorem average_games_per_month :
  (total_games : ℚ) / season_months = 17 := by sorry

end average_games_per_month_l619_61968


namespace quadratic_equation_general_form_l619_61988

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (x - 2) * (x + 3) = 1 ↔ x^2 + x - 7 = 0 :=
by sorry

end quadratic_equation_general_form_l619_61988


namespace mark_takes_tablets_for_12_hours_l619_61957

/-- Represents the number of hours Mark takes Tylenol tablets -/
def hours_taking_tablets (tablets_per_dose : ℕ) (mg_per_tablet : ℕ) (hours_between_doses : ℕ) (total_grams : ℕ) : ℕ :=
  (total_grams * 1000) / (tablets_per_dose * mg_per_tablet) * hours_between_doses

/-- Theorem stating that Mark takes the tablets for 12 hours -/
theorem mark_takes_tablets_for_12_hours :
  hours_taking_tablets 2 500 4 3 = 12 := by
  sorry

end mark_takes_tablets_for_12_hours_l619_61957


namespace solutions_exist_and_finite_l619_61990

theorem solutions_exist_and_finite :
  ∃ (n : ℕ) (S : Finset ℝ),
    (∀ θ ∈ S, 0 < θ ∧ θ < 2 * Real.pi) ∧
    (∀ θ ∈ S, Real.sin (7 * Real.pi * Real.cos θ) = Real.cos (7 * Real.pi * Real.sin θ)) ∧
    S.card = n :=
by sorry

end solutions_exist_and_finite_l619_61990


namespace polynomial_identity_sum_l619_61917

theorem polynomial_identity_sum (d1 d2 d3 e1 e2 e3 : ℝ) : 
  (∀ x : ℝ, x^7 - x^6 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + d1*x + e1) * (x^2 + d2*x + e2) * (x^2 + d3*x + e3)) →
  d1*e1 + d2*e2 + d3*e3 = -1 := by
sorry

end polynomial_identity_sum_l619_61917


namespace butterfat_mixture_l619_61978

/-- Proves that mixing 8 gallons of 50% butterfat milk with 24 gallons of 10% butterfat milk results in a mixture that is 20% butterfat. -/
theorem butterfat_mixture : 
  let milk_50_percent : ℝ := 8
  let milk_10_percent : ℝ := 24
  let butterfat_50_percent : ℝ := 0.5
  let butterfat_10_percent : ℝ := 0.1
  let total_volume : ℝ := milk_50_percent + milk_10_percent
  let total_butterfat : ℝ := milk_50_percent * butterfat_50_percent + milk_10_percent * butterfat_10_percent
  total_butterfat / total_volume = 0.2 := by
sorry

end butterfat_mixture_l619_61978


namespace stratified_sampling_sum_l619_61960

/-- Represents the number of items in each stratum -/
structure Strata :=
  (grains : ℕ)
  (vegetable_oils : ℕ)
  (animal_foods : ℕ)
  (fruits_and_vegetables : ℕ)

/-- Calculates the total number of items across all strata -/
def total_items (s : Strata) : ℕ :=
  s.grains + s.vegetable_oils + s.animal_foods + s.fruits_and_vegetables

/-- Calculates the number of items to sample from a stratum -/
def stratum_sample (total_sample : ℕ) (stratum_size : ℕ) (s : Strata) : ℕ :=
  (total_sample * stratum_size) / (total_items s)

/-- The main theorem to prove -/
theorem stratified_sampling_sum (s : Strata) (total_sample : ℕ) :
  s.grains = 40 →
  s.vegetable_oils = 10 →
  s.animal_foods = 30 →
  s.fruits_and_vegetables = 20 →
  total_sample = 20 →
  (stratum_sample total_sample s.vegetable_oils s +
   stratum_sample total_sample s.fruits_and_vegetables s) = 6 := by
  sorry

end stratified_sampling_sum_l619_61960


namespace friend_rides_80_times_more_l619_61902

/-- Tommy's effective riding area in square blocks -/
def tommy_area : ℚ := 1

/-- Tommy's friend's riding area in square blocks -/
def friend_area : ℚ := 80

/-- The ratio of Tommy's friend's riding area to Tommy's effective riding area -/
def area_ratio : ℚ := friend_area / tommy_area

theorem friend_rides_80_times_more : area_ratio = 80 := by
  sorry

end friend_rides_80_times_more_l619_61902


namespace square_tiles_count_l619_61912

/-- Represents the number of edges for each type of tile -/
def edges_per_tile : Fin 3 → ℕ
  | 0 => 3  -- triangular
  | 1 => 4  -- square
  | 2 => 5  -- pentagonal
  | _ => 0  -- should never happen

/-- The proposition that given the conditions, there are 10 square tiles -/
theorem square_tiles_count 
  (total_tiles : ℕ) 
  (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 30)
  (h_total_edges : total_edges = 120) :
  ∃ (t s p : ℕ), 
    t + s + p = total_tiles ∧ 
    3*t + 4*s + 5*p = total_edges ∧
    s = 10 :=
by sorry

end square_tiles_count_l619_61912


namespace only_setD_forms_triangle_l619_61976

-- Define a structure for a set of three line segments
structure SegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle inequality condition
def satisfiesTriangleInequality (s : SegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

-- Define the given sets of line segments
def setA : SegmentSet := ⟨1, 2, 3.5⟩
def setB : SegmentSet := ⟨4, 5, 9⟩
def setC : SegmentSet := ⟨5, 8, 15⟩
def setD : SegmentSet := ⟨6, 8, 9⟩

-- Theorem stating that only setD satisfies the triangle inequality
theorem only_setD_forms_triangle :
  ¬(satisfiesTriangleInequality setA) ∧
  ¬(satisfiesTriangleInequality setB) ∧
  ¬(satisfiesTriangleInequality setC) ∧
  satisfiesTriangleInequality setD :=
sorry

end only_setD_forms_triangle_l619_61976


namespace vector_complex_correspondence_l619_61982

theorem vector_complex_correspondence (z : ℂ) :
  z = -3 + 2*I → (-z) = 3 - 2*I := by sorry

end vector_complex_correspondence_l619_61982


namespace coefficient_of_monomial_degree_of_monomial_l619_61962

-- Define the monomial
def monomial : ℚ × (ℕ × ℕ) := (-4/3, (2, 1))

-- Theorem for the coefficient
theorem coefficient_of_monomial :
  (monomial.fst : ℚ) = -4/3 := by sorry

-- Theorem for the degree
theorem degree_of_monomial :
  (monomial.snd.fst + monomial.snd.snd : ℕ) = 3 := by sorry

end coefficient_of_monomial_degree_of_monomial_l619_61962


namespace f_zero_value_l619_61946

def is_nonneg_int (x : ℤ) : Prop := x ≥ 0

def functional_equation (f : ℤ → ℤ) : Prop :=
  ∀ m n, is_nonneg_int m → is_nonneg_int n →
    f (m^2 + n^2) = (f m - f n)^2 + f (2*m*n)

theorem f_zero_value (f : ℤ → ℤ) :
  (∀ x, is_nonneg_int (f x)) →
  functional_equation f →
  8 * f 0 + 9 * f 1 = 2006 →
  f 0 = 118 := by sorry

end f_zero_value_l619_61946


namespace intersection_AB_XOZ_plane_l619_61995

/-- Given two points A and B in 3D space, this function returns the coordinates of the 
    intersection point of the line passing through A and B with the XOZ plane. -/
def intersectionWithXOZPlane (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem stating that the intersection of the line passing through A(1,-2,-3) and B(2,-1,-1) 
    with the XOZ plane is the point (3,0,1). -/
theorem intersection_AB_XOZ_plane :
  let A : ℝ × ℝ × ℝ := (1, -2, -3)
  let B : ℝ × ℝ × ℝ := (2, -1, -1)
  intersectionWithXOZPlane A B = (3, 0, 1) := by
  sorry

end intersection_AB_XOZ_plane_l619_61995


namespace sin_cos_pi_12_l619_61925

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end sin_cos_pi_12_l619_61925


namespace bus_speed_with_stoppages_l619_61958

/-- Calculates the speed of a bus including stoppages -/
theorem bus_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (h1 : speed_without_stoppages = 50) 
  (h2 : stoppage_time = 8.4 / 60) : 
  ∃ (speed_with_stoppages : ℝ), 
    speed_with_stoppages = speed_without_stoppages * (1 - stoppage_time) ∧ 
    speed_with_stoppages = 43 := by
  sorry

end bus_speed_with_stoppages_l619_61958


namespace sum_of_sqrt_geq_sum_of_products_l619_61975

theorem sum_of_sqrt_geq_sum_of_products (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a + b + c = 3) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ a * b + b * c + a * c := by
  sorry

end sum_of_sqrt_geq_sum_of_products_l619_61975


namespace convex_polyhedron_max_intersections_non_convex_polyhedron_96_intersections_no_full_intersection_l619_61929

/-- A polyhedron with a specified number of edges. -/
structure Polyhedron where
  edges : ℕ
  is_convex : Bool

/-- A plane that can intersect edges of a polyhedron. -/
structure IntersectingPlane where
  intersected_edges : ℕ
  passes_through_vertices : Bool

/-- Theorem about the maximum number of edges a plane can intersect in a convex polyhedron. -/
theorem convex_polyhedron_max_intersections
  (p : Polyhedron)
  (plane : IntersectingPlane)
  (h1 : p.edges = 100)
  (h2 : p.is_convex = true)
  (h3 : plane.passes_through_vertices = false) :
  plane.intersected_edges ≤ 66 :=
sorry

/-- Theorem about the existence of a non-convex polyhedron where a plane can intersect 96 edges. -/
theorem non_convex_polyhedron_96_intersections
  (p : Polyhedron)
  (h : p.edges = 100) :
  ∃ (plane : IntersectingPlane), plane.intersected_edges = 96 ∧ p.is_convex = false :=
sorry

/-- Theorem stating that it's impossible for a plane to intersect all 100 edges of a polyhedron. -/
theorem no_full_intersection
  (p : Polyhedron)
  (h : p.edges = 100) :
  ¬ ∃ (plane : IntersectingPlane), plane.intersected_edges = 100 :=
sorry

end convex_polyhedron_max_intersections_non_convex_polyhedron_96_intersections_no_full_intersection_l619_61929


namespace primitive_poly_count_l619_61930

/-- A polynomial with integer coefficients -/
structure IntPoly :=
  (a₂ a₁ a₀ : ℤ)

/-- The set of integers from 1 to 5 -/
def S : Set ℤ := {1, 2, 3, 4, 5}

/-- A polynomial is primitive if the gcd of its coefficients is 1 -/
def isPrimitive (p : IntPoly) : Prop :=
  Nat.gcd p.a₂.natAbs (Nat.gcd p.a₁.natAbs p.a₀.natAbs) = 1

/-- The product of two polynomials -/
def polyMul (p q : IntPoly) : IntPoly :=
  ⟨p.a₂ * q.a₂,
   p.a₂ * q.a₁ + p.a₁ * q.a₂,
   p.a₂ * q.a₀ + p.a₁ * q.a₁ + p.a₀ * q.a₂⟩

/-- The number of pairs of polynomials (f, g) such that f * g is primitive -/
def N : ℕ := sorry

theorem primitive_poly_count :
  N ≡ 689 [MOD 1000] := by sorry

end primitive_poly_count_l619_61930


namespace total_children_l619_61970

/-- The number of children who like cabbage -/
def cabbage_lovers : ℕ := 7

/-- The number of children who like carrots -/
def carrot_lovers : ℕ := 6

/-- The number of children who like peas -/
def pea_lovers : ℕ := 5

/-- The number of children who like both cabbage and carrots -/
def cabbage_carrot_lovers : ℕ := 4

/-- The number of children who like both cabbage and peas -/
def cabbage_pea_lovers : ℕ := 3

/-- The number of children who like both carrots and peas -/
def carrot_pea_lovers : ℕ := 2

/-- The number of children who like all three vegetables -/
def all_veg_lovers : ℕ := 1

/-- The theorem stating the total number of children in the family -/
theorem total_children : 
  cabbage_lovers + carrot_lovers + pea_lovers - 
  cabbage_carrot_lovers - cabbage_pea_lovers - carrot_pea_lovers + 
  all_veg_lovers = 10 := by
  sorry

end total_children_l619_61970


namespace remy_used_19_gallons_l619_61933

/-- Represents the water usage of three people taking showers -/
structure ShowerUsage where
  roman : ℕ
  remy : ℕ
  riley : ℕ

/-- Defines the conditions of the shower usage problem -/
def validShowerUsage (u : ShowerUsage) : Prop :=
  u.remy = 3 * u.roman + 1 ∧
  u.riley = u.roman + u.remy - 2 ∧
  u.roman + u.remy + u.riley = 48

/-- Theorem stating that if the shower usage is valid, Remy used 19 gallons -/
theorem remy_used_19_gallons (u : ShowerUsage) : 
  validShowerUsage u → u.remy = 19 := by
  sorry

#check remy_used_19_gallons

end remy_used_19_gallons_l619_61933


namespace freds_change_is_correct_l619_61940

/-- The amount of change Fred received after buying movie tickets and borrowing a movie -/
def freds_change (ticket_price : ℚ) (num_tickets : ℕ) (borrowed_movie_price : ℚ) (paid_amount : ℚ) : ℚ :=
  paid_amount - (ticket_price * num_tickets + borrowed_movie_price)

/-- Theorem: Fred's change is $1.37 -/
theorem freds_change_is_correct : 
  freds_change (92/100 + 5) 2 (79/100 + 6) 20 = 37/100 + 1 :=
by sorry

end freds_change_is_correct_l619_61940


namespace georgia_muffins_l619_61938

theorem georgia_muffins (students : ℕ) (muffins_per_batch : ℕ) (months : ℕ) :
  students = 24 →
  muffins_per_batch = 6 →
  months = 9 →
  (students / muffins_per_batch) * months = 36 :=
by
  sorry

end georgia_muffins_l619_61938


namespace sum_of_fractions_simplest_form_l619_61956

theorem sum_of_fractions_simplest_form : 
  (6 : ℚ) / 7 + (7 : ℚ) / 9 = (103 : ℚ) / 63 ∧ 
  ∀ n d : ℤ, (n : ℚ) / d = (103 : ℚ) / 63 → (n.gcd d = 1 → n = 103 ∧ d = 63) :=
by sorry

end sum_of_fractions_simplest_form_l619_61956


namespace intersection_slope_of_circles_l619_61921

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y + 10 = 0

-- Define the slope of the line passing through the intersection points
def intersection_slope : ℝ := 0.4

-- Theorem statement
theorem intersection_slope_of_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → 
  ∃ m b : ℝ, m = intersection_slope ∧ y = m * x + b :=
by sorry

end intersection_slope_of_circles_l619_61921
