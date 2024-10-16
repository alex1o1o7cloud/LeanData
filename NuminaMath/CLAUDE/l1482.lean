import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l1482_148224

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hab : 1 / a + 1 / b = 1) (n : ℕ) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1482_148224


namespace NUMINAMATH_CALUDE_lcm_count_theorem_exists_valid_k_upper_bound_valid_k_l1482_148295

-- Define the function to count valid k values
def count_valid_k : ℕ :=
  -- Count the number of a values from 0 to 18 (inclusive)
  -- where k = 2^a * 3^36 satisfies the LCM condition
  (Finset.range 19).card

-- State the theorem
theorem lcm_count_theorem : 
  count_valid_k = 19 :=
sorry

-- Define the LCM condition
def is_valid_k (k : ℕ) : Prop :=
  Nat.lcm (Nat.lcm (9^9) (16^16)) k = 18^18

-- State the existence of valid k values
theorem exists_valid_k :
  ∃ k : ℕ, k > 0 ∧ is_valid_k k :=
sorry

-- State the upper bound of valid k values
theorem upper_bound_valid_k :
  ∀ k : ℕ, is_valid_k k → k ≤ 18^18 :=
sorry

end NUMINAMATH_CALUDE_lcm_count_theorem_exists_valid_k_upper_bound_valid_k_l1482_148295


namespace NUMINAMATH_CALUDE_integers_with_consecutive_twos_l1482_148220

def fibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def valid_integers (n : ℕ) : ℕ := 2^n

def integers_without_consecutive_twos (n : ℕ) : ℕ := fibonacci (n - 1)

theorem integers_with_consecutive_twos (n : ℕ) : 
  valid_integers n - integers_without_consecutive_twos n = 880 → n = 10 := by
  sorry

#eval valid_integers 10 - integers_without_consecutive_twos 10

end NUMINAMATH_CALUDE_integers_with_consecutive_twos_l1482_148220


namespace NUMINAMATH_CALUDE_base9_to_base10_653_l1482_148223

/-- Converts a base-9 number to base 10 --/
def base9_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9^i)) 0

/-- The base-9 representation of the number --/
def base9_number : List Nat := [3, 5, 6]

theorem base9_to_base10_653 :
  base9_to_base10 base9_number = 534 := by
  sorry

end NUMINAMATH_CALUDE_base9_to_base10_653_l1482_148223


namespace NUMINAMATH_CALUDE_dog_grouping_combinations_l1482_148252

def total_dogs : ℕ := 12
def group1_size : ℕ := 4
def group2_size : ℕ := 5
def group3_size : ℕ := 3

def buster_in_group1 : Prop := True
def whiskers_in_group2 : Prop := True

def remaining_dogs : ℕ := total_dogs - 2
def remaining_group1 : ℕ := group1_size - 1
def remaining_group2 : ℕ := group2_size - 1

theorem dog_grouping_combinations :
  buster_in_group1 →
  whiskers_in_group2 →
  Nat.choose remaining_dogs remaining_group1 * Nat.choose (remaining_dogs - remaining_group1) remaining_group2 = 4200 := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_combinations_l1482_148252


namespace NUMINAMATH_CALUDE_evaluate_expression_l1482_148246

theorem evaluate_expression (a : ℝ) : 
  let x : ℝ := a + 5
  (2 * x - a + 4) = (a + 14) := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1482_148246


namespace NUMINAMATH_CALUDE_f_zero_at_three_l1482_148221

def f (x r : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + r

theorem f_zero_at_three (r : ℝ) : f 3 r = 0 ↔ r = -276 := by sorry

end NUMINAMATH_CALUDE_f_zero_at_three_l1482_148221


namespace NUMINAMATH_CALUDE_keiko_text_messages_l1482_148208

/-- The number of text messages Keiko sent in the first week -/
def first_week : ℕ := 111

/-- The number of text messages Keiko sent in the second week -/
def second_week : ℕ := 2 * first_week - 50

/-- The number of text messages Keiko sent in the third week -/
def third_week : ℕ := second_week + (second_week / 4)

/-- The total number of text messages Keiko sent over three weeks -/
def total_messages : ℕ := first_week + second_week + third_week

theorem keiko_text_messages : total_messages = 498 := by
  sorry

end NUMINAMATH_CALUDE_keiko_text_messages_l1482_148208


namespace NUMINAMATH_CALUDE_rectangular_solid_width_l1482_148226

/-- The surface area of a rectangular solid given its length, width, and height. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The width of a rectangular solid with length 5, depth 1, and surface area 58 is 4. -/
theorem rectangular_solid_width :
  ∃ w : ℝ, w = 4 ∧ surface_area 5 w 1 = 58 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_width_l1482_148226


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l1482_148206

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The first focus of the ellipse -/
  F₁ : ℝ × ℝ
  /-- The second focus of the ellipse -/
  F₂ : ℝ × ℝ
  /-- A point on the ellipse -/
  P : ℝ × ℝ
  /-- The first focus is at (-4, 0) -/
  h_F₁ : F₁ = (-4, 0)
  /-- The second focus is at (4, 0) -/
  h_F₂ : F₂ = (4, 0)
  /-- The dot product of PF₁ and PF₂ is zero -/
  h_perpendicular : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0
  /-- The area of triangle PF₁F₂ is 9 -/
  h_area : abs ((P.1 - F₁.1) * (P.2 - F₂.2) - (P.2 - F₁.2) * (P.1 - F₂.1)) / 2 = 9

/-- The standard equation of the special ellipse -/
def standardEquation (e : SpecialEllipse) : Prop :=
  ∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 - e.F₁.1)^2 + (p.2 - e.F₁.2)^2 + (p.1 - e.F₂.1)^2 + (p.2 - e.F₂.2)^2 = 100}

/-- The main theorem: The standard equation of the special ellipse is x²/25 + y²/9 = 1 -/
theorem special_ellipse_equation (e : SpecialEllipse) : standardEquation e := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l1482_148206


namespace NUMINAMATH_CALUDE_alpha_squared_plus_one_times_one_plus_cos_two_alpha_equals_two_l1482_148218

theorem alpha_squared_plus_one_times_one_plus_cos_two_alpha_equals_two
  (α : ℝ) (h1 : α ≠ 0) (h2 : α + Real.tan α = 0) :
  (α^2 + 1) * (1 + Real.cos (2 * α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_squared_plus_one_times_one_plus_cos_two_alpha_equals_two_l1482_148218


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l1482_148207

theorem ratio_x_to_y (x y : ℝ) (h : y = 0.2 * x) : x / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l1482_148207


namespace NUMINAMATH_CALUDE_bobs_hair_length_l1482_148278

/-- Calculates the final hair length after a given time period. -/
def final_hair_length (initial_length : ℝ) (growth_rate : ℝ) (time : ℝ) : ℝ :=
  initial_length + growth_rate * 12 * time

/-- Proves that Bob's hair length after 5 years is 36 inches. -/
theorem bobs_hair_length :
  let initial_length : ℝ := 6
  let growth_rate : ℝ := 0.5
  let time : ℝ := 5
  final_hair_length initial_length growth_rate time = 36 := by
  sorry

end NUMINAMATH_CALUDE_bobs_hair_length_l1482_148278


namespace NUMINAMATH_CALUDE_array_sum_divisibility_l1482_148293

/-- Represents the sum of all terms in a 1/2011-array -/
def arraySum : ℚ :=
  (2011^2 : ℚ) / ((4011 : ℚ) * 2010)

/-- Numerator of the array sum when expressed as a simplified fraction -/
def m : ℕ := 2011^2

/-- Denominator of the array sum when expressed as a simplified fraction -/
def n : ℕ := 4011 * 2010

/-- Theorem stating that m + n is divisible by 2011 -/
theorem array_sum_divisibility : (m + n) % 2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_array_sum_divisibility_l1482_148293


namespace NUMINAMATH_CALUDE_painting_time_equation_l1482_148258

theorem painting_time_equation (doug_time dave_time lunch_break : ℝ) 
  (h_doug : doug_time = 6)
  (h_dave : dave_time = 8)
  (h_lunch : lunch_break = 2)
  (t : ℝ) :
  (1 / doug_time + 1 / dave_time) * (t - lunch_break) = 1 :=
by sorry

end NUMINAMATH_CALUDE_painting_time_equation_l1482_148258


namespace NUMINAMATH_CALUDE_opposite_of_one_half_l1482_148288

theorem opposite_of_one_half : 
  (1 / 2 : ℚ) + (-1 / 2 : ℚ) = 0 := by sorry

end NUMINAMATH_CALUDE_opposite_of_one_half_l1482_148288


namespace NUMINAMATH_CALUDE_system_solution_l1482_148202

theorem system_solution :
  let x : ℝ := -13
  let y : ℝ := -1
  let z : ℝ := 2
  (x + y + 16 * z = 18) ∧
  (x - 3 * y + 8 * z = 6) ∧
  (2 * x - y - 4 * z = -33) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1482_148202


namespace NUMINAMATH_CALUDE_three_person_subcommittees_from_seven_l1482_148255

theorem three_person_subcommittees_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → 
  Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_from_seven_l1482_148255


namespace NUMINAMATH_CALUDE_marcus_baseball_cards_l1482_148227

/-- Given that Carter has 152 baseball cards and Marcus has 58 more than Carter,
    prove that Marcus has 210 baseball cards. -/
theorem marcus_baseball_cards :
  let carter_cards : ℕ := 152
  let difference : ℕ := 58
  let marcus_cards : ℕ := carter_cards + difference
  marcus_cards = 210 := by sorry

end NUMINAMATH_CALUDE_marcus_baseball_cards_l1482_148227


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l1482_148248

/-- A configuration of tangents to a circle -/
structure TangentConfiguration where
  r : ℝ  -- radius of the circle
  AB : ℝ  -- length of tangent AB
  CD : ℝ  -- length of tangent CD
  EF : ℝ  -- length of EF

/-- The theorem stating the radius of the circle given the tangent configuration -/
theorem tangent_circle_radius (config : TangentConfiguration) 
  (h1 : config.AB = 12)
  (h2 : config.CD = 20)
  (h3 : config.EF = 8) :
  config.r = 6 := by
  sorry

#check tangent_circle_radius

end NUMINAMATH_CALUDE_tangent_circle_radius_l1482_148248


namespace NUMINAMATH_CALUDE_quadratic_with_inequality_has_negative_root_l1482_148289

/-- A quadratic polynomial with two distinct roots satisfying a specific inequality has at least one negative root. -/
theorem quadratic_with_inequality_has_negative_root 
  (f : ℝ → ℝ) 
  (h_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (h_distinct_roots : ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ f r₁ = 0 ∧ f r₂ = 0)
  (h_inequality : ∀ a b : ℝ, f (a^2 + b^2) ≥ f (2*a*b)) :
  ∃ r : ℝ, f r = 0 ∧ r < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_with_inequality_has_negative_root_l1482_148289


namespace NUMINAMATH_CALUDE_time_per_furniture_piece_l1482_148228

theorem time_per_furniture_piece (chairs tables total_time : ℕ) : 
  chairs = 7 → tables = 3 → total_time = 40 → (chairs + tables) * 4 = total_time := by
  sorry

end NUMINAMATH_CALUDE_time_per_furniture_piece_l1482_148228


namespace NUMINAMATH_CALUDE_max_a_right_angle_circle_l1482_148237

/-- Given points A(-a, 0) and B(a, 0) where a > 0, and a point C on the circle (x-2)²+(y-2)²=2
    such that ∠ACB = 90°, the maximum value of a is 3√2. -/
theorem max_a_right_angle_circle (a : ℝ) (C : ℝ × ℝ) : 
  a > 0 → 
  (C.1 - 2)^2 + (C.2 - 2)^2 = 2 →
  (C.1 + a) * (C.1 - a) + C.2 * C.2 = 0 →
  a ≤ 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_a_right_angle_circle_l1482_148237


namespace NUMINAMATH_CALUDE_dog_tricks_conversion_l1482_148285

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem dog_tricks_conversion :
  base7ToBase10 [3, 5, 6] = 332 := by
  sorry

end NUMINAMATH_CALUDE_dog_tricks_conversion_l1482_148285


namespace NUMINAMATH_CALUDE_problem_2011_l1482_148238

theorem problem_2011 : (2011^2 - 2011) / 2011 = 2010 := by sorry

end NUMINAMATH_CALUDE_problem_2011_l1482_148238


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l1482_148269

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
structure SimilarTriangles (P Q R X Y Z : ℝ × ℝ) : Prop where
  ratio_eq : (dist P Q) / (dist X Y) = (dist Q R) / (dist Y Z)

theorem similar_triangles_side_length 
  (P Q R X Y Z : ℝ × ℝ) 
  (h_similar : SimilarTriangles P Q R X Y Z) 
  (h_PQ : dist P Q = 9)
  (h_QR : dist Q R = 15)
  (h_YZ : dist Y Z = 30) :
  dist X Y = 18 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l1482_148269


namespace NUMINAMATH_CALUDE_percentage_less_l1482_148231

theorem percentage_less (x y : ℝ) (h : x = 3 * y) : 
  (x - y) / x * 100 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_less_l1482_148231


namespace NUMINAMATH_CALUDE_solution_set_subset_interval_l1482_148217

def solution_set (a : ℝ) : Set ℝ :=
  {x | x^2 - 2*a*x + a + 2 ≤ 0}

theorem solution_set_subset_interval (a : ℝ) :
  solution_set a ⊆ Set.Icc 1 4 ↔ a ∈ Set.Ioo (-1) (18/7) :=
sorry

end NUMINAMATH_CALUDE_solution_set_subset_interval_l1482_148217


namespace NUMINAMATH_CALUDE_prob_yellow_twice_is_one_ninth_l1482_148298

/-- A fair 12-sided die with 4 yellow faces -/
structure YellowDie :=
  (sides : ℕ)
  (yellow_faces : ℕ)
  (is_fair : sides = 12)
  (yellow_count : yellow_faces = 4)

/-- The probability of rolling yellow twice with a YellowDie -/
def prob_yellow_twice (d : YellowDie) : ℚ :=
  (d.yellow_faces : ℚ) / (d.sides : ℚ) * (d.yellow_faces : ℚ) / (d.sides : ℚ)

/-- Theorem: The probability of rolling yellow twice with a YellowDie is 1/9 -/
theorem prob_yellow_twice_is_one_ninth (d : YellowDie) :
  prob_yellow_twice d = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_yellow_twice_is_one_ninth_l1482_148298


namespace NUMINAMATH_CALUDE_messenger_catches_up_l1482_148210

/-- Represents the scenario of a messenger catching up to Ilya Muromets --/
def catchUpScenario (ilyaSpeed : ℝ) : Prop :=
  let messengerSpeed := 2 * ilyaSpeed
  let horseSpeed := 5 * messengerSpeed
  let initialDelay := 10 -- seconds
  let ilyaDistance := ilyaSpeed * initialDelay
  let horseDistance := horseSpeed * initialDelay
  let totalDistance := ilyaDistance + horseDistance
  let relativeSpeed := messengerSpeed - ilyaSpeed
  let catchUpTime := totalDistance / relativeSpeed
  catchUpTime = 110

/-- Theorem stating that under the given conditions, 
    the messenger catches up to Ilya Muromets in 110 seconds --/
theorem messenger_catches_up (ilyaSpeed : ℝ) (ilyaSpeed_pos : 0 < ilyaSpeed) :
  catchUpScenario ilyaSpeed := by
  sorry

#check messenger_catches_up

end NUMINAMATH_CALUDE_messenger_catches_up_l1482_148210


namespace NUMINAMATH_CALUDE_expression_simplification_l1482_148219

theorem expression_simplification (x : ℝ) : 
  (1 + Real.sin (2 * x) - Real.cos (2 * x)) / (1 + Real.sin (2 * x) + Real.cos (2 * x)) = Real.tan x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1482_148219


namespace NUMINAMATH_CALUDE_A_intersect_B_l1482_148273

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < 2 - x ∧ 2 - x < 3}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1482_148273


namespace NUMINAMATH_CALUDE_daily_wage_c_value_l1482_148235

/-- The daily wage of worker c given the conditions of the problem -/
def daily_wage_c (days_a days_b days_c : ℕ) 
                 (ratio_a ratio_b ratio_c : ℕ) 
                 (total_earning : ℚ) : ℚ :=
  let wage_a := total_earning * 3 / (ratio_a * days_a + ratio_b * days_b + ratio_c * days_c)
  wage_a * ratio_c / ratio_a

/-- Theorem stating that the daily wage of c is $66.67 given the problem conditions -/
theorem daily_wage_c_value : 
  daily_wage_c 6 9 4 3 4 5 1480 = 200/3 := by sorry

end NUMINAMATH_CALUDE_daily_wage_c_value_l1482_148235


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l1482_148265

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l1482_148265


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l1482_148262

theorem min_value_quadratic_sum (a b c : ℝ) (h : a + 2 * b + 3 * c = 6) :
  a^2 + 4 * b^2 + 9 * c^2 ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l1482_148262


namespace NUMINAMATH_CALUDE_prob_odd_sum_is_half_l1482_148211

/-- Represents a wheel with numbers from 1 to n -/
def Wheel (n : ℕ) := Finset (Fin n)

/-- The probability of selecting an odd number from a wheel -/
def prob_odd (w : Wheel n) : ℚ :=
  (w.filter (λ x => x.val % 2 = 1)).card / w.card

/-- The probability of selecting an even number from a wheel -/
def prob_even (w : Wheel n) : ℚ :=
  (w.filter (λ x => x.val % 2 = 0)).card / w.card

/-- The first wheel with numbers 1 to 5 -/
def wheel1 : Wheel 5 := Finset.univ

/-- The second wheel with numbers 1 to 4 -/
def wheel2 : Wheel 4 := Finset.univ

theorem prob_odd_sum_is_half :
  prob_odd wheel1 * prob_even wheel2 + prob_even wheel1 * prob_odd wheel2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_sum_is_half_l1482_148211


namespace NUMINAMATH_CALUDE_men_entered_room_l1482_148299

theorem men_entered_room (initial_men initial_women : ℕ) 
  (men_entered : ℕ) (h1 : initial_men * 5 = initial_women * 4) 
  (h2 : initial_men + men_entered = 14) 
  (h3 : 2 * (initial_women - 3) = 24) : men_entered = 2 := by
  sorry

end NUMINAMATH_CALUDE_men_entered_room_l1482_148299


namespace NUMINAMATH_CALUDE_equation_solution_range_l1482_148205

-- Define the equation
def equation (x a : ℝ) : Prop := Real.sqrt (x^2 - 1) = a*x - 2

-- Define the condition of having exactly one solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! x, equation x a

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  (a ∈ Set.Icc (-Real.sqrt 5) (-1) ∪ Set.Ioc 1 (Real.sqrt 5))

-- Theorem statement
theorem equation_solution_range :
  ∀ a : ℝ, has_unique_solution a ↔ a_range a :=
sorry

end NUMINAMATH_CALUDE_equation_solution_range_l1482_148205


namespace NUMINAMATH_CALUDE_system_solution_inequality_solution_l1482_148257

theorem system_solution :
  ∃! (x y : ℝ), (6 * x - 2 * y = 1 ∧ 2 * x + y = 2) ∧
  x = (1/2 : ℝ) ∧ y = 1 := by sorry

theorem inequality_solution :
  ∀ x : ℝ, (2 * x - 10 < 0 ∧ (x + 1) / 3 < x - 1) ↔ (2 < x ∧ x < 5) := by sorry

end NUMINAMATH_CALUDE_system_solution_inequality_solution_l1482_148257


namespace NUMINAMATH_CALUDE_coin_distribution_impossibility_l1482_148203

theorem coin_distribution_impossibility : ∀ n : ℕ,
  n = 44 →
  n < (10 * 9) / 2 :=
by
  sorry

#check coin_distribution_impossibility

end NUMINAMATH_CALUDE_coin_distribution_impossibility_l1482_148203


namespace NUMINAMATH_CALUDE_extended_quad_ratio_gt_one_ratio_always_gt_one_l1482_148209

/-- Represents a convex quadrilateral ABCD with an extended construction --/
structure ExtendedQuadrilateral where
  /-- The sum of all internal angles of the quadrilateral --/
  internal_sum : ℝ
  /-- The sum of angles BAD and ABC --/
  partial_sum : ℝ
  /-- Assumption that the quadrilateral is convex --/
  convex : 0 < partial_sum ∧ partial_sum < internal_sum

/-- The ratio of external angle sum to partial internal angle sum is greater than 1 --/
theorem extended_quad_ratio_gt_one (q : ExtendedQuadrilateral) : 
  q.internal_sum / q.partial_sum > 1 := by
  sorry

/-- Main theorem: For any convex quadrilateral with the given construction, 
    the ratio r is always greater than 1 --/
theorem ratio_always_gt_one : 
  ∀ q : ExtendedQuadrilateral, q.internal_sum / q.partial_sum > 1 := by
  sorry

end NUMINAMATH_CALUDE_extended_quad_ratio_gt_one_ratio_always_gt_one_l1482_148209


namespace NUMINAMATH_CALUDE_decimal_shift_difference_l1482_148282

theorem decimal_shift_difference (x : ℝ) : 10 * x - x / 10 = 23.76 → x = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_decimal_shift_difference_l1482_148282


namespace NUMINAMATH_CALUDE_notebooks_needed_correct_l1482_148236

/-- The number of notebooks needed to achieve a profit of $40 -/
def notebooks_needed : ℕ := 96

/-- The cost of 4 notebooks in dollars -/
def cost_of_four : ℚ := 15

/-- The selling price of 6 notebooks in dollars -/
def sell_price_of_six : ℚ := 25

/-- The desired profit in dollars -/
def desired_profit : ℚ := 40

/-- Theorem stating that the number of notebooks needed to achieve the desired profit is correct -/
theorem notebooks_needed_correct : 
  (notebooks_needed : ℚ) * (sell_price_of_six / 6 - cost_of_four / 4) ≥ desired_profit ∧
  ((notebooks_needed - 1) : ℚ) * (sell_price_of_six / 6 - cost_of_four / 4) < desired_profit :=
by sorry

end NUMINAMATH_CALUDE_notebooks_needed_correct_l1482_148236


namespace NUMINAMATH_CALUDE_trig_identity_proof_l1482_148284

theorem trig_identity_proof : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 1 / (Real.cos (20 * π / 180))^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l1482_148284


namespace NUMINAMATH_CALUDE_not_both_prime_2n_plus_minus_one_l1482_148243

theorem not_both_prime_2n_plus_minus_one (n : ℕ) (h : n > 2) :
  ¬(Nat.Prime (2^n - 1) ∧ Nat.Prime (2^n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_not_both_prime_2n_plus_minus_one_l1482_148243


namespace NUMINAMATH_CALUDE_equation_solutions_l1482_148268

def solutions_7 : Set (ℤ × ℤ) := {(6, -42), (-42, 6), (8, 56), (56, 8), (14, 14)}
def solutions_25 : Set (ℤ × ℤ) := {(24, -600), (-600, 24), (26, 650), (650, 26), (50, 50)}

theorem equation_solutions (a b : ℤ) :
  (1 / a + 1 / b = 1 / 7 → (a, b) ∈ solutions_7) ∧
  (1 / a + 1 / b = 1 / 25 → (a, b) ∈ solutions_25) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1482_148268


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1482_148247

/-- Given a line L1 with equation 4x + 5y - 8 = 0 and a point A(3,2),
    the line L2 passing through A and perpendicular to L1 has equation 4y - 5x + 7 = 0 -/
theorem perpendicular_line_equation (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 4 * x + 5 * y - 8 = 0
  let A : ℝ × ℝ := (3, 2)
  let L2 : ℝ → ℝ → Prop := λ x y => 4 * y - 5 * x + 7 = 0
  (∀ x y, L2 x y ↔ (y - A.2) = -(4/5) * (x - A.1)) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = 4/5) →
  L2 A.1 A.2 ∧ ∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L2 x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = -5/4 :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l1482_148247


namespace NUMINAMATH_CALUDE_unique_digit_sum_l1482_148222

theorem unique_digit_sum (A₁₂ B C D : ℕ) : 
  (∃! (B C D : ℕ), 
    (10 > A₁₂ ∧ A₁₂ > B ∧ B > C ∧ C > D ∧ D > 0) ∧
    (1000 * A₁₂ + 100 * B + 10 * C + D) - (1000 * D + 100 * C + 10 * B + A₁₂) = 
    (1000 * B + 100 * D + 10 * A₁₂ + C)) →
  B + C + D = 11 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_sum_l1482_148222


namespace NUMINAMATH_CALUDE_sum_of_ages_l1482_148264

/-- Proves that the sum of Henry and Jill's present ages is 43 years -/
theorem sum_of_ages (henry_age jill_age : ℕ) : 
  henry_age = 27 →
  jill_age = 16 →
  henry_age - 5 = 2 * (jill_age - 5) →
  henry_age + jill_age = 43 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1482_148264


namespace NUMINAMATH_CALUDE_variation_problem_l1482_148241

theorem variation_problem (c : ℝ) (R₀ S₀ T₀ R₁ S₁ T₁ : ℝ) :
  R₀ = c * S₀^2 / T₀^2 →
  R₀ = 3 →
  S₀ = 1 →
  T₀ = 2 →
  R₁ = 75 →
  T₁ = 5 →
  R₁ = c * S₁^2 / T₁^2 →
  S₁ = 12.5 := by
sorry

end NUMINAMATH_CALUDE_variation_problem_l1482_148241


namespace NUMINAMATH_CALUDE_probability_two_white_same_color_l1482_148245

-- Define the number of white and black balls
def num_white_balls : ℕ := 3
def num_black_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := num_white_balls + num_black_balls

-- Define the number of ways to draw two balls of the same color
def same_color_draws : ℕ := (num_white_balls.choose 2) + (num_black_balls.choose 2)

-- Define the number of ways to draw two white balls
def white_draws : ℕ := num_white_balls.choose 2

-- Theorem: The probability of drawing two white balls, given that the two drawn balls are of the same color, is 3/10
theorem probability_two_white_same_color :
  (white_draws : ℚ) / same_color_draws = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_white_same_color_l1482_148245


namespace NUMINAMATH_CALUDE_red_envelope_prob_is_one_third_l1482_148270

def red_envelope_prob : ℚ :=
  let total_people : ℕ := 4
  let one_yuan_envelopes : ℕ := 3
  let five_yuan_envelopes : ℕ := 1
  let total_events : ℕ := Nat.choose total_people 2
  let favorable_events : ℕ := one_yuan_envelopes * five_yuan_envelopes
  favorable_events / total_events

theorem red_envelope_prob_is_one_third : 
  red_envelope_prob = 1/3 := by sorry

end NUMINAMATH_CALUDE_red_envelope_prob_is_one_third_l1482_148270


namespace NUMINAMATH_CALUDE_sequence_properties_l1482_148260

def a (n : ℕ) : ℤ := n^2 - 7*n + 6

theorem sequence_properties :
  (a 4 = -6) ∧
  (a 16 = 150) ∧
  (∀ n : ℕ, n ≥ 7 → a n > 0) := by
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1482_148260


namespace NUMINAMATH_CALUDE_susanna_purchase_l1482_148259

/-- The cost of each item in pounds and pence -/
structure ItemCost where
  pounds : ℕ
  pence : Fin 100
  pence_eq : pence = 99

/-- The total amount spent by Susanna in pence -/
def total_spent : ℕ := 65 * 100 + 76

/-- The number of items Susanna bought -/
def items_bought : ℕ := 24

theorem susanna_purchase :
  ∀ (cost : ItemCost),
  (cost.pounds * 100 + cost.pence) * items_bought = total_spent :=
sorry

end NUMINAMATH_CALUDE_susanna_purchase_l1482_148259


namespace NUMINAMATH_CALUDE_translate_line_upward_l1482_148200

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically --/
def translateLine (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift }

theorem translate_line_upward (original : Line) (shift : ℝ) :
  original.slope = -2 ∧ shift = 4 →
  translateLine original shift = { slope := -2, intercept := 4 } := by
  sorry

#check translate_line_upward

end NUMINAMATH_CALUDE_translate_line_upward_l1482_148200


namespace NUMINAMATH_CALUDE_homeless_donation_distribution_l1482_148213

theorem homeless_donation_distribution (total spent second_set third_set first_set : ℚ) : 
  total = 900 ∧ second_set = 260 ∧ third_set = 315 ∧ 
  total = first_set + second_set + third_set →
  first_set = 325 := by sorry

end NUMINAMATH_CALUDE_homeless_donation_distribution_l1482_148213


namespace NUMINAMATH_CALUDE_expenditure_recording_l1482_148251

/-- Represents the sign of a financial transaction -/
inductive TransactionSign
| Positive
| Negative

/-- Represents a financial transaction -/
structure Transaction where
  amount : ℕ
  sign : TransactionSign

/-- Records a transaction with the given amount and sign -/
def recordTransaction (amount : ℕ) (sign : TransactionSign) : Transaction :=
  { amount := amount, sign := sign }

/-- The rule for recording incomes and expenditures -/
axiom opposite_signs : 
  ∀ (income expenditure : Transaction), 
    income.sign = TransactionSign.Positive → 
    expenditure.sign = TransactionSign.Negative

/-- The main theorem -/
theorem expenditure_recording 
  (income : Transaction) 
  (h_income : income = recordTransaction 500 TransactionSign.Positive) :
  ∃ (expenditure : Transaction), 
    expenditure = recordTransaction 200 TransactionSign.Negative :=
sorry

end NUMINAMATH_CALUDE_expenditure_recording_l1482_148251


namespace NUMINAMATH_CALUDE_simple_interest_solution_l1482_148276

/-- Simple interest calculation -/
def simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) : Prop :=
  let principal := interest / (rate * time / 100)
  principal = 8935

/-- Theorem stating the solution to the simple interest problem -/
theorem simple_interest_solution :
  simple_interest_problem 4020.75 9 5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_solution_l1482_148276


namespace NUMINAMATH_CALUDE_common_difference_is_two_l1482_148291

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition : a 2 + a 6 = 8
  fifth_term : a 5 = 6

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_is_two (seq : ArithmeticSequence) :
  common_difference seq = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l1482_148291


namespace NUMINAMATH_CALUDE_cubic_equation_unique_solution_l1482_148233

theorem cubic_equation_unique_solution :
  ∃! x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_unique_solution_l1482_148233


namespace NUMINAMATH_CALUDE_last_day_of_second_quarter_365_day_year_l1482_148240

/-- Represents a day in a month -/
structure DayInMonth where
  month : Nat
  day : Nat

/-- Represents a year -/
structure Year where
  totalDays : Nat

/-- Defines the last day of the second quarter for a given year -/
def lastDayOfSecondQuarter (y : Year) : DayInMonth :=
  { month := 6, day := 30 }

/-- Theorem: In a year with 365 days, the last day of the second quarter is June 30 -/
theorem last_day_of_second_quarter_365_day_year :
  ∀ (y : Year), y.totalDays = 365 → lastDayOfSecondQuarter y = { month := 6, day := 30 } := by
  sorry

end NUMINAMATH_CALUDE_last_day_of_second_quarter_365_day_year_l1482_148240


namespace NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l1482_148297

theorem min_value_x_plus_four_over_x :
  ∃ (min : ℝ), min > 0 ∧
  (∀ x : ℝ, x > 0 → x + 4 / x ≥ min) ∧
  (∃ x : ℝ, x > 0 ∧ x + 4 / x = min) :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l1482_148297


namespace NUMINAMATH_CALUDE_ratio_sum_quotient_l1482_148292

theorem ratio_sum_quotient (x y : ℚ) (h : x / y = 1 / 2) : (x + y) / y = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_quotient_l1482_148292


namespace NUMINAMATH_CALUDE_work_completion_time_l1482_148230

-- Define the work rates and time
def work_rate_B : ℚ := 1 / 18
def work_rate_A : ℚ := 2 * work_rate_B
def time_together : ℚ := 6

-- State the theorem
theorem work_completion_time :
  (work_rate_A = 2 * work_rate_B) →
  (work_rate_B = 1 / 18) →
  (time_together * (work_rate_A + work_rate_B) = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1482_148230


namespace NUMINAMATH_CALUDE_birds_flew_up_l1482_148244

/-- The number of birds that flew up to a tree -/
theorem birds_flew_up (initial : ℕ) (total : ℕ) (h1 : initial = 14) (h2 : total = 35) :
  total - initial = 21 := by
  sorry

end NUMINAMATH_CALUDE_birds_flew_up_l1482_148244


namespace NUMINAMATH_CALUDE_min_sum_given_product_minus_sum_l1482_148272

theorem min_sum_given_product_minus_sum (a b : ℝ) 
  (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) : 
  a + b ≥ 2 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_given_product_minus_sum_l1482_148272


namespace NUMINAMATH_CALUDE_quadratic_equation_complete_square_l1482_148294

theorem quadratic_equation_complete_square (m n : ℝ) : 
  (∀ x, 15 * x^2 - 30 * x - 45 = 0 ↔ (x + m)^2 = n) → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_complete_square_l1482_148294


namespace NUMINAMATH_CALUDE_a_2n_is_perfect_square_a_2n_specific_form_l1482_148250

/-- The number of natural numbers with digit sum n, using only digits 1, 3, and 4 -/
def a (n : ℕ) : ℕ :=
  sorry

/-- The main theorem: a₂ₙ is a perfect square for all natural numbers n -/
theorem a_2n_is_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k ^ 2 := by
  sorry

/-- The specific form of a₂ₙ as (aₙ + aₙ₋₂)² -/
theorem a_2n_specific_form (n : ℕ) : a (2 * n) = (a n + a (n - 2)) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_2n_is_perfect_square_a_2n_specific_form_l1482_148250


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l1482_148266

/-- The y-coordinate of the point on the y-axis equidistant from C(-3, 0) and D(-2, 5) is 2 -/
theorem equidistant_point_y_coordinate : 
  ∃ y : ℝ, ((-3 : ℝ)^2 + 0^2 + 0^2 + y^2 = (-2 : ℝ)^2 + 5^2 + 0^2 + (y - 5)^2) ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l1482_148266


namespace NUMINAMATH_CALUDE_normal_vector_of_l_l1482_148271

/-- Definition of the line l: 2x - 3y + 4 = 0 -/
def l (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0

/-- Definition of a normal vector to a line -/
def is_normal_vector (v : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v = (k * 2, k * (-3))

/-- Theorem: (4, -6) is a normal vector to the line l -/
theorem normal_vector_of_l : is_normal_vector (4, -6) l := by
  sorry

end NUMINAMATH_CALUDE_normal_vector_of_l_l1482_148271


namespace NUMINAMATH_CALUDE_complex_power_sum_l1482_148253

theorem complex_power_sum (w : ℂ) (hw : w^2 - w + 1 = 0) : 
  w^103 + w^104 + w^105 + w^106 + w^107 = -1 := by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1482_148253


namespace NUMINAMATH_CALUDE_type_b_first_is_better_l1482_148283

/-- Represents the score for a correct answer to a question type -/
def score (questionType : Bool) : ℝ :=
  if questionType then 80 else 20

/-- Represents the probability of correctly answering a question type -/
def probability (questionType : Bool) : ℝ :=
  if questionType then 0.6 else 0.8

/-- Calculates the expected score when choosing a specific question type first -/
def expectedScore (firstQuestionType : Bool) : ℝ :=
  let p1 := probability firstQuestionType
  let p2 := probability (!firstQuestionType)
  let s1 := score firstQuestionType
  let s2 := score (!firstQuestionType)
  p1 * s1 + p1 * p2 * s2

/-- Theorem stating that choosing type B questions first yields a higher expected score -/
theorem type_b_first_is_better :
  expectedScore true > expectedScore false :=
sorry

end NUMINAMATH_CALUDE_type_b_first_is_better_l1482_148283


namespace NUMINAMATH_CALUDE_total_cleaner_needed_l1482_148256

/-- Amount of cleaner needed for a dog stain in ounces -/
def dog_cleaner : ℕ := 6

/-- Amount of cleaner needed for a cat stain in ounces -/
def cat_cleaner : ℕ := 4

/-- Amount of cleaner needed for a rabbit stain in ounces -/
def rabbit_cleaner : ℕ := 1

/-- Number of dogs -/
def num_dogs : ℕ := 6

/-- Number of cats -/
def num_cats : ℕ := 3

/-- Number of rabbits -/
def num_rabbits : ℕ := 1

/-- Theorem stating the total amount of cleaner needed -/
theorem total_cleaner_needed : 
  dog_cleaner * num_dogs + cat_cleaner * num_cats + rabbit_cleaner * num_rabbits = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_cleaner_needed_l1482_148256


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_450_l1482_148216

theorem distinct_prime_factors_of_450 : Nat.card (Nat.factors 450).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_450_l1482_148216


namespace NUMINAMATH_CALUDE_ninety_squared_l1482_148267

theorem ninety_squared : 90 * 90 = 8100 := by
  sorry

end NUMINAMATH_CALUDE_ninety_squared_l1482_148267


namespace NUMINAMATH_CALUDE_simplify_expression_l1482_148279

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 - 2*b + 1) - 2*b^2 = 9*b^3 - 8*b^2 + 3*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1482_148279


namespace NUMINAMATH_CALUDE_angle_Q_measure_l1482_148214

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

-- Define the extended sides and point Q
def extended_sides (octagon : RegularOctagon) : sorry := sorry

def point_Q (octagon : RegularOctagon) : ℝ × ℝ := sorry

-- Define the angle at Q
def angle_Q (octagon : RegularOctagon) : ℝ := sorry

-- Theorem statement
theorem angle_Q_measure (octagon : RegularOctagon) : 
  angle_Q octagon = 22.5 := by sorry

end NUMINAMATH_CALUDE_angle_Q_measure_l1482_148214


namespace NUMINAMATH_CALUDE_aftershave_alcohol_percentage_l1482_148296

/-- Proves that the initial alcohol percentage in an after-shave lotion is 30% -/
theorem aftershave_alcohol_percentage
  (initial_volume : ℝ)
  (water_volume : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 50)
  (h2 : water_volume = 30)
  (h3 : final_percentage = 18.75)
  (h4 : (initial_volume * x / 100) = ((initial_volume + water_volume) * final_percentage / 100)) :
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_aftershave_alcohol_percentage_l1482_148296


namespace NUMINAMATH_CALUDE_dave_winfield_home_runs_l1482_148201

/-- Dave Winfield's career home run count -/
def dave_winfield_hr : ℕ := 465

/-- Hank Aaron's career home run count -/
def hank_aaron_hr : ℕ := 755

/-- Theorem stating Dave Winfield's home run count based on the given conditions -/
theorem dave_winfield_home_runs :
  dave_winfield_hr = 465 ∧
  hank_aaron_hr = 2 * dave_winfield_hr - 175 :=
by sorry

end NUMINAMATH_CALUDE_dave_winfield_home_runs_l1482_148201


namespace NUMINAMATH_CALUDE_expression_evaluation_l1482_148204

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 3 * x + y / 3 ≠ 0) :
  (3 * x + y / 3)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹) = (x * y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1482_148204


namespace NUMINAMATH_CALUDE_square_position_after_2007_transformations_l1482_148215

/-- Represents the vertices of a square in clockwise order -/
inductive SquarePosition
  | ABCD
  | DABC
  | CBAD
  | DCBA

/-- Applies one full cycle of transformations to a square -/
def applyTransformationCycle (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.ABCD
  | SquarePosition.DABC => SquarePosition.DABC
  | SquarePosition.CBAD => SquarePosition.CBAD
  | SquarePosition.DCBA => SquarePosition.DCBA

/-- Applies n cycles of transformations to a square -/
def applyNCycles (pos : SquarePosition) (n : Nat) : SquarePosition :=
  match n with
  | 0 => pos
  | n + 1 => applyNCycles (applyTransformationCycle pos) n

/-- Applies a specific number of individual transformations to a square -/
def applyTransformations (pos : SquarePosition) (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => pos
  | 1 => match pos with
         | SquarePosition.ABCD => SquarePosition.DABC
         | SquarePosition.DABC => SquarePosition.CBAD
         | SquarePosition.CBAD => SquarePosition.DCBA
         | SquarePosition.DCBA => SquarePosition.ABCD
  | 2 => match pos with
         | SquarePosition.ABCD => SquarePosition.CBAD
         | SquarePosition.DABC => SquarePosition.DCBA
         | SquarePosition.CBAD => SquarePosition.ABCD
         | SquarePosition.DCBA => SquarePosition.DABC
  | 3 => match pos with
         | SquarePosition.ABCD => SquarePosition.DCBA
         | SquarePosition.DABC => SquarePosition.ABCD
         | SquarePosition.CBAD => SquarePosition.DABC
         | SquarePosition.DCBA => SquarePosition.CBAD
  | _ => pos  -- This case should never occur due to % 4

theorem square_position_after_2007_transformations :
  applyTransformations SquarePosition.ABCD 2007 = SquarePosition.DCBA :=
by sorry

end NUMINAMATH_CALUDE_square_position_after_2007_transformations_l1482_148215


namespace NUMINAMATH_CALUDE_circle_radius_in_rectangle_l1482_148239

/-- A rectangle with length 10 and width 6 -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (length_eq : length = 10)
  (width_eq : width = 6)

/-- A circle passing through two vertices of the rectangle and tangent to the opposite side -/
structure Circle (rect : Rectangle) :=
  (radius : ℝ)
  (passes_through_vertices : Bool)
  (tangent_to_opposite_side : Bool)

/-- The theorem stating that the radius of the circle is 3 -/
theorem circle_radius_in_rectangle (rect : Rectangle) (circ : Circle rect) :
  circ.passes_through_vertices ∧ circ.tangent_to_opposite_side → circ.radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_in_rectangle_l1482_148239


namespace NUMINAMATH_CALUDE_hyperbola_point_distance_to_x_axis_l1482_148249

/-- The distance from a point on a hyperbola to the x-axis, given specific conditions -/
theorem hyperbola_point_distance_to_x_axis 
  (P : ℝ × ℝ) 
  (F₁ F₂ : ℝ × ℝ) 
  (h_hyperbola : (P.1^2 / 16) - (P.2^2 / 9) = 1) 
  (h_on_hyperbola : P ∈ {p : ℝ × ℝ | (p.1^2 / 16) - (p.2^2 / 9) = 1}) 
  (h_focal_points : F₁ ∈ {f : ℝ × ℝ | (f.1^2 / 16) - (f.2^2 / 9) = 1} ∧ 
                    F₂ ∈ {f : ℝ × ℝ | (f.1^2 / 16) - (f.2^2 / 9) = 1}) 
  (h_perpendicular : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0) : 
  |P.2| = 9/5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_point_distance_to_x_axis_l1482_148249


namespace NUMINAMATH_CALUDE_distance_traveled_l1482_148254

/-- Represents the actual distance traveled in kilometers -/
def actual_distance : ℝ := 33.75

/-- Represents the initial walking speed in km/hr -/
def initial_speed : ℝ := 15

/-- Represents the faster walking speed in km/hr -/
def faster_speed : ℝ := 35

/-- Represents the fraction of the distance that is uphill -/
def uphill_fraction : ℝ := 0.6

/-- Represents the decrease in speed for uphill portion -/
def uphill_speed_decrease : ℝ := 0.1

/-- Represents the additional distance covered at faster speed -/
def additional_distance : ℝ := 45

theorem distance_traveled :
  ∃ (time : ℝ),
    actual_distance = initial_speed * time ∧
    actual_distance + additional_distance = faster_speed * time ∧
    actual_distance * uphill_fraction = (faster_speed * (1 - uphill_speed_decrease)) * (time * uphill_fraction) ∧
    actual_distance * (1 - uphill_fraction) = faster_speed * (time * (1 - uphill_fraction)) :=
by sorry

end NUMINAMATH_CALUDE_distance_traveled_l1482_148254


namespace NUMINAMATH_CALUDE_face_card_proportion_l1482_148225

theorem face_card_proportion (p : ℝ) : 
  (p ≥ 0) → (p ≤ 1) → (1 - (1 - p)^3 = 19/27) → p = 1/3 := by
sorry

end NUMINAMATH_CALUDE_face_card_proportion_l1482_148225


namespace NUMINAMATH_CALUDE_min_distance_to_perpendicular_bisector_l1482_148277

open Complex

theorem min_distance_to_perpendicular_bisector (z : ℂ) :
  (abs z = abs (z + 2 + 2*I)) →
  (∃ (min_val : ℝ), ∀ (w : ℂ), abs w = abs (w + 2 + 2*I) → abs (w - 1 + I) ≥ min_val) ∧
  (∃ (z₀ : ℂ), abs z₀ = abs (z₀ + 2 + 2*I) ∧ abs (z₀ - 1 + I) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_perpendicular_bisector_l1482_148277


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l1482_148280

theorem hot_dogs_remainder : 25197629 % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l1482_148280


namespace NUMINAMATH_CALUDE_race_time_l1482_148234

/-- In a 1000-meter race, runner A beats runner B by 48 meters or 6 seconds -/
def Race (t : ℝ) : Prop :=
  -- A's distance in t seconds
  1000 = t * (1000 / t) ∧
  -- B's distance in t seconds
  952 = t * (952 / (t + 6)) ∧
  -- A and B have the same speed
  1000 / t = 952 / (t + 6)

/-- The time taken by runner A to complete the race is 125 seconds -/
theorem race_time : ∃ t : ℝ, Race t ∧ t = 125 := by sorry

end NUMINAMATH_CALUDE_race_time_l1482_148234


namespace NUMINAMATH_CALUDE_max_visible_blue_cubes_l1482_148261

/-- Represents a column of cubes with red and blue colors -/
structure CubeColumn :=
  (total : Nat)
  (blue : Nat)
  (red : Nat)
  (h_sum : blue + red = total)

/-- Represents a row of three columns on the board -/
structure BoardRow :=
  (left : CubeColumn)
  (middle : CubeColumn)
  (right : CubeColumn)

/-- The entire 3x3 board configuration -/
structure Board :=
  (front : BoardRow)
  (middle : BoardRow)
  (back : BoardRow)

/-- Calculates the maximum number of visible blue cubes in a row -/
def maxVisibleBlueInRow (row : BoardRow) : Nat :=
  row.left.blue + max 0 (row.middle.total - row.left.total) + max 0 (row.right.total - max row.left.total row.middle.total)

/-- The main theorem stating the maximum number of visible blue cubes -/
theorem max_visible_blue_cubes (board : Board) : 
  maxVisibleBlueInRow board.front + maxVisibleBlueInRow board.middle + maxVisibleBlueInRow board.back ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_max_visible_blue_cubes_l1482_148261


namespace NUMINAMATH_CALUDE_solve_for_z_l1482_148290

theorem solve_for_z (x y z : ℚ) 
  (h1 : x = 11)
  (h2 : y = 8)
  (h3 : 2 * x + 3 * z = 5 * y) :
  z = 6 := by
sorry

end NUMINAMATH_CALUDE_solve_for_z_l1482_148290


namespace NUMINAMATH_CALUDE_surface_area_union_cones_l1482_148263

/-- The surface area of the union of two right cones with specific dimensions -/
theorem surface_area_union_cones (r h : ℝ) (hr : r = 4) (hh : h = 3) :
  let L := Real.sqrt (r^2 + h^2)
  let surface_area_one_cone := π * r^2 + π * r * L
  let lateral_area_half_cone := π * (r/2) * (Real.sqrt ((r/2)^2 + (h/2)^2))
  2 * (surface_area_one_cone - lateral_area_half_cone) = 62 * π :=
by sorry

end NUMINAMATH_CALUDE_surface_area_union_cones_l1482_148263


namespace NUMINAMATH_CALUDE_sqrt_x_plus_y_equals_three_l1482_148286

theorem sqrt_x_plus_y_equals_three (x y : ℝ) (h : y = 4 + Real.sqrt (5 - x) + Real.sqrt (x - 5)) : 
  Real.sqrt (x + y) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_y_equals_three_l1482_148286


namespace NUMINAMATH_CALUDE_range_of_a_l1482_148281

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = B a → -2 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1482_148281


namespace NUMINAMATH_CALUDE_quadratic_intersection_l1482_148229

theorem quadratic_intersection
  (a b c d h : ℝ)
  (h_a : a ≠ 0)
  (h_b : b ≠ 0)
  (h_h : h ≠ 0)
  (h_d : d ≠ c) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := a * (x - h)^2 + b * (x - h) + d
  ∃ x y : ℝ, f x = g x ∧ f x = y ∧ x = (d - c) / b ∧ y = a * ((d - c) / b)^2 + d :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l1482_148229


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l1482_148275

theorem polynomial_division_degree (f d q r : Polynomial ℝ) : 
  Polynomial.degree f = 15 →
  f = d * q + r →
  Polynomial.degree q = 8 →
  r = 5 * X^2 + 3 * X - 9 →
  Polynomial.degree d = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l1482_148275


namespace NUMINAMATH_CALUDE_binomial_not_divisible_l1482_148232

theorem binomial_not_divisible (P n k : ℕ) (hP : P > 1) :
  ∃ i ∈ Finset.range (k + 1), ¬(P ∣ Nat.choose (n + i) k) := by
  sorry

end NUMINAMATH_CALUDE_binomial_not_divisible_l1482_148232


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_and_half_l1482_148212

theorem reciprocal_of_negative_three_and_half (x : ℚ) :
  x = -3.5 → (1 / x) = -2/7 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_and_half_l1482_148212


namespace NUMINAMATH_CALUDE_fraction_equality_l1482_148274

theorem fraction_equality (a b : ℝ) (h : (2*a - b) / (a + b) = 3/4) : b / a = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1482_148274


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1482_148242

theorem complex_magnitude_problem (z₁ z₂ : ℂ) : 
  z₁ = -1 + I → z₁ * z₂ = -2 → Complex.abs (z₂ + 2*I) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1482_148242


namespace NUMINAMATH_CALUDE_neznaika_expression_problem_l1482_148287

theorem neznaika_expression_problem :
  ∃ (f : ℝ → ℝ → ℝ → ℝ),
    (∀ x y z, f x y z = x / (y - Real.sqrt z)) →
    f 20 2 2 > 30 := by
  sorry

end NUMINAMATH_CALUDE_neznaika_expression_problem_l1482_148287
