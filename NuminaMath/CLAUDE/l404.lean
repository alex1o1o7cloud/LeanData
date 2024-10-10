import Mathlib

namespace license_plate_sampling_is_systematic_l404_40410

/-- Represents a car's license plate --/
structure LicensePlate where
  number : ℕ

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Lottery
  | Systematic
  | Stratified

/-- Function to check if a license plate ends with a specific digit --/
def endsWithDigit (plate : LicensePlate) (digit : ℕ) : Prop :=
  plate.number % 10 = digit

/-- Definition of systematic sampling for this context --/
def isSystematicSampling (sample : Set LicensePlate) (digit : ℕ) : Prop :=
  ∀ plate, plate ∈ sample ↔ endsWithDigit plate digit

/-- Theorem stating that selecting cars with license plates ending in a specific digit
    is equivalent to systematic sampling --/
theorem license_plate_sampling_is_systematic (sample : Set LicensePlate) (digit : ℕ) :
  (∀ plate, plate ∈ sample ↔ endsWithDigit plate digit) →
  isSystematicSampling sample digit :=
by sorry

end license_plate_sampling_is_systematic_l404_40410


namespace floor_ceiling_sum_l404_40442

theorem floor_ceiling_sum : ⌊(-3.87 : ℝ)⌋ + ⌈(30.75 : ℝ)⌉ = 27 := by
  sorry

end floor_ceiling_sum_l404_40442


namespace households_without_car_or_bike_l404_40451

theorem households_without_car_or_bike 
  (total : ℕ) 
  (both : ℕ) 
  (with_car : ℕ) 
  (bike_only : ℕ) 
  (h1 : total = 90) 
  (h2 : both = 22) 
  (h3 : with_car = 44) 
  (h4 : bike_only = 35) : 
  total - (with_car + bike_only + both - both) = 11 := by
  sorry

end households_without_car_or_bike_l404_40451


namespace gcd_1729_867_l404_40422

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 := by
  sorry

end gcd_1729_867_l404_40422


namespace four_numbers_theorem_l404_40482

theorem four_numbers_theorem (a b c d : ℕ) : 
  a + b + c = 17 ∧ 
  a + b + d = 21 ∧ 
  a + c + d = 25 ∧ 
  b + c + d = 30 → 
  (a = 14 ∧ b = 10 ∧ c = 6 ∧ d = 1) ∨
  (a = 14 ∧ b = 10 ∧ c = 1 ∧ d = 6) ∨
  (a = 14 ∧ b = 6 ∧ c = 10 ∧ d = 1) ∨
  (a = 14 ∧ b = 6 ∧ c = 1 ∧ d = 10) ∨
  (a = 14 ∧ b = 1 ∧ c = 10 ∧ d = 6) ∨
  (a = 14 ∧ b = 1 ∧ c = 6 ∧ d = 10) ∨
  (a = 10 ∧ b = 14 ∧ c = 6 ∧ d = 1) ∨
  (a = 10 ∧ b = 14 ∧ c = 1 ∧ d = 6) ∨
  (a = 10 ∧ b = 6 ∧ c = 14 ∧ d = 1) ∨
  (a = 10 ∧ b = 6 ∧ c = 1 ∧ d = 14) ∨
  (a = 10 ∧ b = 1 ∧ c = 14 ∧ d = 6) ∨
  (a = 10 ∧ b = 1 ∧ c = 6 ∧ d = 14) ∨
  (a = 6 ∧ b = 14 ∧ c = 10 ∧ d = 1) ∨
  (a = 6 ∧ b = 14 ∧ c = 1 ∧ d = 10) ∨
  (a = 6 ∧ b = 10 ∧ c = 14 ∧ d = 1) ∨
  (a = 6 ∧ b = 10 ∧ c = 1 ∧ d = 14) ∨
  (a = 6 ∧ b = 1 ∧ c = 14 ∧ d = 10) ∨
  (a = 6 ∧ b = 1 ∧ c = 10 ∧ d = 14) ∨
  (a = 1 ∧ b = 14 ∧ c = 10 ∧ d = 6) ∨
  (a = 1 ∧ b = 14 ∧ c = 6 ∧ d = 10) ∨
  (a = 1 ∧ b = 10 ∧ c = 14 ∧ d = 6) ∨
  (a = 1 ∧ b = 10 ∧ c = 6 ∧ d = 14) ∨
  (a = 1 ∧ b = 6 ∧ c = 14 ∧ d = 10) ∨
  (a = 1 ∧ b = 6 ∧ c = 10 ∧ d = 14) :=
by sorry


end four_numbers_theorem_l404_40482


namespace inequality_solution_l404_40474

theorem inequality_solution (x : ℝ) : 
  (x^3 - 4*x) / (x^2 - 4*x + 4) > 0 ↔ (x > -2 ∧ x < 0) ∨ x > 2 :=
by sorry

end inequality_solution_l404_40474


namespace union_of_A_and_B_l404_40409

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by sorry

end union_of_A_and_B_l404_40409


namespace chord_length_line_circle_intersection_specific_chord_length_l404_40475

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length_line_circle_intersection 
  (line : ℝ → ℝ → Prop) 
  (circle : ℝ → ℝ → Prop) : ℝ :=
by
  sorry

/-- Main theorem: The length of the chord formed by the intersection of 
    x + √3 y - 2 = 0 and x² + y² = 4 is 2√3 -/
theorem specific_chord_length : 
  chord_length_line_circle_intersection 
    (fun x y => x + Real.sqrt 3 * y - 2 = 0) 
    (fun x y => x^2 + y^2 = 4) = 2 * Real.sqrt 3 :=
by
  sorry

end chord_length_line_circle_intersection_specific_chord_length_l404_40475


namespace set_intersection_union_theorem_l404_40477

def A : Set ℝ := {x | 2*x - x^2 ≤ x}
def B : Set ℝ := {x | x/(1-x) ≤ x/(1-x)}
def C (a b : ℝ) : Set ℝ := {x | a*x^2 + x + b < 0}

theorem set_intersection_union_theorem (a b : ℝ) :
  (A ∪ B) ∩ (C a b) = ∅ ∧ (A ∪ B) ∪ (C a b) = Set.univ →
  a = -1/3 ∧ b = 0 := by
  sorry

end set_intersection_union_theorem_l404_40477


namespace largest_multiple_of_11_below_negative_150_l404_40404

theorem largest_multiple_of_11_below_negative_150 :
  ∀ n : ℤ, n * 11 < -150 → n * 11 ≤ -154 :=
by
  sorry

end largest_multiple_of_11_below_negative_150_l404_40404


namespace log_difference_equals_negative_two_l404_40466

-- Define the common logarithm (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_difference_equals_negative_two :
  log10 (1/4) - log10 25 = -2 := by sorry

end log_difference_equals_negative_two_l404_40466


namespace count_valid_triangles_l404_40467

/-- A triangle with integral side lengths --/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  sum_eq_12 : a + b + c = 12
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of all valid IntTriangles --/
def validTriangles : Finset IntTriangle := sorry

theorem count_valid_triangles : Finset.card validTriangles = 6 := by sorry

end count_valid_triangles_l404_40467


namespace square_sum_minus_triple_product_l404_40497

theorem square_sum_minus_triple_product (x y : ℝ) 
  (h1 : x * y = 3) 
  (h2 : x + y = 4) : 
  x^2 + y^2 - 3*x*y = 1 := by
sorry

end square_sum_minus_triple_product_l404_40497


namespace quadratic_function_properties_l404_40429

-- Define the quadratic function
def f (x : ℝ) : ℝ := -6 * x^2 + 36 * x - 48

-- State the theorem
theorem quadratic_function_properties :
  f 2 = 0 ∧ f 4 = 0 ∧ f 3 = 6 := by
  sorry

end quadratic_function_properties_l404_40429


namespace negation_of_implication_l404_40456

theorem negation_of_implication (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end negation_of_implication_l404_40456


namespace monomial_sum_l404_40414

/-- Given constants a and b, if the sum of 4xy^2, axy^b, and -5xy is a monomial, 
    then a+b = -2 or a+b = 6 -/
theorem monomial_sum (a b : ℝ) : 
  (∃ (x y : ℝ), ∀ (z : ℝ), z = 4*x*y^2 + a*x*y^b - 5*x*y → ∃ (c : ℝ), z = c*x*y^k) → 
  a + b = -2 ∨ a + b = 6 :=
sorry

end monomial_sum_l404_40414


namespace two_solutions_system_l404_40420

theorem two_solutions_system (x y : ℝ) : 
  (x = 3 * x^2 + y^2 ∧ y = 3 * x * y) → 
  (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y ∧ p.1 = 3 * p.1^2 + p.2^2 ∧ p.2 = 3 * p.1 * p.2) ∧
  (∃! q : ℝ × ℝ, q ≠ p ∧ q.1 = x ∧ q.2 = y ∧ q.1 = 3 * q.1^2 + q.2^2 ∧ q.2 = 3 * q.1 * q.2) ∧
  (∀ r : ℝ × ℝ, r ≠ p ∧ r ≠ q → ¬(r.1 = 3 * r.1^2 + r.2^2 ∧ r.2 = 3 * r.1 * r.2)) :=
by sorry

end two_solutions_system_l404_40420


namespace cargo_realization_time_l404_40436

/-- Represents the speed of a boat in still water -/
structure BoatSpeed where
  speed : ℝ
  positive : speed > 0

/-- Represents the current speed of the river -/
structure RiverCurrent where
  speed : ℝ

/-- Represents a boat on the river -/
structure Boat where
  speed : BoatSpeed
  position : ℝ
  direction : Bool  -- True for downstream, False for upstream

/-- The time it takes for Boat 1 to realize its cargo is missing -/
def timeToCargo (boat1 : Boat) (boat2 : Boat) (river : RiverCurrent) : ℝ :=
  sorry

/-- Theorem stating that the time taken for Boat 1 to realize its cargo is missing is 40 minutes -/
theorem cargo_realization_time
  (boat1 : Boat)
  (boat2 : Boat)
  (river : RiverCurrent)
  (h1 : boat1.speed.speed = 2 * boat2.speed.speed)
  (h2 : boat1.direction = false)  -- Boat 1 starts upstream
  (h3 : boat2.direction = true)   -- Boat 2 starts downstream
  (h4 : ∃ (t : ℝ), t > 0 ∧ t < timeToCargo boat1 boat2 river ∧ 
        boat1.position + t * (boat1.speed.speed + river.speed) = 
        boat2.position + t * (boat2.speed.speed - river.speed))  -- Boats meet before cargo realization
  (h5 : ∃ (t : ℝ), t = 20 ∧ 
        boat1.position + t * (boat1.speed.speed + river.speed) = 
        boat2.position + t * (boat2.speed.speed - river.speed))  -- Boats meet at 20 minutes
  : timeToCargo boat1 boat2 river = 40 := by
  sorry

end cargo_realization_time_l404_40436


namespace is_root_of_polynomial_l404_40492

theorem is_root_of_polynomial (x : ℝ) : 
  x = 4 → x^3 - 5*x^2 + 7*x - 12 = 0 := by
  sorry

end is_root_of_polynomial_l404_40492


namespace dans_song_book_cost_l404_40489

/-- The cost of Dan's song book is equal to the total amount spent at the music store
    minus the cost of the clarinet. -/
theorem dans_song_book_cost (clarinet_cost total_spent : ℚ) 
  (h1 : clarinet_cost = 130.30)
  (h2 : total_spent = 141.54) :
  total_spent - clarinet_cost = 11.24 := by
  sorry

end dans_song_book_cost_l404_40489


namespace smallest_factor_perfect_square_l404_40472

theorem smallest_factor_perfect_square : 
  (∀ k : ℕ, k < 14 → ¬ ∃ m : ℕ, 3150 * k = m * m) ∧ 
  ∃ n : ℕ, 3150 * 14 = n * n := by
  sorry

end smallest_factor_perfect_square_l404_40472


namespace parallel_lines_theorem_l404_40413

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

/-- Slope of the line ax + y - 1 - a = 0 -/
def slope1 (a : ℝ) : ℝ := -a

/-- Slope of the line x - 1/2y = 0 -/
def slope2 : ℝ := 2

/-- Theorem: If ax + y - 1 - a = 0 is parallel to x - 1/2y = 0, then a = -2 -/
theorem parallel_lines_theorem (a : ℝ) : 
  parallel_lines (slope1 a) slope2 → a = -2 := by
  sorry

end parallel_lines_theorem_l404_40413


namespace tvs_on_auction_site_l404_40450

def tvs_in_person : ℕ := 8
def tvs_online_multiplier : ℕ := 3
def total_tvs : ℕ := 42

theorem tvs_on_auction_site :
  let tvs_online := tvs_online_multiplier * tvs_in_person
  let tvs_before_auction := tvs_in_person + tvs_online
  total_tvs - tvs_before_auction = 10 := by
sorry

end tvs_on_auction_site_l404_40450


namespace passing_percentage_l404_40469

theorem passing_percentage (student_marks : ℕ) (failed_by : ℕ) (max_marks : ℕ) 
  (h1 : student_marks = 175)
  (h2 : failed_by = 56)
  (h3 : max_marks = 700) :
  (((student_marks + failed_by : ℚ) / max_marks) * 100).floor = 33 := by
sorry

end passing_percentage_l404_40469


namespace tennis_tournament_has_three_cycle_l404_40417

/-- Represents a tennis tournament as a directed graph -/
structure TennisTournament where
  -- The set of participants
  V : Type
  -- The "wins against" relation
  E : V → V → Prop
  -- There are at least three participants
  atleastThree : ∃ (a b c : V), a ≠ b ∧ b ≠ c ∧ a ≠ c
  -- Every participant plays against every other participant exactly once
  complete : ∀ (a b : V), a ≠ b → (E a b ∨ E b a) ∧ ¬(E a b ∧ E b a)
  -- Every participant wins at least one match
  hasWin : ∀ (a : V), ∃ (b : V), E a b

/-- A 3-cycle in the tournament -/
def HasThreeCycle (T : TennisTournament) : Prop :=
  ∃ (a b c : T.V), T.E a b ∧ T.E b c ∧ T.E c a

/-- The main theorem: every tennis tournament has a 3-cycle -/
theorem tennis_tournament_has_three_cycle (T : TennisTournament) : HasThreeCycle T := by
  sorry

end tennis_tournament_has_three_cycle_l404_40417


namespace expression_simplification_l404_40496

theorem expression_simplification (x : ℝ) : 
  3 * x - 7 * x^2 + 5 - (6 - 5 * x + 7 * x^2) = -14 * x^2 + 8 * x - 1 := by
  sorry

end expression_simplification_l404_40496


namespace isosceles_triangles_in_right_triangle_l404_40498

theorem isosceles_triangles_in_right_triangle :
  ∀ (a b c : ℝ) (S₁ S₂ S₃ : ℝ) (x : ℝ),
    a = 1 →
    b = Real.sqrt 3 →
    c^2 = a^2 + b^2 →
    S₁ + S₂ + S₃ = (1/2) * a * b →
    S₁ = (1/2) * (a/3) * x →
    S₂ = (1/2) * (b/3) * x →
    S₃ = (1/2) * (c/3) * x →
    x = Real.sqrt 109 / 6 :=
by
  sorry

end isosceles_triangles_in_right_triangle_l404_40498


namespace fermat_number_units_digit_F5_l404_40460

theorem fermat_number_units_digit_F5 :
  (2^(2^5) + 1) % 10 = 7 := by
  sorry

end fermat_number_units_digit_F5_l404_40460


namespace inequality_proof_l404_40444

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end inequality_proof_l404_40444


namespace correct_division_result_l404_40424

theorem correct_division_result (student_divisor student_quotient correct_divisor : ℕ) 
  (h1 : student_divisor = 63)
  (h2 : student_quotient = 24)
  (h3 : correct_divisor = 36) :
  (student_divisor * student_quotient) / correct_divisor = 42 :=
by sorry

end correct_division_result_l404_40424


namespace triangle_ratio_l404_40490

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  2 * (Real.cos (A / 2))^2 = (Real.sqrt 3 / 3) * Real.sin A →
  Real.sin (B - C) = 4 * Real.cos B * Real.sin C →
  b / c = 1 + Real.sqrt 6 := by
sorry

end triangle_ratio_l404_40490


namespace reciprocal_sum_quarters_fifths_l404_40468

theorem reciprocal_sum_quarters_fifths : (1 / (1 / 4 + 1 / 5) : ℚ) = 20 / 9 := by
  sorry

end reciprocal_sum_quarters_fifths_l404_40468


namespace solution_set_l404_40440

theorem solution_set (y : ℝ) : 2 ≤ y / (3 * y - 4) ∧ y / (3 * y - 4) < 5 ↔ 10 / 7 < y ∧ y ≤ 8 / 5 := by
  sorry

end solution_set_l404_40440


namespace tricia_age_l404_40443

theorem tricia_age (tricia amilia yorick eugene khloe rupert vincent : ℕ) : 
  tricia = amilia / 3 →
  amilia = yorick / 4 →
  ∃ k : ℕ, yorick = k * eugene →
  khloe = eugene / 3 →
  rupert = khloe + 10 →
  rupert = vincent - 2 →
  vincent = 22 →
  tricia = 5 →
  tricia = 5 := by sorry

end tricia_age_l404_40443


namespace identify_burned_bulb_l404_40441

/-- Represents the time in seconds for screwing or unscrewing a bulb -/
def operation_time : ℕ := 10

/-- Represents the number of bulbs in the series -/
def num_bulbs : ℕ := 4

/-- Represents the minimum time to identify the burned-out bulb -/
def min_identification_time : ℕ := 60

/-- Theorem stating that the minimum time to identify the burned-out bulb is 60 seconds -/
theorem identify_burned_bulb :
  ∀ (burned_bulb_position : Fin num_bulbs),
  min_identification_time = operation_time * (2 * (num_bulbs - 1)) :=
by sorry

end identify_burned_bulb_l404_40441


namespace remainder_sum_l404_40433

theorem remainder_sum (a b : ℤ) (ha : a % 70 = 64) (hb : b % 105 = 99) :
  (a + b) % 35 = 23 := by
  sorry

end remainder_sum_l404_40433


namespace news_spread_time_correct_total_time_l404_40495

/-- The number of people in the city -/
def city_population : ℕ := 3000000

/-- The time interval in minutes for each round of information spreading -/
def time_interval : ℕ := 10

/-- The number of new people informed by each person in one interval -/
def spread_rate : ℕ := 2

/-- The total number of people who know the news after k intervals -/
def people_informed (k : ℕ) : ℕ := 2^(k+1) - 1

/-- The minimum number of intervals needed to inform the entire city -/
def min_intervals : ℕ := 21

theorem news_spread_time :
  (people_informed min_intervals ≥ city_population) ∧
  (∀ k < min_intervals, people_informed k < city_population) :=
sorry

/-- The total time needed to inform the entire city in minutes -/
def total_time : ℕ := min_intervals * time_interval

theorem correct_total_time : total_time = 210 :=
sorry

end news_spread_time_correct_total_time_l404_40495


namespace gcd_of_specific_numbers_l404_40428

theorem gcd_of_specific_numbers : Nat.gcd 33333333 666666666 = 2 := by sorry

end gcd_of_specific_numbers_l404_40428


namespace johnsonville_marching_band_max_members_l404_40458

theorem johnsonville_marching_band_max_members :
  ∀ n : ℕ,
  (∃ k : ℕ, 15 * n = 30 * k + 6) →
  15 * n < 900 →
  (∀ m : ℕ, (∃ j : ℕ, 15 * m = 30 * j + 6) → 15 * m < 900 → 15 * m ≤ 15 * n) →
  15 * n = 810 :=
by sorry

end johnsonville_marching_band_max_members_l404_40458


namespace right_triangle_equality_l404_40438

theorem right_triangle_equality (a b c p : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : a^2 + b^2 = c^2) (h5 : 2*p = a + b + c) : 
  let S := (1/2) * a * b
  p * (p - c) = (p - a) * (p - b) ∧ p * (p - c) = S := by
  sorry

end right_triangle_equality_l404_40438


namespace solution_set_of_inequality_l404_40401

theorem solution_set_of_inequality (x : ℝ) :
  (2 / (x - 3) ≤ 5) ↔ (x < 3 ∨ x ≥ 17/5) :=
by sorry

end solution_set_of_inequality_l404_40401


namespace cube_surface_area_increase_l404_40421

/-- If each edge of a cube increases by 20%, the surface area of the cube increases by 44%. -/
theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_edge := 1.2 * L
  let new_area := 6 * new_edge^2
  (new_area - original_area) / original_area = 0.44 := by
  sorry

end cube_surface_area_increase_l404_40421


namespace second_divisor_exists_l404_40455

theorem second_divisor_exists : ∃ (x y : ℕ), 0 < y ∧ y < 61 ∧ x % 61 = 24 ∧ x % y = 4 := by
  sorry

end second_divisor_exists_l404_40455


namespace na_minimum_at_3_l404_40449

-- Define the sequence S_n
def S (n : ℕ) : ℤ := n^2 - 10*n

-- Define a_n as the difference between consecutive S_n terms
def a (n : ℕ) : ℤ := S n - S (n-1)

-- Define na_n
def na (n : ℕ) : ℤ := n * (a n)

-- Theorem statement
theorem na_minimum_at_3 :
  ∀ k : ℕ, k ≥ 1 → na 3 ≤ na k :=
sorry

end na_minimum_at_3_l404_40449


namespace complex_squared_norm_l404_40402

theorem complex_squared_norm (w : ℂ) (h : w^2 + Complex.abs w^2 = 7 + 2*I) : 
  Complex.abs w^2 = 53/14 := by
  sorry

end complex_squared_norm_l404_40402


namespace solution_difference_l404_40487

theorem solution_difference (r s : ℝ) : 
  r ≠ s →
  (6 * r - 18) / (r^2 + 2*r - 15) = r + 3 →
  (6 * s - 18) / (s^2 + 2*s - 15) = s + 3 →
  r > s →
  r - s = 8 := by sorry

end solution_difference_l404_40487


namespace routes_to_n_2_l404_40408

/-- The number of possible routes from (0, 0) to (x, y) moving only right or up -/
def f (x y : ℕ) : ℕ := sorry

/-- Theorem: The number of routes from (0, 0) to (n, 2) is (n^2 + 3n + 2) / 2 -/
theorem routes_to_n_2 (n : ℕ) : f n 2 = (n^2 + 3*n + 2) / 2 := by sorry

end routes_to_n_2_l404_40408


namespace profit_margin_increase_l404_40499

theorem profit_margin_increase (P S : ℝ) (r : ℝ) : 
  P > 0 → S > P →
  (S - P) / P * 100 = r →
  (S - 0.92 * P) / (0.92 * P) * 100 = r + 10 →
  r = 15 := by
sorry

end profit_margin_increase_l404_40499


namespace solve_for_m_l404_40479

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x - 8

-- State the theorem
theorem solve_for_m :
  ∃ m : ℝ, (f 8 - g m 8 = 20) ∧ (m = -25.5) := by
  sorry

end solve_for_m_l404_40479


namespace no_ratio_p_squared_l404_40494

theorem no_ratio_p_squared (p : ℕ) (hp : Prime p) :
  ∀ (x y l : ℕ), l ≥ 1 →
    (x * (x + 1)) / (y * (y + 1)) ≠ p^(2 * l) :=
by sorry

end no_ratio_p_squared_l404_40494


namespace factor_implies_k_equals_8_l404_40491

theorem factor_implies_k_equals_8 (m k : ℝ) : 
  (∃ q : ℝ, m^3 - k*m^2 - 24*m + 16 = (m^2 - 8*m) * q) → k = 8 := by
  sorry

end factor_implies_k_equals_8_l404_40491


namespace binomial_coeff_not_arithmetic_seq_l404_40470

theorem binomial_coeff_not_arithmetic_seq (n r : ℕ) (h1 : n ≥ r + 3) (h2 : r > 0) :
  ¬ (∃ d : ℚ, Nat.choose n r + d = Nat.choose n (r + 1) ∧
               Nat.choose n (r + 1) + d = Nat.choose n (r + 2) ∧
               Nat.choose n (r + 2) + d = Nat.choose n (r + 3)) :=
by sorry

end binomial_coeff_not_arithmetic_seq_l404_40470


namespace income_calculation_l404_40486

/-- Represents a person's financial situation -/
structure FinancialSituation where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- Theorem: Given a person's financial situation where the income to expenditure ratio
    is 3:2 and savings are 7000, the income is 21000 -/
theorem income_calculation (fs : FinancialSituation) 
  (h1 : fs.income = 3 * (fs.expenditure / 2))
  (h2 : fs.savings = 7000)
  (h3 : fs.income = fs.expenditure + fs.savings) : 
  fs.income = 21000 := by
  sorry

end income_calculation_l404_40486


namespace share_decrease_proof_l404_40431

theorem share_decrease_proof (total : ℕ) (c_share : ℕ) (b_decrease : ℕ) (c_decrease : ℕ) 
  (h_total : total = 1010)
  (h_c_share : c_share = 495)
  (h_b_decrease : b_decrease = 10)
  (h_c_decrease : c_decrease = 15) :
  ∃ (a_share b_share : ℕ) (x : ℕ),
    a_share + b_share + c_share = total ∧
    (a_share - x) / 3 = (b_share - b_decrease) / 2 ∧
    (a_share - x) / 3 = (c_share - c_decrease) / 5 ∧
    x = 25 := by
  sorry

end share_decrease_proof_l404_40431


namespace min_gennadies_for_festival_l404_40473

/-- Represents the number of people with a specific name -/
structure NameCount where
  alexanders : Nat
  borises : Nat
  vasilies : Nat

/-- Calculates the minimum number of Gennadies required -/
def minGennadies (counts : NameCount) : Nat :=
  max 0 (counts.borises - 1 - counts.alexanders - counts.vasilies)

/-- Theorem stating the minimum number of Gennadies required for the given scenario -/
theorem min_gennadies_for_festival (counts : NameCount) 
  (h1 : counts.alexanders = 45)
  (h2 : counts.borises = 122)
  (h3 : counts.vasilies = 27) :
  minGennadies counts = 49 := by
  sorry

#eval minGennadies { alexanders := 45, borises := 122, vasilies := 27 }

end min_gennadies_for_festival_l404_40473


namespace five_letter_words_count_l404_40485

def alphabet_size : Nat := 26
def excluded_letter : Nat := 1

theorem five_letter_words_count :
  let available_letters := alphabet_size - excluded_letter
  (available_letters ^ 4 : Nat) = 390625 := by
  sorry

end five_letter_words_count_l404_40485


namespace intersection_A_B_l404_40476

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x : ℕ | Real.log x < 1}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end intersection_A_B_l404_40476


namespace coefficients_of_given_equation_l404_40481

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given quadratic equation x^2 - x + 3 = 0 -/
def givenEquation : QuadraticEquation :=
  { a := 1, b := -1, c := 3 }

theorem coefficients_of_given_equation :
  givenEquation.a = 1 ∧ givenEquation.b = -1 ∧ givenEquation.c = 3 := by
  sorry

end coefficients_of_given_equation_l404_40481


namespace regular_polygon_sides_l404_40447

/-- A regular polygon with perimeter 180 cm and side length 15 cm has 12 sides -/
theorem regular_polygon_sides (P : ℝ) (s : ℝ) (n : ℕ) 
  (h_perimeter : P = 180) 
  (h_side : s = 15) 
  (h_regular : P = n * s) : n = 12 := by
  sorry

end regular_polygon_sides_l404_40447


namespace trig_identity_l404_40434

theorem trig_identity : 
  Real.cos (70 * π / 180) * Real.cos (335 * π / 180) + 
  Real.sin (110 * π / 180) * Real.sin (25 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end trig_identity_l404_40434


namespace book_purchase_problem_l404_40407

/-- Given information about book purchases, prove the number of people who purchased both books --/
theorem book_purchase_problem (A B AB : ℕ) 
  (h1 : A = 2 * B)
  (h2 : AB = 2 * (B - AB))
  (h3 : A - AB = 1000) :
  AB = 500 := by
  sorry

end book_purchase_problem_l404_40407


namespace alternative_plan_more_expensive_l404_40416

/-- Represents a phone plan with its pricing structure -/
structure PhonePlan where
  text_cost : ℚ  -- Cost per 30 texts
  call_cost : ℚ  -- Cost per 20 minutes of calls
  data_cost : ℚ  -- Cost per 2GB of data
  intl_cost : ℚ  -- Additional cost for international calls

/-- Represents a user's monthly usage -/
structure Usage where
  texts : ℕ
  call_minutes : ℕ
  data_gb : ℚ
  intl_calls : Bool

def calculate_cost (plan : PhonePlan) (usage : Usage) : ℚ :=
  let text_units := (usage.texts + 29) / 30
  let call_units := (usage.call_minutes + 19) / 20
  let data_units := ⌈usage.data_gb / 2⌉
  plan.text_cost * text_units +
  plan.call_cost * call_units +
  plan.data_cost * data_units +
  if usage.intl_calls then plan.intl_cost else 0

theorem alternative_plan_more_expensive :
  let current_plan_cost : ℚ := 12
  let alternative_plan := PhonePlan.mk 1 3 5 2
  let darnell_usage := Usage.mk 60 60 3 true
  calculate_cost alternative_plan darnell_usage - current_plan_cost = 11 := by
  sorry

end alternative_plan_more_expensive_l404_40416


namespace expression_simplification_l404_40426

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 3) (h3 : a ≠ -3) :
  (3 / (a - 3) - a / (a + 3)) * ((a^2 - 9) / a) = (-a^2 + 6*a + 9) / a := by
  sorry

end expression_simplification_l404_40426


namespace rectangle_shorter_side_l404_40423

/-- A rectangle with perimeter 60 meters and area 221 square meters has a shorter side of 13 meters. -/
theorem rectangle_shorter_side (a b : ℝ) : 
  a > 0 → b > 0 →  -- positive sides
  2 * (a + b) = 60 →  -- perimeter condition
  a * b = 221 →  -- area condition
  min a b = 13 := by
sorry

end rectangle_shorter_side_l404_40423


namespace infinitely_many_n_squared_divides_b_power_n_plus_one_l404_40463

theorem infinitely_many_n_squared_divides_b_power_n_plus_one
  (b : ℕ) (hb : b > 2) :
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, n^2 ∣ b^n + 1) ↔ ¬∃ k : ℕ, b + 1 = 2^k :=
sorry

end infinitely_many_n_squared_divides_b_power_n_plus_one_l404_40463


namespace tangent_slope_of_circle_l404_40465

/-- Given a circle with center (2,3) and a point (7,4) on the circle,
    the slope of the tangent line at (7,4) is -5. -/
theorem tangent_slope_of_circle (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  center = (2, 3) →
  point = (7, 4) →
  (point.2 - center.2) / (point.1 - center.1) = 1/5 →
  -(point.1 - center.1) / (point.2 - center.2) = -5 :=
by sorry


end tangent_slope_of_circle_l404_40465


namespace chips_sales_third_fourth_week_l404_40478

/-- Proves that the number of bags of chips sold in each of the third and fourth week is 20 --/
theorem chips_sales_third_fourth_week :
  let total_sales : ℕ := 100
  let first_week_sales : ℕ := 15
  let second_week_sales : ℕ := 3 * first_week_sales
  let remaining_sales : ℕ := total_sales - (first_week_sales + second_week_sales)
  let third_fourth_week_sales : ℕ := remaining_sales / 2
  third_fourth_week_sales = 20 := by
  sorry

end chips_sales_third_fourth_week_l404_40478


namespace homework_completion_l404_40448

/-- The fraction of homework Sanjay completed on Monday -/
def sanjay_monday : ℚ := 3/5

/-- The fraction of homework Deepak completed on Monday -/
def deepak_monday : ℚ := 2/7

/-- The fraction of remaining homework Sanjay completed on Tuesday -/
def sanjay_tuesday : ℚ := 1/3

/-- The fraction of remaining homework Deepak completed on Tuesday -/
def deepak_tuesday : ℚ := 3/10

/-- The combined fraction of original homework left for Sanjay and Deepak on Wednesday -/
def homework_left : ℚ := 23/30

theorem homework_completion :
  let sanjay_left := (1 - sanjay_monday) * (1 - sanjay_tuesday)
  let deepak_left := (1 - deepak_monday) * (1 - deepak_tuesday)
  sanjay_left + deepak_left = homework_left := by sorry

end homework_completion_l404_40448


namespace perfect_squares_product_sum_l404_40406

theorem perfect_squares_product_sum (a b : ℕ) : 
  (∃ x : ℕ, a = x^2) → 
  (∃ y : ℕ, b = y^2) → 
  a * b = a + b + 4844 →
  (Real.sqrt a + 1) * (Real.sqrt b + 1) * (Real.sqrt a - 1) * (Real.sqrt b - 1) - 
  (Real.sqrt 68 + 1) * (Real.sqrt 63 + 1) * (Real.sqrt 68 - 1) * (Real.sqrt 63 - 1) = 691 := by
sorry

end perfect_squares_product_sum_l404_40406


namespace pentagon_rectangle_apothem_ratio_l404_40461

theorem pentagon_rectangle_apothem_ratio :
  let pentagon_side := (40 : ℝ) / (1 + Real.sqrt 5)
  let pentagon_apothem := pentagon_side * ((1 + Real.sqrt 5) / 4)
  let rectangle_width := (3 : ℝ) / 2
  let rectangle_apothem := rectangle_width / 2
  pentagon_apothem / rectangle_apothem = 40 / 3 := by
  sorry

end pentagon_rectangle_apothem_ratio_l404_40461


namespace quadratic_function_point_comparison_l404_40454

/-- Given a quadratic function y = x² - 4x + k passing through (-1, y₁) and (3, y₂), prove y₁ > y₂ -/
theorem quadratic_function_point_comparison (k : ℝ) (y₁ y₂ : ℝ)
  (h₁ : y₁ = (-1)^2 - 4*(-1) + k)
  (h₂ : y₂ = 3^2 - 4*3 + k) :
  y₁ > y₂ :=
by sorry

end quadratic_function_point_comparison_l404_40454


namespace complex_equation_solution_l404_40405

theorem complex_equation_solution (z : ℂ) :
  (z - 2*I) * (2 - I) = 5 → z = 2 + 3*I :=
by sorry

end complex_equation_solution_l404_40405


namespace infinite_gcd_condition_l404_40462

open Set Function Nat

/-- A permutation of positive integers -/
def PositiveIntegerPermutation := ℕ+ → ℕ+

/-- The set of indices satisfying the GCD condition -/
def GcdConditionSet (a : PositiveIntegerPermutation) : Set ℕ+ :=
  {i | Nat.gcd (a i) (a (i + 1)) ≤ (3 * i) / 4}

/-- The main theorem -/
theorem infinite_gcd_condition (a : PositiveIntegerPermutation) 
  (h : Bijective a) : Infinite (GcdConditionSet a) := by
  sorry


end infinite_gcd_condition_l404_40462


namespace total_water_volume_l404_40471

def water_volume (num_containers : ℕ) (container_volume : ℝ) : ℝ :=
  (num_containers : ℝ) * container_volume

theorem total_water_volume : 
  water_volume 2812 4 = 11248 := by sorry

end total_water_volume_l404_40471


namespace polynomial_sum_l404_40452

theorem polynomial_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₄ - a₃ + a₂ - a₁ + a₀ = 16 := by
sorry

end polynomial_sum_l404_40452


namespace circle_and_tangent_properties_l404_40419

/-- Given a circle with center C on the line x-y+1=0 and passing through points A(1,1) and B(2,-2) -/
structure CircleData where
  C : ℝ × ℝ
  center_on_line : C.1 - C.2 + 1 = 0
  passes_through_A : (C.1 - 1)^2 + (C.2 - 1)^2 = (C.1 - 2)^2 + (C.2 + 2)^2

/-- The standard equation of the circle is (x+3)^2 + (y+2)^2 = 25 -/
def circle_equation (x y : ℝ) : Prop :=
  (x + 3)^2 + (y + 2)^2 = 25

/-- The equation of the tangent line passing through point (1,1) is 4x + 3y - 7 = 0 -/
def tangent_line_equation (x y : ℝ) : Prop :=
  4*x + 3*y - 7 = 0

theorem circle_and_tangent_properties (data : CircleData) :
  (∀ x y, circle_equation x y ↔ ((x - data.C.1)^2 + (y - data.C.2)^2 = (1 - data.C.1)^2 + (1 - data.C.2)^2)) ∧
  tangent_line_equation 1 1 ∧
  (∀ x y, tangent_line_equation x y → (x - 1) * (1 - data.C.1) + (y - 1) * (1 - data.C.2) = 0) :=
sorry

end circle_and_tangent_properties_l404_40419


namespace max_sum_given_sum_squares_and_product_l404_40453

theorem max_sum_given_sum_squares_and_product (x y : ℝ) : 
  x^2 + y^2 = 100 → xy = 40 → x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end max_sum_given_sum_squares_and_product_l404_40453


namespace pencil_distribution_l404_40435

theorem pencil_distribution (total_pencils : ℕ) (num_students : ℕ) (pencils_per_student : ℕ) :
  total_pencils = 125 →
  num_students = 25 →
  pencils_per_student * num_students = total_pencils →
  pencils_per_student = 5 := by
  sorry

end pencil_distribution_l404_40435


namespace zeros_in_fraction_l404_40425

-- Define the fraction
def fraction : ℚ := 18 / 50000

-- Define the function to count zeros after the decimal point
def count_zeros_after_decimal (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem zeros_in_fraction :
  count_zeros_after_decimal fraction = 3 := by sorry

end zeros_in_fraction_l404_40425


namespace invalid_votes_percentage_l404_40412

-- Define the total number of votes
def total_votes : ℕ := 560000

-- Define the percentage of valid votes received by candidate A
def candidate_A_percentage : ℚ := 55 / 100

-- Define the number of valid votes received by candidate A
def candidate_A_votes : ℕ := 261800

-- Define the percentage of invalid votes
def invalid_vote_percentage : ℚ := 15 / 100

-- Theorem statement
theorem invalid_votes_percentage :
  (1 - (candidate_A_votes : ℚ) / (candidate_A_percentage * total_votes)) = invalid_vote_percentage := by
  sorry

end invalid_votes_percentage_l404_40412


namespace vector_operation_result_l404_40403

theorem vector_operation_result :
  let v₁ : Fin 3 → ℝ := ![(-3), 2, (-5)]
  let v₂ : Fin 3 → ℝ := ![1, 6, (-3)]
  2 • v₁ + v₂ = ![-5, 10, (-13)] := by
sorry

end vector_operation_result_l404_40403


namespace abs_x_minus_two_geq_abs_x_l404_40411

theorem abs_x_minus_two_geq_abs_x (x : ℝ) : |x - 2| ≥ |x| ↔ x ≤ 1 := by sorry

end abs_x_minus_two_geq_abs_x_l404_40411


namespace number_remainder_l404_40464

theorem number_remainder (N : ℤ) 
  (h1 : N % 195 = 79)
  (h2 : N % 273 = 109) : 
  N % 39 = 1 := by
  sorry

end number_remainder_l404_40464


namespace infinite_series_sum_l404_40415

/-- The sum of the infinite series ∑(k=1 to ∞) [12^k / ((4^k - 3^k)(4^(k+1) - 3^(k+1)))] is equal to 3. -/
theorem infinite_series_sum : 
  (∑' k, (12 : ℝ)^k / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))) = 3 :=
sorry

end infinite_series_sum_l404_40415


namespace comet_watch_percentage_l404_40488

-- Define the total time spent on activities in minutes
def total_time : ℕ := 655

-- Define the time spent watching the comet in minutes
def comet_watch_time : ℕ := 20

-- Function to calculate percentage
def calculate_percentage (part : ℕ) (whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

-- Function to round to nearest integer
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

-- Theorem statement
theorem comet_watch_percentage :
  round_to_nearest (calculate_percentage comet_watch_time total_time) = 3 := by
  sorry

end comet_watch_percentage_l404_40488


namespace spherical_segment_volume_l404_40446

/-- Given a sphere of radius 10 cm, prove that a spherical segment with a ratio of 10:7 for its curved surface area to base area has a volume of 288π cm³ -/
theorem spherical_segment_volume (r : ℝ) (m : ℝ) (h_r : r = 10) 
  (h_ratio : (2 * r * m) / (m * (2 * r - m)) = 10 / 7) : 
  (m^2 * π / 3) * (3 * r - m) = 288 * π := by
  sorry

#check spherical_segment_volume

end spherical_segment_volume_l404_40446


namespace orchids_unchanged_l404_40430

/-- The number of orchids in a vase remains unchanged when only roses are added --/
theorem orchids_unchanged 
  (initial_roses : ℕ) 
  (initial_orchids : ℕ) 
  (final_roses : ℕ) 
  (roses_added : ℕ) : 
  initial_roses = 15 → 
  initial_orchids = 62 → 
  final_roses = 17 → 
  roses_added = 2 → 
  final_roses = initial_roses + roses_added → 
  initial_orchids = 62 := by
sorry

end orchids_unchanged_l404_40430


namespace number_of_students_l404_40445

/-- Given an initial average of 100, and a correction of one student's mark from 60 to 10
    resulting in a new average of 98, prove that the number of students in the class is 25. -/
theorem number_of_students (initial_average : ℝ) (wrong_mark : ℝ) (correct_mark : ℝ) (new_average : ℝ)
  (h1 : initial_average = 100)
  (h2 : wrong_mark = 60)
  (h3 : correct_mark = 10)
  (h4 : new_average = 98) :
  ∃ n : ℕ, n * new_average = n * initial_average - (wrong_mark - correct_mark) ∧ n = 25 := by
  sorry

end number_of_students_l404_40445


namespace simplify_expression_l404_40432

theorem simplify_expression (a b : ℝ) : (22*a + 60*b) + (10*a + 29*b) - (9*a + 50*b) = 23*a + 39*b := by
  sorry

end simplify_expression_l404_40432


namespace function_expression_l404_40457

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (x + 2) = x^2 - x + 1) :
  ∀ x, f x = x^2 - 5*x + 7 := by
sorry

end function_expression_l404_40457


namespace carol_initial_amount_l404_40483

/-- Carol's initial amount of money -/
def carol_initial : ℕ := sorry

/-- Carol's weekly savings -/
def carol_weekly_savings : ℕ := 9

/-- Mike's initial amount of money -/
def mike_initial : ℕ := 90

/-- Mike's weekly savings -/
def mike_weekly_savings : ℕ := 3

/-- Number of weeks -/
def weeks : ℕ := 5

theorem carol_initial_amount :
  carol_initial = 60 :=
by
  have h1 : carol_initial + weeks * carol_weekly_savings = mike_initial + weeks * mike_weekly_savings :=
    sorry
  sorry

end carol_initial_amount_l404_40483


namespace ellipse_min_area_l404_40427

/-- An ellipse containing two specific circles has a minimum area of π -/
theorem ellipse_min_area (a b : ℝ) (h_positive : a > 0 ∧ b > 0) :
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 →
    ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) →
  ∃ k : ℝ, k = 1 ∧ π * a * b ≥ k * π :=
sorry

end ellipse_min_area_l404_40427


namespace mrs_hilt_reading_l404_40480

theorem mrs_hilt_reading (books : ℝ) (chapters_per_book : ℝ) 
  (h1 : books = 4.0) (h2 : chapters_per_book = 4.25) : 
  books * chapters_per_book = 17 := by
  sorry

end mrs_hilt_reading_l404_40480


namespace function_equality_l404_40437

theorem function_equality (f : ℝ → ℝ) :
  (∀ x : ℝ, f (2 * x + 1) = 4 * x^2 + 14 * x + 7) →
  (∀ x : ℝ, f x = x^2 + 5 * x + 1) := by
  sorry

end function_equality_l404_40437


namespace inequality_proof_l404_40459

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a * b * c ≤ (a + b) * (b + c) * (c + a) := by
  sorry

end inequality_proof_l404_40459


namespace equation_solution_l404_40484

theorem equation_solution : ∃! y : ℚ, (4 * y + 2) / (5 * y - 5) = 3 / 4 ∧ 5 * y - 5 ≠ 0 := by
  sorry

end equation_solution_l404_40484


namespace product_of_roots_l404_40439

theorem product_of_roots (x₁ x₂ k m : ℝ) : 
  x₁ ≠ x₂ →
  5 * x₁^2 - k * x₁ = m →
  5 * x₂^2 - k * x₂ = m →
  x₁ * x₂ = -m / 5 := by
sorry

end product_of_roots_l404_40439


namespace maria_savings_l404_40418

/-- The amount left in Maria's savings after buying sweaters and scarves -/
def savings_left (sweater_price scarf_price sweater_count scarf_count initial_savings : ℕ) : ℕ :=
  initial_savings - (sweater_price * sweater_count + scarf_price * scarf_count)

/-- Theorem stating that Maria will have $200 left in her savings -/
theorem maria_savings : savings_left 30 20 6 6 500 = 200 := by
  sorry

end maria_savings_l404_40418


namespace geometric_series_first_term_l404_40493

theorem geometric_series_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80)
  : a = 20 / 3 := by
  sorry

end geometric_series_first_term_l404_40493


namespace altitude_length_of_triangle_on_rectangle_diagonal_l404_40400

/-- Given a rectangle with sides a and b, and a triangle constructed on its diagonal
    with an area equal to the rectangle's area, the length of the altitude drawn to
    the base (diagonal) of the triangle is (2ab) / √(a² + b²). -/
theorem altitude_length_of_triangle_on_rectangle_diagonal
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let rectangle_area := a * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let triangle_area := (1/2) * diagonal * (2 * rectangle_area / diagonal)
  triangle_area = rectangle_area →
  (2 * rectangle_area / diagonal) = (2 * a * b) / Real.sqrt (a^2 + b^2) := by
sorry


end altitude_length_of_triangle_on_rectangle_diagonal_l404_40400
