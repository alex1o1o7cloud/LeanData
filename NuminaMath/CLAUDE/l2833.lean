import Mathlib

namespace range_of_s_is_composite_positive_integers_l2833_283322

-- Define the set of composite positive integers
def CompositePositiveIntegers : Set ℕ := {n : ℕ | n > 1 ∧ ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b}

-- Define the function s
def s (n : ℕ) : ℕ := n

-- State the theorem
theorem range_of_s_is_composite_positive_integers :
  {s n | n ∈ CompositePositiveIntegers} = CompositePositiveIntegers := by
  sorry


end range_of_s_is_composite_positive_integers_l2833_283322


namespace min_value_implies_a_inequality_solution_set_l2833_283374

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

-- Theorem for part 1
theorem min_value_implies_a (a : ℝ) (h1 : a > 1) :
  (∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f x a ≥ m) → a = 7 :=
sorry

-- Theorem for part 2
theorem inequality_solution_set (x : ℝ) :
  f x 7 ≤ 5 ↔ 3 ≤ x ∧ x ≤ 8 :=
sorry

end min_value_implies_a_inequality_solution_set_l2833_283374


namespace existence_of_multiple_representations_l2833_283357

def V_n (n : ℕ) : Set ℕ := {m | ∃ k : ℕ, m = 1 + k * n}

def indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → p * q ≠ m

theorem existence_of_multiple_representations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ a b c d : ℕ,
      indecomposable n a ∧
      indecomposable n b ∧
      indecomposable n c ∧
      indecomposable n d ∧
      r = a * b ∧
      r = c * d ∧
      (a ≠ c ∨ b ≠ d) ∧
      (a ≠ d ∨ b ≠ c) :=
by sorry

end existence_of_multiple_representations_l2833_283357


namespace new_rope_length_l2833_283377

/-- Proves that given an initial rope length of 12 m and an additional grazing area of 565.7142857142857 m², 
    the new rope length that allows this additional grazing area is 18 m. -/
theorem new_rope_length 
  (initial_length : ℝ) 
  (additional_area : ℝ) 
  (h1 : initial_length = 12)
  (h2 : additional_area = 565.7142857142857) : 
  ∃ (new_length : ℝ), 
    new_length = 18 ∧ 
    π * new_length ^ 2 = π * initial_length ^ 2 + additional_area :=
by sorry

end new_rope_length_l2833_283377


namespace list_number_relation_l2833_283394

theorem list_number_relation (l : List ℝ) (n : ℝ) : 
  l.length = 21 ∧ 
  n ∈ l ∧ 
  n = (l.sum / 6) →
  n = 4 * ((l.sum - n) / 20) := by
sorry

end list_number_relation_l2833_283394


namespace parallelogram_area_l2833_283345

/-- The area of a parallelogram with vertices at (0, 0), (4, 0), (1, 6), and (5, 6) is 24 square units. -/
theorem parallelogram_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (4, 0)
  let C : ℝ × ℝ := (5, 6)
  let D : ℝ × ℝ := (1, 6)
  let area := abs ((B.1 - A.1) * (D.2 - A.2) - (D.1 - A.1) * (B.2 - A.2))
  area = 24 := by sorry

end parallelogram_area_l2833_283345


namespace function_positivity_implies_ab_bound_l2833_283358

/-- Given a function f(x) = (x - 1/x - a)(x - b), if f(x) > 0 for all x > 0, then ab > -1 -/
theorem function_positivity_implies_ab_bound (a b : ℝ) : 
  (∀ x > 0, (x - 1/x - a) * (x - b) > 0) → a * b > -1 := by
  sorry

end function_positivity_implies_ab_bound_l2833_283358


namespace intersection_nonempty_iff_a_greater_than_neg_five_l2833_283350

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | 1 - a < x}

-- State the theorem
theorem intersection_nonempty_iff_a_greater_than_neg_five (a : ℝ) :
  (A ∩ B a).Nonempty ↔ a > -5 := by
  sorry

end intersection_nonempty_iff_a_greater_than_neg_five_l2833_283350


namespace rainfall_problem_l2833_283329

/-- Rainfall problem -/
theorem rainfall_problem (total_time : ℕ) (total_rainfall : ℕ) 
  (storm1_rate : ℕ) (storm1_duration : ℕ) :
  total_time = 45 →
  total_rainfall = 975 →
  storm1_rate = 30 →
  storm1_duration = 20 →
  ∃ storm2_rate : ℕ, 
    storm2_rate * (total_time - storm1_duration) = 
      total_rainfall - (storm1_rate * storm1_duration) ∧
    storm2_rate = 15 := by
  sorry


end rainfall_problem_l2833_283329


namespace journey_time_equation_l2833_283316

theorem journey_time_equation (x : ℝ) (h : x > 0) :
  let distance : ℝ := 15
  let cyclist_speed : ℝ := x
  let car_speed : ℝ := 2 * x
  let head_start : ℝ := 1 / 2
  distance / cyclist_speed = distance / car_speed + head_start :=
by sorry

end journey_time_equation_l2833_283316


namespace algebraic_simplification_l2833_283314

theorem algebraic_simplification (a b : ℝ) : 2*a - 3*(a-b) = -a + 3*b := by
  sorry

end algebraic_simplification_l2833_283314


namespace geometric_series_common_ratio_l2833_283393

theorem geometric_series_common_ratio :
  ∀ (a r : ℚ),
    a = 4/7 →
    a * r = 16/21 →
    r = 4/3 :=
by
  sorry

end geometric_series_common_ratio_l2833_283393


namespace set_equality_l2833_283383

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end set_equality_l2833_283383


namespace milk_water_mixture_volume_l2833_283337

/-- Proves that given a mixture of milk and water with an initial ratio of 3:2,
    if adding 46 liters of water changes the ratio to 3:4,
    then the initial volume of the mixture was 115 liters. -/
theorem milk_water_mixture_volume
  (initial_milk : ℝ)
  (initial_water : ℝ)
  (h1 : initial_milk / initial_water = 3 / 2)
  (h2 : initial_milk / (initial_water + 46) = 3 / 4) :
  initial_milk + initial_water = 115 :=
by sorry

end milk_water_mixture_volume_l2833_283337


namespace percentage_less_problem_l2833_283378

theorem percentage_less_problem (C A B : ℝ) : 
  B = 0.58 * C →
  B = 0.8923076923076923 * A →
  ∃ (ε : ℝ), abs (A - 0.65 * C) < ε ∧ ε > 0 := by
sorry

end percentage_less_problem_l2833_283378


namespace factorial_expression_equality_l2833_283351

theorem factorial_expression_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 6 - 6 * Nat.factorial 5 = 7920 := by
  sorry

end factorial_expression_equality_l2833_283351


namespace parallel_vectors_t_value_l2833_283362

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_t_value :
  ∀ t : ℝ, 
  let a : ℝ × ℝ := (1, t)
  let b : ℝ × ℝ := (t, 9)
  parallel a b → t = 3 ∨ t = -3 := by
sorry

end parallel_vectors_t_value_l2833_283362


namespace complex_equation_solution_l2833_283348

theorem complex_equation_solution :
  ∀ z : ℂ, z / (1 + Complex.I) = Complex.I ^ 2015 + Complex.I ^ 2016 → z = -2 * Complex.I :=
by
  sorry

end complex_equation_solution_l2833_283348


namespace ice_cube_freeze_time_l2833_283375

/-- The time in minutes to turn frozen ice cubes into one smoothie -/
def time_per_smoothie : ℕ := 3

/-- The total number of smoothies made -/
def num_smoothies : ℕ := 5

/-- The total time in minutes to make all smoothies, including freezing ice cubes -/
def total_time : ℕ := 55

/-- The time in minutes to freeze ice cubes -/
def freeze_time : ℕ := total_time - (time_per_smoothie * num_smoothies)

theorem ice_cube_freeze_time :
  freeze_time = 40 := by sorry

end ice_cube_freeze_time_l2833_283375


namespace complement_A_equals_negative_reals_l2833_283385

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A as the set of non-negative real numbers
def A : Set ℝ := { x : ℝ | x ≥ 0 }

-- Define the complement of A in U
def complement_A : Set ℝ := U \ A

-- Theorem statement
theorem complement_A_equals_negative_reals :
  complement_A = { x : ℝ | x < 0 } :=
sorry

end complement_A_equals_negative_reals_l2833_283385


namespace mitchell_gum_chewing_l2833_283325

/-- 
Given 8 packets of gum with 7 pieces each, and leaving 2 pieces unchewed,
prove that the number of pieces chewed is equal to 54.
-/
theorem mitchell_gum_chewing (packets : Nat) (pieces_per_packet : Nat) (unchewed : Nat) : 
  packets = 8 → pieces_per_packet = 7 → unchewed = 2 →
  packets * pieces_per_packet - unchewed = 54 := by
sorry

end mitchell_gum_chewing_l2833_283325


namespace al_ben_weight_difference_l2833_283361

theorem al_ben_weight_difference :
  ∀ (al_weight ben_weight carl_weight : ℕ),
    ben_weight = carl_weight - 16 →
    al_weight = 146 + 38 →
    carl_weight = 175 →
    al_weight - ben_weight = 25 := by
  sorry

end al_ben_weight_difference_l2833_283361


namespace num_positive_divisors_1386_l2833_283364

/-- The number of positive divisors of a natural number -/
def numPositiveDivisors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of positive divisors of 1386 is 24 -/
theorem num_positive_divisors_1386 : numPositiveDivisors 1386 = 24 := by sorry

end num_positive_divisors_1386_l2833_283364


namespace light_bulb_probabilities_l2833_283387

/-- Market share of Factory A -/
def market_share_A : ℝ := 0.6

/-- Market share of Factory B -/
def market_share_B : ℝ := 0.4

/-- Qualification rate of Factory A products -/
def qual_rate_A : ℝ := 0.9

/-- Qualification rate of Factory B products -/
def qual_rate_B : ℝ := 0.8

/-- Probability of exactly one qualified light bulb out of two from Factory A -/
def prob_one_qualified_A : ℝ := 2 * qual_rate_A * (1 - qual_rate_A)

/-- Probability of a randomly purchased light bulb being qualified -/
def prob_random_qualified : ℝ := market_share_A * qual_rate_A + market_share_B * qual_rate_B

theorem light_bulb_probabilities :
  prob_one_qualified_A = 0.18 ∧ prob_random_qualified = 0.86 := by
  sorry

#check light_bulb_probabilities

end light_bulb_probabilities_l2833_283387


namespace outfit_combinations_l2833_283328

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) : 
  shirts = 3 → pants = 4 → shirts * pants = 12 :=
by sorry

end outfit_combinations_l2833_283328


namespace student_arrangement_l2833_283313

/-- The number of arrangements for n male and m female students -/
def arrangement_count (n m : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of permutations of k items from n items -/
def permute (n k : ℕ) : ℕ := sorry

theorem student_arrangement :
  let total_male : ℕ := 5
  let total_female : ℕ := 5
  let females_between : ℕ := 2
  let males_at_ends : ℕ := 2
  
  arrangement_count total_male total_female = 
    choose total_female females_between * 
    permute (total_male - 2) males_at_ends * 
    permute (total_male + total_female - females_between - males_at_ends - 2) 
            (total_male + total_female - females_between - males_at_ends - 2) :=
by sorry

end student_arrangement_l2833_283313


namespace min_sum_squares_l2833_283379

theorem min_sum_squares (p q r s t u v w : Int) : 
  p ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  q ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  r ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  s ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  t ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  u ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  v ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  w ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  p ≠ q → p ≠ r → p ≠ s → p ≠ t → p ≠ u → p ≠ v → p ≠ w →
  q ≠ r → q ≠ s → q ≠ t → q ≠ u → q ≠ v → q ≠ w →
  r ≠ s → r ≠ t → r ≠ u → r ≠ v → r ≠ w →
  s ≠ t → s ≠ u → s ≠ v → s ≠ w →
  t ≠ u → t ≠ v → t ≠ w →
  u ≠ v → u ≠ w →
  v ≠ w →
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 10 :=
by sorry

end min_sum_squares_l2833_283379


namespace jeff_phone_storage_capacity_l2833_283340

theorem jeff_phone_storage_capacity :
  let storage_used : ℕ := 4
  let song_size : ℕ := 30
  let max_songs : ℕ := 400
  let mb_per_gb : ℕ := 1000
  let total_storage : ℕ := 
    storage_used + (song_size * max_songs) / mb_per_gb
  total_storage = 16 := by
  sorry

end jeff_phone_storage_capacity_l2833_283340


namespace indeterminate_product_at_opposite_points_l2833_283386

-- Define a continuous function on an open interval
def ContinuousOnOpenInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → ContinuousAt f x

-- Define the property of having a single root at 0 in the interval (-2, 2)
def SingleRootAtZero (f : ℝ → ℝ) : Prop :=
  (∀ x, -2 < x ∧ x < 2 ∧ f x = 0 → x = 0) ∧
  (f 0 = 0)

-- Theorem statement
theorem indeterminate_product_at_opposite_points
  (f : ℝ → ℝ)
  (h_cont : ContinuousOnOpenInterval f (-2) 2)
  (h_root : SingleRootAtZero f) :
  ∃ (f₁ f₂ f₃ : ℝ → ℝ),
    (ContinuousOnOpenInterval f₁ (-2) 2 ∧ SingleRootAtZero f₁ ∧ f₁ (-1) * f₁ 1 > 0) ∧
    (ContinuousOnOpenInterval f₂ (-2) 2 ∧ SingleRootAtZero f₂ ∧ f₂ (-1) * f₂ 1 < 0) ∧
    (ContinuousOnOpenInterval f₃ (-2) 2 ∧ SingleRootAtZero f₃ ∧ f₃ (-1) * f₃ 1 = 0) :=
  sorry

end indeterminate_product_at_opposite_points_l2833_283386


namespace product_mod_nineteen_l2833_283331

theorem product_mod_nineteen : (2001 * 2002 * 2003 * 2004 * 2005) % 19 = 11 := by
  sorry

end product_mod_nineteen_l2833_283331


namespace president_and_committee_from_ten_l2833_283398

/-- The number of ways to choose a president and a committee from a group --/
def choose_president_and_committee (group_size : ℕ) (committee_size : ℕ) : ℕ :=
  group_size * Nat.choose (group_size - 1) committee_size

/-- Theorem stating the number of ways to choose a president and a 3-person committee from 10 people --/
theorem president_and_committee_from_ten :
  choose_president_and_committee 10 3 = 840 := by
  sorry

end president_and_committee_from_ten_l2833_283398


namespace smallest_possible_d_l2833_283303

theorem smallest_possible_d : ∃ (d : ℝ), d ≥ 0 ∧
  (∀ (d' : ℝ), d' ≥ 0 → (4 * Real.sqrt 3) ^ 2 + (d' - 2) ^ 2 = (4 * d') ^ 2 → d ≤ d') ∧
  (4 * Real.sqrt 3) ^ 2 + (d - 2) ^ 2 = (4 * d) ^ 2 ∧
  d = 4 := by
  sorry

end smallest_possible_d_l2833_283303


namespace searchlight_configuration_exists_l2833_283396

/-- Represents a searchlight with its position and direction --/
structure Searchlight where
  position : ℝ × ℝ
  direction : ℝ

/-- Checks if a point is within the illuminated region of a searchlight --/
def isIlluminated (s : Searchlight) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Calculates the shadow length of a searchlight given a configuration --/
def shadowLength (s : Searchlight) (config : List Searchlight) : ℝ :=
  sorry

/-- Theorem: There exists a configuration of 7 searchlights where each casts a 7km shadow --/
theorem searchlight_configuration_exists : 
  ∃ (config : List Searchlight), 
    config.length = 7 ∧ 
    ∀ s ∈ config, shadowLength s config = 7 :=
  sorry

end searchlight_configuration_exists_l2833_283396


namespace power_sum_sequence_l2833_283343

/-- Given real numbers a and b satisfying certain conditions, prove that a^10 + b^10 = 123 -/
theorem power_sum_sequence (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry


end power_sum_sequence_l2833_283343


namespace sugar_left_l2833_283376

theorem sugar_left (bought spilled : ℝ) (h1 : bought = 9.8) (h2 : spilled = 5.2) :
  bought - spilled = 4.6 := by
  sorry

end sugar_left_l2833_283376


namespace principal_amount_proof_l2833_283365

/-- Proves that for a principal amount P, with an interest rate of 5% per annum over 2 years,
    if the difference between compound interest and simple interest is 17, then P equals 6800. -/
theorem principal_amount_proof (P : ℝ) : 
  P * (1 + 0.05)^2 - P - (P * 0.05 * 2) = 17 → P = 6800 := by
  sorry

end principal_amount_proof_l2833_283365


namespace right_triangle_ratio_l2833_283373

theorem right_triangle_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a / b = 2 / 5 →    -- Given ratio of legs
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  r + s = c →        -- r and s are parts of hypotenuse
  r / s = 4 / 25 :=  -- Conclusion to prove
by sorry

end right_triangle_ratio_l2833_283373


namespace adjacent_probability_four_people_l2833_283353

def num_people : ℕ := 4

def total_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

def favorable_arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 2)

def probability_adjacent (n : ℕ) : ℚ :=
  (favorable_arrangements n : ℚ) / (total_arrangements n : ℚ)

theorem adjacent_probability_four_people :
  probability_adjacent num_people = 2/3 := by
  sorry

end adjacent_probability_four_people_l2833_283353


namespace curve_crosses_at_point_l2833_283367

/-- A curve in the xy-plane defined by parametric equations -/
def curve (t : ℝ) : ℝ × ℝ :=
  (3 * t^2 + 1, t^3 - 6 * t^2 + 4)

/-- The point where the curve crosses itself -/
def crossing_point : ℝ × ℝ := (109, -428)

/-- Theorem stating that the curve crosses itself at the specified point -/
theorem curve_crosses_at_point :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ curve t₁ = curve t₂ ∧ curve t₁ = crossing_point :=
sorry

end curve_crosses_at_point_l2833_283367


namespace largest_prime_factors_difference_l2833_283389

theorem largest_prime_factors_difference (n : Nat) (h : n = 180181) :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p > q ∧
  p ∣ n ∧ q ∣ n ∧
  (∀ r : Nat, Nat.Prime r → r ∣ n → r ≤ p) ∧
  (∀ r : Nat, Nat.Prime r → r ∣ n → r ≠ p → r ≤ q) ∧
  p - q = 2 :=
by sorry

end largest_prime_factors_difference_l2833_283389


namespace sin_angle_A_is_sqrt3_div_2_l2833_283320

/-- An isosceles trapezoid with specific measurements -/
structure IsoscelesTrapezoid where
  -- Side lengths
  AB : ℝ
  CD : ℝ
  AD : ℝ
  -- Angle A in radians
  angleA : ℝ
  -- Conditions
  isIsosceles : AD = AD  -- AD = BC
  isParallel : AB < CD  -- AB parallel to CD implies AB < CD
  angleValue : angleA = 2 * Real.pi / 3  -- 120° in radians
  sideAB : AB = 160
  sideCD : CD = 240
  perimeter : AB + CD + 2 * AD = 800

/-- The sine of angle A in the isosceles trapezoid is √3/2 -/
theorem sin_angle_A_is_sqrt3_div_2 (t : IsoscelesTrapezoid) : Real.sin t.angleA = Real.sqrt 3 / 2 := by
  sorry

end sin_angle_A_is_sqrt3_div_2_l2833_283320


namespace chord_length_l2833_283306

/-- The length of the chord cut by the line y = 3x on the circle (x+1)^2 + (y-2)^2 = 25 is 3√10. -/
theorem chord_length (x y : ℝ) : 
  y = 3 * x →
  (x + 1)^2 + (y - 2)^2 = 25 →
  ∃ (x1 y1 x2 y2 : ℝ), 
    y1 = 3 * x1 ∧
    (x1 + 1)^2 + (y1 - 2)^2 = 25 ∧
    y2 = 3 * x2 ∧
    (x2 + 1)^2 + (y2 - 2)^2 = 25 ∧
    ((x2 - x1)^2 + (y2 - y1)^2) = 90 :=
by sorry

end chord_length_l2833_283306


namespace power_sum_and_division_l2833_283300

theorem power_sum_and_division (a b c : ℕ) : 3^456 + 9^5 / 9^3 = 82 := by
  sorry

end power_sum_and_division_l2833_283300


namespace integral_value_l2833_283310

-- Define the inequality
def inequality (x a : ℝ) : Prop := 1 - 3 / (x + a) < 0

-- Define the solution set
def solution_set : Set ℝ := Set.Ioo (-1) 2

-- Theorem statement
theorem integral_value (a : ℝ) 
  (h1 : ∀ x, x ∈ solution_set ↔ inequality x a) :
  ∫ x in a..3, (1 - 3 / (x + a)) = 2 - 3 * Real.log 3 := by
  sorry

end integral_value_l2833_283310


namespace hyperbola_fixed_point_l2833_283366

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define point A
def A : ℝ × ℝ := (-1, 0)

-- Define the condition that a point is on the hyperbola
def on_hyperbola (p : ℝ × ℝ) : Prop := hyperbola p.1 p.2

-- Define the perpendicularity condition
def perpendicular (p q : ℝ × ℝ) : Prop :=
  (p.1 - A.1) * (q.1 - A.1) + (p.2 - A.2) * (q.2 - A.2) = 0

-- Define the condition that a line passes through a point
def line_passes_through (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (r.1 - p.1) * (q.2 - p.2)

-- Main theorem
theorem hyperbola_fixed_point :
  ∀ (p q : ℝ × ℝ),
    on_hyperbola p →
    on_hyperbola q →
    perpendicular p q →
    line_passes_through p q (3, 0) :=
by sorry

end hyperbola_fixed_point_l2833_283366


namespace Egypt_India_traditional_l2833_283333

-- Define the types of countries and population growth patterns
inductive CountryType
| Developed
| Developing

inductive GrowthPattern
| Traditional
| Modern

-- Define a function to determine the growth pattern based on country type
def typicalGrowthPattern (ct : CountryType) : GrowthPattern :=
  match ct with
  | CountryType.Developed => GrowthPattern.Modern
  | CountryType.Developing => GrowthPattern.Traditional

-- Define specific countries
def Egypt : CountryType := CountryType.Developing
def India : CountryType := CountryType.Developing

-- China is an exception
def China : CountryType := CountryType.Developing
axiom China_exception : typicalGrowthPattern China = GrowthPattern.Modern

-- Theorem to prove
theorem Egypt_India_traditional :
  typicalGrowthPattern Egypt = GrowthPattern.Traditional ∧
  typicalGrowthPattern India = GrowthPattern.Traditional :=
sorry

end Egypt_India_traditional_l2833_283333


namespace expression_simplification_and_evaluation_l2833_283390

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ -1) (h3 : x ≠ 1) :
  (1 - 3 / (x + 2)) / ((x^2 - 1) / (x + 2)) = 1 / (x + 1) ∧
  (2 - 3 / (2 + 2)) / ((2^2 - 1) / (2 + 2)) = 1 / 3 :=
by sorry

end expression_simplification_and_evaluation_l2833_283390


namespace score_distribution_theorem_l2833_283349

/-- Represents the frequency distribution of student scores -/
structure FrequencyDistribution :=
  (f65_70 f70_75 f75_80 f80_85 f85_90 f90_95 f95_100 : ℚ)

/-- Represents the problem setup -/
structure ProblemSetup :=
  (fd : FrequencyDistribution)
  (total_students : ℕ)
  (students_80_90 : ℕ)
  (prob_male_95_100 : ℚ)
  (female_65_70 : ℕ)

/-- The main theorem to prove -/
theorem score_distribution_theorem (setup : ProblemSetup) :
  (setup.total_students * setup.fd.f95_100 = 6) ∧
  (∃ (m : ℕ), m = 2 ∧ m ≤ 6 ∧ 
    (m * (m - 1) / 30 + m * (6 - m) / 15 : ℚ) = 3/5) ∧
  (∃ (p0 p1 p2 : ℚ), p0 + p1 + p2 = 1 ∧
    p0 * 0 + p1 * 1 + p2 * 2 = 1) :=
by sorry

/-- Assumptions about the problem setup -/
axiom setup_valid (setup : ProblemSetup) :
  setup.fd.f65_70 = 1/10 ∧
  setup.fd.f70_75 = 3/20 ∧
  setup.fd.f75_80 = 1/5 ∧
  setup.fd.f80_85 = 1/5 ∧
  setup.fd.f85_90 = 3/20 ∧
  setup.fd.f90_95 = 1/10 ∧
  setup.fd.f95_100 + setup.fd.f65_70 + setup.fd.f70_75 + setup.fd.f75_80 +
    setup.fd.f80_85 + setup.fd.f85_90 + setup.fd.f90_95 = 1 ∧
  setup.students_80_90 = 21 ∧
  setup.prob_male_95_100 = 3/5 ∧
  setup.female_65_70 = 4 ∧
  setup.total_students * (setup.fd.f80_85 + setup.fd.f85_90) = setup.students_80_90

end score_distribution_theorem_l2833_283349


namespace unique_solution_is_four_l2833_283302

theorem unique_solution_is_four :
  ∃! x : ℝ, 2 * x + 20 = 8 * x - 4 := by
  sorry

end unique_solution_is_four_l2833_283302


namespace discount_comparison_l2833_283330

def initial_amount : ℝ := 20000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.25, 0.15, 0.05]
def option2_discounts : List ℝ := [0.35, 0.10, 0.05]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

def final_price_option1 : ℝ :=
  apply_successive_discounts initial_amount option1_discounts

def final_price_option2 : ℝ :=
  apply_successive_discounts initial_amount option2_discounts

theorem discount_comparison :
  final_price_option1 - final_price_option2 = 997.50 ∧
  final_price_option2 < final_price_option1 :=
sorry

end discount_comparison_l2833_283330


namespace notebook_count_l2833_283334

theorem notebook_count : ∃ (N : ℕ), 
  (∃ (S : ℕ), N = 4 * S + 3) ∧ 
  (∃ (S : ℕ), N + 6 = 5 * S) ∧ 
  N = 39 := by
  sorry

end notebook_count_l2833_283334


namespace prism_with_12_edges_has_quadrilateral_base_l2833_283356

/-- A prism with n sides in its base has 3n edges. -/
def prism_edges (n : ℕ) : ℕ := 3 * n

/-- The number of sides in the base of a prism with 12 edges. -/
def base_sides : ℕ := 4

/-- Theorem: A prism with 12 edges has a quadrilateral base. -/
theorem prism_with_12_edges_has_quadrilateral_base :
  prism_edges base_sides = 12 :=
sorry

end prism_with_12_edges_has_quadrilateral_base_l2833_283356


namespace circle_radius_property_l2833_283380

theorem circle_radius_property (r : ℝ) : 
  r > 0 → r * (2 * Real.pi * r) = 2 * (Real.pi * r^2) → 
  ∃ (radius : ℝ), radius > 0 ∧ radius * (2 * Real.pi * radius) = 2 * (Real.pi * radius^2) := by
sorry

end circle_radius_property_l2833_283380


namespace train_speed_l2833_283344

/-- The speed of a train given its length, time to cross a bridge, and total length of train and bridge -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (total_length : ℝ) :
  train_length = 130 →
  crossing_time = 30 →
  total_length = 245 →
  (total_length / crossing_time) * 3.6 = 29.4 := by
  sorry

#check train_speed

end train_speed_l2833_283344


namespace prime_factorization_sum_l2833_283355

theorem prime_factorization_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 1260 → 2*w + 3*x + 5*y + 7*z = 22 := by
  sorry

end prime_factorization_sum_l2833_283355


namespace students_liking_neither_l2833_283371

theorem students_liking_neither (total : ℕ) (chinese : ℕ) (math : ℕ) (both : ℕ)
  (h_total : total = 62)
  (h_chinese : chinese = 37)
  (h_math : math = 49)
  (h_both : both = 30) :
  total - (chinese + math - both) = 6 :=
by sorry

end students_liking_neither_l2833_283371


namespace unique_remainder_mod_10_l2833_283369

theorem unique_remainder_mod_10 : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [ZMOD 10] := by
  sorry

end unique_remainder_mod_10_l2833_283369


namespace inequality_proof_l2833_283382

theorem inequality_proof (x y z : ℝ) 
  (h1 : -2 ≤ x ∧ x ≤ 2) 
  (h2 : -2 ≤ y ∧ y ≤ 2) 
  (h3 : -2 ≤ z ∧ z ≤ 2) 
  (h4 : x^2 + y^2 + z^2 + x*y*z = 4) : 
  z * (x*z + y*z + y) / (x*y + y^2 + z^2 + 1) ≤ 4/3 := by
sorry

end inequality_proof_l2833_283382


namespace toucan_count_l2833_283335

theorem toucan_count (initial : ℕ) (joined : ℕ) (total : ℕ) : 
  joined = 1 → total = 3 → initial = 2 :=
by
  sorry

end toucan_count_l2833_283335


namespace trains_meeting_time_l2833_283317

/-- Two trains meeting problem -/
theorem trains_meeting_time 
  (distance : ℝ) 
  (speed1 speed2 : ℝ) 
  (start_time2 meet_time : ℝ) 
  (h1 : distance = 200) 
  (h2 : speed1 = 20) 
  (h3 : speed2 = 25) 
  (h4 : start_time2 = 8) 
  (h5 : meet_time = 12) : 
  ∃ start_time1 : ℝ, 
    start_time1 = 7 ∧ 
    speed1 * (meet_time - start_time1) + speed2 * (meet_time - start_time2) = distance :=
sorry

end trains_meeting_time_l2833_283317


namespace xy_plus_inverse_min_value_l2833_283301

theorem xy_plus_inverse_min_value (x y : ℝ) 
  (hx : x < 0) (hy : y < 0) (hsum : x + y = -1) :
  ∀ z, z = x * y + 1 / (x * y) → z ≥ 17 / 4 :=
sorry

end xy_plus_inverse_min_value_l2833_283301


namespace complex_number_equality_l2833_283384

theorem complex_number_equality : (7 : ℂ) - 3*I - 3*(2 - 5*I) + 4*I = 1 + 16*I := by
  sorry

end complex_number_equality_l2833_283384


namespace unicorn_journey_flowers_l2833_283341

/-- Calculates the number of flowers that bloom when unicorns walk across a forest -/
def unicorn_flowers (num_unicorns : ℕ) (journey_km : ℕ) (step_meters : ℕ) (flowers_per_step : ℕ) : ℕ :=
  let journey_meters := journey_km * 1000
  let num_steps := journey_meters / step_meters
  let flowers_per_unicorn := num_steps * flowers_per_step
  num_unicorns * flowers_per_unicorn

/-- Theorem stating that 6 unicorns walking 9 km with 3-meter steps, each causing 4 flowers to bloom, results in 72000 flowers -/
theorem unicorn_journey_flowers :
  unicorn_flowers 6 9 3 4 = 72000 := by
  sorry

end unicorn_journey_flowers_l2833_283341


namespace cafeteria_pies_l2833_283352

/-- Given a cafeteria scenario with initial apples, apples handed out, and apples per pie,
    calculate the number of pies that can be made. -/
theorem cafeteria_pies (initial_apples handed_out apples_per_pie : ℕ) 
    (h1 : initial_apples = 62)
    (h2 : handed_out = 8)
    (h3 : apples_per_pie = 9) :
    (initial_apples - handed_out) / apples_per_pie = 6 := by
  sorry

end cafeteria_pies_l2833_283352


namespace min_soldiers_to_add_l2833_283336

theorem min_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) :
  ∃ (k : ℕ), k = 82 ∧ (N + k) % 7 = 0 ∧ (N + k) % 12 = 0 ∧
  ∀ (m : ℕ), m < k → (N + m) % 7 ≠ 0 ∨ (N + m) % 12 ≠ 0 :=
by sorry

end min_soldiers_to_add_l2833_283336


namespace guilty_pair_is_B_and_C_l2833_283372

/-- Represents the guilt status of a defendant -/
inductive GuiltStatus
| Guilty
| Innocent

/-- Represents a defendant -/
inductive Defendant
| A
| B
| C

/-- The guilt status of all defendants -/
def GuiltStatusSet := Defendant → GuiltStatus

/-- At least one of the defendants is guilty -/
def atLeastOneGuilty (gs : GuiltStatusSet) : Prop :=
  ∃ d : Defendant, gs d = GuiltStatus.Guilty

/-- If A is guilty and B is innocent, then C is innocent -/
def conditionalInnocence (gs : GuiltStatusSet) : Prop :=
  (gs Defendant.A = GuiltStatus.Guilty ∧ gs Defendant.B = GuiltStatus.Innocent) →
  gs Defendant.C = GuiltStatus.Innocent

/-- The main theorem stating that B and C are the two defendants such that one of them is definitely guilty -/
theorem guilty_pair_is_B_and_C :
  ∀ gs : GuiltStatusSet,
  atLeastOneGuilty gs →
  conditionalInnocence gs →
  (gs Defendant.B = GuiltStatus.Guilty ∨ gs Defendant.C = GuiltStatus.Guilty) :=
sorry

end guilty_pair_is_B_and_C_l2833_283372


namespace bookstore_shipment_count_l2833_283342

theorem bookstore_shipment_count :
  ∀ (total : ℕ) (displayed : ℕ) (stockroom : ℕ),
    displayed = (30 : ℕ) * total / 100 →
    stockroom = (70 : ℕ) * total / 100 →
    stockroom = 140 →
    total = 200 := by
  sorry

end bookstore_shipment_count_l2833_283342


namespace land_area_proof_l2833_283339

theorem land_area_proof (original_side : ℝ) (cut_width : ℝ) (remaining_area : ℝ) :
  cut_width = 10 →
  remaining_area = 1575 →
  original_side * (original_side - cut_width) = remaining_area →
  original_side * cut_width = 450 :=
by
  sorry

end land_area_proof_l2833_283339


namespace absolute_value_equation_solution_l2833_283381

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 1| = |x - 3| :=
by
  -- Proof goes here
  sorry

end absolute_value_equation_solution_l2833_283381


namespace min_sum_reciprocals_l2833_283309

theorem min_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 24) :
  ∃ (a b : ℕ+), ((1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 24) ∧ (a ≠ b) ∧ (a.val + b.val = 96) ∧ (∀ (c d : ℕ+), ((1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 24) → (c ≠ d) → (c.val + d.val ≥ 96)) :=
by sorry

end min_sum_reciprocals_l2833_283309


namespace smallest_k_for_inequality_l2833_283346

theorem smallest_k_for_inequality : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ n > 0 → a^k * (1-a)^n < 1 / (n+1)^3) ∧
  (∀ (k' : ℕ), 0 < k' ∧ k' < k → 
    ∃ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ n > 0 ∧ a^k' * (1-a)^n ≥ 1 / (n+1)^3) ∧
  k = 4 :=
sorry

end smallest_k_for_inequality_l2833_283346


namespace rope_division_l2833_283318

/-- Given a rope of 1 meter length divided into two parts, where the second part is twice the length of the first part, prove that the length of the first part is 1/3 meter. -/
theorem rope_division (x : ℝ) (h1 : x > 0) (h2 : x + 2*x = 1) : x = 1/3 := by
  sorry

end rope_division_l2833_283318


namespace negation_of_all_teachers_generous_l2833_283397

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a teacher and being generous
variable (teacher : U → Prop)
variable (generous : U → Prop)

-- State the theorem
theorem negation_of_all_teachers_generous :
  (¬ ∀ x, teacher x → generous x) ↔ (∃ x, teacher x ∧ ¬ generous x) :=
by sorry

end negation_of_all_teachers_generous_l2833_283397


namespace initial_birds_count_l2833_283323

theorem initial_birds_count (initial_birds additional_birds total_birds : ℕ) 
  (h1 : additional_birds = 13)
  (h2 : total_birds = 42)
  (h3 : initial_birds + additional_birds = total_birds) : 
  initial_birds = 29 := by
  sorry

end initial_birds_count_l2833_283323


namespace mistake_permutations_four_letter_word_l2833_283308

/-- The number of permutations of a word with repeated letters -/
def permutations_with_repetition (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial r

/-- The number of mistake permutations for a 4-letter word with one letter repeated twice -/
theorem mistake_permutations_four_letter_word : 
  permutations_with_repetition 4 2 - 1 = 11 := by
  sorry

#eval permutations_with_repetition 4 2 - 1

end mistake_permutations_four_letter_word_l2833_283308


namespace x_cubed_coefficient_l2833_283321

theorem x_cubed_coefficient (p q : Polynomial ℤ) (hp : p = 3 * X ^ 4 - 2 * X ^ 3 + X ^ 2 - 3) 
  (hq : q = 2 * X ^ 2 + 5 * X - 4) : 
  (p * q).coeff 3 = 13 := by sorry

end x_cubed_coefficient_l2833_283321


namespace open_box_volume_l2833_283311

/-- The volume of an open box formed by cutting squares from a rectangular sheet -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_square_side : ℝ) 
  (h1 : sheet_length = 48) 
  (h2 : sheet_width = 36) 
  (h3 : cut_square_side = 8) : 
  (sheet_length - 2 * cut_square_side) * 
  (sheet_width - 2 * cut_square_side) * 
  cut_square_side = 5120 := by
sorry

end open_box_volume_l2833_283311


namespace interest_rate_calculation_l2833_283399

/-- The interest rate (as a percentage) at which A lent money to B -/
def interest_rate_A_to_B : ℝ := 10

/-- The principal amount lent -/
def principal : ℝ := 3500

/-- The interest rate (as a percentage) at which B lent money to C -/
def interest_rate_B_to_C : ℝ := 15

/-- The time period in years -/
def time : ℝ := 3

/-- B's gain over the time period -/
def B_gain : ℝ := 525

theorem interest_rate_calculation :
  let interest_C := principal * interest_rate_B_to_C / 100 * time
  let interest_A := interest_C - B_gain
  interest_A = principal * interest_rate_A_to_B / 100 * time := by
  sorry

end interest_rate_calculation_l2833_283399


namespace line_through_points_l2833_283388

-- Define a structure for points
structure Point where
  x : ℝ
  y : ℝ

-- Define the line passing through the given points
def line_equation (x : ℝ) : ℝ := 3 * x + 2

-- Define the given points
def p1 : Point := ⟨2, 8⟩
def p2 : Point := ⟨4, 14⟩
def p3 : Point := ⟨6, 20⟩
def p4 : Point := ⟨35, line_equation 35⟩

-- Theorem statement
theorem line_through_points :
  p1.y = line_equation p1.x ∧
  p2.y = line_equation p2.x ∧
  p3.y = line_equation p3.x ∧
  p4.y = 107 := by
  sorry

end line_through_points_l2833_283388


namespace eunji_class_size_l2833_283360

/-- The number of students in Eunji's class -/
def class_size : ℕ := 24

/-- The number of lines the students stand in -/
def num_lines : ℕ := 3

/-- Eunji's position from the front of her row -/
def position_from_front : ℕ := 3

/-- Eunji's position from the back of her row -/
def position_from_back : ℕ := 6

/-- Theorem stating the number of students in Eunji's class -/
theorem eunji_class_size :
  class_size = num_lines * (position_from_front + position_from_back - 1) :=
by sorry

end eunji_class_size_l2833_283360


namespace computer_printer_price_l2833_283392

/-- The total price of a basic computer and printer -/
def total_price (basic_computer_price printer_price : ℝ) : ℝ :=
  basic_computer_price + printer_price

/-- The price of an enhanced computer -/
def enhanced_computer_price (basic_computer_price : ℝ) : ℝ :=
  basic_computer_price + 500

/-- Condition for printer price with enhanced computer -/
def printer_price_condition (basic_computer_price printer_price : ℝ) : Prop :=
  printer_price = (1/3) * (enhanced_computer_price basic_computer_price + printer_price)

theorem computer_printer_price :
  ∃ (printer_price : ℝ),
    let basic_computer_price := 1500
    printer_price_condition basic_computer_price printer_price ∧
    total_price basic_computer_price printer_price = 2500 := by
  sorry

end computer_printer_price_l2833_283392


namespace prank_combinations_l2833_283354

/-- The number of choices for each day of the week --/
def monday_choices : ℕ := 1
def tuesday_choices : ℕ := 4
def wednesday_choices : ℕ := 7
def thursday_choices : ℕ := 5
def friday_choices : ℕ := 1

/-- The total number of combinations --/
def total_combinations : ℕ := monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

/-- Theorem stating that the total number of combinations is 140 --/
theorem prank_combinations : total_combinations = 140 := by
  sorry

end prank_combinations_l2833_283354


namespace inequality_solution_set_l2833_283307

theorem inequality_solution_set (x : ℝ) :
  (x - 1) * (x + 1) * (x - 2) < 0 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioo 1 2 := by
  sorry

end inequality_solution_set_l2833_283307


namespace cute_5digit_integer_count_l2833_283312

/-- A function that checks if a list of digits forms a palindrome -/
def isPalindrome (digits : List Nat) : Prop :=
  digits = digits.reverse

/-- A function that checks if the first k digits of a number are divisible by k -/
def firstKDigitsDivisibleByK (digits : List Nat) (k : Nat) : Prop :=
  let firstK := digits.take k
  let num := firstK.foldl (fun acc d => acc * 10 + d) 0
  num % k = 0

/-- A function that checks if a list of digits satisfies all conditions -/
def isCute (digits : List Nat) : Prop :=
  digits.length = 5 ∧
  digits.toFinset = {1, 2, 3, 4, 5} ∧
  isPalindrome digits ∧
  ∀ k, 1 ≤ k ∧ k ≤ 5 → firstKDigitsDivisibleByK digits k

theorem cute_5digit_integer_count :
  ∃! digits : List Nat, isCute digits :=
sorry

end cute_5digit_integer_count_l2833_283312


namespace no_solution_to_equation_l2833_283327

theorem no_solution_to_equation :
  ¬∃ x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3 * x^2 - 15 * x) / (x^2 - 5 * x) = x - 2 := by
  sorry

end no_solution_to_equation_l2833_283327


namespace root_equation_difference_l2833_283315

theorem root_equation_difference (a b : ℤ) :
  (∃ x : ℝ, x^2 = 7 - 4 * Real.sqrt 3 ∧ x^2 + a * x + b = 0) →
  b - a = 5 := by
  sorry

end root_equation_difference_l2833_283315


namespace binomial_20_19_l2833_283368

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end binomial_20_19_l2833_283368


namespace altitude_angle_relation_l2833_283324

/-- For an acute-angled triangle with circumradius R and altitude h from a vertex,
    the angle α at that vertex satisfies the given conditions. -/
theorem altitude_angle_relation (α : Real) (R h : ℝ) : 
  (α < Real.pi / 3 ↔ h < R) ∧ 
  (α = Real.pi / 3 ↔ h = R) ∧ 
  (α > Real.pi / 3 ↔ h > R) :=
by sorry

end altitude_angle_relation_l2833_283324


namespace point_identity_l2833_283319

/-- Given a point P(x, y) in the plane, prove that s^2 + c^2 = 1 where
    r is the distance from the origin to P,
    s = y/r,
    c = x/r,
    and c^2 = 4/9 -/
theorem point_identity (x y : ℝ) : 
  let r := Real.sqrt (x^2 + y^2)
  let s := y / r
  let c := x / r
  c^2 = 4/9 →
  s^2 + c^2 = 1 := by
sorry

end point_identity_l2833_283319


namespace joshuas_bottle_caps_l2833_283338

/-- The total number of bottle caps after buying more -/
def total_bottle_caps (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem: Joshua's total bottle caps -/
theorem joshuas_bottle_caps : total_bottle_caps 40 7 = 47 := by
  sorry

end joshuas_bottle_caps_l2833_283338


namespace simplify_complex_fraction_l2833_283332

theorem simplify_complex_fraction (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 3) * (x - 4) * (x - 2)) := by
  sorry

end simplify_complex_fraction_l2833_283332


namespace special_sum_value_l2833_283304

theorem special_sum_value (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ + 13*x₇ = 0)
  (eq2 : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ + 15*x₇ = 10)
  (eq3 : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ + 17*x₇ = 104) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ + 19*x₇ = 282 :=
by
  sorry

end special_sum_value_l2833_283304


namespace siblings_difference_l2833_283363

theorem siblings_difference (masud_siblings : ℕ) : 
  masud_siblings = 60 →
  let janet_siblings := 4 * masud_siblings - 60
  let carlos_siblings := (3 * masud_siblings) / 4
  janet_siblings - carlos_siblings = 45 := by
sorry

end siblings_difference_l2833_283363


namespace perpendicular_line_equation_l2833_283347

/-- Given a line l with y-intercept 1 and perpendicular to y = (1/2)x, 
    prove that the equation of l is y = -2x + 1 -/
theorem perpendicular_line_equation (l : Set (ℝ × ℝ)) 
  (y_intercept : (0, 1) ∈ l)
  (perpendicular : ∀ (x y : ℝ), (x, y) ∈ l → (y - 1) = m * x → m * (1/2) = -1) :
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y = -2 * x + 1 :=
sorry

end perpendicular_line_equation_l2833_283347


namespace symmetric_points_line_equation_l2833_283359

/-- Given two points are symmetric about a line, prove the equation of the line -/
theorem symmetric_points_line_equation (O A : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  O = (0, 0) → 
  A = (-4, 2) → 
  (∀ p : ℝ × ℝ, p ∈ l ↔ (2 : ℝ) * p.1 - p.2 + 5 = 0) →
  (∀ p : ℝ × ℝ, p ∈ l → dist O p = dist A p) →
  True :=
by sorry

end symmetric_points_line_equation_l2833_283359


namespace function_nonnegative_implies_inequalities_l2833_283391

/-- Given real constants a, b, A, B, and a function f(θ) = 1 - a cos θ - b sin θ - A sin 2θ - B cos 2θ,
    if f(θ) ≥ 0 for all real θ, then a² + b² ≤ 2 and A² + B² ≤ 1. -/
theorem function_nonnegative_implies_inequalities (a b A B : ℝ) :
  (∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.sin (2 * θ) - B * Real.cos (2 * θ) ≥ 0) →
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end function_nonnegative_implies_inequalities_l2833_283391


namespace accidental_calculation_l2833_283326

theorem accidental_calculation (x : ℝ) : (x + 12) / 8 = 8 → (x - 12) * 9 = 360 := by
  sorry

end accidental_calculation_l2833_283326


namespace complex_product_real_iff_condition_l2833_283395

theorem complex_product_real_iff_condition (a b c d : ℝ) :
  let Z1 : ℂ := Complex.mk a b
  let Z2 : ℂ := Complex.mk c d
  (Z1 * Z2).im = 0 ↔ a * d + b * c = 0 := by sorry

end complex_product_real_iff_condition_l2833_283395


namespace households_with_only_bike_l2833_283370

/-- Given a neighborhood with the following properties:
  * There are 90 total households
  * 11 households have neither a car nor a bike
  * 18 households have both a car and a bike
  * 44 households have a car (including those with both)
  Then the number of households with only a bike is 35. -/
theorem households_with_only_bike
  (total : ℕ)
  (neither : ℕ)
  (both : ℕ)
  (with_car : ℕ)
  (h_total : total = 90)
  (h_neither : neither = 11)
  (h_both : both = 18)
  (h_with_car : with_car = 44) :
  total - neither - (with_car - both) - both = 35 := by
  sorry

end households_with_only_bike_l2833_283370


namespace rational_equation_solution_l2833_283305

theorem rational_equation_solution :
  ∃! x : ℚ, (x ≠ 2/3) ∧ (x ≠ -3) ∧
  ((7*x + 3) / (3*x^2 + 7*x - 6) = (5*x) / (3*x - 2)) ∧
  x = 1/5 := by
sorry

end rational_equation_solution_l2833_283305
