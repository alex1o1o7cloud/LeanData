import Mathlib

namespace discount_calculation_l3369_336948

theorem discount_calculation (list_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  list_price = 70 →
  final_price = 56.16 →
  second_discount = 10.857142857142863 →
  ∃ first_discount : ℝ,
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    first_discount = 10 :=
by sorry

end discount_calculation_l3369_336948


namespace propositions_truth_l3369_336914

-- Definition of correlation coefficient
def correlation_strength (r : ℝ) : ℝ := 1 - |r|

-- Definition of perpendicular lines
def perpendicular (A B C A' B' C' : ℝ) : Prop := A * A' + B * B' = 0

theorem propositions_truth : 
  -- Proposition 1 (false)
  (∃ x : ℝ, x^2 < 0 ↔ ¬ ∀ x : ℝ, x^2 ≥ 0) ∧
  -- Proposition 2 (true)
  (∀ r : ℝ, |r| ≤ 1 → correlation_strength r ≤ correlation_strength 0) ∧
  -- Proposition 3 (false, not included)
  -- Proposition 4 (true)
  perpendicular 2 10 6 3 (-3/5) (13/5) :=
by sorry

end propositions_truth_l3369_336914


namespace three_distinct_real_roots_l3369_336960

/-- A cubic polynomial with specific conditions -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  b_neg : b < 0
  ab_eq_9c : a * b = 9 * c

/-- The polynomial function -/
def polynomial (p : CubicPolynomial) (x : ℝ) : ℝ :=
  x^3 + p.a * x^2 + p.b * x + p.c

/-- Theorem stating that the polynomial has three different real roots -/
theorem three_distinct_real_roots (p : CubicPolynomial) :
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    polynomial p x = 0 ∧ polynomial p y = 0 ∧ polynomial p z = 0 := by
  sorry

end three_distinct_real_roots_l3369_336960


namespace expression_defined_iff_l3369_336930

theorem expression_defined_iff (x : ℝ) :
  (∃ y : ℝ, y = (Real.log (3 - x)) / Real.sqrt (x - 1)) ↔ 1 < x ∧ x < 3 := by
  sorry

end expression_defined_iff_l3369_336930


namespace tetrahedron_division_l3369_336923

/-- A regular tetrahedron with unit edge length -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_unit : edge_length = 1

/-- Perpendicular bisector plane of a tetrahedron -/
structure PerpendicularBisectorPlane (t : RegularTetrahedron) where

/-- The number of parts the perpendicular bisector planes divide the tetrahedron into -/
def num_parts (t : RegularTetrahedron) : ℕ := sorry

/-- The volume of each part after division -/
def part_volume (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the number of parts and their volumes -/
theorem tetrahedron_division (t : RegularTetrahedron) :
  num_parts t = 24 ∧ part_volume t = Real.sqrt 2 / 288 := by sorry

end tetrahedron_division_l3369_336923


namespace min_value_sum_reciprocals_l3369_336998

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_one : x + y + z = 1) :
  1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 9 / 4 ∧
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 ∧
    1 / (a + 3 * b) + 1 / (b + 3 * c) + 1 / (c + 3 * a) = 9 / 4 := by
  sorry

end min_value_sum_reciprocals_l3369_336998


namespace polynomial_remainder_theorem_l3369_336925

theorem polynomial_remainder_theorem (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 8*x^4 - 18*x^3 + 5*x^2 - 3*x - 30
  let g : ℝ → ℝ := λ x => 2*x - 4
  f 2 = -32 ∧ (∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + f 2) :=
by sorry

end polynomial_remainder_theorem_l3369_336925


namespace english_score_calculation_l3369_336983

theorem english_score_calculation (average_before : ℝ) (average_after : ℝ) : 
  average_before = 92 →
  average_after = 94 →
  (3 * average_before + english_score) / 4 = average_after →
  english_score = 100 :=
by
  sorry

#check english_score_calculation

end english_score_calculation_l3369_336983


namespace smallest_lcm_with_gcd_five_l3369_336941

theorem smallest_lcm_with_gcd_five (k ℓ : ℕ) : 
  k ≥ 1000 → k < 10000 → ℓ ≥ 1000 → ℓ < 10000 → Nat.gcd k ℓ = 5 → 
  Nat.lcm k ℓ ≥ 201000 :=
by sorry

end smallest_lcm_with_gcd_five_l3369_336941


namespace reporters_not_covering_politics_l3369_336977

/-- The percentage of reporters who cover local politics in country X -/
def local_politics_coverage : ℝ := 30

/-- The percentage of reporters who cover politics but not local politics in country X -/
def non_local_politics_coverage : ℝ := 25

/-- Theorem stating that 60% of reporters do not cover politics -/
theorem reporters_not_covering_politics :
  let total_reporters : ℝ := 100
  let reporters_covering_local_politics : ℝ := local_politics_coverage
  let reporters_covering_politics : ℝ := reporters_covering_local_politics / (1 - non_local_politics_coverage / 100)
  let reporters_not_covering_politics : ℝ := total_reporters - reporters_covering_politics
  reporters_not_covering_politics / total_reporters = 0.6 := by
  sorry

end reporters_not_covering_politics_l3369_336977


namespace zeros_of_f_l3369_336913

def f (x : ℝ) := x^2 - 3*x + 2

theorem zeros_of_f :
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 := by
sorry

end zeros_of_f_l3369_336913


namespace plywood_perimeter_difference_l3369_336971

def plywood_width : ℝ := 3
def plywood_length : ℝ := 9
def num_pieces : ℕ := 6

def is_valid_cut (w h : ℝ) : Prop :=
  w * h * num_pieces = plywood_width * plywood_length ∧
  (w = plywood_width ∨ h = plywood_width ∨ w = plywood_length ∨ h = plywood_length ∨
   w * num_pieces = plywood_width ∨ h * num_pieces = plywood_width ∨
   w * num_pieces = plywood_length ∨ h * num_pieces = plywood_length)

def piece_perimeter (w h : ℝ) : ℝ := 2 * (w + h)

def max_perimeter : ℝ := 20
def min_perimeter : ℝ := 8

theorem plywood_perimeter_difference :
  ∀ w h, is_valid_cut w h →
  ∃ max_w max_h min_w min_h,
    is_valid_cut max_w max_h ∧
    is_valid_cut min_w min_h ∧
    piece_perimeter max_w max_h = max_perimeter ∧
    piece_perimeter min_w min_h = min_perimeter ∧
    max_perimeter - min_perimeter = 12 :=
sorry

end plywood_perimeter_difference_l3369_336971


namespace unique_solution_3m_plus_4n_eq_5k_l3369_336929

theorem unique_solution_3m_plus_4n_eq_5k :
  ∀ m n k : ℕ+, 3 * m + 4 * n = 5 * k → m = 2 ∧ n = 2 ∧ k = 4 := by
  sorry

end unique_solution_3m_plus_4n_eq_5k_l3369_336929


namespace apple_stack_theorem_l3369_336945

/-- Calculates the number of apples in a cubic-like stack --/
def appleStack (baseSize : Nat) : Nat :=
  let numLayers := baseSize
  List.range numLayers
    |> List.map (fun i => (baseSize - i) ^ 3)
    |> List.sum

theorem apple_stack_theorem :
  appleStack 4 = 100 := by
  sorry

end apple_stack_theorem_l3369_336945


namespace inequality_proof_l3369_336938

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 1 / b + 1 / (a * b) ≥ 8 := by
  sorry

end inequality_proof_l3369_336938


namespace percentage_students_taking_music_l3369_336908

/-- The percentage of students taking music in a school with various electives -/
theorem percentage_students_taking_music
  (total_students : ℕ)
  (dance_percent : ℝ)
  (art_percent : ℝ)
  (drama_percent : ℝ)
  (sports_percent : ℝ)
  (photography_percent : ℝ)
  (h_total : total_students = 3000)
  (h_dance : dance_percent = 12.5)
  (h_art : art_percent = 22)
  (h_drama : drama_percent = 13.5)
  (h_sports : sports_percent = 15)
  (h_photo : photography_percent = 8) :
  100 - (dance_percent + art_percent + drama_percent + sports_percent + photography_percent) = 29 := by
  sorry

end percentage_students_taking_music_l3369_336908


namespace prob_select_5_times_expectation_X_l3369_336976

-- Define the review types
inductive ReviewType
  | Good
  | Neutral
  | Bad

-- Define the age groups
inductive AgeGroup
  | Below50
  | Above50

-- Define the sample data
def sampleData : Fin 2 → Fin 3 → Nat
  | ⟨0, _⟩ => fun
    | ⟨0, _⟩ => 10000  -- Good reviews for Below50
    | ⟨1, _⟩ => 2000   -- Neutral reviews for Below50
    | ⟨2, _⟩ => 2000   -- Bad reviews for Below50
  | ⟨1, _⟩ => fun
    | ⟨0, _⟩ => 2000   -- Good reviews for Above50
    | ⟨1, _⟩ => 3000   -- Neutral reviews for Above50
    | ⟨2, _⟩ => 1000   -- Bad reviews for Above50

-- Define the total sample size
def totalSampleSize : Nat := 20000

-- Define the probability of selecting a good review
def probGoodReview : Rat :=
  (sampleData ⟨0, sorry⟩ ⟨0, sorry⟩ + sampleData ⟨1, sorry⟩ ⟨0, sorry⟩) / totalSampleSize

-- Theorem for the probability of selecting 5 times
theorem prob_select_5_times :
  (1 - probGoodReview)^5 + (1 - probGoodReview)^4 * probGoodReview = 16/625 := by sorry

-- Define the number of people giving neutral reviews in each age group
def neutralReviews : Fin 2 → Nat
  | ⟨0, _⟩ => sampleData ⟨0, sorry⟩ ⟨1, sorry⟩
  | ⟨1, _⟩ => sampleData ⟨1, sorry⟩ ⟨1, sorry⟩

-- Define the total number of neutral reviews
def totalNeutralReviews : Nat := neutralReviews ⟨0, sorry⟩ + neutralReviews ⟨1, sorry⟩

-- Define the probability distribution of X
def probX : Fin 4 → Rat
  | ⟨0, _⟩ => 1/6
  | ⟨1, _⟩ => 1/2
  | ⟨2, _⟩ => 3/10
  | ⟨3, _⟩ => 1/30

-- Theorem for the mathematical expectation of X
theorem expectation_X :
  (0 : Rat) * probX ⟨0, sorry⟩ + 1 * probX ⟨1, sorry⟩ + 2 * probX ⟨2, sorry⟩ + 3 * probX ⟨3, sorry⟩ = 6/5 := by sorry

end prob_select_5_times_expectation_X_l3369_336976


namespace tree_height_proof_l3369_336965

/-- The growth rate of the tree in inches per year -/
def growth_rate : ℝ := 0.5

/-- The number of years it takes for the tree to reach its final height -/
def years_to_grow : ℕ := 240

/-- The final height of the tree in inches -/
def final_height : ℝ := 720

/-- The current height of the tree in inches -/
def current_height : ℝ := final_height - (growth_rate * years_to_grow)

theorem tree_height_proof :
  current_height = 600 := by sorry

end tree_height_proof_l3369_336965


namespace section_b_average_weight_l3369_336993

/-- Given a class with two sections A and B, prove that the average weight of section B is 35 kg. -/
theorem section_b_average_weight
  (students_a : ℕ)
  (students_b : ℕ)
  (total_students : ℕ)
  (avg_weight_a : ℝ)
  (avg_weight_total : ℝ)
  (h1 : students_a = 30)
  (h2 : students_b = 20)
  (h3 : total_students = students_a + students_b)
  (h4 : avg_weight_a = 40)
  (h5 : avg_weight_total = 38)
  : (total_students * avg_weight_total - students_a * avg_weight_a) / students_b = 35 := by
  sorry

#check section_b_average_weight

end section_b_average_weight_l3369_336993


namespace billy_bumper_rides_l3369_336974

/-- The number of times Billy rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The cost of each ride in tickets -/
def ticket_cost : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := 50

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := (total_tickets - ferris_rides * ticket_cost) / ticket_cost

theorem billy_bumper_rides : bumper_rides = 3 := by
  sorry

end billy_bumper_rides_l3369_336974


namespace second_caterer_more_cost_effective_l3369_336949

/-- Represents the cost function for a caterer -/
structure Caterer where
  base_fee : ℕ
  per_person : ℕ

/-- Calculates the total cost for a given number of people -/
def total_cost (c : Caterer) (people : ℕ) : ℕ :=
  c.base_fee + c.per_person * people

/-- The first caterer's pricing structure -/
def caterer1 : Caterer :=
  { base_fee := 120, per_person := 14 }

/-- The second caterer's pricing structure -/
def caterer2 : Caterer :=
  { base_fee := 210, per_person := 11 }

/-- Theorem stating the minimum number of people for the second caterer to be more cost-effective -/
theorem second_caterer_more_cost_effective :
  (∀ n : ℕ, n ≥ 31 → total_cost caterer2 n < total_cost caterer1 n) ∧
  (∀ n : ℕ, n < 31 → total_cost caterer2 n ≥ total_cost caterer1 n) :=
sorry

end second_caterer_more_cost_effective_l3369_336949


namespace right_triangle_sets_l3369_336927

/-- A function that checks if three numbers can form a right-angled triangle -/
def isRightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

/-- The theorem stating that among the given sets, only {3, 4, 5} forms a right-angled triangle -/
theorem right_triangle_sets : 
  ¬ isRightTriangle 1 2 3 ∧
  isRightTriangle 3 4 5 ∧
  ¬ isRightTriangle 7 8 9 ∧
  ¬ isRightTriangle 5 10 20 :=
sorry

end right_triangle_sets_l3369_336927


namespace same_num_digits_l3369_336950

/-- The number of digits in the decimal representation of a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: If 10^b < a^b and 2^b < 10^b, then a^b and a^b + 2^b have the same number of digits -/
theorem same_num_digits (a b : ℕ) (h1 : 10^b < a^b) (h2 : 2^b < 10^b) :
  num_digits (a^b) = num_digits (a^b + 2^b) := by sorry

end same_num_digits_l3369_336950


namespace solution_set_part_I_range_of_a_part_II_l3369_336954

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| - |2*x - 1|

-- Part I
theorem solution_set_part_I :
  {x : ℝ | f x 2 + 3 ≥ 0} = {x : ℝ | -4 ≤ x ∧ x ≤ 2} := by sorry

-- Part II
theorem range_of_a_part_II :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 3, f x a ≤ 3) ↔ a ∈ Set.Icc (-3) 5 := by sorry

end solution_set_part_I_range_of_a_part_II_l3369_336954


namespace opposite_of_one_minus_cube_root_three_l3369_336905

theorem opposite_of_one_minus_cube_root_three :
  -(1 - Real.rpow 3 (1/3)) = Real.rpow 3 (1/3) - 1 := by
  sorry

end opposite_of_one_minus_cube_root_three_l3369_336905


namespace largest_n_with_unique_k_l3369_336933

theorem largest_n_with_unique_k : ∃ (n : ℕ), n > 0 ∧ 
  (∃! (k : ℤ), (5 : ℚ)/18 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 9/17) ∧
  (∀ (m : ℕ), m > n → ¬(∃! (k : ℤ), (5 : ℚ)/18 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 9/17)) ∧
  n = 1 :=
by sorry

end largest_n_with_unique_k_l3369_336933


namespace interest_rate_calculation_l3369_336996

theorem interest_rate_calculation (P : ℝ) (R : ℝ) : 
  P * (1 + 5 * R / 100) = 9800 →
  P * (1 + 8 * R / 100) = 12005 →
  R = 12 :=
by sorry

end interest_rate_calculation_l3369_336996


namespace credit_card_balance_l3369_336967

theorem credit_card_balance 
  (G : ℝ) 
  (gold_balance : ℝ) 
  (platinum_balance : ℝ) 
  (h1 : gold_balance = G / 3) 
  (h2 : 0.5833333333333334 = 1 - (platinum_balance + gold_balance) / (2 * G)) : 
  platinum_balance = (1 / 4) * (2 * G) := by
sorry

end credit_card_balance_l3369_336967


namespace intersection_when_m_eq_2_sufficient_not_necessary_condition_l3369_336953

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | (x-1+m)*(x-1-m) ≤ 0}

-- Theorem for part (1)
theorem intersection_when_m_eq_2 : 
  A ∩ B 2 = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

-- Theorem for part (2)
theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) ↔ m ≥ 5 := by sorry

end intersection_when_m_eq_2_sufficient_not_necessary_condition_l3369_336953


namespace deepak_age_l3369_336988

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 10 = 26 →
  deepak_age = 12 := by
sorry

end deepak_age_l3369_336988


namespace coupon_savings_difference_coupon_savings_difference_holds_l3369_336975

theorem coupon_savings_difference : ℝ → Prop :=
  fun difference =>
    ∃ (x y : ℝ),
      x > 120 ∧ y > 120 ∧
      (∀ p : ℝ, p > 120 →
        (0.2 * p ≥ 35 ∧ 0.2 * p ≥ 0.3 * (p - 120)) →
        x ≤ p ∧ p ≤ y) ∧
      (0.2 * x ≥ 35 ∧ 0.2 * x ≥ 0.3 * (x - 120)) ∧
      (0.2 * y ≥ 35 ∧ 0.2 * y ≥ 0.3 * (y - 120)) ∧
      difference = y - x ∧
      difference = 185

theorem coupon_savings_difference_holds : coupon_savings_difference 185 := by
  sorry

end coupon_savings_difference_coupon_savings_difference_holds_l3369_336975


namespace cube_cutting_problem_l3369_336910

theorem cube_cutting_problem :
  ∃! n : ℕ, ∃ s : ℕ, n > s ∧ n^3 - s^3 = 152 :=
by sorry

end cube_cutting_problem_l3369_336910


namespace teenager_toddler_ratio_l3369_336956

theorem teenager_toddler_ratio (total_children : ℕ) (toddlers : ℕ) (newborns : ℕ) : 
  total_children = 40 → toddlers = 6 → newborns = 4 → 
  (total_children - toddlers - newborns) / toddlers = 5 := by
  sorry

end teenager_toddler_ratio_l3369_336956


namespace elliot_book_pages_l3369_336962

/-- The number of pages in Elliot's book -/
def total_pages : ℕ := 381

/-- The number of pages Elliot has already read -/
def pages_read : ℕ := 149

/-- The number of pages Elliot reads per day -/
def pages_per_day : ℕ := 20

/-- The number of days Elliot reads -/
def days_reading : ℕ := 7

/-- The number of pages left to be read after reading for 7 days -/
def pages_left : ℕ := 92

theorem elliot_book_pages : 
  total_pages = pages_read + (pages_per_day * days_reading) + pages_left :=
by sorry

end elliot_book_pages_l3369_336962


namespace coin_and_die_probability_l3369_336912

def biased_coin_prob : ℚ := 3/4
def die_sides : ℕ := 6

theorem coin_and_die_probability :
  let heads_prob := biased_coin_prob
  let three_prob := 1 / die_sides
  heads_prob * three_prob = 1/8 := by
  sorry

end coin_and_die_probability_l3369_336912


namespace sallys_gold_card_balance_fraction_l3369_336994

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards -/
structure SallysCards where
  gold : CreditCard
  platinum : CreditCard

/-- The conditions of Sally's credit cards -/
def sallys_cards_conditions (cards : SallysCards) : Prop :=
  cards.platinum.limit = 2 * cards.gold.limit ∧
  cards.platinum.balance = (1 / 6) * cards.platinum.limit ∧
  cards.platinum.balance + cards.gold.balance = (1 / 3) * cards.platinum.limit

/-- The theorem representing the problem -/
theorem sallys_gold_card_balance_fraction (cards : SallysCards) 
  (h : sallys_cards_conditions cards) : 
  cards.gold.balance = (1 / 3) * cards.gold.limit := by
  sorry

end sallys_gold_card_balance_fraction_l3369_336994


namespace sphere_tangent_planes_properties_l3369_336939

/-- Given a sphere with radius r, this theorem proves various geometric properties related to
    tangent planes, spherical caps, and conical frustums. -/
theorem sphere_tangent_planes_properties (r : ℝ) (hr : r > 0) :
  ∃ (locus_radius : ℝ) (cap_area conical_area : ℝ),
    -- The locus of points P forms a sphere with radius r√3
    locus_radius = r * Real.sqrt 3 ∧
    -- The surface area of the smaller spherical cap
    cap_area = 2 * Real.pi * r^2 * (1 - Real.sqrt (2/3)) ∧
    -- The surface area of the conical frustum
    conical_area = Real.pi * r^2 * (2 * Real.sqrt 3 / 3) ∧
    -- The ratio of the two surface areas
    cap_area / conical_area = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end sphere_tangent_planes_properties_l3369_336939


namespace triangle_transformation_correct_l3369_336951

def initial_triangle : List (ℝ × ℝ) := [(1, -2), (-1, -2), (1, 1)]

def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def rotate_270_clockwise (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)

def transform_triangle (triangle : List (ℝ × ℝ)) : List (ℝ × ℝ) :=
  triangle.map (rotate_270_clockwise ∘ reflect_x_axis ∘ rotate_180)

theorem triangle_transformation_correct :
  transform_triangle initial_triangle = [(2, 1), (2, -1), (-1, -1)] := by
  sorry

end triangle_transformation_correct_l3369_336951


namespace quadratic_inequality_range_l3369_336920

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end quadratic_inequality_range_l3369_336920


namespace simplify_and_evaluate_l3369_336958

theorem simplify_and_evaluate : 
  ∀ x : ℝ, x ≠ -2 → x ≠ 1 → 
  (1 - 3 / (x + 2)) / ((x - 1) / (x^2 + 4*x + 4)) = x + 2 ∧
  (1 - 3 / (-1 + 2)) / ((-1 - 1) / ((-1)^2 + 4*(-1) + 4)) = 1 := by
sorry

end simplify_and_evaluate_l3369_336958


namespace crew_average_weight_increase_l3369_336992

theorem crew_average_weight_increase (initial_average : ℝ) : 
  let initial_total_weight := 20 * initial_average
  let new_total_weight := initial_total_weight + (80 - 40)
  let new_average := new_total_weight / 20
  new_average - initial_average = 2 := by
sorry

end crew_average_weight_increase_l3369_336992


namespace y_is_function_of_x_f_a_is_constant_not_always_injective_not_always_analytic_l3369_336991

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Statement 1: y is a function of x
theorem y_is_function_of_x : ∀ x : ℝ, ∃ y : ℝ, y = f x := by sorry

-- Statement 3: f(a) represents the value of the function f(x) when x = a, which is a constant
theorem f_a_is_constant (a : ℝ) : ∃ k : ℝ, f a = k := by sorry

-- Statement 2 (negation): It is not necessarily true that for different x, the value of y is also different
theorem not_always_injective : ¬ (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ ≠ f x₂) := by sorry

-- Statement 4 (negation): It is not always possible to represent f(x) by a specific formula
theorem not_always_analytic : ¬ (∃ formula : ℝ → ℝ, ∀ x : ℝ, f x = formula x) := by sorry

end y_is_function_of_x_f_a_is_constant_not_always_injective_not_always_analytic_l3369_336991


namespace lab_items_per_tech_l3369_336955

/-- Proves that each lab tech gets 14 items (coats and uniforms combined) given the problem conditions -/
theorem lab_items_per_tech (uniforms : ℕ) (coats : ℕ) (lab_techs : ℕ) : 
  uniforms = 12 →
  coats = 6 * uniforms →
  lab_techs = uniforms / 2 →
  (coats + uniforms) / lab_techs = 14 :=
by
  sorry

#check lab_items_per_tech

end lab_items_per_tech_l3369_336955


namespace binomial_square_constant_l3369_336986

theorem binomial_square_constant (a : ℚ) : 
  (∃ b c : ℚ, ∀ x, 9 * x^2 + 21 * x + a = (b * x + c)^2) → a = 49 / 4 := by
  sorry

end binomial_square_constant_l3369_336986


namespace inequality_system_solution_l3369_336969

theorem inequality_system_solution :
  ∃ (x y : ℝ), 
    (13 * x^2 - 4 * x * y + 4 * y^2 ≤ 2) ∧ 
    (2 * x - 4 * y ≤ -3) ∧
    (x = -1/3) ∧ 
    (y = 2/3) := by
  sorry

end inequality_system_solution_l3369_336969


namespace negation_of_universal_proposition_l3369_336972

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.sin x > 1) ↔ (∃ x₀ : ℝ, Real.sin x₀ ≤ 1) := by
  sorry

end negation_of_universal_proposition_l3369_336972


namespace sector_radius_l3369_336922

/-- Given a circular sector with area 10 cm² and arc length 4 cm, prove that the radius is 5 cm -/
theorem sector_radius (area : ℝ) (arc_length : ℝ) (radius : ℝ) 
  (h_area : area = 10) 
  (h_arc : arc_length = 4) 
  (h_sector : area = (arc_length * radius) / 2) : radius = 5 := by
  sorry

end sector_radius_l3369_336922


namespace john_coin_collection_value_l3369_336966

/-- Represents the value of John's coin collection -/
def coin_collection_value (total_coins : ℕ) (silver_coins : ℕ) (gold_coins : ℕ) 
  (silver_coin_value : ℚ) (regular_coin_value : ℚ) : ℚ :=
  let gold_coin_value := 2 * silver_coin_value
  let regular_coins := total_coins - (silver_coins + gold_coins)
  silver_coins * silver_coin_value + gold_coins * gold_coin_value + regular_coins * regular_coin_value

theorem john_coin_collection_value : 
  coin_collection_value 20 10 5 (30/4) 1 = 155 := by
  sorry


end john_coin_collection_value_l3369_336966


namespace parabola_hyperbola_focus_l3369_336995

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the directrix of the parabola
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p / 2

-- Define the left focus of the hyperbola
def left_focus_hyperbola (x y : ℝ) : Prop := x = -2 ∧ y = 0

-- Theorem statement
theorem parabola_hyperbola_focus (p : ℝ) :
  (∃ x y : ℝ, directrix p x ∧ left_focus_hyperbola x y) →
  p = 4 := by sorry

end parabola_hyperbola_focus_l3369_336995


namespace scientific_notation_748_million_l3369_336924

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a real number to scientific notation with given significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- Rounds a real number to a given number of significant figures -/
def roundToSigFigs (x : ℝ) (sigFigs : ℕ) : ℝ :=
  sorry

theorem scientific_notation_748_million :
  let original := (748 : ℝ) * 1000000
  let scientificForm := toScientificNotation original 2
  scientificForm = ScientificNotation.mk 7.5 8 := by
  sorry

end scientific_notation_748_million_l3369_336924


namespace characterize_functions_l3369_336931

-- Define the property of the function f
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = f ((x^2 + y^2) / 2) + (x - y)^2

-- State the theorem
theorem characterize_functions (f : ℝ → ℝ) 
  (hf : Continuous f) 
  (hprop : satisfies_property f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c - 2 * x :=
sorry

end characterize_functions_l3369_336931


namespace function_simplification_l3369_336964

theorem function_simplification (x : ℝ) : 
  Real.sqrt (4 * Real.sin x ^ 4 - 2 * Real.cos (2 * x) + 3) + 
  Real.sqrt (4 * Real.cos x ^ 4 + 2 * Real.cos (2 * x) + 3) = 4 := by
  sorry

end function_simplification_l3369_336964


namespace total_friends_l3369_336947

/-- The number of friends who attended the movie -/
def M : ℕ := 10

/-- The number of friends who attended the picnic -/
def P : ℕ := 20

/-- The number of friends who attended the games -/
def G : ℕ := 5

/-- The number of friends who attended both movie and picnic -/
def MP : ℕ := 4

/-- The number of friends who attended both movie and games -/
def MG : ℕ := 2

/-- The number of friends who attended both picnic and games -/
def PG : ℕ := 0

/-- The number of friends who attended all three events -/
def MPG : ℕ := 2

/-- The total number of unique friends -/
def N : ℕ := M + P + G - MP - MG - PG + MPG

theorem total_friends : N = 31 := by sorry

end total_friends_l3369_336947


namespace smallest_n_for_Q_less_than_threshold_l3369_336937

/-- The probability of stopping at box n, given n ≥ 50 -/
def Q (n : ℕ) : ℚ := 2 / (n + 2)

/-- The smallest n ≥ 50 such that Q(n) < 1/2023 is 1011 -/
theorem smallest_n_for_Q_less_than_threshold : 
  (∀ k : ℕ, k ≥ 50 → k < 1011 → Q k ≥ 1/2023) ∧ 
  (Q 1011 < 1/2023) := by
  sorry

end smallest_n_for_Q_less_than_threshold_l3369_336937


namespace metal_waste_calculation_l3369_336973

theorem metal_waste_calculation (R : ℝ) (h : R = 10) : 
  let original_circle_area := π * R^2
  let max_square_side := R * Real.sqrt 2
  let max_square_area := max_square_side^2
  let inner_circle_radius := max_square_side / 2
  let inner_circle_area := π * inner_circle_radius^2
  original_circle_area - inner_circle_area = 50 * π - 200 :=
by sorry

end metal_waste_calculation_l3369_336973


namespace negation_of_p_l3369_336934

-- Define the original proposition
def p : Prop := ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 3 > 0

-- State the theorem
theorem negation_of_p : 
  ¬p ↔ ∀ x : ℝ, x^2 + 2*x + 3 ≤ 0 := by sorry

end negation_of_p_l3369_336934


namespace narcissus_count_is_75_l3369_336979

/-- The number of narcissus flowers in a florist's inventory -/
def narcissus_count : ℕ := 75

/-- The number of chrysanthemums in the florist's inventory -/
def chrysanthemum_count : ℕ := 90

/-- The number of bouquets that can be made -/
def bouquet_count : ℕ := 33

/-- The number of flowers in each bouquet -/
def flowers_per_bouquet : ℕ := 5

/-- Theorem stating that the number of narcissus flowers is 75 -/
theorem narcissus_count_is_75 : 
  narcissus_count = bouquet_count * flowers_per_bouquet - chrysanthemum_count :=
by
  sorry

#eval narcissus_count -- Should output 75

end narcissus_count_is_75_l3369_336979


namespace complex_number_location_l3369_336989

theorem complex_number_location (z : ℂ) (h : z * (1 - 2*I) = I) :
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end complex_number_location_l3369_336989


namespace joan_found_70_seashells_l3369_336970

/-- The number of seashells Sam gave to Joan -/
def seashells_from_sam : ℕ := 27

/-- The total number of seashells Joan has now -/
def total_seashells : ℕ := 97

/-- The number of seashells Joan found on the beach -/
def seashells_found_on_beach : ℕ := total_seashells - seashells_from_sam

theorem joan_found_70_seashells : seashells_found_on_beach = 70 := by
  sorry

end joan_found_70_seashells_l3369_336970


namespace toothpick_grid_60_32_l3369_336963

/-- Calculates the total number of toothpicks in a rectangular grid -/
def total_toothpicks (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem: A 60x32 toothpick grid uses 3932 toothpicks -/
theorem toothpick_grid_60_32 :
  total_toothpicks 60 32 = 3932 := by
  sorry

end toothpick_grid_60_32_l3369_336963


namespace unique_solution_to_equation_l3369_336900

theorem unique_solution_to_equation : ∃! (x : ℝ), x ≠ 0 ∧ (7 * x)^5 = (14 * x)^4 ∧ x = 16/7 := by
  sorry

end unique_solution_to_equation_l3369_336900


namespace equal_powers_of_negative_one_l3369_336968

theorem equal_powers_of_negative_one : 
  (-7^4 ≠ (-7)^4) ∧ 
  (4^3 ≠ 3^4) ∧ 
  (-(-6) ≠ -|(-6)|) ∧ 
  ((-1)^3 = (-1)^2023) := by
  sorry

end equal_powers_of_negative_one_l3369_336968


namespace factorial_ratio_50_48_l3369_336909

theorem factorial_ratio_50_48 : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end factorial_ratio_50_48_l3369_336909


namespace total_passengers_l3369_336957

def bus_problem (initial_a initial_b new_a new_b : ℕ) : ℕ :=
  (initial_a + new_a) + (initial_b + new_b)

theorem total_passengers :
  bus_problem 4 7 13 9 = 33 := by
  sorry

end total_passengers_l3369_336957


namespace remaining_sausage_meat_l3369_336906

/-- Calculates the remaining sausage meat in ounces after some links are eaten -/
theorem remaining_sausage_meat 
  (total_pounds : ℕ) 
  (total_links : ℕ) 
  (eaten_links : ℕ) 
  (h1 : total_pounds = 10) 
  (h2 : total_links = 40) 
  (h3 : eaten_links = 12) : 
  (total_pounds * 16 - (total_pounds * 16 / total_links) * eaten_links : ℕ) = 112 := by
  sorry

#check remaining_sausage_meat

end remaining_sausage_meat_l3369_336906


namespace A_equals_two_three_l3369_336982

def A : Set ℤ := {x | (3 : ℚ) / (x - 1) > 1}

theorem A_equals_two_three : A = {2, 3} := by sorry

end A_equals_two_three_l3369_336982


namespace gas_pressure_volume_relationship_l3369_336921

/-- Given a gas with initial pressure P1, initial volume V1, and final volume V2,
    where pressure and volume are inversely proportional at constant temperature,
    prove that the final pressure P2 is equal to (P1 * V1) / V2. -/
theorem gas_pressure_volume_relationship (P1 V1 V2 : ℝ) (h1 : P1 > 0) (h2 : V1 > 0) (h3 : V2 > 0) :
  let P2 := (P1 * V1) / V2
  ∀ k : ℝ, (P1 * V1 = k ∧ P2 * V2 = k) → P2 = (P1 * V1) / V2 := by
sorry

end gas_pressure_volume_relationship_l3369_336921


namespace christen_peeled_17_l3369_336999

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initial_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  homer_solo_time : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christens_potatoes (scenario : PotatoPeeling) : ℕ :=
  let potatoes_after_homer := scenario.initial_potatoes - scenario.homer_rate * scenario.homer_solo_time
  let combined_rate := scenario.homer_rate + scenario.christen_rate
  let remaining_time := potatoes_after_homer / combined_rate
  remaining_time * scenario.christen_rate

/-- Theorem stating that Christen peeled 17 potatoes -/
theorem christen_peeled_17 (scenario : PotatoPeeling) 
  (h1 : scenario.initial_potatoes = 58)
  (h2 : scenario.homer_rate = 4)
  (h3 : scenario.christen_rate = 4)
  (h4 : scenario.homer_solo_time = 6) :
  christens_potatoes scenario = 17 := by
  sorry

#eval christens_potatoes { initial_potatoes := 58, homer_rate := 4, christen_rate := 4, homer_solo_time := 6 }

end christen_peeled_17_l3369_336999


namespace smallest_box_volume_l3369_336916

theorem smallest_box_volume (l w h : ℕ) (h1 : l > 0) (h2 : w = 3 * l) (h3 : h = 4 * l) :
  l * w * h = 96 ∨ l * w * h > 96 :=
sorry

end smallest_box_volume_l3369_336916


namespace als_original_investment_l3369_336942

-- Define the original investment amounts
variable (a b c d : ℝ)

-- Define the conditions
axiom total_investment : a + b + c + d = 1200
axiom different_amounts : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
axiom final_total : (a - 150) + 3*b + 3*c + 2*d = 1800

-- Theorem to prove
theorem als_original_investment : a = 825 := by
  sorry

end als_original_investment_l3369_336942


namespace distance_circle_center_to_point_l3369_336901

/-- The distance between the center of a circle with polar equation ρ = 4sin θ 
    and a point with polar coordinates (2√2, π/4) is 2 -/
theorem distance_circle_center_to_point : 
  let circle_equation : ℝ → ℝ := λ θ => 4 * Real.sin θ
  let point_A : ℝ × ℝ := (2 * Real.sqrt 2, Real.pi / 4)
  ∃ center : ℝ × ℝ, 
    Real.sqrt ((center.1 - (point_A.1 * Real.cos point_A.2))^2 + 
               (center.2 - (point_A.1 * Real.sin point_A.2))^2) = 2 :=
by sorry

end distance_circle_center_to_point_l3369_336901


namespace union_perimeter_bound_l3369_336961

/-- A disc in a 2D plane -/
structure Disc where
  center : ℝ × ℝ
  radius : ℝ

/-- A set of discs satisfying the problem conditions -/
structure DiscSet where
  discs : Set Disc
  segment_length : ℝ
  centers_on_segment : ∀ d ∈ discs, ∃ x : ℝ, d.center = (x, 0) ∧ 0 ≤ x ∧ x ≤ segment_length
  radii_bounded : ∀ d ∈ discs, d.radius ≤ 1

/-- The perimeter of the union of discs -/
noncomputable def union_perimeter (ds : DiscSet) : ℝ := sorry

/-- The main theorem -/
theorem union_perimeter_bound (ds : DiscSet) :
  union_perimeter ds ≤ 4 * ds.segment_length + 8 := by
  sorry

end union_perimeter_bound_l3369_336961


namespace exponent_division_l3369_336936

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end exponent_division_l3369_336936


namespace rectangle_area_l3369_336944

/-- The length of the shorter side of each small rectangle -/
def short_side : ℝ := 4

/-- The number of small rectangles -/
def num_rectangles : ℕ := 4

/-- The aspect ratio of each small rectangle -/
def aspect_ratio : ℝ := 2

/-- The length of the longer side of each small rectangle -/
def long_side : ℝ := short_side * aspect_ratio

/-- The width of rectangle EFGH -/
def width : ℝ := long_side

/-- The length of rectangle EFGH -/
def length : ℝ := 2 * long_side

/-- The area of rectangle EFGH -/
def area : ℝ := width * length

theorem rectangle_area : area = 128 := by sorry

end rectangle_area_l3369_336944


namespace fixed_point_of_exponential_function_l3369_336903

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 3
  f 2 = 4 :=
by sorry

end fixed_point_of_exponential_function_l3369_336903


namespace f_min_max_l3369_336911

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the domain
def domain : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem f_min_max :
  (∃ x ∈ domain, f x = 0) ∧
  (∀ x ∈ domain, f x ≥ 0) ∧
  (∃ x ∈ domain, f x = 9) ∧
  (∀ x ∈ domain, f x ≤ 9) := by
  sorry

end f_min_max_l3369_336911


namespace solution_set_f_leq_x_range_of_t_for_f_geq_t_squared_minus_t_l3369_336981

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |2*x + 1|

-- Theorem for part I
theorem solution_set_f_leq_x : 
  {x : ℝ | f x ≤ x} = {x : ℝ | x ≥ 1/4} :=
sorry

-- Theorem for part II
theorem range_of_t_for_f_geq_t_squared_minus_t : 
  {t : ℝ | ∀ x ∈ Set.Icc (-2) (-1), f x ≥ t^2 - t} = 
  Set.Icc ((1 - Real.sqrt 5) / 2) ((1 + Real.sqrt 5) / 2) :=
sorry

end solution_set_f_leq_x_range_of_t_for_f_geq_t_squared_minus_t_l3369_336981


namespace lucas_chocolate_problem_l3369_336943

theorem lucas_chocolate_problem (total_students : ℕ) 
  (candy_per_student : ℕ) 
  (h1 : total_students * candy_per_student = 40) 
  (h2 : (total_students - 3) * candy_per_student = 28) :
  candy_per_student = 4 := by
  sorry

end lucas_chocolate_problem_l3369_336943


namespace rubble_purchase_l3369_336985

/-- Calculates the remaining money after a purchase. -/
def remaining_money (initial_amount notebook_cost pen_cost : ℚ) : ℚ :=
  initial_amount - (2 * notebook_cost + 2 * pen_cost)

/-- Proves that Rubble will have $4.00 left after his purchase. -/
theorem rubble_purchase : remaining_money 15 4 (3/2) = 4 := by
  sorry

end rubble_purchase_l3369_336985


namespace base8_157_equals_111_l3369_336932

/-- Converts a base-8 number to base-10 --/
def base8To10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

/-- The base-8 representation of 157 --/
def base8_157 : List Nat := [1, 5, 7]

theorem base8_157_equals_111 :
  base8To10 base8_157 = 111 := by
  sorry

end base8_157_equals_111_l3369_336932


namespace at_least_one_boy_and_girl_l3369_336928

def probability_boy_or_girl : ℚ := 1 / 2

def number_of_children : ℕ := 4

theorem at_least_one_boy_and_girl :
  let p := probability_boy_or_girl
  let n := number_of_children
  (1 : ℚ) - (p^n + (1 - p)^n) = 7 / 8 := by sorry

end at_least_one_boy_and_girl_l3369_336928


namespace c_condition_l3369_336902

theorem c_condition (a b c : ℝ) (h1 : a < b) (h2 : a * c > b * c) : c < 0 := by
  sorry

end c_condition_l3369_336902


namespace digit_swap_difference_multiple_of_nine_l3369_336978

theorem digit_swap_difference_multiple_of_nine (a b : ℕ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) : 
  ∃ k : ℤ, (10 * a + b) - (10 * b + a) = 9 * k := by
  sorry

end digit_swap_difference_multiple_of_nine_l3369_336978


namespace extreme_value_implies_a_l3369_336984

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x^2

theorem extreme_value_implies_a (a : ℝ) :
  (∃ (ε : ℝ), ∀ (h : ℝ), 0 < |h| → |h| < ε → f a (-2 + h) ≤ f a (-2)) →
  a = 1/3 := by
  sorry

end extreme_value_implies_a_l3369_336984


namespace problem_solution_l3369_336997

theorem problem_solution (x y z : ℕ) : 
  x > 0 ∧ 
  x = 10 * y + 3 ∧ 
  2 * x = 7 * (3 * y) + 1 ∧ 
  3 * x = 5 * z + 2 → 
  11 * y - x + 7 * z = 219 := by
sorry

end problem_solution_l3369_336997


namespace custom_operation_solution_l3369_336940

-- Define the custom operation *
def star (a b : ℝ) : ℝ := 2 * a^2 - b

-- State the theorem
theorem custom_operation_solution :
  ∃ x : ℝ, (star 3 (star 4 x) = 8) ∧ (x = 22) := by
  sorry

end custom_operation_solution_l3369_336940


namespace square_perimeter_9cm_l3369_336987

/-- The perimeter of a square with side length 9 centimeters is 36 centimeters. -/
theorem square_perimeter_9cm (s : ℝ) (h : s = 9) : 4 * s = 36 := by
  sorry

end square_perimeter_9cm_l3369_336987


namespace max_length_sum_l3369_336926

/-- Length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer -/
def length (n : ℕ) : ℕ := sorry

theorem max_length_sum (x y : ℕ) (hx : x > 1) (hy : y > 1) (hsum : x + 3 * y < 920) :
  ∃ (a b : ℕ), length x + length y ≤ a + b ∧ a + b = 16 := by
  sorry

end max_length_sum_l3369_336926


namespace third_grade_girls_sample_l3369_336980

theorem third_grade_girls_sample (total_students : ℕ) (first_grade : ℕ) (second_grade : ℕ) (third_grade : ℕ)
  (first_boys : ℕ) (first_girls : ℕ) (second_boys : ℕ) (second_girls : ℕ) (third_boys : ℕ) (third_girls : ℕ)
  (sample_size : ℕ) :
  total_students = 3000 →
  first_grade = 800 →
  second_grade = 1000 →
  third_grade = 1200 →
  first_boys = 500 →
  first_girls = 300 →
  second_boys = 600 →
  second_girls = 400 →
  third_boys = 800 →
  third_girls = 400 →
  sample_size = 150 →
  first_grade + second_grade + third_grade = total_students →
  first_boys + first_girls = first_grade →
  second_boys + second_girls = second_grade →
  third_boys + third_girls = third_grade →
  (third_grade : ℚ) / (total_students : ℚ) * (sample_size : ℚ) * (third_girls : ℚ) / (third_grade : ℚ) = 20 := by
  sorry

end third_grade_girls_sample_l3369_336980


namespace strings_per_normal_guitar_is_6_l3369_336918

/-- Calculates the number of strings on each normal guitar given the following conditions:
  * There are 3 basses with 4 strings each
  * There are twice as many normal guitars as basses
  * There are 3 fewer 8-string guitars than normal guitars
  * The total number of strings needed is 72
-/
def strings_per_normal_guitar : ℕ :=
  let num_basses : ℕ := 3
  let strings_per_bass : ℕ := 4
  let num_normal_guitars : ℕ := 2 * num_basses
  let num_8string_guitars : ℕ := num_normal_guitars - 3
  let total_strings : ℕ := 72
  (total_strings - num_basses * strings_per_bass - num_8string_guitars * 8) / num_normal_guitars

theorem strings_per_normal_guitar_is_6 : strings_per_normal_guitar = 6 := by
  sorry

end strings_per_normal_guitar_is_6_l3369_336918


namespace x_value_in_set_l3369_336917

theorem x_value_in_set (x : ℝ) : 1 ∈ ({x, x^2} : Set ℝ) → x ≠ x^2 → x = -1 := by
  sorry

end x_value_in_set_l3369_336917


namespace picture_album_distribution_l3369_336959

theorem picture_album_distribution : ∃ (a b c : ℕ), 
  a + b + c = 40 ∧ 
  a + b = 28 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 :=
by sorry

end picture_album_distribution_l3369_336959


namespace min_pieces_correct_l3369_336935

/-- Represents a chessboard of size n x n -/
structure Chessboard (n : ℕ) where
  size : ℕ
  size_pos : size > 0
  size_eq : size = n

/-- A piece on the chessboard -/
structure Piece (n : ℕ) where
  x : Fin n
  y : Fin n

/-- A configuration of pieces on the chessboard -/
def Configuration (n : ℕ) := List (Piece n)

/-- Checks if a configuration satisfies the line coverage property -/
def satisfiesLineCoverage (n : ℕ) (config : Configuration n) : Prop := sorry

/-- The minimum number of pieces required for a valid configuration -/
def minPieces (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 * n else 2 * n + 1

/-- Theorem stating the minimum number of pieces required for a valid configuration -/
theorem min_pieces_correct (n : ℕ) (h : n > 0) :
  ∀ (config : Configuration n),
    satisfiesLineCoverage n config →
    config.length ≥ minPieces n :=
  sorry

end min_pieces_correct_l3369_336935


namespace smallest_part_of_proportional_division_l3369_336952

theorem smallest_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 81)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_proportional : ∃ (k : ℚ), a = 3 * k ∧ b = 5 * k ∧ c = 7 * k)
  (h_sum : a + b + c = total) :
  a = 81 / 5 := by
  sorry

end smallest_part_of_proportional_division_l3369_336952


namespace expression_evaluation_l3369_336915

theorem expression_evaluation : (-6)^6 / 6^4 + 4^5 - 7^2 * 2 = 890 := by
  sorry

end expression_evaluation_l3369_336915


namespace factor_of_x4_plus_8_l3369_336907

theorem factor_of_x4_plus_8 (x : ℝ) : 
  (x^2 - 2*x + 4) * (x^2 + 2*x + 4) = x^4 + 8 := by
  sorry

end factor_of_x4_plus_8_l3369_336907


namespace prime_factors_power_l3369_336946

/-- Given an expression containing 4^11, 7^5, and 11^x, 
    if the total number of prime factors is 29, then x = 2 -/
theorem prime_factors_power (x : ℕ) : 
  (22 + 5 + x = 29) → x = 2 := by
  sorry

end prime_factors_power_l3369_336946


namespace fox_rabbit_bridge_problem_l3369_336990

theorem fox_rabbit_bridge_problem (x : ℝ) : 
  (((2 * ((2 * ((2 * ((2 * x) - 50)) - 50)) - 50)) - 50) = 0) → x = 46.875 := by
  sorry

end fox_rabbit_bridge_problem_l3369_336990


namespace sequence_pattern_l3369_336919

def S : ℕ → ℕ
  | n => if n % 2 = 1 then 1 + 2 * ((n + 1) / 2 - 1) else 2^(n / 2)

theorem sequence_pattern (n : ℕ) : 
  S n = if n % 2 = 1 then 1 + 2 * ((n + 1) / 2 - 1) else 2^(n / 2) := by
  sorry

#eval [S 1, S 2, S 3, S 4, S 5, S 6, S 7, S 8, S 9, S 10, S 11, S 12, S 13, S 14]

end sequence_pattern_l3369_336919


namespace total_cost_is_correct_l3369_336904

-- Define the prices and quantities
def pasta_price : ℝ := 1.70
def pasta_quantity : ℝ := 3
def beef_price : ℝ := 8.20
def beef_quantity : ℝ := 0.5
def sauce_price : ℝ := 2.30
def sauce_quantity : ℝ := 3
def quesadillas_price : ℝ := 11.50
def discount_rate : ℝ := 0.10
def vat_rate : ℝ := 0.05

-- Define the total cost function
def total_cost : ℝ :=
  let pasta_cost := pasta_price * pasta_quantity
  let beef_cost := beef_price * beef_quantity
  let sauce_cost := sauce_price * sauce_quantity
  let discounted_sauce_cost := sauce_cost * (1 - discount_rate)
  let subtotal := pasta_cost + beef_cost + discounted_sauce_cost + quesadillas_price
  let vat := subtotal * vat_rate
  subtotal + vat

-- Theorem statement
theorem total_cost_is_correct : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ abs (total_cost - 28.26) < ε :=
sorry

end total_cost_is_correct_l3369_336904
