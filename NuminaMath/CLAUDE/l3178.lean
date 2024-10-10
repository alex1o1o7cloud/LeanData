import Mathlib

namespace root_in_interval_l3178_317844

-- Define the function f(x) = x^3 - x - 5
def f (x : ℝ) : ℝ := x^3 - x - 5

-- State the theorem
theorem root_in_interval :
  (f 1 < 0) → (f 2 > 0) → (f 1.5 < 0) →
  ∃ x : ℝ, x ∈ Set.Ioo 1.5 2 ∧ f x = 0 :=
by
  sorry


end root_in_interval_l3178_317844


namespace isosceles_triangle_cosine_l3178_317893

def IsoscelesTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ b = c

def LargestAngleThreeTimesSmallest (a b c : ℝ) : Prop :=
  let cosSmallest := (b^2 + c^2 - a^2) / (2 * b * c)
  let cosLargest := (a^2 + b^2 - c^2) / (2 * a * b)
  cosLargest = 4 * cosSmallest^3 - 3 * cosSmallest

theorem isosceles_triangle_cosine (n : ℕ) :
  IsoscelesTriangle n (n + 1) (n + 1) →
  LargestAngleThreeTimesSmallest n (n + 1) (n + 1) →
  let cosSmallest := ((n + 1)^2 + (n + 1)^2 - n^2) / (2 * (n + 1) * (n + 1))
  cosSmallest = 7 / 9 :=
sorry

end isosceles_triangle_cosine_l3178_317893


namespace expression_reduction_l3178_317839

theorem expression_reduction (a b c : ℝ) 
  (h1 : a^2 + c^2 - b^2 - 2*a*c ≠ 0)
  (h2 : a - b + c ≠ 0)
  (h3 : a - c + b ≠ 0) :
  (a^2 + b^2 - c^2 - 2*a*b) / (a^2 + c^2 - b^2 - 2*a*c) = 
  ((a - b + c) * (a - b - c)) / ((a - c + b) * (a - c - b)) := by
  sorry

end expression_reduction_l3178_317839


namespace is_factorization_l3178_317876

/-- Proves that x^2 - 4x + 4 = (x - 2)^2 is a factorization --/
theorem is_factorization (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end is_factorization_l3178_317876


namespace quadratic_roots_k_value_l3178_317811

theorem quadratic_roots_k_value (k : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 10 * x + k = 0 ↔ x = 5 + Real.sqrt 15 ∨ x = 5 - Real.sqrt 15) →
  k = 85 / 8 := by
sorry

end quadratic_roots_k_value_l3178_317811


namespace shoes_lost_l3178_317842

theorem shoes_lost (initial_pairs : ℕ) (remaining_pairs : ℕ) : 
  initial_pairs = 26 → remaining_pairs = 21 → initial_pairs * 2 - remaining_pairs * 2 = 10 := by
  sorry

end shoes_lost_l3178_317842


namespace second_chapter_page_difference_l3178_317815

/-- A book with three chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ
  chapter3_pages : ℕ

/-- The specific book described in the problem -/
def my_book : Book := {
  chapter1_pages := 35
  chapter2_pages := 18
  chapter3_pages := 3
}

/-- Theorem stating the difference in pages between the second and third chapters -/
theorem second_chapter_page_difference (b : Book := my_book) :
  b.chapter2_pages - b.chapter3_pages = 15 := by
  sorry

end second_chapter_page_difference_l3178_317815


namespace final_bacteria_count_l3178_317892

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 50

-- Define the doubling interval in minutes
def doubling_interval : ℕ := 4

-- Define the total time elapsed in minutes
def total_time : ℕ := 15

-- Define the number of complete doubling intervals
def complete_intervals : ℕ := total_time / doubling_interval

-- Function to calculate the bacteria population after a given number of intervals
def bacteria_population (intervals : ℕ) : ℕ :=
  initial_bacteria * (2 ^ intervals)

-- Theorem stating the final bacteria count
theorem final_bacteria_count :
  bacteria_population complete_intervals = 400 := by
  sorry

end final_bacteria_count_l3178_317892


namespace medal_distribution_proof_l3178_317863

def distribute_medals (n : ℕ) : ℕ :=
  Nat.choose (n + 2) 2

theorem medal_distribution_proof (n : ℕ) (h : n = 12) : 
  distribute_medals n = 55 := by
  sorry

end medal_distribution_proof_l3178_317863


namespace trig_identities_l3178_317834

theorem trig_identities (α : Real) (h : Real.tan α = 3) :
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - 4 * Real.cos α) = -11 ∧
  (5 * Real.cos α ^ 2 - 3 * Real.sin α ^ 2) / (1 + Real.sin α ^ 2) = -11/5 := by
  sorry

end trig_identities_l3178_317834


namespace diagonal_length_in_specific_kite_l3178_317805

/-- A kite is a quadrilateral with two pairs of adjacent sides of equal length -/
structure Kite :=
  (A B C D : ℝ × ℝ)
  (is_kite : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2 ∧
             (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2)

/-- The theorem about the diagonal length in a specific kite -/
theorem diagonal_length_in_specific_kite (k : Kite) 
  (ab_length : (k.A.1 - k.B.1)^2 + (k.A.2 - k.B.2)^2 = 100)
  (bc_length : (k.B.1 - k.C.1)^2 + (k.B.2 - k.C.2)^2 = 225)
  (sin_B : Real.sin (Real.arcsin ((k.A.2 - k.B.2) / Real.sqrt ((k.A.1 - k.B.1)^2 + (k.A.2 - k.B.2)^2))) = 4/5)
  (angle_ADB : Real.cos (Real.arccos ((k.A.1 - k.D.1) * (k.B.1 - k.D.1) + 
                                      (k.A.2 - k.D.2) * (k.B.2 - k.D.2)) / 
                        (Real.sqrt ((k.A.1 - k.D.1)^2 + (k.A.2 - k.D.2)^2) * 
                         Real.sqrt ((k.B.1 - k.D.1)^2 + (k.B.2 - k.D.2)^2))) = -1/2) :
  (k.A.1 - k.C.1)^2 + (k.A.2 - k.C.2)^2 = 150 := by sorry

end diagonal_length_in_specific_kite_l3178_317805


namespace elevator_max_weight_elevator_problem_l3178_317810

/-- Calculates the maximum weight of the next person to enter an elevator without overloading it. -/
theorem elevator_max_weight (num_adults : ℕ) (num_children : ℕ) (avg_adult_weight : ℝ) 
  (avg_child_weight : ℝ) (original_capacity : ℝ) (capacity_increase : ℝ) : ℝ :=
  let total_adult_weight := num_adults * avg_adult_weight
  let total_child_weight := num_children * avg_child_weight
  let current_weight := total_adult_weight + total_child_weight
  let new_capacity := original_capacity * (1 + capacity_increase)
  new_capacity - current_weight

/-- Proves that the maximum weight of the next person to enter the elevator is 250 pounds. -/
theorem elevator_problem : 
  elevator_max_weight 7 5 150 70 1500 0.1 = 250 := by
  sorry

end elevator_max_weight_elevator_problem_l3178_317810


namespace second_car_departure_time_l3178_317850

/-- Proves that the second car left 45 minutes after the first car --/
theorem second_car_departure_time (first_car_speed : ℝ) (trip_distance : ℝ) 
  (second_car_speed : ℝ) (time_difference : ℝ) : 
  first_car_speed = 30 →
  trip_distance = 80 →
  second_car_speed = 60 →
  time_difference = 1.5 →
  (time_difference - (first_car_speed * time_difference / second_car_speed)) * 60 = 45 := by
  sorry

#check second_car_departure_time

end second_car_departure_time_l3178_317850


namespace cos_sum_17th_roots_l3178_317883

theorem cos_sum_17th_roots : 
  Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end cos_sum_17th_roots_l3178_317883


namespace custom_mul_theorem_l3178_317868

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 2 * a - b^3

/-- Theorem stating that if a * 3 = 15 under the custom multiplication, then a = 21 -/
theorem custom_mul_theorem (a : ℝ) (h : custom_mul a 3 = 15) : a = 21 := by
  sorry

end custom_mul_theorem_l3178_317868


namespace lesser_fraction_l3178_317812

theorem lesser_fraction (x y : ℚ) (sum_eq : x + y = 5/6) (prod_eq : x * y = 1/8) :
  min x y = 1/6 := by
  sorry

end lesser_fraction_l3178_317812


namespace rectangle_area_with_inscribed_circle_l3178_317820

/-- Given a right triangle XYZ with coordinates X(0,0), Y(a,0), Z(a,a),
    where 'a' is a positive real number, and a circle with radius 'a'
    inscribed in the rectangle formed by extending sides XY and YZ,
    prove that the area of the rectangle is 4a² when the hypotenuse XZ = 2a. -/
theorem rectangle_area_with_inscribed_circle (a : ℝ) (ha : a > 0) :
  let X : ℝ × ℝ := (0, 0)
  let Y : ℝ × ℝ := (a, 0)
  let Z : ℝ × ℝ := (a, a)
  let hypotenuse := Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2)
  hypotenuse = 2 * a →
  (2 * a) * (2 * a) = 4 * a^2 := by
sorry

end rectangle_area_with_inscribed_circle_l3178_317820


namespace cubic_polynomial_property_l3178_317899

/-- A cubic polynomial with integer coefficients -/
def cubic_polynomial (a b : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 9*a

/-- Predicate for a cubic polynomial having two coincident roots -/
def has_coincident_roots (a b : ℤ) : Prop :=
  ∃ r s : ℤ, r ≠ s ∧ 
    ∀ x : ℝ, cubic_polynomial a b x = (x - r)^2 * (x - s)

/-- Theorem stating that under given conditions, |ab| = 1344 -/
theorem cubic_polynomial_property (a b : ℤ) :
  a ≠ 0 → b ≠ 0 → has_coincident_roots a b → |a*b| = 1344 := by
  sorry

end cubic_polynomial_property_l3178_317899


namespace revolver_game_probability_l3178_317884

/-- Represents a six-shot revolver with one bullet -/
def Revolver : Type := Unit

/-- Represents a player in the game -/
inductive Player : Type
| A : Player
| B : Player

/-- The probability of firing the bullet on a single shot -/
def singleShotProbability : ℚ := 1 / 6

/-- The probability of not firing the bullet on a single shot -/
def singleShotMissProbability : ℚ := 1 - singleShotProbability

/-- The starting player of the game -/
def startingPlayer : Player := Player.A

/-- The probability that the gun will fire while player A is holding it -/
noncomputable def probabilityAFires : ℚ := 6 / 11

/-- Theorem stating that the probability of A firing the gun is 6/11 -/
theorem revolver_game_probability :
  probabilityAFires = 6 / 11 :=
sorry

end revolver_game_probability_l3178_317884


namespace basketball_game_score_l3178_317851

theorem basketball_game_score (a b k d : ℕ) : 
  a = b →  -- Tied at the end of first quarter
  (4*a + 14*k = 4*b + 6*d + 2) →  -- Eagles won by two points
  (4*a + 14*k ≤ 100) →  -- Eagles scored no more than 100
  (4*b + 6*d ≤ 100) →  -- Panthers scored no more than 100
  (2*a + k) + (2*b + d) = 59 := by
sorry

end basketball_game_score_l3178_317851


namespace PQ_length_range_l3178_317833

/-- The circle C in the Cartesian coordinate system -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 3)^2 = 2}

/-- A point on the x-axis -/
def A : ℝ → ℝ × ℝ := λ x => (x, 0)

/-- The tangent points P and Q on the circle C -/
noncomputable def P (x : ℝ) : ℝ × ℝ := sorry
noncomputable def Q (x : ℝ) : ℝ × ℝ := sorry

/-- The length of segment PQ -/
noncomputable def PQ_length (x : ℝ) : ℝ :=
  Real.sqrt ((P x).1 - (Q x).1)^2 + ((P x).2 - (Q x).2)^2

/-- The theorem stating the range of PQ length -/
theorem PQ_length_range :
  ∀ x : ℝ, 2 * Real.sqrt 14 / 3 < PQ_length x ∧ PQ_length x < 2 * Real.sqrt 2 := by
  sorry

end PQ_length_range_l3178_317833


namespace point_position_l3178_317807

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space with equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Determines if a point is on the upper right side of a line -/
def isUpperRight (l : Line) (p : Point) : Prop :=
  l.A * p.x + l.B * p.y + l.C < 0 ∧ l.A > 0 ∧ l.B < 0

theorem point_position (l : Line) (p : Point) :
  isUpperRight l p → p.y > (-l.A * p.x - l.C) / l.B :=
by sorry

end point_position_l3178_317807


namespace marble_jar_ratio_l3178_317829

/-- 
Given a collection of marbles distributed in three jars, where:
- The first jar contains 80 marbles
- The second jar contains twice the amount of the first jar
- The total number of marbles is 260

This theorem proves that the ratio of marbles in the third jar to the first jar is 1/4.
-/
theorem marble_jar_ratio : 
  ∀ (jar1 jar2 jar3 : ℕ),
  jar1 = 80 →
  jar2 = 2 * jar1 →
  jar1 + jar2 + jar3 = 260 →
  (jar3 : ℚ) / jar1 = 1 / 4 := by
sorry

end marble_jar_ratio_l3178_317829


namespace good_carrots_count_l3178_317874

theorem good_carrots_count (carol_carrots : ℕ) (mom_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : mom_carrots = 16)
  (h3 : bad_carrots = 7) :
  carol_carrots + mom_carrots - bad_carrots = 38 :=
by
  sorry

end good_carrots_count_l3178_317874


namespace parallel_condition_distance_condition_l3178_317836

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given point P with coordinates (a+2, 2a-8) -/
def P (a : ℝ) : Point := ⟨a + 2, 2 * a - 8⟩

/-- Point Q with fixed coordinates (1, -2) -/
def Q : Point := ⟨1, -2⟩

/-- Condition 1: Line PQ is parallel to x-axis -/
def parallel_to_x_axis (P Q : Point) : Prop := P.y = Q.y

/-- Condition 2: Distance from P to y-axis is 4 -/
def distance_to_y_axis (P : Point) : ℝ := |P.x|

/-- Theorem for Condition 1 -/
theorem parallel_condition (a : ℝ) : 
  parallel_to_x_axis (P a) Q → P a = ⟨5, -2⟩ := by sorry

/-- Theorem for Condition 2 -/
theorem distance_condition (a : ℝ) : 
  distance_to_y_axis (P a) = 4 → (P a = ⟨4, -4⟩ ∨ P a = ⟨-4, -20⟩) := by sorry

end parallel_condition_distance_condition_l3178_317836


namespace area_segment_proportions_l3178_317864

/-- Given areas and segments, prove proportional relationships -/
theorem area_segment_proportions 
  (S S'' S' : ℝ) 
  (a a' : ℝ) 
  (h : S / S'' = a / a') 
  (h_pos : S > 0 ∧ S'' > 0 ∧ S' > 0 ∧ a > 0 ∧ a' > 0) :
  (S / a = S' / a') ∧ (S * a' = S' * a) := by
  sorry

end area_segment_proportions_l3178_317864


namespace arc_measures_l3178_317843

-- Define the circle and angles
def Circle : Type := ℝ × ℝ
def CentralAngle (c : Circle) : ℝ := 60
def InscribedAngle (c : Circle) : ℝ := 30

-- Define the theorem
theorem arc_measures (c : Circle) :
  (2 * CentralAngle c = 120) ∧ (2 * InscribedAngle c = 60) :=
by sorry

end arc_measures_l3178_317843


namespace sqrt_five_squared_times_four_to_sixth_l3178_317898

theorem sqrt_five_squared_times_four_to_sixth (x : ℝ) : x = Real.sqrt (5^2 * 4^6) → x = 320 := by
  sorry

end sqrt_five_squared_times_four_to_sixth_l3178_317898


namespace donation_problem_l3178_317891

theorem donation_problem (day1_amount day2_amount : ℕ) 
  (day2_extra_donors : ℕ) (h1 : day1_amount = 4800) 
  (h2 : day2_amount = 6000) (h3 : day2_extra_donors = 50) : 
  ∃ (day1_donors : ℕ), 
    (day1_donors > 0 ∧ day1_donors + day2_extra_donors > 0) ∧
    (day1_amount : ℚ) / day1_donors = (day2_amount : ℚ) / (day1_donors + day2_extra_donors) ∧
    day1_donors + (day1_donors + day2_extra_donors) = 450 ∧
    (day1_amount : ℚ) / day1_donors = 24 :=
by
  sorry

#check donation_problem

end donation_problem_l3178_317891


namespace remainder_equality_l3178_317890

theorem remainder_equality (a b d s t : ℕ) 
  (h1 : a > b) 
  (h2 : a % d = s % d) 
  (h3 : b % d = t % d) : 
  ((a + 1) * (b + 1)) % d = ((s + 1) * (t + 1)) % d := by
  sorry

end remainder_equality_l3178_317890


namespace solve_seed_problem_l3178_317879

def seed_problem (total_seeds : ℕ) (left_seeds : ℕ) (right_multiplier : ℕ) (seeds_left : ℕ) : Prop :=
  let right_seeds := right_multiplier * left_seeds
  let initially_thrown := left_seeds + right_seeds
  let joined_later := total_seeds - initially_thrown - seeds_left
  joined_later = total_seeds - (left_seeds + right_multiplier * left_seeds) - seeds_left

theorem solve_seed_problem :
  seed_problem 120 20 2 30 := by
  sorry

end solve_seed_problem_l3178_317879


namespace no_primes_in_range_l3178_317872

theorem no_primes_in_range (n : ℕ) (hn : n > 2) :
  ∀ p, Prime p → ¬(n.factorial + 2 < p ∧ p < n.factorial + 2*n) :=
sorry

end no_primes_in_range_l3178_317872


namespace inequality_proof_l3178_317881

theorem inequality_proof (a b c : ℝ) (h : a ≠ b) :
  Real.sqrt ((a - c)^2 + b^2) + Real.sqrt (a^2 + (b - c)^2) > Real.sqrt 2 * abs (a - b) := by
  sorry

end inequality_proof_l3178_317881


namespace oxford_high_total_people_l3178_317837

-- Define the school structure
structure School where
  teachers : Nat
  principal : Nat
  vice_principals : Nat
  other_staff : Nat
  classes : Nat
  avg_students_per_class : Nat

-- Define Oxford High School
def oxford_high : School :=
  { teachers := 75,
    principal := 1,
    vice_principals := 3,
    other_staff := 20,
    classes := 35,
    avg_students_per_class := 23 }

-- Define the function to calculate total people
def total_people (s : School) : Nat :=
  s.teachers + s.principal + s.vice_principals + s.other_staff +
  (s.classes * s.avg_students_per_class)

-- Theorem statement
theorem oxford_high_total_people :
  total_people oxford_high = 904 := by
  sorry

end oxford_high_total_people_l3178_317837


namespace oliver_money_problem_l3178_317830

theorem oliver_money_problem (X : ℤ) :
  X + 5 - 4 - 3 + 8 = 15 → X = 13 := by
  sorry

end oliver_money_problem_l3178_317830


namespace count_integers_satisfying_inequality_l3178_317841

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset Int),
    (∀ n ∈ S, -15 ≤ n ∧ n ≤ 7 ∧ (n - 4) * (n + 2) * (n + 6) < -1) ∧
    (∀ n : Int, -15 ≤ n ∧ n ≤ 7 ∧ (n - 4) * (n + 2) * (n + 6) < -1 → n ∈ S) ∧
    Finset.card S = 12 :=
by sorry

end count_integers_satisfying_inequality_l3178_317841


namespace total_bulbs_needed_l3178_317886

def ceiling_lights (medium_count : ℕ) : ℕ × ℕ × ℕ := 
  let large_count := 2 * medium_count
  let small_count := medium_count + 10
  (small_count, medium_count, large_count)

def bulb_count (lights : ℕ × ℕ × ℕ) : ℕ :=
  let (small, medium, large) := lights
  small * 1 + medium * 2 + large * 3

theorem total_bulbs_needed : 
  bulb_count (ceiling_lights 12) = 118 := by
  sorry

end total_bulbs_needed_l3178_317886


namespace square_area_ratio_l3178_317827

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 16 * b) : a^2 = 16 * b^2 := by
  sorry

end square_area_ratio_l3178_317827


namespace discount_profit_equivalence_l3178_317831

theorem discount_profit_equivalence (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ)
  (h1 : discount_rate = 0.04)
  (h2 : profit_rate = 0.38) :
  let selling_price := cost_price * (1 + profit_rate)
  let discounted_price := selling_price * (1 - discount_rate)
  let profit_with_discount := discounted_price - cost_price
  let profit_without_discount := selling_price - cost_price
  profit_without_discount / cost_price = profit_rate :=
by
  sorry

end discount_profit_equivalence_l3178_317831


namespace years_since_stopped_babysitting_l3178_317800

/-- Represents the age when Jane started babysitting -/
def start_age : ℕ := 18

/-- Represents Jane's current age -/
def current_age : ℕ := 32

/-- Represents the current age of the oldest person Jane could have babysat -/
def oldest_babysat_current_age : ℕ := 23

/-- Represents the maximum age ratio between Jane and the children she babysat -/
def max_age_ratio : ℚ := 1/2

/-- Theorem stating that Jane stopped babysitting 14 years ago -/
theorem years_since_stopped_babysitting :
  current_age - (oldest_babysat_current_age - (start_age * max_age_ratio).floor) = 14 := by
  sorry

end years_since_stopped_babysitting_l3178_317800


namespace temperature_calculation_l3178_317835

/-- Given the average temperatures for two sets of four consecutive days and the temperature of the first day, calculate the temperature of the last day. -/
theorem temperature_calculation (temp_mon tues wed thurs fri : ℝ) :
  (temp_mon + tues + wed + thurs) / 4 = 48 →
  (tues + wed + thurs + fri) / 4 = 46 →
  temp_mon = 39 →
  fri = 31 := by
  sorry

end temperature_calculation_l3178_317835


namespace no_consecutive_power_l3178_317878

theorem no_consecutive_power (n : ℕ) : ¬ ∃ (m k : ℕ), k ≥ 2 ∧ n * (n + 1) = m ^ k := by
  sorry

end no_consecutive_power_l3178_317878


namespace alien_energy_cells_l3178_317852

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- Theorem stating that 321 in base 7 is equal to 162 in base 10 --/
theorem alien_energy_cells : base7ToBase10 3 2 1 = 162 := by
  sorry

end alien_energy_cells_l3178_317852


namespace speech_contest_allocation_l3178_317808

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes,
    with each box containing at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 items from a set of 6 items. -/
def choose_two_from_six : ℕ := sorry

theorem speech_contest_allocation :
  distribute 8 6 = choose_two_from_six + 6 := by sorry

end speech_contest_allocation_l3178_317808


namespace carpet_cost_specific_carpet_cost_l3178_317866

/-- The total cost of carpet squares needed to cover a rectangular floor and an irregular section -/
theorem carpet_cost (rectangular_length : ℝ) (rectangular_width : ℝ) (irregular_area : ℝ)
  (carpet_side : ℝ) (carpet_cost : ℝ) : ℝ :=
  let rectangular_area := rectangular_length * rectangular_width
  let carpet_area := carpet_side * carpet_side
  let rectangular_squares := rectangular_area / carpet_area
  let irregular_squares := irregular_area / carpet_area
  let total_squares := rectangular_squares + irregular_squares + 1 -- Adding 1 for potential waste
  total_squares * carpet_cost

/-- The specific problem statement -/
theorem specific_carpet_cost : carpet_cost 24 64 128 8 24 = 648 := by
  sorry

end carpet_cost_specific_carpet_cost_l3178_317866


namespace equation_solution_l3178_317818

theorem equation_solution : ∃! x : ℝ, (1 / (x - 1) = 3 / (x - 3)) ∧ x = 0 := by
  sorry

end equation_solution_l3178_317818


namespace sum_inequality_l3178_317840

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  c / a + a / (b + c) + b / c ≥ 2 := by
  sorry

end sum_inequality_l3178_317840


namespace range_of_m_for_single_valued_function_l3178_317802

/-- A function is single-valued on an interval if there exists a unique x in the interval
    that satisfies (b-a) * f'(x) = f(b) - f(a) --/
def SingleValued (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ (b - a) * (deriv f x) = f b - f a

/-- The function f(x) = x^3 - x^2 + m --/
def f (m : ℝ) : ℝ → ℝ := fun x ↦ x^3 - x^2 + m

theorem range_of_m_for_single_valued_function (a : ℝ) (h_a : a ≥ 1) :
  ∀ m : ℝ, SingleValued (f m) 0 a ∧ 
  (∃ x y, 0 ≤ x ∧ x < y ∧ y ≤ a ∧ f m x = 0 ∧ f m y = 0) ∧ 
  (∀ z, 0 ≤ z ∧ z ≤ a ∧ f m z = 0 → z = x ∨ z = y) →
  -1 ≤ m ∧ m < 4/27 :=
sorry

end range_of_m_for_single_valued_function_l3178_317802


namespace four_number_sequence_l3178_317826

theorem four_number_sequence (a b c d : ℝ) : 
  (b - a = c - b) →  -- arithmetic sequence condition
  (c * c = b * d) →  -- geometric sequence condition
  (a + d = 16) → 
  (b + c = 12) → 
  ((a = 15 ∧ b = 9 ∧ c = 3 ∧ d = 1) ∨ (a = 0 ∧ b = 4 ∧ c = 8 ∧ d = 16)) :=
by sorry

end four_number_sequence_l3178_317826


namespace parabola_equation_from_axis_and_focus_l3178_317877

/-- A parabola with given axis of symmetry and focus -/
structure Parabola where
  axis_of_symmetry : ℝ
  focus : ℝ × ℝ

/-- The equation of a parabola given its parameters -/
def parabola_equation (p : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y^2 = -4 * x

/-- Theorem: For a parabola with axis of symmetry x = 1 and focus at (-1, 0), its equation is y² = -4x -/
theorem parabola_equation_from_axis_and_focus :
  ∀ (p : Parabola), p.axis_of_symmetry = 1 ∧ p.focus = (-1, 0) →
  parabola_equation p = fun x y => y^2 = -4 * x :=
by sorry

end parabola_equation_from_axis_and_focus_l3178_317877


namespace age_problem_l3178_317869

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 42 → 
  b = 16 := by
sorry

end age_problem_l3178_317869


namespace simplify_expression_1_simplify_expression_2_l3178_317858

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  (1 : ℝ) * x^2 + 5*y - 4*x^2 - 3*y = -3*x^2 + 2*y := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b := by sorry

end simplify_expression_1_simplify_expression_2_l3178_317858


namespace solution_set_x_squared_gt_x_l3178_317846

theorem solution_set_x_squared_gt_x : 
  {x : ℝ | x^2 > x} = {x : ℝ | x < 0 ∨ x > 1} := by sorry

end solution_set_x_squared_gt_x_l3178_317846


namespace snack_machine_purchase_l3178_317861

/-- The number of pieces of chocolate bought -/
def chocolate_pieces : ℕ := 2

/-- The cost of a candy bar in cents -/
def candy_bar_cost : ℕ := 25

/-- The cost of a piece of chocolate in cents -/
def chocolate_cost : ℕ := 75

/-- The cost of a pack of juice in cents -/
def juice_cost : ℕ := 50

/-- The total number of quarters used -/
def total_quarters : ℕ := 11

theorem snack_machine_purchase :
  chocolate_pieces * chocolate_cost + 3 * candy_bar_cost + juice_cost = total_quarters * 25 :=
by sorry

end snack_machine_purchase_l3178_317861


namespace polynomial_evaluation_l3178_317896

theorem polynomial_evaluation : 
  let p : ℝ → ℝ := λ x => 2*x^4 + 3*x^3 - x^2 + 5*x - 2
  p 2 = 60 := by
  sorry

end polynomial_evaluation_l3178_317896


namespace smallest_four_digit_mod_11_l3178_317857

theorem smallest_four_digit_mod_11 :
  ∀ n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧ n % 11 = 2 → n ≥ 1003 :=
by sorry

end smallest_four_digit_mod_11_l3178_317857


namespace candy_count_l3178_317803

theorem candy_count : ∃ n : ℕ, n % 3 = 2 ∧ n % 4 = 3 ∧ 32 ≤ n ∧ n ≤ 35 ∧ n = 35 := by
  sorry

end candy_count_l3178_317803


namespace inequality_solution_set_l3178_317825

theorem inequality_solution_set (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 2) :
  (2 * x) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ↔ x ∈ Set.Ioc 0 (1/5) ∪ Set.Ioc 2 6 := by
  sorry

end inequality_solution_set_l3178_317825


namespace girls_in_class_l3178_317848

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (ratio_nonbinary : ℕ) 
  (h1 : ratio_girls = 3)
  (h2 : ratio_boys = 2)
  (h3 : ratio_nonbinary = 1)
  (h4 : total = 72) :
  (total * ratio_girls) / (ratio_girls + ratio_boys + ratio_nonbinary) = 36 := by
sorry

end girls_in_class_l3178_317848


namespace twenty_triangles_l3178_317887

/-- Represents a rectangle divided into smaller rectangles with diagonal and vertical lines -/
structure DividedRectangle where
  smallRectangles : Nat
  diagonalsPerSmallRectangle : Nat
  verticalLinesPerSmallRectangle : Nat

/-- Counts the total number of triangles in the divided rectangle -/
def countTriangles (r : DividedRectangle) : Nat :=
  sorry

/-- Theorem stating that the specific configuration results in 20 triangles -/
theorem twenty_triangles :
  let r : DividedRectangle := {
    smallRectangles := 4,
    diagonalsPerSmallRectangle := 1,
    verticalLinesPerSmallRectangle := 1
  }
  countTriangles r = 20 := by
  sorry

end twenty_triangles_l3178_317887


namespace diana_tue_thu_hours_l3178_317853

/-- Represents Diana's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  weekly_earnings : ℕ
  hourly_rate : ℕ

/-- Calculates the number of hours Diana works on Tuesday and Thursday --/
def hours_tue_thu (schedule : WorkSchedule) : ℕ :=
  schedule.weekly_earnings / schedule.hourly_rate - 3 * schedule.hours_mon_wed_fri

/-- Theorem stating that Diana works 30 hours on Tuesday and Thursday --/
theorem diana_tue_thu_hours (schedule : WorkSchedule) 
  (h1 : schedule.hours_mon_wed_fri = 10)
  (h2 : schedule.weekly_earnings = 1800)
  (h3 : schedule.hourly_rate = 30) :
  hours_tue_thu schedule = 30 := by
  sorry

#eval hours_tue_thu { hours_mon_wed_fri := 10, weekly_earnings := 1800, hourly_rate := 30 }

end diana_tue_thu_hours_l3178_317853


namespace randy_third_quiz_score_l3178_317873

theorem randy_third_quiz_score 
  (first_quiz : ℕ) 
  (second_quiz : ℕ) 
  (fifth_quiz : ℕ) 
  (desired_average : ℕ) 
  (total_quizzes : ℕ) 
  (third_fourth_sum : ℕ) :
  first_quiz = 90 →
  second_quiz = 98 →
  fifth_quiz = 96 →
  desired_average = 94 →
  total_quizzes = 5 →
  third_fourth_sum = 186 →
  ∃ (fourth_quiz : ℕ), 
    (first_quiz + second_quiz + 94 + fourth_quiz + fifth_quiz) / total_quizzes = desired_average :=
by
  sorry


end randy_third_quiz_score_l3178_317873


namespace color_film_fraction_l3178_317813

theorem color_film_fraction (x y : ℝ) (x_pos : x > 0) (y_pos : y > 0) : 
  let total_bw := 20 * x
  let total_color := 6 * y
  let selected_bw := y / (5 * x) * total_bw
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  selected_color / total_selected = 30 / 31 := by
sorry


end color_film_fraction_l3178_317813


namespace expression_value_l3178_317867

theorem expression_value (x : ℝ) (h : Real.tan (Real.pi - x) = -2) :
  4 * Real.sin x ^ 2 - 3 * Real.sin x * Real.cos x - 5 * Real.cos x ^ 2 = 1 := by
  sorry

end expression_value_l3178_317867


namespace equation_solution_l3178_317875

theorem equation_solution : ∃ x : ℚ, (x - 7) / 2 - (1 + x) / 3 = 1 ∧ x = 29 := by
  sorry

end equation_solution_l3178_317875


namespace probability_of_black_ball_l3178_317849

theorem probability_of_black_ball (p_red p_white p_black : ℝ) :
  p_red = 0.41 →
  p_white = 0.27 →
  p_red + p_white + p_black = 1 →
  p_black = 0.32 := by
sorry

end probability_of_black_ball_l3178_317849


namespace unique_matrix_transformation_l3178_317860

theorem unique_matrix_transformation (A : Matrix (Fin 2) (Fin 2) ℝ) :
  ∃! M : Matrix (Fin 2) (Fin 2) ℝ,
    (∀ i j, (M * A) i j = if j = 1 then (if i = 0 then 2 * A i j else 3 * A i j) else A i j) ∧
    M = ![![1, 0], ![0, 3]] := by
  sorry

end unique_matrix_transformation_l3178_317860


namespace permutation_problem_arrangement_problem_photo_arrangement_problem_l3178_317859

-- Problem 1
theorem permutation_problem (m : ℕ) : 
  (Nat.factorial 10) / (Nat.factorial (10 - m)) = (Nat.factorial 10) / (Nat.factorial 4) → m = 6 := by
sorry

-- Problem 2
theorem arrangement_problem : 
  (Nat.factorial 3) = 6 := by
sorry

-- Problem 3
theorem photo_arrangement_problem : 
  2 * 4 * (Nat.factorial 4) = 192 := by
sorry

end permutation_problem_arrangement_problem_photo_arrangement_problem_l3178_317859


namespace museum_ticket_fraction_l3178_317806

theorem museum_ticket_fraction (total : ℚ) (sandwich_fraction : ℚ) (book_fraction : ℚ) (leftover : ℚ) :
  total = 150 →
  sandwich_fraction = 1/5 →
  book_fraction = 1/2 →
  leftover = 20 →
  (total - (sandwich_fraction * total + book_fraction * total + leftover)) / total = 1/6 := by
  sorry

end museum_ticket_fraction_l3178_317806


namespace b_eq_one_sufficient_not_necessary_l3178_317870

/-- The condition for the line and curve to have common points -/
def has_common_points (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ x^2 + (y - 1)^2 = 1

/-- The statement that b = 1 is sufficient but not necessary for common points -/
theorem b_eq_one_sufficient_not_necessary :
  (∀ k : ℝ, has_common_points k 1) ∧
  (∃ k b : ℝ, b ≠ 1 ∧ has_common_points k b) :=
sorry

end b_eq_one_sufficient_not_necessary_l3178_317870


namespace polynomial_identity_l3178_317845

theorem polynomial_identity : ∀ x : ℝ, 
  (x^2 + 3*x + 2) * (x + 3) = (x + 1) * (x^2 + 5*x + 6) := by
  sorry

end polynomial_identity_l3178_317845


namespace image_of_A_under_f_l3178_317894

def A : Set Int := {-1, 3, 5}

def f (x : Int) : Int := 2 * x - 1

theorem image_of_A_under_f :
  (Set.image f A) = {-3, 5, 9} := by
  sorry

end image_of_A_under_f_l3178_317894


namespace converse_xy_zero_x_zero_is_true_l3178_317816

theorem converse_xy_zero_x_zero_is_true :
  ∀ (x y : ℝ), x = 0 → x * y = 0 :=
by sorry

end converse_xy_zero_x_zero_is_true_l3178_317816


namespace range_of_a_l3178_317828

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + a| < 3 ↔ 2 < x ∧ x < 3) → 
  -5 ≤ a ∧ a ≤ 0 := by
sorry

end range_of_a_l3178_317828


namespace floor_sqrt_50_squared_plus_2_l3178_317823

theorem floor_sqrt_50_squared_plus_2 : ⌊Real.sqrt 50⌋^2 + 2 = 51 := by
  sorry

end floor_sqrt_50_squared_plus_2_l3178_317823


namespace point_M_coordinates_l3178_317817

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 4 * x

theorem point_M_coordinates :
  ∃ (x y : ℝ), f' x = -4 ∧ f x = y ∧ x = -1 ∧ y = 3 :=
by
  sorry

#check point_M_coordinates

end point_M_coordinates_l3178_317817


namespace height_difference_l3178_317871

/-- Prove that the difference between 3 times Kim's height and Tamara's height is 4 inches -/
theorem height_difference (kim_height tamara_height : ℕ) : 
  tamara_height + kim_height = 92 →
  tamara_height = 68 →
  ∃ x, tamara_height = 3 * kim_height - x →
  3 * kim_height - tamara_height = 4 := by
  sorry

end height_difference_l3178_317871


namespace german_team_goals_l3178_317885

def journalist1_correct (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2_correct (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3_correct (x : ℕ) : Prop := x % 2 = 1

def exactly_two_correct (x : ℕ) : Prop :=
  (journalist1_correct x ∧ journalist2_correct x ∧ ¬journalist3_correct x) ∨
  (journalist1_correct x ∧ ¬journalist2_correct x ∧ journalist3_correct x) ∨
  (¬journalist1_correct x ∧ journalist2_correct x ∧ journalist3_correct x)

theorem german_team_goals :
  {x : ℕ | exactly_two_correct x} = {11, 12, 14, 16, 17} := by sorry

end german_team_goals_l3178_317885


namespace factor_x4_minus_16_l3178_317888

theorem factor_x4_minus_16 (x : ℂ) : x^4 - 16 = (x - 2) * (x + 2) * (x - 2*I) * (x + 2*I) := by
  sorry

end factor_x4_minus_16_l3178_317888


namespace waynes_blocks_l3178_317882

/-- Wayne's block collection problem -/
theorem waynes_blocks (initial_blocks final_blocks father_blocks : ℕ) 
  (h1 : father_blocks = 6)
  (h2 : final_blocks = 15)
  (h3 : final_blocks = initial_blocks + father_blocks) : 
  initial_blocks = 9 := by
  sorry

end waynes_blocks_l3178_317882


namespace max_crosses_4x10_impossible_5x10_l3178_317809

/-- Represents a rectangular table with crosses placed in its cells. -/
structure CrossTable (m n : ℕ) :=
  (crosses : Fin m → Fin n → Bool)

/-- Checks if a row has an odd number of crosses. -/
def hasOddCrossesInRow (t : CrossTable m n) (row : Fin m) : Prop :=
  Odd (Finset.sum (Finset.univ : Finset (Fin n)) (λ col => if t.crosses row col then 1 else 0))

/-- Checks if a column has an odd number of crosses. -/
def hasOddCrossesInColumn (t : CrossTable m n) (col : Fin n) : Prop :=
  Odd (Finset.sum (Finset.univ : Finset (Fin m)) (λ row => if t.crosses row col then 1 else 0))

/-- Checks if all rows and columns have an odd number of crosses. -/
def hasOddCrossesEverywhere (t : CrossTable m n) : Prop :=
  (∀ row, hasOddCrossesInRow t row) ∧ (∀ col, hasOddCrossesInColumn t col)

/-- Counts the total number of crosses in the table. -/
def totalCrosses (t : CrossTable m n) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin m)) (λ row =>
    Finset.sum (Finset.univ : Finset (Fin n)) (λ col =>
      if t.crosses row col then 1 else 0))

/-- Theorem for the 4x10 table. -/
theorem max_crosses_4x10 :
  ∀ t : CrossTable 4 10, hasOddCrossesEverywhere t → totalCrosses t ≤ 30 :=
sorry

/-- Theorem for the 5x10 table. -/
theorem impossible_5x10 :
  ¬∃ t : CrossTable 5 10, hasOddCrossesEverywhere t :=
sorry

end max_crosses_4x10_impossible_5x10_l3178_317809


namespace largest_integer_less_than_100_remainder_5_mod_8_l3178_317822

theorem largest_integer_less_than_100_remainder_5_mod_8 : 
  ∀ n : ℕ, n < 100 ∧ n % 8 = 5 → n ≤ 93 :=
by
  sorry

end largest_integer_less_than_100_remainder_5_mod_8_l3178_317822


namespace circle_radius_l3178_317821

theorem circle_radius (x y : ℝ) : 
  (16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 68 = 0) → 
  (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = 1) :=
by sorry

end circle_radius_l3178_317821


namespace money_distribution_l3178_317854

theorem money_distribution (p q r s : ℕ) : 
  p + q + r + s = 10000 →
  r = 2 * p →
  r = 3 * q →
  s = p + q →
  p = 1875 ∧ q = 1250 ∧ r = 3750 ∧ s = 3125 :=
by sorry

end money_distribution_l3178_317854


namespace properties_of_one_minus_sqrt_two_l3178_317856

theorem properties_of_one_minus_sqrt_two :
  let x : ℝ := 1 - Real.sqrt 2
  (- x = Real.sqrt 2 - 1) ∧
  (|x| = Real.sqrt 2 - 1) ∧
  (x⁻¹ = -1 - Real.sqrt 2) := by
  sorry

end properties_of_one_minus_sqrt_two_l3178_317856


namespace completing_square_transformation_l3178_317801

theorem completing_square_transformation (x : ℝ) :
  x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 :=
by sorry

end completing_square_transformation_l3178_317801


namespace lcm_of_ratio_and_hcf_l3178_317862

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 7 / 13 → Nat.gcd a b = 15 → Nat.lcm a b = 91 := by
  sorry

end lcm_of_ratio_and_hcf_l3178_317862


namespace centroid_altitude_length_l3178_317847

/-- Triangle XYZ with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Foot of the altitude from a point to a line segment -/
def altitude_foot (point : ℝ × ℝ) (segment : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem centroid_altitude_length (t : Triangle) (h1 : t.a = 13) (h2 : t.b = 15) (h3 : t.c = 24) :
  let g := centroid t
  let yz := ((0, 0), (t.c, 0))  -- Assuming YZ is on the x-axis
  let q := altitude_foot g yz
  distance g q = 2.4 := by sorry

end centroid_altitude_length_l3178_317847


namespace tangent_slope_at_pi_over_four_l3178_317897

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.sin x + Real.cos x) - 1/2

theorem tangent_slope_at_pi_over_four :
  (deriv f) (π/4) = 1/2 := by sorry

end tangent_slope_at_pi_over_four_l3178_317897


namespace polynomial_remainder_l3178_317804

theorem polynomial_remainder (m : ℚ) : 
  (∃ (f g : ℚ → ℚ) (R : ℚ), 
    (∀ y : ℚ, y^2 + m*y + 2 = (y - 1) * f y + R) ∧
    (∀ y : ℚ, y^2 + m*y + 2 = (y + 1) * g y + R)) →
  m = 0 := by
sorry

end polynomial_remainder_l3178_317804


namespace min_force_to_submerge_cube_l3178_317832

/-- Minimum force required to submerge a cube -/
theorem min_force_to_submerge_cube (V : Real) (ρ_cube ρ_water g : Real) :
  V = 1e-5 →
  ρ_cube = 700 →
  ρ_water = 1000 →
  g = 10 →
  (ρ_water * V * g) - (ρ_cube * V * g) = 0.03 := by
  sorry

end min_force_to_submerge_cube_l3178_317832


namespace ball_placement_theorem_l3178_317838

/-- Represents the number of distinct balls -/
def num_balls : ℕ := 4

/-- Represents the number of distinct boxes -/
def num_boxes : ℕ := 4

/-- Calculates the number of ways to place all balls into boxes leaving exactly one box empty -/
def ways_one_empty : ℕ := sorry

/-- Calculates the number of ways to place all balls into boxes with exactly one box containing two balls -/
def ways_one_two_balls : ℕ := sorry

/-- Calculates the number of ways to place all balls into boxes leaving exactly two boxes empty -/
def ways_two_empty : ℕ := sorry

theorem ball_placement_theorem :
  ways_one_empty = 144 ∧
  ways_one_two_balls = 144 ∧
  ways_two_empty = 84 := by sorry

end ball_placement_theorem_l3178_317838


namespace total_spider_legs_l3178_317814

/-- The number of spiders in the room -/
def num_spiders : ℕ := 5

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs in the room is 40 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 40 := by
  sorry

end total_spider_legs_l3178_317814


namespace profit_share_ratio_l3178_317880

/-- The ratio of profit shares for two investors is proportional to their investments -/
theorem profit_share_ratio (p_investment q_investment : ℕ) :
  p_investment = 30000 →
  q_investment = 45000 →
  (p_investment : ℚ) / (p_investment + q_investment) = 2 / 5 ∧
  (q_investment : ℚ) / (p_investment + q_investment) = 3 / 5 := by
  sorry

end profit_share_ratio_l3178_317880


namespace correct_distribution_probability_l3178_317819

def num_guests : ℕ := 3
def num_roll_types : ℕ := 4
def total_rolls : ℕ := 12
def rolls_per_guest : ℕ := 4

def probability_correct_distribution : ℚ := 2 / 103950

theorem correct_distribution_probability :
  let total_ways := (total_rolls.choose rolls_per_guest) * 
                    ((total_rolls - rolls_per_guest).choose rolls_per_guest) *
                    ((total_rolls - 2*rolls_per_guest).choose rolls_per_guest)
  let correct_ways := (num_roll_types.factorial) * 
                      (2^num_roll_types) * 
                      (1^num_roll_types)
  (correct_ways : ℚ) / total_ways = probability_correct_distribution :=
sorry

end correct_distribution_probability_l3178_317819


namespace a_positive_iff_sum_geq_two_l3178_317865

theorem a_positive_iff_sum_geq_two (a : ℝ) : a > 0 ↔ a + 1/a ≥ 2 := by sorry

end a_positive_iff_sum_geq_two_l3178_317865


namespace inequality_equivalence_l3178_317824

theorem inequality_equivalence (x : Real) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (2 * Real.cos x ≤ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ∧
   Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ≤ Real.sqrt 2) ↔
  x ∈ Set.Icc (Real.pi / 4) (7 * Real.pi / 4) :=
by sorry

end inequality_equivalence_l3178_317824


namespace amoeba_population_day_10_l3178_317855

/-- The number of amoebas on day n, given an initial population of 3 and daily doubling. -/
def amoeba_population (n : ℕ) : ℕ := 3 * 2^n

/-- Theorem stating that after 10 days, the amoeba population is 3072. -/
theorem amoeba_population_day_10 : amoeba_population 10 = 3072 := by
  sorry

end amoeba_population_day_10_l3178_317855


namespace trig_values_for_special_angle_l3178_317895

/-- The intersection point of two lines -/
def intersection_point (l₁ l₂ : ℝ × ℝ → Prop) : ℝ × ℝ :=
  sorry

/-- The angle whose terminal side passes through a given point -/
def angle_from_point (p : ℝ × ℝ) : ℝ :=
  sorry

/-- The sine of an angle -/
def sine (α : ℝ) : ℝ :=
  sorry

/-- The cosine of an angle -/
def cosine (α : ℝ) : ℝ :=
  sorry

/-- The tangent of an angle -/
def tangent (α : ℝ) : ℝ :=
  sorry

theorem trig_values_for_special_angle :
  let l₁ : ℝ × ℝ → Prop := λ (x, y) ↦ x - y = 0
  let l₂ : ℝ × ℝ → Prop := λ (x, y) ↦ 2*x + y - 3 = 0
  let p := intersection_point l₁ l₂
  let α := angle_from_point p
  sine α = Real.sqrt 2 / 2 ∧ cosine α = Real.sqrt 2 / 2 ∧ tangent α = 1 := by
  sorry

end trig_values_for_special_angle_l3178_317895


namespace not_A_union_B_equiv_l3178_317889

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 3) ≥ 0}
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem not_A_union_B_equiv : (Aᶜ ∪ B) = {x : ℝ | x > -2} := by sorry

end not_A_union_B_equiv_l3178_317889
