import Mathlib

namespace rhino_weight_theorem_l2249_224905

/-- The weight of a full-grown white rhino in pounds -/
def full_grown_white_rhino_weight : ℝ := 5100

/-- The weight of a newborn white rhino in pounds -/
def newborn_white_rhino_weight : ℝ := 150

/-- The weight of a full-grown black rhino in pounds -/
def full_grown_black_rhino_weight : ℝ := 2000

/-- The weight of a newborn black rhino in pounds -/
def newborn_black_rhino_weight : ℝ := 100

/-- The conversion factor from pounds to kilograms -/
def pounds_to_kg : ℝ := 0.453592

/-- The number of full-grown white rhinos -/
def num_full_grown_white : ℕ := 6

/-- The number of newborn white rhinos -/
def num_newborn_white : ℕ := 3

/-- The number of full-grown black rhinos -/
def num_full_grown_black : ℕ := 7

/-- The number of newborn black rhinos -/
def num_newborn_black : ℕ := 4

/-- The total weight of all rhinos in kilograms -/
def total_weight_kg : ℝ :=
  ((num_full_grown_white : ℝ) * full_grown_white_rhino_weight +
   (num_newborn_white : ℝ) * newborn_white_rhino_weight +
   (num_full_grown_black : ℝ) * full_grown_black_rhino_weight +
   (num_newborn_black : ℝ) * newborn_black_rhino_weight) * pounds_to_kg

theorem rhino_weight_theorem :
  total_weight_kg = 20616.436 := by
  sorry

end rhino_weight_theorem_l2249_224905


namespace intersection_line_equation_l2249_224931

/-- Given two lines l₁ and l₂, if a line l intersects both l₁ and l₂ such that the midpoint
    of the segment cut off by l₁ and l₂ is at the origin, then l has the equation x + 6y = 0 -/
theorem intersection_line_equation (x y : ℝ) : 
  let l₁ := {(x, y) : ℝ × ℝ | 4*x + y + 6 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | 3*x - 5*y - 6 = 0}
  let midpoint := (0, 0)
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁, y₁) ∈ l₁ ∧ (x₂, y₂) ∈ l₂ ∧
    (x₁ + x₂) / 2 = midpoint.1 ∧ (y₁ + y₂) / 2 = midpoint.2 →
  x + 6*y = 0 := by
sorry


end intersection_line_equation_l2249_224931


namespace problem_statements_l2249_224979

noncomputable section

variable (k : ℝ)

def f (x : ℝ) : ℝ := Real.log x

def g (x : ℝ) : ℝ := x^2 + k*x

def a (x₁ x₂ : ℝ) : ℝ := (f x₁ - f x₂) / (x₁ - x₂)

def b (x₁ x₂ : ℝ) : ℝ := (g k x₁ - g k x₂) / (x₁ - x₂)

theorem problem_statements :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → a x₁ x₂ > 0) ∧
  (∃ k : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ b k x₁ x₂ ≤ 0) ∧
  (∀ k : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ b k x₁ x₂ / a x₁ x₂ = 2) ∧
  (∀ k : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ b k x₁ x₂ / a x₁ x₂ = -2) → k < -4) :=
by sorry

end

end problem_statements_l2249_224979


namespace expression_bounds_l2249_224918

theorem expression_bounds (x y z w : Real) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) : 
  let expr := Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
              Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2)
  2 * Real.sqrt 2 ≤ expr ∧ expr ≤ 4 ∧ 
  (∃ x y z w, (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) ∧ 
              expr = 2 * Real.sqrt 2) ∧
  (∃ x y z w, (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) ∧ 
              expr = 4) := by
  sorry

end expression_bounds_l2249_224918


namespace digit_sum_l2249_224927

theorem digit_sum (P Q R S T : Nat) : 
  (P < 10 ∧ Q < 10 ∧ R < 10 ∧ S < 10 ∧ T < 10) → 
  (4 * (P * 10000 + Q * 1000 + R * 100 + S * 10 + T) = 41024) → 
  (P + Q + R + S + T = 14) := by
sorry

end digit_sum_l2249_224927


namespace sock_ratio_is_7_19_l2249_224952

/-- Represents the ratio of black socks to blue socks -/
structure SockRatio where
  black : ℕ
  blue : ℕ

/-- Represents the order of socks -/
structure SockOrder where
  black : ℕ
  blue : ℕ
  price_ratio : ℚ
  bill_increase : ℚ

/-- Calculates the ratio of black socks to blue socks given a sock order -/
def calculate_sock_ratio (order : SockOrder) : SockRatio :=
  sorry

/-- The specific sock order from the problem -/
def tom_order : SockOrder :=
  { black := 5
  , blue := 0  -- Unknown, to be calculated
  , price_ratio := 3
  , bill_increase := 3/5 }

theorem sock_ratio_is_7_19 : 
  let ratio := calculate_sock_ratio tom_order
  ratio.black = 7 ∧ ratio.blue = 19 := by sorry

end sock_ratio_is_7_19_l2249_224952


namespace cubic_sum_of_roots_l2249_224951

theorem cubic_sum_of_roots (m n r s : ℝ) : 
  (r^2 - m*r - n = 0) → (s^2 - m*s - n = 0) → r^3 + s^3 = m^3 + 3*n*m :=
by sorry

end cubic_sum_of_roots_l2249_224951


namespace pet_store_birds_l2249_224965

theorem pet_store_birds (total_birds talking_birds : ℕ) 
  (h1 : total_birds = 77)
  (h2 : talking_birds = 64) :
  total_birds - talking_birds = 13 := by
  sorry

end pet_store_birds_l2249_224965


namespace line1_passes_through_points_line2_satisfies_conditions_l2249_224904

-- Define the points
def point1 : ℝ × ℝ := (2, 1)
def point2 : ℝ × ℝ := (0, -3)
def point3 : ℝ × ℝ := (0, 5)

-- Define the sum of intercepts
def sum_of_intercepts : ℝ := 2

-- Define the equations of the lines
def line1_equation (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line2_equation (x y : ℝ) : Prop := 5 * x - 3 * y + 15 = 0

-- Theorem for the first line
theorem line1_passes_through_points :
  line1_equation point1.1 point1.2 ∧ line1_equation point2.1 point2.2 :=
sorry

-- Theorem for the second line
theorem line2_satisfies_conditions :
  line2_equation point3.1 point3.2 ∧
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a + b = sum_of_intercepts ∧
  ∀ x y : ℝ, line2_equation x y ↔ x / a + y / b = 1) :=
sorry

end line1_passes_through_points_line2_satisfies_conditions_l2249_224904


namespace mike_score_is_99_percent_l2249_224988

/-- Represents the exam scores of four students -/
structure ExamScores where
  gibi : ℝ
  jigi : ℝ
  mike : ℝ
  lizzy : ℝ

/-- Theorem stating that Mike's score is 99% given the conditions -/
theorem mike_score_is_99_percent 
  (scores : ExamScores)
  (h_gibi : scores.gibi = 59)
  (h_jigi : scores.jigi = 55)
  (h_lizzy : scores.lizzy = 67)
  (h_max_score : ℝ := 700)
  (h_average : (scores.gibi + scores.jigi + scores.mike + scores.lizzy) / 4 * h_max_score / 100 = 490) :
  scores.mike = 99 := by
sorry

end mike_score_is_99_percent_l2249_224988


namespace largest_n_with_2020_sets_l2249_224949

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define a_n as the number of sets S of positive integers
-- such that the sum of F_k for k in S equals n
def a (n : ℕ) : ℕ := sorry

-- State the theorem
theorem largest_n_with_2020_sets :
  ∃ n : ℕ, a n = 2020 ∧ ∀ m : ℕ, m > n → a m ≠ 2020 ∧ n = fib 2022 - 1 := by
  sorry

end largest_n_with_2020_sets_l2249_224949


namespace cookie_process_time_l2249_224926

/-- Represents the cookie-making process with given times for each step -/
structure CookieProcess where
  total_time : ℕ
  baking_time : ℕ
  white_icing_time : ℕ
  chocolate_icing_time : ℕ

/-- Calculates the time to make dough and cool cookies -/
def dough_and_cooling_time (process : CookieProcess) : ℕ :=
  process.total_time - (process.baking_time + process.white_icing_time + process.chocolate_icing_time)

/-- Theorem stating that the time to make dough and cool cookies is 45 minutes -/
theorem cookie_process_time (process : CookieProcess) 
  (h1 : process.total_time = 120)
  (h2 : process.baking_time = 15)
  (h3 : process.white_icing_time = 30)
  (h4 : process.chocolate_icing_time = 30) : 
  dough_and_cooling_time process = 45 := by
  sorry

end cookie_process_time_l2249_224926


namespace calculation_proof_l2249_224966

theorem calculation_proof :
  ((-1/12 - 1/36 + 1/6) * (-36) = -2) ∧
  ((-99 - 11/12) * 24 = -2398) := by
sorry

end calculation_proof_l2249_224966


namespace decimal_representation_symmetry_l2249_224997

/-- The main period of the decimal representation of 1/p -/
def decimal_period (p : ℕ) : List ℕ :=
  sorry

/-- Count occurrences of a digit in a list -/
def count_occurrences (digit : ℕ) (l : List ℕ) : ℕ :=
  sorry

theorem decimal_representation_symmetry (p n : ℕ) (h1 : Nat.Prime p) (h2 : p ∣ 10^n + 1) :
  ∀ i ∈ Finset.range 10,
    count_occurrences i (decimal_period p) = count_occurrences (9 - i) (decimal_period p) :=
by sorry

end decimal_representation_symmetry_l2249_224997


namespace intersection_points_l2249_224961

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

def g (x : ℝ) : ℝ := -f x

def h (x : ℝ) : ℝ := f (-x)

theorem intersection_points :
  (∃ a b : ℝ, a ≠ b ∧ f a = g a ∧ f b = g b) ∧
  (∃! c : ℝ, f c = h c) ∧
  (∀ x y : ℝ, x ≠ y → (f x = g x ∧ f y = g y) → (f x = h x ∧ f y = h y) → False) :=
by sorry

end intersection_points_l2249_224961


namespace sequence_not_convergent_l2249_224957

theorem sequence_not_convergent (x : ℕ → ℝ) (a : ℝ) :
  (∀ ε > 0, ∀ k, ∃ n > k, |x n - a| ≥ ε) →
  ¬ (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - a| < ε) :=
by sorry

end sequence_not_convergent_l2249_224957


namespace mixtape_length_example_l2249_224971

/-- The length of a mixtape given the number of songs on each side and the length of each song. -/
def mixtape_length (side1_songs : ℕ) (side2_songs : ℕ) (song_length : ℕ) : ℕ :=
  (side1_songs + side2_songs) * song_length

/-- Theorem stating that a mixtape with 6 songs on the first side, 4 songs on the second side,
    and each song being 4 minutes long has a total length of 40 minutes. -/
theorem mixtape_length_example : mixtape_length 6 4 4 = 40 := by
  sorry

end mixtape_length_example_l2249_224971


namespace student_fail_marks_l2249_224912

theorem student_fail_marks (pass_percentage : ℝ) (max_score : ℕ) (obtained_score : ℕ) : 
  pass_percentage = 36 / 100 → 
  max_score = 400 → 
  obtained_score = 130 → 
  ⌈pass_percentage * max_score⌉ - obtained_score = 14 := by
  sorry

end student_fail_marks_l2249_224912


namespace calculation_proof_l2249_224922

theorem calculation_proof : (1.2 : ℝ)^3 - (0.9 : ℝ)^3 / (1.2 : ℝ)^2 + 1.08 + (0.9 : ℝ)^2 = 3.11175 := by
  sorry

end calculation_proof_l2249_224922


namespace roberts_ride_time_l2249_224924

/-- The time taken for Robert to ride along a semi-circular path on a highway segment -/
theorem roberts_ride_time 
  (highway_length : ℝ) 
  (highway_width : ℝ) 
  (speed : ℝ) 
  (miles_to_feet : ℝ) 
  (h1 : highway_length = 1) 
  (h2 : highway_width = 40) 
  (h3 : speed = 5) 
  (h4 : miles_to_feet = 5280) : 
  ∃ (time : ℝ), time = π / 10 := by
sorry

end roberts_ride_time_l2249_224924


namespace choir_composition_theorem_l2249_224925

/-- Represents the choir composition and ratio changes -/
structure ChoirComposition where
  b : ℝ  -- Initial number of blonde girls
  x : ℝ  -- Number of blonde girls added

/-- Theorem about the choir composition changes -/
theorem choir_composition_theorem (choir : ChoirComposition) :
  -- Initial ratio of blonde to black-haired girls is 3:5
  (choir.b) / ((5/3) * choir.b) = 3/5 →
  -- After adding x blonde girls, the ratio becomes 3:2
  (choir.b + choir.x) / ((5/3) * choir.b) = 3/2 →
  -- The final number of black-haired girls is (5/3)b
  (5/3) * choir.b = (5/3) * choir.b ∧
  -- The relationship between x and b is x = (3/2)b
  choir.x = (3/2) * choir.b :=
by sorry

end choir_composition_theorem_l2249_224925


namespace largest_angle_cosine_l2249_224998

theorem largest_angle_cosine (A B C : ℝ) (h1 : A = π/6) 
  (h2 : 2 * (B * C * Real.cos A) = 3 * (B^2 + C^2 - 2*B*C*Real.cos A)) :
  Real.cos (max A (max B C)) = -1/2 := by
  sorry

end largest_angle_cosine_l2249_224998


namespace at_least_two_same_number_l2249_224945

/-- The number of dice being rolled -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability of at least two dice showing the same number -/
def prob_same_number : ℝ := 1

theorem at_least_two_same_number :
  num_dice > num_sides → prob_same_number = 1 := by sorry

end at_least_two_same_number_l2249_224945


namespace ellipse_foci_y_axis_m_range_l2249_224987

/-- Represents an ellipse with the given equation -/
structure Ellipse (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (|m| - 1) + y^2 / (2 - m) = 1

/-- Indicates that the ellipse has foci on the y-axis -/
def has_foci_on_y_axis (e : Ellipse m) : Prop :=
  2 - m > |m| - 1 ∧ |m| - 1 > 0

/-- The range of m for which the ellipse has foci on the y-axis -/
def m_range (m : ℝ) : Prop :=
  m < -1 ∨ (1 < m ∧ m < 3/2)

/-- Theorem stating the range of m for an ellipse with foci on the y-axis -/
theorem ellipse_foci_y_axis_m_range (m : ℝ) :
  (∃ e : Ellipse m, has_foci_on_y_axis e) ↔ m_range m :=
sorry

end ellipse_foci_y_axis_m_range_l2249_224987


namespace least_integer_with_twelve_factors_l2249_224981

theorem least_integer_with_twelve_factors : 
  ∀ n : ℕ, n > 0 → (∃ f : Finset ℕ, f = {d : ℕ | d ∣ n ∧ d > 0} ∧ f.card = 12) → n ≥ 60 :=
sorry

end least_integer_with_twelve_factors_l2249_224981


namespace range_of_m_l2249_224900

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 12 * x - x^3 else -2 * x

theorem range_of_m (m : ℝ) :
  (∀ y ∈ Set.Iic m, f y ∈ Set.Ici (-16)) ∧
  (∀ z : ℝ, z ≥ -16 → ∃ x ∈ Set.Iic m, f x = z) →
  m ∈ Set.Icc (-2) 8 :=
sorry

end range_of_m_l2249_224900


namespace swimming_pool_volume_l2249_224960

theorem swimming_pool_volume :
  let shallow_width : ℝ := 9
  let shallow_length : ℝ := 12
  let shallow_depth : ℝ := 1
  let deep_width : ℝ := 15
  let deep_length : ℝ := 18
  let deep_depth : ℝ := 4
  let island_width : ℝ := 3
  let island_length : ℝ := 6
  let island_height : ℝ := 1
  let shallow_volume := shallow_width * shallow_length * shallow_depth
  let deep_volume := deep_width * deep_length * deep_depth
  let island_volume := island_width * island_length * island_height
  let total_volume := shallow_volume + deep_volume - island_volume
  total_volume = 1170 := by
sorry

end swimming_pool_volume_l2249_224960


namespace gcd_n_cubed_plus_16_and_n_plus_4_l2249_224959

theorem gcd_n_cubed_plus_16_and_n_plus_4 (n : ℕ) (h : n > 2^4) :
  Nat.gcd (n^3 + 4^2) (n + 4) = Nat.gcd 48 (n + 4) := by
  sorry

end gcd_n_cubed_plus_16_and_n_plus_4_l2249_224959


namespace meeting_impossible_l2249_224915

-- Define the type for people in the meeting
def Person : Type := ℕ

-- Define the relationship of knowing each other
def knows (p q : Person) : Prop := sorry

-- Define the number of people in the meeting
def num_people : ℕ := 65

-- State the conditions of the problem
axiom condition1 : ∀ p : Person, ∃ S : Finset Person, S.card ≥ 56 ∧ ∀ q ∈ S, ¬knows p q

axiom condition2 : ∀ p q : Person, p ≠ q → ∃ r : Person, r ≠ p ∧ r ≠ q ∧ knows r p ∧ knows r q

-- The theorem to be proved
theorem meeting_impossible : False := sorry

end meeting_impossible_l2249_224915


namespace sum_of_coordinates_X_l2249_224909

def Y : ℝ × ℝ := (2, 9)
def Z : ℝ × ℝ := (1, 5)

theorem sum_of_coordinates_X (X : ℝ × ℝ) 
  (h1 : (dist X Z) / (dist X Y) = 3 / 4)
  (h2 : (dist Z Y) / (dist X Y) = 1 / 4) : 
  X.1 + X.2 = -9 := by
  sorry

#check sum_of_coordinates_X

end sum_of_coordinates_X_l2249_224909


namespace expand_and_simplify_l2249_224993

theorem expand_and_simplify (x : ℝ) : (x + 5) * (4 * x - 9 - 3) = 4 * x^2 + 8 * x - 60 := by
  sorry

end expand_and_simplify_l2249_224993


namespace probability_ten_heads_in_twelve_flips_l2249_224983

theorem probability_ten_heads_in_twelve_flips :
  let n : ℕ := 12  -- Total number of coin flips
  let k : ℕ := 10  -- Number of desired heads
  let p : ℚ := 1/2 -- Probability of getting heads on a single flip (fair coin)
  Nat.choose n k * p^k * (1-p)^(n-k) = 66/4096 :=
by sorry

end probability_ten_heads_in_twelve_flips_l2249_224983


namespace specific_ellipse_area_l2249_224996

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Given the endpoints of the major axis and a point on the ellipse, 
    calculate the ellipse parameters -/
def calculateEllipse (p1 p2 p3 : Point) : Ellipse :=
  sorry

/-- Calculate the area of an ellipse -/
def ellipseArea (e : Ellipse) : ℝ :=
  sorry

/-- The main theorem stating the area of the specific ellipse -/
theorem specific_ellipse_area : 
  let p1 : Point := ⟨-10, 3⟩
  let p2 : Point := ⟨8, 3⟩
  let p3 : Point := ⟨6, 8⟩
  let e : Ellipse := calculateEllipse p1 p2 p3
  ellipseArea e = (405 * Real.pi) / (4 * Real.sqrt 2) := by
  sorry

end specific_ellipse_area_l2249_224996


namespace triangle_angle_b_is_pi_third_l2249_224985

theorem triangle_angle_b_is_pi_third (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  a / Real.sin A = b / Real.sin B ∧  -- Law of sines
  b / Real.sin B = c / Real.sin C ∧  -- Law of sines
  b * Real.cos B = (a * Real.cos C + c * Real.cos A) / 2  -- Given condition
  → B = π / 3 :=
by sorry

end triangle_angle_b_is_pi_third_l2249_224985


namespace curve_self_intersection_l2249_224968

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ := (t^2 - 4, t^3 - 6*t + 3)

-- Define the self-intersection point
def intersection_point : ℝ × ℝ := (2, 3)

-- Theorem statement
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve a = intersection_point :=
sorry

end curve_self_intersection_l2249_224968


namespace lily_trip_distance_l2249_224907

/-- Represents Lily's car and trip details -/
structure CarTrip where
  /-- Miles per gallon of the car -/
  mpg : ℝ
  /-- Capacity of the gas tank in gallons -/
  tank_capacity : ℝ
  /-- Initial distance driven in miles -/
  initial_distance : ℝ
  /-- First gas purchase in gallons -/
  first_gas_purchase : ℝ
  /-- Second gas purchase in gallons -/
  second_gas_purchase : ℝ
  /-- Fraction of tank full at arrival -/
  final_tank_fraction : ℝ

/-- Calculates the total distance driven given the car trip details -/
def total_distance (trip : CarTrip) : ℝ :=
  trip.initial_distance +
  trip.first_gas_purchase * trip.mpg +
  (trip.second_gas_purchase + trip.final_tank_fraction * trip.tank_capacity - trip.tank_capacity) * trip.mpg

/-- Theorem stating that Lily's total distance driven is 880 miles -/
theorem lily_trip_distance :
  let trip : CarTrip := {
    mpg := 40,
    tank_capacity := 12,
    initial_distance := 480,
    first_gas_purchase := 6,
    second_gas_purchase := 4,
    final_tank_fraction := 3/4
  }
  total_distance trip = 880 := by sorry

end lily_trip_distance_l2249_224907


namespace p_hyperbola_range_p_necessary_not_sufficient_for_q_l2249_224954

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (m - 4) = 1
def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 2) + y^2 / (4 - m) = 1

-- Define what it means for p to represent a hyperbola
def p_is_hyperbola (m : ℝ) : Prop := (m - 1) * (m - 4) < 0

-- Define what it means for q to represent an ellipse
def q_is_ellipse (m : ℝ) : Prop := m - 2 > 0 ∧ 4 - m > 0 ∧ m - 2 ≠ 4 - m

-- Theorem 1: The range of m for which p represents a hyperbola
theorem p_hyperbola_range : 
  ∀ m : ℝ, p_is_hyperbola m ↔ (1 < m ∧ m < 4) :=
sorry

-- Theorem 2: p being true is necessary but not sufficient for q being true
theorem p_necessary_not_sufficient_for_q :
  (∀ m : ℝ, q_is_ellipse m → p_is_hyperbola m) ∧
  (∃ m : ℝ, p_is_hyperbola m ∧ ¬q_is_ellipse m) :=
sorry

end p_hyperbola_range_p_necessary_not_sufficient_for_q_l2249_224954


namespace cost_of_pizza_slice_l2249_224942

/-- The cost of a slice of pizza given the conditions of Zoe's purchase -/
theorem cost_of_pizza_slice (num_people : ℕ) (soda_cost : ℚ) (total_spent : ℚ) :
  num_people = 6 →
  soda_cost = 1/2 →
  total_spent = 9 →
  (total_spent - num_people * soda_cost) / num_people = 1 := by
  sorry

#check cost_of_pizza_slice

end cost_of_pizza_slice_l2249_224942


namespace problem_solution_l2249_224939

theorem problem_solution (a : ℚ) : a + a / 4 = 6 / 2 → a = 12 / 5 := by
  sorry

end problem_solution_l2249_224939


namespace same_prime_factors_power_of_two_l2249_224989

theorem same_prime_factors_power_of_two (b m n : ℕ) 
  (hb : b ≠ 1) (hmn : m ≠ n) 
  (h_same_factors : ∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) :
  ∃ k : ℕ, b + 1 = 2^k := by sorry

end same_prime_factors_power_of_two_l2249_224989


namespace project_completion_time_l2249_224947

/-- The time taken for teams A and D to complete a project given the completion times of other team combinations -/
theorem project_completion_time (t_AB t_BC t_CD : ℝ) (h_AB : t_AB = 20) (h_BC : t_BC = 60) (h_CD : t_CD = 30) :
  1 / (1 / t_AB + 1 / t_CD - 1 / t_BC) = 15 := by
  sorry

#check project_completion_time

end project_completion_time_l2249_224947


namespace mod_inverse_sum_five_l2249_224919

theorem mod_inverse_sum_five : ∃ (a b : ℤ), 
  (5 * a) % 17 = 1 ∧ 
  (5^2 * b) % 17 = 1 ∧ 
  (a + b) % 17 = 14 := by
sorry

end mod_inverse_sum_five_l2249_224919


namespace smallest_third_term_geometric_progression_l2249_224921

theorem smallest_third_term_geometric_progression 
  (a : ℝ) -- Common difference of the arithmetic progression
  (h1 : (5 : ℝ) < 9 + a) -- Ensure the second term of GP is positive
  (h2 : 9 + a < 37 + 2*a) -- Ensure the third term of GP is greater than the second
  (h3 : (9 + a)^2 = 5*(37 + 2*a)) -- Condition for geometric progression
  : 
  ∃ (x : ℝ), x = 29 - 20*Real.sqrt 6 ∧ 
  x ≤ 37 + 2*a ∧
  ∀ (y : ℝ), y = 37 + 2*a → x ≤ y :=
by sorry

end smallest_third_term_geometric_progression_l2249_224921


namespace eliot_account_balance_l2249_224929

theorem eliot_account_balance :
  ∀ (A E : ℝ),
  A > E →
  A - E = (1 / 12) * (A + E) →
  1.1 * A = 1.2 * E + 21 →
  E = 210 :=
by
  sorry

end eliot_account_balance_l2249_224929


namespace rectangle_square_problem_l2249_224991

/-- Given a rectangle with length-to-width ratio 2:1 and area 50 cm², 
    and a square with the same area as the rectangle, prove:
    1. The rectangle's length is 10 cm and width is 5 cm
    2. The difference between the square's side length and the rectangle's width is 5(√2 - 1) cm -/
theorem rectangle_square_problem (length width : ℝ) (square_side : ℝ) : 
  length = 2 * width → 
  length * width = 50 → 
  square_side^2 = 50 → 
  (length = 10 ∧ width = 5) ∧ 
  square_side - width = 5 * (Real.sqrt 2 - 1) := by
  sorry

end rectangle_square_problem_l2249_224991


namespace ben_win_probability_l2249_224950

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 3 / 7) (h2 : ¬ ∃ tie_prob : ℚ, tie_prob ≠ 0) : 
  1 - lose_prob = 4 / 7 := by
sorry

end ben_win_probability_l2249_224950


namespace square_sum_from_difference_and_product_l2249_224908

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 := by
  sorry

end square_sum_from_difference_and_product_l2249_224908


namespace waiter_customers_l2249_224914

/-- The initial number of customers before 5 more arrived -/
def initial_customers : ℕ := 3

/-- The number of additional customers that arrived -/
def additional_customers : ℕ := 5

/-- The total number of customers after the additional customers arrived -/
def total_customers : ℕ := 8

theorem waiter_customers : 
  initial_customers + additional_customers = total_customers := by
  sorry

end waiter_customers_l2249_224914


namespace problem_solution_l2249_224930

theorem problem_solution (x y : ℝ) (h : 0.5 * x = y + 20) : x - 2 * y = 40 := by
  sorry

end problem_solution_l2249_224930


namespace jessie_initial_weight_l2249_224963

/-- Represents Jessie's weight change after jogging --/
structure WeightChange where
  lost : ℕ      -- Weight lost in kilograms
  current : ℕ   -- Current weight in kilograms

/-- Calculates the initial weight before jogging --/
def initial_weight (w : WeightChange) : ℕ :=
  w.lost + w.current

/-- Theorem stating Jessie's initial weight was 192 kg --/
theorem jessie_initial_weight :
  let w : WeightChange := { lost := 126, current := 66 }
  initial_weight w = 192 := by
  sorry

end jessie_initial_weight_l2249_224963


namespace inequality_proof_l2249_224901

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : (a + 1) * (b + 1) * (c + 1) = 8) :
  a + b + c ≥ 3 ∧ a * b * c ≤ 1 := by
  sorry

end inequality_proof_l2249_224901


namespace coefficient_of_x_cubed_is_six_l2249_224974

-- Define the polynomials
def p (x : ℝ) : ℝ := -3*x^3 - 8*x^2 + 3*x + 2
def q (x : ℝ) : ℝ := -2*x^2 - 7*x - 4

-- Define the product of the polynomials
def product (x : ℝ) : ℝ := p x * q x

-- Theorem statement
theorem coefficient_of_x_cubed_is_six :
  ∃ (a b c d : ℝ), product = fun x ↦ 6*x^3 + a*x^2 + b*x + c + d*x^4 := by
  sorry

end coefficient_of_x_cubed_is_six_l2249_224974


namespace maggie_bouncy_balls_l2249_224902

/-- The number of bouncy balls in each pack -/
def ballsPerPack : ℕ := 10

/-- The number of packs of red bouncy balls -/
def redPacks : ℕ := 4

/-- The number of packs of yellow bouncy balls -/
def yellowPacks : ℕ := 8

/-- The number of packs of green bouncy balls -/
def greenPacks : ℕ := 4

/-- The total number of bouncy balls Maggie bought -/
def totalBalls : ℕ := ballsPerPack * (redPacks + yellowPacks + greenPacks)

theorem maggie_bouncy_balls : totalBalls = 160 := by
  sorry

end maggie_bouncy_balls_l2249_224902


namespace expand_expression_l2249_224999

theorem expand_expression (x : ℝ) : (3 * x + 5) * (4 * x - 2) = 12 * x^2 + 14 * x - 10 := by
  sorry

end expand_expression_l2249_224999


namespace percentage_problem_l2249_224978

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 600 = (50 / 100) * 1080 → P = 90 := by
  sorry

end percentage_problem_l2249_224978


namespace problems_per_page_l2249_224910

/-- Given the total number of homework problems, the number of finished problems,
    and the number of remaining pages, calculate the number of problems per page. -/
theorem problems_per_page
  (total_problems : ℕ)
  (finished_problems : ℕ)
  (remaining_pages : ℕ)
  (h1 : total_problems = 40)
  (h2 : finished_problems = 26)
  (h3 : remaining_pages = 2)
  (h4 : remaining_pages > 0)
  (h5 : finished_problems ≤ total_problems) :
  (total_problems - finished_problems) / remaining_pages = 7 := by
sorry

end problems_per_page_l2249_224910


namespace count_common_divisors_9240_8820_l2249_224938

/-- The number of positive divisors that 9240 and 8820 have in common -/
def common_divisors_count : ℕ := 24

/-- Theorem stating that the number of positive divisors that 9240 and 8820 have in common is 24 -/
theorem count_common_divisors_9240_8820 : 
  (Nat.divisors 9240 ∩ Nat.divisors 8820).card = common_divisors_count := by
  sorry

end count_common_divisors_9240_8820_l2249_224938


namespace parallel_lines_c_value_l2249_224937

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The problem statement -/
theorem parallel_lines_c_value :
  ∀ c : ℝ, (∀ x y : ℝ, y = 5 * x + 7 ↔ y = (3 * c) * x + 1) → c = 5 / 3 :=
by sorry

end parallel_lines_c_value_l2249_224937


namespace banana_cost_theorem_l2249_224948

def cost_of_fruit (apple_cost banana_cost orange_cost : ℚ)
                  (apple_count banana_count orange_count : ℕ)
                  (average_cost : ℚ) : Prop :=
  let total_count := apple_count + banana_count + orange_count
  let total_cost := apple_cost * apple_count + banana_cost * banana_count + orange_cost * orange_count
  total_cost = average_cost * total_count

theorem banana_cost_theorem :
  ∀ (banana_cost : ℚ),
    cost_of_fruit 2 banana_cost 3 12 4 4 2 →
    banana_cost = 1 :=
by
  sorry

end banana_cost_theorem_l2249_224948


namespace zuminglish_seven_letter_words_l2249_224994

/-- Represents the ending of a word -/
inductive WordEnding
| CC  -- Two consonants
| CV  -- Consonant followed by vowel
| VC  -- Vowel followed by consonant

/-- Represents the rules of Zuminglish -/
structure Zuminglish where
  -- The number of n-letter words ending in each type
  count : ℕ → WordEnding → ℕ
  -- Initial conditions for 2-letter words
  init_CC : count 2 WordEnding.CC = 4
  init_CV : count 2 WordEnding.CV = 2
  init_VC : count 2 WordEnding.VC = 2
  -- Recursive relations
  rec_CC : ∀ n, count (n+1) WordEnding.CC = 2 * (count n WordEnding.CC + count n WordEnding.VC)
  rec_CV : ∀ n, count (n+1) WordEnding.CV = count n WordEnding.CC
  rec_VC : ∀ n, count (n+1) WordEnding.VC = 2 * count n WordEnding.CV

/-- The main theorem stating the number of valid 7-letter words in Zuminglish -/
theorem zuminglish_seven_letter_words (z : Zuminglish) :
  z.count 7 WordEnding.CC + z.count 7 WordEnding.CV + z.count 7 WordEnding.VC = 912 := by
  sorry


end zuminglish_seven_letter_words_l2249_224994


namespace sum_of_max_and_min_is_two_l2249_224958

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x - 4| - |2*x - 6|

-- Define the domain
def domain : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }

-- State the theorem
theorem sum_of_max_and_min_is_two :
  ∃ (max min : ℝ), 
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    (∀ x ∈ domain, min ≤ f x) ∧
    (∃ x ∈ domain, f x = min) ∧
    max + min = 2 := by
  sorry

end sum_of_max_and_min_is_two_l2249_224958


namespace count_divisors_252_not_div_by_seven_l2249_224984

def divisors_not_div_by_seven (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ x => x > 0 ∧ n % x = 0 ∧ x % 7 ≠ 0)

theorem count_divisors_252_not_div_by_seven :
  (divisors_not_div_by_seven 252).card = 9 := by
  sorry

end count_divisors_252_not_div_by_seven_l2249_224984


namespace hyperbola_k_range_l2249_224970

/-- A hyperbola is represented by the equation (x^2 / (k-2)) + (y^2 / (5-k)) = 1 -/
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 / (k-2)) + (y^2 / (5-k)) = 1 ∧ (k-2) * (5-k) < 0

/-- The range of k for which the equation represents a hyperbola -/
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ k < 2 ∨ k > 5 :=
by sorry

end hyperbola_k_range_l2249_224970


namespace instantaneous_velocity_at_3_seconds_l2249_224944

-- Define the displacement function
def displacement (t : ℝ) : ℝ := t^2 - t

-- Define the velocity function as the derivative of displacement
def velocity (t : ℝ) : ℝ := 2 * t - 1

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds : 
  velocity 3 = 5 := by sorry

end instantaneous_velocity_at_3_seconds_l2249_224944


namespace birth_outcome_probabilities_l2249_224923

def num_children : ℕ := 5
def prob_boy : ℚ := 1/2
def prob_girl : ℚ := 1/2

theorem birth_outcome_probabilities :
  let prob_all_boys : ℚ := prob_boy ^ num_children
  let prob_all_girls : ℚ := prob_girl ^ num_children
  let prob_three_girls_two_boys : ℚ := (Nat.choose num_children 3) * (prob_girl ^ 3) * (prob_boy ^ 2)
  let prob_four_one : ℚ := 2 * (Nat.choose num_children 1) * (prob_girl ^ 4) * prob_boy
  prob_three_girls_two_boys = prob_four_one ∧
  prob_three_girls_two_boys > prob_all_boys ∧
  prob_three_girls_two_boys > prob_all_girls :=
by
  sorry

#check birth_outcome_probabilities

end birth_outcome_probabilities_l2249_224923


namespace prob_heart_then_king_is_one_52_l2249_224973

def standard_deck : ℕ := 52
def hearts_in_deck : ℕ := 13
def kings_in_deck : ℕ := 4

def prob_heart_then_king : ℚ :=
  (hearts_in_deck / standard_deck) * (kings_in_deck / (standard_deck - 1))

theorem prob_heart_then_king_is_one_52 :
  prob_heart_then_king = 1 / 52 := by sorry

end prob_heart_then_king_is_one_52_l2249_224973


namespace inequality_condition_l2249_224940

theorem inequality_condition (x : ℝ) : x * (x + 2) > x * (3 - x) + 1 ↔ x < -1/2 ∨ x > 1 := by
  sorry

end inequality_condition_l2249_224940


namespace police_coverage_l2249_224990

-- Define the set of intersections
inductive Intersection : Type
  | A | B | C | D | E | F | G | H | I | J | K

-- Define the streets as sets of intersections
def horizontal_streets : List (List Intersection) :=
  [[Intersection.A, Intersection.B, Intersection.C, Intersection.D],
   [Intersection.E, Intersection.F, Intersection.G],
   [Intersection.H, Intersection.I, Intersection.J, Intersection.K]]

def vertical_streets : List (List Intersection) :=
  [[Intersection.A, Intersection.E, Intersection.H],
   [Intersection.B, Intersection.F, Intersection.I],
   [Intersection.D, Intersection.G, Intersection.J]]

def diagonal_streets : List (List Intersection) :=
  [[Intersection.H, Intersection.F, Intersection.C],
   [Intersection.C, Intersection.G, Intersection.K]]

def all_streets : List (List Intersection) :=
  horizontal_streets ++ vertical_streets ++ diagonal_streets

-- Define the function to check if a street is covered by the given intersections
def street_covered (street : List Intersection) (officers : List Intersection) : Prop :=
  ∃ i ∈ street, i ∈ officers

-- Theorem statement
theorem police_coverage :
  ∀ (street : List Intersection),
    street ∈ all_streets →
    street_covered street [Intersection.B, Intersection.G, Intersection.H] :=
by
  sorry


end police_coverage_l2249_224990


namespace clea_escalator_time_l2249_224934

/-- Represents the scenario of Clea walking on an escalator -/
structure EscalatorScenario where
  /-- Clea's walking speed (units per second) -/
  walking_speed : ℝ
  /-- Total distance of the escalator (units) -/
  escalator_distance : ℝ
  /-- Speed of the moving escalator (units per second) -/
  escalator_speed : ℝ

/-- Time taken for Clea to walk down the stationary escalator -/
def time_stationary (scenario : EscalatorScenario) : ℝ :=
  80

/-- Time taken for Clea to walk down the moving escalator -/
def time_moving (scenario : EscalatorScenario) : ℝ :=
  32

/-- Theorem stating the time taken for the given scenario -/
theorem clea_escalator_time (scenario : EscalatorScenario) :
  scenario.escalator_speed = 1.5 * scenario.walking_speed →
  (scenario.escalator_distance / scenario.walking_speed / 2) +
  (scenario.escalator_distance / (2 * scenario.escalator_speed)) = 200 / 3 := by
  sorry

end clea_escalator_time_l2249_224934


namespace robyn_cookie_sales_l2249_224955

/-- Given that Robyn and Lucy sold a total of 98 packs of cookies,
    and Lucy sold 43 packs, prove that Robyn sold 55 packs. -/
theorem robyn_cookie_sales (total : ℕ) (lucy : ℕ) (robyn : ℕ)
    (h1 : total = 98)
    (h2 : lucy = 43)
    (h3 : total = lucy + robyn) :
  robyn = 55 := by
  sorry

end robyn_cookie_sales_l2249_224955


namespace partner_a_income_increase_l2249_224969

/-- Represents the increase in partner a's income when the profit rate changes --/
def income_increase (capital : ℝ) (initial_rate final_rate : ℝ) (share : ℝ) : ℝ :=
  share * (final_rate - initial_rate) * capital

/-- Theorem stating the increase in partner a's income given the problem conditions --/
theorem partner_a_income_increase :
  let capital : ℝ := 10000
  let initial_rate : ℝ := 0.05
  let final_rate : ℝ := 0.07
  let share : ℝ := 2/3
  income_increase capital initial_rate final_rate share = 400/3 := by sorry

end partner_a_income_increase_l2249_224969


namespace probability_not_rain_l2249_224943

theorem probability_not_rain (p : ℚ) (h : p = 3 / 10) : 1 - p = 7 / 10 := by
  sorry

end probability_not_rain_l2249_224943


namespace equation_solvable_for_small_primes_l2249_224962

theorem equation_solvable_for_small_primes :
  ∀ p : ℕ, p.Prime → p ≤ 100 →
  ∃ x y : ℕ, (y^37 : ℤ) ≡ (x^3 + 11 : ℤ) [ZMOD p] := by
  sorry

end equation_solvable_for_small_primes_l2249_224962


namespace ten_steps_climb_ways_l2249_224975

/-- The number of ways to climb n steps, where each move is either climbing 1 step or 2 steps -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => climbStairs k + climbStairs (k + 1)

/-- Theorem stating that there are 89 ways to climb 10 steps -/
theorem ten_steps_climb_ways : climbStairs 10 = 89 := by
  sorry

end ten_steps_climb_ways_l2249_224975


namespace sum_c_d_eq_nine_l2249_224941

/-- A quadrilateral PQRS with specific vertex coordinates -/
structure Quadrilateral (c d : ℤ) :=
  (c_pos : c > 0)
  (d_pos : d > 0)
  (c_gt_d : c > d)

/-- The area of the quadrilateral PQRS -/
def area (q : Quadrilateral c d) : ℝ := 2 * ((c : ℝ)^2 - (d : ℝ)^2)

theorem sum_c_d_eq_nine {c d : ℤ} (q : Quadrilateral c d) (h : area q = 18) :
  c + d = 9 := by
  sorry

#check sum_c_d_eq_nine

end sum_c_d_eq_nine_l2249_224941


namespace quadratic_roots_theorem_l2249_224906

theorem quadratic_roots_theorem (k : ℝ) (a b : ℝ) : 
  (∀ x, k * (x^2 - x) + x + 2 = 0 ↔ x = a ∨ x = b) →
  (a / b + b / a = 3 / 7) →
  (∃ k₁ k₂ : ℝ, 
    k₁ = (20 + Real.sqrt 988) / 14 ∧
    k₂ = (20 - Real.sqrt 988) / 14 ∧
    (k = k₁ ∨ k = k₂) ∧
    k₁ / k₂ + k₂ / k₁ = -104 / 21) :=
by sorry

end quadratic_roots_theorem_l2249_224906


namespace impossible_chord_length_l2249_224972

theorem impossible_chord_length (r : ℝ) (chord_length : ℝ) : 
  r = 5 → chord_length = 11 → chord_length > 2 * r := by sorry

end impossible_chord_length_l2249_224972


namespace savings_after_twelve_months_l2249_224913

def savings_sequence (n : ℕ) : ℕ := 2 ^ n

theorem savings_after_twelve_months :
  savings_sequence 12 = 4096 := by sorry

end savings_after_twelve_months_l2249_224913


namespace number_of_divisors_5400_l2249_224936

theorem number_of_divisors_5400 : Nat.card (Nat.divisors 5400) = 48 := by
  sorry

end number_of_divisors_5400_l2249_224936


namespace quadratic_surds_problem_l2249_224920

-- Define the variables and equations
theorem quadratic_surds_problem (x y : ℝ) 
  (hA : 5 * Real.sqrt (2 * x + 1) = 5 * Real.sqrt 5)
  (hB : 3 * Real.sqrt (x + 3) = 3 * Real.sqrt 5)
  (hC : Real.sqrt (10 * x + 3 * y) = Real.sqrt 320)
  (hAB : 5 * Real.sqrt (2 * x + 1) + 3 * Real.sqrt (x + 3) = Real.sqrt (10 * x + 3 * y)) :
  Real.sqrt (2 * y - x^2) = 14 := by
  sorry


end quadratic_surds_problem_l2249_224920


namespace school_play_tickets_l2249_224935

/-- Calculates the total number of tickets sold for a school play. -/
def total_tickets (adult_tickets : ℕ) : ℕ :=
  adult_tickets + 2 * adult_tickets

/-- Theorem: Given 122 adult tickets and student tickets being twice the number of adult tickets,
    the total number of tickets sold is 366. -/
theorem school_play_tickets : total_tickets 122 = 366 := by
  sorry

end school_play_tickets_l2249_224935


namespace max_type_a_stationery_l2249_224992

/-- Represents the number of items for each stationery type -/
structure Stationery where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total cost of stationery -/
def totalCost (s : Stationery) : ℕ :=
  3 * s.a + 2 * s.b + s.c

/-- Checks if the stationery purchase satisfies all conditions -/
def isValidPurchase (s : Stationery) : Prop :=
  s.b = s.a - 2 ∧
  3 * s.a ≤ 33 ∧
  totalCost s = 66

/-- Theorem: The maximum number of Type A stationery that can be purchased is 11 -/
theorem max_type_a_stationery :
  ∃ (s : Stationery), isValidPurchase s ∧
  (∀ (t : Stationery), isValidPurchase t → t.a ≤ s.a) ∧
  s.a = 11 := by
  sorry


end max_type_a_stationery_l2249_224992


namespace max_silver_tokens_l2249_224977

/-- Represents the state of tokens -/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange at a booth -/
inductive Exchange
  | First  : Exchange
  | Second : Exchange

/-- Applies an exchange to a token state -/
def applyExchange (s : TokenState) (e : Exchange) : Option TokenState :=
  match e with
  | Exchange.First =>
      if s.red ≥ 3 then
        some { red := s.red - 3, blue := s.blue + 2, silver := s.silver + 1 }
      else
        none
  | Exchange.Second =>
      if s.blue ≥ 4 then
        some { red := s.red + 2, blue := s.blue - 4, silver := s.silver + 1 }
      else
        none

/-- Theorem: The maximum number of silver tokens Alex can obtain is 39 -/
theorem max_silver_tokens (initialState : TokenState)
    (h_initial_red : initialState.red = 100)
    (h_initial_blue : initialState.blue = 90)
    (h_initial_silver : initialState.silver = 0) :
    (∃ (finalState : TokenState),
      (∀ e : Exchange, applyExchange finalState e = none) ∧
      finalState.silver = 39 ∧
      (∀ otherState : TokenState,
        (∀ e : Exchange, applyExchange otherState e = none) →
        otherState.silver ≤ finalState.silver)) :=
  sorry


end max_silver_tokens_l2249_224977


namespace pirate_loot_value_l2249_224953

/-- Converts a number from base 5 to base 10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The loot values in base 5 --/
def silverware : List Nat := [3, 2, 1, 4]
def silkGarments : List Nat := [1, 2, 0, 2]
def rareSpices : List Nat := [1, 3, 2]

theorem pirate_loot_value :
  base5ToBase10 silverware + base5ToBase10 silkGarments + base5ToBase10 rareSpices = 865 := by
  sorry


end pirate_loot_value_l2249_224953


namespace folded_blankets_theorem_l2249_224964

/-- The thickness of a stack of folded blankets -/
def folded_blankets_thickness (initial_thickness : ℕ) (num_blankets : ℕ) (num_folds : ℕ) : ℕ :=
  num_blankets * initial_thickness * (2 ^ num_folds)

/-- Theorem: The thickness of n blankets, each initially 3 inches thick and folded 4 times, is 48n inches -/
theorem folded_blankets_theorem (n : ℕ) :
  folded_blankets_thickness 3 n 4 = 48 * n := by
  sorry

end folded_blankets_theorem_l2249_224964


namespace total_production_cost_l2249_224911

def initial_cost_per_episode : ℕ := 100000
def cost_increase_rate : ℚ := 1.2
def initial_episodes : ℕ := 12
def season_2_increase : ℚ := 1.3
def subsequent_seasons_increase : ℚ := 1.1
def final_season_decrease : ℚ := 0.85
def total_seasons : ℕ := 7

def calculate_total_cost : ℕ := sorry

theorem total_production_cost :
  calculate_total_cost = 25673856 := by sorry

end total_production_cost_l2249_224911


namespace fold_square_crease_length_l2249_224933

-- Define the square ABCD
def square_side : ℝ := 8

-- Define point E on AD
def AE : ℝ := 2
def ED : ℝ := 6

-- Define FD as x
def FD : ℝ → ℝ := λ x => x

-- Define CF and EF
def CF (x : ℝ) : ℝ := square_side - x
def EF (x : ℝ) : ℝ := square_side - x

-- State the theorem
theorem fold_square_crease_length :
  ∃ x : ℝ, FD x = 7/4 ∧ CF x = EF x ∧ CF x^2 = FD x^2 + ED^2 := by
  sorry

end fold_square_crease_length_l2249_224933


namespace correct_rounded_result_l2249_224946

def round_to_nearest_hundred (n : ℤ) : ℤ :=
  (n + 50) / 100 * 100

theorem correct_rounded_result : round_to_nearest_hundred (68 + 57) = 100 := by
  sorry

end correct_rounded_result_l2249_224946


namespace polynomial_division_theorem_l2249_224986

theorem polynomial_division_theorem (z : ℝ) :
  4 * z^4 - 6 * z^3 + 7 * z^2 - 17 * z + 3 =
  (5 * z + 4) * (z^3 - (26/5) * z^2 + (1/5) * z - 67/25) + 331/25 := by
  sorry

end polynomial_division_theorem_l2249_224986


namespace condition_p_sufficient_not_necessary_l2249_224917

theorem condition_p_sufficient_not_necessary :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2 ∧ x * y > 1) ∧
  (∃ x y : ℝ, x + y > 2 ∧ x * y > 1 ∧ ¬(x > 1 ∧ y > 1)) := by
  sorry

end condition_p_sufficient_not_necessary_l2249_224917


namespace system_one_solution_system_two_solution_l2249_224980

-- System (1)
theorem system_one_solution (x y : ℝ) : 
  x + y = 1 ∧ 3 * x + y = 5 → x = 2 ∧ y = -1 := by sorry

-- System (2)
theorem system_two_solution (x y : ℝ) : 
  3 * (x - 1) + 4 * y = 1 ∧ 2 * x + 3 * (y + 1) = 2 → x = 16 ∧ y = -11 := by sorry

end system_one_solution_system_two_solution_l2249_224980


namespace defective_units_shipped_percentage_l2249_224932

theorem defective_units_shipped_percentage
  (total_units : ℝ)
  (defective_percentage : ℝ)
  (defective_shipped_percentage : ℝ)
  (h1 : defective_percentage = 8)
  (h2 : defective_shipped_percentage = 0.4) :
  (defective_shipped_percentage / defective_percentage) * 100 = 5 := by
sorry

end defective_units_shipped_percentage_l2249_224932


namespace point_not_in_region_l2249_224976

/-- The plane region is represented by the inequality 3x + 2y < 6 -/
def in_plane_region (x y : ℝ) : Prop := 3 * x + 2 * y < 6

/-- The point (2, 0) is not in the plane region -/
theorem point_not_in_region : ¬ in_plane_region 2 0 := by
  sorry

end point_not_in_region_l2249_224976


namespace modified_coin_expected_winnings_l2249_224995

/-- A coin with three possible outcomes -/
structure Coin where
  prob_heads : ℚ
  prob_tails : ℚ
  prob_edge : ℚ
  winnings_heads : ℚ
  winnings_tails : ℚ
  loss_edge : ℚ

/-- The modified weighted coin as described in the problem -/
def modified_coin : Coin :=
  { prob_heads := 1/3
  , prob_tails := 1/2
  , prob_edge := 1/6
  , winnings_heads := 2
  , winnings_tails := 2
  , loss_edge := 4 }

/-- Expected winnings from flipping the coin -/
def expected_winnings (c : Coin) : ℚ :=
  c.prob_heads * c.winnings_heads + c.prob_tails * c.winnings_tails - c.prob_edge * c.loss_edge

/-- Theorem stating that the expected winnings from flipping the modified coin is 1 -/
theorem modified_coin_expected_winnings :
  expected_winnings modified_coin = 1 := by
  sorry

end modified_coin_expected_winnings_l2249_224995


namespace museum_ticket_price_museum_ticket_price_is_6_l2249_224928

theorem museum_ticket_price (friday_price : ℝ) (saturday_visitors : ℕ) 
  (saturday_visitor_ratio : ℝ) (saturday_revenue_ratio : ℝ) : ℝ :=
let friday_visitors : ℕ := saturday_visitors / 2
let friday_revenue : ℝ := friday_visitors * friday_price
let saturday_revenue : ℝ := friday_revenue * saturday_revenue_ratio
let k : ℝ := saturday_revenue / saturday_visitors
k

theorem museum_ticket_price_is_6 :
  museum_ticket_price 9 200 2 (4/3) = 6 := by
sorry

end museum_ticket_price_museum_ticket_price_is_6_l2249_224928


namespace sum_extrema_l2249_224956

theorem sum_extrema (x y z w : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ w ≥ 0) 
  (h_eq : x^2 + y^2 + z^2 + w^2 + x + 2*y + 3*z + 4*w = 17/2) :
  (∀ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 + a + 2*b + 3*c + 4*d = 17/2 → 
    a + b + c + d ≤ 3) ∧
  (x + y + z + w ≥ -2 + 5/2 * Real.sqrt 2) :=
by sorry

end sum_extrema_l2249_224956


namespace sqrt_four_twos_to_fourth_l2249_224903

theorem sqrt_four_twos_to_fourth : Real.sqrt (2^4 + 2^4 + 2^4 + 2^4) = 8 := by
  sorry

end sqrt_four_twos_to_fourth_l2249_224903


namespace inscribed_quadrilateral_theorem_l2249_224982

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral :=
  (radius : ℝ)
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

/-- The theorem about the inscribed quadrilateral -/
theorem inscribed_quadrilateral_theorem (q : InscribedQuadrilateral) :
  q.radius = 300 ∧ q.side1 = 300 ∧ q.side2 = 300 ∧ q.side3 = 200 →
  q.side4 = 300 := by
  sorry

end inscribed_quadrilateral_theorem_l2249_224982


namespace uncle_pill_duration_l2249_224967

/-- Represents the duration in days that a bottle of pills lasts -/
def bottle_duration (pills_per_bottle : ℕ) (dose : ℚ) (days_between_doses : ℕ) : ℚ :=
  (pills_per_bottle : ℚ) * (days_between_doses : ℚ) / dose

/-- Converts days to months, assuming 30 days per month -/
def days_to_months (days : ℚ) : ℚ :=
  days / 30

theorem uncle_pill_duration :
  let pills_per_bottle : ℕ := 60
  let dose : ℚ := 3/4
  let days_between_doses : ℕ := 3
  days_to_months (bottle_duration pills_per_bottle dose days_between_doses) = 8 := by
  sorry

#eval days_to_months (bottle_duration 60 (3/4) 3)

end uncle_pill_duration_l2249_224967


namespace range_of_sqrt_function_l2249_224916

theorem range_of_sqrt_function (x : ℝ) :
  (∃ y : ℝ, y = Real.sqrt (2 - x)) ↔ x ≤ 2 := by
  sorry

end range_of_sqrt_function_l2249_224916
