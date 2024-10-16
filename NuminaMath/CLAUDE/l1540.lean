import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l1540_154004

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) (hm : m > 0) (hn : n > 0) :
  let f := fun x => a^(x - 1) - 2
  let A := (1, -1)
  (m * A.1 - n * A.2 - 1 = 0) →
  (∀ x, f x = -1 → x = 1) →
  (∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧
    ∀ (m' n' : ℝ), m' > 0 → n' > 0 → m' * A.1 - n' * A.2 - 1 = 0 →
      1 / m' + 2 / n' ≥ min_val) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1540_154004


namespace NUMINAMATH_CALUDE_position_function_correct_l1540_154042

/-- The velocity function --/
def v (t : ℝ) : ℝ := 3 * t^2 - 1

/-- The position function --/
def s (t : ℝ) : ℝ := t^3 - t + 0.05

/-- Theorem stating that s is the correct position function --/
theorem position_function_correct :
  (∀ t, (deriv s) t = v t) ∧ s 0 = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_position_function_correct_l1540_154042


namespace NUMINAMATH_CALUDE_parallelogram_area_specific_vectors_l1540_154002

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (v w : ℝ × ℝ) : ℝ :=
  |v.1 * w.2 - v.2 * w.1|

theorem parallelogram_area_specific_vectors :
  let v : ℝ × ℝ := (8, -5)
  let w : ℝ × ℝ := (14, -3)
  parallelogramArea v w = 46 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_specific_vectors_l1540_154002


namespace NUMINAMATH_CALUDE_min_distance_exponential_linear_l1540_154034

theorem min_distance_exponential_linear (t : ℝ) : 
  let f (x : ℝ) := Real.exp x
  let g (x : ℝ) := 2 * x
  let distance (t : ℝ) := |f t - g t|
  ∃ (min_dist : ℝ), ∀ (t : ℝ), distance t ≥ min_dist ∧ min_dist = 2 - 2 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_exponential_linear_l1540_154034


namespace NUMINAMATH_CALUDE_fifth_term_of_8998_sequence_l1540_154073

-- Define the sequence generation function
def generateSequence (n : ℕ) : List ℕ :=
  -- Implementation of the sequence generation rules
  sorry

-- Define a function to get the nth term of a sequence
def getNthTerm (sequence : List ℕ) (n : ℕ) : ℕ :=
  -- Implementation to get the nth term
  sorry

-- Theorem statement
theorem fifth_term_of_8998_sequence :
  getNthTerm (generateSequence 8998) 5 = 4625 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_8998_sequence_l1540_154073


namespace NUMINAMATH_CALUDE_existence_of_four_pairs_l1540_154087

theorem existence_of_four_pairs :
  ∃ (a₁ b₁ a₂ b₂ a₃ b₃ a₄ b₄ : ℝ),
    (0 < a₁ ∧ a₁ < 1) ∧ (0 < b₁ ∧ b₁ < 1) ∧ a₁ ≠ b₁ ∧
    (0 < a₂ ∧ a₂ < 1) ∧ (0 < b₂ ∧ b₂ < 1) ∧ a₂ ≠ b₂ ∧
    (0 < a₃ ∧ a₃ < 1) ∧ (0 < b₃ ∧ b₃ < 1) ∧ a₃ ≠ b₃ ∧
    (0 < a₄ ∧ a₄ < 1) ∧ (0 < b₄ ∧ b₄ < 1) ∧ a₄ ≠ b₄ ∧
    (Real.sqrt ((1 - a₁^2) * (1 - b₁^2)) > a₁ / (2 * b₁) + b₁ / (2 * a₁) - a₁ * b₁ - 1 / (8 * a₁ * b₁)) ∧
    (Real.sqrt ((1 - a₂^2) * (1 - b₂^2)) > a₂ / (2 * b₂) + b₂ / (2 * a₂) - a₂ * b₂ - 1 / (8 * a₂ * b₂)) ∧
    (Real.sqrt ((1 - a₃^2) * (1 - b₃^2)) > a₃ / (2 * b₃) + b₃ / (2 * a₃) - a₃ * b₃ - 1 / (8 * a₃ * b₃)) ∧
    (Real.sqrt ((1 - a₄^2) * (1 - b₄^2)) > a₄ / (2 * b₄) + b₄ / (2 * a₄) - a₄ * b₄ - 1 / (8 * a₄ * b₄)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_four_pairs_l1540_154087


namespace NUMINAMATH_CALUDE_c_investment_is_2000_l1540_154043

/-- Represents a partnership investment and profit distribution --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit : ℕ

/-- Theorem stating that under given conditions, C's investment is 2000 --/
theorem c_investment_is_2000 (p : Partnership) 
  (h1 : p.a_investment = 8000)
  (h2 : p.b_investment = 4000)
  (h3 : p.total_profit = 252000)
  (h4 : p.c_profit = 36000)
  (h5 : p.c_profit * (p.a_investment + p.b_investment + p.c_investment) = 
        p.c_investment * p.total_profit) : 
  p.c_investment = 2000 := by
  sorry

#check c_investment_is_2000

end NUMINAMATH_CALUDE_c_investment_is_2000_l1540_154043


namespace NUMINAMATH_CALUDE_train_stoppage_time_l1540_154075

/-- Calculates the stoppage time per hour for a train given its speeds with and without stoppages. -/
theorem train_stoppage_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 48) 
  (h2 : speed_with_stops = 40) : 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_stoppage_time_l1540_154075


namespace NUMINAMATH_CALUDE_container_capacity_l1540_154040

theorem container_capacity (container_volume : ℝ) (num_containers : ℕ) : 
  (8 : ℝ) = 0.2 * container_volume → 
  num_containers = 40 → 
  num_containers * container_volume = 1600 := by
sorry

end NUMINAMATH_CALUDE_container_capacity_l1540_154040


namespace NUMINAMATH_CALUDE_first_discount_percentage_l1540_154093

def original_price : ℝ := 345
def final_price : ℝ := 227.70
def second_discount : ℝ := 0.25

theorem first_discount_percentage :
  ∃ (d : ℝ), d ≥ 0 ∧ d ≤ 1 ∧
  original_price * (1 - d) * (1 - second_discount) = final_price ∧
  d = 0.12 :=
sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l1540_154093


namespace NUMINAMATH_CALUDE_square_sum_inequality_l1540_154060

theorem square_sum_inequality (a b x y : ℝ) : (a^2 + b^2) * (x^2 + y^2) ≥ (a*x + b*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l1540_154060


namespace NUMINAMATH_CALUDE_train_speed_l1540_154072

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 170 →
  crossing_time = 17.39860811135109 →
  (train_length + bridge_length) / crossing_time * 3.6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1540_154072


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1540_154099

/-- Proves that in a triangle ABC, if angle C is triple angle B and angle B is 18°, then angle A is 108° -/
theorem triangle_angle_calculation (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- Angles are positive
  A + B + C = 180 → -- Sum of angles in a triangle
  C = 3 * B → -- Angle C is triple angle B
  B = 18 → -- Angle B is 18°
  A = 108 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1540_154099


namespace NUMINAMATH_CALUDE_ruble_payment_l1540_154096

theorem ruble_payment (n : ℤ) (h : n > 7) : ∃ x y : ℕ, 3 * x + 5 * y = n := by
  sorry

end NUMINAMATH_CALUDE_ruble_payment_l1540_154096


namespace NUMINAMATH_CALUDE_second_division_remainder_l1540_154037

theorem second_division_remainder (n : ℕ) : 
  n % 68 = 0 ∧ n / 68 = 269 → n % 18291 = 1 :=
by sorry

end NUMINAMATH_CALUDE_second_division_remainder_l1540_154037


namespace NUMINAMATH_CALUDE_bryan_collected_from_four_continents_l1540_154008

/-- The number of books Bryan collected per continent -/
def books_per_continent : ℕ := 122

/-- The total number of books Bryan collected -/
def total_books : ℕ := 488

/-- The number of continents Bryan collected books from -/
def num_continents : ℕ := total_books / books_per_continent

theorem bryan_collected_from_four_continents :
  num_continents = 4 := by sorry

end NUMINAMATH_CALUDE_bryan_collected_from_four_continents_l1540_154008


namespace NUMINAMATH_CALUDE_complex_number_opposite_parts_l1540_154083

theorem complex_number_opposite_parts (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_opposite_parts_l1540_154083


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1540_154009

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 30)
  (h2 : 4 * (a + b + c) = 28) :
  (a^2 + b^2 + c^2).sqrt = (19 : ℝ).sqrt := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1540_154009


namespace NUMINAMATH_CALUDE_train_speed_train_speed_result_l1540_154079

/-- The speed of a train given its length, the time to cross a man, and the man's speed --/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / crossing_time
  let train_speed_ms := relative_speed + man_speed_ms
  let train_speed_kmh := train_speed_ms * (3600 / 1000)
  train_speed_kmh

/-- The speed of the train is approximately 63.0036 km/hr --/
theorem train_speed_result :
  ∃ ε > 0, abs (train_speed 250 14.998800095992321 3 - 63.0036) < ε :=
sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_result_l1540_154079


namespace NUMINAMATH_CALUDE_tanner_video_game_cost_l1540_154078

/-- The cost of Tanner's video game purchase -/
def video_game_cost (september_savings october_savings november_savings remaining_amount : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - remaining_amount

/-- Theorem stating the cost of Tanner's video game -/
theorem tanner_video_game_cost :
  video_game_cost 17 48 25 41 = 49 := by
  sorry

end NUMINAMATH_CALUDE_tanner_video_game_cost_l1540_154078


namespace NUMINAMATH_CALUDE_alternating_series_ratio_l1540_154001

theorem alternating_series_ratio : 
  (1 - 2 + 4 - 8 + 16 - 32 + 64 - 128) / 
  (1^2 + 2^2 - 4^2 + 8^2 + 16^2 - 32^2 + 64^2 - 128^2) = 1 / 113 := by
  sorry

end NUMINAMATH_CALUDE_alternating_series_ratio_l1540_154001


namespace NUMINAMATH_CALUDE_highest_score_calculation_l1540_154010

theorem highest_score_calculation (scores : Finset ℕ) (lowest highest : ℕ) :
  Finset.card scores = 15 →
  (Finset.sum scores id) / 15 = 90 →
  ((Finset.sum scores id) - lowest - highest) / 13 = 92 →
  lowest = 65 →
  highest = 89 := by
  sorry

end NUMINAMATH_CALUDE_highest_score_calculation_l1540_154010


namespace NUMINAMATH_CALUDE_business_school_size_l1540_154092

/-- The number of students in the law school -/
def law_students : ℕ := 800

/-- The number of sibling pairs -/
def sibling_pairs : ℕ := 30

/-- The probability of selecting a sibling pair -/
def sibling_pair_probability : ℚ := 75 / 1000000

/-- The number of students in the business school -/
def business_students : ℕ := 5000

theorem business_school_size :
  (sibling_pairs : ℚ) / (business_students * law_students) = sibling_pair_probability :=
by sorry

end NUMINAMATH_CALUDE_business_school_size_l1540_154092


namespace NUMINAMATH_CALUDE_sqrt_two_equality_l1540_154044

theorem sqrt_two_equality : (2 : ℝ) / Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_equality_l1540_154044


namespace NUMINAMATH_CALUDE_fish_upstream_speed_l1540_154077

/-- The upstream speed of a fish given its downstream speed and speed in still water -/
theorem fish_upstream_speed (downstream_speed still_water_speed : ℝ) :
  downstream_speed = 55 →
  still_water_speed = 45 →
  still_water_speed - (downstream_speed - still_water_speed) = 35 := by
  sorry

#check fish_upstream_speed

end NUMINAMATH_CALUDE_fish_upstream_speed_l1540_154077


namespace NUMINAMATH_CALUDE_computer_price_increase_l1540_154006

theorem computer_price_increase (b : ℝ) : 
  2 * b = 540 → (351 - b) / b * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l1540_154006


namespace NUMINAMATH_CALUDE_range_of_x_when_m_is_2_range_of_m_when_q_necessary_not_sufficient_l1540_154017

-- Define propositions p and q
def p (x m : ℝ) : Prop := x^2 - 5*m*x + 6*m^2 < 0

def q (x : ℝ) : Prop := (x - 5) / (x - 1) < 0

-- Theorem 1
theorem range_of_x_when_m_is_2 (x : ℝ) :
  (p x 2 ∨ q x) → 1 < x ∧ x < 6 := by sorry

-- Theorem 2
theorem range_of_m_when_q_necessary_not_sufficient (m : ℝ) :
  (m > 0 ∧ ∀ x, p x m → q x) ∧ (∃ x, q x ∧ ¬p x m) →
  1/2 ≤ m ∧ m ≤ 5/3 := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_m_is_2_range_of_m_when_q_necessary_not_sufficient_l1540_154017


namespace NUMINAMATH_CALUDE_fiftieth_term_is_448_l1540_154016

/-- Checks if a natural number contains the digit 4 --/
def containsDigitFour (n : ℕ) : Bool :=
  n.repr.any (· = '4')

/-- The sequence of positive multiples of 4 that contain at least one digit 4 --/
def specialSequence : ℕ → ℕ
  | 0 => 4  -- The first term is always 4
  | n + 1 => 
      let next := specialSequence n + 4
      if containsDigitFour next then next
      else specialSequence (n + 1)

/-- The 50th term of the special sequence is 448 --/
theorem fiftieth_term_is_448 : specialSequence 49 = 448 := by
  sorry

#eval specialSequence 49  -- This should output 448

end NUMINAMATH_CALUDE_fiftieth_term_is_448_l1540_154016


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l1540_154027

theorem greatest_prime_factor_of_sum (p : ℕ) :
  (∃ (q : ℕ), Nat.Prime q ∧ q ∣ (5^7 + 6^6) ∧ q ≥ p) →
  p ≤ 211 :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l1540_154027


namespace NUMINAMATH_CALUDE_trig_simplification_l1540_154094

theorem trig_simplification (α : ℝ) : 
  (1 + Real.sin (4 * α) - Real.cos (4 * α)) / (1 + Real.sin (4 * α) + Real.cos (4 * α)) = Real.tan (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l1540_154094


namespace NUMINAMATH_CALUDE_equation_solution_l1540_154088

theorem equation_solution : 
  ∃ t : ℝ, (1 / (t + 3) + 3 * t / (t + 3) - 4 / (t + 3) = 5) ∧ (t = -9) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1540_154088


namespace NUMINAMATH_CALUDE_intersection_line_slope_l1540_154058

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 12 = 0

-- Define the intersection points
def intersection_points (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- Theorem stating that the slope of the line connecting intersection points is 1
theorem intersection_line_slope :
  ∃ (x1 y1 x2 y2 : ℝ),
    intersection_points x1 y1 ∧
    intersection_points x2 y2 ∧
    x1 ≠ x2 →
    (y2 - y1) / (x2 - x1) = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l1540_154058


namespace NUMINAMATH_CALUDE_unique_outstanding_defeats_all_l1540_154031

/-- Represents a tournament with n participants. -/
structure Tournament (n : ℕ) where
  -- n ≥ 3
  n_ge_three : n ≥ 3
  -- Defeat relation
  defeats : Fin n → Fin n → Prop
  -- Every match has a definite winner
  winner_exists : ∀ i j : Fin n, i ≠ j → (defeats i j ∨ defeats j i) ∧ ¬(defeats i j ∧ defeats j i)

/-- Definition of an outstanding participant -/
def is_outstanding (t : Tournament n) (a : Fin n) : Prop :=
  ∀ b : Fin n, b ≠ a → t.defeats a b ∨ ∃ c : Fin n, t.defeats c b ∧ t.defeats a c

/-- The main theorem -/
theorem unique_outstanding_defeats_all (t : Tournament n) (a : Fin n) :
  (∀ b : Fin n, b ≠ a → is_outstanding t b → b = a) →
  is_outstanding t a →
  ∀ b : Fin n, b ≠ a → t.defeats a b :=
by sorry

end NUMINAMATH_CALUDE_unique_outstanding_defeats_all_l1540_154031


namespace NUMINAMATH_CALUDE_min_value_theorem_l1540_154045

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 8/n = 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 8/y = 4 → 8*m + n ≤ 8*x + y :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1540_154045


namespace NUMINAMATH_CALUDE_distinct_scores_count_l1540_154067

/-- Represents the possible scores for a basketball player who made 7 baskets,
    where each basket is worth either 2 or 3 points. -/
def basketball_scores : Finset ℕ :=
  Finset.image (fun x => x + 14) (Finset.range 8)

/-- The number of distinct possible scores for the basketball player. -/
theorem distinct_scores_count : basketball_scores.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_distinct_scores_count_l1540_154067


namespace NUMINAMATH_CALUDE_equation_solution_l1540_154015

theorem equation_solution :
  ∀ x : ℝ, (1 / x^2 + 2 / x = 5/4) ↔ (x = 2 ∨ x = -2/5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1540_154015


namespace NUMINAMATH_CALUDE_distance_between_points_l1540_154064

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (5, 10)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1540_154064


namespace NUMINAMATH_CALUDE_sin_cos_difference_l1540_154020

theorem sin_cos_difference (θ₁ θ₂ θ₃ θ₄ : Real) 
  (h₁ : θ₁ = 17 * π / 180)
  (h₂ : θ₂ = 47 * π / 180)
  (h₃ : θ₃ = 73 * π / 180)
  (h₄ : θ₄ = 43 * π / 180) : 
  Real.sin θ₁ * Real.cos θ₂ - Real.sin θ₃ * Real.cos θ₄ = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_l1540_154020


namespace NUMINAMATH_CALUDE_prob_same_length_in_hexagon_l1540_154023

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℕ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of elements in T -/
def total_elements : ℕ := num_sides + num_diagonals

/-- The probability of selecting two elements of the same length from T -/
def prob_same_length : ℚ := 17 / 35

theorem prob_same_length_in_hexagon :
  (num_sides * (num_sides - 1) + num_diagonals * (num_diagonals - 1)) / (total_elements * (total_elements - 1)) = prob_same_length := by
  sorry

end NUMINAMATH_CALUDE_prob_same_length_in_hexagon_l1540_154023


namespace NUMINAMATH_CALUDE_vector_dot_product_cos_2x_l1540_154056

theorem vector_dot_product_cos_2x (x : ℝ) : 
  let a := (Real.sqrt 3 * Real.sin x, Real.cos x)
  let b := (Real.cos x, -Real.cos x)
  x ∈ Set.Ioo (7 * Real.pi / 12) (5 * Real.pi / 6) →
  a.1 * b.1 + a.2 * b.2 = -5/4 →
  Real.cos (2 * x) = (3 - Real.sqrt 21) / 8 := by
    sorry

end NUMINAMATH_CALUDE_vector_dot_product_cos_2x_l1540_154056


namespace NUMINAMATH_CALUDE_least_number_of_cookies_cookies_solution_mohan_cookies_l1540_154063

theorem least_number_of_cookies (x : ℕ) : 
  (x % 6 = 5) ∧ (x % 9 = 3) ∧ (x % 11 = 7) → x ≥ 83 :=
by sorry

theorem cookies_solution : 
  (83 % 6 = 5) ∧ (83 % 9 = 3) ∧ (83 % 11 = 7) :=
by sorry

theorem mohan_cookies : 
  ∃ (x : ℕ), (x % 6 = 5) ∧ (x % 9 = 3) ∧ (x % 11 = 7) ∧ 
  (∀ (y : ℕ), (y % 6 = 5) ∧ (y % 9 = 3) ∧ (y % 11 = 7) → x ≤ y) ∧
  x = 83 :=
by sorry

end NUMINAMATH_CALUDE_least_number_of_cookies_cookies_solution_mohan_cookies_l1540_154063


namespace NUMINAMATH_CALUDE_twenty_paise_coins_count_l1540_154051

theorem twenty_paise_coins_count (total_coins : ℕ) (total_value : ℚ) :
  total_coins = 342 →
  total_value = 71 →
  ∃ (coins_20 coins_25 : ℕ),
    coins_20 + coins_25 = total_coins ∧
    (20 * coins_20 + 25 * coins_25 : ℚ) / 100 = total_value ∧
    coins_20 = 290 := by
  sorry

end NUMINAMATH_CALUDE_twenty_paise_coins_count_l1540_154051


namespace NUMINAMATH_CALUDE_line_equation_l1540_154036

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- A line passes through the origin if its y-intercept is 0 -/
def passes_through_origin (l : Line) : Prop :=
  l.y_intercept = 0

/-- A line has equal x and y intercepts if y_intercept = -slope * y_intercept -/
def equal_intercepts (l : Line) : Prop :=
  l.y_intercept = -l.slope * l.y_intercept

/-- The main theorem -/
theorem line_equation (l m : Line) :
  passes_through_origin l →
  parallel l m →
  equal_intercepts m →
  l.slope = -1 :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1540_154036


namespace NUMINAMATH_CALUDE_max_divisors_in_range_20_l1540_154024

def divisor_count (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def max_divisor_count (upper_bound : ℕ) : ℕ :=
  Finset.sup (Finset.range upper_bound.succ) divisor_count

theorem max_divisors_in_range_20 :
  max_divisor_count 20 = 6 ∧
  {12, 18, 20} = {n : ℕ | n ≤ 20 ∧ divisor_count n = max_divisor_count 20} :=
by sorry

end NUMINAMATH_CALUDE_max_divisors_in_range_20_l1540_154024


namespace NUMINAMATH_CALUDE_point_A_in_fourth_quadrant_l1540_154081

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point A -/
def point_A : Point :=
  { x := 5, y := -4 }

/-- Theorem: Point A is in the fourth quadrant -/
theorem point_A_in_fourth_quadrant : is_in_fourth_quadrant point_A := by
  sorry


end NUMINAMATH_CALUDE_point_A_in_fourth_quadrant_l1540_154081


namespace NUMINAMATH_CALUDE_field_trip_group_size_l1540_154007

/-- Calculates the number of students in each group excluding the student themselves -/
def students_per_group (total_bread : ℕ) (num_groups : ℕ) (sandwiches_per_student : ℕ) (bread_per_sandwich : ℕ) : ℕ :=
  (total_bread / (num_groups * sandwiches_per_student * bread_per_sandwich)) - 1

/-- Theorem: Given the specified conditions, there are 5 students in each group excluding the student themselves -/
theorem field_trip_group_size :
  students_per_group 120 5 2 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_group_size_l1540_154007


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1540_154026

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- The theorem states that the opposite of 2023 is -2023. -/
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1540_154026


namespace NUMINAMATH_CALUDE_calculate_withdrawal_l1540_154091

/-- Calculates the withdrawal amount given initial balance and transactions --/
theorem calculate_withdrawal 
  (initial_balance : ℕ) 
  (deposit_last_month : ℕ) 
  (deposit_this_month : ℕ) 
  (balance_increase : ℕ) 
  (h1 : initial_balance = 150)
  (h2 : deposit_last_month = 17)
  (h3 : deposit_this_month = 21)
  (h4 : balance_increase = 16) :
  ∃ (withdrawal : ℕ), 
    initial_balance + deposit_last_month - withdrawal + deposit_this_month 
    = initial_balance + balance_increase ∧ 
    withdrawal = 22 := by
sorry

end NUMINAMATH_CALUDE_calculate_withdrawal_l1540_154091


namespace NUMINAMATH_CALUDE_digit_difference_base_4_9_l1540_154057

theorem digit_difference_base_4_9 (n : ℕ) (h : n = 523) : 
  (Nat.log 4 n + 1) - (Nat.log 9 n + 1) = 2 := by sorry

end NUMINAMATH_CALUDE_digit_difference_base_4_9_l1540_154057


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1540_154098

theorem trigonometric_identity (α : Real) : 
  (Real.sin (45 * π / 180 + α))^2 - (Real.sin (30 * π / 180 - α))^2 - 
  Real.sin (15 * π / 180) * Real.cos (15 * π / 180 + 2 * α) = 
  Real.sin (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1540_154098


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l1540_154049

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_transitivity
  (m n : Line) (α β : Plane)
  (hm : m ≠ n) (hαβ : α ≠ β)
  (hmβ : perp m β) (hnβ : perp n β) (hnα : perp n α) :
  perp m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l1540_154049


namespace NUMINAMATH_CALUDE_parallel_vectors_l1540_154033

theorem parallel_vectors (a b : ℝ × ℝ) :
  a.1 = 2 ∧ a.2 = -1 ∧ b.2 = 3 ∧ a.1 * b.2 = a.2 * b.1 → b.1 = -6 :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_l1540_154033


namespace NUMINAMATH_CALUDE_y_satisfies_equation_l1540_154012

/-- The function y defined as the cube root of a quadratic expression -/
def y (x : ℝ) : ℝ := (2 + 3*x - 3*x^2)^(1/3)

/-- The statement that y satisfies the given differential equation -/
theorem y_satisfies_equation :
  ∀ x : ℝ, (y x) * (deriv y x) = (1 - 2*x) / (y x) := by
  sorry

end NUMINAMATH_CALUDE_y_satisfies_equation_l1540_154012


namespace NUMINAMATH_CALUDE_sum_of_f_92_and_neg_92_l1540_154028

/-- Given a polynomial function f(x) = ax^7 + bx^5 - cx^3 + dx + 3 where f(92) = 2,
    prove that f(92) + f(-92) = 6 -/
theorem sum_of_f_92_and_neg_92 (a b c d : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^7 + b * x^5 - c * x^3 + d * x + 3) 
  (h2 : f 92 = 2) : 
  f 92 + f (-92) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_92_and_neg_92_l1540_154028


namespace NUMINAMATH_CALUDE_tangent_directrix_parabola_circle_l1540_154074

/-- Given a circle and a parabola with a tangent directrix, prove the value of m -/
theorem tangent_directrix_parabola_circle (m : ℝ) : 
  m > 0 → 
  (∃ (x y : ℝ), x^2 + y^2 = 1/4) →
  (∃ (x y : ℝ), y = m * x^2) →
  (∃ (d : ℝ), d = 1/(4*m) ∧ d = 1/2) →
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_tangent_directrix_parabola_circle_l1540_154074


namespace NUMINAMATH_CALUDE_remaining_problems_to_grade_l1540_154014

theorem remaining_problems_to_grade 
  (problems_per_paper : ℕ) 
  (total_papers : ℕ) 
  (graded_papers : ℕ) 
  (h1 : problems_per_paper = 15)
  (h2 : total_papers = 45)
  (h3 : graded_papers = 18)
  : (total_papers - graded_papers) * problems_per_paper = 405 :=
by sorry

end NUMINAMATH_CALUDE_remaining_problems_to_grade_l1540_154014


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l1540_154070

theorem cone_volume_from_cylinder (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  let cylinder_volume := π * r^2 * h
  let cone_volume := (1/3) * π * r^2 * h
  cylinder_volume = 108 * π → cone_volume = 36 * π := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l1540_154070


namespace NUMINAMATH_CALUDE_smallest_marble_count_l1540_154084

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles in the urn -/
def totalMarbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green + mc.yellow

/-- Represents the probability of drawing a specific combination of marbles -/
def drawProbability (mc : MarbleCount) (r w b g y : ℕ) : ℚ :=
  (mc.red.choose r * mc.white.choose w * mc.blue.choose b * mc.green.choose g * mc.yellow.choose y : ℚ) /
  (totalMarbles mc).choose 4

/-- Checks if the four specified events are equally likely -/
def eventsEquallyLikely (mc : MarbleCount) : Prop :=
  drawProbability mc 4 0 0 0 0 = drawProbability mc 3 1 0 0 0 ∧
  drawProbability mc 4 0 0 0 0 = drawProbability mc 1 1 1 0 1 ∧
  drawProbability mc 4 0 0 0 0 = drawProbability mc 1 1 1 1 0

/-- The main theorem stating the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count : ∃ (mc : MarbleCount), 
  eventsEquallyLikely mc ∧ 
  totalMarbles mc = 11 ∧ 
  (∀ (mc' : MarbleCount), eventsEquallyLikely mc' → totalMarbles mc' ≥ totalMarbles mc) :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l1540_154084


namespace NUMINAMATH_CALUDE_max_xy_value_l1540_154000

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y + x + 2 * y = 4) :
  x * y ≤ 8 + 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_xy_value_l1540_154000


namespace NUMINAMATH_CALUDE_basketball_shots_l1540_154047

theorem basketball_shots (shots_made : ℝ) (shots_missed : ℝ) :
  shots_made = 0.8 * (shots_made + shots_missed) →
  shots_missed = 4 →
  shots_made + shots_missed = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_shots_l1540_154047


namespace NUMINAMATH_CALUDE_regular_soda_count_l1540_154052

/-- The number of bottles of regular soda in a grocery store -/
def regular_soda_bottles : ℕ := 83

/-- The number of bottles of diet soda in the grocery store -/
def diet_soda_bottles : ℕ := 4

/-- The difference between the number of regular soda bottles and diet soda bottles -/
def soda_difference : ℕ := 79

theorem regular_soda_count :
  regular_soda_bottles = diet_soda_bottles + soda_difference := by
  sorry

end NUMINAMATH_CALUDE_regular_soda_count_l1540_154052


namespace NUMINAMATH_CALUDE_product_sum_theorem_l1540_154030

theorem product_sum_theorem (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (6 - a) * (6 - b) * (6 - c) * (6 - d) * (6 - e) = 120 →
  a + b + c + d + e = 27 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l1540_154030


namespace NUMINAMATH_CALUDE_trout_weight_fishing_scenario_l1540_154061

/-- Calculates the weight of trout caught given the fishing conditions -/
theorem trout_weight (num_campers : ℕ) (fish_per_camper : ℕ) 
                     (num_bass : ℕ) (bass_weight : ℕ) 
                     (num_salmon : ℕ) (salmon_weight : ℕ) : ℕ :=
  let total_fish_needed := num_campers * fish_per_camper
  let total_bass_weight := num_bass * bass_weight
  let total_salmon_weight := num_salmon * salmon_weight
  total_fish_needed - (total_bass_weight + total_salmon_weight)

/-- The specific fishing scenario described in the problem -/
theorem fishing_scenario : trout_weight 22 2 6 2 2 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_trout_weight_fishing_scenario_l1540_154061


namespace NUMINAMATH_CALUDE_first_grade_allocation_l1540_154097

theorem first_grade_allocation (total : ℕ) (ratio_first : ℕ) (ratio_second : ℕ) (ratio_third : ℕ) 
  (h_total : total = 160)
  (h_ratio : ratio_first = 6 ∧ ratio_second = 5 ∧ ratio_third = 5) :
  (total * ratio_first) / (ratio_first + ratio_second + ratio_third) = 60 := by
  sorry

end NUMINAMATH_CALUDE_first_grade_allocation_l1540_154097


namespace NUMINAMATH_CALUDE_kit_price_difference_l1540_154068

-- Define the prices
def kit_price : ℚ := 145.75
def filter_price_1 : ℚ := 9.50
def filter_price_2 : ℚ := 15.30
def filter_price_3 : ℚ := 20.75
def filter_price_4 : ℚ := 25.80

-- Define the quantities
def quantity_1 : ℕ := 3
def quantity_2 : ℕ := 2
def quantity_3 : ℕ := 1
def quantity_4 : ℕ := 2

-- Calculate the total price of individual filters
def total_individual_price : ℚ :=
  filter_price_1 * quantity_1 +
  filter_price_2 * quantity_2 +
  filter_price_3 * quantity_3 +
  filter_price_4 * quantity_4

-- Define the theorem
theorem kit_price_difference :
  kit_price - total_individual_price = 14.30 := by
  sorry

end NUMINAMATH_CALUDE_kit_price_difference_l1540_154068


namespace NUMINAMATH_CALUDE_cone_base_circumference_l1540_154090

/-- The circumference of the base of a right circular cone formed from a 180° sector of a circle --/
theorem cone_base_circumference (r : ℝ) (h : r = 5) :
  let full_circle_circumference := 2 * π * r
  let sector_angle := π  -- 180° in radians
  let full_angle := 2 * π  -- 360° in radians
  let base_circumference := (sector_angle / full_angle) * full_circle_circumference
  base_circumference = 5 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l1540_154090


namespace NUMINAMATH_CALUDE_fixed_point_unique_l1540_154085

/-- The line l passes through the point (x, y) for all real values of m -/
def passes_through (x y : ℝ) : Prop :=
  ∀ m : ℝ, (2 + m) * x + (1 - 2*m) * y + (4 - 3*m) = 0

/-- The point M is the unique point that the line l passes through for all m -/
theorem fixed_point_unique :
  ∃! p : ℝ × ℝ, passes_through p.1 p.2 ∧ p = (-1, -2) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_unique_l1540_154085


namespace NUMINAMATH_CALUDE_sum_of_absolute_differences_l1540_154032

theorem sum_of_absolute_differences (a b c : ℤ) 
  (h : (a - b)^10 + (a - c)^10 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_differences_l1540_154032


namespace NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l1540_154013

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_with_twelve_divisors : 
  ∀ n : ℕ, n > 0 → divisor_count n = 12 → n ≥ 96 :=
by sorry

end NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l1540_154013


namespace NUMINAMATH_CALUDE_function_maximum_implies_a_range_l1540_154086

/-- Given a function f(x) = 4x³ - 3x with a maximum in the interval (a, a+2), prove that a is in the range (-5/2, -1]. -/
theorem function_maximum_implies_a_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h₁ : ∀ x, f x = 4 * x^3 - 3 * x)
  (h₂ : ∃ x₀ ∈ Set.Ioo a (a + 2), ∀ x ∈ Set.Ioo a (a + 2), f x ≤ f x₀) :
  a ∈ Set.Ioc (-5/2) (-1) :=
sorry

end NUMINAMATH_CALUDE_function_maximum_implies_a_range_l1540_154086


namespace NUMINAMATH_CALUDE_cube_sphere_volume_l1540_154048

theorem cube_sphere_volume (cube_surface_area : ℝ) (h_surface_area : cube_surface_area = 18) :
  let cube_edge := Real.sqrt (cube_surface_area / 6)
  let sphere_radius := (Real.sqrt 3 * cube_edge) / 2
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = 9 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sphere_volume_l1540_154048


namespace NUMINAMATH_CALUDE_three_pairs_square_product_400_l1540_154025

/-- The number of pairs of positive integers whose squares multiply to 400 -/
def count_pairs : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 * p.2^2 = 400)
    (Finset.product (Finset.range 21) (Finset.range 21))).card

/-- Theorem stating that there are exactly 3 pairs of positive integers
    whose squares multiply to 400 -/
theorem three_pairs_square_product_400 :
  count_pairs = 3 := by sorry

end NUMINAMATH_CALUDE_three_pairs_square_product_400_l1540_154025


namespace NUMINAMATH_CALUDE_coffee_stock_solution_l1540_154065

/-- Represents the coffee stock problem --/
def coffee_stock_problem (initial_stock : ℝ) (initial_decaf_percent : ℝ) 
  (new_batch_decaf_percent : ℝ) (final_decaf_percent : ℝ) (new_batch : ℝ) : Prop :=
  let initial_decaf := initial_stock * initial_decaf_percent
  let new_batch_decaf := new_batch * new_batch_decaf_percent
  let total_stock := initial_stock + new_batch
  let total_decaf := initial_decaf + new_batch_decaf
  (total_decaf / total_stock) = final_decaf_percent

/-- Theorem stating the solution to the coffee stock problem --/
theorem coffee_stock_solution :
  coffee_stock_problem 400 0.25 0.60 0.32 100 := by
  sorry

#check coffee_stock_solution

end NUMINAMATH_CALUDE_coffee_stock_solution_l1540_154065


namespace NUMINAMATH_CALUDE_directrix_of_hyperbola_l1540_154050

/-- The directrix of the hyperbola xy = 1 -/
def directrix_equation (x y : ℝ) : Prop :=
  y = -x + Real.sqrt 2 ∨ y = -x - Real.sqrt 2

/-- The hyperbola equation xy = 1 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x * y = 1

/-- Theorem stating that the directrix of the hyperbola xy = 1 has the equation y = -x ± √2 -/
theorem directrix_of_hyperbola (x y : ℝ) :
  hyperbola_equation x y → directrix_equation x y :=
sorry

end NUMINAMATH_CALUDE_directrix_of_hyperbola_l1540_154050


namespace NUMINAMATH_CALUDE_solution_of_equation_l1540_154095

theorem solution_of_equation (x : ℝ) : (1 / (3 * x) = 2 / (x + 5)) ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l1540_154095


namespace NUMINAMATH_CALUDE_ternary_1021_is_34_l1540_154021

def ternary_to_decimal (t : List Nat) : Nat :=
  List.foldl (fun acc d => acc * 3 + d) 0 t.reverse

theorem ternary_1021_is_34 :
  ternary_to_decimal [1, 0, 2, 1] = 34 := by
  sorry

end NUMINAMATH_CALUDE_ternary_1021_is_34_l1540_154021


namespace NUMINAMATH_CALUDE_linear_function_proof_l1540_154041

/-- A linear function of the form y = kx - 3 -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x - 3

/-- The k value for which the linear function passes through (1, 7) -/
def k : ℝ := 10

theorem linear_function_proof :
  (linear_function k 1 = 7) ∧
  (linear_function k 2 ≠ 15) := by
  sorry

#check linear_function_proof

end NUMINAMATH_CALUDE_linear_function_proof_l1540_154041


namespace NUMINAMATH_CALUDE_fraction_equality_l1540_154018

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 3 / 5) 
  (h2 : s / u = 8 / 11) : 
  (4 * p * s - 3 * q * u) / (5 * q * u - 8 * p * s) = -69 / 83 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1540_154018


namespace NUMINAMATH_CALUDE_max_digit_sum_is_24_l1540_154069

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  h_range : hours ≤ 23
  m_range : minutes ≤ 59

/-- Calculates the sum of digits for a natural number -/
def sumDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumDigits (n / 10)

/-- Calculates the sum of all digits in a Time24 -/
def totalDigitSum (t : Time24) : Nat :=
  sumDigits t.hours + sumDigits t.minutes

/-- The maximum sum of digits possible in a 24-hour format display -/
def maxDigitSum : Nat := 24

/-- Theorem stating that the maximum sum of digits in a 24-hour format display is 24 -/
theorem max_digit_sum_is_24 : 
  ∀ t : Time24, totalDigitSum t ≤ maxDigitSum :=
by sorry

end NUMINAMATH_CALUDE_max_digit_sum_is_24_l1540_154069


namespace NUMINAMATH_CALUDE_point_c_satisfies_inequalities_l1540_154011

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point satisfies the given system of inequalities -/
def satisfiesInequalities (p : Point2D) : Prop :=
  p.x + p.y - 1 < 0 ∧ p.x - p.y + 1 > 0

theorem point_c_satisfies_inequalities :
  satisfiesInequalities ⟨0, -2⟩ ∧
  ¬satisfiesInequalities ⟨0, 2⟩ ∧
  ¬satisfiesInequalities ⟨-2, 0⟩ ∧
  ¬satisfiesInequalities ⟨2, 0⟩ := by
  sorry


end NUMINAMATH_CALUDE_point_c_satisfies_inequalities_l1540_154011


namespace NUMINAMATH_CALUDE_total_balloons_count_l1540_154038

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 60

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := 85

/-- The number of blue balloons Alex has -/
def alex_balloons : ℕ := 37

/-- The total number of blue balloons -/
def total_balloons : ℕ := joan_balloons + melanie_balloons + alex_balloons

theorem total_balloons_count : total_balloons = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_count_l1540_154038


namespace NUMINAMATH_CALUDE_fraction_equality_l1540_154046

theorem fraction_equality (x : ℝ) : 
  (3/4 : ℝ) * (1/2 : ℝ) * (2/5 : ℝ) * x = 750.0000000000001 → x = 5000.000000000001 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1540_154046


namespace NUMINAMATH_CALUDE_polynomial_identity_l1540_154071

theorem polynomial_identity (a b c : ℝ) : 
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 + 3 * a * b * c * (a - b) * (b - c) * (c - a) = 
  (a - b) * (b - c) * (c - a) * (a + b + c + 3 * a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1540_154071


namespace NUMINAMATH_CALUDE_no_valid_k_exists_l1540_154005

/-- The nth odd prime number -/
def nthOddPrime (n : ℕ) : ℕ := sorry

/-- The product of the first n odd prime numbers -/
def productFirstNOddPrimes (n : ℕ) : ℕ := sorry

/-- Statement: There does not exist a natural number k such that the product 
    of the first k odd prime numbers, decreased by 1, is an exact power 
    of a natural number greater than 1 -/
theorem no_valid_k_exists : 
  ¬ ∃ (k : ℕ), ∃ (a n : ℕ), n > 1 ∧ productFirstNOddPrimes k - 1 = a^n := by
  sorry


end NUMINAMATH_CALUDE_no_valid_k_exists_l1540_154005


namespace NUMINAMATH_CALUDE_dubblefud_yellow_count_l1540_154089

/-- The game of dubblefud with yellow, blue, and green chips -/
def dubblefud (yellow blue green : ℕ) : Prop :=
  2^yellow * 4^blue * 5^green = 16000 ∧ blue = green

theorem dubblefud_yellow_count :
  ∀ y b g : ℕ, dubblefud y b g → y = 1 :=
by sorry

end NUMINAMATH_CALUDE_dubblefud_yellow_count_l1540_154089


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l1540_154035

theorem sum_of_roots_cubic_equation : 
  let f : ℝ → ℝ := fun x ↦ 3 * x^3 - 9 * x^2 - 72 * x + 6
  ∃ r p q : ℝ, (∀ x : ℝ, f x = 0 ↔ x = r ∨ x = p ∨ x = q) ∧ r + p + q = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l1540_154035


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1540_154053

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 4 + a 10 + a 16 = 30) : 
  a 18 - 2 * a 14 = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1540_154053


namespace NUMINAMATH_CALUDE_machine_output_2023_l1540_154029

/-- A function that computes the output of Ava's machine for a four-digit number -/
def machine_output (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a * b + c * d

/-- Theorem stating that the machine output for 2023 is 6 -/
theorem machine_output_2023 : machine_output 2023 = 6 := by
  sorry

end NUMINAMATH_CALUDE_machine_output_2023_l1540_154029


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1540_154066

theorem geometric_series_sum (x : ℝ) :
  (|x| < 1) →
  (∑' n, x^n = 4) →
  x = 3/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1540_154066


namespace NUMINAMATH_CALUDE_ellipse_focus_d_value_l1540_154019

/-- Definition of an ellipse with given properties -/
structure Ellipse where
  /-- The ellipse is in the first quadrant -/
  first_quadrant : Bool
  /-- The ellipse is tangent to both x-axis and y-axis -/
  tangent_to_axes : Bool
  /-- One focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The other focus of the ellipse -/
  focus2 : ℝ × ℝ

/-- Theorem stating the value of d for the given ellipse -/
theorem ellipse_focus_d_value (e : Ellipse) : 
  e.first_quadrant = true ∧ 
  e.tangent_to_axes = true ∧ 
  e.focus1 = (4, 10) ∧ 
  e.focus2.1 = e.focus2.2 ∧ 
  e.focus2.2 = 10 → 
  e.focus2.1 = 25 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focus_d_value_l1540_154019


namespace NUMINAMATH_CALUDE_sales_volume_correct_profit_10000_prices_max_profit_under_constraints_l1540_154039

/-- Toy sales model -/
structure ToySalesModel where
  purchase_price : ℝ
  initial_price : ℝ
  initial_volume : ℝ
  price_sensitivity : ℝ
  min_price : ℝ
  min_volume : ℝ

/-- Given toy sales model -/
def given_model : ToySalesModel :=
  { purchase_price := 30
  , initial_price := 40
  , initial_volume := 600
  , price_sensitivity := 10
  , min_price := 44
  , min_volume := 540 }

/-- Sales volume as a function of price -/
def sales_volume (model : ToySalesModel) (x : ℝ) : ℝ :=
  model.initial_volume - model.price_sensitivity * (x - model.initial_price)

/-- Profit as a function of price -/
def profit (model : ToySalesModel) (x : ℝ) : ℝ :=
  (x - model.purchase_price) * (sales_volume model x)

/-- Theorem stating the correctness of the sales volume function -/
theorem sales_volume_correct (x : ℝ) (h : x > given_model.initial_price) :
  sales_volume given_model x = 1000 - 10 * x := by sorry

/-- Theorem stating the selling prices for a profit of 10,000 yuan -/
theorem profit_10000_prices :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ profit given_model x₁ = 10000 ∧ profit given_model x₂ = 10000 ∧
  (x₁ = 50 ∨ x₁ = 80) ∧ (x₂ = 50 ∨ x₂ = 80) := by sorry

/-- Theorem stating the maximum profit under constraints -/
theorem max_profit_under_constraints :
  ∃ x : ℝ, x ≥ given_model.min_price ∧ 
    sales_volume given_model x ≥ given_model.min_volume ∧
    ∀ y : ℝ, y ≥ given_model.min_price → 
      sales_volume given_model y ≥ given_model.min_volume →
      profit given_model x ≥ profit given_model y ∧
      profit given_model x = 8640 := by sorry

end NUMINAMATH_CALUDE_sales_volume_correct_profit_10000_prices_max_profit_under_constraints_l1540_154039


namespace NUMINAMATH_CALUDE_complement_union_equals_set_l1540_154055

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {2,4,5}
def B : Set Nat := {1,2,5}

theorem complement_union_equals_set : 
  (U \ (A ∪ B)) = {3,6} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_set_l1540_154055


namespace NUMINAMATH_CALUDE_positive_xy_l1540_154076

theorem positive_xy (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ -2) : x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_xy_l1540_154076


namespace NUMINAMATH_CALUDE_proposition_equivalence_l1540_154054

theorem proposition_equivalence (p q : Prop) : 
  (¬(p ∨ q)) → ((¬p) ∧ (¬q)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l1540_154054


namespace NUMINAMATH_CALUDE_number_wall_problem_l1540_154003

/-- Represents a row in the Number Wall -/
structure NumberWallRow :=
  (a b c d : ℕ)

/-- Calculates the next row in the Number Wall -/
def nextRow (row : NumberWallRow) : NumberWallRow :=
  ⟨row.a + row.b, row.b + row.c, row.c + row.d, 0⟩

/-- The Number Wall problem -/
theorem number_wall_problem (m : ℕ) : m = 2 :=
  let row1 := NumberWallRow.mk m 5 9 6
  let row2 := nextRow row1
  let row3 := nextRow row2
  let row4 := nextRow row3
  have h1 : row2.c = 18 := by sorry
  have h2 : row4.a = 55 := by sorry
  sorry

end NUMINAMATH_CALUDE_number_wall_problem_l1540_154003


namespace NUMINAMATH_CALUDE_inverse_function_property_l1540_154080

-- Define a function f and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Given condition: the graph of y = x - f(x) passes through (1, 2)
axiom condition : 1 - f 1 = 2

-- Theorem to prove
theorem inverse_function_property : f_inv (-1) - (-1) = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_function_property_l1540_154080


namespace NUMINAMATH_CALUDE_remainder_theorem_l1540_154022

theorem remainder_theorem (n : ℤ) (h : n % 5 = 3) : (7 * n + 4) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1540_154022


namespace NUMINAMATH_CALUDE_inequality_proof_l1540_154082

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a^4 + b^4 + c^4 = 3) :
  1 / (4 - a*b) + 1 / (4 - b*c) + 1 / (4 - c*a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1540_154082


namespace NUMINAMATH_CALUDE_triangle_with_ap_angles_and_altitudes_is_equilateral_l1540_154059

/-- A triangle with angles and altitudes in arithmetic progression is equilateral -/
theorem triangle_with_ap_angles_and_altitudes_is_equilateral 
  (A B C : ℝ) (a b c : ℝ) (ha hb hc : ℝ) : 
  (∃ (d : ℝ), A = B - d ∧ C = B + d) →  -- Angles in arithmetic progression
  (A + B + C = 180) →                   -- Sum of angles in a triangle
  (ha + hc = 2 * hb) →                  -- Altitudes in arithmetic progression
  (ha = 2 * area / a) →                 -- Relation between altitude and side
  (hb = 2 * area / b) → 
  (hc = 2 * area / c) → 
  (b^2 = a^2 + c^2 - a*c) →             -- Law of cosines for 60° angle
  (a = b ∧ b = c) :=                    -- Triangle is equilateral
by sorry

end NUMINAMATH_CALUDE_triangle_with_ap_angles_and_altitudes_is_equilateral_l1540_154059


namespace NUMINAMATH_CALUDE_exists_pair_satisfying_condition_l1540_154062

theorem exists_pair_satisfying_condition (r : Fin 5 → ℝ) : 
  ∃ (i j : Fin 5), i ≠ j ∧ 0 ≤ (r i - r j) / (1 + r i * r j) ∧ (r i - r j) / (1 + r i * r j) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_pair_satisfying_condition_l1540_154062
