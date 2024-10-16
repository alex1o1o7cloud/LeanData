import Mathlib

namespace NUMINAMATH_CALUDE_triangle_properties_l3138_313813

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.b * Real.sin abc.A = Real.sqrt 3 * abc.a * Real.cos abc.B) 
  (h2 : abc.b = 2 * Real.sqrt 3) : 
  abc.B = π / 3 ∧ 
  (∃ (S : ℝ), S = 3 * Real.sqrt 3 ∧ ∀ (area : ℝ), area ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3138_313813


namespace NUMINAMATH_CALUDE_opposite_face_is_A_l3138_313809

/-- Represents the labels of the squares --/
inductive Label
  | A | B | C | D | E | F

/-- Represents a cube formed by folding six squares --/
structure Cube where
  top : Label
  bottom : Label
  front : Label
  back : Label
  left : Label
  right : Label

/-- Represents the linear arrangement of squares before folding --/
def LinearArrangement := List Label

/-- Function to create a cube from a linear arrangement of squares --/
def foldCube (arrangement : LinearArrangement) (top : Label) : Cube :=
  sorry

/-- The theorem to be proved --/
theorem opposite_face_is_A 
  (arrangement : LinearArrangement) 
  (h1 : arrangement = [Label.A, Label.B, Label.C, Label.D, Label.E, Label.F]) 
  (cube : Cube) 
  (h2 : cube = foldCube arrangement Label.B) : 
  cube.bottom = Label.A :=
sorry

end NUMINAMATH_CALUDE_opposite_face_is_A_l3138_313809


namespace NUMINAMATH_CALUDE_toby_girl_friends_l3138_313805

/-- Represents the number of Toby's friends in each category -/
structure FriendCounts where
  boys : ℕ
  girls : ℕ
  imaginary : ℕ

/-- Represents the percentages of Toby's friends in each category -/
structure FriendPercentages where
  boys : ℚ
  girls : ℚ
  imaginary : ℚ

/-- Given the conditions of the problem, prove that Toby has 21 girl friends -/
theorem toby_girl_friends 
  (percentages : FriendPercentages)
  (counts : FriendCounts)
  (h1 : percentages.boys = 55 / 100)
  (h2 : percentages.girls = 35 / 100)
  (h3 : percentages.imaginary = 10 / 100)
  (h4 : percentages.boys + percentages.girls + percentages.imaginary = 1)
  (h5 : counts.boys = 33)
  : counts.girls = 21 := by
  sorry


end NUMINAMATH_CALUDE_toby_girl_friends_l3138_313805


namespace NUMINAMATH_CALUDE_dave_initial_apps_l3138_313884

/-- The number of apps Dave had initially -/
def initial_apps : ℕ := sorry

/-- The number of apps Dave had after adding one -/
def apps_after_adding : ℕ := 18

theorem dave_initial_apps : 
  initial_apps = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_dave_initial_apps_l3138_313884


namespace NUMINAMATH_CALUDE_downstream_speed_is_48_l3138_313855

/-- The speed of a man rowing in a stream -/
structure RowingSpeed :=
  (upstream : ℝ)
  (stillWater : ℝ)

/-- Calculate the downstream speed of a man rowing in a stream -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  s.stillWater + (s.stillWater - s.upstream)

/-- Theorem: Given the upstream and still water speeds, the downstream speed is 48 -/
theorem downstream_speed_is_48 (s : RowingSpeed) 
    (h1 : s.upstream = 34) 
    (h2 : s.stillWater = 41) : 
  downstreamSpeed s = 48 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_is_48_l3138_313855


namespace NUMINAMATH_CALUDE_sum_over_subsets_equals_power_of_two_l3138_313880

def S : Finset ℕ := Finset.range 1999

def f (T : Finset ℕ) : ℕ := T.sum id

theorem sum_over_subsets_equals_power_of_two :
  (Finset.powerset S).sum (fun E => (f E : ℚ) / (f S : ℚ)) = 2^1998 := by sorry

end NUMINAMATH_CALUDE_sum_over_subsets_equals_power_of_two_l3138_313880


namespace NUMINAMATH_CALUDE_train_length_l3138_313856

/-- Given a train passing a bridge, calculate its length. -/
theorem train_length
  (train_speed : Real) -- Speed of the train in km/hour
  (bridge_length : Real) -- Length of the bridge in meters
  (passing_time : Real) -- Time to pass the bridge in seconds
  (h1 : train_speed = 45) -- Train speed is 45 km/hour
  (h2 : bridge_length = 160) -- Bridge length is 160 meters
  (h3 : passing_time = 41.6) -- Time to pass the bridge is 41.6 seconds
  : Real := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3138_313856


namespace NUMINAMATH_CALUDE_quadratic_properties_l3138_313872

def f (x : ℝ) := x^2 - 4*x + 6

theorem quadratic_properties :
  (∀ x : ℝ, f x = 2 ↔ x = 2) ∧
  (∀ x y : ℝ, x > 2 ∧ y > x → f y > f x) ∧
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m) ∧
  (∀ x : ℝ, f x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3138_313872


namespace NUMINAMATH_CALUDE_largest_x_for_prime_expression_l3138_313812

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem largest_x_for_prime_expression :
  ∀ x : ℤ, x > 2 → ¬(is_prime (|4 * x^2 - 41 * x + 21|.toNat)) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_for_prime_expression_l3138_313812


namespace NUMINAMATH_CALUDE_train_journey_time_l3138_313839

/-- Calculate the total travel time for a train journey with multiple stops and varying speeds -/
theorem train_journey_time (d1 d2 d3 : ℝ) (v1 v2 v3 : ℝ) (t1 t2 : ℝ) :
  d1 = 30 →
  d2 = 40 →
  d3 = 50 →
  v1 = 60 →
  v2 = 40 →
  v3 = 80 →
  t1 = 10 / 60 →
  t2 = 5 / 60 →
  (d1 / v1 + t1 + d2 / v2 + t2 + d3 / v3) * 60 = 142.5 :=
by sorry

end NUMINAMATH_CALUDE_train_journey_time_l3138_313839


namespace NUMINAMATH_CALUDE_circles_intersection_l3138_313804

/-- Two circles intersect if and only if m is between 9 and 49 -/
theorem circles_intersection (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = m ∧ x^2 + y^2 + 6*x - 8*y + 21 = 0) ↔ 9 < m ∧ m < 49 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersection_l3138_313804


namespace NUMINAMATH_CALUDE_dog_food_total_l3138_313825

/-- Theorem: Given an initial amount of dog food and two additional purchases,
    prove the total amount of dog food. -/
theorem dog_food_total (initial : ℕ) (bag1 : ℕ) (bag2 : ℕ) :
  initial = 15 → bag1 = 15 → bag2 = 10 → initial + bag1 + bag2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_total_l3138_313825


namespace NUMINAMATH_CALUDE_probability_second_white_given_first_white_l3138_313861

/-- The probability of drawing a white ball second, given that the first ball drawn is white,
    when there are 5 white balls and 4 black balls initially. -/
theorem probability_second_white_given_first_white :
  let total_balls : ℕ := 9
  let white_balls : ℕ := 5
  let black_balls : ℕ := 4
  let prob_first_white : ℚ := white_balls / total_balls
  let prob_both_white : ℚ := (white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1))
  prob_both_white / prob_first_white = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_second_white_given_first_white_l3138_313861


namespace NUMINAMATH_CALUDE_coin_value_difference_l3138_313836

theorem coin_value_difference :
  ∀ (x : ℕ),
  1 ≤ x ∧ x ≤ 3029 →
  (30300 - 9 * 1) - (30300 - 9 * 3029) = 27252 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_value_difference_l3138_313836


namespace NUMINAMATH_CALUDE_I_max_min_zero_l3138_313831

noncomputable def f (x : ℝ) : ℝ := x^2 + 3

noncomputable def g (a x : ℝ) : ℝ := a * x + 3

noncomputable def I (a : ℝ) : ℝ := 3 * ∫ x in (-1)..(1), |f x - g a x|

theorem I_max_min_zero :
  (∀ a : ℝ, I a ≤ 0) ∧ (∃ a : ℝ, I a = 0) :=
sorry

end NUMINAMATH_CALUDE_I_max_min_zero_l3138_313831


namespace NUMINAMATH_CALUDE_manuscript_year_count_l3138_313853

/-- The number of possible 6-digit years formed from the digits 2, 2, 2, 2, 3, and 9,
    where the year must begin with an odd digit -/
def manuscript_year_possibilities : ℕ :=
  let total_digits : ℕ := 6
  let repeated_digit_count : ℕ := 4
  let odd_digit_choices : ℕ := 2
  odd_digit_choices * (Nat.factorial total_digits) / (Nat.factorial repeated_digit_count)

theorem manuscript_year_count : manuscript_year_possibilities = 60 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_year_count_l3138_313853


namespace NUMINAMATH_CALUDE_odd_function_minimum_value_l3138_313886

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- F is defined as a linear combination of f and x, plus a constant -/
def F (f : ℝ → ℝ) (a b : ℝ) (x : ℝ) : ℝ :=
  a * f x + b * x + 1

theorem odd_function_minimum_value
    (f : ℝ → ℝ) (a b : ℝ)
    (h_odd : IsOdd f)
    (h_max : ∀ x > 0, F f a b x ≤ 2) :
    ∀ x < 0, F f a b x ≥ 0 :=
  sorry

end NUMINAMATH_CALUDE_odd_function_minimum_value_l3138_313886


namespace NUMINAMATH_CALUDE_remainder_theorem_l3138_313808

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 60 * k - 3) :
  (n^3 + 2*n^2 + 3*n + 4) % 60 = 46 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3138_313808


namespace NUMINAMATH_CALUDE_probability_of_shaded_triangle_l3138_313811

/-- A triangle in the diagram -/
structure Triangle where
  label : String

/-- The set of all triangles in the diagram -/
def all_triangles : Finset Triangle := sorry

/-- The set of shaded triangles -/
def shaded_triangles : Finset Triangle := sorry

/-- Each triangle has the same probability of being selected -/
axiom equal_probability : ∀ t : Triangle, t ∈ all_triangles → 
  (Finset.card {t} : ℚ) / (Finset.card all_triangles : ℚ) = 1 / (Finset.card all_triangles : ℚ)

theorem probability_of_shaded_triangle :
  (Finset.card shaded_triangles : ℚ) / (Finset.card all_triangles : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_shaded_triangle_l3138_313811


namespace NUMINAMATH_CALUDE_hair_cut_length_l3138_313891

/-- The length of hair cut off is equal to the difference between the initial and final hair lengths. -/
theorem hair_cut_length (initial_length final_length cut_length : ℝ) 
  (h1 : initial_length = 11)
  (h2 : final_length = 7)
  (h3 : cut_length = initial_length - final_length) :
  cut_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_length_l3138_313891


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l3138_313833

theorem weekend_rain_probability
  (p_friday : ℝ)
  (p_saturday_given_friday : ℝ)
  (p_saturday_given_not_friday : ℝ)
  (p_sunday : ℝ)
  (h1 : p_friday = 0.3)
  (h2 : p_saturday_given_friday = 0.6)
  (h3 : p_saturday_given_not_friday = 0.25)
  (h4 : p_sunday = 0.4) :
  1 - (1 - p_friday) * (1 - p_saturday_given_not_friday * (1 - p_friday)) * (1 - p_sunday) = 0.685 := by
sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l3138_313833


namespace NUMINAMATH_CALUDE_sum_of_digits_8_pow_2004_l3138_313879

/-- The sum of the tens digit and the units digit of 8^2004 in its decimal representation -/
def sum_of_digits : ℕ :=
  let n := 8^2004
  let tens_digit := (n / 10) % 10
  let units_digit := n % 10
  tens_digit + units_digit

/-- Theorem stating that the sum of the tens digit and the units digit of 8^2004 is 7 -/
theorem sum_of_digits_8_pow_2004 : sum_of_digits = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_8_pow_2004_l3138_313879


namespace NUMINAMATH_CALUDE_max_product_of_digits_l3138_313847

theorem max_product_of_digits (A B : ℕ) : 
  (A ≤ 9) → 
  (B ≤ 9) → 
  (∃ (n : ℕ), A * 100000 + 2021 * 10 + B = 9 * n) →
  A * B ≤ 42 :=
sorry

end NUMINAMATH_CALUDE_max_product_of_digits_l3138_313847


namespace NUMINAMATH_CALUDE_lowest_sale_price_percentage_l3138_313866

theorem lowest_sale_price_percentage (list_price : ℝ) (max_regular_discount : ℝ) (additional_discount : ℝ) : 
  list_price = 80 ∧ 
  max_regular_discount = 0.5 ∧ 
  additional_discount = 0.2 → 
  (list_price * (1 - max_regular_discount) - list_price * additional_discount) / list_price = 0.3 := by
sorry

end NUMINAMATH_CALUDE_lowest_sale_price_percentage_l3138_313866


namespace NUMINAMATH_CALUDE_base3_11111_is_121_l3138_313814

def base3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem base3_11111_is_121 :
  base3_to_decimal [1, 1, 1, 1, 1] = 121 := by
  sorry

end NUMINAMATH_CALUDE_base3_11111_is_121_l3138_313814


namespace NUMINAMATH_CALUDE_jakes_earnings_l3138_313834

/-- Jake's lawn mowing and flower planting problem -/
theorem jakes_earnings (mowing_time mowing_pay planting_time desired_rate : ℝ) 
  (h1 : mowing_time = 1)
  (h2 : mowing_pay = 15)
  (h3 : planting_time = 2)
  (h4 : desired_rate = 20) :
  let total_time := mowing_time + planting_time
  let total_desired_earnings := desired_rate * total_time
  let planting_charge := total_desired_earnings - mowing_pay
  planting_charge = 45 := by sorry

end NUMINAMATH_CALUDE_jakes_earnings_l3138_313834


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3138_313877

/-- The complex number z = (-2-3i)/i is in the second quadrant of the complex plane -/
theorem complex_number_in_second_quadrant : 
  let z : ℂ := (-2 - 3*Complex.I) / Complex.I
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3138_313877


namespace NUMINAMATH_CALUDE_procedure_cost_l3138_313849

theorem procedure_cost (insurance_coverage : Real) (amount_saved : Real) :
  insurance_coverage = 0.80 →
  amount_saved = 3520 →
  ∃ (cost : Real), cost = 4400 ∧ insurance_coverage * cost = amount_saved :=
by sorry

end NUMINAMATH_CALUDE_procedure_cost_l3138_313849


namespace NUMINAMATH_CALUDE_sin_theta_value_l3138_313821

theorem sin_theta_value (θ : Real) 
  (h1 : Real.sin θ + Real.cos θ = 7/5)
  (h2 : Real.tan θ < 1) : 
  Real.sin θ = 3/5 := by
sorry

end NUMINAMATH_CALUDE_sin_theta_value_l3138_313821


namespace NUMINAMATH_CALUDE_correct_proposition_l3138_313810

-- Define the parallel relation
def parallel (x y : Type) : Prop := sorry

-- Define the intersection of two planes
def intersection (α β : Type) : Type := sorry

-- Define proposition p
def p : Prop :=
  ∀ (a α β : Type), parallel a β ∧ parallel a α → parallel a β

-- Define proposition q
def q : Prop :=
  ∀ (a α β b : Type), parallel a α ∧ parallel a β ∧ intersection α β = b → parallel a b

-- Theorem to prove
theorem correct_proposition : (¬p) ∧ q := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l3138_313810


namespace NUMINAMATH_CALUDE_student_selection_l3138_313827

theorem student_selection (boys girls : ℕ) (ways : ℕ) : 
  boys = 15 → 
  girls = 10 → 
  ways = 1050 → 
  ways = (girls.choose 1) * (boys.choose 2) →
  1 + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_l3138_313827


namespace NUMINAMATH_CALUDE_parabola_vertex_l3138_313815

/-- The parabola defined by y = (x-1)^2 + 2 has its vertex at (1,2) -/
theorem parabola_vertex (x y : ℝ) : 
  y = (x - 1)^2 + 2 → (1, 2) = (x, y) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3138_313815


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3138_313844

/-- Given two vectors a and b in R², if (a + b) is parallel to (m*a - b), then m = -1 -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
    (h1 : a = (4, -1))
    (h2 : b = (-5, 2))
    (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (m • a - b)) : 
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3138_313844


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l3138_313841

theorem not_necessarily_right_triangle 
  (a b c : ℝ) 
  (ha : a^2 = 5) 
  (hb : b^2 = 12) 
  (hc : c^2 = 13) : 
  ¬ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) := by
sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l3138_313841


namespace NUMINAMATH_CALUDE_square_of_complex_l3138_313867

theorem square_of_complex (z : ℂ) (i : ℂ) : z = 5 + 2 * i → i ^ 2 = -1 → z ^ 2 = 21 + 20 * i := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_l3138_313867


namespace NUMINAMATH_CALUDE_mandy_quarters_l3138_313846

theorem mandy_quarters : 
  ∃ q : ℕ, (40 < q ∧ q < 400) ∧ 
           (q % 6 = 2) ∧ 
           (q % 7 = 2) ∧ 
           (q % 8 = 2) ∧ 
           (q = 170 ∨ q = 338) := by
  sorry

end NUMINAMATH_CALUDE_mandy_quarters_l3138_313846


namespace NUMINAMATH_CALUDE_four_star_three_equals_nineteen_l3138_313806

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := a^2 + a*b - b^2

-- State the theorem
theorem four_star_three_equals_nineteen :
  customOp 4 3 = 19 := by sorry

end NUMINAMATH_CALUDE_four_star_three_equals_nineteen_l3138_313806


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3138_313852

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution (x : ℝ) : 
  (1 - i) * (x + i) = 1 + i → x = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3138_313852


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3138_313883

theorem fraction_sum_equality : (3 : ℚ) / 5 - 1 / 10 + 2 / 15 = 19 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3138_313883


namespace NUMINAMATH_CALUDE_shepherd_a_has_seven_sheep_l3138_313842

/-- Represents the number of sheep each shepherd has -/
structure ShepherdSheep where
  a : ℕ
  b : ℕ

/-- The conditions of the problem are satisfied -/
def satisfiesConditions (s : ShepherdSheep) : Prop :=
  (s.a + 1 = 2 * (s.b - 1)) ∧ (s.a - 1 = s.b + 1)

/-- Theorem stating that shepherd A has 7 sheep -/
theorem shepherd_a_has_seven_sheep :
  ∃ s : ShepherdSheep, satisfiesConditions s ∧ s.a = 7 :=
sorry

end NUMINAMATH_CALUDE_shepherd_a_has_seven_sheep_l3138_313842


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l3138_313873

/-- The probability of drawing a red ball from a bag with white and red balls -/
theorem probability_of_red_ball (total_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ) 
  (h1 : total_balls = white_balls + red_balls)
  (h2 : white_balls = 3)
  (h3 : red_balls = 7) :
  (red_balls : ℚ) / total_balls = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l3138_313873


namespace NUMINAMATH_CALUDE_number_of_female_students_l3138_313824

theorem number_of_female_students 
  (total_average : ℝ) 
  (num_male : ℕ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (h1 : total_average = 90)
  (h2 : num_male = 8)
  (h3 : male_average = 82)
  (h4 : female_average = 92) :
  ∃ (num_female : ℕ), 
    (num_male : ℝ) * male_average + (num_female : ℝ) * female_average = 
    ((num_male : ℝ) + (num_female : ℝ)) * total_average ∧ 
    num_female = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_of_female_students_l3138_313824


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l3138_313802

theorem no_solution_fractional_equation :
  ∀ x : ℝ, (1 - x) / (x - 2) ≠ 1 / (2 - x) + 1 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l3138_313802


namespace NUMINAMATH_CALUDE_smaller_circle_area_l3138_313899

/-- Two circles are externally tangent with common tangents. Given specific conditions, 
    prove that the area of the smaller circle is 5π/3. -/
theorem smaller_circle_area (r : ℝ) : 
  r > 0 → -- radius of smaller circle is positive
  (∃ (P A B : ℝ × ℝ), 
    -- PA and AB are tangent lines
    dist P A = dist A B ∧ 
    dist P A = 5 ∧
    -- Larger circle has radius 3r
    (∃ (C : ℝ × ℝ), dist C B = 3 * r)) →
  π * r^2 = 5 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_area_l3138_313899


namespace NUMINAMATH_CALUDE_jeans_card_collection_l3138_313885

theorem jeans_card_collection (num_groups : ℕ) (cards_per_group : ℕ) 
  (h1 : num_groups = 9) (h2 : cards_per_group = 8) :
  num_groups * cards_per_group = 72 := by
  sorry

end NUMINAMATH_CALUDE_jeans_card_collection_l3138_313885


namespace NUMINAMATH_CALUDE_three_correct_propositions_l3138_313892

theorem three_correct_propositions :
  (∀ a b : ℝ, a > b → a + 1 > b + 1) ∧
  (∀ a b : ℝ, a > b → a - 1 > b - 1) ∧
  (∀ a b : ℝ, a > b → -2 * a < -2 * b) ∧
  ¬(∀ a b : ℝ, a > b → 2 * a < 2 * b) :=
by sorry

end NUMINAMATH_CALUDE_three_correct_propositions_l3138_313892


namespace NUMINAMATH_CALUDE_bob_pizza_calorie_intake_l3138_313807

/-- Calculates the average calorie intake per slice for the slices Bob ate from a pizza -/
def average_calorie_intake (total_slices : ℕ) (low_cal_slices : ℕ) (high_cal_slices : ℕ) (low_cal : ℕ) (high_cal : ℕ) : ℚ :=
  (low_cal_slices * low_cal + high_cal_slices * high_cal) / (low_cal_slices + high_cal_slices)

/-- Theorem stating that the average calorie intake per slice for the slices Bob ate is approximately 357.14 calories -/
theorem bob_pizza_calorie_intake :
  average_calorie_intake 12 3 4 300 400 = 2500 / 7 := by
  sorry

end NUMINAMATH_CALUDE_bob_pizza_calorie_intake_l3138_313807


namespace NUMINAMATH_CALUDE_monicas_first_class_size_l3138_313840

/-- Represents the number of students in Monica's classes -/
structure MonicasClasses where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ

/-- Theorem stating the number of students in Monica's first class -/
theorem monicas_first_class_size (c : MonicasClasses) : c.first = 20 :=
  by
  have h1 : c.second = 25 := by sorry
  have h2 : c.third = 25 := by sorry
  have h3 : c.fourth = c.first / 2 := by sorry
  have h4 : c.fifth = 28 := by sorry
  have h5 : c.sixth = 28 := by sorry
  have h6 : c.first + c.second + c.third + c.fourth + c.fifth + c.sixth = 136 := by sorry
  sorry

#check monicas_first_class_size

end NUMINAMATH_CALUDE_monicas_first_class_size_l3138_313840


namespace NUMINAMATH_CALUDE_product_with_seven_zeros_is_odd_l3138_313896

def binary_num (n : ℕ) : Prop := ∀ d : ℕ, d ∈ n.digits 2 → d = 0 ∨ d = 1

def count_zeros (n : ℕ) : ℕ := (n.digits 2).filter (· = 0) |>.length

theorem product_with_seven_zeros_is_odd (m : ℕ) :
  binary_num m →
  count_zeros (17 * m) = 7 →
  Odd (17 * m) :=
by sorry

end NUMINAMATH_CALUDE_product_with_seven_zeros_is_odd_l3138_313896


namespace NUMINAMATH_CALUDE_min_sum_of_distances_l3138_313843

-- Define the curve
def curve (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the distance from a point to line y = 2
def dist_to_line1 (x y : ℝ) : ℝ := |y - 2|

-- Define the distance from a point to line x = -1
def dist_to_line2 (x y : ℝ) : ℝ := |x + 1|

-- Define the sum of distances
def sum_of_distances (x y : ℝ) : ℝ := dist_to_line1 x y + dist_to_line2 x y

-- Theorem statement
theorem min_sum_of_distances :
  ∃ (min : ℝ), min = 4 - Real.sqrt 2 ∧
  (∀ (x y : ℝ), curve x y → sum_of_distances x y ≥ min) ∧
  (∃ (x y : ℝ), curve x y ∧ sum_of_distances x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_distances_l3138_313843


namespace NUMINAMATH_CALUDE_at_least_ten_same_weight_l3138_313874

/-- Represents the weight of a coin as measured by the scale -/
structure MeasuredWeight where
  value : ℝ
  is_valid : value > 11

/-- Represents the actual weight of a coin -/
structure ActualWeight where
  value : ℝ
  is_valid : value > 10

/-- The scale's measurement is always off by exactly 1 gram -/
def scale_error (actual : ActualWeight) (measured : MeasuredWeight) : Prop :=
  (measured.value = actual.value + 1) ∨ (measured.value = actual.value - 1)

/-- A collection of 12 coin measurements -/
def CoinMeasurements := Fin 12 → MeasuredWeight

/-- The actual weights corresponding to the measurements -/
def ActualWeights := Fin 12 → ActualWeight

theorem at_least_ten_same_weight 
  (measurements : CoinMeasurements) 
  (actual_weights : ActualWeights) 
  (h : ∀ i, scale_error (actual_weights i) (measurements i)) :
  ∃ (w : ℝ) (s : Finset (Fin 12)), s.card ≥ 10 ∧ ∀ i ∈ s, (actual_weights i).value = w :=
sorry

end NUMINAMATH_CALUDE_at_least_ten_same_weight_l3138_313874


namespace NUMINAMATH_CALUDE_determinant_value_l3138_313828

-- Define the operation
def determinant (a b c d : ℚ) : ℚ := a * d - b * c

-- Define the theorem
theorem determinant_value :
  let a : ℚ := -(1^2)
  let b : ℚ := (-2)^2 - 1
  let c : ℚ := -(3^2) + 5
  let d : ℚ := (3/4) / (-1/4)
  determinant a b c d = 15 := by
  sorry

end NUMINAMATH_CALUDE_determinant_value_l3138_313828


namespace NUMINAMATH_CALUDE_pqr_product_l3138_313822

theorem pqr_product (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (h1 : p ∣ (q * r - 1)) (h2 : q ∣ (r * p - 1)) (h3 : r ∣ (p * q - 1)) :
  p * q * r = 30 := by
  sorry

end NUMINAMATH_CALUDE_pqr_product_l3138_313822


namespace NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l3138_313897

theorem lattice_points_on_hyperbola : 
  ∃! (points : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ points ↔ x^2 - y^2 = 65) ∧ 
    points.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l3138_313897


namespace NUMINAMATH_CALUDE_robie_has_five_boxes_l3138_313887

/-- Calculates the number of boxes Robie has left after giving some away -/
def robies_boxes (total_cards : ℕ) (cards_per_box : ℕ) (unboxed_cards : ℕ) (boxes_given_away : ℕ) : ℕ :=
  ((total_cards - unboxed_cards) / cards_per_box) - boxes_given_away

/-- Theorem stating that Robie has 5 boxes left given the initial conditions -/
theorem robie_has_five_boxes :
  robies_boxes 75 10 5 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_robie_has_five_boxes_l3138_313887


namespace NUMINAMATH_CALUDE_transylvanian_identity_l3138_313800

-- Define the possible states of being
inductive State
| Human
| Vampire

-- Define the possible states of mind
inductive Mind
| Sane
| Insane

-- Define a person as a combination of state and mind
structure Person :=
  (state : State)
  (mind : Mind)

-- Define the statement made by the Transylvanian
def transylvanian_statement (p : Person) : Prop :=
  p.state = State.Human ∨ p.mind = Mind.Sane

-- Define the condition that insane vampires only make true statements
axiom insane_vampire_truth (p : Person) :
  p.state = State.Vampire ∧ p.mind = Mind.Insane → transylvanian_statement p

-- Theorem: The Transylvanian must be a human and sane
theorem transylvanian_identity :
  ∃ (p : Person), p.state = State.Human ∧ p.mind = Mind.Sane ∧ transylvanian_statement p :=
by sorry

end NUMINAMATH_CALUDE_transylvanian_identity_l3138_313800


namespace NUMINAMATH_CALUDE_max_xy_given_constraint_l3138_313818

theorem max_xy_given_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + 9 * y = 60) :
  x * y ≤ 25 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 4 * x₀ + 9 * y₀ = 60 ∧ x₀ * y₀ = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_given_constraint_l3138_313818


namespace NUMINAMATH_CALUDE_davids_physics_marks_l3138_313816

/-- Calculates the marks in Physics given marks in other subjects and the average --/
def physics_marks (english : ℕ) (mathematics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + mathematics + chemistry + biology)

/-- Theorem: Given David's marks and average, his Physics marks are 82 --/
theorem davids_physics_marks :
  physics_marks 86 89 87 81 85 = 82 := by
  sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l3138_313816


namespace NUMINAMATH_CALUDE_f_increasing_on_neg_reals_l3138_313863

/-- The function f(x) = -x^2 + 2x is monotonically increasing on (-∞, 0) -/
theorem f_increasing_on_neg_reals (x y : ℝ) :
  x < y → x < 0 → y < 0 → (-x^2 + 2*x) < (-y^2 + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_neg_reals_l3138_313863


namespace NUMINAMATH_CALUDE_investment_problem_l3138_313860

/-- Proves that given the conditions of the investment problem, the invested sum is 4200 --/
theorem investment_problem (P : ℝ) 
  (h1 : P * (15 / 100) * 2 - P * (10 / 100) * 2 = 840) : 
  P = 4200 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l3138_313860


namespace NUMINAMATH_CALUDE_tangent_product_equality_l3138_313830

theorem tangent_product_equality : 
  (1 + Real.tan (20 * π / 180)) * (1 + Real.tan (25 * π / 180)) = 2 :=
by
  sorry

/- Proof hints:
   1. Use the fact that 45° = 20° + 25°
   2. Recall that tan(45°) = 1
   3. Apply the tangent sum formula: tan(A+B) = (tan A + tan B) / (1 - tan A * tan B)
   4. Algebraically manipulate the expressions
-/

end NUMINAMATH_CALUDE_tangent_product_equality_l3138_313830


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3138_313850

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_7 : a + b + c + d + e + f = 7) :
  (1/a) + (4/b) + (9/c) + (16/d) + (25/e) + (36/f) ≥ 63 ∧
  ((1/a) + (4/b) + (9/c) + (16/d) + (25/e) + (36/f) = 63 ↔ 
   a = 1/3 ∧ b = 2/3 ∧ c = 1 ∧ d = 4/3 ∧ e = 5/3 ∧ f = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3138_313850


namespace NUMINAMATH_CALUDE_parabola_focus_centroid_l3138_313858

/-- Given three points A, B, C in a 2D plane, and a parabola y^2 = ax,
    if the focus of the parabola is exactly the centroid of triangle ABC,
    then a = 8. -/
theorem parabola_focus_centroid (A B C : ℝ × ℝ) (a : ℝ) : 
  A = (-1, 2) →
  B = (3, 4) →
  C = (4, -6) →
  let centroid := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  let focus := (a / 4, 0)
  centroid = focus →
  a = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_centroid_l3138_313858


namespace NUMINAMATH_CALUDE_triangle_cosine_double_angle_l3138_313878

theorem triangle_cosine_double_angle 
  (A B C : Real) (a b c : Real) (S : Real) :
  c = 5 →
  B = 2 * Real.pi / 3 →
  S = 15 * Real.sqrt 3 / 4 →
  S = 1/2 * a * c * Real.sin B →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  Real.sin A / a = Real.sin B / b →
  Real.cos (2*A) = 71/98 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_double_angle_l3138_313878


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3138_313832

theorem arithmetic_calculation : -1^4 * 8 - 2^3 / (-4) * (-7 + 5) = -12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3138_313832


namespace NUMINAMATH_CALUDE_circle_ratio_theorem_l3138_313868

noncomputable def circle_ratio (r : ℝ) (A B C : ℝ × ℝ) : Prop :=
  let O := (0, 0)  -- Center of the circle
  ∃ (θ : ℝ),
    -- Points A, B, and C are on a circle of radius r
    dist O A = r ∧ dist O B = r ∧ dist O C = r ∧
    -- AB = AC
    dist A B = dist A C ∧
    -- AB > r
    dist A B > r ∧
    -- Length of minor arc BC is r
    θ = 1 ∧
    -- Ratio AB/BC
    dist A B / dist B C = (1/2) * (1 / Real.sin (1/4))

theorem circle_ratio_theorem (r : ℝ) (A B C : ℝ × ℝ) 
  (h : circle_ratio r A B C) : 
  ∃ (θ : ℝ), dist A B / dist B C = (1/2) * (1 / Real.sin (1/4)) :=
sorry

end NUMINAMATH_CALUDE_circle_ratio_theorem_l3138_313868


namespace NUMINAMATH_CALUDE_no_solution_x4_plus_6_eq_y3_l3138_313871

theorem no_solution_x4_plus_6_eq_y3 :
  ∀ (x y : ℤ), (x^4 + 6) % 13 ≠ y^3 % 13 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_x4_plus_6_eq_y3_l3138_313871


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l3138_313889

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l3138_313889


namespace NUMINAMATH_CALUDE_salary_increase_with_manager_l3138_313829

/-- Calculates the increase in average salary when a manager's salary is added to a group of employees. -/
theorem salary_increase_with_manager 
  (num_employees : ℕ) 
  (avg_salary : ℚ) 
  (manager_salary : ℚ) : 
  num_employees = 18 → 
  avg_salary = 2000 → 
  manager_salary = 5800 → 
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 200 :=
by sorry

end NUMINAMATH_CALUDE_salary_increase_with_manager_l3138_313829


namespace NUMINAMATH_CALUDE_red_knights_magical_swords_fraction_l3138_313817

/-- Represents the color of a knight -/
inductive KnightColor
  | Red
  | Blue
  | Green

/-- Represents the total number of knights -/
def totalKnights : ℕ := 40

/-- The fraction of knights that are red -/
def redFraction : ℚ := 3/8

/-- The fraction of knights that are blue -/
def blueFraction : ℚ := 1/4

/-- The fraction of knights that are green -/
def greenFraction : ℚ := 1 - redFraction - blueFraction

/-- The fraction of all knights that wield magical swords -/
def magicalSwordsFraction : ℚ := 1/5

/-- The ratio of red knights with magical swords to blue knights with magical swords -/
def redToBlueMagicalRatio : ℚ := 3/2

/-- The ratio of red knights with magical swords to green knights with magical swords -/
def redToGreenMagicalRatio : ℚ := 2

theorem red_knights_magical_swords_fraction :
  ∃ (redMagicalFraction : ℚ),
    redMagicalFraction = 48/175 ∧
    redMagicalFraction * redFraction * totalKnights +
    (redMagicalFraction / redToBlueMagicalRatio) * blueFraction * totalKnights +
    (redMagicalFraction / redToGreenMagicalRatio) * greenFraction * totalKnights =
    magicalSwordsFraction * totalKnights :=
by sorry

end NUMINAMATH_CALUDE_red_knights_magical_swords_fraction_l3138_313817


namespace NUMINAMATH_CALUDE_octal_sum_equality_l3138_313890

/-- Converts a list of digits in base 8 to a natural number -/
def fromOctal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- The sum of 235₈, 647₈, and 54₈ is equal to 1160₈ -/
theorem octal_sum_equality :
  fromOctal [2, 3, 5] + fromOctal [6, 4, 7] + fromOctal [5, 4] = fromOctal [1, 1, 6, 0] := by
  sorry

#eval fromOctal [2, 3, 5] + fromOctal [6, 4, 7] + fromOctal [5, 4]
#eval fromOctal [1, 1, 6, 0]

end NUMINAMATH_CALUDE_octal_sum_equality_l3138_313890


namespace NUMINAMATH_CALUDE_rainfall_ratio_l3138_313838

/-- Given the total rainfall over two weeks and the rainfall in the second week,
    calculate the ratio of the second week's rainfall to the first week's rainfall. -/
theorem rainfall_ratio (total : ℝ) (second_week : ℝ) :
  total = 30 →
  second_week = 18 →
  (second_week / (total - second_week) = 3 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_rainfall_ratio_l3138_313838


namespace NUMINAMATH_CALUDE_three_digit_combinations_l3138_313876

def set1 : Finset Nat := {0, 2, 4}
def set2 : Finset Nat := {1, 3, 5}

theorem three_digit_combinations : 
  (Finset.card set1) * (Finset.card set2) * (Finset.card set2 - 1) = 48 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_combinations_l3138_313876


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3138_313865

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0) ↔ (k ≥ -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3138_313865


namespace NUMINAMATH_CALUDE_ngon_construction_l3138_313837

/-- A line in 2D space -/
structure Line where
  -- Define a line using two points
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ

/-- An n-gon in 2D space -/
structure Polygon where
  -- List of vertices
  vertices : List (ℝ × ℝ)

/-- Function to check if a line is a perpendicular bisector of a polygon side -/
def isPerpBisector (l : Line) (p : Polygon) : Prop :=
  sorry

/-- Function to check if a line is an angle bisector of a polygon vertex -/
def isAngleBisector (l : Line) (p : Polygon) : Prop :=
  sorry

/-- Main theorem: Given n lines, there exists an n-gon such that these lines
    are either perpendicular bisectors of its sides or angle bisectors -/
theorem ngon_construction (n : ℕ) (lines : List Line) :
  (lines.length = n) →
  ∃ (p : Polygon),
    (p.vertices.length = n) ∧
    (∀ l ∈ lines, isPerpBisector l p ∨ isAngleBisector l p) :=
by sorry

end NUMINAMATH_CALUDE_ngon_construction_l3138_313837


namespace NUMINAMATH_CALUDE_digital_earth_implies_science_technology_expression_l3138_313826

-- Define the concept of Digital Earth
def DigitalEarth : Prop := sorry

-- Define the concept of technological innovation paradigm
def TechnologicalInnovationParadigm : Prop := sorry

-- Define the concept of science and technology as expression of advanced productive forces
def ScienceTechnologyExpression : Prop := sorry

-- Theorem statement
theorem digital_earth_implies_science_technology_expression :
  (DigitalEarth → TechnologicalInnovationParadigm) →
  (TechnologicalInnovationParadigm → ScienceTechnologyExpression) :=
by
  sorry

end NUMINAMATH_CALUDE_digital_earth_implies_science_technology_expression_l3138_313826


namespace NUMINAMATH_CALUDE_sin_product_18_54_72_36_l3138_313875

theorem sin_product_18_54_72_36 :
  Real.sin (18 * π / 180) * Real.sin (54 * π / 180) *
  Real.sin (72 * π / 180) * Real.sin (36 * π / 180) =
  (Real.sqrt 5 + 1) / 16 := by sorry

end NUMINAMATH_CALUDE_sin_product_18_54_72_36_l3138_313875


namespace NUMINAMATH_CALUDE_power_function_not_through_origin_l3138_313888

theorem power_function_not_through_origin (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (m^2 - 3*m + 3) * x^(m^2 - m - 2)
  (∀ x ≠ 0, f x ≠ 0) → (m = 1 ∨ m = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_power_function_not_through_origin_l3138_313888


namespace NUMINAMATH_CALUDE_wallet_value_theorem_l3138_313864

/-- Represents the total value of bills in a wallet -/
def wallet_value (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ) : ℕ :=
  5 * five_dollar_bills + 10 * ten_dollar_bills

/-- Theorem: The total value of 4 $5 bills and 8 $10 bills is $100 -/
theorem wallet_value_theorem : wallet_value 4 8 = 100 := by
  sorry

#eval wallet_value 4 8

end NUMINAMATH_CALUDE_wallet_value_theorem_l3138_313864


namespace NUMINAMATH_CALUDE_no_solution_for_digit_equation_l3138_313835

theorem no_solution_for_digit_equation : 
  ¬ ∃ (x : ℕ), x ≤ 9 ∧ ((x : ℤ) - (10 * x + x) = 801 ∨ (x : ℤ) - (10 * x + x) = 812) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_digit_equation_l3138_313835


namespace NUMINAMATH_CALUDE_decreasing_power_function_m_values_l3138_313823

theorem decreasing_power_function_m_values (m : ℤ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → x₁^(m^2 - m - 2) > x₂^(m^2 - m - 2)) →
  m = 0 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_power_function_m_values_l3138_313823


namespace NUMINAMATH_CALUDE_direct_proportion_through_points_l3138_313869

/-- A direct proportion function passing through (-1, 2) also passes through (1, -2) -/
theorem direct_proportion_through_points :
  ∀ (f : ℝ → ℝ) (k : ℝ),
    (∀ x, f x = k * x) →  -- f is a direct proportion function
    f (-1) = 2 →          -- f passes through (-1, 2)
    f 1 = -2 :=           -- f passes through (1, -2)
by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_through_points_l3138_313869


namespace NUMINAMATH_CALUDE_sticker_problem_l3138_313895

theorem sticker_problem (initial_stickers : ℚ) : 
  let lost_stickers := (1 : ℚ) / 3 * initial_stickers
  let found_stickers := (3 : ℚ) / 4 * lost_stickers
  let remaining_stickers := initial_stickers - lost_stickers + found_stickers
  initial_stickers - remaining_stickers = (1 : ℚ) / 12 * initial_stickers :=
by sorry

end NUMINAMATH_CALUDE_sticker_problem_l3138_313895


namespace NUMINAMATH_CALUDE_unique_solution_to_system_l3138_313803

theorem unique_solution_to_system :
  ∃! (x y z : ℝ), 
    x^2 - 23*y + 66*z + 612 = 0 ∧
    y^2 + 62*x - 20*z + 296 = 0 ∧
    z^2 - 22*x + 67*y + 505 = 0 ∧
    x = -20 ∧ y = -22 ∧ z = -23 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_system_l3138_313803


namespace NUMINAMATH_CALUDE_pipe_A_fill_time_l3138_313848

-- Define the flow rates of pipes A, B, and C
def flow_rate_A : ℝ := by sorry
def flow_rate_B : ℝ := 2 * flow_rate_A
def flow_rate_C : ℝ := 2 * flow_rate_B

-- Define the time it takes for all three pipes to fill the tank
def total_fill_time : ℝ := 4

-- Theorem stating that pipe A alone takes 28 hours to fill the tank
theorem pipe_A_fill_time :
  1 / flow_rate_A = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_pipe_A_fill_time_l3138_313848


namespace NUMINAMATH_CALUDE_existence_of_sum_of_cubes_l3138_313894

theorem existence_of_sum_of_cubes :
  ∃ (a b c d : ℕ), a^3 + b^3 + c^3 + d^3 = 100^100 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_sum_of_cubes_l3138_313894


namespace NUMINAMATH_CALUDE_trig_function_properties_l3138_313851

open Real

theorem trig_function_properties :
  (∀ x, cos (x + π/3) = cos (π/3 - x)) ∧
  (∀ x, 3 * sin (2 * (x - π/6) + π/3) = 3 * sin (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_trig_function_properties_l3138_313851


namespace NUMINAMATH_CALUDE_ernie_circles_l3138_313820

def total_boxes : ℕ := 80
def ali_boxes_per_circle : ℕ := 8
def ernie_boxes_per_circle : ℕ := 10
def ali_circles : ℕ := 5

theorem ernie_circles : 
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle = 4 := by
  sorry

end NUMINAMATH_CALUDE_ernie_circles_l3138_313820


namespace NUMINAMATH_CALUDE_even_function_increasing_interval_l3138_313819

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The interval (-∞, 0] -/
def NegativeRealsAndZero : Set ℝ := { x | x ≤ 0 }

/-- A function f : ℝ → ℝ is increasing on a set S if f(x) ≤ f(y) for all x, y ∈ S with x ≤ y -/
def IncreasingOn (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

/-- The main theorem -/
theorem even_function_increasing_interval (a : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (a - 2) * x^2 + (a - 1) * x + 3
  IsEven f →
  IncreasingOn f NegativeRealsAndZero ∧
  ∀ S, IncreasingOn f S → S ⊆ NegativeRealsAndZero :=
sorry

end NUMINAMATH_CALUDE_even_function_increasing_interval_l3138_313819


namespace NUMINAMATH_CALUDE_fencing_requirement_l3138_313898

/-- Given a rectangular field with area 210 sq. feet and one side 20 feet,
    prove that the sum of the other three sides is 41 feet. -/
theorem fencing_requirement (area : ℝ) (length : ℝ) (width : ℝ) : 
  area = 210 →
  length = 20 →
  area = length * width →
  2 * width + length = 41 := by
  sorry

end NUMINAMATH_CALUDE_fencing_requirement_l3138_313898


namespace NUMINAMATH_CALUDE_hyperbola_in_trilinear_coordinates_l3138_313845

/-- Trilinear coordinates -/
structure TrilinearCoord where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Triangle with angles A, B, C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Hyperbola equation in trilinear coordinates -/
def hyperbola_equation (t : Triangle) (p : TrilinearCoord) : Prop :=
  (Real.sin (2 * t.A) * Real.cos (t.B - t.C)) / p.x +
  (Real.sin (2 * t.B) * Real.cos (t.C - t.A)) / p.y +
  (Real.sin (2 * t.C) * Real.cos (t.A - t.B)) / p.z = 0

/-- Theorem: The equation of the hyperbola in trilinear coordinates -/
theorem hyperbola_in_trilinear_coordinates (t : Triangle) (p : TrilinearCoord) :
  hyperbola_equation t p := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_in_trilinear_coordinates_l3138_313845


namespace NUMINAMATH_CALUDE_trivia_team_grouping_l3138_313854

theorem trivia_team_grouping (total_students : ℕ) (students_not_picked : ℕ) (num_groups : ℕ)
  (h1 : total_students = 120)
  (h2 : students_not_picked = 22)
  (h3 : num_groups = 14)
  : (total_students - students_not_picked) / num_groups = 7 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_grouping_l3138_313854


namespace NUMINAMATH_CALUDE_max_correct_is_42_l3138_313893

/-- Represents the exam scoring system and Xiaolong's result -/
structure ExamResult where
  total_questions : Nat
  correct_points : Int
  incorrect_points : Int
  no_answer_points : Int
  total_score : Int

/-- Calculates the maximum number of correctly answered questions -/
def max_correct_answers (exam : ExamResult) : Nat :=
  sorry

/-- Theorem stating that the maximum number of correct answers is 42 -/
theorem max_correct_is_42 (exam : ExamResult) 
  (h1 : exam.total_questions = 50)
  (h2 : exam.correct_points = 3)
  (h3 : exam.incorrect_points = -1)
  (h4 : exam.no_answer_points = 0)
  (h5 : exam.total_score = 120) :
  max_correct_answers exam = 42 :=
  sorry

end NUMINAMATH_CALUDE_max_correct_is_42_l3138_313893


namespace NUMINAMATH_CALUDE_vector_angle_solution_l3138_313857

/-- Given two plane vectors a and b with unit length and 60° angle between them,
    prove that t = 0 is a valid solution when the angle between a+b and ta-b is obtuse. -/
theorem vector_angle_solution (a b : ℝ × ℝ) (t : ℝ) :
  (norm a = 1) →
  (norm b = 1) →
  (a • b = 1 / 2) →  -- cos 60° = 1/2
  ((a + b) • (t • a - b) < 0) →
  (t = 0) →
  True := by sorry

end NUMINAMATH_CALUDE_vector_angle_solution_l3138_313857


namespace NUMINAMATH_CALUDE_vovochka_candy_theorem_l3138_313859

/-- Represents the candy distribution problem --/
structure CandyDistribution where
  total_candies : ℕ
  num_classmates : ℕ
  min_group_size : ℕ
  min_group_candies : ℕ

/-- The maximum number of candies that can be kept while satisfying the distribution condition --/
def max_kept_candies (cd : CandyDistribution) : ℕ :=
  cd.total_candies - (cd.num_classmates * (cd.min_group_candies / cd.min_group_size))

/-- Theorem stating the maximum number of candies that can be kept in the given scenario --/
theorem vovochka_candy_theorem (cd : CandyDistribution) 
  (h1 : cd.total_candies = 200)
  (h2 : cd.num_classmates = 25)
  (h3 : cd.min_group_size = 16)
  (h4 : cd.min_group_candies = 100) :
  max_kept_candies cd = 37 := by
  sorry

#eval max_kept_candies { total_candies := 200, num_classmates := 25, min_group_size := 16, min_group_candies := 100 }

end NUMINAMATH_CALUDE_vovochka_candy_theorem_l3138_313859


namespace NUMINAMATH_CALUDE_retailer_profit_percent_l3138_313882

/-- Calculates the profit percent given purchase price, overhead expenses, and selling price -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percent for the given values is 45.83% -/
theorem retailer_profit_percent :
  profit_percent 225 15 350 = 45.83 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percent_l3138_313882


namespace NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_271_l3138_313870

theorem first_nonzero_digit_after_decimal_1_271 :
  ∃ (n : ℕ) (r : ℚ), 1000 * (1 / 271) = n + r ∧ n = 3 ∧ 0 < r ∧ r < 1 := by
  sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_271_l3138_313870


namespace NUMINAMATH_CALUDE_cubic_function_property_l3138_313801

/-- A cubic function g(x) with specific properties -/
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

theorem cubic_function_property (p q r s : ℝ) :
  g p q r s 1 = 1 →
  g p q r s 3 = 1 →
  g p q r s 2 = 2 →
  (fun x ↦ 3 * p * x^2 + 2 * q * x + r) 2 = 0 →
  3 * p - 2 * q + r - 4 * s = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3138_313801


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l3138_313862

-- Define the polynomial g(x)
def g (d : ℝ) (x : ℝ) : ℝ := d * x^3 + 25 * x^2 - 5 * d * x + 45

-- State the theorem
theorem factor_implies_d_value :
  ∀ d : ℝ, (∀ x : ℝ, (x + 5) ∣ g d x) → d = 6.7 :=
by sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l3138_313862


namespace NUMINAMATH_CALUDE_lengths_form_triangle_l3138_313881

/-- A function that checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the lengths 4, 6, and 9 can form a triangle -/
theorem lengths_form_triangle : can_form_triangle 4 6 9 := by
  sorry

end NUMINAMATH_CALUDE_lengths_form_triangle_l3138_313881
