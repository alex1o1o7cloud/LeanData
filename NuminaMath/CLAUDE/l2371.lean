import Mathlib

namespace percentage_of_2_to_50_l2371_237176

theorem percentage_of_2_to_50 : (2 : ℝ) / 50 * 100 = 4 := by
  sorry

end percentage_of_2_to_50_l2371_237176


namespace percentage_increase_l2371_237141

theorem percentage_increase (initial final : ℝ) (h1 : initial = 60) (h2 : final = 90) :
  (final - initial) / initial * 100 = 50 := by
  sorry

end percentage_increase_l2371_237141


namespace stripes_calculation_l2371_237123

/-- The number of stripes on one of Olga's shoes -/
def olga_stripes_per_shoe : ℕ := 3

/-- The number of stripes on one of Rick's shoes -/
def rick_stripes_per_shoe : ℕ := olga_stripes_per_shoe - 1

/-- The number of stripes on one of Hortense's shoes -/
def hortense_stripes_per_shoe : ℕ := 2 * olga_stripes_per_shoe

/-- The number of stripes on one of Ethan's shoes -/
def ethan_stripes_per_shoe : ℕ := hortense_stripes_per_shoe + 2

/-- The total number of stripes on all shoes -/
def total_stripes : ℕ := 2 * (olga_stripes_per_shoe + rick_stripes_per_shoe + hortense_stripes_per_shoe + ethan_stripes_per_shoe)

/-- The final result after dividing by 2 and rounding up -/
def final_result : ℕ := (total_stripes + 1) / 2

theorem stripes_calculation :
  final_result = 19 := by sorry

end stripes_calculation_l2371_237123


namespace average_of_numbers_l2371_237103

def numbers : List ℕ := [1, 2, 4, 5, 6, 9, 9, 10, 12, 12]

theorem average_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 7 := by sorry

end average_of_numbers_l2371_237103


namespace solve_equation_l2371_237195

theorem solve_equation (y : ℤ) (h : 7 - y = 13) : y = -6 := by
  sorry

end solve_equation_l2371_237195


namespace intersection_equals_closed_interval_l2371_237194

-- Define the sets M and N
def M : Set ℝ := {y | y ≥ -1}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define the closed interval [-1, 3]
def closedInterval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_equals_closed_interval : M ∩ N = closedInterval := by sorry

end intersection_equals_closed_interval_l2371_237194


namespace max_hands_in_dance_l2371_237174

/-- Represents a Martian participating in the dance --/
structure Martian :=
  (hands : Nat)
  (hands_le_three : hands ≤ 3)

/-- Represents the dance configuration --/
structure DanceConfiguration :=
  (participants : List Martian)
  (participant_count_le_seven : participants.length ≤ 7)

/-- Calculates the total number of hands in a dance configuration --/
def total_hands (config : DanceConfiguration) : Nat :=
  config.participants.foldl (λ sum martian => sum + martian.hands) 0

/-- Theorem: The maximum number of hands involved in the dance is 20 --/
theorem max_hands_in_dance :
  ∃ (config : DanceConfiguration),
    (∀ (other_config : DanceConfiguration),
      total_hands other_config ≤ total_hands config) ∧
    total_hands config = 20 ∧
    total_hands config % 2 = 0 :=
  sorry

end max_hands_in_dance_l2371_237174


namespace cyclic_identity_l2371_237112

theorem cyclic_identity (a b c : ℝ) : 
  a * (a - c)^2 + b * (b - c)^2 - (a - c) * (b - c) * (a + b - c) = 
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) ∧
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) = 
  c * (c - b)^2 + a * (a - b)^2 - (c - b) * (a - b) * (c + a - b) ∧
  a * (a - c)^2 + b * (b - c)^2 - (a - c) * (b - c) * (a + b - c) = 
  a^3 + b^3 + c^3 - (a^2 * b + a * b^2 + b^2 * c + b * c^2 + c^2 * a + a^2 * c) + 3 * a * b * c :=
by sorry

end cyclic_identity_l2371_237112


namespace equal_quantities_solution_l2371_237159

theorem equal_quantities_solution (x y : ℝ) (h : y ≠ 0) :
  (((x + y = x - y ∧ x + y = x * y) ∨
    (x + y = x - y ∧ x + y = x / y) ∨
    (x + y = x * y ∧ x + y = x / y) ∨
    (x - y = x * y ∧ x - y = x / y)) →
   ((x = 1/2 ∧ y = -1) ∨ (x = -1/2 ∧ y = -1))) :=
by sorry

end equal_quantities_solution_l2371_237159


namespace candy_distribution_l2371_237128

/-- Represents the number of positions moved for the k-th candy distribution -/
def a (k : ℕ) : ℕ := k * (k + 1) / 2

/-- Checks if all students in a circle of size n receive a candy -/
def all_receive_candy (n : ℕ) : Prop :=
  ∀ m : ℕ, m < n → ∃ k : ℕ, a k % n = m

/-- Main theorem: All students receive a candy iff n is a power of 2 -/
theorem candy_distribution (n : ℕ) :
  all_receive_candy n ↔ ∃ m : ℕ, n = 2^m :=
sorry

/-- Helper lemma: If n is not a power of 2, not all students receive a candy -/
lemma not_power_of_two_not_all_receive (n : ℕ) :
  (¬ ∃ m : ℕ, n = 2^m) → ¬ all_receive_candy n :=
sorry

/-- Helper lemma: If n is a power of 2, all students receive a candy -/
lemma power_of_two_all_receive (m : ℕ) :
  all_receive_candy (2^m) :=
sorry

end candy_distribution_l2371_237128


namespace rose_sale_earnings_l2371_237175

theorem rose_sale_earnings :
  ∀ (price : ℕ) (initial : ℕ) (remaining : ℕ),
    price = 7 →
    initial = 9 →
    remaining = 4 →
    (initial - remaining) * price = 35 :=
by sorry

end rose_sale_earnings_l2371_237175


namespace muffin_banana_cost_ratio_l2371_237130

/-- The cost ratio of a muffin to a banana given Susie and Calvin's purchases -/
theorem muffin_banana_cost_ratio :
  let muffin_cost : ℚ := muffin_cost
  let banana_cost : ℚ := banana_cost
  (6 * muffin_cost + 4 * banana_cost) * 3 = 3 * muffin_cost + 24 * banana_cost →
  muffin_cost / banana_cost = 4 / 5 := by
sorry


end muffin_banana_cost_ratio_l2371_237130


namespace smallest_number_with_20_divisors_l2371_237122

/-- The number of divisors of a natural number n -/
def numDivisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- A natural number n has exactly 20 divisors -/
def has20Divisors (n : ℕ) : Prop := numDivisors n = 20

theorem smallest_number_with_20_divisors :
  ∀ n : ℕ, has20Divisors n → n ≥ 240 :=
by sorry

end smallest_number_with_20_divisors_l2371_237122


namespace yellow_highlighters_count_l2371_237186

theorem yellow_highlighters_count (total : ℕ) (pink : ℕ) (blue : ℕ) 
  (h1 : total = 12) 
  (h2 : pink = 6) 
  (h3 : blue = 4) : 
  total - pink - blue = 2 := by
  sorry

end yellow_highlighters_count_l2371_237186


namespace consecutive_binomial_ratio_sum_n_plus_k_l2371_237114

theorem consecutive_binomial_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 3 / 6 →
  n = 11 ∧ k = 2 :=
by sorry

theorem sum_n_plus_k (n k : ℕ) :
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 3 / 6 →
  n + k = 13 :=
by sorry

end consecutive_binomial_ratio_sum_n_plus_k_l2371_237114


namespace z_in_fourth_quadrant_l2371_237161

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for z
def z_condition (z : ℂ) : Prop := z * (1 + i) = 2 * i

-- Define what it means for a complex number to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant :
  ∃ z : ℂ, z_condition z ∧ in_fourth_quadrant z :=
sorry

end z_in_fourth_quadrant_l2371_237161


namespace hyperbola_asymptote_slope_l2371_237116

/-- Given a hyperbola with equation (x^2 / 144) - (y^2 / 81) = 1 and asymptotes y = ±mx, prove that m = 3/4 -/
theorem hyperbola_asymptote_slope (x y m : ℝ) : 
  ((x^2 / 144) - (y^2 / 81) = 1) → 
  (∃ (k : ℝ), y = k * m * x ∨ y = -k * m * x) → 
  m = 3/4 := by
sorry

end hyperbola_asymptote_slope_l2371_237116


namespace c_share_value_l2371_237120

/-- Proves that given the conditions, c's share is 398.75 -/
theorem c_share_value (total : ℚ) (a b c d : ℚ) : 
  total = 1500 →
  5/2 * a = 7/3 * b →
  5/2 * a = 2 * c →
  5/2 * a = 11/6 * d →
  a + b + c + d = total →
  c = 398.75 := by
sorry

end c_share_value_l2371_237120


namespace circle_equations_correct_l2371_237169

-- Define the parallel lines
def line1 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + y + Real.sqrt 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 2 * Real.sqrt 2 * a = 0

-- Define the circle N
def circleN (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 9

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x + 5)^2 + (y + 2)^2 = 49

-- Define point B
def pointB : ℝ × ℝ := (3, -2)

-- Define the line of symmetry
def lineOfSymmetry (x : ℝ) : Prop := x = -1

-- Main theorem
theorem circle_equations_correct :
  ∃ (a : ℝ),
    (∀ x y, line1 a x y ↔ line2 a x y) ∧  -- Lines are parallel
    (∃ r, r > 0 ∧ r = Real.sqrt ((3 - (-5))^2 + (4 - (-2))^2) - 3) ∧  -- Distance between centers minus radius of N
    (∀ x, lineOfSymmetry x → x = -1) ∧
    (∃ c : ℝ × ℝ, c.1 = -5 ∧ c.2 = -2) →  -- Point C exists
  (∀ x y, circleN x y) ∧ (∀ x y, circleC x y) :=
sorry

end circle_equations_correct_l2371_237169


namespace polynomial_simplification_l2371_237136

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 - 5 * x + 2) - (2 * x^3 + x^2 - 7 * x - 6) = x^3 + 3 * x^2 + 2 * x + 8 := by
  sorry

end polynomial_simplification_l2371_237136


namespace quadratic_factorization_l2371_237177

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end quadratic_factorization_l2371_237177


namespace square_difference_pattern_l2371_237129

theorem square_difference_pattern (n : ℕ) : (n + 1)^2 - n^2 = 2*n + 1 := by
  sorry

end square_difference_pattern_l2371_237129


namespace roots_relation_l2371_237150

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

-- Define the polynomial j(x)
def j (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- Theorem statement
theorem roots_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, h x = 0 → j b c d (x^3) = 0) →
  b = 6 ∧ c = 12 ∧ d = 8 :=
by sorry

end roots_relation_l2371_237150


namespace night_ride_ratio_l2371_237135

def ferris_wheel_total : ℕ := 13
def roller_coaster_total : ℕ := 9
def ferris_wheel_day : ℕ := 7
def roller_coaster_day : ℕ := 4

theorem night_ride_ratio :
  (ferris_wheel_total - ferris_wheel_day) * 5 = (roller_coaster_total - roller_coaster_day) * 6 := by
  sorry

end night_ride_ratio_l2371_237135


namespace transaction_difference_l2371_237121

theorem transaction_difference (mabel_transactions : ℕ) 
  (anthony_transactions : ℕ) (cal_transactions : ℕ) (jade_transactions : ℕ) :
  mabel_transactions = 90 →
  anthony_transactions = mabel_transactions + mabel_transactions / 10 →
  cal_transactions = anthony_transactions * 2 / 3 →
  jade_transactions = 85 →
  jade_transactions - cal_transactions = 19 := by
sorry

end transaction_difference_l2371_237121


namespace pythagorean_theorem_l2371_237180

-- Define a right triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleBAC_is_right : angleBAC = 90

-- State the theorem
theorem pythagorean_theorem (t : RightTriangle) : t.b^2 + t.c^2 = t.a^2 := by
  sorry

end pythagorean_theorem_l2371_237180


namespace budget_allocation_theorem_l2371_237198

def budget_allocation (microphotonics home_electronics food_additives industrial_lubricants basic_astrophysics_degrees : ℝ) : Prop :=
  let total_degrees : ℝ := 360
  let total_percentage : ℝ := 100
  let basic_astrophysics_percentage : ℝ := (basic_astrophysics_degrees / total_degrees) * total_percentage
  let known_percentage : ℝ := microphotonics + home_electronics + food_additives + industrial_lubricants + basic_astrophysics_percentage
  let gmo_percentage : ℝ := total_percentage - known_percentage
  gmo_percentage = 19

theorem budget_allocation_theorem :
  budget_allocation 14 24 15 8 72 := by
  sorry

end budget_allocation_theorem_l2371_237198


namespace min_sheets_for_boats_is_one_l2371_237106

/-- The minimum number of sheets needed to make paper boats -/
def min_sheets_for_boats : ℕ := 1

/-- The total number of paper toys to be made -/
def total_toys : ℕ := 250

/-- The number of paper boats that can be made from one sheet -/
def boats_per_sheet : ℕ := 9

/-- The number of paper planes that can be made from one sheet -/
def planes_per_sheet : ℕ := 5

/-- The number of paper helicopters that can be made from one sheet -/
def helicopters_per_sheet : ℕ := 3

/-- Theorem stating that the minimum number of sheets needed for paper boats is 1 -/
theorem min_sheets_for_boats_is_one :
  ∃ (boats planes helicopters : ℕ),
    boats + planes + helicopters = total_toys ∧
    boats ≤ min_sheets_for_boats * boats_per_sheet ∧
    planes ≤ (total_toys / helicopters_per_sheet) * planes_per_sheet ∧
    helicopters = (total_toys / helicopters_per_sheet) * helicopters_per_sheet :=
by sorry

end min_sheets_for_boats_is_one_l2371_237106


namespace final_average_is_23_l2371_237172

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

end final_average_is_23_l2371_237172


namespace place_mat_length_l2371_237143

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ) :
  r = 5 →
  n = 8 →
  w = 1 →
  (x - w/2)^2 + (w/2)^2 = r^2 →
  x = (3 * Real.sqrt 11 + 1) / 2 := by
  sorry

end place_mat_length_l2371_237143


namespace G₁_intersects_x_axis_range_of_n_minus_m_plus_a_right_triangle_BNB_l2371_237113

-- Define the parabola G₁
def G₁ (a x : ℝ) : ℝ := x^2 - 2*a*x + a^2 - 4

-- Define the coordinates of point N
def N : ℝ × ℝ := (0, -4)

-- Theorem 1: G₁ intersects x-axis at two points
theorem G₁_intersects_x_axis (a : ℝ) :
  ∃ m n : ℝ, m < n ∧ G₁ a m = 0 ∧ G₁ a n = 0 := by sorry

-- Theorem 2: Range of n - m + a when NA ≥ 5
theorem range_of_n_minus_m_plus_a (a m n : ℝ) :
  m < n → G₁ a m = 0 → G₁ a n = 0 →
  Real.sqrt ((m - N.1)^2 + (N.2)^2) ≥ 5 →
  (n - m + a ≥ 9 ∨ n - m + a ≤ 3) := by sorry

-- Define the parabola G₂ (symmetric to G₁ with respect to A)
def G₂ (a x : ℝ) : ℝ := G₁ a (2*a - 2 - x)

-- Theorem 3: Conditions for right triangle BNB'
theorem right_triangle_BNB' (a : ℝ) :
  (∃ m n b : ℝ, m < n ∧ G₁ a m = 0 ∧ G₁ a n = 0 ∧ G₂ a b = 0 ∧ b ≠ m ∧
   (n - N.1)^2 + N.2^2 + (b - N.1)^2 + N.2^2 = (n - b)^2) ↔
  (a = 2 ∨ a = -2 ∨ a = 6) := by sorry

end G₁_intersects_x_axis_range_of_n_minus_m_plus_a_right_triangle_BNB_l2371_237113


namespace range_a_theorem_l2371_237100

/-- Proposition p: The solution set of the inequality x^2+(a-1)x+1≤0 is the empty set ∅ -/
def p (a : ℝ) : Prop :=
  ∀ x, x^2 + (a-1)*x + 1 > 0

/-- Proposition q: The function y=(a-1)^x is an increasing function -/
def q (a : ℝ) : Prop :=
  ∀ x y, x < y → (a-1)^x < (a-1)^y

/-- The range of a satisfying the given conditions -/
def range_a : Set ℝ :=
  {a | (-1 < a ∧ a ≤ 2) ∨ a ≥ 3}

/-- Theorem stating that given the conditions, the range of a is as specified -/
theorem range_a_theorem (a : ℝ) :
  (¬(p a ∧ q a)) → (p a ∨ q a) → a ∈ range_a :=
by sorry

end range_a_theorem_l2371_237100


namespace laurence_to_missy_relation_keith_receives_32_messages_l2371_237178

/-- Messages sent from Juan to Laurence -/
def messages_juan_to_laurence : ℕ := sorry

/-- Messages sent from Juan to Keith -/
def messages_juan_to_keith : ℕ := 8 * messages_juan_to_laurence

/-- Messages sent from Laurence to Missy -/
def messages_laurence_to_missy : ℕ := 18

/-- Relation between messages from Laurence to Missy and from Juan to Laurence -/
theorem laurence_to_missy_relation : 
  messages_laurence_to_missy = (4.5 : ℚ) * messages_juan_to_laurence := sorry

theorem keith_receives_32_messages : messages_juan_to_keith = 32 := by sorry

end laurence_to_missy_relation_keith_receives_32_messages_l2371_237178


namespace unique_polynomial_function_l2371_237164

-- Define a polynomial function of degree 3
def PolynomialDegree3 (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^3 + b * x^2 + c * x + d

-- Define the conditions given in the problem
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x^2) = (f x)^2) ∧
  (∀ x, f (x^2) = f (f x)) ∧
  f 1 = f (-1)

-- Theorem statement
theorem unique_polynomial_function :
  ∃! f : ℝ → ℝ, PolynomialDegree3 f ∧ SatisfiesConditions f ∧ (∀ x, f x = x^3) :=
sorry

end unique_polynomial_function_l2371_237164


namespace sams_initial_dimes_l2371_237140

theorem sams_initial_dimes (initial_dimes final_dimes dimes_from_dad : ℕ) 
  (h1 : final_dimes = initial_dimes + dimes_from_dad)
  (h2 : final_dimes = 16)
  (h3 : dimes_from_dad = 7) : 
  initial_dimes = 9 := by
  sorry

end sams_initial_dimes_l2371_237140


namespace arithmetic_sequence_problem_l2371_237138

theorem arithmetic_sequence_problem (a b c : ℤ) :
  (∃ d : ℤ, -1 = a - d ∧ a = b - d ∧ b = c - d ∧ c = -9 + d) →
  b = -5 ∧ a * c = 21 := by
  sorry

end arithmetic_sequence_problem_l2371_237138


namespace product_of_sums_equal_difference_of_powers_l2371_237173

theorem product_of_sums_equal_difference_of_powers : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end product_of_sums_equal_difference_of_powers_l2371_237173


namespace quadratic_equation_transformation_l2371_237193

theorem quadratic_equation_transformation (x : ℝ) :
  x^2 - 6*x + 5 = 0 ↔ (x - 3)^2 = 14 := by
  sorry

end quadratic_equation_transformation_l2371_237193


namespace art_club_artworks_art_club_two_years_collection_l2371_237160

theorem art_club_artworks (num_students : ℕ) (artworks_per_student_per_quarter : ℕ) 
  (quarters_per_year : ℕ) (num_years : ℕ) : ℕ :=
  num_students * artworks_per_student_per_quarter * quarters_per_year * num_years

theorem art_club_two_years_collection : 
  art_club_artworks 15 2 4 2 = 240 := by sorry

end art_club_artworks_art_club_two_years_collection_l2371_237160


namespace sqrt_x_minus_2_real_l2371_237165

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by sorry

end sqrt_x_minus_2_real_l2371_237165


namespace boys_dropped_out_l2371_237189

/-- Proves the number of boys who dropped out from a school, given initial counts and final total -/
theorem boys_dropped_out (initial_boys initial_girls girls_dropped final_total : ℕ) : 
  initial_boys = 14 →
  initial_girls = 10 →
  girls_dropped = 3 →
  final_total = 17 →
  initial_boys - (final_total - (initial_girls - girls_dropped)) = 4 :=
by sorry

end boys_dropped_out_l2371_237189


namespace subtract_sum_digits_100_times_is_zero_l2371_237125

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value < 1000

/-- Computes the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Performs one iteration of subtracting the sum of digits -/
def subtract_sum_of_digits (n : ThreeDigitNumber) : ℕ := 
  n.value - sum_of_digits n.value

/-- Performs the subtraction process n times -/
def iterate_subtraction (n : ThreeDigitNumber) (iterations : ℕ) : ℕ := sorry

/-- Theorem: After 100 iterations of subtracting the sum of digits from any three-digit number, the result is zero -/
theorem subtract_sum_digits_100_times_is_zero (n : ThreeDigitNumber) : 
  iterate_subtraction n 100 = 0 := by sorry

end subtract_sum_digits_100_times_is_zero_l2371_237125


namespace tangent_slope_at_pi_l2371_237133

theorem tangent_slope_at_pi (f : ℝ → ℝ) (h : f = λ x => 2*x + Real.sin x) :
  HasDerivAt f 1 π := by sorry

end tangent_slope_at_pi_l2371_237133


namespace sum_of_squares_of_roots_l2371_237181

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (5 * x₁^2 - 7 * x₁ + 2 = 0) → 
  (5 * x₂^2 - 7 * x₂ + 2 = 0) → 
  (x₁^2 + x₂^2 = 29/25) :=
by sorry

end sum_of_squares_of_roots_l2371_237181


namespace dans_age_l2371_237156

theorem dans_age (x : ℕ) : x + 20 = 7 * (x - 4) → x = 8 := by
  sorry

end dans_age_l2371_237156


namespace A_power_50_l2371_237152

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  !![5, 2; -16, -6]

theorem A_power_50 : A^50 = !![301, 100; -800, -249] := by
  sorry

end A_power_50_l2371_237152


namespace total_weight_of_rings_l2371_237157

-- Define the weights of the rings
def orange_weight : ℚ := 0.08
def purple_weight : ℚ := 0.33
def white_weight : ℚ := 0.42

-- Theorem statement
theorem total_weight_of_rings : orange_weight + purple_weight + white_weight = 0.83 := by
  sorry

end total_weight_of_rings_l2371_237157


namespace vector_equation_l2371_237182

def a : ℝ × ℝ × ℝ := (-1, 3, 2)
def b : ℝ × ℝ × ℝ := (4, -6, 2)
def c (t : ℝ) : ℝ × ℝ × ℝ := (-3, 12, t)

theorem vector_equation (m n t : ℝ) :
  c t = m • a + n • b → t = 11 ∧ m + n = 11/2 := by
  sorry

end vector_equation_l2371_237182


namespace min_draws_for_even_product_l2371_237146

theorem min_draws_for_even_product (n : ℕ) (h : n = 16) :
  let S := Finset.range n
  let even_count := (S.filter (λ x => x % 2 = 0)).card
  let odd_count := (S.filter (λ x => x % 2 ≠ 0)).card
  odd_count + 1 = 9 ∧ 
  ∀ k : ℕ, k < odd_count + 1 → ∃ subset : Finset ℕ, 
    subset.card = k ∧ 
    subset ⊆ S ∧ 
    ∀ x ∈ subset, x % 2 ≠ 0 :=
by sorry

end min_draws_for_even_product_l2371_237146


namespace train_overtake_time_l2371_237108

/-- The time it takes for a train to overtake a motorbike -/
theorem train_overtake_time (train_speed : ℝ) (motorbike_speed : ℝ) (train_length : ℝ) :
  train_speed = 100 →
  motorbike_speed = 64 →
  train_length = 180.0144 →
  (train_length / ((train_speed - motorbike_speed) / 3.6)) = 18.00144 := by
  sorry

end train_overtake_time_l2371_237108


namespace pure_imaginary_product_l2371_237107

theorem pure_imaginary_product (a : ℝ) : 
  (∃ b : ℝ, (2*a + Complex.I) * (1 - 2*Complex.I) = b * Complex.I ∧ b ≠ 0) → a = -1 := by
  sorry

end pure_imaginary_product_l2371_237107


namespace knight_traversal_coloring_l2371_237115

/-- Represents a chessboard of arbitrary size -/
structure Chessboard where
  size : ℕ
  canBeTraversed : Bool

/-- Represents a position on the chessboard -/
structure Position where
  x : ℕ
  y : ℕ

/-- Represents a knight's move -/
def knightMove (p : Position) : Position :=
  sorry

/-- Checks if a position is even in the knight's traversal -/
def isEvenPosition (p : Position) : Bool :=
  sorry

/-- Checks if a position should be colored black in a properly colored chessboard -/
def isBlackInProperColoring (p : Position) : Bool :=
  sorry

/-- The main theorem stating that shading even-numbered squares in a knight's traversal
    reproduces the proper coloring of a chessboard -/
theorem knight_traversal_coloring (board : Chessboard) :
  board.canBeTraversed →
  ∀ p : Position, isEvenPosition p = isBlackInProperColoring p :=
sorry

end knight_traversal_coloring_l2371_237115


namespace area_at_stage_8_l2371_237124

/-- Represents the width of a rectangle at a given stage -/
def width (stage : ℕ) : ℕ :=
  if stage ≤ 4 then 4 else 2 * stage - 6

/-- Represents the area of a rectangle at a given stage -/
def area (stage : ℕ) : ℕ := 4 * width stage

/-- The total area of the figure at Stage 8 -/
def totalArea : ℕ := (List.range 8).map (fun i => area (i + 1)) |>.sum

theorem area_at_stage_8 : totalArea = 176 := by
  sorry

end area_at_stage_8_l2371_237124


namespace number_ratio_problem_l2371_237109

theorem number_ratio_problem (x : ℚ) : 
  (x / 6 = 16 / 480) → x = 1/5 := by
sorry

end number_ratio_problem_l2371_237109


namespace percentage_calculation_l2371_237179

theorem percentage_calculation (x : ℝ) (h : 0.2 * x = 1000) : 1.2 * x = 6000 := by
  sorry

end percentage_calculation_l2371_237179


namespace greatest_perimeter_l2371_237139

/-- A rectangle with whole number side lengths and an area of 12 square metres. -/
structure Rectangle where
  width : ℕ
  length : ℕ
  area_eq : width * length = 12

/-- The perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.length)

/-- The theorem stating that the greatest possible perimeter is 26. -/
theorem greatest_perimeter :
  ∀ r : Rectangle, perimeter r ≤ 26 ∧ ∃ r' : Rectangle, perimeter r' = 26 := by
  sorry

end greatest_perimeter_l2371_237139


namespace cindys_homework_l2371_237127

theorem cindys_homework (x : ℝ) : (x - 7) * 4 = 48 → (x * 4) - 7 = 69 := by
  sorry

end cindys_homework_l2371_237127


namespace same_suit_probability_l2371_237144

theorem same_suit_probability (total_cards : ℕ) (num_suits : ℕ) (cards_per_suit : ℕ) 
  (h1 : total_cards = 52)
  (h2 : num_suits = 4)
  (h3 : cards_per_suit = 13)
  (h4 : total_cards = num_suits * cards_per_suit) :
  (4 : ℚ) / 17 = (num_suits * (cards_per_suit.choose 2)) / (total_cards.choose 2) :=
by sorry

end same_suit_probability_l2371_237144


namespace coefficient_of_x_squared_l2371_237142

-- Define the polynomials
def p1 (x : ℝ) : ℝ := 3 * x^3 - 4 * x^2 + 5 * x - 2
def p2 (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 4

-- Define the product of the polynomials
def product (x : ℝ) : ℝ := p1 x * p2 x

-- Theorem statement
theorem coefficient_of_x_squared :
  ∃ (a b c d : ℝ), product = fun x => a * x^3 + (-5) * x^2 + b * x + c + d * x^4 :=
sorry

end coefficient_of_x_squared_l2371_237142


namespace system_solution_ratio_l2371_237132

theorem system_solution_ratio (x y c d : ℝ) : 
  x ≠ 0 →
  y ≠ 0 →
  d ≠ 0 →
  8 * x - 6 * y = c →
  10 * y - 15 * x = d →
  c / d = -2 / 5 := by
sorry

end system_solution_ratio_l2371_237132


namespace biggest_measure_for_containers_l2371_237137

theorem biggest_measure_for_containers (a b c : ℕ) 
  (ha : a = 496) (hb : b = 403) (hc : c = 713) : 
  Nat.gcd a (Nat.gcd b c) = 31 := by
  sorry

end biggest_measure_for_containers_l2371_237137


namespace sum_of_coordinates_equals_16_l2371_237110

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 6 = (y - 2)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem sum_of_coordinates_equals_16 :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧ (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 16 :=
by
  sorry

end sum_of_coordinates_equals_16_l2371_237110


namespace min_distance_between_ships_l2371_237148

/-- The minimum distance between two ships given specific conditions -/
theorem min_distance_between_ships 
  (d : ℝ) -- Initial distance between ships
  (k : ℝ) -- Speed ratio v₁/v₂
  (h₁ : k > 0) -- Speed ratio is positive
  (h₂ : k < 1) -- Speed ratio is less than 1
  : ∃ (min_dist : ℝ), min_dist = d * Real.sqrt (1 - k^2) :=
by
  sorry

end min_distance_between_ships_l2371_237148


namespace constant_calculation_l2371_237126

theorem constant_calculation (N : ℝ) (C : ℝ) : 
  N = 12.0 → C + 0.6667 * N = 0.75 * N → C = 0.9996 := by
  sorry

end constant_calculation_l2371_237126


namespace dot_product_theorem_l2371_237190

variable (a b : ℝ × ℝ)

theorem dot_product_theorem (h1 : a.1 + 2 * b.1 = 0 ∧ a.2 + 2 * b.2 = 0) 
                            (h2 : (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 2) : 
  a.1 * b.1 + a.2 * b.2 = -2 := by
  sorry

end dot_product_theorem_l2371_237190


namespace negative_one_minus_two_times_negative_two_l2371_237151

theorem negative_one_minus_two_times_negative_two : -1 - 2 * (-2) = 3 := by
  sorry

end negative_one_minus_two_times_negative_two_l2371_237151


namespace det_necessary_not_sufficient_for_parallel_l2371_237158

/-- Determinant of a 2x2 matrix --/
def det (a₁ b₁ a₂ b₂ : ℝ) : ℝ := a₁ * b₂ - a₂ * b₁

/-- Two lines are parallel --/
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a₁ = k * a₂ ∧ b₁ = k * b₂ ∧ c₁ ≠ k * c₂

theorem det_necessary_not_sufficient_for_parallel
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) (h₁ : a₁^2 + b₁^2 ≠ 0) (h₂ : a₂^2 + b₂^2 ≠ 0) :
  (det a₁ b₁ a₂ b₂ = 0 → parallel a₁ b₁ c₁ a₂ b₂ c₂) ∧
  ¬(parallel a₁ b₁ c₁ a₂ b₂ c₂ → det a₁ b₁ a₂ b₂ = 0) :=
sorry

end det_necessary_not_sufficient_for_parallel_l2371_237158


namespace nicholas_bottle_caps_l2371_237170

theorem nicholas_bottle_caps :
  let initial_caps : ℕ := 8
  let additional_caps : ℕ := 85
  initial_caps + additional_caps = 93
:= by sorry

end nicholas_bottle_caps_l2371_237170


namespace quadratic_equal_roots_l2371_237111

theorem quadratic_equal_roots : ∃ x : ℝ, x^2 - x + (1/4 : ℝ) = 0 ∧
  ∀ y : ℝ, y^2 - y + (1/4 : ℝ) = 0 → y = x :=
by sorry

end quadratic_equal_roots_l2371_237111


namespace triangle_properties_l2371_237154

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ  -- Angle A in radians
  b : ℝ  -- Side length b
  c : ℝ  -- Side length c
  h1 : A = π / 3  -- A = 60° in radians
  h2 : b = 5
  h3 : c = 4

/-- Main theorem about the triangle -/
theorem triangle_properties (t : Triangle) :
  ∃ (a : ℝ), 
    a ^ 2 = 21 ∧ 
    Real.sin (Real.arcsin (t.b / a)) * Real.sin (Real.arcsin (t.c / a)) = 5 / 7 := by
  sorry

end triangle_properties_l2371_237154


namespace product_remainder_l2371_237118

theorem product_remainder (a b m : ℕ) (h : a = 98) (h' : b = 102) (h'' : m = 8) :
  (a * b) % m = 4 := by
  sorry

end product_remainder_l2371_237118


namespace quadratic_single_solution_l2371_237101

theorem quadratic_single_solution (a : ℝ) (ha : a ≠ 0) :
  (∃! x : ℝ, a * x^2 + 20 * x + 7 = 0) → 
  (∀ x : ℝ, a * x^2 + 20 * x + 7 = 0 → x = -7/10) :=
by sorry

end quadratic_single_solution_l2371_237101


namespace ceiling_negative_three_point_seven_l2371_237166

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end ceiling_negative_three_point_seven_l2371_237166


namespace used_car_clients_l2371_237196

theorem used_car_clients (num_cars : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) :
  num_cars = 18 →
  selections_per_client = 3 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / selections_per_client = 18 := by
  sorry

end used_car_clients_l2371_237196


namespace direction_525_to_527_l2371_237168

/-- Represents the directions of movement -/
inductive Direction
| Right
| Up
| Left
| Down
| Diagonal

/-- Defines the cyclic pattern of directions -/
def directionPattern : Fin 5 → Direction
| 0 => Direction.Right
| 1 => Direction.Up
| 2 => Direction.Left
| 3 => Direction.Down
| 4 => Direction.Diagonal

/-- Returns the direction for a given point number -/
def directionAtPoint (n : Nat) : Direction :=
  directionPattern (n % 5)

/-- Theorem: The sequence of directions from point 525 to 527 is Right, Up -/
theorem direction_525_to_527 :
  (directionAtPoint 525, directionAtPoint 526) = (Direction.Right, Direction.Up) := by
  sorry

#check direction_525_to_527

end direction_525_to_527_l2371_237168


namespace salary_increase_after_reduction_l2371_237147

theorem salary_increase_after_reduction : ∀ (original_salary : ℝ),
  original_salary > 0 →
  let reduced_salary := original_salary * (1 - 0.25)
  let increase_factor := (1 + 1/3)
  reduced_salary * increase_factor = original_salary :=
by
  sorry

end salary_increase_after_reduction_l2371_237147


namespace michaels_pets_cats_percentage_l2371_237104

/-- Proves that the percentage of cats among Michael's pets is 50% -/
theorem michaels_pets_cats_percentage
  (total_pets : ℕ)
  (dog_percentage : ℚ)
  (bunny_count : ℕ)
  (h1 : total_pets = 36)
  (h2 : dog_percentage = 1/4)
  (h3 : bunny_count = 9)
  (h4 : (dog_percentage * total_pets).num + bunny_count + (total_pets - (dog_percentage * total_pets).num - bunny_count) = total_pets) :
  (total_pets - (dog_percentage * total_pets).num - bunny_count) / total_pets = 1/2 := by
  sorry

#check michaels_pets_cats_percentage

end michaels_pets_cats_percentage_l2371_237104


namespace system_negative_solution_l2371_237191

/-- The system of equations has at least one negative solution if and only if a + b + c = 0 -/
theorem system_negative_solution (a b c : ℝ) :
  (∃ (x y : ℝ), x < 0 ∧ y < 0 ∧
    a * x + b * y = c ∧
    b * x + c * y = a ∧
    c * x + a * y = b) ↔
  a + b + c = 0 := by
  sorry

end system_negative_solution_l2371_237191


namespace shoe_picking_probability_l2371_237167

def total_pairs : ℕ := 15
def black_pairs : ℕ := 8
def blue_pairs : ℕ := 4
def green_pairs : ℕ := 3

def total_shoes : ℕ := 2 * total_pairs

theorem shoe_picking_probability :
  let black_shoes : ℕ := 2 * black_pairs
  let blue_shoes : ℕ := 2 * blue_pairs
  let green_shoes : ℕ := 2 * green_pairs
  let prob_black := (black_shoes : ℚ) / total_shoes * (black_pairs : ℚ) / (total_shoes - 1)
  let prob_blue := (blue_shoes : ℚ) / total_shoes * (blue_pairs : ℚ) / (total_shoes - 1)
  let prob_green := (green_shoes : ℚ) / total_shoes * (green_pairs : ℚ) / (total_shoes - 1)
  prob_black + prob_blue + prob_green = 89 / 435 :=
by sorry

end shoe_picking_probability_l2371_237167


namespace problem_solution_l2371_237153

theorem problem_solution : 
  ((-1)^3 + |1 - Real.sqrt 2| + (8 : ℝ)^(1/3) = Real.sqrt 2) ∧
  (((-5 : ℝ)^3)^(1/3) + (-3)^2 - Real.sqrt 25 + |Real.sqrt 3 - 2| + (Real.sqrt 3)^2 = 4 - Real.sqrt 3) := by
  sorry

end problem_solution_l2371_237153


namespace simplify_and_evaluate_l2371_237171

theorem simplify_and_evaluate :
  ∀ x : ℝ, x ≠ 1 → x ≠ 3 →
  (1 - 2 / (x - 1)) * ((x^2 - x) / (x^2 - 6*x + 9)) = x / (x - 3) ∧
  (2 : ℝ) / ((2 : ℝ) - 3) = -2 :=
by sorry

end simplify_and_evaluate_l2371_237171


namespace vector_subtraction_l2371_237197

/-- Given vectors a and b in ℝ², prove that a - 2b equals (6, -7) -/
theorem vector_subtraction (a b : ℝ × ℝ) 
  (ha : a = (2, -1)) (hb : b = (-2, 3)) : 
  a - 2 • b = (6, -7) := by
  sorry

end vector_subtraction_l2371_237197


namespace pumpkin_patch_pie_filling_l2371_237192

/-- Calculates the number of cans of pie filling produced given the total pumpkins,
    price per pumpkin, total money made, and pumpkins per can. -/
def cans_of_pie_filling (total_pumpkins : ℕ) (price_per_pumpkin : ℕ) 
                        (total_made : ℕ) (pumpkins_per_can : ℕ) : ℕ :=
  (total_pumpkins - total_made / price_per_pumpkin) / pumpkins_per_can

/-- Theorem stating that given the specific conditions, 
    the number of cans of pie filling produced is 17. -/
theorem pumpkin_patch_pie_filling : 
  cans_of_pie_filling 83 3 96 3 = 17 := by
  sorry

end pumpkin_patch_pie_filling_l2371_237192


namespace sum_squared_l2371_237117

theorem sum_squared (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 + b^2 = 25) : (a + b)^2 = 49 := by
  sorry

end sum_squared_l2371_237117


namespace black_ball_prob_compare_l2371_237134

-- Define the number of balls in each box
def box_a_red : ℕ := 40
def box_a_black : ℕ := 10
def box_b_red : ℕ := 60
def box_b_black : ℕ := 40
def box_b_white : ℕ := 50

-- Define the total number of balls in each box
def total_a : ℕ := box_a_red + box_a_black
def total_b : ℕ := box_b_red + box_b_black + box_b_white

-- Define the probabilities of drawing a black ball from each box
def prob_a : ℚ := box_a_black / total_a
def prob_b : ℚ := box_b_black / total_b

-- Theorem statement
theorem black_ball_prob_compare : prob_b > prob_a := by
  sorry

end black_ball_prob_compare_l2371_237134


namespace square_product_closed_l2371_237102

def P : Set ℕ := {n : ℕ | ∃ m : ℕ+, n = m ^ 2}

theorem square_product_closed (a b : ℕ) (ha : a ∈ P) (hb : b ∈ P) : 
  a * b ∈ P := by sorry

end square_product_closed_l2371_237102


namespace polynomial_simplification_l2371_237155

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + 3 * x^3 - 5 * x + 6) + (-6 * x^4 - 2 * x^3 + 3 * x^2 + 5 * x - 4) =
  -4 * x^4 + x^3 + 3 * x^2 + 2 := by
  sorry

end polynomial_simplification_l2371_237155


namespace ascending_order_proof_l2371_237188

def base_16_to_decimal (n : ℕ) : ℕ := n

def base_7_to_decimal (n : ℕ) : ℕ := n

def base_4_to_decimal (n : ℕ) : ℕ := n

theorem ascending_order_proof (a b c : ℕ) 
  (ha : a = base_16_to_decimal 0x12)
  (hb : b = base_7_to_decimal 25)
  (hc : c = base_4_to_decimal 33) :
  c < a ∧ a < b := by
  sorry

end ascending_order_proof_l2371_237188


namespace sufficient_condition_for_product_greater_than_one_l2371_237162

theorem sufficient_condition_for_product_greater_than_one :
  ∀ (a b : ℝ), a > 1 ∧ b > 1 → a * b > 1 := by
  sorry

end sufficient_condition_for_product_greater_than_one_l2371_237162


namespace manufacturing_cost_is_210_l2371_237105

/-- Calculates the manufacturing cost of a shoe given transportation cost, selling price, and gain percentage. -/
def manufacturing_cost (transportation_cost : ℚ) (shoes_per_transport : ℕ) (selling_price : ℚ) (gain_percentage : ℚ) : ℚ :=
  let transportation_cost_per_shoe := transportation_cost / shoes_per_transport
  let cost_price := selling_price / (1 + gain_percentage)
  cost_price - transportation_cost_per_shoe

/-- Proves that the manufacturing cost of a shoe is 210, given the specified conditions. -/
theorem manufacturing_cost_is_210 :
  manufacturing_cost 500 100 258 (20/100) = 210 := by
  sorry

#eval manufacturing_cost 500 100 258 (20/100)

end manufacturing_cost_is_210_l2371_237105


namespace max_min_difference_d_l2371_237119

theorem max_min_difference_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 5)
  (sum_sq_eq : a^2 + b^2 + c^2 + d^2 = 18) : 
  ∃ (d_max d_min : ℝ),
    (∀ d', a + b + c + d' = 5 ∧ a^2 + b^2 + c^2 + d'^2 = 18 → d' ≤ d_max) ∧
    (∀ d', a + b + c + d' = 5 ∧ a^2 + b^2 + c^2 + d'^2 = 18 → d_min ≤ d') ∧
    d_max - d_min = 6.75 := by
  sorry

end max_min_difference_d_l2371_237119


namespace line_intersects_x_axis_at_2_0_l2371_237187

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- The x-axis -/
def x_axis : Line := { p1 := ⟨0, 0⟩, p2 := ⟨1, 0⟩ }

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

/-- Check if a point lies on the x-axis -/
def point_on_x_axis (p : Point) : Prop := p.y = 0

/-- The main theorem -/
theorem line_intersects_x_axis_at_2_0 :
  let l : Line := { p1 := ⟨4, -2⟩, p2 := ⟨0, 2⟩ }
  let intersection : Point := ⟨2, 0⟩
  point_on_line intersection l ∧ point_on_x_axis intersection := by
  sorry

end line_intersects_x_axis_at_2_0_l2371_237187


namespace tan_45_degrees_l2371_237149

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_l2371_237149


namespace extremum_implies_f_2_l2371_237145

/-- A function f with an extremum at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f_2 (a b : ℝ) :
  f' a b 1 = 0 → f a b 1 = 10 → f a b 2 = 18 := by
  sorry

end extremum_implies_f_2_l2371_237145


namespace sum_remainder_l2371_237131

theorem sum_remainder (n : ℤ) : (7 - 2*n + (n + 5)) % 8 = (4 - n) % 8 := by
  sorry

end sum_remainder_l2371_237131


namespace infinite_divisible_by_76_and_unique_centers_l2371_237184

/-- Represents a cell in the spiral grid -/
structure Cell where
  x : ℤ
  y : ℤ

/-- The value at a node of the grid -/
def node_value (c : Cell) : ℕ := sorry

/-- The value at the center of a cell -/
def center_value (c : Cell) : ℕ := sorry

/-- The set of all cells in the infinite grid -/
def all_cells : Set Cell := sorry

theorem infinite_divisible_by_76_and_unique_centers :
  (∃ (S : Set Cell), Set.Infinite S ∧ ∀ c ∈ S, 76 ∣ center_value c) ∧
  (∀ c₁ c₂ : Cell, c₁ ≠ c₂ → center_value c₁ ≠ center_value c₂) := by
  sorry

end infinite_divisible_by_76_and_unique_centers_l2371_237184


namespace correct_philosophies_l2371_237183

-- Define the philosophies
inductive Philosophy
  | GraspMeasure
  | ComprehensiveView
  | AnalyzeSpecifically
  | EmphasizeKeyPoints

-- Define the conditions
structure IodineScenario where
  iodineEssential : Bool
  oneSizeFitsAllRisky : Bool
  nonIodineDeficientArea : Bool
  increasedNonIodizedSalt : Bool
  allowAdjustment : Bool

-- Define the function to check if a philosophy is reflected
def reflectsPhilosophy (scenario : IodineScenario) (philosophy : Philosophy) : Prop :=
  match philosophy with
  | Philosophy.GraspMeasure => scenario.oneSizeFitsAllRisky
  | Philosophy.ComprehensiveView => scenario.iodineEssential ∧ scenario.oneSizeFitsAllRisky
  | Philosophy.AnalyzeSpecifically => scenario.nonIodineDeficientArea ∧ scenario.increasedNonIodizedSalt ∧ scenario.allowAdjustment
  | Philosophy.EmphasizeKeyPoints => False

-- Theorem to prove
theorem correct_philosophies (scenario : IodineScenario) 
  (h1 : scenario.iodineEssential = true)
  (h2 : scenario.oneSizeFitsAllRisky = true)
  (h3 : scenario.nonIodineDeficientArea = true)
  (h4 : scenario.increasedNonIodizedSalt = true)
  (h5 : scenario.allowAdjustment = true) :
  reflectsPhilosophy scenario Philosophy.GraspMeasure ∧
  reflectsPhilosophy scenario Philosophy.ComprehensiveView ∧
  reflectsPhilosophy scenario Philosophy.AnalyzeSpecifically ∧
  ¬reflectsPhilosophy scenario Philosophy.EmphasizeKeyPoints :=
sorry

end correct_philosophies_l2371_237183


namespace infinite_intersection_l2371_237185

def sequence_a : ℕ → ℤ
| 0 => 2
| 1 => 14
| (n + 2) => 14 * sequence_a (n + 1) + sequence_a n

def sequence_b : ℕ → ℤ
| 0 => 2
| 1 => 14
| (n + 2) => 6 * sequence_b (n + 1) - sequence_b n

theorem infinite_intersection :
  Set.Infinite {n : ℕ | ∃ m : ℕ, sequence_a n = sequence_b m} :=
sorry

end infinite_intersection_l2371_237185


namespace town_population_l2371_237199

theorem town_population (P : ℝ) : 
  (P * (1 - 0.2)^2 = 12800) → P = 20000 := by
  sorry

end town_population_l2371_237199


namespace chessboard_coloring_limit_l2371_237163

/-- Represents the minimum number of colored vertices required on an n × n chessboard
    such that any k × k square has at least one edge with a colored vertex. -/
noncomputable def l (n : ℕ) : ℕ := sorry

/-- The limit of l(n)/n² as n approaches infinity is 2/7. -/
theorem chessboard_coloring_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |l n / (n^2 : ℝ) - 2/7| < ε :=
sorry

end chessboard_coloring_limit_l2371_237163
