import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3620_362034

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 230) : x + y = 660 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3620_362034


namespace NUMINAMATH_CALUDE_watermelon_pineapple_weight_difference_l3620_362019

/-- Given that 4 watermelons weigh 5200 grams and 3 watermelons plus 4 pineapples
    weigh 5700 grams, prove that a watermelon is 850 grams heavier than a pineapple. -/
theorem watermelon_pineapple_weight_difference :
  let watermelon_weight : ℕ := 5200 / 4
  let pineapple_weight : ℕ := (5700 - 3 * watermelon_weight) / 4
  watermelon_weight - pineapple_weight = 850 := by
sorry

end NUMINAMATH_CALUDE_watermelon_pineapple_weight_difference_l3620_362019


namespace NUMINAMATH_CALUDE_folded_square_distance_l3620_362027

/-- Given a square sheet of paper with area 18 cm², prove that when folded so that
    a corner touches the line from midpoint of adjacent side to opposite corner,
    creating equal visible areas, the distance from corner to original position is 3 cm. -/
theorem folded_square_distance (s : ℝ) (h1 : s^2 = 18) : 
  let d := s * Real.sqrt 2 / 2
  d = 3 := by sorry

end NUMINAMATH_CALUDE_folded_square_distance_l3620_362027


namespace NUMINAMATH_CALUDE_special_triangle_sum_range_l3620_362014

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Ensure angles are positive and sum to π
  angle_sum : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  -- Ensure sides are positive
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Define the specific conditions for our triangle
def SpecialTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 + t.a * t.b = 4 ∧ t.c = 2

-- State the theorem
theorem special_triangle_sum_range (t : Triangle) (h : SpecialTriangle t) :
  2 < 2 * t.a + t.b ∧ 2 * t.a + t.b < 4 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sum_range_l3620_362014


namespace NUMINAMATH_CALUDE_cylinder_volume_scaling_l3620_362035

theorem cylinder_volume_scaling (r h V : ℝ) :
  V = π * r^2 * h →
  ∀ (k : ℝ), k > 0 →
    π * (k*r)^2 * (k*h) = k^3 * V :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_scaling_l3620_362035


namespace NUMINAMATH_CALUDE_unique_solution_l3620_362083

def is_valid_digit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 8

def are_distinct (a b c d e f g h : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h

def four_digit_number (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

theorem unique_solution (a b c d e f g h : ℕ) :
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧
  is_valid_digit e ∧ is_valid_digit f ∧ is_valid_digit g ∧ is_valid_digit h ∧
  are_distinct a b c d e f g h ∧
  four_digit_number a b c d + e * f * g * h = 2011 →
  four_digit_number a b c d = 1563 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l3620_362083


namespace NUMINAMATH_CALUDE_real_estate_investment_l3620_362021

def total_investment : ℝ := 200000
def real_estate_ratio : ℝ := 7

theorem real_estate_investment (mutual_funds : ℝ) 
  (h1 : mutual_funds + real_estate_ratio * mutual_funds = total_investment) :
  real_estate_ratio * mutual_funds = 175000 := by
  sorry

end NUMINAMATH_CALUDE_real_estate_investment_l3620_362021


namespace NUMINAMATH_CALUDE_english_not_russian_count_l3620_362044

/-- Represents the set of teachers who know English -/
def E : Finset Nat := sorry

/-- Represents the set of teachers who know Russian -/
def R : Finset Nat := sorry

theorem english_not_russian_count :
  (E.card = 75) →
  (R.card = 55) →
  ((E ∩ R).card = 110) →
  ((E \ R).card = 55) := by
  sorry

end NUMINAMATH_CALUDE_english_not_russian_count_l3620_362044


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l3620_362055

def grape_quantity : ℝ := 7
def grape_rate : ℝ := 70
def mango_quantity : ℝ := 9
def total_paid : ℝ := 985

theorem mango_rate_calculation :
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 55 :=
by sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l3620_362055


namespace NUMINAMATH_CALUDE_pascal_triangle_prob_one_20_l3620_362013

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : Set ℕ := sorry

/-- The number of elements in the first n rows of Pascal's Triangle -/
def numElements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of ones in the first n rows of Pascal's Triangle -/
def numOnes (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def probSelectOne (n : ℕ) : ℚ := (numOnes n : ℚ) / (numElements n : ℚ)

theorem pascal_triangle_prob_one_20 : 
  probSelectOne 20 = 39 / 210 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_prob_one_20_l3620_362013


namespace NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l3620_362089

theorem greatest_x_quadratic_inequality :
  ∀ x : ℝ, x^2 - 16*x + 63 ≤ 0 → x ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l3620_362089


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3620_362077

theorem trig_identity_proof : 
  Real.sin (21 * π / 180) * Real.cos (81 * π / 180) - 
  Real.cos (21 * π / 180) * Real.sin (81 * π / 180) = 
  -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3620_362077


namespace NUMINAMATH_CALUDE_concert_total_cost_l3620_362076

/-- Calculate the total cost of a concert for two people -/
theorem concert_total_cost
  (ticket_price : ℚ)
  (num_people : ℕ)
  (processing_fee_rate : ℚ)
  (parking_fee : ℚ)
  (entrance_fee_per_person : ℚ)
  (refreshments_cost : ℚ)
  (tshirts_cost : ℚ)
  (h1 : ticket_price = 75)
  (h2 : num_people = 2)
  (h3 : processing_fee_rate = 0.15)
  (h4 : parking_fee = 10)
  (h5 : entrance_fee_per_person = 5)
  (h6 : refreshments_cost = 20)
  (h7 : tshirts_cost = 40) :
  let total_ticket_cost := ticket_price * num_people
  let processing_fee := total_ticket_cost * processing_fee_rate
  let entrance_fee_total := entrance_fee_per_person * num_people
  ticket_price * num_people +
  total_ticket_cost * processing_fee_rate +
  parking_fee +
  entrance_fee_total +
  refreshments_cost +
  tshirts_cost = 252.5 := by
    sorry


end NUMINAMATH_CALUDE_concert_total_cost_l3620_362076


namespace NUMINAMATH_CALUDE_problems_finished_at_school_l3620_362037

def math_problems : ℕ := 18
def science_problems : ℕ := 11
def problems_left : ℕ := 5

theorem problems_finished_at_school :
  math_problems + science_problems - problems_left = 24 := by
  sorry

end NUMINAMATH_CALUDE_problems_finished_at_school_l3620_362037


namespace NUMINAMATH_CALUDE_work_increase_per_person_l3620_362022

/-- Calculates the increase in work per person when 1/6 of the workforce is absent -/
theorem work_increase_per_person (p : ℕ) (W : ℝ) (h : p > 0) :
  let initial_work_per_person := W / p
  let remaining_workers := p - p / 6
  let new_work_per_person := W / remaining_workers
  new_work_per_person - initial_work_per_person = W / (5 * p) :=
by sorry

end NUMINAMATH_CALUDE_work_increase_per_person_l3620_362022


namespace NUMINAMATH_CALUDE_range_of_a_l3620_362063

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) → a > -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3620_362063


namespace NUMINAMATH_CALUDE_jake_sausage_cost_l3620_362098

-- Define the parameters
def package_weight : ℝ := 2
def num_packages : ℕ := 3
def price_per_pound : ℝ := 4

-- Define the theorem
theorem jake_sausage_cost :
  package_weight * num_packages * price_per_pound = 24 := by
  sorry

end NUMINAMATH_CALUDE_jake_sausage_cost_l3620_362098


namespace NUMINAMATH_CALUDE_red_and_large_toys_l3620_362048

/-- Represents the color of a toy -/
inductive Color
| Red
| Green
| Blue
| Yellow
| Orange

/-- Represents the size of a toy -/
inductive Size
| Small
| Medium
| Large
| ExtraLarge

/-- Represents the distribution of toys by color and size -/
structure ToyDistribution where
  red_small : Rat
  red_medium : Rat
  red_large : Rat
  red_extra_large : Rat
  green_small : Rat
  green_medium : Rat
  green_large : Rat
  green_extra_large : Rat
  blue_small : Rat
  blue_medium : Rat
  blue_large : Rat
  blue_extra_large : Rat
  yellow_small : Rat
  yellow_medium : Rat
  yellow_large : Rat
  yellow_extra_large : Rat
  orange_small : Rat
  orange_medium : Rat
  orange_large : Rat
  orange_extra_large : Rat

/-- The given distribution of toys -/
def given_distribution : ToyDistribution :=
  { red_small := 6/100, red_medium := 8/100, red_large := 7/100, red_extra_large := 4/100,
    green_small := 4/100, green_medium := 7/100, green_large := 5/100, green_extra_large := 4/100,
    blue_small := 6/100, blue_medium := 3/100, blue_large := 4/100, blue_extra_large := 2/100,
    yellow_small := 8/100, yellow_medium := 10/100, yellow_large := 5/100, yellow_extra_large := 2/100,
    orange_small := 9/100, orange_medium := 6/100, orange_large := 5/100, orange_extra_large := 5/100 }

/-- Theorem stating the number of red and large toys -/
theorem red_and_large_toys (total_toys : ℕ) (h : total_toys * given_distribution.green_large = 47) :
  total_toys * given_distribution.red_large = 329 := by
  sorry

end NUMINAMATH_CALUDE_red_and_large_toys_l3620_362048


namespace NUMINAMATH_CALUDE_relay_race_sarah_speed_l3620_362097

/-- Relay race problem -/
theorem relay_race_sarah_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (sadie_speed : ℝ) 
  (sadie_time : ℝ) 
  (ariana_speed : ℝ) 
  (ariana_time : ℝ) 
  (h1 : total_distance = 17) 
  (h2 : total_time = 4.5) 
  (h3 : sadie_speed = 3) 
  (h4 : sadie_time = 2) 
  (h5 : ariana_speed = 6) 
  (h6 : ariana_time = 0.5) : 
  (total_distance - (sadie_speed * sadie_time + ariana_speed * ariana_time)) / 
  (total_time - sadie_time - ariana_time) = 4 := by
  sorry


end NUMINAMATH_CALUDE_relay_race_sarah_speed_l3620_362097


namespace NUMINAMATH_CALUDE_unusual_bicycle_spokes_l3620_362039

/-- A bicycle with an unusual spoke configuration. -/
structure Bicycle where
  front_spokes : ℕ
  back_spokes : ℕ

/-- The total number of spokes on a bicycle. -/
def total_spokes (b : Bicycle) : ℕ := b.front_spokes + b.back_spokes

/-- Theorem: The total number of spokes on the unusual bicycle is 60. -/
theorem unusual_bicycle_spokes :
  ∃ (b : Bicycle), b.front_spokes = 20 ∧ b.back_spokes = 2 * b.front_spokes ∧ total_spokes b = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_unusual_bicycle_spokes_l3620_362039


namespace NUMINAMATH_CALUDE_tangent_line_at_pi_third_l3620_362052

noncomputable def f (x : ℝ) : ℝ := (1/2) * x + Real.sin x

def tangent_line_equation (x y : ℝ) : Prop :=
  6 * x - 6 * y + 3 * Real.sqrt 3 - Real.pi = 0

theorem tangent_line_at_pi_third :
  tangent_line_equation (π/3) (f (π/3)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_pi_third_l3620_362052


namespace NUMINAMATH_CALUDE_square_root_problem_l3620_362065

theorem square_root_problem (m n : ℝ) (hm : m = 3^2) (hn : n = (-4)^3) :
  Real.sqrt (2 * m - n - 1) = 9 ∨ Real.sqrt (2 * m - n - 1) = -9 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l3620_362065


namespace NUMINAMATH_CALUDE_combination_sum_theorem_l3620_362081

theorem combination_sum_theorem : 
  ∃ (n : ℕ+), 
    (0 ≤ 38 - n.val ∧ 38 - n.val ≤ 3 * n.val) ∧ 
    (n.val + 21 ≥ 3 * n.val) ∧ 
    (Nat.choose (3 * n.val) (38 - n.val) + Nat.choose (n.val + 21) (3 * n.val) = 466) := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_theorem_l3620_362081


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l3620_362099

def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m + 6)*x + 1

theorem cubic_function_extrema (m : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    IsLocalMax (f m) x₁ ∧ 
    IsLocalMin (f m) x₂) ↔ 
  (m < -3 ∨ m > 6) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l3620_362099


namespace NUMINAMATH_CALUDE_sqrt_360000_equals_600_l3620_362095

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_equals_600_l3620_362095


namespace NUMINAMATH_CALUDE_two_male_two_female_selection_methods_at_least_one_male_one_female_selection_methods_l3620_362028

-- Define the number of female and male students
def num_female : ℕ := 5
def num_male : ℕ := 4

-- Define the number of students to be selected
def num_selected : ℕ := 4

-- Theorem for scenario 1
theorem two_male_two_female_selection_methods : 
  (num_male.choose 2 * num_female.choose 2) * num_selected.factorial = 1440 := by sorry

-- Theorem for scenario 2
theorem at_least_one_male_one_female_selection_methods :
  (num_male.choose 1 * num_female.choose 3 + 
   num_male.choose 2 * num_female.choose 2 + 
   num_male.choose 3 * num_female.choose 1) * num_selected.factorial = 2880 := by sorry

end NUMINAMATH_CALUDE_two_male_two_female_selection_methods_at_least_one_male_one_female_selection_methods_l3620_362028


namespace NUMINAMATH_CALUDE_simplify_fraction_l3620_362069

theorem simplify_fraction : (90 + 54) / (150 - 90) = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3620_362069


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_l3620_362054

theorem sum_of_four_consecutive_integers (a b c d : ℤ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c + 1 = d) ∧ (d = 27) → a + b + c + d = 102 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_l3620_362054


namespace NUMINAMATH_CALUDE_base_conversion_l3620_362071

/-- Given that 26 in decimal is equal to 32 in base k, prove that k = 8 -/
theorem base_conversion (k : ℕ) (h : 3 * k + 2 = 26) : k = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l3620_362071


namespace NUMINAMATH_CALUDE_tangent_line_equation_f_decreasing_intervals_l3620_362012

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Theorem for the equation of the tangent line
theorem tangent_line_equation :
  ∀ x y : ℝ, y = f x → (x = 0 → 9*x - y - 2 = 0) :=
sorry

-- Theorem for the intervals where f is decreasing
theorem f_decreasing_intervals :
  ∀ x : ℝ, (x < -1 ∨ x > 3) → (f' x < 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_f_decreasing_intervals_l3620_362012


namespace NUMINAMATH_CALUDE_gcf_and_sum_proof_l3620_362045

def a : ℕ := 198
def b : ℕ := 396

theorem gcf_and_sum_proof : 
  (Nat.gcd a b = a) ∧ 
  (a + 4 * a = 990) := by
sorry

end NUMINAMATH_CALUDE_gcf_and_sum_proof_l3620_362045


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3620_362004

-- Problem 1
theorem factorization_problem_1 (a b : ℝ) :
  12 * a^3 * b - 12 * a^2 * b + 3 * a * b = 3 * a * b * (2*a - 1)^2 := by
  sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  9 - x^2 + 2*x*y - y^2 = (3 + x - y) * (3 - x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3620_362004


namespace NUMINAMATH_CALUDE_gymnasts_count_l3620_362078

/-- The number of gymnastics teams --/
def num_teams : ℕ := 4

/-- The total number of handshakes --/
def total_handshakes : ℕ := 595

/-- The number of gymnasts each coach shakes hands with --/
def coach_handshakes : ℕ := 6

/-- The total number of gymnasts across all teams --/
def total_gymnasts : ℕ := 34

/-- Theorem stating that the total number of gymnasts is 34 --/
theorem gymnasts_count : 
  (total_gymnasts * (total_gymnasts - 1)) / 2 + num_teams * coach_handshakes = total_handshakes :=
by sorry

end NUMINAMATH_CALUDE_gymnasts_count_l3620_362078


namespace NUMINAMATH_CALUDE_stratified_sampling_teachers_l3620_362062

theorem stratified_sampling_teachers :
  let total_teachers : ℕ := 150
  let senior_teachers : ℕ := 45
  let intermediate_teachers : ℕ := 90
  let junior_teachers : ℕ := 15
  let sample_size : ℕ := 30
  let sample_senior : ℕ := 9
  let sample_intermediate : ℕ := 18
  let sample_junior : ℕ := 3
  
  (total_teachers = senior_teachers + intermediate_teachers + junior_teachers) →
  (sample_size = sample_senior + sample_intermediate + sample_junior) →
  (sample_senior : ℚ) / senior_teachers = (sample_intermediate : ℚ) / intermediate_teachers →
  (sample_senior : ℚ) / senior_teachers = (sample_junior : ℚ) / junior_teachers →
  (sample_size : ℚ) / total_teachers = (sample_senior : ℚ) / senior_teachers :=
by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_teachers_l3620_362062


namespace NUMINAMATH_CALUDE_incorrect_division_l3620_362067

theorem incorrect_division (D : ℕ) (h : D / 36 = 48) : D / 72 = 24 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_division_l3620_362067


namespace NUMINAMATH_CALUDE_car_rental_cost_l3620_362032

/-- The daily rental cost of a car, given specific conditions. -/
theorem car_rental_cost (daily_rate : ℝ) (cost_per_mile : ℝ) (budget : ℝ) (miles : ℝ) : 
  cost_per_mile = 0.23 →
  budget = 76 →
  miles = 200 →
  daily_rate + cost_per_mile * miles = budget →
  daily_rate = 30 := by
sorry

end NUMINAMATH_CALUDE_car_rental_cost_l3620_362032


namespace NUMINAMATH_CALUDE_amount_over_limit_l3620_362026

/-- Calculates the amount spent over a given limit when purchasing a necklace and a book,
    where the book costs $5 more than the necklace. -/
theorem amount_over_limit (necklace_cost book_cost limit : ℕ) : 
  necklace_cost = 34 →
  book_cost = necklace_cost + 5 →
  limit = 70 →
  (necklace_cost + book_cost) - limit = 3 := by
sorry


end NUMINAMATH_CALUDE_amount_over_limit_l3620_362026


namespace NUMINAMATH_CALUDE_nonnegative_integer_solutions_l3620_362096

theorem nonnegative_integer_solutions : 
  {(x, y) : ℕ × ℕ | 3 * x^2 + 2 * 9^y = x * (4^(y + 1) - 1)} = {(3, 1), (2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_integer_solutions_l3620_362096


namespace NUMINAMATH_CALUDE_rope_length_difference_l3620_362036

/-- Given three ropes with lengths in ratio 4 : 5 : 6, where the shortest is 80 meters,
    prove that the sum of the longest and shortest is 100 meters more than the middle. -/
theorem rope_length_difference (shortest middle longest : ℝ) : 
  shortest = 80 ∧ 
  5 * shortest = 4 * middle ∧ 
  6 * shortest = 4 * longest →
  longest + shortest = middle + 100 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_difference_l3620_362036


namespace NUMINAMATH_CALUDE_linear_function_negative_slope_l3620_362047

/-- Given a linear function y = kx + b passing through points A(1, m) and B(-1, n), 
    where m < n and k ≠ 0, prove that k < 0. -/
theorem linear_function_negative_slope (k b m n : ℝ) 
  (h1 : k ≠ 0)
  (h2 : m < n)
  (h3 : m = k + b)  -- Point A(1, m) satisfies the equation
  (h4 : n = -k + b) -- Point B(-1, n) satisfies the equation
  : k < 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_negative_slope_l3620_362047


namespace NUMINAMATH_CALUDE_mix_alloys_theorem_l3620_362060

/-- Represents an alloy of copper and zinc -/
structure Alloy where
  copper : ℝ
  zinc : ℝ

/-- The first alloy with twice as much copper as zinc -/
def alloy1 : Alloy := { copper := 2, zinc := 1 }

/-- The second alloy with five times less copper than zinc -/
def alloy2 : Alloy := { copper := 1, zinc := 5 }

/-- Mixing two alloys in a given ratio -/
def mixAlloys (a b : Alloy) (ratio : ℝ) : Alloy :=
  { copper := ratio * a.copper + b.copper,
    zinc := ratio * a.zinc + b.zinc }

/-- Theorem stating that mixing alloy1 and alloy2 in 1:2 ratio results in an alloy with twice as much zinc as copper -/
theorem mix_alloys_theorem :
  let mixedAlloy := mixAlloys alloy1 alloy2 0.5
  mixedAlloy.zinc = 2 * mixedAlloy.copper := by sorry

end NUMINAMATH_CALUDE_mix_alloys_theorem_l3620_362060


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3620_362074

theorem complex_equation_solution (z : ℂ) :
  Complex.abs z = 2 + z + Complex.I * 3 → z = 5 / 4 - Complex.I * 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3620_362074


namespace NUMINAMATH_CALUDE_distance_traveled_l3620_362080

theorem distance_traveled (original_speed : ℝ) (increased_speed : ℝ) (additional_distance : ℝ) 
  (h1 : original_speed = 8)
  (h2 : increased_speed = 12)
  (h3 : additional_distance = 20)
  : ∃ (actual_distance : ℝ) (time : ℝ),
    actual_distance = original_speed * time ∧
    actual_distance + additional_distance = increased_speed * time ∧
    actual_distance = 40 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3620_362080


namespace NUMINAMATH_CALUDE_book_pages_problem_l3620_362058

theorem book_pages_problem :
  ∃ (n k : ℕ), 
    n > 0 ∧ 
    k > 0 ∧ 
    k < n ∧ 
    n * (n + 1) / 2 - (2 * k + 1) = 4979 :=
sorry

end NUMINAMATH_CALUDE_book_pages_problem_l3620_362058


namespace NUMINAMATH_CALUDE_complex_number_difference_l3620_362002

theorem complex_number_difference : 
  let z : ℂ := (Complex.I * (-6 + Complex.I)) / Complex.abs (3 - 4 * Complex.I)
  (z.re - z.im) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_number_difference_l3620_362002


namespace NUMINAMATH_CALUDE_correct_remaining_time_l3620_362000

/-- Represents a food item with its cooking times -/
structure FoodItem where
  name : String
  recommendedTime : Nat
  actualTime : Nat

/-- Calculates the remaining cooking time in seconds for a food item -/
def remainingTimeInSeconds (food : FoodItem) : Nat :=
  (food.recommendedTime - food.actualTime) * 60

/-- The main theorem to prove -/
theorem correct_remaining_time (frenchFries chickenNuggets mozzarellaSticks : FoodItem)
  (h1 : frenchFries.name = "French Fries" ∧ frenchFries.recommendedTime = 12 ∧ frenchFries.actualTime = 2)
  (h2 : chickenNuggets.name = "Chicken Nuggets" ∧ chickenNuggets.recommendedTime = 18 ∧ chickenNuggets.actualTime = 5)
  (h3 : mozzarellaSticks.name = "Mozzarella Sticks" ∧ mozzarellaSticks.recommendedTime = 8 ∧ mozzarellaSticks.actualTime = 3) :
  remainingTimeInSeconds frenchFries = 600 ∧
  remainingTimeInSeconds chickenNuggets = 780 ∧
  remainingTimeInSeconds mozzarellaSticks = 300 := by
  sorry


end NUMINAMATH_CALUDE_correct_remaining_time_l3620_362000


namespace NUMINAMATH_CALUDE_cos_two_alpha_zero_l3620_362017

theorem cos_two_alpha_zero (α : Real) 
  (h : Real.sin (π / 6 - α) = Real.cos (π / 6 + α)) : 
  Real.cos (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_zero_l3620_362017


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3620_362025

theorem division_remainder_problem : ∃ (r : ℕ), 15968 = 179 * 89 + r ∧ r < 179 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3620_362025


namespace NUMINAMATH_CALUDE_sqrt_144_div_6_l3620_362090

theorem sqrt_144_div_6 : Real.sqrt 144 / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_144_div_6_l3620_362090


namespace NUMINAMATH_CALUDE_complex_product_sum_l3620_362079

theorem complex_product_sum (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (1 + i) * (2 + i) = a + b * i) : a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_sum_l3620_362079


namespace NUMINAMATH_CALUDE_x_power_125_minus_reciprocal_l3620_362057

theorem x_power_125_minus_reciprocal (x : ℝ) (h : x - 1/x = Real.sqrt 3) :
  x^125 - 1/x^125 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_power_125_minus_reciprocal_l3620_362057


namespace NUMINAMATH_CALUDE_max_rock_value_is_58_l3620_362053

/-- Represents a type of rock with its weight and value -/
structure Rock where
  weight : ℕ
  value : ℕ

/-- Calculates the maximum value of rocks that can be carried given the constraints -/
def maxRockValue (rocks : List Rock) (maxWeight : ℕ) (maxSixPoundRocks : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the maximum value of rocks Carl can carry -/
theorem max_rock_value_is_58 :
  let rocks : List Rock := [
    { weight := 3, value := 9 },
    { weight := 6, value := 20 },
    { weight := 2, value := 5 }
  ]
  let maxWeight : ℕ := 20
  let maxSixPoundRocks : ℕ := 2
  maxRockValue rocks maxWeight maxSixPoundRocks = 58 := by
  sorry

end NUMINAMATH_CALUDE_max_rock_value_is_58_l3620_362053


namespace NUMINAMATH_CALUDE_sqrt_five_power_l3620_362041

theorem sqrt_five_power : (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 5 ^ (15 / 2) := by sorry

end NUMINAMATH_CALUDE_sqrt_five_power_l3620_362041


namespace NUMINAMATH_CALUDE_hoseok_additional_jumps_theorem_l3620_362033

/-- The number of additional jumps Hoseok needs to match Minyoung's total -/
def additional_jumps (hoseok_jumps minyoung_jumps : ℕ) : ℕ :=
  minyoung_jumps - hoseok_jumps

theorem hoseok_additional_jumps_theorem (hoseok_jumps minyoung_jumps : ℕ) 
    (h : minyoung_jumps > hoseok_jumps) :
  additional_jumps hoseok_jumps minyoung_jumps = 17 :=
by
  sorry

#eval additional_jumps 34 51

end NUMINAMATH_CALUDE_hoseok_additional_jumps_theorem_l3620_362033


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l3620_362087

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, x < 0 ∧ P x) ↔ (∀ x, x < 0 → ¬ P x) :=
by sorry

-- The specific proposition
def proposition (x : ℝ) : Prop := 3 * x < 4 * x

theorem negation_of_specific_proposition :
  (¬ ∃ x, x < 0 ∧ proposition x) ↔ (∀ x, x < 0 → 3 * x ≥ 4 * x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l3620_362087


namespace NUMINAMATH_CALUDE_abc_product_l3620_362086

theorem abc_product (a b c : ℤ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  a + b + c = 30 → 
  1 / a + 1 / b + 1 / c + 450 / (a * b * c) = 1 → 
  a * b * c = 1920 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l3620_362086


namespace NUMINAMATH_CALUDE_tan_x_value_l3620_362075

theorem tan_x_value (x : Real) (h : Real.tan (x + π/4) = 2) : Real.tan x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_value_l3620_362075


namespace NUMINAMATH_CALUDE_organizing_related_to_excellent_scores_expectation_X_l3620_362050

-- Define the total number of students
def total_students : ℕ := 100

-- Define the number of students with excellent and poor math scores
def excellent_scores : ℕ := 40
def poor_scores : ℕ := 60

-- Define the number of students not organizing regularly
def not_organizing_excellent : ℕ := 8  -- 20% of 40
def not_organizing_poor : ℕ := 32

-- Define the chi-square statistic
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99% certainty
def critical_value : ℚ := 6635 / 1000

-- Theorem for the relationship between organizing regularly and excellent math scores
theorem organizing_related_to_excellent_scores :
  chi_square not_organizing_excellent (excellent_scores - not_organizing_excellent)
              not_organizing_poor (poor_scores - not_organizing_poor) > critical_value := by
  sorry

-- Define the probability distribution of X
def prob_X (x : ℕ) : ℚ :=
  match x with
  | 0 => 28 / 45
  | 1 => 16 / 45
  | 2 => 1 / 45
  | _ => 0

-- Theorem for the expectation of X
theorem expectation_X :
  (0 : ℚ) * prob_X 0 + 1 * prob_X 1 + 2 * prob_X 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_organizing_related_to_excellent_scores_expectation_X_l3620_362050


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3620_362042

theorem complex_sum_theorem : 
  let i : ℂ := Complex.I
  let z₁ : ℂ := 2 + i
  let z₂ : ℂ := 1 - 2*i
  z₁ + z₂ = 3 - i := by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3620_362042


namespace NUMINAMATH_CALUDE_duck_difference_l3620_362024

/-- The number of Muscovy ducks is greater than the number of Cayuga ducks,
    and there are 3 more than twice as many Cayugas as Khaki Campbells.
    There are 90 ducks in total, and 39 Muscovy ducks.
    This theorem proves that there are 27 more Muscovy ducks than Cayugas. -/
theorem duck_difference (m c k : ℕ) : 
  m + c + k = 90 →
  m = 39 →
  m > c →
  m = 2 * c + 3 + k →
  m - c = 27 := by
  sorry

end NUMINAMATH_CALUDE_duck_difference_l3620_362024


namespace NUMINAMATH_CALUDE_M_mod_1000_l3620_362007

def M : ℕ := Nat.choose 14 8

theorem M_mod_1000 : M % 1000 = 3 := by
  sorry

end NUMINAMATH_CALUDE_M_mod_1000_l3620_362007


namespace NUMINAMATH_CALUDE_total_third_graders_l3620_362061

/-- The number of third grade girl students -/
def girl_students : ℕ := 57

/-- The number of third grade boy students -/
def boy_students : ℕ := 66

/-- The total number of third grade students -/
def total_students : ℕ := girl_students + boy_students

/-- Theorem stating that the total number of third grade students is 123 -/
theorem total_third_graders : total_students = 123 := by
  sorry

end NUMINAMATH_CALUDE_total_third_graders_l3620_362061


namespace NUMINAMATH_CALUDE_mirror_area_l3620_362092

/-- Calculates the area of a rectangular mirror inside a frame -/
theorem mirror_area (frame_width frame_height frame_thickness : ℕ) : 
  frame_width = 65 ∧ frame_height = 85 ∧ frame_thickness = 15 →
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 1925 := by
  sorry

end NUMINAMATH_CALUDE_mirror_area_l3620_362092


namespace NUMINAMATH_CALUDE_chi_square_independence_hypothesis_l3620_362064

/-- Represents a χ² test of independence -/
structure ChiSquareTest where
  /-- The statistical hypothesis of the test -/
  hypothesis : Prop

/-- Represents events in a statistical context -/
structure Event

/-- Defines mutual independence for a list of events -/
def mutually_independent (events : List Event) : Prop :=
  sorry -- Definition of mutual independence

/-- The χ² test of independence assumes mutual independence of events -/
theorem chi_square_independence_hypothesis :
  ∀ (test : ChiSquareTest) (events : List Event),
    test.hypothesis ↔ mutually_independent events := by
  sorry

end NUMINAMATH_CALUDE_chi_square_independence_hypothesis_l3620_362064


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3620_362010

/-- The sum of the first n terms of an arithmetic sequence -/
def T (a d : ℚ) (n : ℕ+) : ℚ := n * (2 * a + (n - 1) * d) / 2

/-- The theorem states that if T_{4n} / T_n is constant for an arithmetic sequence
    with common difference 5, then the first term of the sequence is 5/2 -/
theorem arithmetic_sequence_first_term
  (h : ∃ (k : ℚ), ∀ (n : ℕ+), T a 5 (4 * n) / T a 5 n = k) :
  a = 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3620_362010


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l3620_362072

theorem quadratic_equation_condition (m : ℝ) : 
  (abs m + 1 = 2 ∧ m + 1 ≠ 0) ↔ m = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l3620_362072


namespace NUMINAMATH_CALUDE_no_guaranteed_primes_l3620_362094

theorem no_guaranteed_primes (n : ℕ) (h : n > 1) :
  ∀ p : ℕ, Prime p → (p ∉ Set.Ioo (n.factorial) (n.factorial + 2*n)) :=
sorry

end NUMINAMATH_CALUDE_no_guaranteed_primes_l3620_362094


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3620_362011

/-- A line passing through (1,0) parallel to x-2y-2=0 has equation x-2y-1=0 -/
theorem parallel_line_through_point (x y : ℝ) : 
  (x - 2*y - 1 = 0) ↔ 
  (∃ (m b : ℝ), y = m*x + b ∧ 
                 m = (1 : ℝ)/(2 : ℝ) ∧ 
                 1 = m*1 + b ∧ 
                 0 = m*0 + b) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3620_362011


namespace NUMINAMATH_CALUDE_number_equals_nine_l3620_362023

theorem number_equals_nine (x : ℝ) (N : ℝ) (h1 : x = 0.5) (h2 : N / (1 + 4 / x) = 1) : N = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_nine_l3620_362023


namespace NUMINAMATH_CALUDE_distance_major_minor_endpoints_l3620_362084

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  4 * (x - 3)^2 + 16 * (y + 2)^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (3, -2)

-- Define the semi-major and semi-minor axes
def a : ℝ := 4
def b : ℝ := 2

-- Define a point on the major axis
def point_on_major_axis (x y : ℝ) : Prop :=
  ellipse x y ∧ y = center.2

-- Define a point on the minor axis
def point_on_minor_axis (x y : ℝ) : Prop :=
  ellipse x y ∧ x = center.1

-- Theorem: The distance between an endpoint of the major axis
-- and an endpoint of the minor axis is 2√5
theorem distance_major_minor_endpoints :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    point_on_major_axis x₁ y₁ ∧
    point_on_minor_axis x₂ y₂ ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 20 :=
sorry

end NUMINAMATH_CALUDE_distance_major_minor_endpoints_l3620_362084


namespace NUMINAMATH_CALUDE_button_up_shirt_cost_l3620_362066

def total_budget : ℕ := 200
def suit_pants : ℕ := 46
def suit_coat : ℕ := 38
def socks : ℕ := 11
def belt : ℕ := 18
def shoes : ℕ := 41
def amount_left : ℕ := 16

theorem button_up_shirt_cost : 
  total_budget - (suit_pants + suit_coat + socks + belt + shoes + amount_left) = 30 := by
  sorry

end NUMINAMATH_CALUDE_button_up_shirt_cost_l3620_362066


namespace NUMINAMATH_CALUDE_numeral_system_base_proof_l3620_362018

theorem numeral_system_base_proof (x : ℕ) : 
  (3 * x + 4)^2 = x^3 + 5 * x^2 + 5 * x + 2 → x = 7 := by
sorry

end NUMINAMATH_CALUDE_numeral_system_base_proof_l3620_362018


namespace NUMINAMATH_CALUDE_down_payment_percentage_l3620_362003

def house_price : ℝ := 100000
def parents_contribution_rate : ℝ := 0.30
def remaining_balance : ℝ := 56000

theorem down_payment_percentage :
  ∃ (down_payment_rate : ℝ),
    down_payment_rate * house_price +
    parents_contribution_rate * (house_price - down_payment_rate * house_price) +
    remaining_balance = house_price ∧
    down_payment_rate = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_down_payment_percentage_l3620_362003


namespace NUMINAMATH_CALUDE_cash_realized_approx_103_74_l3620_362088

/-- The cash realized on selling a stock, given the brokerage rate and total amount including brokerage -/
def cash_realized (brokerage_rate : ℚ) (total_amount : ℚ) : ℚ :=
  total_amount / (1 + brokerage_rate)

/-- Theorem stating that the cash realized is approximately 103.74 given the problem conditions -/
theorem cash_realized_approx_103_74 :
  let brokerage_rate : ℚ := 1 / 400  -- 1/4% expressed as a fraction
  let total_amount : ℚ := 104
  |cash_realized brokerage_rate total_amount - 103.74| < 0.01 := by
  sorry

#eval cash_realized (1/400) 104

end NUMINAMATH_CALUDE_cash_realized_approx_103_74_l3620_362088


namespace NUMINAMATH_CALUDE_bicycle_oil_requirement_l3620_362049

/-- The number of wheels on a bicycle -/
def num_wheels : ℕ := 2

/-- The amount of oil needed for each wheel (in ml) -/
def oil_per_wheel : ℕ := 10

/-- The amount of oil needed for the rest of the bike (in ml) -/
def oil_for_rest : ℕ := 5

/-- The total amount of oil needed to fix the bicycle (in ml) -/
def total_oil_needed : ℕ := num_wheels * oil_per_wheel + oil_for_rest

theorem bicycle_oil_requirement :
  total_oil_needed = 25 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_oil_requirement_l3620_362049


namespace NUMINAMATH_CALUDE_unit_circle_from_sin_cos_l3620_362016

-- Define the set of points (x,y) = (sin t, cos t) for all real t
def unitCirclePoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = Real.sin t ∧ p.2 = Real.cos t}

-- Theorem: The set of points forms a circle with radius 1 centered at the origin
theorem unit_circle_from_sin_cos :
  unitCirclePoints = {p : ℝ × ℝ | p.1^2 + p.2^2 = 1} := by
  sorry


end NUMINAMATH_CALUDE_unit_circle_from_sin_cos_l3620_362016


namespace NUMINAMATH_CALUDE_chris_age_l3620_362056

theorem chris_age (a b c : ℚ) : 
  (a + b + c) / 3 = 12 →
  c - 5 = 2 * a →
  b + 2 = (a + 2) / 2 →
  c = 163 / 7 := by
sorry

end NUMINAMATH_CALUDE_chris_age_l3620_362056


namespace NUMINAMATH_CALUDE_p_satisfies_conditions_l3620_362015

/-- The quadratic polynomial p(x) that satisfies given conditions -/
def p (x : ℚ) : ℚ := (12/5) * x^2 - (36/5) * x - 216/5

/-- Theorem stating that p(x) satisfies the required conditions -/
theorem p_satisfies_conditions : 
  p (-3) = 0 ∧ p 6 = 0 ∧ p 2 = -48 := by
  sorry

end NUMINAMATH_CALUDE_p_satisfies_conditions_l3620_362015


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l3620_362030

theorem merchant_pricing_strategy (L : ℝ) (h : L > 0) :
  let purchase_price := L * 0.7
  let marked_price := L * 1.25
  let selling_price := marked_price * 0.8
  selling_price = purchase_price * 1.3 := by
sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l3620_362030


namespace NUMINAMATH_CALUDE_tims_age_l3620_362085

theorem tims_age (tim rommel jenny : ℕ) 
  (h1 : rommel = 3 * tim)
  (h2 : jenny = rommel + 2)
  (h3 : tim + 12 = jenny) :
  tim = 5 := by
sorry

end NUMINAMATH_CALUDE_tims_age_l3620_362085


namespace NUMINAMATH_CALUDE_ballon_arrangements_l3620_362046

theorem ballon_arrangements :
  let total_letters : Nat := 6
  let repeated_letters : Nat := 2
  Nat.factorial total_letters / Nat.factorial repeated_letters = 360 := by
  sorry

end NUMINAMATH_CALUDE_ballon_arrangements_l3620_362046


namespace NUMINAMATH_CALUDE_op_35_77_l3620_362051

-- Define the operation @
def op (a b : ℕ+) : ℚ := (a * b) / (a + b)

-- Theorem statement
theorem op_35_77 : op 35 77 = 2695 / 112 := by
  sorry

end NUMINAMATH_CALUDE_op_35_77_l3620_362051


namespace NUMINAMATH_CALUDE_work_completion_l3620_362070

/-- Problem: Work completion by two workers --/
theorem work_completion (a_days b_days b_worked_days : ℕ) 
  (ha : a_days > 0) 
  (hb : b_days > 0) 
  (hw : b_worked_days < b_days) :
  let b_work_rate := 1 / b_days
  let b_work_done := b_worked_days * b_work_rate
  let remaining_work := 1 - b_work_done
  let a_work_rate := 1 / a_days
  a_days = 12 ∧ b_days = 15 ∧ b_worked_days = 10 →
  remaining_work / a_work_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_work_completion_l3620_362070


namespace NUMINAMATH_CALUDE_cube_equation_solution_l3620_362008

theorem cube_equation_solution (a w : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * w) : w = 49 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l3620_362008


namespace NUMINAMATH_CALUDE_mike_ride_length_l3620_362020

/-- Represents the taxi fare structure and trip details for Mike and Annie -/
structure TaxiTrip where
  initialCharge : ℝ
  costPerMile : ℝ
  surcharge : ℝ
  tollFees : ℝ
  miles : ℝ

/-- Calculates the total cost of a taxi trip -/
def tripCost (trip : TaxiTrip) : ℝ :=
  trip.initialCharge + trip.costPerMile * trip.miles + trip.surcharge + trip.tollFees

/-- Theorem stating that Mike's ride was 30 miles long -/
theorem mike_ride_length 
  (mike : TaxiTrip)
  (annie : TaxiTrip)
  (h1 : mike.initialCharge = 2.5)
  (h2 : mike.costPerMile = 0.25)
  (h3 : mike.surcharge = 3)
  (h4 : mike.tollFees = 0)
  (h5 : annie.initialCharge = 2.5)
  (h6 : annie.costPerMile = 0.25)
  (h7 : annie.surcharge = 0)
  (h8 : annie.tollFees = 5)
  (h9 : annie.miles = 22)
  (h10 : tripCost mike = tripCost annie) :
  mike.miles = 30 := by
  sorry

end NUMINAMATH_CALUDE_mike_ride_length_l3620_362020


namespace NUMINAMATH_CALUDE_distance_solution_l3620_362005

/-- The distance from a dormitory to a city -/
def distance_problem (D : ℝ) : Prop :=
  (1/5 : ℝ) * D + (2/3 : ℝ) * D + 12 = D

theorem distance_solution : ∃ D : ℝ, distance_problem D ∧ D = 90 := by
  sorry

end NUMINAMATH_CALUDE_distance_solution_l3620_362005


namespace NUMINAMATH_CALUDE_tan_sum_product_l3620_362006

theorem tan_sum_product (α β : Real) (h : α + β = 3 * Real.pi / 4) :
  (1 - Real.tan α) * (1 - Real.tan β) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_product_l3620_362006


namespace NUMINAMATH_CALUDE_intersection_distance_l3620_362040

/-- The distance between the intersection points of two curves in polar coordinates -/
theorem intersection_distance (θ : Real) : 
  ∃ (A B : ℝ × ℝ), 
    (∀ (ρ : ℝ), ρ * Real.sin (θ + π/4) = 1 → (ρ * Real.cos θ, ρ * Real.sin θ) = A ∨ (ρ * Real.cos θ, ρ * Real.sin θ) = B) ∧
    (∀ (ρ : ℝ), ρ = Real.sqrt 2 → (ρ * Real.cos θ, ρ * Real.sin θ) = A ∨ (ρ * Real.cos θ, ρ * Real.sin θ) = B) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l3620_362040


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3620_362091

theorem smallest_integer_with_given_remainders : ∃ (b : ℕ), 
  b > 0 ∧ 
  b % 3 = 0 ∧ 
  b % 4 = 2 ∧ 
  b % 5 = 3 ∧ 
  ∀ (n : ℕ), n > 0 ∧ n % 3 = 0 ∧ n % 4 = 2 ∧ n % 5 = 3 → b ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3620_362091


namespace NUMINAMATH_CALUDE_extended_euclidean_algorithm_l3620_362059

theorem extended_euclidean_algorithm (m₀ m₁ : ℤ) (h : 0 < m₁ ∧ m₁ ≤ m₀) :
  ∃ u v : ℤ, m₀ * u + m₁ * v = Int.gcd m₀ m₁ := by
  sorry

end NUMINAMATH_CALUDE_extended_euclidean_algorithm_l3620_362059


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3620_362038

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 * (2 * π * r₁)) = (48 / 360 * (2 * π * r₂)) →
  (π * r₁^2) / (π * r₂^2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3620_362038


namespace NUMINAMATH_CALUDE_sports_meeting_medals_l3620_362029

/-- The number of medals awarded on day x -/
def f (x m : ℕ) : ℚ :=
  if x = 1 then
    1 + (m - 1) / 7
  else
    (6 / 7) ^ (x - 1) * ((m - 36) / 7) + 6

/-- The total number of medals awarded over n days -/
def total_medals (n : ℕ) : ℕ := 36

/-- The number of days the sports meeting lasted -/
def meeting_duration : ℕ := 6

theorem sports_meeting_medals :
  ∀ n m : ℕ,
  (∀ i : ℕ, i < n → f i m = i + (f (i+1) m - i) / 7) →
  f n m = n →
  n = meeting_duration ∧ m = total_medals n :=
sorry

end NUMINAMATH_CALUDE_sports_meeting_medals_l3620_362029


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3620_362093

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 5 * a 14 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3620_362093


namespace NUMINAMATH_CALUDE_expected_coincidences_value_l3620_362031

/-- The number of questions in the test -/
def num_questions : ℕ := 20

/-- Vasya's probability of guessing correctly -/
def p_vasya : ℚ := 6 / 20

/-- Misha's probability of guessing correctly -/
def p_misha : ℚ := 8 / 20

/-- The probability of a coincidence (both correct or both incorrect) for a single question -/
def p_coincidence : ℚ := p_vasya * p_misha + (1 - p_vasya) * (1 - p_misha)

/-- The expected number of coincidences -/
def expected_coincidences : ℚ := num_questions * p_coincidence

theorem expected_coincidences_value :
  expected_coincidences = 54 / 5 := by sorry

end NUMINAMATH_CALUDE_expected_coincidences_value_l3620_362031


namespace NUMINAMATH_CALUDE_initial_shirts_count_l3620_362068

/-- The number of shirts Haley returned -/
def returned_shirts : ℕ := 6

/-- The number of shirts Haley ended up with -/
def final_shirts : ℕ := 5

/-- The initial number of shirts Haley bought -/
def initial_shirts : ℕ := returned_shirts + final_shirts

/-- Theorem stating that the initial number of shirts is 11 -/
theorem initial_shirts_count : initial_shirts = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_shirts_count_l3620_362068


namespace NUMINAMATH_CALUDE_sum_of_variables_l3620_362001

theorem sum_of_variables (a b c : ℝ) 
  (eq1 : a + 2*b + 3*c = 13) 
  (eq2 : 4*a + 3*b + 2*c = 17) : 
  a + b + c = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l3620_362001


namespace NUMINAMATH_CALUDE_x_value_in_sequence_l3620_362082

def fibonacci_like_sequence (a b : ℤ) : ℕ → ℤ
  | 0 => a
  | 1 => b
  | n+2 => fibonacci_like_sequence a b n + fibonacci_like_sequence a b (n+1)

theorem x_value_in_sequence :
  ∃ (start : ℕ), 
    (fibonacci_like_sequence (-1) 2 (start + 2) = 3) ∧
    (fibonacci_like_sequence (-1) 2 (start + 3) = 5) ∧
    (fibonacci_like_sequence (-1) 2 (start + 4) = 8) ∧
    (fibonacci_like_sequence (-1) 2 (start + 5) = 13) ∧
    (fibonacci_like_sequence (-1) 2 (start + 6) = 21) ∧
    (fibonacci_like_sequence (-1) 2 (start + 7) = 34) ∧
    (fibonacci_like_sequence (-1) 2 (start + 8) = 55) := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_sequence_l3620_362082


namespace NUMINAMATH_CALUDE_number_of_boys_l3620_362009

/-- Proves that the number of boys is 5 given the problem conditions -/
theorem number_of_boys (men : ℕ) (women : ℕ) (boys : ℕ) (total_earnings : ℕ) (men_wage : ℕ) :
  men = 5 →
  men = women →
  women = boys →
  total_earnings = 90 →
  men_wage = 6 →
  boys = 5 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l3620_362009


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3620_362043

def A (a : ℝ) := {x : ℝ | 0 < x ∧ x < a}
def B := {x : ℝ | 1 < x ∧ x < 2}

theorem possible_values_of_a (a : ℝ) :
  (A a).Nonempty ∧ B ⊆ (Aᶜ a) ↔ 0 < a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3620_362043


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l3620_362073

/-- Calculate the cost of plastering a tank's walls and bottom -/
def plasteringCost (length width depth : ℝ) (costPerSquareMeter : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * costPerSquareMeter

/-- Theorem: The cost of plastering a tank with given dimensions is 558 rupees -/
theorem tank_plastering_cost :
  plasteringCost 25 12 6 0.75 = 558 := by
  sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l3620_362073
