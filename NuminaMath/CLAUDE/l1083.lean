import Mathlib

namespace min_value_of_expression_l1083_108356

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3 ≥ 6 := by
  sorry

end min_value_of_expression_l1083_108356


namespace log_product_equals_four_l1083_108302

theorem log_product_equals_four : Real.log 9 / Real.log 2 * (Real.log 4 / Real.log 3) = 4 := by
  sorry

end log_product_equals_four_l1083_108302


namespace exponent_sum_l1083_108385

theorem exponent_sum (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m+n) = 6 := by
  sorry

end exponent_sum_l1083_108385


namespace purely_imaginary_quotient_implies_a_l1083_108384

def z₁ (a : ℝ) : ℂ := a + 2 * Complex.I
def z₂ : ℂ := 3 - 4 * Complex.I

theorem purely_imaginary_quotient_implies_a (a : ℝ) :
  (z₁ a / z₂).re = 0 → (z₁ a / z₂).im ≠ 0 → a = 8/3 := by
  sorry

end purely_imaginary_quotient_implies_a_l1083_108384


namespace star_value_l1083_108380

-- Define the operation *
def star (a b : ℤ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem star_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 12) (prod_eq : a * b = 32) : 
  star a b = 3 / 8 := by
  sorry

end star_value_l1083_108380


namespace square_side_length_l1083_108326

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s * Real.sqrt 2 = d ∧ s = 2 := by
  sorry

end square_side_length_l1083_108326


namespace sandwich_problem_l1083_108367

theorem sandwich_problem (sandwich_cost soda_cost total_cost : ℚ) 
                         (num_sodas : ℕ) :
  sandwich_cost = 245/100 →
  soda_cost = 87/100 →
  num_sodas = 4 →
  total_cost = 838/100 →
  ∃ (num_sandwiches : ℕ), 
    num_sandwiches * sandwich_cost + num_sodas * soda_cost = total_cost ∧
    num_sandwiches = 2 :=
by sorry

end sandwich_problem_l1083_108367


namespace asian_games_ticket_scientific_notation_l1083_108364

theorem asian_games_ticket_scientific_notation :
  ∃ (a : ℝ) (b : ℤ), 1 ≤ a ∧ a < 10 ∧ 113700 = a * (10 : ℝ) ^ b ∧ a = 1.137 ∧ b = 5 := by
  sorry

end asian_games_ticket_scientific_notation_l1083_108364


namespace least_common_multiple_first_ten_l1083_108393

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∧
  n = 2520 := by
sorry

end least_common_multiple_first_ten_l1083_108393


namespace min_value_of_f_l1083_108323

def f (x : ℝ) : ℝ := x^2 - 2*x

theorem min_value_of_f :
  ∃ (m : ℝ), m = -1 ∧ ∀ x ∈ Set.Icc 0 3, f x ≥ m :=
sorry

end min_value_of_f_l1083_108323


namespace test_questions_l1083_108314

theorem test_questions (total_points : ℕ) (five_point_questions : ℕ) (points_per_five_point : ℕ) (points_per_ten_point : ℕ) : 
  total_points = 200 →
  five_point_questions = 20 →
  points_per_five_point = 5 →
  points_per_ten_point = 10 →
  ∃ (ten_point_questions : ℕ),
    five_point_questions * points_per_five_point + ten_point_questions * points_per_ten_point = total_points ∧
    five_point_questions + ten_point_questions = 30 :=
by sorry

end test_questions_l1083_108314


namespace evaluate_expression_l1083_108321

theorem evaluate_expression : (1500^2 : ℚ) / (306^2 - 294^2) = 312.5 := by sorry

end evaluate_expression_l1083_108321


namespace no_solutions_for_2500_l1083_108313

theorem no_solutions_for_2500 :
  ¬ ∃ (a₂ a₀ : ℤ), 2500 = a₂ * 10^4 + a₀ ∧ 0 ≤ a₂ ∧ a₂ ≤ 9 ∧ 0 ≤ a₀ ∧ a₀ ≤ 1000 := by
  sorry

end no_solutions_for_2500_l1083_108313


namespace reciprocal_of_repeating_decimal_l1083_108363

/-- The decimal representation of x as 0.36̅ -/
def x : ℚ := 36 / 99

/-- The reciprocal of the common fraction form of 0.36̅ -/
def reciprocal : ℚ := 11 / 4

theorem reciprocal_of_repeating_decimal :
  (1 : ℚ) / x = reciprocal := by sorry

end reciprocal_of_repeating_decimal_l1083_108363


namespace decreasing_f_implies_a_leq_neg_three_l1083_108306

/-- The function f(x) defined as x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The theorem stating that if f(x) is decreasing on (-∞, 4), then a ≤ -3 -/
theorem decreasing_f_implies_a_leq_neg_three (a : ℝ) :
  (∀ x y, x < y → y < 4 → f a x > f a y) → a ≤ -3 := by
  sorry

end decreasing_f_implies_a_leq_neg_three_l1083_108306


namespace sum_triangles_eq_sixteen_l1083_108322

/-- The triangle operation -/
def triangle (a b c : ℕ) : ℕ := a * b - c

/-- The sum of two triangle operations -/
def sum_triangles (a1 b1 c1 a2 b2 c2 : ℕ) : ℕ :=
  triangle a1 b1 c1 + triangle a2 b2 c2

/-- Theorem: The sum of the triangle operations for the given sets of numbers equals 16 -/
theorem sum_triangles_eq_sixteen :
  sum_triangles 2 4 3 3 6 7 = 16 := by sorry

end sum_triangles_eq_sixteen_l1083_108322


namespace rowing_distance_problem_l1083_108362

/-- Proves that the distance to a destination is 72 km given specific rowing conditions -/
theorem rowing_distance_problem (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) : 
  rowing_speed = 10 → 
  current_speed = 2 → 
  total_time = 15 → 
  (rowing_speed + current_speed) * (rowing_speed - current_speed) * total_time / 
    (rowing_speed + current_speed + rowing_speed - current_speed) = 72 := by
  sorry

#check rowing_distance_problem

end rowing_distance_problem_l1083_108362


namespace cos_ninety_degrees_l1083_108365

theorem cos_ninety_degrees : Real.cos (π / 2) = 0 := by
  sorry

end cos_ninety_degrees_l1083_108365


namespace min_value_of_f_l1083_108375

def f (x : ℝ) := x^3 - 3*x + 1

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = -1 ∧ ∀ y ∈ Set.Icc 0 3, f y ≥ f x :=
sorry

end min_value_of_f_l1083_108375


namespace divided_triangle_area_l1083_108304

/-- Represents a triangle divided into six smaller triangles -/
structure DividedTriangle where
  /-- Areas of four known smaller triangles -/
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ

/-- The theorem stating that if a triangle is divided as described, with the given areas, its total area is 380 -/
theorem divided_triangle_area (t : DividedTriangle) 
  (h1 : t.area1 = 84) 
  (h2 : t.area2 = 70) 
  (h3 : t.area3 = 35) 
  (h4 : t.area4 = 65) : 
  ∃ (area5 area6 : ℝ), t.area1 + t.area2 + t.area3 + t.area4 + area5 + area6 = 380 := by
  sorry

end divided_triangle_area_l1083_108304


namespace wong_grandchildren_probability_l1083_108388

/-- The number of grandchildren Mr. Wong has -/
def num_grandchildren : ℕ := 12

/-- The probability of a grandchild being male (or female) -/
def gender_probability : ℚ := 1/2

/-- The probability of having an unequal number of grandsons and granddaughters -/
def unequal_gender_prob : ℚ := 793/1024

theorem wong_grandchildren_probability :
  let total_outcomes := 2^num_grandchildren
  let equal_gender_outcomes := (num_grandchildren.choose (num_grandchildren / 2))
  (total_outcomes - equal_gender_outcomes : ℚ) / total_outcomes = unequal_gender_prob :=
sorry

end wong_grandchildren_probability_l1083_108388


namespace martha_juice_bottles_l1083_108354

theorem martha_juice_bottles (initial_pantry : ℕ) (bought : ℕ) (consumed : ℕ) (final_total : ℕ) 
  (h1 : initial_pantry = 4)
  (h2 : bought = 5)
  (h3 : consumed = 3)
  (h4 : final_total = 10) :
  ∃ (initial_fridge : ℕ), 
    initial_fridge + initial_pantry + bought - consumed = final_total ∧ 
    initial_fridge = 4 := by
  sorry

end martha_juice_bottles_l1083_108354


namespace inequality_implies_upper_bound_l1083_108350

open Real

theorem inequality_implies_upper_bound (a : ℝ) : 
  (∀ x > 0, 2 * x * log x ≥ -x^2 + a*x - 3) → a ≤ 4 := by
  sorry

end inequality_implies_upper_bound_l1083_108350


namespace linear_function_k_value_l1083_108337

/-- Given a linear function y = kx + 2 passing through the point (-2, -1), prove that k = 3/2 -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 2) →  -- Linear function condition
  (-1 : ℝ) = k * (-2 : ℝ) + 2 →  -- Point (-2, -1) condition
  k = 3/2 := by sorry

end linear_function_k_value_l1083_108337


namespace even_quadratic_implies_k_eq_one_l1083_108307

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The quadratic function f(x) = kx^2 + (k-1)x + 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

/-- If f(x) = kx^2 + (k-1)x + 2 is an even function, then k = 1 -/
theorem even_quadratic_implies_k_eq_one :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
  sorry

end even_quadratic_implies_k_eq_one_l1083_108307


namespace vacuum_time_per_room_l1083_108329

-- Define the total vacuuming time in hours
def total_time : ℝ := 2

-- Define the number of rooms
def num_rooms : ℕ := 6

-- Define the function to convert hours to minutes
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

-- Theorem statement
theorem vacuum_time_per_room : 
  (hours_to_minutes total_time) / num_rooms = 20 := by
  sorry

end vacuum_time_per_room_l1083_108329


namespace geometric_sequence_third_term_l1083_108312

/-- Represents a geometric sequence -/
def GeometricSequence (a r : ℝ) : ℕ → ℝ := fun n ↦ a * r ^ (n - 1)

/-- Theorem: In a geometric sequence where the first term is 3 and the fifth term is 243, the third term is 27 -/
theorem geometric_sequence_third_term
  (a r : ℝ)
  (h1 : GeometricSequence a r 1 = 3)
  (h5 : GeometricSequence a r 5 = 243) :
  GeometricSequence a r 3 = 27 := by
sorry

end geometric_sequence_third_term_l1083_108312


namespace negation_of_universal_statement_l1083_108369

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end negation_of_universal_statement_l1083_108369


namespace fraction_equality_l1083_108343

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (x - 4 * y) = 3) : 
  (x + 4 * y) / (4 * x - y) = 10 / 57 := by
  sorry

end fraction_equality_l1083_108343


namespace homework_ratio_proof_l1083_108394

/-- Given a total of 15 problems and 6 problems finished, 
    prove that the simplified ratio of problems still to complete 
    to problems already finished is 3:2. -/
theorem homework_ratio_proof (total : ℕ) (finished : ℕ) 
    (h1 : total = 15) (h2 : finished = 6) : 
    (total - finished) / Nat.gcd (total - finished) finished = 3 ∧ 
    finished / Nat.gcd (total - finished) finished = 2 := by
  sorry

end homework_ratio_proof_l1083_108394


namespace binomial_divisibility_theorem_l1083_108371

theorem binomial_divisibility_theorem (k : ℕ) (h : k > 1) :
  ∃ n : ℕ, n > 0 ∧ 
    (n ∣ Nat.choose n k) ∧ 
    (∀ m : ℕ, 2 ≤ m → m < k → ¬(n ∣ Nat.choose n m)) := by
  sorry

end binomial_divisibility_theorem_l1083_108371


namespace sequence_is_increasing_l1083_108320

theorem sequence_is_increasing (a : ℕ → ℝ) (h : ∀ n, a (n + 1) - a n - 3 = 0) :
  ∀ n, a (n + 1) > a n :=
sorry

end sequence_is_increasing_l1083_108320


namespace special_list_median_l1083_108368

/-- The sum of the first n positive integers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list of integers where each n appears n times for 1 ≤ n ≤ 150 -/
def special_list : List ℕ := sorry

/-- The median of a list is the middle value when the list is sorted -/
def median (l : List ℕ) : ℕ := sorry

theorem special_list_median :
  median special_list = 107 := by sorry

end special_list_median_l1083_108368


namespace probability_of_one_in_20_rows_l1083_108398

/-- The number of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1's in the first n rows of Pascal's Triangle -/
def pascal_triangle_ones (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def probability_of_one (n : ℕ) : ℚ :=
  (pascal_triangle_ones n : ℚ) / (pascal_triangle_elements n : ℚ)

theorem probability_of_one_in_20_rows :
  probability_of_one 20 = 13 / 70 := by
  sorry

end probability_of_one_in_20_rows_l1083_108398


namespace power_zero_eq_one_l1083_108334

theorem power_zero_eq_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end power_zero_eq_one_l1083_108334


namespace combined_age_in_eight_years_l1083_108387

/-- Given the current age and the relation between your age and your brother's age 5 years ago,
    calculate the combined age of you and your brother in 8 years. -/
theorem combined_age_in_eight_years
  (your_current_age : ℕ)
  (h1 : your_current_age = 13)
  (h2 : your_current_age - 5 = (your_current_age + 3) - 5) :
  your_current_age + 8 + (your_current_age + 3) + 8 = 50 := by
  sorry

end combined_age_in_eight_years_l1083_108387


namespace inscribed_cylinder_radius_l1083_108378

/-- A right circular cone with an inscribed right circular cylinder -/
structure ConeWithCylinder where
  -- Cone properties
  cone_diameter : ℝ
  cone_altitude : ℝ
  -- Cylinder properties
  cylinder_radius : ℝ
  -- Conditions
  cone_diameter_positive : 0 < cone_diameter
  cone_altitude_positive : 0 < cone_altitude
  cylinder_radius_positive : 0 < cylinder_radius
  cylinder_inscribed : cylinder_radius ≤ cone_diameter / 2
  cylinder_height_eq_diameter : cylinder_radius * 2 = cylinder_radius * 2
  shared_axis : True

/-- Theorem: The radius of the inscribed cylinder is 24/5 -/
theorem inscribed_cylinder_radius 
  (c : ConeWithCylinder) 
  (h1 : c.cone_diameter = 16) 
  (h2 : c.cone_altitude = 24) : 
  c.cylinder_radius = 24 / 5 := by sorry

end inscribed_cylinder_radius_l1083_108378


namespace apples_per_box_is_correct_l1083_108345

/-- The number of apples packed in a box -/
def apples_per_box : ℕ := 40

/-- The number of boxes packed per day in the first week -/
def boxes_per_day : ℕ := 50

/-- The number of fewer apples packed per day in the second week -/
def fewer_apples_per_day : ℕ := 500

/-- The total number of apples packed in two weeks -/
def total_apples : ℕ := 24500

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem apples_per_box_is_correct :
  (boxes_per_day * days_per_week * apples_per_box) +
  ((boxes_per_day * apples_per_box - fewer_apples_per_day) * days_per_week) = total_apples :=
by sorry

end apples_per_box_is_correct_l1083_108345


namespace food_bank_donations_boudin_del_monte_multiple_of_seven_l1083_108351

/-- Represents the total number of food items donated by five companies to a local food bank. -/
def total_donations (foster_farms : ℕ) : ℕ :=
  let american_summits := 2 * foster_farms
  let hormel := 3 * foster_farms
  let boudin_butchers := hormel / 3
  let del_monte := american_summits - 30
  foster_farms + american_summits + hormel + boudin_butchers + del_monte

/-- Theorem stating the total number of food items donated by the five companies. -/
theorem food_bank_donations : 
  total_donations 45 = 375 ∧ 
  (total_donations 45 - (45 + (2 * 45 - 30))) % 7 = 0 := by
  sorry

/-- Verification that the combined donations from Boudin Butchers and Del Monte Foods is a multiple of 7. -/
theorem boudin_del_monte_multiple_of_seven (foster_farms : ℕ) : 
  ((3 * foster_farms) / 3 + (2 * foster_farms - 30)) % 7 = 0 := by
  sorry

end food_bank_donations_boudin_del_monte_multiple_of_seven_l1083_108351


namespace digit_permutation_theorem_l1083_108324

/-- A k-digit number -/
def kDigitNumber (k : ℕ) := { n : ℕ // n < 10^k ∧ n ≥ 10^(k-1) }

/-- Inserting a k-digit number between two adjacent digits of another number -/
def insertNumber (n : ℕ) (k : ℕ) (a : kDigitNumber k) : ℕ := sorry

/-- Permutation of digits -/
def isPermutationOf (a b : ℕ) : Prop := sorry

theorem digit_permutation_theorem (k : ℕ) (p : ℕ) (A B : kDigitNumber k) :
  Prime p →
  p > 10^k →
  (∀ m : ℕ, m % p = 0 → (insertNumber m k A) % p = 0) →
  (∃ n : ℕ, (insertNumber n k A) % p = 0 ∧ (insertNumber (insertNumber n k A) k B) % p = 0) →
  isPermutationOf A.val B.val := by sorry

end digit_permutation_theorem_l1083_108324


namespace sequence_sum_bound_l1083_108379

theorem sequence_sum_bound (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n : ℕ, a n > 0) →
  (∀ n : ℕ, S n^2 - (n^2 + n - 1) * S n - (n^2 + n) = 0) →
  (∀ n : ℕ, b n = (n + 1) / ((n + 2)^2 * (a n)^2)) →
  (∀ n : ℕ, T (n + 1) = T n + b (n + 1)) →
  T 0 = 0 →
  ∀ n : ℕ, 0 < n → T n < 5/64 := by
sorry

end sequence_sum_bound_l1083_108379


namespace regular_9gon_coloring_l1083_108330

-- Define a regular 9-gon
structure RegularNineGon where
  vertices : Fin 9 → Point

-- Define a coloring of the vertices
inductive Color
| Black
| White

def Coloring := Fin 9 → Color

-- Define adjacency in the 9-gon
def adjacent (i j : Fin 9) : Prop :=
  (i.val + 1) % 9 = j.val ∨ (j.val + 1) % 9 = i.val

-- Define an isosceles triangle in the 9-gon
def isoscelesTriangle (i j k : Fin 9) (polygon : RegularNineGon) : Prop :=
  let d := (i.val - j.val + 9) % 9
  (i.val - k.val + 9) % 9 = d ∨ (j.val - k.val + 9) % 9 = d

theorem regular_9gon_coloring 
  (polygon : RegularNineGon) 
  (coloring : Coloring) : 
  (∃ i j : Fin 9, adjacent i j ∧ coloring i = coloring j) ∧ 
  (∃ i j k : Fin 9, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    coloring i = coloring j ∧ coloring j = coloring k ∧ 
    isoscelesTriangle i j k polygon) :=
by sorry

end regular_9gon_coloring_l1083_108330


namespace hoseok_number_l1083_108348

theorem hoseok_number : ∃ n : ℤ, n / 6 = 11 ∧ n = 66 := by
  sorry

end hoseok_number_l1083_108348


namespace simplify_and_evaluate_l1083_108331

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2) :
  (x + 2) * (x - 2) + 3 * (1 - x) = 1 - 3 * Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l1083_108331


namespace cubic_minus_linear_factorization_l1083_108338

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end cubic_minus_linear_factorization_l1083_108338


namespace derivative_at_negative_third_l1083_108358

/-- Given a function f(x) = x^2 + 2f'(-1/3)x, prove that f'(-1/3) = 2/3 -/
theorem derivative_at_negative_third (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + 2 * (deriv f (-1/3)) * x) :
  deriv f (-1/3) = 2/3 := by
  sorry

end derivative_at_negative_third_l1083_108358


namespace pirate_costume_group_size_l1083_108366

theorem pirate_costume_group_size 
  (costume_cost : ℕ) 
  (total_spent : ℕ) 
  (h1 : costume_cost = 5)
  (h2 : total_spent = 40) :
  total_spent / costume_cost = 8 :=
by
  sorry

end pirate_costume_group_size_l1083_108366


namespace max_pages_for_15_dollars_l1083_108342

/-- The cost in cents to copy 4 pages -/
def cost_per_4_pages : ℕ := 7

/-- The number of pages that can be copied for 4 cents -/
def pages_per_4_cents : ℕ := 4

/-- The amount in dollars available for copying -/
def available_dollars : ℕ := 15

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

/-- Calculates the maximum number of whole pages that can be copied -/
def max_pages : ℕ := 
  (dollars_to_cents available_dollars * pages_per_4_cents) / cost_per_4_pages

theorem max_pages_for_15_dollars : max_pages = 857 := by
  sorry

end max_pages_for_15_dollars_l1083_108342


namespace complex_trajectory_l1083_108399

theorem complex_trajectory (x y : ℝ) (z : ℂ) :
  z = x + y * I ∧ Complex.abs (z - 1) = x →
  y^2 = 2 * x - 1 :=
by sorry

end complex_trajectory_l1083_108399


namespace ball_travel_distance_l1083_108383

/-- The total distance traveled by a bouncing ball -/
def total_distance (initial_height : ℝ) (rebound_ratio : ℝ) : ℝ :=
  let first_rebound := initial_height * rebound_ratio
  let second_rebound := first_rebound * rebound_ratio
  initial_height + first_rebound + first_rebound + second_rebound + second_rebound

/-- Theorem: The ball travels 260 cm when it touches the floor for the third time -/
theorem ball_travel_distance :
  total_distance 104 0.5 = 260 := by
  sorry

end ball_travel_distance_l1083_108383


namespace stationery_sales_l1083_108319

theorem stationery_sales (total_sales : ℕ) (fabric_fraction : ℚ) (jewelry_fraction : ℚ)
  (h_total : total_sales = 36)
  (h_fabric : fabric_fraction = 1/3)
  (h_jewelry : jewelry_fraction = 1/4)
  (h_stationery : fabric_fraction + jewelry_fraction < 1) :
  total_sales - (total_sales * fabric_fraction).floor - (total_sales * jewelry_fraction).floor = 15 :=
by sorry

end stationery_sales_l1083_108319


namespace medium_stores_sampled_l1083_108391

/-- Represents the total number of stores in the city -/
def total_stores : ℕ := 1500

/-- Represents the ratio of large stores in the city -/
def large_ratio : ℕ := 1

/-- Represents the ratio of medium stores in the city -/
def medium_ratio : ℕ := 5

/-- Represents the ratio of small stores in the city -/
def small_ratio : ℕ := 9

/-- Represents the total number of stores to be sampled -/
def sample_size : ℕ := 30

/-- Theorem stating that the number of medium-sized stores to be sampled is 10 -/
theorem medium_stores_sampled : ℕ := by
  sorry

end medium_stores_sampled_l1083_108391


namespace unique_solution_l1083_108386

/-- Represents a three-digit number (abc) -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9
  h3 : c ≥ 0 ∧ c ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of the five rearranged numbers plus the original number -/
def N : Nat := 3306

/-- The equation that needs to be satisfied -/
def satisfiesEquation (n : ThreeDigitNumber) : Prop :=
  N + n.toNat = 222 * (n.a + n.b + n.c)

theorem unique_solution :
  ∃! n : ThreeDigitNumber, satisfiesEquation n ∧ n.toNat = 753 := by
  sorry

end unique_solution_l1083_108386


namespace weekly_syrup_cost_l1083_108344

/-- Calculates the weekly cost of syrup for a convenience store selling soda -/
theorem weekly_syrup_cost
  (weekly_soda_sales : ℕ)
  (gallons_per_box : ℕ)
  (cost_per_box : ℕ)
  (h_weekly_soda_sales : weekly_soda_sales = 180)
  (h_gallons_per_box : gallons_per_box = 30)
  (h_cost_per_box : cost_per_box = 40) :
  (weekly_soda_sales / gallons_per_box) * cost_per_box = 240 :=
by sorry

end weekly_syrup_cost_l1083_108344


namespace polygon_with_45_degree_exterior_angles_has_8_sides_l1083_108359

/-- A polygon with exterior angles measuring 45° has 8 sides. -/
theorem polygon_with_45_degree_exterior_angles_has_8_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    exterior_angle = 45 →
    (n : ℝ) * exterior_angle = 360 →
    n = 8 := by
  sorry

end polygon_with_45_degree_exterior_angles_has_8_sides_l1083_108359


namespace absolute_value_inequality_l1083_108382

theorem absolute_value_inequality (x : ℝ) : 
  |x + 1| - |x - 4| > 3 ↔ x > 3 := by sorry

end absolute_value_inequality_l1083_108382


namespace binomial_50_2_l1083_108308

theorem binomial_50_2 : Nat.choose 50 2 = 1225 := by
  sorry

end binomial_50_2_l1083_108308


namespace parallelogram_angle_ratio_l1083_108309

-- Define a parallelogram
structure Parallelogram :=
  (A B C D : Point)

-- Define the angles of the parallelogram
def angle (p : Parallelogram) (v : Fin 4) : ℝ :=
  sorry

-- State the theorem
theorem parallelogram_angle_ratio (p : Parallelogram) :
  ∃ (k : ℝ), k > 0 ∧
    angle p 0 = k ∧
    angle p 1 = 2 * k ∧
    angle p 2 = k ∧
    angle p 3 = 2 * k :=
  sorry

end parallelogram_angle_ratio_l1083_108309


namespace vacuum_cleaner_theorem_l1083_108389

def vacuum_cleaner_problem (initial_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) 
  (dog_walking_earnings : List ℕ) (discount_percent : ℕ) : ℕ × ℕ :=
  let discounted_cost := initial_cost - (initial_cost * discount_percent / 100)
  let total_savings := initial_savings + weekly_allowance * 3 + dog_walking_earnings.sum
  let amount_needed := discounted_cost - total_savings
  let weekly_savings := weekly_allowance + dog_walking_earnings.getLast!
  let weeks_needed := (amount_needed + weekly_savings - 1) / weekly_savings
  (amount_needed, weeks_needed)

theorem vacuum_cleaner_theorem : 
  vacuum_cleaner_problem 420 65 25 [40, 50, 30] 15 = (97, 2) := by
  sorry

end vacuum_cleaner_theorem_l1083_108389


namespace total_sheets_is_400_l1083_108300

/-- Calculates the total number of sheets of paper used for all students --/
def total_sheets (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) : ℕ :=
  num_classes * students_per_class * sheets_per_student

/-- Proves that the total number of sheets used is 400 --/
theorem total_sheets_is_400 :
  total_sheets 4 20 5 = 400 := by
  sorry

end total_sheets_is_400_l1083_108300


namespace chocolate_bar_cost_is_3_l1083_108349

def chocolate_bar_cost (total_cost : ℕ) (num_chocolate_bars : ℕ) (num_gummy_packs : ℕ) (num_chip_bags : ℕ) (gummy_cost : ℕ) (chip_cost : ℕ) : ℕ :=
  (total_cost - (num_gummy_packs * gummy_cost + num_chip_bags * chip_cost)) / num_chocolate_bars

theorem chocolate_bar_cost_is_3 :
  chocolate_bar_cost 150 10 10 20 2 5 = 3 := by
  sorry

end chocolate_bar_cost_is_3_l1083_108349


namespace solve_for_a_l1083_108347

theorem solve_for_a (x : ℝ) (a : ℝ) (h1 : x = 0.3) 
  (h2 : (a * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3) : a = 10 := by
  sorry

end solve_for_a_l1083_108347


namespace cubic_root_ratio_l1083_108361

theorem cubic_root_ratio (a b c d : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 1/2 ∨ x = 4) →
  c / d = 9/4 := by
sorry

end cubic_root_ratio_l1083_108361


namespace student_score_l1083_108341

theorem student_score (num_questions num_correct_answers points_per_question : ℕ) 
  (h1 : num_questions = 5)
  (h2 : num_correct_answers = 3)
  (h3 : points_per_question = 2) :
  num_correct_answers * points_per_question = 6 := by sorry

end student_score_l1083_108341


namespace geometric_sequence_product_l1083_108360

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a → a 5 * a 14 = 5 → a 8 * a 9 * a 10 * a 11 = 10 := by
  sorry

end geometric_sequence_product_l1083_108360


namespace dedekind_cut_B_dedekind_cut_D_l1083_108333

-- Define a Dedekind cut
def DedekindCut (M N : Set ℚ) : Prop :=
  (M ∪ N = Set.univ) ∧ 
  (M ∩ N = ∅) ∧ 
  (∀ x ∈ M, ∀ y ∈ N, x < y) ∧
  M.Nonempty ∧ 
  N.Nonempty

-- Statement B
theorem dedekind_cut_B : 
  ∃ M N : Set ℚ, DedekindCut M N ∧ 
  (¬∃ x, x = Sup M) ∧ 
  (∃ y, y = Inf N) :=
sorry

-- Statement D
theorem dedekind_cut_D : 
  ∃ M N : Set ℚ, DedekindCut M N ∧ 
  (¬∃ x, x = Sup M) ∧ 
  (¬∃ y, y = Inf N) :=
sorry

end dedekind_cut_B_dedekind_cut_D_l1083_108333


namespace henry_final_book_count_l1083_108374

/-- Calculates the final number of books Henry has after decluttering and acquiring new ones. -/
def final_book_count (initial_books : ℕ) (boxed_books : ℕ) (room_books : ℕ) 
  (coffee_table_books : ℕ) (cookbooks : ℕ) (new_books : ℕ) : ℕ :=
  initial_books - (3 * boxed_books + room_books + coffee_table_books + cookbooks) + new_books

/-- Theorem stating that Henry ends up with 23 books after the process. -/
theorem henry_final_book_count : 
  final_book_count 99 15 21 4 18 12 = 23 := by
  sorry

end henry_final_book_count_l1083_108374


namespace expression_value_l1083_108303

theorem expression_value (a : ℝ) (h : a^2 + 2*a - 1 = 0) : 
  ((a^2 - 1)/(a^2 - 2*a + 1) - 1/(1-a)) / (1/(a^2 - a)) = 1 := by
  sorry

end expression_value_l1083_108303


namespace students_in_both_band_and_chorus_l1083_108332

theorem students_in_both_band_and_chorus 
  (total_students : ℕ) 
  (band_students : ℕ) 
  (chorus_students : ℕ) 
  (band_or_chorus_students : ℕ) : 
  total_students = 200 →
  band_students = 70 →
  chorus_students = 95 →
  band_or_chorus_students = 150 →
  band_students + chorus_students - band_or_chorus_students = 15 := by
sorry

end students_in_both_band_and_chorus_l1083_108332


namespace proportion_inconsistency_l1083_108315

theorem proportion_inconsistency : ¬ ∃ (x : ℚ), (x / 2 = 2 / 6) ∧ (x = 3 / 4) := by
  sorry

end proportion_inconsistency_l1083_108315


namespace shaded_area_is_half_l1083_108340

/-- Represents a rectangle with a given area -/
structure Rectangle where
  area : ℝ

/-- Represents the shaded region after transformation -/
structure ShadedRegion where
  rectangle : Rectangle
  -- The rectangle is cut in two by a vertical cut joining the midpoints of its longer edges
  is_cut_in_half : Bool
  -- The right-hand half is given a quarter turn (90 degrees) about its center
  is_quarter_turned : Bool

/-- The area of the shaded region is half the area of the original rectangle -/
theorem shaded_area_is_half (r : Rectangle) (s : ShadedRegion) 
  (h1 : s.rectangle = r)
  (h2 : s.is_cut_in_half = true)
  (h3 : s.is_quarter_turned = true) :
  (s.rectangle.area / 2 : ℝ) = r.area / 2 :=
by sorry

#check shaded_area_is_half

end shaded_area_is_half_l1083_108340


namespace odd_function_sum_zero_l1083_108373

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_sum_zero (f : ℝ → ℝ) (h : OddFunction f) :
  f (-2) + f (-1) + f 0 + f 1 + f 2 = 0 := by
  sorry

end odd_function_sum_zero_l1083_108373


namespace percentage_of_seats_filled_l1083_108376

/-- Given a public show with 600 seats in total and 330 vacant seats,
    prove that 45% of the seats were filled. -/
theorem percentage_of_seats_filled (total_seats : ℕ) (vacant_seats : ℕ) : 
  total_seats = 600 →
  vacant_seats = 330 →
  (((total_seats - vacant_seats : ℚ) / total_seats) * 100 : ℚ) = 45 := by
  sorry

end percentage_of_seats_filled_l1083_108376


namespace second_day_average_speed_l1083_108317

/-- Represents the driving conditions and results over two days -/
structure DrivingData where
  total_distance : ℝ
  total_time : ℝ
  total_fuel : ℝ
  first_day_time_diff : ℝ
  first_day_speed_diff : ℝ
  first_day_efficiency : ℝ
  second_day_efficiency : ℝ

/-- Theorem stating that given the driving conditions, the average speed on the second day is 35 mph -/
theorem second_day_average_speed
  (data : DrivingData)
  (h1 : data.total_distance = 680)
  (h2 : data.total_time = 18)
  (h3 : data.total_fuel = 22.5)
  (h4 : data.first_day_time_diff = 2)
  (h5 : data.first_day_speed_diff = 5)
  (h6 : data.first_day_efficiency = 25)
  (h7 : data.second_day_efficiency = 30) :
  ∃ (second_day_speed : ℝ),
    second_day_speed = 35 ∧
    (second_day_speed + data.first_day_speed_diff) * (data.total_time / 2 + data.first_day_time_diff / 2) +
    second_day_speed * (data.total_time / 2 - data.first_day_time_diff / 2) = data.total_distance :=
by sorry

#check second_day_average_speed

end second_day_average_speed_l1083_108317


namespace square_sum_given_sum_square_and_product_l1083_108357

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 9) 
  (h2 : x * y = 2) : 
  x^2 + y^2 = 5 := by
  sorry

end square_sum_given_sum_square_and_product_l1083_108357


namespace probability_greater_than_three_l1083_108335

-- Define a standard die
def StandardDie : ℕ := 6

-- Define the favorable outcomes (numbers greater than 3)
def FavorableOutcomes : ℕ := 3

-- Theorem statement
theorem probability_greater_than_three (d : ℕ) (h : d = StandardDie) : 
  (FavorableOutcomes : ℚ) / d = 1 / 2 := by
  sorry


end probability_greater_than_three_l1083_108335


namespace min_value_reciprocal_sum_l1083_108390

/-- Given m > 0, n > 0, and the line y = (1/e)x + m + 1 is tangent to the curve y = ln x - n + 2,
    the minimum value of 1/m + 1/n is 4. -/
theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (h_tangent : ∃ x : ℝ, (1 / Real.exp 1) * x + m + 1 = Real.log x - n + 2 ∧
                        (1 / Real.exp 1) = 1 / x) :
  (1 / m + 1 / n) ≥ 4 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 1 / m₀ + 1 / n₀ = 4 :=
by sorry

end min_value_reciprocal_sum_l1083_108390


namespace hoseok_social_studies_score_l1083_108325

/-- Represents Hoseok's test scores -/
structure HoseokScores where
  average_three : ℝ  -- Average score of Korean, English, and Science
  average_four : ℝ   -- Average score after including Social studies
  social_studies : ℝ -- Score of Social studies test

/-- Theorem stating that given Hoseok's average scores, his Social studies score must be 93 -/
theorem hoseok_social_studies_score (scores : HoseokScores)
  (h1 : scores.average_three = 89)
  (h2 : scores.average_four = 90) :
  scores.social_studies = 93 := by
  sorry

#check hoseok_social_studies_score

end hoseok_social_studies_score_l1083_108325


namespace dime_difference_l1083_108397

/-- Represents the number of each type of coin in the piggy bank -/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  half_dollars : ℕ

/-- The total number of coins in the piggy bank -/
def total_coins : ℕ := 120

/-- The total value of coins in cents -/
def total_value : ℕ := 1240

/-- Calculates the total number of coins for a given CoinCount -/
def count_coins (c : CoinCount) : ℕ :=
  c.nickels + c.dimes + c.quarters + c.half_dollars

/-- Calculates the total value in cents for a given CoinCount -/
def calculate_value (c : CoinCount) : ℕ :=
  5 * c.nickels + 10 * c.dimes + 25 * c.quarters + 50 * c.half_dollars

/-- Defines a valid CoinCount that satisfies the problem conditions -/
def is_valid_count (c : CoinCount) : Prop :=
  count_coins c = total_coins ∧ calculate_value c = total_value

/-- Finds the maximum number of dimes possible -/
def max_dimes : ℕ := 128

/-- Finds the minimum number of dimes possible -/
def min_dimes : ℕ := 2

theorem dime_difference :
  ∃ (max min : CoinCount),
    is_valid_count max ∧
    is_valid_count min ∧
    max.dimes = max_dimes ∧
    min.dimes = min_dimes ∧
    max_dimes - min_dimes = 126 := by
  sorry

end dime_difference_l1083_108397


namespace sequence_non_positive_l1083_108339

theorem sequence_non_positive
  (n : ℕ)
  (a : ℕ → ℝ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h_ineq : ∀ k : ℕ, k < n → a k.pred - 2 * a k + a k.succ ≥ 0) :
  ∀ k : ℕ, k ≤ n → a k ≤ 0 :=
sorry

end sequence_non_positive_l1083_108339


namespace max_value_and_inequality_l1083_108301

noncomputable def f (x : ℝ) := Real.log (x + 1)

noncomputable def g (x : ℝ) := f x - x / 4 - 1

theorem max_value_and_inequality :
  (∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x) ∧
  (g (3 : ℝ) = 2 * Real.log 2 - 7 / 4) ∧
  (∀ (x : ℝ), x > 0 → f x < (Real.exp x - 1) / (x^2)) := by sorry

end max_value_and_inequality_l1083_108301


namespace tile_arrangements_l1083_108353

/-- The number of distinguishable arrangements of tiles of different colors -/
def distinguishable_arrangements (blue red green : ℕ) : ℕ :=
  Nat.factorial (blue + red + green) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green)

/-- Theorem stating that the number of distinguishable arrangements
    of 3 blue tiles, 2 red tiles, and 4 green tiles is 1260 -/
theorem tile_arrangements :
  distinguishable_arrangements 3 2 4 = 1260 := by
  sorry

#eval distinguishable_arrangements 3 2 4

end tile_arrangements_l1083_108353


namespace negation_of_forall_abs_sum_nonnegative_l1083_108327

theorem negation_of_forall_abs_sum_nonnegative :
  (¬ (∀ x : ℝ, x + |x| ≥ 0)) ↔ (∃ x : ℝ, x + |x| < 0) := by
  sorry

end negation_of_forall_abs_sum_nonnegative_l1083_108327


namespace min_distance_for_ten_trees_l1083_108392

/-- Calculates the minimum distance to water trees -/
def min_distance_to_water_trees (num_trees : ℕ) (tree_spacing : ℕ) : ℕ :=
  let max_trees_per_trip := 2  -- Xiao Zhang can water 2 trees per trip
  let num_full_trips := (num_trees - 1) / max_trees_per_trip
  let trees_on_last_trip := (num_trees - 1) % max_trees_per_trip + 1
  let full_trip_distance := num_full_trips * (max_trees_per_trip * tree_spacing * 2)
  let last_trip_distance := trees_on_last_trip * tree_spacing
  full_trip_distance + last_trip_distance

/-- The theorem to be proved -/
theorem min_distance_for_ten_trees :
  min_distance_to_water_trees 10 10 = 410 :=
sorry

end min_distance_for_ten_trees_l1083_108392


namespace scientific_notation_conversion_l1083_108316

theorem scientific_notation_conversion :
  (4.6 : ℝ) * (10 ^ 8) = 460000000 := by sorry

end scientific_notation_conversion_l1083_108316


namespace maximize_x_cubed_y_fourth_l1083_108336

theorem maximize_x_cubed_y_fourth (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 36) :
  x^3 * y^4 ≤ 18^3 * 6^4 ∧ (x^3 * y^4 = 18^3 * 6^4 ↔ x = 18 ∧ y = 6) :=
by sorry

end maximize_x_cubed_y_fourth_l1083_108336


namespace prime_square_minus_one_divisible_by_24_l1083_108305

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ k : ℕ, p^2 - 1 = 24 * k := by
  sorry

end prime_square_minus_one_divisible_by_24_l1083_108305


namespace tennis_ball_cost_l1083_108395

theorem tennis_ball_cost (num_packs : ℕ) (total_cost : ℚ) (balls_per_pack : ℕ) 
  (h1 : num_packs = 4)
  (h2 : total_cost = 24)
  (h3 : balls_per_pack = 3) :
  total_cost / (num_packs * balls_per_pack) = 2 := by
  sorry

end tennis_ball_cost_l1083_108395


namespace light_glow_time_l1083_108377

/-- The number of seconds between 1:57:58 am and 3:20:47 am -/
def total_seconds : ℕ := 4969

/-- The maximum number of times the light glowed -/
def max_glows : ℚ := 155.28125

/-- The time it takes for one glow in seconds -/
def time_per_glow : ℕ := 32

theorem light_glow_time :
  (total_seconds : ℚ) / max_glows = time_per_glow := by sorry

end light_glow_time_l1083_108377


namespace yellow_surface_fraction_proof_l1083_108352

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cubes : ℕ
  yellow_cubes : ℕ
  blue_cubes : ℕ

/-- Calculates the fraction of yellow surface area -/
def yellow_surface_fraction (cube : LargeCube) : ℚ :=
  sorry

theorem yellow_surface_fraction_proof (cube : LargeCube) :
  cube.edge_length = 4 →
  cube.small_cubes = 64 →
  cube.yellow_cubes = 15 →
  cube.blue_cubes = 49 →
  yellow_surface_fraction cube = 1/6 :=
sorry

end yellow_surface_fraction_proof_l1083_108352


namespace sqrt_eight_equals_two_sqrt_two_l1083_108381

theorem sqrt_eight_equals_two_sqrt_two : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_eight_equals_two_sqrt_two_l1083_108381


namespace unpainted_cubes_count_l1083_108318

/-- Represents a 4x4x4 cube composed of unit cubes -/
structure Cube :=
  (size : Nat)
  (total_units : Nat)
  (painted_per_face : Nat)

/-- The number of unpainted unit cubes in the cube -/
def unpainted_cubes (c : Cube) : Nat :=
  c.total_units - (c.painted_per_face * 6)

/-- Theorem stating the number of unpainted cubes in the specific cube configuration -/
theorem unpainted_cubes_count (c : Cube) 
  (h1 : c.size = 4)
  (h2 : c.total_units = 64)
  (h3 : c.painted_per_face = 4) :
  unpainted_cubes c = 40 := by
  sorry


end unpainted_cubes_count_l1083_108318


namespace symmetric_points_on_parabola_l1083_108310

/-- Two points on a parabola, symmetric with respect to a line -/
theorem symmetric_points_on_parabola (x₁ x₂ y₁ y₂ m : ℝ) :
  y₁ = 2 * x₁^2 →                          -- A is on the parabola
  y₂ = 2 * x₂^2 →                          -- B is on the parabola
  (y₂ - y₁) / (x₂ - x₁) = -1 →             -- A and B are symmetric (slope condition)
  (y₂ + y₁) / 2 = (x₂ + x₁) / 2 + m →      -- Midpoint of A and B lies on y = x + m
  x₁ * x₂ = -3/4 →                         -- Given condition
  m = 2 := by
sorry

end symmetric_points_on_parabola_l1083_108310


namespace prime_divisor_form_l1083_108346

theorem prime_divisor_form (a p : ℕ) (ha : a > 0) (hp : Nat.Prime p) 
  (hdiv : p ∣ a^3 - 3*a + 1) (hp_neq_3 : p ≠ 3) :
  ∃ k : ℤ, p = 9*k + 1 ∨ p = 9*k - 1 := by
sorry

end prime_divisor_form_l1083_108346


namespace circle_equation_from_diameter_l1083_108370

theorem circle_equation_from_diameter (A B : ℝ × ℝ) :
  A = (2, 0) →
  B = (0, 4) →
  ∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 5 ↔
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
      x = 2 * (1 - t) + 0 * t ∧
      y = 0 * (1 - t) + 4 * t :=
by sorry

end circle_equation_from_diameter_l1083_108370


namespace paco_cookies_proof_l1083_108396

/-- The number of cookies Paco had initially -/
def initial_cookies : ℕ := 2

theorem paco_cookies_proof :
  (∃ (x : ℕ), 
    (x - 2 + 36 = 2 + 34) ∧ 
    (x = initial_cookies)) := by
  sorry

end paco_cookies_proof_l1083_108396


namespace total_volume_of_prisms_l1083_108355

theorem total_volume_of_prisms (length width height : ℝ) (num_prisms : ℕ) 
  (h1 : length = 5)
  (h2 : width = 3)
  (h3 : height = 6)
  (h4 : num_prisms = 4) :
  length * width * height * num_prisms = 360 := by
  sorry

end total_volume_of_prisms_l1083_108355


namespace dalton_needs_four_more_l1083_108372

/-- The amount of additional money Dalton needs to buy his desired items -/
def additional_money_needed (jump_rope_cost board_game_cost ball_cost saved_allowance uncle_gift : ℕ) : ℕ :=
  let total_cost := jump_rope_cost + board_game_cost + ball_cost
  let available_money := saved_allowance + uncle_gift
  if total_cost > available_money then
    total_cost - available_money
  else
    0

/-- Theorem stating that Dalton needs $4 more to buy his desired items -/
theorem dalton_needs_four_more :
  additional_money_needed 7 12 4 6 13 = 4 := by
  sorry

end dalton_needs_four_more_l1083_108372


namespace purple_bows_count_l1083_108328

/-- Given a bag of bows with the following properties:
    - 1/4 of the bows are red
    - 1/3 of the bows are blue
    - 1/6 of the bows are purple
    - The remaining 60 bows are yellow
    This theorem proves that there are 40 purple bows. -/
theorem purple_bows_count (total : ℕ) (red blue purple yellow : ℕ) : 
  red + blue + purple + yellow = total →
  4 * red = total →
  3 * blue = total →
  6 * purple = total →
  yellow = 60 →
  purple = 40 := by
  sorry

#check purple_bows_count

end purple_bows_count_l1083_108328


namespace opposite_of_negative_two_l1083_108311

theorem opposite_of_negative_two :
  ∀ x : ℤ, x + (-2) = 0 → x = 2 := by
  sorry

end opposite_of_negative_two_l1083_108311
