import Mathlib

namespace cos_150_degrees_l573_57341

theorem cos_150_degrees : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_150_degrees_l573_57341


namespace intersection_A_complement_B_l573_57360

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 0}

-- Define set B
def B : Set ℝ := {x | x ≤ -1}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (Set.compl B) = {x | -1 < x ∧ x < 0} := by sorry

end intersection_A_complement_B_l573_57360


namespace power_two_vs_square_l573_57379

theorem power_two_vs_square (n : ℕ+) :
  (n = 2 ∨ n = 4 → 2^(n:ℕ) = n^2) ∧
  (n = 3 → 2^(n:ℕ) < n^2) ∧
  (n = 1 ∨ n > 4 → 2^(n:ℕ) > n^2) := by
  sorry

end power_two_vs_square_l573_57379


namespace valid_schedules_l573_57350

/-- Represents the number of periods in a day -/
def total_periods : ℕ := 9

/-- Represents the number of morning periods -/
def morning_periods : ℕ := 5

/-- Represents the number of afternoon periods -/
def afternoon_periods : ℕ := 4

/-- Represents the number of classes to be taught -/
def classes_to_teach : ℕ := 3

/-- Calculates the number of ways to arrange n items taken k at a time -/
def arrangement (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- Calculates the number of prohibited arrangements in the morning -/
def morning_prohibited : ℕ := 3 * Nat.factorial classes_to_teach

/-- Calculates the number of prohibited arrangements in the afternoon -/
def afternoon_prohibited : ℕ := 2 * Nat.factorial classes_to_teach

/-- The main theorem stating the number of valid schedules -/
theorem valid_schedules : 
  arrangement total_periods classes_to_teach - morning_prohibited - afternoon_prohibited = 474 := by
  sorry


end valid_schedules_l573_57350


namespace emily_holidays_l573_57357

/-- The number of holidays Emily takes in a year -/
def holidays_per_year (days_off_per_month : ℕ) (months_in_year : ℕ) : ℕ :=
  days_off_per_month * months_in_year

/-- Theorem: Emily takes 24 holidays in a year -/
theorem emily_holidays :
  holidays_per_year 2 12 = 24 := by
  sorry

end emily_holidays_l573_57357


namespace kyle_caught_14_fish_l573_57381

/-- The number of fish Kyle caught given the conditions of the problem -/
def kyles_fish (total : ℕ) (carlas : ℕ) : ℕ :=
  (total - carlas) / 2

/-- Theorem stating that Kyle caught 14 fish under the given conditions -/
theorem kyle_caught_14_fish (total : ℕ) (carlas : ℕ) 
  (h1 : total = 36) 
  (h2 : carlas = 8) : 
  kyles_fish total carlas = 14 := by
  sorry

#eval kyles_fish 36 8

end kyle_caught_14_fish_l573_57381


namespace crayons_distribution_l573_57386

theorem crayons_distribution (total_crayons : ℝ) (x : ℝ) : 
  total_crayons = 210 →
  x / total_crayons = 1 / 30 →
  30 * x = 0.7 * total_crayons →
  x = 4.9 := by
sorry

end crayons_distribution_l573_57386


namespace intersection_equality_implies_a_geq_one_l573_57303

-- Define sets A and B
def A : Set ℝ := {x | 3 * x + 1 < 4}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

-- State the theorem
theorem intersection_equality_implies_a_geq_one (a : ℝ) :
  A ∩ B a = A → a ≥ 1 := by sorry

end intersection_equality_implies_a_geq_one_l573_57303


namespace range_of_positive_values_l573_57384

def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem range_of_positive_values 
  (f : ℝ → ℝ) 
  (odd : OddFunction f)
  (incr_neg : ∀ x y, x < y ∧ y ≤ 0 → f x < f y)
  (f_neg_one_zero : f (-1) = 0) :
  {x : ℝ | f x > 0} = Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi 1 := by
sorry

end range_of_positive_values_l573_57384


namespace distance_between_cities_l573_57367

theorem distance_between_cities (t : ℝ) : ∃ x : ℝ,
  x / 50 = t - 1 ∧ x / 35 = t + 2 → x = 350 := by
  sorry

end distance_between_cities_l573_57367


namespace remainder_problem_l573_57336

theorem remainder_problem (N : ℤ) : N % 296 = 75 → N % 37 = 1 := by
  sorry

end remainder_problem_l573_57336


namespace zeros_in_fraction_l573_57383

def count_leading_zeros (n : ℚ) : ℕ :=
  sorry

theorem zeros_in_fraction : count_leading_zeros (1 / (2^7 * 5^9)) = 8 := by
  sorry

end zeros_in_fraction_l573_57383


namespace remainder_4123_div_32_l573_57302

theorem remainder_4123_div_32 : 4123 % 32 = 27 := by
  sorry

end remainder_4123_div_32_l573_57302


namespace sector_central_angle_l573_57329

/-- Given a sector with perimeter 10 and area 4, prove that its central angle is 1/2 radian -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 10) (h2 : (1/2) * l * r = 4) :
  l / r = 1 / 2 := by
  sorry

end sector_central_angle_l573_57329


namespace quadratic_form_only_trivial_solution_l573_57345

theorem quadratic_form_only_trivial_solution (a b c d : ℤ) :
  a^2 + 5*b^2 - 2*c^2 - 2*c*d - 3*d^2 = 0 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry

end quadratic_form_only_trivial_solution_l573_57345


namespace convex_polygon_interior_angles_l573_57370

theorem convex_polygon_interior_angles (n : ℕ) : 
  n ≥ 3 →  -- Convex polygon has at least 3 sides
  (∀ k, k ∈ Finset.range n → 
    100 + k * 10 < 180) →  -- All interior angles are less than 180°
  (100 + (n - 1) * 10 ≥ 180) →  -- The largest angle is at least 180°
  n = 8 := by
  sorry

end convex_polygon_interior_angles_l573_57370


namespace cosine_sine_square_difference_l573_57349

theorem cosine_sine_square_difference (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.cos α ^ 2 - Real.sin α ^ 2 = -(Real.sqrt 5 / 3) := by
  sorry

end cosine_sine_square_difference_l573_57349


namespace roses_cut_l573_57378

theorem roses_cut (initial_roses final_roses : ℕ) (h1 : initial_roses = 3) (h2 : final_roses = 14) :
  final_roses - initial_roses = 11 := by
  sorry

end roses_cut_l573_57378


namespace arithmetic_sequence_sum_l573_57305

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Theorem: In an arithmetic sequence where a_4 = 5, a_3 + a_5 = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) (h_a4 : a 4 = 5) : 
  a 3 + a 5 = 10 := by
  sorry

end arithmetic_sequence_sum_l573_57305


namespace jills_yard_area_l573_57314

/-- Represents a rectangular yard with fence posts -/
structure FencedYard where
  shorterSidePosts : ℕ
  longerSidePosts : ℕ
  postSpacing : ℕ

/-- The total number of fence posts -/
def FencedYard.totalPosts (yard : FencedYard) : ℕ :=
  2 * (yard.shorterSidePosts + yard.longerSidePosts) - 4

/-- The length of the shorter side of the yard -/
def FencedYard.shorterSide (yard : FencedYard) : ℕ :=
  yard.postSpacing * (yard.shorterSidePosts - 1)

/-- The length of the longer side of the yard -/
def FencedYard.longerSide (yard : FencedYard) : ℕ :=
  yard.postSpacing * (yard.longerSidePosts - 1)

/-- The area of the yard -/
def FencedYard.area (yard : FencedYard) : ℕ :=
  yard.shorterSide * yard.longerSide

/-- Theorem: The area of Jill's yard is 144 square yards -/
theorem jills_yard_area :
  ∃ (yard : FencedYard),
    yard.totalPosts = 24 ∧
    yard.postSpacing = 3 ∧
    yard.longerSidePosts = 3 * yard.shorterSidePosts ∧
    yard.area = 144 :=
by
  sorry


end jills_yard_area_l573_57314


namespace exist_consecutive_amazing_numbers_l573_57344

/-- Definition of an amazing number -/
def is_amazing (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    n = (Nat.gcd b c) * (Nat.gcd a (b*c)) + 
        (Nat.gcd c a) * (Nat.gcd b (c*a)) + 
        (Nat.gcd a b) * (Nat.gcd c (a*b))

/-- Theorem: There exist 2011 consecutive amazing numbers -/
theorem exist_consecutive_amazing_numbers : 
  ∃ start : ℕ, ∀ i : ℕ, i < 2011 → is_amazing (start + i) := by
  sorry

end exist_consecutive_amazing_numbers_l573_57344


namespace circle_tangent_to_line_l573_57385

/-- The line to which the circle is tangent -/
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 9

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -1)

/-- The theorem stating that the given equation represents the circle with the specified properties -/
theorem circle_tangent_to_line :
  ∀ x y : ℝ,
  circle_equation x y ↔
  (∃ r : ℝ, r > 0 ∧
    (x - circle_center.1)^2 + (y - circle_center.2)^2 = r^2 ∧
    r = |3 * circle_center.1 - 4 * circle_center.2 + 5| / 5) :=
sorry

end circle_tangent_to_line_l573_57385


namespace greatest_prime_factor_of_3_7_plus_6_6_l573_57307

theorem greatest_prime_factor_of_3_7_plus_6_6 :
  ∃ p : ℕ, p.Prime ∧ p ∣ (3^7 + 6^6) ∧ ∀ q : ℕ, q.Prime → q ∣ (3^7 + 6^6) → q ≤ p ∧ p = 67 :=
sorry

end greatest_prime_factor_of_3_7_plus_6_6_l573_57307


namespace triangle_altitude_segment_l573_57338

/-- Given a triangle with sides 40, 60, and 80 units, prove that the larger segment
    cut off by an altitude to the side of length 80 is 52.5 units long. -/
theorem triangle_altitude_segment (a b c h x : ℝ) : 
  a = 40 ∧ b = 60 ∧ c = 80 ∧ 
  a^2 = x^2 + h^2 ∧
  b^2 = (c - x)^2 + h^2 →
  c - x = 52.5 := by
  sorry

end triangle_altitude_segment_l573_57338


namespace problem_solution_l573_57312

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x - 4 = 21 * (1 / x)) (h2 : x + y^2 = 45) : x = 7 ∧ y = Real.sqrt 38 := by
  sorry

end problem_solution_l573_57312


namespace point_in_fourth_quadrant_l573_57323

theorem point_in_fourth_quadrant (a : ℤ) : 
  (2*a + 6 > 0) ∧ (3*a + 3 < 0) → (2*a + 6 = 2 ∧ 3*a + 3 = -3) :=
by sorry

end point_in_fourth_quadrant_l573_57323


namespace min_sum_of_parallel_vectors_l573_57343

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![1, x - 1]
def b (y : ℝ) : Fin 2 → ℝ := ![y, 2]

-- Theorem statement
theorem min_sum_of_parallel_vectors (x y : ℝ) 
  (h1 : x - 1 ≥ 0) 
  (h2 : y ≥ 0) 
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ a x = k • b y) : 
  x + y ≥ 2 * Real.sqrt 2 := by
  sorry

end min_sum_of_parallel_vectors_l573_57343


namespace calculate_c_investment_c_investment_is_20000_l573_57340

/-- Calculates C's investment in a partnership given the investments of A and B,
    C's share of profit, and the total profit. -/
theorem calculate_c_investment (a_investment b_investment : ℕ)
                                (c_profit_share total_profit : ℕ) : ℕ :=
  let x := c_profit_share * (a_investment + b_investment + c_profit_share * total_profit / c_profit_share) / 
           (total_profit - c_profit_share)
  x

/-- Proves that C's investment is 20,000 given the specified conditions -/
theorem c_investment_is_20000 :
  calculate_c_investment 12000 16000 36000 86400 = 20000 := by
  sorry

end calculate_c_investment_c_investment_is_20000_l573_57340


namespace xyz_value_l573_57376

theorem xyz_value (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 280 / 3 := by
  sorry

end xyz_value_l573_57376


namespace one_right_intersection_implies_negative_n_l573_57362

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a quadratic function has one intersection point with the x-axis to the right of the y-axis -/
def hasOneRightIntersection (f : QuadraticFunction) : Prop :=
  ∃ x : ℝ, x > 0 ∧ f.a * x^2 + f.b * x + f.c = 0 ∧
  ∀ y : ℝ, y ≠ x → f.a * y^2 + f.b * y + f.c ≠ 0

/-- Theorem: If a quadratic function y = x^2 + 3x + n has one intersection point
    with the x-axis to the right of the y-axis, then n < 0 -/
theorem one_right_intersection_implies_negative_n :
  ∀ n : ℝ, hasOneRightIntersection ⟨1, 3, n⟩ → n < 0 := by
  sorry


end one_right_intersection_implies_negative_n_l573_57362


namespace finite_solutions_l573_57371

def f (z : ℂ) : ℂ := z^2 + Complex.I * z + 1

theorem finite_solutions :
  ∃ (S : Finset ℂ), ∀ z : ℂ,
    Complex.im z > 0 ∧
    (∃ (a b : ℤ), f z = ↑a + ↑b * Complex.I ∧ abs a ≤ 15 ∧ abs b ≤ 15) →
    z ∈ S :=
sorry

end finite_solutions_l573_57371


namespace corn_harvest_problem_l573_57313

/-- Represents the corn harvest problem -/
theorem corn_harvest_problem 
  (initial_harvest : ℝ) 
  (planned_harvest : ℝ) 
  (area_increase : ℝ) 
  (yield_improvement : ℝ) 
  (h1 : initial_harvest = 4340)
  (h2 : planned_harvest = 5520)
  (h3 : area_increase = 14)
  (h4 : yield_improvement = 5)
  (h5 : initial_harvest / 124 < 40) :
  ∃ (initial_area yield : ℝ),
    initial_area = 124 ∧ 
    yield = 35 ∧
    initial_harvest = initial_area * yield ∧
    planned_harvest = (initial_area + area_increase) * (yield + yield_improvement) := by
  sorry

end corn_harvest_problem_l573_57313


namespace not_all_nonnegative_l573_57380

theorem not_all_nonnegative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  ¬(0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) := by
  sorry

end not_all_nonnegative_l573_57380


namespace arcsin_cos_two_pi_thirds_l573_57308

theorem arcsin_cos_two_pi_thirds : 
  Real.arcsin (Real.cos (2 * π / 3)) = -π / 6 := by
  sorry

end arcsin_cos_two_pi_thirds_l573_57308


namespace arts_students_count_l573_57355

/-- Represents the number of arts students in the college -/
def arts_students : ℕ := sorry

/-- Represents the number of local arts students -/
def local_arts_students : ℕ := sorry

/-- Represents the number of local science students -/
def local_science_students : ℕ := 25

/-- Represents the number of local commerce students -/
def local_commerce_students : ℕ := 102

/-- Represents the total number of local students -/
def total_local_students : ℕ := 327

/-- Theorem stating that the number of arts students is 400 -/
theorem arts_students_count :
  (local_arts_students = arts_students / 2) ∧
  (local_arts_students + local_science_students + local_commerce_students = total_local_students) →
  arts_students = 400 :=
by sorry

end arts_students_count_l573_57355


namespace untouched_area_of_tetrahedron_l573_57304

/-- The area of a regular tetrahedron's inner wall that cannot be touched by an inscribed sphere -/
theorem untouched_area_of_tetrahedron (r : ℝ) (a : ℝ) (h : a = 4 * Real.sqrt 6) :
  let total_surface_area := a^2 * Real.sqrt 3
  let touched_area := (a^2 * Real.sqrt 3) / 4
  total_surface_area - touched_area = 108 * Real.sqrt 3 :=
by sorry

end untouched_area_of_tetrahedron_l573_57304


namespace max_consecutive_working_days_l573_57352

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Checks if a given date is a working day for the guard -/
def isWorkingDay (d : Date) : Prop :=
  d.dayOfWeek = DayOfWeek.Tuesday ∨ 
  d.dayOfWeek = DayOfWeek.Friday ∨ 
  d.day % 2 = 1

/-- Represents a sequence of consecutive dates -/
def ConsecutiveDates (n : Nat) := Fin n → Date

/-- Checks if all dates in a sequence are working days -/
def allWorkingDays (dates : ConsecutiveDates n) : Prop :=
  ∀ i, isWorkingDay (dates i)

/-- The main theorem: The maximum number of consecutive working days is 6 -/
theorem max_consecutive_working_days :
  (∃ (dates : ConsecutiveDates 6), allWorkingDays dates) ∧
  (∀ (dates : ConsecutiveDates 7), ¬ allWorkingDays dates) :=
sorry

end max_consecutive_working_days_l573_57352


namespace system_solution_l573_57373

theorem system_solution : ∃! (x y : ℝ), 
  (x^2 * y + x * y^2 + 3*x + 3*y + 24 = 0) ∧ 
  (x^3 * y - x * y^3 + 3*x^2 - 3*y^2 - 48 = 0) ∧
  (x = -3) ∧ (y = -1) := by
sorry

end system_solution_l573_57373


namespace fraction_difference_equality_l573_57374

theorem fraction_difference_equality (x y : ℝ) : 
  let P := x^2 + y^2
  let Q := x - y
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = 4 * x * y / ((x^2 + y^2)^2 - (x - y)^2) :=
by sorry

end fraction_difference_equality_l573_57374


namespace total_guests_proof_l573_57317

def number_of_guests (adults children seniors teenagers toddlers vip : ℕ) : ℕ :=
  adults + children + seniors + teenagers + toddlers + vip

theorem total_guests_proof :
  ∃ (adults children seniors teenagers toddlers vip : ℕ),
    adults = 58 ∧
    children = adults - 35 ∧
    seniors = 2 * children ∧
    teenagers = seniors - 15 ∧
    toddlers = teenagers / 2 ∧
    vip = teenagers - 20 ∧
    ∃ (n : ℕ), vip = n^2 ∧
    number_of_guests adults children seniors teenagers toddlers vip = 198 :=
by
  sorry

end total_guests_proof_l573_57317


namespace height_difference_proof_l573_57389

/-- Proves that the height difference between Vlad and his sister is 104.14 cm -/
theorem height_difference_proof (vlad_height_m : ℝ) (sister_height_cm : ℝ) 
  (h1 : vlad_height_m = 1.905) (h2 : sister_height_cm = 86.36) : 
  vlad_height_m * 100 - sister_height_cm = 104.14 := by
  sorry

end height_difference_proof_l573_57389


namespace fourth_term_expansion_l573_57318

/-- The fourth term in the binomial expansion of (1/x + x)^n, where n is determined by the condition that the binomial coefficients of the third and seventh terms are equal. -/
def fourth_term (x : ℝ) : ℝ := 56 * x^2

/-- The condition that the binomial coefficients of the third and seventh terms are equal. -/
def binomial_coefficient_condition (n : ℕ) : Prop :=
  Nat.choose n 2 = Nat.choose n 6

theorem fourth_term_expansion (x : ℝ) :
  binomial_coefficient_condition 8 →
  fourth_term x = Nat.choose 8 3 * (1/x)^3 * x^5 :=
sorry

end fourth_term_expansion_l573_57318


namespace roommate_condition_not_satisfied_l573_57334

-- Define the functions for John's and Bob's roommates
def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 1
def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 5

-- Theorem stating that the condition is not satisfied after 3 years
theorem roommate_condition_not_satisfied : f 3 ≠ 2 * g 3 + 5 := by
  sorry

end roommate_condition_not_satisfied_l573_57334


namespace yuri_total_puppies_l573_57361

def puppies_week1 : ℕ := 20

def puppies_week2 : ℕ := (2 * puppies_week1) / 5

def puppies_week3 : ℕ := 2 * puppies_week2

def puppies_week4 : ℕ := puppies_week1 + 10

def total_puppies : ℕ := puppies_week1 + puppies_week2 + puppies_week3 + puppies_week4

theorem yuri_total_puppies : total_puppies = 74 := by
  sorry

end yuri_total_puppies_l573_57361


namespace min_sum_squares_l573_57316

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             m = 4 := by
  sorry

#check min_sum_squares

end min_sum_squares_l573_57316


namespace product_of_three_numbers_l573_57346

theorem product_of_three_numbers (a b c m : ℝ) 
  (sum_eq : a + b + c = 195)
  (m_eq_8a : m = 8 * a)
  (m_eq_b_minus_10 : m = b - 10)
  (m_eq_c_plus_10 : m = c + 10)
  (a_smallest : a < b ∧ a < c) :
  a * b * c = 95922 := by
sorry

end product_of_three_numbers_l573_57346


namespace shorterToLongerBaseRatio_l573_57372

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Length of the shorter base -/
  s : ℝ
  /-- Length of the longer base -/
  t : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The length of the longer base is equal to the length of its diagonals -/
  longerBaseEqualsDiagonal : True
  /-- The length of the shorter base is equal to the height -/
  shorterBaseEqualsHeight : True

/-- The ratio of the shorter base to the longer base is 3/5 -/
theorem shorterToLongerBaseRatio (trap : IsoscelesTrapezoid) : 
  trap.s / trap.t = 3 / 5 := by
  sorry

end shorterToLongerBaseRatio_l573_57372


namespace cubic_sum_ratio_l573_57382

theorem cubic_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end cubic_sum_ratio_l573_57382


namespace probability_not_jim_pictures_l573_57342

/-- Given a set of pictures, calculate the probability of picking two pictures
    that are not among those bought by Jim. -/
theorem probability_not_jim_pictures
  (total_pictures : ℕ)
  (jim_bought : ℕ)
  (pick_count : ℕ)
  (h_total : total_pictures = 10)
  (h_jim : jim_bought = 3)
  (h_pick : pick_count = 2) :
  (pick_count.choose (total_pictures - jim_bought)) / (pick_count.choose total_pictures) = 7 / 15 := by
  sorry

end probability_not_jim_pictures_l573_57342


namespace original_cost_after_discount_l573_57332

theorem original_cost_after_discount (decreased_cost : ℝ) (discount_rate : ℝ) :
  decreased_cost = 100 ∧ discount_rate = 0.5 → 
  ∃ original_cost : ℝ, original_cost = 200 ∧ decreased_cost = original_cost * (1 - discount_rate) :=
by
  sorry

end original_cost_after_discount_l573_57332


namespace solution_set_f_positive_l573_57391

/-- A function f is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem solution_set_f_positive
    (f : ℝ → ℝ)
    (h_even : EvenFunction f)
    (h_nonneg : ∀ x ≥ 0, f x = 2^x - 4) :
    {x : ℝ | f x > 0} = Set.Ioi 2 ∪ Set.Iio (-2) :=
  sorry

end solution_set_f_positive_l573_57391


namespace kelvins_classes_l573_57397

theorem kelvins_classes (grant_vacations kelvin_classes : ℕ) 
  (h1 : grant_vacations = 4 * kelvin_classes)
  (h2 : grant_vacations + kelvin_classes = 450) :
  kelvin_classes = 90 := by
  sorry

end kelvins_classes_l573_57397


namespace negation_of_absolute_value_implication_is_true_l573_57326

theorem negation_of_absolute_value_implication_is_true : 
  (∃ x : ℝ, (|x| ≤ 1 ∧ x > 1) ∨ (|x| > 1 ∧ x ≤ 1)) = False :=
by sorry

end negation_of_absolute_value_implication_is_true_l573_57326


namespace continued_fraction_equality_l573_57353

/-- Defines the continued fraction [2; 2, ..., 2] with n occurrences of 2 -/
def continued_fraction (n : ℕ) : ℚ :=
  if n = 0 then 2
  else 2 + 1 / continued_fraction (n - 1)

/-- The main theorem stating the equality of the continued fraction and the algebraic expression -/
theorem continued_fraction_equality (n : ℕ) :
  continued_fraction n = (((1 + Real.sqrt 2) ^ (n + 1) - (1 - Real.sqrt 2) ^ (n + 1)) /
                          ((1 + Real.sqrt 2) ^ n - (1 - Real.sqrt 2) ^ n)) := by
  sorry


end continued_fraction_equality_l573_57353


namespace solution_set_eq_neg_reals_l573_57387

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Axiom: f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Given conditions
axiom condition1 : ∀ x, f x + f' x < 1
axiom condition2 : f 0 = 2016

-- Define the solution set
def solution_set : Set ℝ := {x | Real.exp x * f x - Real.exp x > 2015}

-- Theorem statement
theorem solution_set_eq_neg_reals : solution_set f = Set.Iio 0 := by sorry

end solution_set_eq_neg_reals_l573_57387


namespace open_cells_are_perfect_squares_l573_57358

/-- Represents whether a cell is open (true) or closed (false) -/
def CellState := Bool

/-- The state of a cell after the jailer's procedure -/
def final_cell_state (n : ℕ) : CellState :=
  sorry

/-- A number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- The main theorem: a cell remains open iff its number is a perfect square -/
theorem open_cells_are_perfect_squares (n : ℕ) :
  final_cell_state n = true ↔ is_perfect_square n :=
  sorry

end open_cells_are_perfect_squares_l573_57358


namespace john_driving_distance_l573_57327

/-- Represents the efficiency of John's car in miles per gallon -/
def car_efficiency : ℝ := 40

/-- Represents the current price of gas in dollars per gallon -/
def gas_price : ℝ := 5

/-- Represents the amount of money John has to spend on gas in dollars -/
def available_money : ℝ := 25

/-- Theorem stating that John can drive exactly 200 miles with the given conditions -/
theorem john_driving_distance : 
  (available_money / gas_price) * car_efficiency = 200 := by
  sorry

end john_driving_distance_l573_57327


namespace no_integer_solutions_l573_57300

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ),
    (x^6 + x^3 + x^3*y + y = 147^157) ∧
    (x^3 + x^3*y + y^2 + y + z^9 = 157^147) := by
  sorry

end no_integer_solutions_l573_57300


namespace quadratic_equation_at_most_one_solution_l573_57324

theorem quadratic_equation_at_most_one_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a ≥ 9/8 ∨ a = 0) :=
by sorry

end quadratic_equation_at_most_one_solution_l573_57324


namespace no_solution_for_sock_problem_l573_57388

theorem no_solution_for_sock_problem : ¬ ∃ (m n : ℕ), 
  m + n = 2009 ∧ 
  (m^2 - m + n^2 - n : ℚ) / (2009 * 2008) = 1/2 := by
sorry

end no_solution_for_sock_problem_l573_57388


namespace money_distribution_l573_57322

theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 500)
  (h2 : B + C = 340)
  (h3 : C = 40) :
  A + C = 200 := by
sorry

end money_distribution_l573_57322


namespace constant_ratio_problem_l573_57398

theorem constant_ratio_problem (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) (k : ℝ) :
  (3 * x₁ - 4) / (y₁ + 7) = k →
  (3 * x₂ - 4) / (y₂ + 7) = k →
  x₁ = 3 →
  y₁ = 5 →
  y₂ = 20 →
  x₂ = 5.0833 := by
sorry

end constant_ratio_problem_l573_57398


namespace graph_decomposition_l573_57375

/-- The graph of the equation (x^2 - 1)(x+y) = y^2(x+y) -/
def GraphEquation (x y : ℝ) : Prop :=
  (x^2 - 1) * (x + y) = y^2 * (x + y)

/-- The line y = -x -/
def Line (x y : ℝ) : Prop :=
  y = -x

/-- The hyperbola (x+y)(x-y) = 1 -/
def Hyperbola (x y : ℝ) : Prop :=
  (x + y) * (x - y) = 1

theorem graph_decomposition :
  ∀ x y : ℝ, GraphEquation x y ↔ (Line x y ∨ Hyperbola x y) :=
sorry

end graph_decomposition_l573_57375


namespace line_tangent_to_parabola_l573_57306

/-- A line is tangent to a parabola if and only if there exists a point (x₀, y₀) that satisfies
    the following conditions:
    1. The point lies on the line: x₀ - y₀ - 1 = 0
    2. The point lies on the parabola: y₀ = a * x₀^2
    3. The slope of the tangent line equals the derivative of the parabola at that point: 1 = 2 * a * x₀
-/
theorem line_tangent_to_parabola (a : ℝ) :
  (∃ x₀ y₀ : ℝ, x₀ - y₀ - 1 = 0 ∧ y₀ = a * x₀^2 ∧ 1 = 2 * a * x₀) ↔ a = 1/4 := by
  sorry

end line_tangent_to_parabola_l573_57306


namespace shirt_cost_percentage_l573_57311

theorem shirt_cost_percentage (pants_cost shirt_cost total_cost : ℝ) : 
  pants_cost = 50 →
  total_cost = 130 →
  shirt_cost + pants_cost = total_cost →
  (shirt_cost - pants_cost) / pants_cost * 100 = 60 := by
sorry

end shirt_cost_percentage_l573_57311


namespace kwik_e_tax_center_problem_l573_57393

/-- The Kwik-e-Tax Center problem -/
theorem kwik_e_tax_center_problem 
  (federal_charge : ℕ) 
  (state_charge : ℕ) 
  (quarterly_charge : ℕ)
  (federal_sold : ℕ) 
  (state_sold : ℕ) 
  (total_revenue : ℕ)
  (h1 : federal_charge = 50)
  (h2 : state_charge = 30)
  (h3 : quarterly_charge = 80)
  (h4 : federal_sold = 60)
  (h5 : state_sold = 20)
  (h6 : total_revenue = 4400) :
  ∃ (quarterly_sold : ℕ), 
    federal_charge * federal_sold + 
    state_charge * state_sold + 
    quarterly_charge * quarterly_sold = total_revenue ∧ 
    quarterly_sold = 10 := by
  sorry

end kwik_e_tax_center_problem_l573_57393


namespace circle_radius_increase_l573_57395

/-- If a circle's radius r is increased by n, and its new area is twice the original area,
    then r = n(√2 + 1) -/
theorem circle_radius_increase (n : ℝ) (h : n > 0) :
  ∃ (r : ℝ), r > 0 ∧ π * (r + n)^2 = 2 * π * r^2 → r = n * (Real.sqrt 2 + 1) := by
  sorry

end circle_radius_increase_l573_57395


namespace book_reading_end_day_l573_57331

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (startDay : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => startDay
  | n + 1 => nextDay (advanceDays startDay n)

theorem book_reading_end_day :
  let numBooks : Nat := 20
  let startDay := DayOfWeek.Wednesday
  let totalDays := (numBooks * (numBooks + 1)) / 2
  advanceDays startDay totalDays = startDay := by
  sorry


end book_reading_end_day_l573_57331


namespace probability_different_colors_bags_l573_57335

/-- Represents a bag of colored balls -/
structure Bag where
  white : ℕ
  red : ℕ
  black : ℕ

/-- Calculates the total number of balls in a bag -/
def Bag.total (b : Bag) : ℕ := b.white + b.red + b.black

/-- Calculates the probability of drawing a ball of a specific color from a bag -/
def probability_color (b : Bag) (color : ℕ) : ℚ :=
  color / b.total

/-- Calculates the probability of drawing balls of different colors from two bags -/
def probability_different_colors (a b : Bag) : ℚ :=
  1 - (probability_color a a.white * probability_color b b.white +
       probability_color a a.red * probability_color b b.red +
       probability_color a a.black * probability_color b b.black)

theorem probability_different_colors_bags :
  let bag_a : Bag := { white := 4, red := 5, black := 6 }
  let bag_b : Bag := { white := 7, red := 6, black := 2 }
  probability_different_colors bag_a bag_b = 31 / 45 := by
  sorry

end probability_different_colors_bags_l573_57335


namespace tangent_circles_m_value_l573_57365

/-- The value of m for which the circle x² + y² = 1 is tangent to the circle x² + y² + 6x - 8y + m = 0 -/
theorem tangent_circles_m_value : ∃ m : ℝ, 
  (∀ x y : ℝ, x^2 + y^2 = 1 → x^2 + y^2 + 6*x - 8*y + m = 0 → 
    (x + 3)^2 + (y - 4)^2 = 5^2 ∨ (x + 3)^2 + (y - 4)^2 = 4^2) ∧
  (m = -11 ∨ m = 9) := by
  sorry

end tangent_circles_m_value_l573_57365


namespace coefficient_x4_is_correct_l573_57330

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ :=
  4 * (x^4 - 2*x^5) + 3 * (x^3 - 3*x^4 + 2*x^6) - (5*x^5 - 2*x^4)

/-- The coefficient of x^4 in the simplified expression -/
def coefficient_x4 : ℝ := -3

theorem coefficient_x4_is_correct :
  (deriv (deriv (deriv (deriv expression)))) 0 / 24 = coefficient_x4 := by
  sorry

end coefficient_x4_is_correct_l573_57330


namespace f_min_at_neg_four_l573_57309

/-- The quadratic function f(x) = x^2 + 8x + 7 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 7

/-- The theorem stating that f(x) has a minimum value of -9 at x = -4 -/
theorem f_min_at_neg_four :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min) ∧
  (∀ (x : ℝ), f x ≥ f (-4)) ∧
  f (-4) = -9 :=
sorry

end f_min_at_neg_four_l573_57309


namespace cylinder_radius_comparison_l573_57364

theorem cylinder_radius_comparison (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let original_volume := (6/7) * π * r^2 * h
  let new_height := (7/10) * h
  let new_volume := original_volume
  let new_radius := Real.sqrt ((5/3) * new_volume / (π * new_height))
  (new_radius - r) / r = 3/7 := by
sorry

end cylinder_radius_comparison_l573_57364


namespace sin_plus_tan_special_angle_l573_57347

/-- 
If the terminal side of angle α passes through point (4,-3), 
then sin α + tan α = -27/20 
-/
theorem sin_plus_tan_special_angle (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 4 ∧ r * Real.sin α = -3) → 
  Real.sin α + Real.tan α = -27/20 := by
  sorry

end sin_plus_tan_special_angle_l573_57347


namespace blue_notebook_cost_l573_57328

/-- Represents the cost of notebooks in dollars -/
def TotalCost : ℕ := 37

/-- Represents the total number of notebooks -/
def TotalNotebooks : ℕ := 12

/-- Represents the number of red notebooks -/
def RedNotebooks : ℕ := 3

/-- Represents the cost of each red notebook in dollars -/
def RedNotebookCost : ℕ := 4

/-- Represents the number of green notebooks -/
def GreenNotebooks : ℕ := 2

/-- Represents the cost of each green notebook in dollars -/
def GreenNotebookCost : ℕ := 2

/-- Calculates the number of blue notebooks -/
def BlueNotebooks : ℕ := TotalNotebooks - RedNotebooks - GreenNotebooks

/-- Theorem: The cost of each blue notebook is 3 dollars -/
theorem blue_notebook_cost : 
  (TotalCost - RedNotebooks * RedNotebookCost - GreenNotebooks * GreenNotebookCost) / BlueNotebooks = 3 := by
  sorry

end blue_notebook_cost_l573_57328


namespace simplify_expression_l573_57333

theorem simplify_expression : 0.3 * 0.8 + 0.1 * 0.5 = 0.29 := by
  sorry

end simplify_expression_l573_57333


namespace quadratic_real_roots_l573_57377

theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, (a * (1 + Complex.I)) * x^2 + (1 + a^2 * Complex.I) * x + (a^2 + Complex.I) = 0) ↔ a = -1 := by
  sorry

end quadratic_real_roots_l573_57377


namespace box_fits_blocks_l573_57339

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.height * d.width * d.length

/-- Calculates how many smaller objects can fit into a larger object -/
def fitCount (larger smaller : Dimensions) : ℕ :=
  (volume larger) / (volume smaller)

theorem box_fits_blocks :
  let box : Dimensions := { height := 8, width := 10, length := 12 }
  let block : Dimensions := { height := 3, width := 2, length := 4 }
  fitCount box block = 40 := by
  sorry

end box_fits_blocks_l573_57339


namespace quadratic_inequality_solution_range_l573_57315

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, 2 * x^2 - 9 * x + a < 0 ∧ (x^2 - 4 * x + 3 < 0 ∨ x^2 - 6 * x + 8 < 0)) ↔
  a < 4 := by
  sorry

end quadratic_inequality_solution_range_l573_57315


namespace nine_keys_required_l573_57354

/-- Represents the warehouse setup and retrieval task -/
structure WarehouseSetup where
  total_cabinets : ℕ
  boxes_per_cabinet : ℕ
  phones_per_box : ℕ
  phones_to_retrieve : ℕ

/-- Calculates the minimum number of keys required for the given warehouse setup -/
def min_keys_required (setup : WarehouseSetup) : ℕ :=
  let boxes_needed := (setup.phones_to_retrieve + setup.phones_per_box - 1) / setup.phones_per_box
  let cabinets_needed := (boxes_needed + setup.boxes_per_cabinet - 1) / setup.boxes_per_cabinet
  boxes_needed + cabinets_needed + 1

/-- Theorem stating that for the given warehouse setup, 9 keys are required -/
theorem nine_keys_required : 
  let setup : WarehouseSetup := {
    total_cabinets := 8,
    boxes_per_cabinet := 4,
    phones_per_box := 10,
    phones_to_retrieve := 52
  }
  min_keys_required setup = 9 := by
  sorry

end nine_keys_required_l573_57354


namespace factor_implies_k_value_l573_57321

theorem factor_implies_k_value (k : ℚ) :
  (∀ x : ℚ, (x + 5) ∣ (k * x^3 + 27 * x^2 - k * x + 55)) →
  k = 73 / 12 := by
  sorry

end factor_implies_k_value_l573_57321


namespace infinitely_many_pairs_exist_l573_57366

/-- A function that checks if all digits in the decimal representation of a natural number are greater than or equal to 7. -/
def allDigitsAtLeastSeven (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≥ 7

/-- A function that generates a pair of natural numbers based on the input n. -/
noncomputable def f (n : ℕ) : ℕ × ℕ :=
  let a := (887 : ℕ).pow n
  let b := 10^(3*n) - 123
  (a, b)

/-- Theorem stating that there exist infinitely many pairs of integers satisfying the given conditions. -/
theorem infinitely_many_pairs_exist :
  ∀ n : ℕ, 
    let (a, b) := f n
    allDigitsAtLeastSeven a ∧
    allDigitsAtLeastSeven b ∧
    allDigitsAtLeastSeven (a * b) :=
by
  sorry

end infinitely_many_pairs_exist_l573_57366


namespace odd_factors_of_420_l573_57368

-- Define 420 as a natural number
def n : ℕ := 420

-- Define a function to count odd factors
def count_odd_factors (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem odd_factors_of_420 : count_odd_factors n = 8 := by sorry

end odd_factors_of_420_l573_57368


namespace page_number_added_twice_l573_57310

theorem page_number_added_twice (n : ℕ) (h : n > 0) :
  (∃ (p : ℕ) (h_p : p ≤ n), n * (n + 1) / 2 + p = 1986) →
  (∃ (p : ℕ) (h_p : p ≤ n), n * (n + 1) / 2 + p = 1986 ∧ p = 33) :=
by sorry

end page_number_added_twice_l573_57310


namespace system_of_inequalities_solution_l573_57363

theorem system_of_inequalities_solution (x : ℝ) :
  (x - 1 ≤ x / 2 ∧ x + 2 > 3 * (x - 2)) ↔ x ≤ 2 := by
  sorry

end system_of_inequalities_solution_l573_57363


namespace sum_of_three_consecutive_cubes_divisible_by_nine_l573_57390

theorem sum_of_three_consecutive_cubes_divisible_by_nine (n : ℕ) :
  ∃ k : ℕ, n^3 + (n+1)^3 + (n+2)^3 = 9 * k := by
  sorry

end sum_of_three_consecutive_cubes_divisible_by_nine_l573_57390


namespace number_added_before_division_l573_57392

theorem number_added_before_division (x n : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) →
  (∃ m : ℤ, x + n = 41 * m + 22) →
  n = 5 := by
sorry

end number_added_before_division_l573_57392


namespace point_B_coordinates_l573_57399

def point_A : ℝ × ℝ := (-3, 2)

def move_right (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1 + units, p.2)

def move_down (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - units)

def point_B : ℝ × ℝ :=
  move_down (move_right point_A 1) 2

theorem point_B_coordinates :
  point_B = (-2, 0) := by sorry

end point_B_coordinates_l573_57399


namespace marcel_corn_count_l573_57369

/-- The number of ears of corn Marcel bought -/
def marcel_corn : ℕ := sorry

/-- The number of ears of corn Dale bought -/
def dale_corn : ℕ := sorry

/-- The number of potatoes Dale bought -/
def dale_potatoes : ℕ := 8

/-- The number of potatoes Marcel bought -/
def marcel_potatoes : ℕ := 4

/-- The total number of vegetables bought -/
def total_vegetables : ℕ := 27

theorem marcel_corn_count :
  (dale_corn = marcel_corn / 2) →
  (dale_potatoes = 8) →
  (marcel_potatoes = 4) →
  (marcel_corn + dale_corn + dale_potatoes + marcel_potatoes = total_vegetables) →
  marcel_corn = 10 := by sorry

end marcel_corn_count_l573_57369


namespace fifth_term_value_l573_57301

/-- A geometric sequence with a_3 and a_7 as roots of x^2 - 4x + 3 = 0 -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  root_condition : a 3 ^ 2 - 4 * a 3 + 3 = 0 ∧ a 7 ^ 2 - 4 * a 7 + 3 = 0

theorem fifth_term_value (seq : GeometricSequence) : seq.a 5 = Real.sqrt 3 := by
  sorry

end fifth_term_value_l573_57301


namespace area_of_square_e_l573_57348

/-- Given a rectangle composed of squares a, b, c, d, and e, prove the area of square e. -/
theorem area_of_square_e (a b c d e : ℝ) : 
  a + b + c = 30 →
  a + b = 22 →
  2 * c + e = 22 →
  e^2 = 36 := by
  sorry

end area_of_square_e_l573_57348


namespace f_max_value_l573_57396

/-- The function f(x) = -x^2 + 4x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 4*x + a

/-- Theorem: If f(x) has a minimum value of -2 on [0, 1], then its maximum value on [0, 1] is 1 -/
theorem f_max_value (a : ℝ) :
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f a x ≤ f a y) →
  (∃ x ∈ Set.Icc 0 1, f a x = -2) →
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f a y ≤ f a x) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = 1) :=
by sorry

#check f_max_value

end f_max_value_l573_57396


namespace smallest_four_digit_divisible_by_35_l573_57356

theorem smallest_four_digit_divisible_by_35 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 35 ∣ n → 1015 ≤ n :=
by sorry

end smallest_four_digit_divisible_by_35_l573_57356


namespace wall_painting_problem_l573_57351

theorem wall_painting_problem (heidi_rate peter_rate : ℚ) 
  (heidi_time peter_time painting_time : ℕ) :
  heidi_rate = 1 / 60 →
  peter_rate = 1 / 75 →
  heidi_time = 60 →
  peter_time = 75 →
  painting_time = 15 →
  (heidi_rate + peter_rate) * painting_time = 9 / 20 := by
  sorry

end wall_painting_problem_l573_57351


namespace tangent_sum_product_l573_57319

theorem tangent_sum_product (a b c : Real) (h1 : a = 117 * π / 180)
                                           (h2 : b = 118 * π / 180)
                                           (h3 : c = 125 * π / 180)
                                           (h4 : a + b + c = 2 * π) :
  Real.tan a * Real.tan b * Real.tan c = Real.tan a + Real.tan b + Real.tan c := by
  sorry

end tangent_sum_product_l573_57319


namespace projective_transformation_uniqueness_l573_57325

/-- A projective transformation on a line -/
structure ProjectiveTransformation (α : Type*) where
  transform : α → α

/-- The statement that two projective transformations are equal if they agree on three distinct points -/
theorem projective_transformation_uniqueness 
  {α : Type*} [LinearOrder α] 
  (P Q : ProjectiveTransformation α) 
  (A B C : α) 
  (hABC : A < B ∧ B < C) 
  (hP : P.transform A = Q.transform A ∧ 
        P.transform B = Q.transform B ∧ 
        P.transform C = Q.transform C) : 
  P = Q :=
sorry

end projective_transformation_uniqueness_l573_57325


namespace cubic_roots_problem_l573_57320

theorem cubic_roots_problem (p q r : ℂ) (u v w : ℂ) : 
  (p^3 + 5*p^2 + 6*p - 8 = 0) →
  (q^3 + 5*q^2 + 6*q - 8 = 0) →
  (r^3 + 5*r^2 + 6*r - 8 = 0) →
  ((p+q)^3 + u*(p+q)^2 + v*(p+q) + w = 0) →
  ((q+r)^3 + u*(q+r)^2 + v*(q+r) + w = 0) →
  ((r+p)^3 + u*(r+p)^2 + v*(r+p) + w = 0) →
  w = 38 := by
sorry

end cubic_roots_problem_l573_57320


namespace fatima_phone_probability_l573_57394

def first_three_digits : List ℕ := [295, 296, 299]
def base_last_four : List ℕ := [1, 6, 7]

def possible_numbers : ℕ := sorry

theorem fatima_phone_probability :
  (1 : ℚ) / possible_numbers = (1 : ℚ) / 72 := by sorry

end fatima_phone_probability_l573_57394


namespace intersection_implies_m_range_l573_57359

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2 + m * p.1 + 2}

def B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ 2}

-- State the theorem
theorem intersection_implies_m_range (m : ℝ) :
  (A m ∩ B).Nonempty → m ≤ -1 := by
  sorry


end intersection_implies_m_range_l573_57359


namespace units_digit_problem_l573_57337

theorem units_digit_problem : ∃ n : ℕ, 
  33 * 83^1001 * 7^1002 * 13^1003 ≡ 9 [ZMOD 10] ∧ n * 10 + 9 = 33 * 83^1001 * 7^1002 * 13^1003 :=
by sorry

end units_digit_problem_l573_57337
