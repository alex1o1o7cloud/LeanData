import Mathlib

namespace kaydence_sister_age_l1948_194820

/-- Given the ages of family members, prove the age of Kaydence's sister -/
theorem kaydence_sister_age 
  (total_age : ℕ) 
  (father_age : ℕ) 
  (mother_age : ℕ) 
  (brother_age : ℕ) 
  (kaydence_age : ℕ) 
  (h1 : total_age = 200)
  (h2 : father_age = 60)
  (h3 : mother_age = father_age - 2)
  (h4 : brother_age = father_age / 2)
  (h5 : kaydence_age = 12) :
  total_age - (father_age + mother_age + brother_age + kaydence_age) = 40 := by
  sorry


end kaydence_sister_age_l1948_194820


namespace ae_length_l1948_194803

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of side AB -/
  ab : ℝ
  /-- The length of side CD -/
  cd : ℝ
  /-- The property that AD = BC -/
  ad_eq_bc : True
  /-- Point E such that BC = EC -/
  bc_eq_ec : True
  /-- AE is perpendicular to EC -/
  ae_perp_ec : True

/-- The main theorem stating the length of AE in the specific isosceles trapezoid -/
theorem ae_length (t : IsoscelesTrapezoid) (h1 : t.ab = 3) (h2 : t.cd = 8) : 
  ∃ ae : ℝ, ae = 2 * Real.sqrt 6 := by
  sorry

end ae_length_l1948_194803


namespace negation_of_universal_statement_l1948_194842

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) := by sorry

end negation_of_universal_statement_l1948_194842


namespace subscription_ratio_l1948_194843

/-- Represents the number of magazine subscriptions sold to different people --/
structure Subscriptions where
  parents : ℕ
  grandfather : ℕ
  nextDoorNeighbor : ℕ
  otherNeighbor : ℕ

/-- Calculates the total earnings from selling subscriptions --/
def totalEarnings (s : Subscriptions) (pricePerSubscription : ℕ) : ℕ :=
  (s.parents + s.grandfather + s.nextDoorNeighbor + s.otherNeighbor) * pricePerSubscription

/-- Theorem stating the ratio of subscriptions sold to other neighbor vs next-door neighbor --/
theorem subscription_ratio (s : Subscriptions) (pricePerSubscription totalEarned : ℕ) :
  s.parents = 4 →
  s.grandfather = 1 →
  s.nextDoorNeighbor = 2 →
  pricePerSubscription = 5 →
  totalEarnings s pricePerSubscription = totalEarned →
  totalEarned = 55 →
  s.otherNeighbor = 2 * s.nextDoorNeighbor :=
by sorry


end subscription_ratio_l1948_194843


namespace smallest_positive_b_l1948_194861

/-- Definition of circle w1 -/
def w1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y - 23 = 0

/-- Definition of circle w2 -/
def w2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 8*y + 9 = 0

/-- Definition of a circle externally tangent to w2 and internally tangent to w1 -/
def tangent_circle (h k r : ℝ) : Prop :=
  (r + 2)^2 = (h + 3)^2 + (k + 4)^2 ∧ (6 - r)^2 = (h - 3)^2 + (k + 4)^2

/-- The line y = bx contains the center of the tangent circle -/
def center_on_line (h k b : ℝ) : Prop := k = b * h

/-- The main theorem -/
theorem smallest_positive_b :
  ∃ (b : ℝ), b > 0 ∧
  (∀ (h k r : ℝ), tangent_circle h k r → center_on_line h k b) ∧
  (∀ (b' : ℝ), 0 < b' ∧ b' < b →
    ¬(∀ (h k r : ℝ), tangent_circle h k r → center_on_line h k b')) ∧
  b^2 = 64/25 := by sorry

end smallest_positive_b_l1948_194861


namespace new_people_count_l1948_194811

/-- The number of people born in the country last year -/
def born : ℕ := 90171

/-- The number of people who immigrated to the country last year -/
def immigrated : ℕ := 16320

/-- The total number of new people who began living in the country last year -/
def total_new_people : ℕ := born + immigrated

/-- Theorem stating that the total number of new people is 106,491 -/
theorem new_people_count : total_new_people = 106491 := by
  sorry

end new_people_count_l1948_194811


namespace jeromes_contact_list_l1948_194850

theorem jeromes_contact_list (classmates : ℕ) (out_of_school_friends : ℕ) (family_members : ℕ) : 
  classmates = 20 →
  out_of_school_friends = classmates / 2 →
  family_members = 3 →
  classmates + out_of_school_friends + family_members = 33 := by
  sorry

end jeromes_contact_list_l1948_194850


namespace smartphone_cost_decrease_l1948_194862

def original_cost : ℝ := 600
def new_cost : ℝ := 450

theorem smartphone_cost_decrease :
  (original_cost - new_cost) / original_cost * 100 = 25 := by
  sorry

end smartphone_cost_decrease_l1948_194862


namespace sixth_rack_dvds_l1948_194828

def dvd_sequence : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * dvd_sequence n

theorem sixth_rack_dvds : dvd_sequence 5 = 64 := by
  sorry

end sixth_rack_dvds_l1948_194828


namespace largest_gold_coins_l1948_194883

theorem largest_gold_coins : ∃ n : ℕ, n < 150 ∧ ∃ k : ℕ, n = 13 * k + 3 ∧ 
  ∀ m : ℕ, m < 150 → (∃ j : ℕ, m = 13 * j + 3) → m ≤ n :=
by sorry

end largest_gold_coins_l1948_194883


namespace A_satisfies_conditions_l1948_194896

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 0}

-- Define set A
def A : Set ℝ := {1, 2}

-- Theorem statement
theorem A_satisfies_conditions : (A ∩ B = A) := by sorry

end A_satisfies_conditions_l1948_194896


namespace smallest_percentage_both_l1948_194844

/-- The smallest possible percentage of people eating both ice cream and chocolate in a town -/
theorem smallest_percentage_both (ice_cream_eaters chocolate_eaters : ℝ) 
  (h_ice_cream : ice_cream_eaters = 0.9)
  (h_chocolate : chocolate_eaters = 0.8) :
  ∃ (both : ℝ), both ≥ 0.7 ∧ 
    ∀ (x : ℝ), x ≥ 0 ∧ x ≤ 1 ∧ ice_cream_eaters + chocolate_eaters - x ≤ 1 → x ≥ both := by
  sorry


end smallest_percentage_both_l1948_194844


namespace sqrt_equation_solution_l1948_194867

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (1 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 68 / 3 := by
  sorry

end sqrt_equation_solution_l1948_194867


namespace square_root_divided_by_three_l1948_194801

theorem square_root_divided_by_three : Real.sqrt 81 / 3 = 3 := by
  sorry

end square_root_divided_by_three_l1948_194801


namespace negation_of_even_multiple_of_two_l1948_194812

theorem negation_of_even_multiple_of_two :
  ¬(∀ n : ℕ, Even n → (∃ k : ℕ, n = 2 * k)) ↔ 
  (∃ n : ℕ, Even n ∧ ¬(∃ k : ℕ, n = 2 * k)) :=
sorry

end negation_of_even_multiple_of_two_l1948_194812


namespace movie_ticket_change_l1948_194839

/-- Represents the movie ticket formats --/
inductive TicketFormat
  | Regular
  | ThreeD
  | IMAX

/-- Returns the price of a ticket based on its format --/
def ticketPrice (format : TicketFormat) : ℝ :=
  match format with
  | TicketFormat.Regular => 8
  | TicketFormat.ThreeD => 12
  | TicketFormat.IMAX => 15

/-- Calculates the discounted price of a ticket --/
def discountedPrice (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

theorem movie_ticket_change : 
  let format := TicketFormat.ThreeD
  let fullPrice := ticketPrice format
  let discountPercent := 0.25
  let discountedTicket := discountedPrice fullPrice discountPercent
  let totalCost := fullPrice + discountedTicket
  let moneyBrought := 25
  moneyBrought - totalCost = 4 := by sorry


end movie_ticket_change_l1948_194839


namespace ian_money_left_l1948_194815

/-- Calculates the amount of money left after spending half of earnings from surveys -/
def money_left (hours_worked : ℕ) (hourly_rate : ℚ) : ℚ :=
  (hours_worked : ℚ) * hourly_rate / 2

/-- Theorem: Given the conditions, prove that Ian has $72 left -/
theorem ian_money_left : money_left 8 18 = 72 := by
  sorry

end ian_money_left_l1948_194815


namespace right_triangle_sides_l1948_194887

theorem right_triangle_sides (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 30 → 
  x^2 + y^2 + z^2 = 338 → 
  x^2 + y^2 = z^2 →
  ((x = 5 ∧ y = 12 ∧ z = 13) ∨ (x = 12 ∧ y = 5 ∧ z = 13)) :=
by sorry

end right_triangle_sides_l1948_194887


namespace isabels_pop_albums_l1948_194821

theorem isabels_pop_albums (total_songs : ℕ) (country_albums : ℕ) (songs_per_album : ℕ) 
  (h1 : total_songs = 72)
  (h2 : country_albums = 4)
  (h3 : songs_per_album = 8) :
  total_songs - country_albums * songs_per_album = 5 * songs_per_album :=
by sorry

end isabels_pop_albums_l1948_194821


namespace joined_right_triangles_fourth_square_l1948_194800

theorem joined_right_triangles_fourth_square 
  (PQ QR RS : ℝ) 
  (h1 : PQ^2 = 25) 
  (h2 : QR^2 = 49) 
  (h3 : RS^2 = 64) 
  (h4 : PQ > 0 ∧ QR > 0 ∧ RS > 0) : 
  (PQ^2 + QR^2) + RS^2 = 138 := by
  sorry

end joined_right_triangles_fourth_square_l1948_194800


namespace bill_face_value_l1948_194888

/-- Calculates the face value of a bill given the true discount, time period, and annual interest rate -/
def calculate_face_value (true_discount : ℚ) (time_months : ℚ) (annual_rate : ℚ) : ℚ :=
  let time_years := time_months / 12
  let rate_decimal := annual_rate / 100
  (true_discount * (100 + (rate_decimal * time_years * 100))) / (rate_decimal * time_years * 100)

/-- Theorem stating that given the specific conditions, the face value of the bill is 1764 -/
theorem bill_face_value :
  let true_discount : ℚ := 189
  let time_months : ℚ := 9
  let annual_rate : ℚ := 16
  calculate_face_value true_discount time_months annual_rate = 1764 := by
  sorry

end bill_face_value_l1948_194888


namespace expression_simplification_l1948_194805

def x : ℚ := -2
def y : ℚ := 1/2

theorem expression_simplification :
  (x + 4*y) * (x - 4*y) + (x - 4*y)^2 - (4*x^2 - x*y) = -1 :=
by sorry

end expression_simplification_l1948_194805


namespace exam_type_a_time_l1948_194889

/-- Represents the examination setup -/
structure Exam where
  totalTime : ℕ  -- Total time in minutes
  totalQuestions : ℕ
  typeAQuestions : ℕ
  typeAMultiplier : ℕ  -- How many times longer type A questions take compared to type B

/-- Calculates the time spent on type A problems -/
def timeOnTypeA (e : Exam) : ℚ :=
  let totalTypeB := e.totalQuestions - e.typeAQuestions
  let x := e.totalTime / (e.typeAQuestions * e.typeAMultiplier + totalTypeB)
  e.typeAQuestions * e.typeAMultiplier * x

/-- Theorem stating that for the given exam setup, 40 minutes should be spent on type A problems -/
theorem exam_type_a_time :
  let e : Exam := {
    totalTime := 180,  -- 3 hours * 60 minutes
    totalQuestions := 200,
    typeAQuestions := 25,
    typeAMultiplier := 2
  }
  timeOnTypeA e = 40 := by sorry


end exam_type_a_time_l1948_194889


namespace no_negative_exponents_l1948_194860

theorem no_negative_exponents (a b c d : ℤ) (h : (4:ℝ)^a + (4:ℝ)^b = (8:ℝ)^c + (27:ℝ)^d) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 := by
sorry

end no_negative_exponents_l1948_194860


namespace min_value_sum_reciprocals_l1948_194802

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  (1 / (a + 2*b)) + (1 / (b + 2*c)) + (1 / (c + 2*a)) ≥ 3 := by
  sorry

end min_value_sum_reciprocals_l1948_194802


namespace plastic_for_rulers_l1948_194884

theorem plastic_for_rulers (plastic_per_ruler : ℕ) (rulers_made : ℕ) : 
  plastic_per_ruler = 8 → rulers_made = 103 → plastic_per_ruler * rulers_made = 824 := by
  sorry

end plastic_for_rulers_l1948_194884


namespace new_student_weight_l1948_194848

theorem new_student_weight (n : ℕ) (original_avg replaced_weight new_avg : ℝ) :
  n = 5 →
  replaced_weight = 72 →
  new_avg = original_avg - 12 →
  n * original_avg - replaced_weight = n * new_avg - (n * original_avg - n * new_avg) →
  n * original_avg - replaced_weight = 12 :=
by sorry

end new_student_weight_l1948_194848


namespace binomial_coefficient_30_3_l1948_194851

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_coefficient_30_3_l1948_194851


namespace fraction_calculation_l1948_194806

theorem fraction_calculation (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  ((x + 1) / (y - 1)) / ((y + 2) / (x - 2)) = 5 / 14 := by
  sorry

end fraction_calculation_l1948_194806


namespace intersection_complement_equality_l1948_194863

def R : Set ℝ := Set.univ

def A : Set ℝ := {1, 2, 3, 4, 5}

def B : Set ℝ := {x : ℝ | x * (4 - x) < 0}

theorem intersection_complement_equality :
  A ∩ (Set.compl B) = {1, 2, 3, 4} := by
  sorry

end intersection_complement_equality_l1948_194863


namespace max_value_theorem_l1948_194829

theorem max_value_theorem (x : ℝ) :
  x^4 / (x^8 + 2*x^6 + 4*x^4 + 4*x^2 + 16) ≤ 1/16 ∧
  ∃ y : ℝ, y^4 / (y^8 + 2*y^6 + 4*y^4 + 4*y^2 + 16) = 1/16 :=
by sorry

end max_value_theorem_l1948_194829


namespace at_home_workforce_trend_l1948_194819

/-- Represents the percentage of working adults in Parkertown working at home for a given year -/
def AtHomeWorkforce : ℕ → ℚ
  | 1990 => 12/100
  | 1995 => 15/100
  | 2000 => 14/100
  | 2005 => 28/100
  | _ => 0

/-- The trend of the at-home workforce in Parkertown from 1990 to 2005 -/
theorem at_home_workforce_trend :
  AtHomeWorkforce 1995 > AtHomeWorkforce 1990 ∧
  AtHomeWorkforce 2000 < AtHomeWorkforce 1995 ∧
  AtHomeWorkforce 2005 > AtHomeWorkforce 2000 ∧
  (AtHomeWorkforce 2005 - AtHomeWorkforce 2000) > (AtHomeWorkforce 1995 - AtHomeWorkforce 1990) :=
by sorry

end at_home_workforce_trend_l1948_194819


namespace pool_water_proof_l1948_194818

def initial_volume : ℝ := 300
def evaporation_rate_1 : ℝ := 1
def evaporation_rate_2 : ℝ := 2
def days_1 : ℝ := 15
def days_2 : ℝ := 15
def total_days : ℝ := days_1 + days_2

def remaining_volume : ℝ :=
  initial_volume - (evaporation_rate_1 * days_1 + evaporation_rate_2 * days_2)

theorem pool_water_proof :
  remaining_volume = 255 := by sorry

end pool_water_proof_l1948_194818


namespace solution_count_l1948_194885

open Complex

/-- The number of complex solutions to e^z = (z + i)/(z - i) with |z| < 20 -/
def num_solutions : ℕ := 14

/-- The equation e^z = (z + i)/(z - i) -/
def equation (z : ℂ) : Prop := exp z = (z + I) / (z - I)

/-- The condition |z| < 20 -/
def magnitude_condition (z : ℂ) : Prop := abs z < 20

theorem solution_count :
  (∃ (S : Finset ℂ), S.card = num_solutions ∧
    (∀ z ∈ S, equation z ∧ magnitude_condition z) ∧
    (∀ z : ℂ, equation z ∧ magnitude_condition z → z ∈ S)) := by
  sorry

end solution_count_l1948_194885


namespace quadratic_polynomial_identification_l1948_194886

-- Define the polynomials
def p1 (x : ℝ) : ℝ := 3^2 * x + 1
def p2 (x : ℝ) : ℝ := 3 * x^2
def p3 (x y : ℝ) : ℝ := 3 * x * y + 1
def p4 (x : ℝ) : ℝ := 3 * x - 5^2

-- Define what it means for a polynomial to be quadratic
def is_quadratic (p : ℝ → ℝ) : Prop := ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, p x = a * x^2 + b * x + c

-- Define what it means for a two-variable polynomial to be quadratic
def is_quadratic_two_var (p : ℝ → ℝ → ℝ) : Prop := 
  ∃ a b c d e f : ℝ, (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ 
  ∀ x y, p x y = a * x^2 + b * y^2 + c * x * y + d * x + e * y + f

-- State the theorem
theorem quadratic_polynomial_identification :
  ¬ is_quadratic p1 ∧
  is_quadratic p2 ∧
  is_quadratic_two_var p3 ∧
  ¬ is_quadratic p4 :=
sorry

end quadratic_polynomial_identification_l1948_194886


namespace science_project_cans_l1948_194804

def empty_cans_problem (alyssa_cans abigail_cans more_needed : ℕ) : Prop :=
  alyssa_cans + abigail_cans + more_needed = 100

theorem science_project_cans : empty_cans_problem 30 43 27 := by
  sorry

end science_project_cans_l1948_194804


namespace car_distance_l1948_194857

theorem car_distance (efficiency : ℝ) (gas : ℝ) (distance : ℝ) :
  efficiency = 20 →
  gas = 5 →
  distance = efficiency * gas →
  distance = 100 := by sorry

end car_distance_l1948_194857


namespace vector_expression_l1948_194840

-- Define vectors a, b, and c
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, -2)

-- Theorem statement
theorem vector_expression :
  c = (-3/2 : ℝ) • a + (1/2 : ℝ) • b := by sorry

end vector_expression_l1948_194840


namespace triangle_abc_properties_l1948_194880

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  cos C + (cos A - Real.sqrt 3 * sin A) * cos B = 0 ∧
  b = Real.sqrt 3 ∧
  c = 1 →
  B = π / 3 ∧
  (1 / 2) * a * c * sin B = Real.sqrt 3 / 2 :=
by sorry

end triangle_abc_properties_l1948_194880


namespace highest_divisible_digit_l1948_194813

theorem highest_divisible_digit : ∃ (a : ℕ), a ≤ 9 ∧ 
  (365 * 10 + a) * 100 + 16 % 8 = 0 ∧ 
  ∀ (b : ℕ), b ≤ 9 → b > a → (365 * 10 + b) * 100 + 16 % 8 ≠ 0 :=
by sorry

end highest_divisible_digit_l1948_194813


namespace last_four_digits_l1948_194826

theorem last_four_digits : (301 * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 4 % 10000 = 5856 := by
  sorry

end last_four_digits_l1948_194826


namespace negation_of_existence_negation_of_proposition_l1948_194858

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end negation_of_existence_negation_of_proposition_l1948_194858


namespace profit_in_toys_l1948_194823

/-- 
Given:
- A man sold 18 toys for Rs. 18900
- The cost price of a toy is Rs. 900
Prove that the number of toys' cost price gained as profit is 3
-/
theorem profit_in_toys (total_toys : ℕ) (selling_price : ℕ) (cost_per_toy : ℕ) :
  total_toys = 18 →
  selling_price = 18900 →
  cost_per_toy = 900 →
  (selling_price - total_toys * cost_per_toy) / cost_per_toy = 3 :=
by sorry

end profit_in_toys_l1948_194823


namespace quadratic_inequality_theorem_1_quadratic_inequality_theorem_2_quadratic_inequality_theorem_3_l1948_194846

-- Define the quadratic inequality
def quadratic_inequality (k x : ℝ) : Prop := k * x^2 - 2 * x + 6 * k < 0

-- Define the solution sets
def solution_set_1 (x : ℝ) : Prop := x < -3 ∨ x > -2
def solution_set_2 (k x : ℝ) : Prop := x ≠ 1 / k
def solution_set_3 : Set ℝ := ∅

-- Theorem statements
theorem quadratic_inequality_theorem_1 (k : ℝ) :
  k ≠ 0 →
  (∀ x, quadratic_inequality k x ↔ solution_set_1 x) →
  k = -2/5 :=
sorry

theorem quadratic_inequality_theorem_2 (k : ℝ) :
  k ≠ 0 →
  (∀ x, quadratic_inequality k x ↔ solution_set_2 k x) →
  k = -Real.sqrt 6 / 6 :=
sorry

theorem quadratic_inequality_theorem_3 (k : ℝ) :
  k ≠ 0 →
  (∀ x, ¬quadratic_inequality k x) →
  k ≥ Real.sqrt 6 / 6 :=
sorry

end quadratic_inequality_theorem_1_quadratic_inequality_theorem_2_quadratic_inequality_theorem_3_l1948_194846


namespace toms_seashells_l1948_194882

theorem toms_seashells (sally_shells : ℕ) (jessica_shells : ℕ) (total_shells : ℕ) 
  (h1 : sally_shells = 9)
  (h2 : jessica_shells = 5)
  (h3 : total_shells = 21) :
  total_shells - (sally_shells + jessica_shells) = 7 := by
  sorry

end toms_seashells_l1948_194882


namespace cubic_polynomial_third_root_l1948_194808

theorem cubic_polynomial_third_root 
  (a b : ℚ) 
  (h1 : a * (-1)^3 + (a + 3*b) * (-1)^2 + (b - 2*a) * (-1) + (10 - a) = 0)
  (h2 : a * 4^3 + (a + 3*b) * 4^2 + (b - 2*a) * 4 + (10 - a) = 0) :
  ∃ (r : ℚ), a * r^3 + (a + 3*b) * r^2 + (b - 2*a) * r + (10 - a) = 0 ∧ 
              r = -67/88 :=
by sorry

end cubic_polynomial_third_root_l1948_194808


namespace abc_product_is_one_l1948_194893

theorem abc_product_is_one (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : a^2 + 1/b^2 = b^2 + 1/c^2 ∧ b^2 + 1/c^2 = c^2 + 1/a^2) :
  |a * b * c| = 1 := by
  sorry

end abc_product_is_one_l1948_194893


namespace unit_conversion_l1948_194891

/-- Conversion rates --/
def hectare_to_square_meter : ℝ := 10000
def meter_to_centimeter : ℝ := 100
def square_kilometer_to_hectare : ℝ := 100
def hour_to_minute : ℝ := 60
def kilogram_to_gram : ℝ := 1000

/-- Unit conversion theorem --/
theorem unit_conversion :
  (360 / hectare_to_square_meter = 0.036) ∧
  (504 / meter_to_centimeter = 5.04) ∧
  (0.06 * square_kilometer_to_hectare = 6) ∧
  (15 / hour_to_minute = 0.25) ∧
  (5.45 = 5 + 450 / kilogram_to_gram) :=
by sorry

end unit_conversion_l1948_194891


namespace binary_101101_eq_45_l1948_194832

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101101₂ -/
def binary_101101 : List Bool := [true, false, true, true, false, true]

/-- Theorem stating that the decimal equivalent of 101101₂ is 45 -/
theorem binary_101101_eq_45 : binary_to_decimal binary_101101 = 45 := by
  sorry

#eval binary_to_decimal binary_101101

end binary_101101_eq_45_l1948_194832


namespace f_derivative_l1948_194873

noncomputable def f (x : ℝ) : ℝ := x^3 * (Real.cos x)^5 * (Real.arctan x)^7 * (Real.log x)^4 * (Real.arcsin x)^10

theorem f_derivative (x : ℝ) (hx : x ≠ 0 ∧ x^2 < 1) : 
  deriv f x = f x * (3/x - 5*Real.tan x + 7/(Real.arctan x * (1 + x^2)) + 4/(x * Real.log x) + 10/(Real.arcsin x * Real.sqrt (1 - x^2))) := by
  sorry

end f_derivative_l1948_194873


namespace line_points_k_value_l1948_194849

/-- 
Given two points (m, n) and (m + 5, n + k) on a line with equation x = 2y + 5,
prove that k = 2.5
-/
theorem line_points_k_value 
  (m n k : ℝ) 
  (point1_on_line : m = 2 * n + 5)
  (point2_on_line : m + 5 = 2 * (n + k) + 5) :
  k = 2.5 := by
sorry

end line_points_k_value_l1948_194849


namespace complex_distance_theorem_l1948_194856

theorem complex_distance_theorem (z z₁ z₂ : ℂ) :
  z₁ ≠ z₂ →
  z₁^2 = -2 - 2 * Complex.I * Real.sqrt 3 →
  z₂^2 = -2 - 2 * Complex.I * Real.sqrt 3 →
  Complex.abs (z - z₁) = 4 →
  Complex.abs (z - z₂) = 4 →
  Complex.abs z = 2 * Real.sqrt 3 := by
  sorry

end complex_distance_theorem_l1948_194856


namespace semicircle_perimeter_semicircle_area_l1948_194877

-- Define constants
def rectangle_length : ℝ := 10
def rectangle_width : ℝ := 8
def π : ℝ := 3.14

-- Define the semicircle
def semicircle_diameter : ℝ := rectangle_length

-- Theorem for the perimeter of the semicircle
theorem semicircle_perimeter :
  π * semicircle_diameter / 2 + semicircle_diameter = 25.7 :=
sorry

-- Theorem for the area of the semicircle
theorem semicircle_area :
  π * (semicircle_diameter / 2)^2 / 2 = 39.25 :=
sorry

end semicircle_perimeter_semicircle_area_l1948_194877


namespace daily_savings_amount_l1948_194809

def total_savings : ℝ := 8760
def days_in_year : ℕ := 365

theorem daily_savings_amount :
  total_savings / days_in_year = 24 := by
  sorry

end daily_savings_amount_l1948_194809


namespace right_triangle_sin_a_l1948_194872

theorem right_triangle_sin_a (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : C = Real.pi / 2) (h3 : Real.cos B = 1 / 2) : Real.sin A = 1 / 2 := by
  sorry

end right_triangle_sin_a_l1948_194872


namespace fourth_term_value_l1948_194825

def S (n : ℕ) : ℤ := n^2 - 3*n

def a (n : ℕ) : ℤ := S n - S (n-1)

theorem fourth_term_value : a 4 = 4 := by sorry

end fourth_term_value_l1948_194825


namespace directrix_of_specific_parabola_l1948_194874

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Defines the directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop := sorry

/-- The specific parabola with equation x^2 = 8y -/
def specific_parabola : Parabola :=
  { equation := fun x y => x^2 = 8*y }

/-- Theorem stating that the directrix of the specific parabola is y = -2 -/
theorem directrix_of_specific_parabola :
  directrix specific_parabola = fun y => y = -2 := by sorry

end directrix_of_specific_parabola_l1948_194874


namespace max_balls_in_cube_l1948_194894

theorem max_balls_in_cube (cube_side : ℝ) (ball_radius : ℝ) : 
  cube_side = 8 → 
  ball_radius = 1.5 → 
  ⌊(cube_side^3) / ((4/3) * π * ball_radius^3)⌋ = 36 := by
  sorry

end max_balls_in_cube_l1948_194894


namespace pet_shelter_problem_l1948_194853

theorem pet_shelter_problem (total dogs_watermelon dogs_salmon dogs_chicken 
  dogs_watermelon_salmon dogs_salmon_chicken dogs_watermelon_chicken dogs_all_three : ℕ) 
  (h_total : total = 150)
  (h_watermelon : dogs_watermelon = 30)
  (h_salmon : dogs_salmon = 70)
  (h_chicken : dogs_chicken = 15)
  (h_watermelon_salmon : dogs_watermelon_salmon = 10)
  (h_salmon_chicken : dogs_salmon_chicken = 7)
  (h_watermelon_chicken : dogs_watermelon_chicken = 5)
  (h_all_three : dogs_all_three = 3) :
  total - (dogs_watermelon + dogs_salmon + dogs_chicken 
    - dogs_watermelon_salmon - dogs_salmon_chicken - dogs_watermelon_chicken 
    + dogs_all_three) = 54 := by
  sorry


end pet_shelter_problem_l1948_194853


namespace sequence_sum_of_squares_l1948_194876

theorem sequence_sum_of_squares (n : ℕ) :
  ∃ y : ℤ, (1 / 4 : ℝ) * ((2 + Real.sqrt 3)^(2*n - 1) + (2 - Real.sqrt 3)^(2*n - 1)) = y^2 + (y + 1)^2 := by
  sorry

end sequence_sum_of_squares_l1948_194876


namespace beetle_speed_l1948_194864

/-- Calculates the speed of a beetle given the ant's distance and the beetle's relative speed -/
theorem beetle_speed (ant_distance : Real) (time_minutes : Real) (beetle_relative_speed : Real) :
  let beetle_distance := ant_distance * (1 - beetle_relative_speed)
  let time_hours := time_minutes / 60
  let speed_km_h := (beetle_distance / 1000) / time_hours
  speed_km_h = 2.55 :=
by
  sorry

#check beetle_speed 600 12 0.15

end beetle_speed_l1948_194864


namespace combined_return_percentage_l1948_194892

theorem combined_return_percentage (investment1 investment2 return1 return2 : ℝ) :
  investment1 = 500 →
  investment2 = 1500 →
  return1 = 0.07 →
  return2 = 0.19 →
  (investment1 * return1 + investment2 * return2) / (investment1 + investment2) = 0.16 := by
  sorry

end combined_return_percentage_l1948_194892


namespace unique_root_implies_k_range_l1948_194814

/-- A function f(x) with parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + (1-k)*x - k

/-- Theorem: If f(x) has exactly one root in (2,3), then k is in (2,3) -/
theorem unique_root_implies_k_range (k : ℝ) :
  (∃! x, x ∈ (Set.Ioo 2 3) ∧ f k x = 0) → k ∈ Set.Ioo 2 3 :=
by sorry

end unique_root_implies_k_range_l1948_194814


namespace least_number_with_remainder_four_l1948_194854

theorem least_number_with_remainder_four (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → 
    (m % 6 ≠ 4 ∨ m % 9 ≠ 4 ∨ m % 12 ≠ 4 ∨ m % 18 ≠ 4)) ∧
  n % 6 = 4 ∧ n % 9 = 4 ∧ n % 12 = 4 ∧ n % 18 = 4 → 
  n = 40 := by
sorry

end least_number_with_remainder_four_l1948_194854


namespace game_correct_answers_l1948_194834

theorem game_correct_answers : 
  ∀ (total_questions : ℕ) 
    (correct_reward incorrect_penalty : ℕ) 
    (correct_answers : ℕ),
  total_questions = 50 →
  correct_reward = 7 →
  incorrect_penalty = 3 →
  correct_answers * correct_reward = 
    (total_questions - correct_answers) * incorrect_penalty →
  correct_answers = 15 := by
sorry

end game_correct_answers_l1948_194834


namespace granola_bar_distribution_l1948_194855

/-- Given that Monroe made x granola bars, she and her husband ate 2/3 of them,
    and the rest were divided equally among y children, with each child receiving z granola bars,
    prove that z = x / (3 * y) -/
theorem granola_bar_distribution (x y z : ℚ) (hx : x > 0) (hy : y > 0) : 
  (2 / 3 * x + y * z = x) → z = x / (3 * y) := by
  sorry

end granola_bar_distribution_l1948_194855


namespace polygon_with_900_degree_sum_is_heptagon_l1948_194898

theorem polygon_with_900_degree_sum_is_heptagon :
  ∀ n : ℕ, 
    n ≥ 3 →
    (n - 2) * 180 = 900 →
    n = 7 :=
by
  sorry

end polygon_with_900_degree_sum_is_heptagon_l1948_194898


namespace integral_comparison_l1948_194831

theorem integral_comparison : ∃ (a b c : ℝ),
  (a = ∫ x in (0:ℝ)..1, x) ∧
  (b = ∫ x in (0:ℝ)..1, x^2) ∧
  (c = ∫ x in (0:ℝ)..1, Real.sqrt x) ∧
  (b < a ∧ a < c) :=
by sorry

end integral_comparison_l1948_194831


namespace extreme_value_and_minimum_a_l1948_194865

noncomputable def f (a : ℤ) (x : ℝ) : ℝ := (1/2) * a * x^2 - Real.log x

theorem extreme_value_and_minimum_a :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 1 x ≤ f 1 y) ∧
  f 1 1 = (1/2) ∧
  (∀ (a : ℤ), (∀ (x : ℝ), x > 0 → f a x ≥ (1 - a) * x + 1) → a ≥ 2) ∧
  (∀ (x : ℝ), x > 0 → f 2 x ≥ (1 - 2) * x + 1) := by
  sorry

end extreme_value_and_minimum_a_l1948_194865


namespace tims_drive_distance_l1948_194816

/-- Represents the scenario of Tim's drive to work -/
def TimsDrive (totalDistance : ℝ) : Prop :=
  let normalTime : ℝ := 120
  let newTime : ℝ := 165
  let speedReduction : ℝ := 30 / 60 -- 30 mph converted to miles per minute
  let normalSpeed : ℝ := totalDistance / normalTime
  let newSpeed : ℝ := normalSpeed - speedReduction
  let halfDistance : ℝ := totalDistance / 2
  normalTime / 2 + halfDistance / newSpeed = newTime

/-- Theorem stating that the total distance of Tim's drive is 140 miles -/
theorem tims_drive_distance : ∃ (d : ℝ), TimsDrive d ∧ d = 140 :=
sorry

end tims_drive_distance_l1948_194816


namespace football_club_balance_l1948_194833

def initial_balance : ℝ := 100
def players_sold : ℕ := 2
def selling_price : ℝ := 10
def players_bought : ℕ := 4
def buying_price : ℝ := 15

theorem football_club_balance :
  initial_balance + players_sold * selling_price - players_bought * buying_price = 60 :=
by sorry

end football_club_balance_l1948_194833


namespace cube_face_sum_theorem_l1948_194852

/-- Represents a cube with numbers on its faces -/
structure NumberedCube where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- Calculates the sum of vertex products for a NumberedCube -/
def vertexProductSum (cube : NumberedCube) : ℕ :=
  cube.a * cube.b * cube.c +
  cube.a * cube.e * cube.c +
  cube.a * cube.b * cube.f +
  cube.a * cube.e * cube.f +
  cube.d * cube.b * cube.c +
  cube.d * cube.e * cube.c +
  cube.d * cube.b * cube.f +
  cube.d * cube.e * cube.f

/-- Calculates the sum of face numbers for a NumberedCube -/
def faceSum (cube : NumberedCube) : ℕ :=
  cube.a + cube.b + cube.c + cube.d + cube.e + cube.f

/-- Theorem: If the sum of vertex products is 357, then the sum of face numbers is 27 -/
theorem cube_face_sum_theorem (cube : NumberedCube) :
  vertexProductSum cube = 357 → faceSum cube = 27 := by
  sorry

end cube_face_sum_theorem_l1948_194852


namespace race_probability_l1948_194895

theorem race_probability (total_cars : ℕ) (prob_Y prob_Z prob_XYZ : ℝ) : 
  total_cars = 15 →
  prob_Y = 1/8 →
  prob_Z = 1/12 →
  prob_XYZ = 0.4583333333333333 →
  ∃ (prob_X : ℝ), prob_X + prob_Y + prob_Z = prob_XYZ ∧ prob_X = 0.25 := by
  sorry

end race_probability_l1948_194895


namespace coin_distribution_theorem_l1948_194868

/-- Represents the number of rounds in the coin distribution -/
def x : ℕ := sorry

/-- Pete's coins after distribution -/
def pete_coins (x : ℕ) : ℕ := x * (x + 1) / 2

/-- Paul's coins after distribution -/
def paul_coins (x : ℕ) : ℕ := x

/-- The condition that Pete has three times as many coins as Paul -/
axiom pete_triple_paul : pete_coins x = 3 * paul_coins x

/-- The total number of coins -/
def total_coins (x : ℕ) : ℕ := pete_coins x + paul_coins x

theorem coin_distribution_theorem : total_coins x = 20 := by
  sorry

end coin_distribution_theorem_l1948_194868


namespace tom_payment_l1948_194807

/-- The amount Tom paid to the shopkeeper for apples and mangoes -/
def total_amount (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Theorem stating that Tom paid 1145 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 65 = 1145 := by
  sorry

#eval total_amount 8 70 9 65

end tom_payment_l1948_194807


namespace investment_amount_from_interest_difference_l1948_194847

/-- Proves that given two equal investments with specific interest rates and time period, 
    the investment amount can be determined from the interest difference. -/
theorem investment_amount_from_interest_difference 
  (P : ℝ) -- The amount invested (same for both investments)
  (r1 : ℝ) -- Interest rate for first investment
  (r2 : ℝ) -- Interest rate for second investment
  (t : ℝ) -- Time period in years
  (diff : ℝ) -- Difference in interest earned
  (h1 : r1 = 0.04) -- First interest rate is 4%
  (h2 : r2 = 0.045) -- Second interest rate is 4.5%
  (h3 : t = 7) -- Time period is 7 years
  (h4 : P * r2 * t - P * r1 * t = diff) -- Difference in interest equation
  (h5 : diff = 31.5) -- Interest difference is $31.50
  : P = 900 := by
  sorry

#check investment_amount_from_interest_difference

end investment_amount_from_interest_difference_l1948_194847


namespace abc_relationship_l1948_194899

theorem abc_relationship :
  let a : Real := Real.rpow 0.3 0.2
  let b : Real := Real.rpow 0.2 0.3
  let c : Real := Real.rpow 0.3 0.3
  a > c ∧ c > b := by
  sorry

end abc_relationship_l1948_194899


namespace divisibility_constraint_l1948_194845

theorem divisibility_constraint (m n : ℕ) : 
  m ≥ 1 → n ≥ 1 → 
  (m * n ∣ 3^m + 1) → 
  (m * n ∣ 3^n + 1) → 
  ((m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1)) :=
by sorry

end divisibility_constraint_l1948_194845


namespace field_trip_girls_fraction_l1948_194836

theorem field_trip_girls_fraction (total_boys : ℕ) (total_girls : ℕ) 
  (boys_fraction : ℚ) (girls_fraction : ℚ) :
  total_boys = 200 →
  total_girls = 150 →
  boys_fraction = 3 / 5 →
  girls_fraction = 4 / 5 →
  let boys_on_trip := (boys_fraction * total_boys : ℚ)
  let girls_on_trip := (girls_fraction * total_girls : ℚ)
  let total_on_trip := boys_on_trip + girls_on_trip
  (girls_on_trip / total_on_trip : ℚ) = 1 / 2 := by
sorry

end field_trip_girls_fraction_l1948_194836


namespace real_part_of_fraction_l1948_194817

theorem real_part_of_fraction (z : ℂ) (h1 : z ≠ 0) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 2) : 
  (2 / (1 - z)).re = 2/5 := by
sorry

end real_part_of_fraction_l1948_194817


namespace bret_frog_count_l1948_194824

theorem bret_frog_count :
  ∀ (alster_frogs quinn_frogs bret_frogs : ℕ),
    alster_frogs = 2 →
    quinn_frogs = 2 * alster_frogs →
    bret_frogs = 3 * quinn_frogs →
    bret_frogs = 12 := by
  sorry

end bret_frog_count_l1948_194824


namespace hyperbola_sum_l1948_194866

/-- Proves that for a hyperbola with given properties, h + k + a + b = 11 -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 3 ∧ 
  k = -1 ∧ 
  c = Real.sqrt 41 ∧ 
  a = 4 ∧ 
  c^2 = a^2 + b^2 →
  h + k + a + b = 11 := by sorry

end hyperbola_sum_l1948_194866


namespace boys_at_reunion_l1948_194871

/-- The number of handshakes when each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: There are 12 boys at the reunion given the conditions. -/
theorem boys_at_reunion : ∃ (n : ℕ), n > 0 ∧ handshakes n = 66 ∧ n = 12 := by
  sorry

end boys_at_reunion_l1948_194871


namespace optimal_inequality_values_l1948_194897

theorem optimal_inequality_values (x : ℝ) (hx : x ∈ Set.Icc 0 1) :
  let a : ℝ := 2
  let b : ℝ := 1/4
  (∀ (a' : ℝ) (b' : ℝ), a' > 0 → b' > 0 →
    (∀ (y : ℝ), y ∈ Set.Icc 0 1 → Real.sqrt (1 - y) + Real.sqrt (1 + y) ≤ 2 - b' * y ^ a') →
    a' ≥ a) ∧
  (∀ (b' : ℝ), b' > b →
    ∃ (y : ℝ), y ∈ Set.Icc 0 1 ∧ Real.sqrt (1 - y) + Real.sqrt (1 + y) > 2 - b' * y ^ a) ∧
  Real.sqrt (1 - x) + Real.sqrt (1 + x) ≤ 2 - b * x ^ a :=
by sorry

end optimal_inequality_values_l1948_194897


namespace f_negative_nine_l1948_194878

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_negative_nine (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_period : has_period f 4) 
  (h_f_one : f 1 = 1) : 
  f (-9) = 1 := by
  sorry

end f_negative_nine_l1948_194878


namespace sodium_hydride_requirement_l1948_194869

-- Define the chemical reaction
structure ChemicalReaction where
  naH : ℚ  -- moles of Sodium hydride
  h2o : ℚ  -- moles of Water
  naOH : ℚ -- moles of Sodium hydroxide
  h2 : ℚ   -- moles of Hydrogen

-- Define the balanced equation
def balancedReaction (r : ChemicalReaction) : Prop :=
  r.naH = r.h2o ∧ r.naH = r.naOH ∧ r.naH = r.h2

-- Theorem statement
theorem sodium_hydride_requirement 
  (r : ChemicalReaction) 
  (h1 : r.naOH = 2) 
  (h2 : r.h2 = 2) 
  (h3 : r.h2o = 2) 
  (h4 : balancedReaction r) : 
  r.naH = 2 := by
  sorry

end sodium_hydride_requirement_l1948_194869


namespace binomial_variance_l1948_194881

/-- A binomial distribution with parameter p -/
structure BinomialDistribution (p : ℝ) where
  (h1 : 0 < p)
  (h2 : p < 1)

/-- The variance of a binomial distribution -/
def variance (p : ℝ) (X : BinomialDistribution p) : ℝ := sorry

theorem binomial_variance (p : ℝ) (X : BinomialDistribution p) :
  variance p X = p * (1 - p) := by sorry

end binomial_variance_l1948_194881


namespace largest_three_digit_product_l1948_194810

def is_prime (p : ℕ) : Prop := sorry

theorem largest_three_digit_product (n x y : ℕ) :
  n ≥ 100 ∧ n < 1000 ∧
  is_prime x ∧ is_prime y ∧ is_prime (10 * y - x) ∧
  x < 10 ∧ y < 10 ∧
  n = x * y * (10 * y - x) ∧
  x ≠ y ∧ x ≠ (10 * y - x) ∧ y ≠ (10 * y - x) →
  n ≤ 705 :=
sorry

end largest_three_digit_product_l1948_194810


namespace y_value_l1948_194879

theorem y_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := by
  sorry

end y_value_l1948_194879


namespace log_sum_theorem_l1948_194822

theorem log_sum_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : y = 2016 * x) (h2 : x^y = y^x) : 
  Real.log x / Real.log 2016 + Real.log y / Real.log 2016 = 2017 / 2015 := by
  sorry

end log_sum_theorem_l1948_194822


namespace enemies_left_undefeated_l1948_194890

theorem enemies_left_undefeated 
  (total_enemies : ℕ) 
  (points_per_enemy : ℕ) 
  (points_earned : ℕ) : 
  total_enemies = 6 → 
  points_per_enemy = 3 → 
  points_earned = 12 → 
  total_enemies - (points_earned / points_per_enemy) = 2 := by
sorry

end enemies_left_undefeated_l1948_194890


namespace c_leq_one_sufficient_not_necessary_l1948_194859

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

def sequence_a (c : ℝ) (n : ℕ) : ℝ :=
  |n - c|

theorem c_leq_one_sufficient_not_necessary (c : ℝ) :
  (c ≤ 1 → is_increasing (sequence_a c)) ∧
  ¬(is_increasing (sequence_a c) → c ≤ 1) :=
sorry

end c_leq_one_sufficient_not_necessary_l1948_194859


namespace hostel_mess_expenditure_l1948_194837

/-- Given a hostel with students and mess expenses, calculate the original expenditure --/
theorem hostel_mess_expenditure 
  (initial_students : ℕ) 
  (student_increase : ℕ) 
  (expense_increase : ℕ) 
  (avg_expense_decrease : ℕ) 
  (h1 : initial_students = 35)
  (h2 : student_increase = 7)
  (h3 : expense_increase = 42)
  (h4 : avg_expense_decrease = 1) :
  ∃ (original_expenditure : ℕ), 
    original_expenditure = initial_students * 
      ((initial_students + student_increase) * 
        (original_expenditure / initial_students - avg_expense_decrease) - 
      original_expenditure) / student_increase ∧
    original_expenditure = 420 :=
by sorry

end hostel_mess_expenditure_l1948_194837


namespace quadratic_roots_l1948_194875

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3
  ∃ x₁ x₂ : ℝ, x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end quadratic_roots_l1948_194875


namespace solution_set_equals_target_set_l1948_194870

/-- The set of solutions for the system of equations with parameter a -/
def SolutionSet : Set (ℝ × ℝ) :=
  {(x, y) | ∃ a : ℝ, a * x + y = 2 * a + 3 ∧ x - a * y = a + 4}

/-- The circle with center (3, 1) and radius √5, excluding (2, -1) -/
def TargetSet : Set (ℝ × ℝ) :=
  {(x, y) | (x - 3)^2 + (y - 1)^2 = 5 ∧ (x, y) ≠ (2, -1)}

theorem solution_set_equals_target_set : SolutionSet = TargetSet := by sorry

end solution_set_equals_target_set_l1948_194870


namespace chocolate_eggs_duration_l1948_194838

/-- Proves that given 40 chocolate eggs, eating 2 eggs per day for 5 days a week will result in the eggs lasting for 4 weeks. -/
theorem chocolate_eggs_duration (total_eggs : ℕ) (eggs_per_day : ℕ) (school_days_per_week : ℕ) : 
  total_eggs = 40 → 
  eggs_per_day = 2 → 
  school_days_per_week = 5 → 
  (total_eggs / (eggs_per_day * school_days_per_week) : ℚ) = 4 := by
sorry


end chocolate_eggs_duration_l1948_194838


namespace total_surveys_per_week_l1948_194830

/-- Proves that the total number of surveys completed per week is 50 given the problem conditions -/
theorem total_surveys_per_week 
  (regular_rate : ℝ)
  (cellphone_rate_increase : ℝ)
  (cellphone_surveys : ℕ)
  (total_earnings : ℝ)
  (h1 : regular_rate = 30)
  (h2 : cellphone_rate_increase = 0.2)
  (h3 : cellphone_surveys = 50)
  (h4 : total_earnings = 3300)
  (h5 : total_earnings = cellphone_surveys * (regular_rate * (1 + cellphone_rate_increase))) :
  cellphone_surveys = 50 := by
  sorry

#check total_surveys_per_week

end total_surveys_per_week_l1948_194830


namespace x_is_bounded_l1948_194827

/-- Product of all decimal digits of a natural number -/
def P (x : ℕ) : ℕ := sorry

/-- Sequence defined recursively by xₙ₊₁ = xₙ + P(xₙ) -/
def x : ℕ → ℕ
  | 0 => sorry  -- x₁ is some positive integer
  | n + 1 => x n + P (x n)

/-- The sequence (xₙ) is bounded -/
theorem x_is_bounded : ∃ (M : ℕ), ∀ (n : ℕ), x n ≤ M := by sorry

end x_is_bounded_l1948_194827


namespace ratio_of_fractions_l1948_194841

theorem ratio_of_fractions (P Q : ℤ) : 
  (∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 → 
    P / (x + 7) + Q / (x^2 - 6*x) = (x^2 - x + 15) / (x^3 + x^2 - 42*x)) →
  Q / P = 7 := by
sorry

end ratio_of_fractions_l1948_194841


namespace tangent_point_abscissa_l1948_194835

/-- The curve function f(x) = x³ - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 1

/-- The slope of the line perpendicular to x + 2y - 1 = 0 -/
def m : ℝ := 2

theorem tangent_point_abscissa :
  ∃ (x : ℝ), (f' x = m) ∧ (x = 1 ∨ x = -1) := by
  sorry

end tangent_point_abscissa_l1948_194835
