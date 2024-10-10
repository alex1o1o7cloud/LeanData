import Mathlib

namespace sum_of_integers_l3412_341278

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 128) : x + y = 24 := by
  sorry

end sum_of_integers_l3412_341278


namespace football_tournament_yardage_l3412_341247

/-- Represents a football team's yardage progress --/
structure TeamProgress where
  gains : List Int
  losses : List Int
  bonus : Int
  penalty : Int

/-- Calculates the total yardage progress for a team --/
def totalYardage (team : TeamProgress) : Int :=
  (team.gains.sum - team.losses.sum) + team.bonus - team.penalty

/-- The football tournament scenario --/
def footballTournament : Prop :=
  let teamA : TeamProgress := {
    gains := [8, 6],
    losses := [5, 3],
    bonus := 0,
    penalty := 2
  }
  let teamB : TeamProgress := {
    gains := [4, 9],
    losses := [2, 7],
    bonus := 0,
    penalty := 3
  }
  let teamC : TeamProgress := {
    gains := [2, 11],
    losses := [6, 4],
    bonus := 3,
    penalty := 4
  }
  (totalYardage teamA = 4) ∧
  (totalYardage teamB = 1) ∧
  (totalYardage teamC = 2)

theorem football_tournament_yardage : footballTournament := by
  sorry

end football_tournament_yardage_l3412_341247


namespace drive_time_calculation_l3412_341204

/-- Given a person drives 120 miles in 3 hours, prove that driving 200 miles
    at the same speed will take 5 hours. -/
theorem drive_time_calculation (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ)
    (h1 : distance1 = 120)
    (h2 : time1 = 3)
    (h3 : distance2 = 200) :
  let speed := distance1 / time1
  distance2 / speed = 5 := by
  sorry

end drive_time_calculation_l3412_341204


namespace abc_product_l3412_341267

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (b + c) = 154) (h2 : b * (c + a) = 164) (h3 : c * (a + b) = 172) :
  a * b * c = Real.sqrt 538083 := by
sorry

end abc_product_l3412_341267


namespace min_sum_of_two_digits_is_one_l3412_341238

/-- A digit is a natural number from 0 to 9 -/
def Digit := { n : ℕ // n ≤ 9 }

/-- The theorem states that the minimum sum of two digits P and Q is 1,
    given that P, Q, R, and S are four different digits,
    and (P+Q)/(R+S) is an integer and as small as possible. -/
theorem min_sum_of_two_digits_is_one
  (P Q R S : Digit)
  (h_distinct : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S)
  (h_integer : ∃ k : ℕ, (P.val + Q.val : ℚ) / (R.val + S.val) = k)
  (h_min : ∀ (P' Q' R' S' : Digit),
           P' ≠ Q' ∧ P' ≠ R' ∧ P' ≠ S' ∧ Q' ≠ R' ∧ Q' ≠ S' ∧ R' ≠ S' →
           (∃ k : ℕ, (P'.val + Q'.val : ℚ) / (R'.val + S'.val) = k) →
           (P.val + Q.val : ℚ) / (R.val + S.val) ≤ (P'.val + Q'.val : ℚ) / (R'.val + S'.val)) :
  P.val + Q.val = 1 :=
sorry

end min_sum_of_two_digits_is_one_l3412_341238


namespace carter_school_earnings_l3412_341250

/-- Represents the number of students from each school --/
def students_adams : ℕ := 8
def students_bentley : ℕ := 6
def students_carter : ℕ := 7

/-- Represents the number of days worked by students from each school --/
def days_adams : ℕ := 4
def days_bentley : ℕ := 6
def days_carter : ℕ := 10

/-- Total amount paid for all students' work --/
def total_paid : ℚ := 1020

/-- Theorem stating that the earnings for Carter school students is approximately $517.39 --/
theorem carter_school_earnings : 
  let total_student_days := students_adams * days_adams + students_bentley * days_bentley + students_carter * days_carter
  let daily_wage := total_paid / total_student_days
  let carter_earnings := daily_wage * (students_carter * days_carter)
  ∃ ε > 0, |carter_earnings - 517.39| < ε :=
sorry

end carter_school_earnings_l3412_341250


namespace correct_number_value_l3412_341264

theorem correct_number_value (n : ℕ) (initial_avg correct_avg wrong_value : ℚ) :
  n = 10 →
  initial_avg = 5 →
  wrong_value = 26 →
  correct_avg = 6 →
  ∃ (correct_value : ℚ),
    correct_value = wrong_value + n * (correct_avg - initial_avg) ∧
    correct_value = 36 := by
  sorry

end correct_number_value_l3412_341264


namespace equation_transformation_l3412_341276

theorem equation_transformation (x : ℝ) : (3 * x - 7 = 2 * x) ↔ (3 * x - 2 * x = 7) := by
  sorry

end equation_transformation_l3412_341276


namespace cone_prism_volume_ratio_l3412_341201

/-- The ratio of the volume of a right circular cone to the volume of its circumscribing right rectangular prism -/
theorem cone_prism_volume_ratio :
  ∀ (r h : ℝ), r > 0 → h > 0 →
  (1 / 3 * π * r^2 * h) / (9 * r^2 * h) = π / 27 := by
sorry

end cone_prism_volume_ratio_l3412_341201


namespace second_child_birth_year_l3412_341269

theorem second_child_birth_year (first_child_age : ℕ) (fourth_child_age : ℕ) 
  (h1 : first_child_age = 15)
  (h2 : fourth_child_age = 8)
  (h3 : ∃ (third_child_age : ℕ), third_child_age = fourth_child_age + 2)
  (h4 : ∃ (second_child_age : ℕ), second_child_age + 4 = third_child_age) :
  first_child_age - (fourth_child_age + 6) = 1 := by
sorry

end second_child_birth_year_l3412_341269


namespace x_values_l3412_341200

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 7) (h2 : y + 1 / x = 1 / 3) :
  x = 3 ∨ x = 4 := by
  sorry

end x_values_l3412_341200


namespace andrew_grape_purchase_l3412_341232

/-- The amount of grapes Andrew purchased in kg -/
def G : ℝ := by sorry

/-- The price of grapes per kg -/
def grape_price : ℝ := 70

/-- The amount of mangoes Andrew purchased in kg -/
def mango_amount : ℝ := 9

/-- The price of mangoes per kg -/
def mango_price : ℝ := 55

/-- The total amount Andrew paid -/
def total_paid : ℝ := 1055

theorem andrew_grape_purchase :
  G * grape_price + mango_amount * mango_price = total_paid ∧ G = 8 := by sorry

end andrew_grape_purchase_l3412_341232


namespace unique_four_digit_number_l3412_341234

theorem unique_four_digit_number : ∃! n : ℕ,
  (1000 ≤ n ∧ n < 10000) ∧ 
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n % 1000 = b^3) ∧
  (∃ c : ℕ, n % 100 = c^4) ∧
  n = 9216 :=
by sorry

end unique_four_digit_number_l3412_341234


namespace parametric_to_general_equation_l3412_341230

/-- Parametric equations to general equation conversion -/
theorem parametric_to_general_equation :
  ∀ θ : ℝ,
  let x : ℝ := 2 + Real.sin θ ^ 2
  let y : ℝ := -1 + Real.cos (2 * θ)
  2 * x + y - 4 = 0 ∧ x ∈ Set.Icc 2 3 := by
  sorry

end parametric_to_general_equation_l3412_341230


namespace stripe_area_on_cylindrical_tank_l3412_341237

/-- The area of a stripe painted on a cylindrical tank -/
theorem stripe_area_on_cylindrical_tank 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h1 : diameter = 40)
  (h2 : stripe_width = 4)
  (h3 : revolutions = 3) : 
  stripe_width * (Real.pi * diameter * revolutions) = 480 * Real.pi := by
  sorry

end stripe_area_on_cylindrical_tank_l3412_341237


namespace last_student_score_l3412_341248

theorem last_student_score (total_students : ℕ) (average_19 : ℝ) (average_20 : ℝ) :
  total_students = 20 →
  average_19 = 82 →
  average_20 = 84 →
  ∃ (last_score oliver_score : ℝ),
    (19 * average_19 + oliver_score) / total_students = average_20 ∧
    oliver_score = 2 * last_score →
    last_score = 61 := by
  sorry

end last_student_score_l3412_341248


namespace centroid_quadrilateral_area_l3412_341290

/-- Given a square ABCD with side length 40 and a point Q inside the square
    such that AQ = 16 and BQ = 34, the area of the quadrilateral formed by
    the centroids of △ABQ, △BCQ, △CDQ, and △DAQ is 6400/9. -/
theorem centroid_quadrilateral_area (A B C D Q : ℝ × ℝ) : 
  let square_side : ℝ := 40
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  -- Square ABCD conditions
  (dist A B = square_side) ∧ 
  (dist B C = square_side) ∧ 
  (dist C D = square_side) ∧ 
  (dist D A = square_side) ∧ 
  -- Right angles
  ((B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) ∧
  -- Q inside square
  (0 < Q.1) ∧ (Q.1 < square_side) ∧ (0 < Q.2) ∧ (Q.2 < square_side) ∧
  -- AQ and BQ distances
  (dist A Q = 16) ∧ 
  (dist B Q = 34) →
  -- Area of quadrilateral formed by centroids
  let centroid (P1 P2 P3 : ℝ × ℝ) := 
    ((P1.1 + P2.1 + P3.1) / 3, (P1.2 + P2.2 + P3.2) / 3)
  let G1 := centroid A B Q
  let G2 := centroid B C Q
  let G3 := centroid C D Q
  let G4 := centroid D A Q
  let area := (dist G1 G3 * dist G2 G4) / 2
  area = 6400 / 9 := by
sorry

end centroid_quadrilateral_area_l3412_341290


namespace equation_solution_l3412_341268

theorem equation_solution (x : ℝ) : 
  (Real.sqrt (6 * x^2 + 1)) / (Real.sqrt (3 * x^2 + 4)) = 2 / Real.sqrt 3 ↔ 
  x = Real.sqrt (13/6) ∨ x = -Real.sqrt (13/6) := by
sorry

end equation_solution_l3412_341268


namespace Q_equals_N_l3412_341252

-- Define the sets Q and N
def Q : Set ℝ := {y | ∃ x, y = x^2 + 1}
def N : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem Q_equals_N : Q = N := by sorry

end Q_equals_N_l3412_341252


namespace expression_evaluation_l3412_341253

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end expression_evaluation_l3412_341253


namespace polynomial_range_l3412_341294

noncomputable def P (p q x : ℝ) : ℝ := x^2 + p*x + q

theorem polynomial_range (p q : ℝ) :
  let rangeP := {y | ∃ x ∈ Set.Icc (-1 : ℝ) 1, P p q x = y}
  (p < -2 → rangeP = Set.Icc (1 + p + q) (1 - p + q)) ∧
  (-2 ≤ p ∧ p ≤ 0 → rangeP = Set.Icc (q - p^2/4) (1 - p + q)) ∧
  (0 ≤ p ∧ p ≤ 2 → rangeP = Set.Icc (q - p^2/4) (1 + p + q)) ∧
  (p > 2 → rangeP = Set.Icc (1 - p + q) (1 + p + q)) :=
by sorry

end polynomial_range_l3412_341294


namespace log_division_simplification_l3412_341293

theorem log_division_simplification :
  (Real.log 16) / (Real.log (1/16)) = -1 := by
  sorry

end log_division_simplification_l3412_341293


namespace camp_girls_count_l3412_341263

theorem camp_girls_count (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 133 → difference = 33 → girls + (girls + difference) = total → girls = 50 := by
  sorry

end camp_girls_count_l3412_341263


namespace two_dogs_food_consumption_l3412_341291

/-- The amount of dog food consumed by two dogs in a day -/
def total_dog_food_consumption (dog1_consumption dog2_consumption : Real) : Real :=
  dog1_consumption + dog2_consumption

/-- Theorem: Two dogs each consuming 0.125 scoop of dog food per day eat 0.25 scoop in total -/
theorem two_dogs_food_consumption :
  total_dog_food_consumption 0.125 0.125 = 0.25 := by
  sorry

end two_dogs_food_consumption_l3412_341291


namespace vikki_hourly_rate_l3412_341255

/-- Vikki's weekly work hours -/
def work_hours : ℝ := 42

/-- Tax deduction rate -/
def tax_rate : ℝ := 0.20

/-- Insurance deduction rate -/
def insurance_rate : ℝ := 0.05

/-- Union dues deduction -/
def union_dues : ℝ := 5

/-- Vikki's take-home pay after deductions -/
def take_home_pay : ℝ := 310

/-- Vikki's hourly pay rate -/
def hourly_rate : ℝ := 10

theorem vikki_hourly_rate :
  work_hours * hourly_rate * (1 - tax_rate - insurance_rate) - union_dues = take_home_pay :=
sorry

end vikki_hourly_rate_l3412_341255


namespace polynomial_identity_l3412_341296

theorem polynomial_identity (x : ℝ) : 
  (x + 2)^4 + 4*(x + 2)^3 + 6*(x + 2)^2 + 4*(x + 2) + 1 = (x + 3)^4 := by
  sorry

end polynomial_identity_l3412_341296


namespace range_of_g_l3412_341273

def f (x : ℝ) : ℝ := 4 * x - 3

def g (x : ℝ) : ℝ := f (f (f (f (f x))))

def domain_g : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }

theorem range_of_g :
  ∀ x ∈ domain_g, 1 ≤ g x ∧ g x ≤ 2049 ∧
  ∃ y ∈ domain_g, g y = 1 ∧
  ∃ z ∈ domain_g, g z = 2049 :=
sorry

end range_of_g_l3412_341273


namespace sequence_proof_l3412_341207

theorem sequence_proof (a : Fin 8 → ℕ) 
  (h1 : a 0 = 11)
  (h2 : a 7 = 12)
  (h3 : ∀ i : Fin 6, a i + a (i + 1) + a (i + 2) = 50) :
  a = ![11, 12, 27, 11, 12, 27, 11, 12] := by
sorry

end sequence_proof_l3412_341207


namespace complex_equation_solution_l3412_341271

theorem complex_equation_solution :
  ∃ z : ℂ, (5 : ℂ) + (3 - 2*I)*z = 2 + 5*I*z ∧ z = -(9/58) - (21/58)*I :=
by sorry

end complex_equation_solution_l3412_341271


namespace exist_six_points_similar_triangles_l3412_341260

/-- A point in a plane represented by its coordinates -/
structure Point (α : Type*) where
  x : α
  y : α

/-- A triangle represented by its three vertices -/
structure Triangle (α : Type*) where
  A : Point α
  B : Point α
  C : Point α

/-- Predicate to check if two triangles are similar -/
def similar {α : Type*} (t1 t2 : Triangle α) : Prop :=
  sorry

/-- Theorem stating the existence of six points forming similar triangles -/
theorem exist_six_points_similar_triangles :
  ∃ (X₁ X₂ Y₁ Y₂ Z₁ Z₂ : Point ℝ),
    ∀ (i j k : Fin 2),
      similar
        (Triangle.mk (if i = 0 then X₁ else X₂) (if j = 0 then Y₁ else Y₂) (if k = 0 then Z₁ else Z₂))
        (Triangle.mk X₁ Y₁ Z₁) :=
  sorry

end exist_six_points_similar_triangles_l3412_341260


namespace circle_and_line_properties_l3412_341298

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 4}

-- Define the line
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 - 1}

-- Define the center of the circle
def center : ℝ × ℝ := (2, 3)

-- Define the property of being tangent to y-axis
def tangent_to_y_axis (C : Set (ℝ × ℝ)) : Prop :=
  ∃ y, (0, y) ∈ C ∧ ∀ x ≠ 0, (x, y) ∉ C

-- Define the perpendicularity condition
def perpendicular (M N : ℝ × ℝ) : Prop :=
  (M.1 - center.1) * (N.1 - center.1) + (M.2 - center.2) * (N.2 - center.2) = 0

theorem circle_and_line_properties :
  tangent_to_y_axis circle_C →
  ∀ k, ∃ M N, M ∈ circle_C ∧ N ∈ circle_C ∧ M ∈ line k ∧ N ∈ line k ∧ perpendicular M N →
  (k = 1 ∨ k = 7) := by
  sorry

end circle_and_line_properties_l3412_341298


namespace least_n_without_square_l3412_341205

theorem least_n_without_square : ∃ (N : ℕ), N = 282 ∧ 
  (∀ (k : ℕ), k < N → ∃ (x : ℕ), ∃ (i : ℕ), i < 1000 ∧ x^2 = 1000*k + i) ∧
  (∀ (x : ℕ), ¬∃ (i : ℕ), i < 1000 ∧ x^2 = 1000*N + i) :=
by sorry

end least_n_without_square_l3412_341205


namespace prime_between_40_and_50_and_largest_below_100_l3412_341266

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_between_40_and_50_and_largest_below_100 :
  (∀ p : ℕ, 40 < p ∧ p < 50 ∧ isPrime p ↔ p = 41 ∨ p = 43 ∨ p = 47) ∧
  (∀ q : ℕ, q < 100 ∧ isPrime q → q ≤ 97) ∧
  isPrime 97 :=
sorry

end prime_between_40_and_50_and_largest_below_100_l3412_341266


namespace square_plus_self_even_l3412_341209

theorem square_plus_self_even (n : ℤ) : Even (n^2 + n) := by sorry

end square_plus_self_even_l3412_341209


namespace capri_sun_pouches_per_box_l3412_341226

theorem capri_sun_pouches_per_box 
  (total_boxes : ℕ) 
  (total_paid : ℚ) 
  (cost_per_pouch : ℚ) 
  (h1 : total_boxes = 10) 
  (h2 : total_paid = 12) 
  (h3 : cost_per_pouch = 1/5) : 
  (total_paid / cost_per_pouch) / total_boxes = 6 := by
sorry

end capri_sun_pouches_per_box_l3412_341226


namespace weight_of_new_person_l3412_341272

theorem weight_of_new_person (initial_weight : ℝ) (weight_increase : ℝ) :
  initial_weight = 65 →
  weight_increase = 4.5 →
  ∃ (new_weight : ℝ), new_weight = initial_weight + 2 * weight_increase :=
by
  sorry

end weight_of_new_person_l3412_341272


namespace arithmetic_computation_l3412_341254

theorem arithmetic_computation : -9 * 5 - (-7 * -4) + (-12 * -6) = -1 := by
  sorry

end arithmetic_computation_l3412_341254


namespace max_value_of_expression_l3412_341283

theorem max_value_of_expression (x y z : ℝ) (h : x + 3 * y + z = 5) :
  ∃ (max : ℝ), max = 125 / 4 ∧ ∀ (a b c : ℝ), a + 3 * b + c = 5 → a * b + a * c + b * c ≤ max :=
by sorry

end max_value_of_expression_l3412_341283


namespace largest_n_satisfying_property_l3412_341216

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

/-- A function that checks if a number is an odd prime -/
def isOddPrime (p : ℕ) : Prop := isPrime p ∧ p % 2 ≠ 0

/-- The property that n satisfies: for any odd prime p < n, n - p is prime -/
def satisfiesProperty (n : ℕ) : Prop :=
  ∀ p : ℕ, p < n → isOddPrime p → isPrime (n - p)

theorem largest_n_satisfying_property :
  (satisfiesProperty 10) ∧ 
  (∀ m : ℕ, m > 10 → ¬(satisfiesProperty m)) :=
sorry

end largest_n_satisfying_property_l3412_341216


namespace yellow_marbles_count_l3412_341270

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by
  sorry

end yellow_marbles_count_l3412_341270


namespace sum_of_fractions_l3412_341284

theorem sum_of_fractions : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + 
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) = 3 / 8 := by
  sorry

end sum_of_fractions_l3412_341284


namespace system_solution_l3412_341227

theorem system_solution :
  ∃! (x y z : ℚ),
    2 * x - 3 * y + z = 8 ∧
    4 * x - 6 * y + 2 * z = 16 ∧
    x + y - z = 1 ∧
    x = 11 / 3 ∧
    y = 1 ∧
    z = 11 / 3 := by
  sorry

end system_solution_l3412_341227


namespace mutual_fund_share_price_increase_l3412_341213

theorem mutual_fund_share_price_increase (initial_price : ℝ) : 
  let first_quarter_price := initial_price * 1.25
  let second_quarter_price := initial_price * 1.55
  (second_quarter_price - first_quarter_price) / first_quarter_price * 100 = 24 := by
  sorry

end mutual_fund_share_price_increase_l3412_341213


namespace line_inclination_angle_l3412_341225

def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x + 3 * y + 2 = 0

def inclination_angle (eq : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem line_inclination_angle :
  inclination_angle line_equation = 150 * Real.pi / 180 :=
sorry

end line_inclination_angle_l3412_341225


namespace farmer_land_usage_l3412_341257

theorem farmer_land_usage (beans wheat corn total : ℕ) : 
  beans + wheat + corn = total →
  5 * wheat = 2 * beans →
  2 * corn = beans →
  corn = 376 →
  total = 1034 := by
sorry

end farmer_land_usage_l3412_341257


namespace smallest_multiplier_for_perfect_square_l3412_341249

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem smallest_multiplier_for_perfect_square :
  ∃ n : ℕ, n > 0 ∧ is_perfect_square (n * y) ∧
  ∀ m : ℕ, 0 < m ∧ m < n → ¬is_perfect_square (m * y) :=
sorry

end smallest_multiplier_for_perfect_square_l3412_341249


namespace notebook_cost_l3412_341288

theorem notebook_cost (notebook_cost pencil_cost : ℝ) 
  (total_cost : notebook_cost + pencil_cost = 2.40)
  (cost_difference : notebook_cost = pencil_cost + 2) : 
  notebook_cost = 2.20 := by
sorry

end notebook_cost_l3412_341288


namespace eight_power_fifteen_div_sixtyfour_power_six_l3412_341219

theorem eight_power_fifteen_div_sixtyfour_power_six : 8^15 / 64^6 = 512 := by
  sorry

end eight_power_fifteen_div_sixtyfour_power_six_l3412_341219


namespace tangent_point_coordinates_l3412_341258

theorem tangent_point_coordinates (x y : ℝ) : 
  y = x^2 → -- Point (x, y) is on the curve y = x^2
  (2*x = 1) → -- Tangent line has slope 1 (tan(π/4) = 1)
  (x = 1/2 ∧ y = 1/4) := by -- The coordinates are (1/2, 1/4)
sorry

end tangent_point_coordinates_l3412_341258


namespace multiples_of_five_most_representative_l3412_341212

/-- Represents a sampling method for the math test --/
inductive SamplingMethod
  | TopStudents
  | BottomStudents
  | FemaleStudents
  | MultiplesOfFive

/-- Represents a student in the seventh grade --/
structure Student where
  id : Nat
  gender : Bool  -- True for female, False for male
  score : Nat

/-- The population of students who took the test --/
def population : Finset Student := sorry

/-- The total number of students in the population --/
axiom total_students : Finset.card population = 400

/-- Defines what makes a sampling method representative --/
def is_representative (method : SamplingMethod) : Prop := sorry

/-- Theorem stating that selecting students with numbers that are multiples of 5 
    is the most representative sampling method --/
theorem multiples_of_five_most_representative : 
  is_representative SamplingMethod.MultiplesOfFive ∧ 
  ∀ m : SamplingMethod, m ≠ SamplingMethod.MultiplesOfFive → 
    ¬(is_representative m) :=
sorry

end multiples_of_five_most_representative_l3412_341212


namespace calculate_number_of_bs_l3412_341240

/-- Calculates the number of Bs given the recess rules and report card results -/
theorem calculate_number_of_bs (
  normal_recess : ℕ)
  (extra_time_per_a : ℕ)
  (extra_time_per_b : ℕ)
  (extra_time_per_c : ℕ)
  (less_time_per_d : ℕ)
  (num_as : ℕ)
  (num_cs : ℕ)
  (num_ds : ℕ)
  (total_recess : ℕ)
  (h1 : normal_recess = 20)
  (h2 : extra_time_per_a = 2)
  (h3 : extra_time_per_b = 1)
  (h4 : extra_time_per_c = 0)
  (h5 : less_time_per_d = 1)
  (h6 : num_as = 10)
  (h7 : num_cs = 14)
  (h8 : num_ds = 5)
  (h9 : total_recess = 47) :
  ∃ (num_bs : ℕ), num_bs = 12 ∧
    total_recess = normal_recess + num_as * extra_time_per_a + num_bs * extra_time_per_b + num_cs * extra_time_per_c - num_ds * less_time_per_d :=
by
  sorry


end calculate_number_of_bs_l3412_341240


namespace additive_function_value_l3412_341228

/-- A function satisfying f(x+y) = f(x) + f(y) for all real x and y -/
def AdditiveFunctionR (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

/-- Theorem: If f is an additive function on ℝ and f(2) = 4, then f(1) = 2 -/
theorem additive_function_value (f : ℝ → ℝ) (h1 : AdditiveFunctionR f) (h2 : f 2 = 4) : f 1 = 2 := by
  sorry

end additive_function_value_l3412_341228


namespace find_y_l3412_341236

theorem find_y (x y : ℤ) (h1 : x + y = 300) (h2 : x - y = 200) : y = 50 := by
  sorry

end find_y_l3412_341236


namespace class_size_proof_l3412_341297

theorem class_size_proof (total : ℕ) 
  (h1 : 20 < total ∧ total < 30)
  (h2 : ∃ n : ℕ, total = 8 * n + 2)
  (h3 : ∃ M F : ℕ, M = 5 * n ∧ F = 4 * n) 
  (h4 : ∃ n : ℕ, n = (20 * M) / 100 ∧ n = (25 * F) / 100) :
  total = 26 := by
sorry

end class_size_proof_l3412_341297


namespace product_digit_sum_base7_l3412_341233

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a number in base-7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem product_digit_sum_base7 :
  sumOfDigitsBase7 (toBase7 (toBase10 35 * toBase10 13)) = 11 := by sorry

end product_digit_sum_base7_l3412_341233


namespace set_d_forms_triangle_l3412_341215

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. --/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem set_d_forms_triangle :
  can_form_triangle 6 6 6 := by
  sorry

end set_d_forms_triangle_l3412_341215


namespace imaginary_part_of_z_l3412_341287

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = (1 + Complex.I) / 2) :
  Complex.im z = -1/2 := by sorry

end imaginary_part_of_z_l3412_341287


namespace amy_height_l3412_341282

def angela_height : ℕ := 157
def angela_helen_diff : ℕ := 4
def helen_amy_diff : ℕ := 3

theorem amy_height : 
  angela_height - angela_helen_diff - helen_amy_diff = 150 :=
by sorry

end amy_height_l3412_341282


namespace paradise_park_ferris_wheel_seats_l3412_341262

/-- The number of seats on a Ferris wheel -/
def ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) : ℕ :=
  total_people / people_per_seat

/-- Theorem: The Ferris wheel in paradise park has 4 seats -/
theorem paradise_park_ferris_wheel_seats :
  ferris_wheel_seats 16 4 = 4 := by
  sorry

end paradise_park_ferris_wheel_seats_l3412_341262


namespace unique_solution_for_inequality_l3412_341274

theorem unique_solution_for_inequality : 
  ∃! n : ℕ+, -46 ≤ (2023 : ℝ) / (46 - n.val) ∧ (2023 : ℝ) / (46 - n.val) ≤ 46 - n.val :=
by
  -- Proof goes here
  sorry

end unique_solution_for_inequality_l3412_341274


namespace factor_expression_l3412_341285

theorem factor_expression (b c : ℝ) : 55 * b^2 + 165 * b * c = 55 * b * (b + 3 * c) := by
  sorry

end factor_expression_l3412_341285


namespace plane_perpendicularity_l3412_341208

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (m n : Line) (α β : Plane) :
  parallel m n → 
  contained_in m α → 
  perpendicular n β → 
  plane_perpendicular α β := by
sorry

end plane_perpendicularity_l3412_341208


namespace trapezoid_side_length_l3412_341265

theorem trapezoid_side_length (square_side : ℝ) (trapezoid_area hexagon_area : ℝ) 
  (x : ℝ) : 
  square_side = 1 →
  trapezoid_area = hexagon_area →
  trapezoid_area = 1/4 →
  x = trapezoid_area * 4 / (1 + square_side) →
  x = 1/2 :=
by sorry

end trapezoid_side_length_l3412_341265


namespace perfect_square_binomial_l3412_341281

theorem perfect_square_binomial (k : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (x + b)^2) ↔ k = 100 := by
  sorry

end perfect_square_binomial_l3412_341281


namespace bridge_length_l3412_341277

/-- The length of a bridge given train parameters -/
theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 120) 
  (h2 : train_speed_kmh = 45) 
  (h3 : crossing_time = 30) : 
  ∃ (bridge_length : ℝ), bridge_length = 255 := by
sorry

end bridge_length_l3412_341277


namespace triangle_area_l3412_341223

theorem triangle_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2) 
  (h_angle : Real.cos (30 * π / 180) = c / (2 * a)) (h_side : b = 8) : 
  (1/2) * a * b = 32 * Real.sqrt 3 := by
  sorry

end triangle_area_l3412_341223


namespace smallest_natural_with_eight_divisors_ending_in_zero_l3412_341242

theorem smallest_natural_with_eight_divisors_ending_in_zero (N : ℕ) :
  (N % 10 = 0) →  -- N ends with 0
  (Finset.card (Nat.divisors N) = 8) →  -- N has exactly 8 divisors
  (∀ M : ℕ, M % 10 = 0 ∧ Finset.card (Nat.divisors M) = 8 → N ≤ M) →  -- N is the smallest such number
  N = 30 := by
sorry

end smallest_natural_with_eight_divisors_ending_in_zero_l3412_341242


namespace min_xy_value_l3412_341243

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/(4*y) = 1) :
  x * y ≥ 2 := by
  sorry

end min_xy_value_l3412_341243


namespace complex_magnitude_l3412_341239

theorem complex_magnitude (z : ℂ) (h : z - 2 * Complex.I = 1 + z * Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_magnitude_l3412_341239


namespace algebraic_expression_value_l3412_341279

theorem algebraic_expression_value : ∀ a : ℝ, a^2 + a = 3 → 2*a^2 + 2*a - 1 = 5 := by
  sorry

end algebraic_expression_value_l3412_341279


namespace scientific_notation_correct_l3412_341259

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 2000000

/-- The scientific notation representation of the original number -/
def scientific_repr : ScientificNotation := {
  coefficient := 2
  exponent := 6
  is_valid := by sorry
}

/-- Theorem stating that the scientific notation representation is correct -/
theorem scientific_notation_correct : 
  (scientific_repr.coefficient * (10 ^ scientific_repr.exponent : ℝ)) = original_number := by sorry

end scientific_notation_correct_l3412_341259


namespace first_term_is_two_l3412_341211

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  monotone_increasing : ∀ n, a n < a (n + 1)
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_first_three : a 1 + a 2 + a 3 = 12
  product_first_three : a 1 * a 2 * a 3 = 48

/-- The first term of the arithmetic sequence is 2 -/
theorem first_term_is_two (seq : ArithmeticSequence) : seq.a 1 = 2 := by
  sorry

end first_term_is_two_l3412_341211


namespace right_triangle_leg_length_l3412_341256

theorem right_triangle_leg_length 
  (a b c : ℝ) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 13) 
  (h_leg : a = 5) : 
  b = 12 := by
sorry

end right_triangle_leg_length_l3412_341256


namespace reservoir_capacity_l3412_341280

theorem reservoir_capacity : ∀ (capacity : ℚ),
  (1/8 : ℚ) * capacity + 200 = (1/2 : ℚ) * capacity →
  capacity = 1600/3 := by
  sorry

end reservoir_capacity_l3412_341280


namespace smallest_sum_arithmetic_cubic_sequence_l3412_341246

theorem smallest_sum_arithmetic_cubic_sequence (A B C D : ℕ+) : 
  (∃ r : ℚ, B = A + r ∧ C = B + r) →  -- A, B, C form an arithmetic sequence
  (D - C = (C - B)^2) →  -- B, C, D form a cubic sequence
  (C : ℚ) / B = 4 / 3 →  -- C/B = 4/3
  (∀ A' B' C' D' : ℕ+, 
    (∃ r' : ℚ, B' = A' + r' ∧ C' = B' + r') → 
    (D' - C' = (C' - B')^2) → 
    (C' : ℚ) / B' = 4 / 3 → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 14 :=
by sorry

end smallest_sum_arithmetic_cubic_sequence_l3412_341246


namespace quadratic_negative_roots_probability_l3412_341299

/-- The probability that a quadratic equation with a randomly selected coefficient has two negative roots -/
theorem quadratic_negative_roots_probability : 
  ∃ (f : ℝ → ℝ → ℝ → Prop) (P : Set ℝ → ℝ),
    (∀ p x₁ x₂, f p x₁ x₂ ↔ x₁^2 + 2*p*x₁ + 3*p - 2 = 0 ∧ x₂^2 + 2*p*x₂ + 3*p - 2 = 0 ∧ x₁ < 0 ∧ x₂ < 0) →
    (P (Set.Icc 0 5) = 5) →
    P {p ∈ Set.Icc 0 5 | ∃ x₁ x₂, f p x₁ x₂} / P (Set.Icc 0 5) = 2/3 := by
  sorry

end quadratic_negative_roots_probability_l3412_341299


namespace car_trip_distance_l3412_341206

theorem car_trip_distance (D : ℝ) :
  let remaining_after_first_stop := D / 2
  let remaining_after_second_stop := remaining_after_first_stop * 2 / 3
  let remaining_after_third_stop := remaining_after_second_stop * 3 / 5
  remaining_after_third_stop = 180
  → D = 900 := by
  sorry

end car_trip_distance_l3412_341206


namespace dantes_recipe_total_l3412_341241

def dantes_recipe (eggs : ℕ) : ℕ :=
  eggs + eggs / 2

theorem dantes_recipe_total : dantes_recipe 60 = 90 := by
  sorry

end dantes_recipe_total_l3412_341241


namespace tissue_magnification_l3412_341203

/-- Given a circular piece of tissue magnified by an electron microscope, 
    this theorem proves the relationship between the magnified image diameter 
    and the actual tissue diameter. -/
theorem tissue_magnification (magnification : ℝ) (magnified_diameter : ℝ) 
  (h1 : magnification = 1000) 
  (h2 : magnified_diameter = 2) :
  magnified_diameter / magnification = 0.002 := by
  sorry

end tissue_magnification_l3412_341203


namespace distance_p_to_y_axis_l3412_341245

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate -/
def distance_to_y_axis (x : ℝ) : ℝ := |x|

/-- Given a point P(-3, 2) in the second quadrant, its distance to the y-axis is 3 -/
theorem distance_p_to_y_axis :
  let P : ℝ × ℝ := (-3, 2)
  distance_to_y_axis P.1 = 3 := by sorry

end distance_p_to_y_axis_l3412_341245


namespace orchard_trees_l3412_341244

theorem orchard_trees (total : ℕ) (peach : ℕ) (pear : ℕ) 
  (h1 : total = 480) 
  (h2 : pear = 3 * peach) 
  (h3 : total = peach + pear) : 
  peach = 120 ∧ pear = 360 := by
  sorry

end orchard_trees_l3412_341244


namespace game_gameplay_hours_l3412_341231

theorem game_gameplay_hours (T : ℝ) (h1 : 0.2 * T + 30 = 50) : T = 100 := by
  sorry

end game_gameplay_hours_l3412_341231


namespace optimal_sampling_methods_l3412_341221

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents income levels -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Community structure -/
structure Community where
  totalFamilies : Nat
  highIncomeFamilies : Nat
  middleIncomeFamilies : Nat
  lowIncomeFamilies : Nat

/-- Sample size for family survey -/
def familySampleSize : Nat := 100

/-- Student selection parameters -/
structure StudentSelection where
  totalStudents : Nat
  studentsToSelect : Nat

/-- Function to determine the optimal sampling method for family survey -/
def optimalFamilySamplingMethod (c : Community) : SamplingMethod := sorry

/-- Function to determine the optimal sampling method for student selection -/
def optimalStudentSamplingMethod (s : StudentSelection) : SamplingMethod := sorry

/-- Theorem stating the optimal sampling methods for the given scenario -/
theorem optimal_sampling_methods 
  (community : Community)
  (studentSelection : StudentSelection)
  (h1 : community.totalFamilies = 800)
  (h2 : community.highIncomeFamilies = 200)
  (h3 : community.middleIncomeFamilies = 480)
  (h4 : community.lowIncomeFamilies = 120)
  (h5 : studentSelection.totalStudents = 10)
  (h6 : studentSelection.studentsToSelect = 3) :
  optimalFamilySamplingMethod community = SamplingMethod.Stratified ∧
  optimalStudentSamplingMethod studentSelection = SamplingMethod.SimpleRandom := by
  sorry


end optimal_sampling_methods_l3412_341221


namespace jamesFinalNumber_l3412_341217

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define Kyle's result
def kylesResult : ℕ := sumOfDigits (2014^2014)

-- Define Shannon's result
def shannonsResult : ℕ := sumOfDigits kylesResult

-- Theorem to prove
theorem jamesFinalNumber : sumOfDigits shannonsResult = 7 := by sorry

end jamesFinalNumber_l3412_341217


namespace sum_of_bases_equal_1193_l3412_341289

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 14 to base 10 -/
def base14_to_base10 (n : ℕ) : ℕ := sorry

/-- Represents the digit C in base 14 -/
def C : ℕ := 12

theorem sum_of_bases_equal_1193 :
  base8_to_base10 356 + base14_to_base10 (4 * 14^2 + C * 14 + 3) = 1193 := by sorry

end sum_of_bases_equal_1193_l3412_341289


namespace average_problem_l3412_341235

theorem average_problem (numbers : List ℕ) (x : ℕ) : 
  numbers = [201, 202, 204, 205, 206, 209, 209, 210, 212] →
  (numbers.sum + x) / 10 = 207 →
  x = 212 := by
sorry

end average_problem_l3412_341235


namespace triangle_base_length_l3412_341218

/-- Given a triangle with area 615 and height 10, prove its base is 123 -/
theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) 
  (h_area : area = 615) 
  (h_height : height = 10) 
  (h_triangle_area : area = (base * height) / 2) : base = 123 := by
  sorry

end triangle_base_length_l3412_341218


namespace range_of_a_minus_b_l3412_341214

theorem range_of_a_minus_b (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 2) :
  -3 < a - b ∧ a - b < 0 := by
  sorry

end range_of_a_minus_b_l3412_341214


namespace chess_tournament_ratio_l3412_341295

theorem chess_tournament_ratio (total_students : ℕ) (tournament_students : ℕ) :
  total_students = 24 →
  tournament_students = 4 →
  (total_students / 3 : ℚ) = (total_students / 3 : ℕ) →
  (tournament_students : ℚ) / (total_students / 3 : ℚ) = 1 / 2 := by
  sorry

end chess_tournament_ratio_l3412_341295


namespace honey_barrel_distribution_l3412_341222

/-- Represents a barrel of honey -/
inductive Barrel
  | Full
  | Half
  | Empty

/-- Represents a distribution of barrels to a person -/
structure Distribution :=
  (full : ℕ)
  (half : ℕ)
  (empty : ℕ)

/-- Calculates the amount of honey in a distribution -/
def honey_amount (d : Distribution) : ℚ :=
  d.full + d.half / 2

/-- Calculates the total number of barrels in a distribution -/
def barrel_count (d : Distribution) : ℕ :=
  d.full + d.half + d.empty

/-- Checks if a distribution is valid (7 barrels and 3.5 units of honey) -/
def is_valid_distribution (d : Distribution) : Prop :=
  barrel_count d = 7 ∧ honey_amount d = 7/2

/-- Represents a solution to the honey distribution problem -/
structure Solution :=
  (person1 : Distribution)
  (person2 : Distribution)
  (person3 : Distribution)

/-- Checks if a solution is valid -/
def is_valid_solution (s : Solution) : Prop :=
  is_valid_distribution s.person1 ∧
  is_valid_distribution s.person2 ∧
  is_valid_distribution s.person3 ∧
  s.person1.full + s.person2.full + s.person3.full = 7 ∧
  s.person1.half + s.person2.half + s.person3.half = 7 ∧
  s.person1.empty + s.person2.empty + s.person3.empty = 7

theorem honey_barrel_distribution :
  ∃ (s : Solution), is_valid_solution s :=
sorry

end honey_barrel_distribution_l3412_341222


namespace complex_magnitude_sum_l3412_341286

theorem complex_magnitude_sum (i : ℂ) : i^2 = -1 →
  Complex.abs ((2 + i)^24 + (2 - i)^24) = 488281250 := by
  sorry

end complex_magnitude_sum_l3412_341286


namespace village_population_l3412_341261

theorem village_population (P : ℝ) : 
  (P > 0) →
  (0.85 * (0.9 * P) = 3213) →
  P = 4200 := by
sorry

end village_population_l3412_341261


namespace circle_polar_equation_l3412_341202

/-- The polar coordinate equation of a circle C, given specific conditions -/
theorem circle_polar_equation (C : Set (ℝ × ℝ)) (P : ℝ × ℝ) (l : ℝ → ℝ → Prop) :
  (P = (Real.sqrt 2, π / 4)) →
  (∀ ρ θ, l ρ θ ↔ ρ * Real.sin (θ - π / 3) = -Real.sqrt 3 / 2) →
  (∃ x, x ∈ C ∧ x.1 = 1 ∧ x.2 = 0) →
  (P ∈ C) →
  (∀ ρ θ, (ρ, θ) ∈ C ↔ ρ = 2 * Real.cos θ) :=
by sorry

end circle_polar_equation_l3412_341202


namespace gala_tree_count_l3412_341229

/-- Represents an apple orchard with Fuji and Gala trees -/
structure AppleOrchard where
  totalTrees : ℕ
  pureFuji : ℕ
  pureGala : ℕ
  crossPollinated : ℕ

/-- The conditions of the orchard as described in the problem -/
def orchardConditions (o : AppleOrchard) : Prop :=
  o.crossPollinated = o.totalTrees / 10 ∧
  o.pureFuji = (o.totalTrees * 3) / 4 ∧
  o.pureFuji + o.crossPollinated = 170 ∧
  o.totalTrees = o.pureFuji + o.pureGala + o.crossPollinated

/-- The theorem stating that under the given conditions, there are 50 pure Gala trees -/
theorem gala_tree_count (o : AppleOrchard) 
  (h : orchardConditions o) : o.pureGala = 50 := by
  sorry

#check gala_tree_count

end gala_tree_count_l3412_341229


namespace number_of_balls_l3412_341292

theorem number_of_balls (x : ℕ) : x - 92 = 156 - x → x = 124 := by
  sorry

end number_of_balls_l3412_341292


namespace inequality_proof_l3412_341251

theorem inequality_proof (a b : Real) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  a^5 + b^3 + (a - b)^2 ≤ 2 := by
sorry

end inequality_proof_l3412_341251


namespace parabola_rotation_l3412_341224

/-- A parabola in the xy-plane -/
structure Parabola where
  a : ℝ  -- coefficient of (x-h)^2
  h : ℝ  -- x-coordinate of vertex
  k : ℝ  -- y-coordinate of vertex

/-- Rotate a parabola by 180 degrees around its vertex -/
def rotate180 (p : Parabola) : Parabola :=
  { a := -p.a, h := p.h, k := p.k }

theorem parabola_rotation (p : Parabola) (hp : p = ⟨2, 3, -2⟩) :
  rotate180 p = ⟨-2, 3, -2⟩ := by
  sorry

#check parabola_rotation

end parabola_rotation_l3412_341224


namespace greatest_common_multiple_15_20_less_than_150_l3412_341210

def is_common_multiple (m n k : ℕ) : Prop := k % m = 0 ∧ k % n = 0

theorem greatest_common_multiple_15_20_less_than_150 : 
  ∃ (k : ℕ), k = 120 ∧ 
  is_common_multiple 15 20 k ∧ 
  k < 150 ∧ 
  ∀ (m : ℕ), is_common_multiple 15 20 m → m < 150 → m ≤ k :=
sorry

end greatest_common_multiple_15_20_less_than_150_l3412_341210


namespace work_completion_time_l3412_341220

theorem work_completion_time (renu_rate suma_rate : ℚ) 
  (h1 : renu_rate = 1 / 8)
  (h2 : suma_rate = 1 / (24 / 5))
  : (1 / (renu_rate + suma_rate) : ℚ) = 3 := by
  sorry

end work_completion_time_l3412_341220


namespace power_multiplication_l3412_341275

theorem power_multiplication (a : ℝ) : 3 * a^4 * (4 * a) = 12 * a^5 := by
  sorry

end power_multiplication_l3412_341275
