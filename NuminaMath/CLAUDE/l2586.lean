import Mathlib

namespace complement_of_M_in_U_l2586_258686

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2, 4}

theorem complement_of_M_in_U :
  U \ M = {3, 5} := by sorry

end complement_of_M_in_U_l2586_258686


namespace base_sum_22_l2586_258607

def F₁ (R : ℕ) : ℚ := (4*R + 5) / (R^2 - 1)
def F₂ (R : ℕ) : ℚ := (5*R + 4) / (R^2 - 1)

theorem base_sum_22 (R₁ R₂ : ℕ) : 
  (F₁ R₁ = 0.454545 ∧ F₂ R₁ = 0.545454) →
  (F₁ R₂ = 3 / 10 ∧ F₂ R₂ = 7 / 10) →
  R₁ + R₂ = 22 := by sorry

end base_sum_22_l2586_258607


namespace same_solution_implies_c_value_l2586_258606

theorem same_solution_implies_c_value (x c : ℚ) : 
  (3 * x + 5 = 1) ∧ (c * x - 8 = -5) → c = -9/4 := by
  sorry

end same_solution_implies_c_value_l2586_258606


namespace solve_for_a_l2586_258660

-- Define the equations as functions of x
def eq1 (x : ℝ) : Prop := 6 * (x + 8) = 18 * x
def eq2 (a x : ℝ) : Prop := 6 * x - 2 * (a - x) = 2 * a + x

-- State the theorem
theorem solve_for_a : ∃ (a : ℝ), ∃ (x : ℝ), eq1 x ∧ eq2 a x ∧ a = 7 := by sorry

end solve_for_a_l2586_258660


namespace z_value_l2586_258642

theorem z_value (x : ℝ) (z : ℝ) (h1 : 3 * x = 0.75 * z) (h2 : x = 20) : z = 80 := by
  sorry

end z_value_l2586_258642


namespace cubic_roots_relation_l2586_258659

theorem cubic_roots_relation (p q r : ℝ) (u v w : ℝ) : 
  (∀ x, x^3 + 5*x^2 + 6*x - 7 = (x - p) * (x - q) * (x - r)) →
  (∀ x, x^3 + u*x^2 + v*x + w = (x - (p + q)) * (x - (q + r)) * (x - (r + p))) →
  w = 37 := by
sorry

end cubic_roots_relation_l2586_258659


namespace solution_pairs_count_l2586_258690

theorem solution_pairs_count : 
  let equation := λ (x y : ℕ) => 4 * x + 7 * y = 600
  ∃! n : ℕ, n = (Finset.filter (λ p : ℕ × ℕ => equation p.1 p.2) (Finset.product (Finset.range 601) (Finset.range 601))).card ∧ n = 22 := by
  sorry

end solution_pairs_count_l2586_258690


namespace square_area_difference_l2586_258605

theorem square_area_difference (small_side large_side : ℝ) 
  (h1 : small_side = 4)
  (h2 : large_side = 9)
  (h3 : small_side < large_side) : 
  large_side^2 - small_side^2 = 65 := by
sorry

end square_area_difference_l2586_258605


namespace distance_AA_l2586_258664

/-- Two unit circles intersecting at X and Y with distance 1 between them -/
structure IntersectingCircles where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  dist_X_Y : dist X Y = 1

/-- Point C on one circle with tangents to the other circle -/
structure TangentPoint (ic : IntersectingCircles) where
  C : ℝ × ℝ
  on_circle : (∃ center, dist center C = 1 ∧ (center = ic.X ∨ center = ic.Y))
  A : ℝ × ℝ  -- Point where tangent CA touches the other circle
  B : ℝ × ℝ  -- Point where tangent CB touches the other circle
  is_tangent_A : ∃ center, dist center A = 1 ∧ center ≠ C
  is_tangent_B : ∃ center, dist center B = 1 ∧ center ≠ C

/-- A' is the point where CB intersects the first circle again -/
def A' (ic : IntersectingCircles) (tp : TangentPoint ic) : ℝ × ℝ :=
  sorry  -- Definition of A' based on the given conditions

/-- The main theorem to prove -/
theorem distance_AA'_is_sqrt3 (ic : IntersectingCircles) (tp : TangentPoint ic) :
  dist tp.A (A' ic tp) = Real.sqrt 3 :=
sorry

end distance_AA_l2586_258664


namespace square_of_difference_l2586_258698

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 25) :
  (5 - Real.sqrt (y^2 - 25))^2 = y^2 - 10 * Real.sqrt (y^2 - 25) := by
  sorry

end square_of_difference_l2586_258698


namespace odd_prime_divisor_property_l2586_258615

theorem odd_prime_divisor_property (n : ℕ+) : 
  (∀ d : ℕ+, d ∣ n → (d + 1) ∣ (n + 1)) ↔ Nat.Prime n.val ∧ n.val % 2 = 1 := by
  sorry

end odd_prime_divisor_property_l2586_258615


namespace village_foods_customers_l2586_258646

/-- The number of customers per month for Village Foods --/
def customers_per_month (lettuce_cost tomato_cost total_cost_per_customer total_sales : ℚ) : ℚ :=
  total_sales / total_cost_per_customer

/-- Theorem: Village Foods gets 500 customers per month --/
theorem village_foods_customers :
  customers_per_month 2 2 4 2000 = 500 := by
  sorry

end village_foods_customers_l2586_258646


namespace min_odd_integers_l2586_258676

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum_ab : a + b = 30)
  (sum_abcd : a + b + c + d = 45)
  (sum_all : a + b + c + d + e + f = 62) :
  ∃ (odd_count : ℕ), 
    odd_count ≥ 2 ∧ 
    (∃ (odd_integers : Finset ℤ), 
      odd_integers.card = odd_count ∧
      odd_integers ⊆ {a, b, c, d, e, f} ∧
      ∀ x ∈ odd_integers, Odd x) ∧
    ∀ (other_odd_count : ℕ),
      other_odd_count < odd_count →
      ¬∃ (other_odd_integers : Finset ℤ),
        other_odd_integers.card = other_odd_count ∧
        other_odd_integers ⊆ {a, b, c, d, e, f} ∧
        ∀ x ∈ other_odd_integers, Odd x :=
sorry

end min_odd_integers_l2586_258676


namespace average_increase_theorem_l2586_258619

/-- Represents a cricket player's batting statistics -/
structure CricketStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional innings -/
def newAverage (stats : CricketStats) (newInningsRuns : ℕ) : ℚ :=
  (stats.totalRuns + newInningsRuns) / (stats.innings + 1)

theorem average_increase_theorem (initialStats : CricketStats) :
  initialStats.innings = 9 →
  newAverage initialStats 200 = initialStats.average + 8 →
  newAverage initialStats 200 = 128 := by
  sorry


end average_increase_theorem_l2586_258619


namespace total_spending_is_correct_l2586_258650

def lunch_cost : ℚ := 50.50
def dessert_cost : ℚ := 8.25
def beverage_cost : ℚ := 3.75
def lunch_discount : ℚ := 0.10
def dessert_tax : ℚ := 0.07
def beverage_tax : ℚ := 0.05
def lunch_tip : ℚ := 0.20
def other_tip : ℚ := 0.15

def discounted_lunch : ℚ := lunch_cost * (1 - lunch_discount)
def taxed_dessert : ℚ := dessert_cost * (1 + dessert_tax)
def taxed_beverage : ℚ := beverage_cost * (1 + beverage_tax)

def lunch_tip_amount : ℚ := discounted_lunch * lunch_tip
def other_tip_amount : ℚ := (taxed_dessert + taxed_beverage) * other_tip

def total_spending : ℚ := discounted_lunch + taxed_dessert + taxed_beverage + lunch_tip_amount + other_tip_amount

theorem total_spending_is_correct : total_spending = 69.23 := by sorry

end total_spending_is_correct_l2586_258650


namespace irrational_root_theorem_l2586_258677

theorem irrational_root_theorem (a : ℝ) :
  (¬ (∃ (q : ℚ), a = q)) →
  (∃ (s p : ℤ), a + (a^3 - 6*a) = s ∧ a*(a^3 - 6*a) = p) →
  (a = -1 - Real.sqrt 2 ∨
   a = -Real.sqrt 5 ∨
   a = 1 - Real.sqrt 2 ∨
   a = -1 + Real.sqrt 2 ∨
   a = Real.sqrt 5 ∨
   a = 1 + Real.sqrt 2) :=
by sorry

end irrational_root_theorem_l2586_258677


namespace absolute_value_equation_solution_l2586_258631

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 4| + 3 * y = 11 :=
by
  -- The unique solution is y = 3.5
  use 3.5
  sorry

end absolute_value_equation_solution_l2586_258631


namespace five_distinct_naturals_product_1000_l2586_258630

theorem five_distinct_naturals_product_1000 :
  ∃ (a b c d e : ℕ), a * b * c * d * e = 1000 ∧
                     a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
                     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
                     c ≠ d ∧ c ≠ e ∧
                     d ≠ e :=
by
  use 1, 2, 4, 5, 25
  sorry

end five_distinct_naturals_product_1000_l2586_258630


namespace candy_bar_calculation_l2586_258685

theorem candy_bar_calculation :
  let f : ℕ := 12
  let b : ℕ := f + 6
  let j : ℕ := 10 * (f + b)
  (40 : ℚ) / 100 * (j ^ 2 : ℚ) = 36000 := by sorry

end candy_bar_calculation_l2586_258685


namespace B_inverse_proof_l2586_258693

variable (A B : Matrix (Fin 2) (Fin 2) ℚ)

def A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1, 2; 3, 4]

theorem B_inverse_proof :
  A⁻¹ = A_inv →
  B * A = 1 →
  B⁻¹ = !![(-2), 1; (3/2), (-1/2)] := by sorry

end B_inverse_proof_l2586_258693


namespace exponential_functional_equation_l2586_258624

theorem exponential_functional_equation 
  (a : ℝ) (ha : a > 0 ∧ a ≠ 1) : 
  ∀ x y : ℝ, (fun x => a^x) x * (fun x => a^x) y = (fun x => a^x) (x + y) :=
by sorry

end exponential_functional_equation_l2586_258624


namespace douglas_county_x_votes_l2586_258679

/-- The percentage of votes Douglas won in county X -/
def douglas_votes_x : ℝ := 74

/-- The ratio of voters in county X to county Y -/
def voter_ratio : ℝ := 2

/-- The percentage of total votes Douglas won in both counties -/
def douglas_total_percent : ℝ := 66

/-- The percentage of votes Douglas won in county Y -/
def douglas_votes_y : ℝ := 50.00000000000002

theorem douglas_county_x_votes :
  let total_votes := voter_ratio + 1
  let douglas_total_votes := douglas_total_percent / 100 * total_votes
  let douglas_y_votes := douglas_votes_y / 100
  douglas_votes_x / 100 * voter_ratio + douglas_y_votes = douglas_total_votes :=
by sorry

end douglas_county_x_votes_l2586_258679


namespace preimage_of_two_one_l2586_258665

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (2 * p.1 + p.2, p.1 - 2 * p.2)

/-- Theorem stating that (1, 0) is the pre-image of (2, 1) under f -/
theorem preimage_of_two_one :
  f (1, 0) = (2, 1) ∧ ∀ p : ℝ × ℝ, f p = (2, 1) → p = (1, 0) := by
  sorry

end preimage_of_two_one_l2586_258665


namespace carnival_rides_l2586_258609

theorem carnival_rides (total_time hours roller_coaster_time tilt_a_whirl_time giant_slide_time : ℕ) 
  (roller_coaster_rides tilt_a_whirl_rides : ℕ) : 
  total_time = hours * 60 →
  roller_coaster_time = 30 →
  tilt_a_whirl_time = 60 →
  giant_slide_time = 15 →
  hours = 4 →
  roller_coaster_rides = 4 →
  tilt_a_whirl_rides = 1 →
  (total_time - (roller_coaster_rides * roller_coaster_time + tilt_a_whirl_rides * tilt_a_whirl_time)) / giant_slide_time = 4 :=
by sorry

end carnival_rides_l2586_258609


namespace smallest_with_eight_divisors_l2586_258643

/-- A function that returns the number of distinct positive divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- Proposition: 24 is the smallest positive integer with exactly eight distinct positive divisors -/
theorem smallest_with_eight_divisors :
  (∀ m : ℕ, m > 0 → m < 24 → numDivisors m ≠ 8) ∧ numDivisors 24 = 8 := by sorry

end smallest_with_eight_divisors_l2586_258643


namespace parallel_vectors_x_value_l2586_258602

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem parallel_vectors_x_value :
  let p : ℝ × ℝ := (2, -3)
  let q : ℝ × ℝ := (x, 6)
  are_parallel p q → x = -4 := by
sorry

end parallel_vectors_x_value_l2586_258602


namespace line_equation_to_slope_intercept_l2586_258611

/-- Given a line equation, prove it can be expressed in slope-intercept form --/
theorem line_equation_to_slope_intercept :
  ∀ (x y : ℝ),
  3 * (x + 2) - 4 * (y - 8) = 0 →
  y = (3 / 4) * x + (19 / 2) :=
by
  sorry

#check line_equation_to_slope_intercept

end line_equation_to_slope_intercept_l2586_258611


namespace f_derivative_at_one_l2586_258657

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem f_derivative_at_one : 
  deriv f 1 = 2 * Real.log 2 - 3 := by sorry

end f_derivative_at_one_l2586_258657


namespace tank_volume_ratio_l2586_258668

/-- Represents the volume ratio of two tanks given specific oil transfer conditions -/
theorem tank_volume_ratio (tank1 tank2 : ℚ) : 
  tank1 > 0 → 
  tank2 > 0 → 
  (3/4 : ℚ) * tank1 = (2/5 : ℚ) * tank2 → 
  tank1 / tank2 = 8/15 := by
  sorry

#check tank_volume_ratio

end tank_volume_ratio_l2586_258668


namespace monthly_income_of_P_l2586_258628

/-- Given the average monthly incomes of three people, prove the monthly income of P. -/
theorem monthly_income_of_P (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (Q + R) / 2 = 6250 →
  (P + R) / 2 = 5200 →
  P = 4000 := by
  sorry

end monthly_income_of_P_l2586_258628


namespace absolute_value_equality_l2586_258681

theorem absolute_value_equality (x : ℚ) :
  (|x + 3| = |x - 4|) ↔ (x = 1/2) := by
  sorry

end absolute_value_equality_l2586_258681


namespace squirrel_acorns_l2586_258692

theorem squirrel_acorns :
  let num_squirrels : ℕ := 5
  let acorns_needed : ℕ := 130
  let acorns_to_collect : ℕ := 15
  let acorns_per_squirrel : ℕ := acorns_needed - acorns_to_collect
  num_squirrels * acorns_per_squirrel = 575 :=
by sorry

end squirrel_acorns_l2586_258692


namespace max_xyz_value_l2586_258608

theorem max_xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y + 2 * z = (x + z) * (y + z))
  (h2 : x + y + 2 * z = 2) :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
  a * b + 2 * c = (a + c) * (b + c) →
  a + b + 2 * c = 2 →
  x * y * z ≥ a * b * c :=
by sorry

end max_xyz_value_l2586_258608


namespace greatest_three_digit_number_with_conditions_l2586_258675

theorem greatest_three_digit_number_with_conditions : ∃ n : ℕ, 
  (n ≤ 999 ∧ n ≥ 100) ∧ 
  (∃ k : ℕ, n = 7 * k + 2) ∧ 
  (∃ m : ℕ, n = 6 * m + 4) ∧
  (∀ x : ℕ, (x ≤ 999 ∧ x ≥ 100) → 
    (∃ a : ℕ, x = 7 * a + 2) → 
    (∃ b : ℕ, x = 6 * b + 4) → 
    x ≤ n) ∧
  n = 994 :=
by sorry

end greatest_three_digit_number_with_conditions_l2586_258675


namespace absolute_value_inequality_solution_set_l2586_258610

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 * x - 3| < 5} = Set.Ioo (-1 : ℝ) 4 := by sorry

end absolute_value_inequality_solution_set_l2586_258610


namespace product_sum_l2586_258699

theorem product_sum (a b : ℕ) (h1 : a / 3 = 16) (h2 : b = a - 1) : a + b = 95 := by
  sorry

end product_sum_l2586_258699


namespace polynomial_root_problem_l2586_258641

theorem polynomial_root_problem (a b : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (2 - 3 * Complex.I : ℂ) ^ 3 + a * (2 - 3 * Complex.I : ℂ) ^ 2 - 2 * (2 - 3 * Complex.I : ℂ) + b = 0 →
  a = -1/4 ∧ b = 195/4 := by
sorry

end polynomial_root_problem_l2586_258641


namespace mother_ate_five_cookies_l2586_258647

def total_cookies : ℕ := 30
def charlie_cookies : ℕ := 15
def father_cookies : ℕ := 10

def mother_cookies : ℕ := total_cookies - (charlie_cookies + father_cookies)

theorem mother_ate_five_cookies : mother_cookies = 5 := by
  sorry

end mother_ate_five_cookies_l2586_258647


namespace common_tangent_range_a_l2586_258617

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

def has_common_tangent (f g : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (deriv f x₁ = deriv g x₂) ∧ 
    (f x₁ - g x₂ = deriv f x₁ * (x₁ - x₂))

theorem common_tangent_range_a :
  ∀ a : ℝ, (∃ x < 0, has_common_tangent f (g a)) → 
    a ∈ Set.Ioi (Real.log (1/(2*Real.exp 1))) :=
sorry

end common_tangent_range_a_l2586_258617


namespace quadratic_residue_characterization_l2586_258623

theorem quadratic_residue_characterization (a b c : ℕ+) :
  (∀ (p : ℕ) (hp : Prime p) (n : ℤ), 
    (∃ (m : ℤ), n ≡ m^2 [ZMOD p]) → 
    (∃ (k : ℤ), (a.val : ℤ) * n^2 + (b.val : ℤ) * n + (c.val : ℤ) ≡ k^2 [ZMOD p])) ↔
  (∃ (d e : ℤ), (a : ℤ) = d^2 ∧ (b : ℤ) = 2*d*e ∧ (c : ℤ) = e^2) :=
sorry

end quadratic_residue_characterization_l2586_258623


namespace hike_consumption_ratio_l2586_258662

/-- Proves the ratio of food to water consumption given hiking conditions --/
theorem hike_consumption_ratio 
  (initial_water : ℝ) 
  (initial_food : ℝ) 
  (initial_gear : ℝ)
  (water_rate : ℝ) 
  (time : ℝ) 
  (final_weight : ℝ) :
  initial_water = 20 →
  initial_food = 10 →
  initial_gear = 20 →
  water_rate = 2 →
  time = 6 →
  final_weight = 34 →
  ∃ (food_rate : ℝ), 
    final_weight = initial_water - water_rate * time + 
                   initial_food - food_rate * time + 
                   initial_gear ∧
    food_rate / water_rate = 2 / 3 :=
by sorry

end hike_consumption_ratio_l2586_258662


namespace length_of_AB_is_10_l2586_258618

-- Define the triangle structures
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the isosceles property
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define the perimeter
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Theorem statement
theorem length_of_AB_is_10 
  (ABC : Triangle) 
  (CBD : Triangle) 
  (isIsoscelesABC : isIsosceles ABC)
  (isIsoscelesCBD : isIsosceles CBD)
  (angle_BAC_twice_ABC : True)  -- We can't directly represent angle relationships, so we use a placeholder
  (perim_CBD : perimeter CBD = 21)
  (perim_ABC : perimeter ABC = 26)
  (length_BD : CBD.c = 9)
  : ABC.a = 10 := by
  sorry


end length_of_AB_is_10_l2586_258618


namespace difference_c_minus_a_l2586_258661

theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) :
  c - a = 90 := by
  sorry

end difference_c_minus_a_l2586_258661


namespace expected_balls_in_original_position_l2586_258687

-- Define the number of balls
def num_balls : ℕ := 10

-- Define the probability of a ball being in its original position after two transpositions
def prob_original_position : ℚ := 18 / 25

-- Theorem statement
theorem expected_balls_in_original_position :
  (num_balls : ℚ) * prob_original_position = 72 / 10 := by
sorry

end expected_balls_in_original_position_l2586_258687


namespace jacket_price_l2586_258620

theorem jacket_price (jacket_count : ℕ) (shorts_count : ℕ) (pants_count : ℕ) 
  (shorts_price : ℚ) (pants_price : ℚ) (total_spent : ℚ) :
  jacket_count = 3 → 
  shorts_count = 2 →
  pants_count = 4 →
  shorts_price = 6 →
  pants_price = 12 →
  total_spent = 90 →
  ∃ (jacket_price : ℚ), 
    jacket_price * jacket_count + shorts_price * shorts_count + pants_price * pants_count = total_spent ∧
    jacket_price = 10 :=
by sorry

end jacket_price_l2586_258620


namespace fraction_equality_l2586_258637

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 7 := by
  sorry

end fraction_equality_l2586_258637


namespace fraction_equality_l2586_258616

theorem fraction_equality : (1000^2 : ℚ) / (252^2 - 248^2) = 500 := by sorry

end fraction_equality_l2586_258616


namespace inequality_proof_l2586_258601

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^3 + b^3) / 2 ≥ ((a^2 + b^2) / 2) * ((a + b) / 2) := by
  sorry

end inequality_proof_l2586_258601


namespace difference_of_squares_l2586_258614

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end difference_of_squares_l2586_258614


namespace sports_enthusiasts_difference_l2586_258694

theorem sports_enthusiasts_difference (total : ℕ) (basketball : ℕ) (football : ℕ)
  (h_total : total = 46)
  (h_basketball : basketball = 23)
  (h_football : football = 29) :
  basketball - (basketball + football - total) = 17 :=
by sorry

end sports_enthusiasts_difference_l2586_258694


namespace consecutive_non_primes_l2586_258603

theorem consecutive_non_primes (k : ℕ+) : ∃ n : ℕ, ∀ i : ℕ, i < k → ¬ Nat.Prime (n + i) := by
  sorry

end consecutive_non_primes_l2586_258603


namespace profit_and_marginal_profit_l2586_258633

/-- The marginal function of f -/
def marginal (f : ℕ → ℝ) : ℕ → ℝ := fun x ↦ f (x + 1) - f x

/-- The revenue function -/
def R : ℕ → ℝ := fun x ↦ 300 * x - 2 * x^2

/-- The cost function -/
def C : ℕ → ℝ := fun x ↦ 50 * x + 300

/-- The profit function -/
def p : ℕ → ℝ := fun x ↦ R x - C x

/-- The marginal profit function -/
def Mp : ℕ → ℝ := marginal p

theorem profit_and_marginal_profit (x : ℕ) (h : 1 ≤ x ∧ x ≤ 100) :
  p x = -2 * x^2 + 250 * x - 300 ∧
  Mp x = 248 - 4 * x ∧
  (∃ y : ℕ, 1 ≤ y ∧ y ≤ 100 ∧ p y = 7512 ∧ ∀ z : ℕ, 1 ≤ z ∧ z ≤ 100 → p z ≤ p y) ∧
  (Mp 1 = 244 ∧ ∀ z : ℕ, 1 < z ∧ z ≤ 100 → Mp z ≤ Mp 1) :=
by sorry

#check profit_and_marginal_profit

end profit_and_marginal_profit_l2586_258633


namespace paper_cutting_l2586_258674

theorem paper_cutting (k : ℕ) : 
  (¬ ∃ (n m : ℕ), 1 + 7 * n + 11 * m = 60) ∧
  (k > 60 → ∃ (n m : ℕ), 1 + 7 * n + 11 * m = k) := by
  sorry

end paper_cutting_l2586_258674


namespace quadratic_properties_l2586_258652

theorem quadratic_properties (a b c m : ℝ) : 
  a < 0 →
  -2 < m →
  m < -1 →
  a * 1^2 + b * 1 + c = 0 →
  a * m^2 + b * m + c = 0 →
  b < 0 ∧ 
  a + b + c = 0 ∧ 
  a * (m + 1) - b + c > 0 := by
sorry

end quadratic_properties_l2586_258652


namespace fruit_rate_proof_l2586_258640

/-- The rate per kg for both apples and mangoes -/
def R : ℝ := 70

/-- The weight of apples purchased in kg -/
def apple_weight : ℝ := 8

/-- The weight of mangoes purchased in kg -/
def mango_weight : ℝ := 9

/-- The total amount paid -/
def total_paid : ℝ := 1190

theorem fruit_rate_proof :
  apple_weight * R + mango_weight * R = total_paid :=
by sorry

end fruit_rate_proof_l2586_258640


namespace pau_total_chicken_l2586_258680

/-- Calculates the total number of chicken pieces Pau eats given the initial orders and a second round of ordering. -/
theorem pau_total_chicken (kobe_order : ℝ) (pau_multiplier : ℝ) (pau_extra : ℝ) (shaq_extra_percent : ℝ) : 
  kobe_order = 5 →
  pau_multiplier = 2 →
  pau_extra = 2.5 →
  shaq_extra_percent = 0.5 →
  2 * (pau_multiplier * kobe_order + pau_extra) = 25 := by
  sorry

end pau_total_chicken_l2586_258680


namespace fixed_point_of_line_family_l2586_258636

theorem fixed_point_of_line_family (k : ℝ) : 
  (3 * k - 1) * (2 / 7) + (k + 2) * (1 / 7) - k = 0 := by
  sorry

end fixed_point_of_line_family_l2586_258636


namespace unique_solution_for_equation_l2586_258697

theorem unique_solution_for_equation : 
  ∃! x y : ℕ+, x^2 - 2 * Nat.factorial y.val = 2021 ∧ x = 45 ∧ y = 2 := by
  sorry

#check unique_solution_for_equation

end unique_solution_for_equation_l2586_258697


namespace sequence_gcd_is_one_l2586_258684

theorem sequence_gcd_is_one (n : ℕ+) : 
  let a : ℕ+ → ℕ := fun k => 100 + 2 * k^2
  Nat.gcd (a n) (a (n + 1)) = 1 := by
sorry

end sequence_gcd_is_one_l2586_258684


namespace alice_favorite_number_l2586_258678

def is_favorite_number (n : ℕ) : Prop :=
  30 ≤ n ∧ n ≤ 70 ∧
  n % 7 = 0 ∧
  n % 3 ≠ 0 ∧
  (n / 10 + n % 10) % 4 = 0

theorem alice_favorite_number :
  ∀ n : ℕ, is_favorite_number n ↔ n = 35 := by
  sorry

end alice_favorite_number_l2586_258678


namespace exponential_function_characterization_l2586_258654

/-- A function f is exponential if it satisfies f(x+y) = f(x)f(y) for all x and y -/
def IsExponential (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x * f y

/-- A function f is monotonically increasing if f(x) ≤ f(y) whenever x ≤ y -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem exponential_function_characterization (f : ℝ → ℝ) 
  (h_exp : IsExponential f) (h_mono : MonoIncreasing f) :
  ∃ a : ℝ, a > 1 ∧ (∀ x, f x = a^x) := by
  sorry

end exponential_function_characterization_l2586_258654


namespace grace_total_pennies_l2586_258604

/-- The value of a dime in pennies -/
def dime_value : ℕ := 10

/-- The value of a coin in pennies -/
def coin_value : ℕ := 5

/-- The number of dimes Grace has -/
def grace_dimes : ℕ := 10

/-- The number of coins Grace has -/
def grace_coins : ℕ := 10

/-- The total value of Grace's dimes and coins in pennies -/
def total_value : ℕ := grace_dimes * dime_value + grace_coins * coin_value

theorem grace_total_pennies : total_value = 150 := by sorry

end grace_total_pennies_l2586_258604


namespace wall_bricks_count_l2586_258653

theorem wall_bricks_count (x : ℝ) 
  (h1 : x > 0)  -- Ensure positive number of bricks
  (h2 : (x / 8 + x / 12 - 15) > 0)  -- Ensure positive combined rate
  (h3 : 6 * (x / 8 + x / 12 - 15) = x)  -- Equation from working together for 6 hours
  : x = 360 := by
  sorry

end wall_bricks_count_l2586_258653


namespace juggler_balls_l2586_258682

theorem juggler_balls (total_jugglers : ℕ) (total_balls : ℕ) 
  (h1 : total_jugglers = 378) 
  (h2 : total_balls = 2268) 
  (h3 : total_balls % total_jugglers = 0) : 
  total_balls / total_jugglers = 6 := by
  sorry

end juggler_balls_l2586_258682


namespace equation_solution_l2586_258627

theorem equation_solution : 
  ∀ x : ℝ, 
    (9 / (Real.sqrt (x - 5) - 10) + 
     2 / (Real.sqrt (x - 5) - 5) + 
     8 / (Real.sqrt (x - 5) + 5) + 
     15 / (Real.sqrt (x - 5) + 10) = 0) ↔ 
    (x = 14 ∨ x = 1335 / 17) := by
  sorry

end equation_solution_l2586_258627


namespace statement_analysis_l2586_258625

theorem statement_analysis (m n : ℝ) : 
  (∀ m n, m + n ≤ 0 → m ≤ 0 ∨ n ≤ 0) ∧ 
  (∀ m n, m > 0 ∧ n > 0 → m + n > 0) ∧ 
  (∃ m n, m + n > 0 ∧ ¬(m > 0 ∧ n > 0)) :=
by sorry

end statement_analysis_l2586_258625


namespace salary_increase_l2586_258667

theorem salary_increase (S : ℝ) (savings_rate_year1 savings_rate_year2 savings_ratio : ℝ) :
  savings_rate_year1 = 0.10 →
  savings_rate_year2 = 0.06 →
  savings_ratio = 0.6599999999999999 →
  ∃ (P : ℝ), 
    savings_rate_year2 * S * (1 + P / 100) = savings_ratio * (savings_rate_year1 * S) ∧
    P = 10 := by
  sorry

end salary_increase_l2586_258667


namespace quadratic_coefficient_sign_l2586_258670

theorem quadratic_coefficient_sign 
  (a b c : ℝ) 
  (h1 : a + b + c < 0) 
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) : 
  c < 0 := by
sorry

end quadratic_coefficient_sign_l2586_258670


namespace clock_time_after_hours_l2586_258626

theorem clock_time_after_hours (current_time hours_passed : ℕ) : 
  current_time = 2 → 
  hours_passed = 3467 → 
  (current_time + hours_passed) % 12 = 9 :=
by
  sorry

end clock_time_after_hours_l2586_258626


namespace right_triangle_inequality_l2586_258696

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
    (h4 : a^2 + b^2 = c^2) : (a + b) / c ≤ Real.sqrt 2 := by
  sorry

end right_triangle_inequality_l2586_258696


namespace a_zero_necessary_not_sufficient_l2586_258691

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The condition "a = 0" is necessary but not sufficient for a complex number z = a + bi to be purely imaginary. -/
theorem a_zero_necessary_not_sufficient :
  (∀ z : ℂ, is_purely_imaginary z → z.re = 0) ∧
  ¬(∀ z : ℂ, z.re = 0 → is_purely_imaginary z) :=
sorry

end a_zero_necessary_not_sufficient_l2586_258691


namespace incorrect_transformation_l2586_258695

theorem incorrect_transformation (a b c : ℝ) : 
  (a = b) → ¬(∀ c, a / c = b / c) := by
  sorry

end incorrect_transformation_l2586_258695


namespace unique_postage_arrangements_l2586_258688

/-- Represents the quantity of stamps for each denomination -/
def stamp_quantities : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

/-- Represents the denominations of stamps available -/
def stamp_denominations : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

/-- The target postage amount -/
def target_postage : Nat := 12

/-- A function to calculate the number of unique arrangements -/
noncomputable def count_arrangements (quantities : List Nat) (denominations : List Nat) (target : Nat) : Nat :=
  sorry  -- Implementation details omitted

/-- Theorem stating that there are 82 unique arrangements -/
theorem unique_postage_arrangements :
  count_arrangements stamp_quantities stamp_denominations target_postage = 82 := by
  sorry

#check unique_postage_arrangements

end unique_postage_arrangements_l2586_258688


namespace line_intercept_sum_l2586_258612

/-- Given a line 3x + 5y + c = 0 where the sum of its x-intercept and y-intercept is 16, prove that c = -30 -/
theorem line_intercept_sum (c : ℝ) : 
  (∃ (x y : ℝ), 3 * x + 5 * y + c = 0 ∧ x + y = 16) → c = -30 := by
  sorry

end line_intercept_sum_l2586_258612


namespace unique_root_in_interval_l2586_258613

theorem unique_root_in_interval (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ x^3 - a*x^2 + 1 = 0 := by
  sorry

end unique_root_in_interval_l2586_258613


namespace fraction_equality_l2586_258673

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 3) 
  (h2 : r / t = 8 / 15) : 
  (4 * m * r - 2 * n * t) / (5 * n * t - 9 * m * r) = -14 / 27 := by
  sorry

end fraction_equality_l2586_258673


namespace delegates_without_badges_l2586_258639

theorem delegates_without_badges (total : ℕ) (pre_printed : ℚ) (break_fraction : ℚ) (hand_written : ℚ) 
  (h_total : total = 100)
  (h_pre_printed : pre_printed = 1/5)
  (h_break : break_fraction = 3/7)
  (h_hand_written : hand_written = 2/9) :
  ↑total - (↑total * pre_printed).floor - 
  ((↑total - (↑total * pre_printed).floor) * break_fraction).floor - 
  (((↑total - (↑total * pre_printed).floor) - ((↑total - (↑total * pre_printed).floor) * break_fraction).floor) * hand_written).floor = 36 :=
by sorry

end delegates_without_badges_l2586_258639


namespace tenth_term_of_sequence_l2586_258655

/-- An arithmetic sequence {aₙ} where a₂ = 2 and a₃ = 4 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 3 = 4 ∧ ∀ n : ℕ, a (n + 1) - a n = a 3 - a 2

theorem tenth_term_of_sequence (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 18 :=
sorry

end tenth_term_of_sequence_l2586_258655


namespace sin_minus_cos_sqrt_two_l2586_258663

theorem sin_minus_cos_sqrt_two (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi →
  Real.sin x - Real.cos x = Real.sqrt 2 →
  x = 3 * Real.pi / 4 := by
sorry

end sin_minus_cos_sqrt_two_l2586_258663


namespace trapezoid_segment_length_l2586_258629

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ
  base_difference : shorter_base + 150 = longer_base
  midline_ratio_condition : (shorter_base + (shorter_base + 150) / 2) / 
    ((shorter_base + 150 + (shorter_base + 150)) / 2) = 3 / 4
  equal_area_condition : ∃ h₁ : ℝ, 
    2 * (1/2 * h₁ * (shorter_base + equal_area_segment)) = 
    1/2 * height * (shorter_base + longer_base)

/-- The main theorem to be proved -/
theorem trapezoid_segment_length (t : Trapezoid) : 
  ⌊t.equal_area_segment^2 / 150⌋ = 300 := by
  sorry

end trapezoid_segment_length_l2586_258629


namespace three_double_derivative_l2586_258656

-- Define the derivative operation
noncomputable def derive (f : ℝ → ℝ) : ℝ → ℝ := sorry

-- Define the given equation as a property
axiom equation (q : ℝ) : derive (λ x => x) q = 3 * q - 3

-- State the theorem
theorem three_double_derivative : derive (derive (λ x => x)) 3 = 15 := by
  sorry

end three_double_derivative_l2586_258656


namespace leftover_value_is_seven_l2586_258600

/-- Calculates the value of leftover coins after pooling and rolling --/
def leftover_value (james_quarters james_dimes rebecca_quarters rebecca_dimes : ℕ) 
  (quarters_per_roll dimes_per_roll : ℕ) : ℚ :=
  let total_quarters := james_quarters + rebecca_quarters
  let total_dimes := james_dimes + rebecca_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * (1 / 4) + (leftover_dimes : ℚ) * (1 / 10)

theorem leftover_value_is_seven :
  leftover_value 50 80 170 340 40 50 = 7 := by
  sorry

end leftover_value_is_seven_l2586_258600


namespace frog_safety_probability_l2586_258645

/-- Represents the probability of the frog reaching stone 14 safely when starting from stone n -/
def safe_probability (n : ℕ) : ℚ := sorry

/-- The total number of stones -/
def total_stones : ℕ := 15

/-- The probability of jumping backwards from stone n -/
def back_prob (n : ℕ) : ℚ := (n + 1) / total_stones

/-- The probability of jumping forwards from stone n -/
def forward_prob (n : ℕ) : ℚ := 1 - back_prob n

theorem frog_safety_probability :
  0 < 2 ∧ 2 < 14 →
  (∀ n : ℕ, 0 < n ∧ n < 14 →
    safe_probability n = back_prob n * safe_probability (n - 1) +
                         forward_prob n * safe_probability (n + 1)) →
  safe_probability 0 = 0 →
  safe_probability 14 = 1 →
  safe_probability 2 = 85 / 256 :=
sorry

end frog_safety_probability_l2586_258645


namespace perpendicular_planes_parallel_l2586_258666

structure Line3D where
  -- Placeholder for 3D line properties

structure Plane3D where
  -- Placeholder for 3D plane properties

def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

theorem perpendicular_planes_parallel (m : Line3D) (α β : Plane3D) :
  perpendicular m α → perpendicular m β → parallel α β := by
  sorry

end perpendicular_planes_parallel_l2586_258666


namespace modular_inverse_13_mod_300_l2586_258671

theorem modular_inverse_13_mod_300 :
  ∃ (x : ℕ), x < 300 ∧ (13 * x) % 300 = 1 :=
by
  use 277
  sorry

end modular_inverse_13_mod_300_l2586_258671


namespace complement_of_α_l2586_258632

-- Define a custom type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the given angle α
def α : Angle := ⟨25, 39⟩

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let total_minutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨total_minutes / 60, total_minutes % 60⟩

-- Theorem statement
theorem complement_of_α :
  complement α = ⟨64, 21⟩ := by
  sorry

end complement_of_α_l2586_258632


namespace animal_arrangement_count_l2586_258683

def number_of_cages : ℕ := 15
def empty_cages : ℕ := 3
def number_of_chickens : ℕ := 3
def number_of_dogs : ℕ := 3
def number_of_cats : ℕ := 6

def arrangement_count : ℕ := Nat.choose number_of_cages empty_cages * 
                              Nat.factorial 3 * 
                              Nat.factorial number_of_chickens * 
                              Nat.factorial number_of_dogs * 
                              Nat.factorial number_of_cats

theorem animal_arrangement_count : arrangement_count = 70761600 := by
  sorry

end animal_arrangement_count_l2586_258683


namespace square_lake_area_l2586_258622

/-- Represents a square lake with a given boat speed and crossing times -/
structure SquareLake where
  boat_speed : ℝ  -- Speed of the boat in miles per hour
  length_time : ℝ  -- Time to cross the length in hours
  width_time : ℝ  -- Time to cross the width in hours

/-- Calculates the area of a square lake based on boat speed and crossing times -/
def lake_area (lake : SquareLake) : ℝ :=
  (lake.boat_speed * lake.length_time) * (lake.boat_speed * lake.width_time)

/-- Theorem: The area of the specified square lake is 100 square miles -/
theorem square_lake_area :
  let lake := SquareLake.mk 10 2 (1/2)
  lake_area lake = 100 := by
  sorry


end square_lake_area_l2586_258622


namespace segments_AB_CD_parallel_l2586_258621

-- Define points in 2D space
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (2, -1)
def C : ℝ × ℝ := (0, 4)
def D : ℝ × ℝ := (2, -4)

-- Define a function to check if two segments are parallel
def are_parallel (p1 p2 q1 q2 : ℝ × ℝ) : Prop :=
  let v1 := (p2.1 - p1.1, p2.2 - p1.2)
  let v2 := (q2.1 - q1.1, q2.2 - q1.2)
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

-- Theorem statement
theorem segments_AB_CD_parallel :
  are_parallel A B C D := by
  sorry

end segments_AB_CD_parallel_l2586_258621


namespace sum_a_b_eq_neg_four_l2586_258638

theorem sum_a_b_eq_neg_four (a b : ℝ) (h : |1 - 2*a + b| + 2*a = -a^2 - 1) : 
  a + b = -4 := by sorry

end sum_a_b_eq_neg_four_l2586_258638


namespace f_neg_two_eq_one_fourth_l2586_258672

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

-- Theorem statement
theorem f_neg_two_eq_one_fourth :
  f (-2) = 1/4 := by sorry

end f_neg_two_eq_one_fourth_l2586_258672


namespace triangle_folding_theorem_l2586_258634

/-- A triangle represented by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A folding method is a function that takes a triangle and produces a set of fold lines -/
def FoldingMethod := Triangle → Set (ℝ × ℝ → ℝ × ℝ)

/-- The result of applying a folding method to a triangle -/
structure FoldedObject where
  original : Triangle
  foldLines : Set (ℝ × ℝ → ℝ × ℝ)
  thickness : ℕ

/-- A folding method is valid if it produces a folded object with uniform thickness -/
def isValidFolding (method : FoldingMethod) : Prop :=
  ∀ t : Triangle, ∃ fo : FoldedObject, 
    fo.original = t ∧ 
    fo.foldLines = method t ∧ 
    fo.thickness = 2020

/-- The main theorem: there exists a valid folding method for any triangle -/
theorem triangle_folding_theorem : ∃ (method : FoldingMethod), isValidFolding method := by
  sorry

end triangle_folding_theorem_l2586_258634


namespace fraction_transformation_impossibility_l2586_258669

theorem fraction_transformation_impossibility : ¬∃ (a b : ℕ), (2 + 2013 * a) / (3 + 2014 * b) = 3 / 5 := by
  sorry

end fraction_transformation_impossibility_l2586_258669


namespace fraction_equivalence_l2586_258648

theorem fraction_equivalence : 
  ∃ (n : ℚ), (3 + n) / (5 + n) = 9 / 11 ∧ n = 6 := by sorry

end fraction_equivalence_l2586_258648


namespace average_increment_l2586_258635

theorem average_increment (a b c : ℝ) (h : (a + b + c) / 3 = 8) :
  ((a + 1) + (b + 2) + (c + 3)) / 3 = 10 := by
  sorry

end average_increment_l2586_258635


namespace bus_driver_compensation_l2586_258658

/-- A bus driver's compensation problem -/
theorem bus_driver_compensation 
  (regular_rate : ℝ) 
  (overtime_rate_increase : ℝ) 
  (max_regular_hours : ℕ) 
  (total_compensation : ℝ) 
  (h1 : regular_rate = 16)
  (h2 : overtime_rate_increase = 0.75)
  (h3 : max_regular_hours = 40)
  (h4 : total_compensation = 864) :
  ∃ (total_hours : ℕ), 
    total_hours = 48 ∧ 
    (↑max_regular_hours * regular_rate + 
     (↑total_hours - ↑max_regular_hours) * (regular_rate * (1 + overtime_rate_increase)) = 
     total_compensation) :=
by sorry

end bus_driver_compensation_l2586_258658


namespace AC_length_l2586_258649

/-- A right triangle with a circle passing through its altitude --/
structure RightTriangleWithCircle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  -- ABC is a right triangle with right angle at A
  right_angle_at_A : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- AH is perpendicular to BC
  AH_perpendicular_BC : (H.1 - A.1) * (C.1 - B.1) + (H.2 - A.2) * (C.2 - B.2) = 0
  -- A circle passes through A, H, X, and Y
  circle_passes : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (H.1 - center.1)^2 + (H.2 - center.2)^2 = radius^2 ∧
    (X.1 - center.1)^2 + (X.2 - center.2)^2 = radius^2 ∧
    (Y.1 - center.1)^2 + (Y.2 - center.2)^2 = radius^2
  -- X is on AB
  X_on_AB : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ X = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  -- Y is on AC
  Y_on_AC : ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 1 ∧ Y = (A.1 + s * (C.1 - A.1), A.2 + s * (C.2 - A.2))
  -- Given lengths
  AX_length : ((X.1 - A.1)^2 + (X.2 - A.2)^2)^(1/2 : ℝ) = 5
  AY_length : ((Y.1 - A.1)^2 + (Y.2 - A.2)^2)^(1/2 : ℝ) = 6
  AB_length : ((B.1 - A.1)^2 + (B.2 - A.2)^2)^(1/2 : ℝ) = 9

/-- The main theorem --/
theorem AC_length (t : RightTriangleWithCircle) : 
  ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)^(1/2 : ℝ) = 13.5 := by
  sorry

end AC_length_l2586_258649


namespace sail_pressure_velocity_l2586_258644

/-- The pressure-area-velocity relationship for a boat sail -/
theorem sail_pressure_velocity 
  (k : ℝ) 
  (A₁ A₂ V₁ V₂ P₁ P₂ : ℝ) 
  (h1 : P₁ = k * A₁ * V₁^2) 
  (h2 : P₂ = k * A₂ * V₂^2) 
  (h3 : A₁ = 2) 
  (h4 : V₁ = 20) 
  (h5 : P₁ = 5) 
  (h6 : A₂ = 4) 
  (h7 : P₂ = 20) : 
  V₂ = 20 * Real.sqrt 2 := by
sorry

end sail_pressure_velocity_l2586_258644


namespace sock_pairs_combinations_l2586_258689

/-- Given 7 pairs of socks, proves that the number of ways to choose 2 socks 
    from different pairs is 84. -/
theorem sock_pairs_combinations (n : ℕ) (h : n = 7) : 
  (2 * n * (2 * n - 2)) / 2 = 84 := by
  sorry

end sock_pairs_combinations_l2586_258689


namespace equation_has_real_root_l2586_258651

theorem equation_has_real_root (M : ℝ) : ∃ x : ℝ, x = M^2 * (x - 1) * (x - 2) * (x - 3) := by
  sorry

end equation_has_real_root_l2586_258651
