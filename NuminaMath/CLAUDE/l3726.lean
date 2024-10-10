import Mathlib

namespace max_negative_integers_l3726_372647

theorem max_negative_integers (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) :
  ∃ (w : ℕ), w ≤ 4 ∧
  ∀ (n : ℕ), (∃ (s : Finset (Fin 6)), s.card = n ∧
    (∀ i ∈ s, match i with
      | 0 => a < 0
      | 1 => b < 0
      | 2 => c < 0
      | 3 => d < 0
      | 4 => e < 0
      | 5 => f < 0
    )) → n ≤ w :=
by sorry

end max_negative_integers_l3726_372647


namespace fence_cost_l3726_372676

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 56) :
  4 * Real.sqrt area * price_per_foot = 3808 := by
  sorry

#check fence_cost

end fence_cost_l3726_372676


namespace arithmetic_sequence_common_difference_l3726_372660

/-- An arithmetic sequence with first term 2 and the property a_2 + a_4 = a_6 has common difference 2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first_term : a 1 = 2) 
  (h_sum_property : a 2 + a 4 = a 6) :
  ∀ n : ℕ, a (n + 1) - a n = 2 := by
sorry

end arithmetic_sequence_common_difference_l3726_372660


namespace age_calculation_l3726_372640

/-- Given a two-digit birth year satisfying certain conditions, prove the person's age in 1955 --/
theorem age_calculation (x y : ℕ) (h : 10 * x + y + 4 = 43) : 1955 - (1900 + 10 * x + y) = 16 := by
  sorry

end age_calculation_l3726_372640


namespace bleach_time_is_correct_l3726_372667

/-- Represents the hair dyeing process with given time constraints -/
def HairDyeingProcess (total_time bleach_time : ℝ) : Prop :=
  bleach_time > 0 ∧
  total_time = bleach_time + (4 * bleach_time) + (1/3 * bleach_time)

/-- Theorem stating that given the constraints, the bleaching time is 1.875 hours -/
theorem bleach_time_is_correct (total_time : ℝ) 
  (h : total_time = 10) : 
  ∃ (bleach_time : ℝ), HairDyeingProcess total_time bleach_time ∧ bleach_time = 1.875 := by
  sorry

end bleach_time_is_correct_l3726_372667


namespace min_m_for_solution_non_monotonic_range_l3726_372669

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x^2 - 1| + x

-- Part I
theorem min_m_for_solution (a : ℝ) (h : a = 2) :
  (∃ m : ℝ, ∀ x : ℝ, f a x - m ≤ 0) ↔ m ≥ -Real.sqrt 2 / 2 :=
sorry

-- Part II
theorem non_monotonic_range (a : ℝ) :
  (∃ x y : ℝ, x ∈ Set.Icc (-3) 2 ∧ y ∈ Set.Icc (-3) 2 ∧ x < y ∧ f a x > f a y) ↔
  a < -1/6 ∨ a > 1/6 :=
sorry

end min_m_for_solution_non_monotonic_range_l3726_372669


namespace uno_card_price_l3726_372627

/-- The original price of an Uno Giant Family Card -/
def original_price : ℝ := 12

/-- The number of cards purchased -/
def num_cards : ℕ := 10

/-- The discount applied to each card -/
def discount : ℝ := 2

/-- The total amount paid -/
def total_paid : ℝ := 100

/-- Theorem stating that the original price satisfies the given conditions -/
theorem uno_card_price : 
  num_cards * (original_price - discount) = total_paid := by
  sorry


end uno_card_price_l3726_372627


namespace escalator_time_l3726_372687

/-- The time taken for a person to cover the entire length of an escalator -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) 
  (h1 : escalator_speed = 20)
  (h2 : person_speed = 4)
  (h3 : escalator_length = 210) : 
  escalator_length / (escalator_speed + person_speed) = 8.75 := by
sorry

end escalator_time_l3726_372687


namespace roots_sum_of_squares_l3726_372643

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 5*a + 5 = 0) → (b^2 - 5*b + 5 = 0) → a^2 + b^2 = 15 := by
  sorry

end roots_sum_of_squares_l3726_372643


namespace sector_central_angle_l3726_372679

/-- Given a sector with circumference 6 and area 2, its central angle in radians is either 1 or 4. -/
theorem sector_central_angle (r l : ℝ) (h1 : 2*r + l = 6) (h2 : (1/2)*l*r = 2) :
  l/r = 1 ∨ l/r = 4 := by
  sorry

end sector_central_angle_l3726_372679


namespace average_sum_of_abs_diff_l3726_372693

def sum_of_abs_diff (perm : Fin 8 → Fin 8) : ℕ :=
  |perm 0 - perm 1| + |perm 2 - perm 3| + |perm 4 - perm 5| + |perm 6 - perm 7|

def all_permutations : Finset (Fin 8 → Fin 8) :=
  Finset.univ.filter (fun f => Function.Injective f)

theorem average_sum_of_abs_diff :
  (Finset.sum all_permutations sum_of_abs_diff) / all_permutations.card = 20 := by
  sorry

end average_sum_of_abs_diff_l3726_372693


namespace min_value_problem_l3726_372600

/-- The problem statement -/
theorem min_value_problem (m n : ℝ) : 
  (∃ (x y : ℝ), x + y - 1 = 0 ∧ 3 * x - y - 7 = 0 ∧ m * x + y + n = 0) →
  (m * n > 0) →
  (∀ k : ℝ, (1 / m + 2 / n ≥ k) → k ≤ 8) ∧ 
  (∃ m₀ n₀ : ℝ, (∃ (x y : ℝ), x + y - 1 = 0 ∧ 3 * x - y - 7 = 0 ∧ m₀ * x + y + n₀ = 0) ∧ 
                (m₀ * n₀ > 0) ∧ 
                (1 / m₀ + 2 / n₀ = 8)) :=
by sorry

end min_value_problem_l3726_372600


namespace complement_intersection_theorem_l3726_372635

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {1, 3}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl A ∩ B) = {3} := by
  sorry

end complement_intersection_theorem_l3726_372635


namespace convex_bodies_with_coinciding_projections_intersect_l3726_372626

/-- A convex body in 3D space -/
structure ConvexBody3D where
  -- Add necessary fields/axioms for a convex body

/-- Projection of a convex body onto a coordinate plane -/
def projection (body : ConvexBody3D) (plane : Fin 3) : Set (Fin 2 → ℝ) :=
  sorry

/-- Two convex bodies intersect if they have a common point -/
def intersect (body1 body2 : ConvexBody3D) : Prop :=
  sorry

/-- Main theorem: If two convex bodies have coinciding projections on all coordinate planes, 
    then they must intersect -/
theorem convex_bodies_with_coinciding_projections_intersect 
  (body1 body2 : ConvexBody3D) 
  (h : ∀ (plane : Fin 3), projection body1 plane = projection body2 plane) : 
  intersect body1 body2 :=
sorry

end convex_bodies_with_coinciding_projections_intersect_l3726_372626


namespace shadow_length_indeterminate_l3726_372603

/-- Represents a person's shadow length under different light sources -/
structure Shadow where
  sunLength : ℝ
  streetLightLength : ℝ → ℝ

/-- The theorem states that given Xiao Ming's shadow is longer than Xiao Qiang's under sunlight,
    it's impossible to determine their relative shadow lengths under a streetlight -/
theorem shadow_length_indeterminate 
  (xiaoming xioaqiang : Shadow)
  (h_sun : xiaoming.sunLength > xioaqiang.sunLength) :
  ∃ (d₁ d₂ : ℝ), 
    xiaoming.streetLightLength d₁ > xioaqiang.streetLightLength d₂ ∧
    ∃ (d₃ d₄ : ℝ), 
      xiaoming.streetLightLength d₃ < xioaqiang.streetLightLength d₄ :=
by sorry

end shadow_length_indeterminate_l3726_372603


namespace credit_card_more_beneficial_l3726_372620

/-- Represents the purchase amount in rubles -/
def purchase_amount : ℝ := 8000

/-- Represents the credit card cashback rate -/
def credit_cashback_rate : ℝ := 0.005

/-- Represents the debit card cashback rate -/
def debit_cashback_rate : ℝ := 0.0075

/-- Represents the monthly interest rate on the debit card -/
def debit_interest_rate : ℝ := 0.005

/-- Calculates the total income when using the credit card -/
def credit_card_income (amount : ℝ) : ℝ :=
  amount * credit_cashback_rate + amount * debit_interest_rate

/-- Calculates the total income when using the debit card -/
def debit_card_income (amount : ℝ) : ℝ :=
  amount * debit_cashback_rate

/-- Theorem stating that using the credit card is more beneficial -/
theorem credit_card_more_beneficial :
  credit_card_income purchase_amount > debit_card_income purchase_amount :=
sorry


end credit_card_more_beneficial_l3726_372620


namespace fewer_bees_than_flowers_l3726_372652

theorem fewer_bees_than_flowers (flowers : ℕ) (bees : ℕ) 
  (h1 : flowers = 5) (h2 : bees = 3) : flowers - bees = 2 := by
  sorry

end fewer_bees_than_flowers_l3726_372652


namespace g_constant_value_l3726_372684

-- Define the function g
def g : ℝ → ℝ := fun x ↦ 5

-- Theorem statement
theorem g_constant_value (x : ℝ) : g (x + 3) = 5 := by
  sorry

end g_constant_value_l3726_372684


namespace inscribed_square_area_l3726_372629

theorem inscribed_square_area (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0) 
  (h4 : a * b = x^2) (h5 : a = 34) (h6 : b = 66) : x^2 = 2244 := by
  sorry

end inscribed_square_area_l3726_372629


namespace combined_teaching_experience_l3726_372695

theorem combined_teaching_experience (james_experience partner_experience : ℕ) 
  (h1 : james_experience = 40)
  (h2 : partner_experience = james_experience - 10) :
  james_experience + partner_experience = 70 := by
sorry

end combined_teaching_experience_l3726_372695


namespace inequality_proof_l3726_372674

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 :=
by sorry

end inequality_proof_l3726_372674


namespace product_of_numbers_with_given_sum_and_difference_l3726_372608

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 100 ∧ x - y = 8 → x * y = 2484 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l3726_372608


namespace star_property_l3726_372601

def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  λ (a, b) (c, d) ↦ (a - c, b + d)

theorem star_property :
  ∀ x y : ℤ, star (x, y) (2, 3) = star (5, 4) (1, 1) → x = 6 := by
  sorry

end star_property_l3726_372601


namespace church_full_capacity_l3726_372622

/-- Calculates the number of people needed to fill a church given the number of rows, chairs per row, and people per chair. -/
def church_capacity (rows : ℕ) (chairs_per_row : ℕ) (people_per_chair : ℕ) : ℕ :=
  rows * chairs_per_row * people_per_chair

/-- Theorem stating that a church with 20 rows, 6 chairs per row, and 5 people per chair can hold 600 people. -/
theorem church_full_capacity : church_capacity 20 6 5 = 600 := by
  sorry

end church_full_capacity_l3726_372622


namespace complex_number_properties_l3726_372602

def z : ℂ := 3 - 4 * Complex.I

theorem complex_number_properties : 
  (Complex.abs z = 5) ∧ 
  (∃ (y : ℝ), z - 3 = y * Complex.I) ∧
  (z.re > 0 ∧ z.im < 0) := by
  sorry

end complex_number_properties_l3726_372602


namespace fred_weekend_earnings_l3726_372657

/-- Fred's earnings over the weekend -/
def fred_earnings (initial_amount final_amount : ℕ) : ℕ :=
  final_amount - initial_amount

/-- Theorem stating Fred's earnings -/
theorem fred_weekend_earnings :
  fred_earnings 19 40 = 21 := by
  sorry

end fred_weekend_earnings_l3726_372657


namespace perimeter_ratio_l3726_372624

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- The original large rectangle -/
def largeRectangle : Rectangle := { width := 4, height := 6 }

/-- One of the small rectangles after folding and cutting -/
def smallRectangle : Rectangle := { width := 2, height := 3 }

/-- Theorem stating the ratio of perimeters -/
theorem perimeter_ratio :
  perimeter smallRectangle / perimeter largeRectangle = 1 / 2 := by
  sorry

end perimeter_ratio_l3726_372624


namespace geometric_series_sum_l3726_372681

theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/7
  let S := ∑' n, a * r^n
  S = 7/6 := by sorry

end geometric_series_sum_l3726_372681


namespace number_of_products_l3726_372612

/-- Given fixed cost, marginal cost, and total cost, prove the number of products. -/
theorem number_of_products (fixed_cost marginal_cost total_cost : ℚ) (n : ℕ) : 
  fixed_cost = 12000 →
  marginal_cost = 200 →
  total_cost = 16000 →
  n = 20 := by
  sorry

end number_of_products_l3726_372612


namespace cricket_bat_price_l3726_372685

/-- Represents the cost and selling prices of an item -/
structure PriceData where
  cost_price_a : ℝ
  selling_price_b : ℝ
  selling_price_c : ℝ

/-- Theorem stating the relationship between the prices and profits -/
theorem cricket_bat_price (p : PriceData) 
  (profit_a : p.selling_price_b = 1.20 * p.cost_price_a)
  (profit_b : p.selling_price_c = 1.25 * p.selling_price_b)
  (final_price : p.selling_price_c = 222) :
  p.cost_price_a = 148 := by
  sorry

#check cricket_bat_price

end cricket_bat_price_l3726_372685


namespace fifth_month_sale_is_13562_l3726_372692

/-- The sale amount in the fifth month given the conditions of the problem -/
def fifth_month_sale (first_month : ℕ) (second_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (sixth_month : ℕ) (average : ℕ) : ℕ :=
  average * 6 - (first_month + second_month + third_month + fourth_month + sixth_month)

/-- Theorem stating that the fifth month sale is 13562 given the problem conditions -/
theorem fifth_month_sale_is_13562 :
  fifth_month_sale 6435 6927 6855 7230 5591 6600 = 13562 := by
  sorry

end fifth_month_sale_is_13562_l3726_372692


namespace cube_root_simplification_l3726_372680

theorem cube_root_simplification :
  ∃ (a b : ℕ+), (a.val : ℝ) * (b.val : ℝ)^(1/3 : ℝ) = (2^11 * 3^8 : ℝ)^(1/3 : ℝ) ∧ 
  a.val = 72 ∧ b.val = 36 := by
sorry

end cube_root_simplification_l3726_372680


namespace triangle_similarity_problem_l3726_372637

theorem triangle_similarity_problem (DC CB AD : ℝ) (h1 : DC = 13) (h2 : CB = 9) 
  (h3 : AD > 0) (h4 : (1/3) * AD + DC + CB = AD) : 
  ∃ FC : ℝ, FC = 40/3 := by
  sorry

end triangle_similarity_problem_l3726_372637


namespace expression_evaluation_l3726_372613

theorem expression_evaluation : (4^2 - 4) + (5^2 - 5) - (7^3 - 7) + (3^2 - 3) = -298 := by
  sorry

end expression_evaluation_l3726_372613


namespace system_solution_l3726_372641

theorem system_solution (x y : ℝ) :
  (2 * x + 3 * y = 14) → (x + 4 * y = 11) → (x - y = 3) := by
  sorry

end system_solution_l3726_372641


namespace equation_solutions_l3726_372677

theorem equation_solutions : 
  let f (x : ℝ) := 1/((x-1)*(x-2)) + 1/((x-2)*(x-3)) + 1/((x-3)*(x-4)) + 1/((x-4)*(x-5))
  ∀ x : ℝ, f x = 1/12 ↔ x = 12 ∨ x = -4.5 :=
by sorry

end equation_solutions_l3726_372677


namespace tv_selection_theorem_l3726_372655

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of Type A televisions -/
def typeA : ℕ := 4

/-- The number of Type B televisions -/
def typeB : ℕ := 5

/-- The total number of televisions to be chosen -/
def totalChosen : ℕ := 3

/-- The number of ways to choose the televisions -/
def waysToChoose : ℕ := choose typeA 2 * choose typeB 1 + choose typeA 1 * choose typeB 2

theorem tv_selection_theorem : waysToChoose = 70 := by sorry

end tv_selection_theorem_l3726_372655


namespace smallest_number_satisfying_conditions_l3726_372604

def satisfies_remainder_conditions (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 ∧
  n % 8 = 7 ∧ n % 9 = 8 ∧ n % 10 = 9 ∧ n % 11 = 10 ∧ n % 12 = 11

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → n % m ≠ 0

theorem smallest_number_satisfying_conditions :
  ∃ (n : ℕ),
    n = 10079 ∧
    satisfies_remainder_conditions n ∧
    ¬∃ (m : ℕ), m * m = n ∧
    is_prime (sum_of_digits n) ∧
    n > 10000 ∧
    ∀ (k : ℕ), 10000 < k ∧ k < n →
      ¬(satisfies_remainder_conditions k ∧
        ¬∃ (m : ℕ), m * m = k ∧
        is_prime (sum_of_digits k)) :=
by sorry

end smallest_number_satisfying_conditions_l3726_372604


namespace fraction_problem_l3726_372682

theorem fraction_problem (n : ℝ) (h : n = 180) : ∃ f : ℝ, f * (1/3 * 1/5 * n) + 6 = 1/15 * n ∧ f = 1/2 := by
  sorry

end fraction_problem_l3726_372682


namespace sqrt_equation_solution_l3726_372697

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end sqrt_equation_solution_l3726_372697


namespace first_term_of_geometric_series_l3726_372648

/-- The first term of an infinite geometric series with common ratio 1/4 and sum 80 is 60. -/
theorem first_term_of_geometric_series : ∀ (a : ℝ),
  (∑' n, a * (1/4)^n = 80) → a = 60 :=
by
  sorry

end first_term_of_geometric_series_l3726_372648


namespace work_rate_ratio_l3726_372619

/-- Given three workers with work rates, prove the ratio of combined work rates -/
theorem work_rate_ratio 
  (R₁ R₂ R₃ : ℝ) 
  (h₁ : R₂ + R₃ = 2 * R₁) 
  (h₂ : R₁ + R₃ = 3 * R₂) : 
  (R₁ + R₂) / R₃ = 7 / 5 := by
sorry

end work_rate_ratio_l3726_372619


namespace min_value_theorem_l3726_372662

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (2 / a) + (3 / b) ≥ 25 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ (2 / a₀) + (3 / b₀) = 25 :=
by sorry

end min_value_theorem_l3726_372662


namespace multiplicative_inverse_600_mod_3599_l3726_372686

theorem multiplicative_inverse_600_mod_3599 :
  ∃ (n : ℕ), n < 3599 ∧ (600 * n) % 3599 = 1 :=
by
  -- Define the right triangle
  let a : ℕ := 45
  let b : ℕ := 336
  let c : ℕ := 339
  
  -- Assert that a, b, c form a right triangle
  have right_triangle : a^2 + b^2 = c^2 := by sorry
  
  -- Define the multiplicative inverse
  let inverse : ℕ := 1200
  
  -- Prove that inverse is less than 3599
  have inverse_bound : inverse < 3599 := by sorry
  
  -- Prove that inverse is the multiplicative inverse of 600 modulo 3599
  have inverse_property : (600 * inverse) % 3599 = 1 := by sorry
  
  -- Combine the proofs
  exact ⟨inverse, inverse_bound, inverse_property⟩

#eval (600 * 1200) % 3599  -- Should output 1

end multiplicative_inverse_600_mod_3599_l3726_372686


namespace candy_distribution_l3726_372606

theorem candy_distribution (n : ℕ) : n > 0 → (100 % n = 0) → (99 % n = 0) → n = 11 := by
  sorry

end candy_distribution_l3726_372606


namespace f_monotonicity_l3726_372611

-- Define the function
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- State the theorem
theorem f_monotonicity :
  (∀ x y, x < y ∧ y < -1/3 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (∀ x y, -1/3 < x ∧ x < y ∧ y < 1 → f x > f y) := by
  sorry

end f_monotonicity_l3726_372611


namespace multiples_properties_l3726_372630

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a - b = 4 * p) := by
sorry

end multiples_properties_l3726_372630


namespace scientific_notation_correct_l3726_372688

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 2700000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 2.7
    exponent := 6
    property := by sorry }

/-- Theorem stating that the scientific notation representation is correct -/
theorem scientific_notation_correct :
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end scientific_notation_correct_l3726_372688


namespace julio_earnings_l3726_372639

/-- Calculates the total earnings for Julio over 3 weeks --/
def total_earnings (commission_rate : ℕ) (first_week_customers : ℕ) (salary : ℕ) (bonus : ℕ) : ℕ :=
  let second_week_customers := 2 * first_week_customers
  let third_week_customers := 3 * first_week_customers
  let total_customers := first_week_customers + second_week_customers + third_week_customers
  let commission := commission_rate * total_customers
  salary + commission + bonus

/-- Theorem stating that Julio's total earnings for 3 weeks is $760 --/
theorem julio_earnings : 
  total_earnings 1 35 500 50 = 760 := by
  sorry

end julio_earnings_l3726_372639


namespace complex_equation_solution_l3726_372607

theorem complex_equation_solution (z : ℂ) (m n : ℝ) : 
  (Complex.abs (1 - z) + z = 10 - 3 * Complex.I) →
  (z = 5 - 3 * Complex.I) ∧
  (z^2 + m * z + n = 1 - 3 * Complex.I) →
  (m = 14 ∧ n = -103) := by
sorry

end complex_equation_solution_l3726_372607


namespace fraction_multiple_l3726_372698

theorem fraction_multiple (numerator denominator : ℕ) : 
  denominator = 5 →
  numerator = denominator + 4 →
  (numerator + 6) / denominator = 3 := by
sorry

end fraction_multiple_l3726_372698


namespace set_operations_l3726_372683

open Set

def A : Set ℝ := {x | 2 * x - 8 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}

theorem set_operations :
  (A ∩ B = {x : ℝ | 0 < x ∧ x < 4}) ∧
  ((Aᶜ ∪ B) = {x : ℝ | 0 < x}) := by sorry

end set_operations_l3726_372683


namespace davantes_girl_friends_l3726_372628

def days_in_week : ℕ := 7

def davantes_friends (days : ℕ) : ℕ := 2 * days

def boy_friends : ℕ := 11

theorem davantes_girl_friends :
  davantes_friends days_in_week - boy_friends = 3 := by
  sorry

end davantes_girl_friends_l3726_372628


namespace volume_range_l3726_372650

/-- A rectangular prism with given surface area and sum of edge lengths -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  surface_area_eq : 2 * (a * b + b * c + a * c) = 48
  edge_sum_eq : 4 * (a + b + c) = 36

/-- The volume of a rectangular prism -/
def volume (p : RectangularPrism) : ℝ := p.a * p.b * p.c

/-- Theorem stating the range of possible volumes for the given rectangular prism -/
theorem volume_range (p : RectangularPrism) : 
  16 ≤ volume p ∧ volume p ≤ 20 :=
sorry

end volume_range_l3726_372650


namespace remainder_problem_l3726_372661

theorem remainder_problem (n : ℤ) (h : n % 18 = 10) : (2 * n) % 9 = 2 := by
  sorry

end remainder_problem_l3726_372661


namespace vector_sum_magnitude_l3726_372605

def a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def b (y : ℝ) : Fin 2 → ℝ := ![1, y]
def c : Fin 2 → ℝ := ![2, -4]

theorem vector_sum_magnitude : 
  ∃ (x y : ℝ), 
    (∀ i : Fin 2, (a x) i * c i = 0) ∧ 
    (∃ (k : ℝ), ∀ i : Fin 2, (b y) i = k * c i) →
    Real.sqrt ((a x 0 + b y 0)^2 + (a x 1 + b y 1)^2) = Real.sqrt 10 := by
  sorry

end vector_sum_magnitude_l3726_372605


namespace tony_rollercoasters_l3726_372699

/-- The number of rollercoasters Tony went on -/
def num_rollercoasters : ℕ := 5

/-- The speeds of the rollercoasters Tony went on -/
def rollercoaster_speeds : List ℝ := [50, 62, 73, 70, 40]

/-- The average speed of all rollercoasters Tony went on -/
def average_speed : ℝ := 59

/-- Theorem stating that the number of rollercoasters Tony went on is correct -/
theorem tony_rollercoasters :
  num_rollercoasters = rollercoaster_speeds.length ∧
  (rollercoaster_speeds.sum / num_rollercoasters : ℝ) = average_speed := by
  sorry

end tony_rollercoasters_l3726_372699


namespace cubic_inequality_l3726_372690

theorem cubic_inequality (x : ℝ) : x > 1 → 2 * x^3 > x^2 + 1 := by
  sorry

end cubic_inequality_l3726_372690


namespace sequence_problem_l3726_372623

-- Define arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_geom : is_geometric_sequence b)
  (h_eq : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
  (h_b7 : b 7 = a 7)
  (h_nonzero : ∀ n : ℕ, a n ≠ 0) :
  b 6 * b 8 = 16 := by
sorry

end sequence_problem_l3726_372623


namespace intersection_equality_implies_m_values_l3726_372653

def A (m : ℝ) : Set ℝ := {3, m^2}
def B (m : ℝ) : Set ℝ := {-1, 3, 3*m-2}

theorem intersection_equality_implies_m_values (m : ℝ) :
  A m ∩ B m = A m → m = 1 ∨ m = 2 := by
  sorry

end intersection_equality_implies_m_values_l3726_372653


namespace dice_throw_pigeonhole_l3726_372633

/-- Represents a throw of four fair six-sided dice -/
def DiceThrow := Fin 4 → Fin 6

/-- The sum of a dice throw -/
def throwSum (t : DiceThrow) : ℕ := (t 0).val + 1 + (t 1).val + 1 + (t 2).val + 1 + (t 3).val + 1

/-- A sequence of dice throws -/
def ThrowSequence (n : ℕ) := Fin n → DiceThrow

theorem dice_throw_pigeonhole :
  ∀ (s : ThrowSequence 22), ∃ (i j : Fin 22), i ≠ j ∧ throwSum (s i) = throwSum (s j) :=
sorry

end dice_throw_pigeonhole_l3726_372633


namespace remainder_property_l3726_372609

theorem remainder_property (N : ℤ) : ∃ (k : ℤ), N = 35 * k + 25 → ∃ (m : ℤ), N = 15 * m + 10 := by
  sorry

end remainder_property_l3726_372609


namespace function_equation_solution_l3726_372632

theorem function_equation_solution (f : ℚ → ℚ) 
  (h0 : f 0 = 0) 
  (h1 : ∀ x y : ℚ, f (f x + f y) = x + y) : 
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) := by
sorry

end function_equation_solution_l3726_372632


namespace triangle_inequality_incenter_l3726_372668

/-- Given a triangle ABC with sides a, b, c and a point P inside the triangle with distances
    r₁, r₂, r₃ to the sides respectively, prove that (a/r₁ + b/r₂ + c/r₃) ≥ (a + b + c)²/(2S),
    where S is the area of triangle ABC, and equality holds iff P is the incenter. -/
theorem triangle_inequality_incenter (a b c r₁ r₂ r₃ S : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ S > 0)
  (h_area : a * r₁ + b * r₂ + c * r₃ = 2 * S) :
  a / r₁ + b / r₂ + c / r₃ ≥ (a + b + c)^2 / (2 * S) ∧
  (a / r₁ + b / r₂ + c / r₃ = (a + b + c)^2 / (2 * S) ↔ r₁ = r₂ ∧ r₂ = r₃) :=
by sorry

end triangle_inequality_incenter_l3726_372668


namespace king_middle_school_teachers_l3726_372614

theorem king_middle_school_teachers (total_students : ℕ) 
  (classes_per_student : ℕ) (classes_per_teacher : ℕ) 
  (students_per_class : ℕ) :
  total_students = 1500 →
  classes_per_student = 6 →
  classes_per_teacher = 5 →
  students_per_class = 25 →
  (total_students * classes_per_student) / (students_per_class * classes_per_teacher) = 72 :=
by
  sorry

end king_middle_school_teachers_l3726_372614


namespace mrs_hilt_impressed_fans_l3726_372615

/-- The number of sets of bleachers -/
def num_bleachers : ℕ := 3

/-- The number of fans on each set of bleachers -/
def fans_per_bleacher : ℕ := 812

/-- The total number of fans Mrs. Hilt impressed -/
def total_fans : ℕ := num_bleachers * fans_per_bleacher

theorem mrs_hilt_impressed_fans : total_fans = 2436 := by
  sorry

end mrs_hilt_impressed_fans_l3726_372615


namespace johnny_emily_meeting_distance_l3726_372665

-- Define the total distance
def total_distance : ℝ := 60

-- Define walking rates
def matthew_rate : ℝ := 3
def johnny_rate : ℝ := 4
def emily_rate : ℝ := 5

-- Define the time difference between Matthew's start and Johnny/Emily's start
def time_diff : ℝ := 1

-- Define the function to calculate the distance Johnny walked
def johnny_distance (t : ℝ) : ℝ := johnny_rate * t

-- Theorem statement
theorem johnny_emily_meeting_distance :
  ∃ t : ℝ, t > 0 ∧ 
    matthew_rate * (t + time_diff) + johnny_distance t + emily_rate * t = total_distance ∧
    johnny_distance t = 19 := by
  sorry

end johnny_emily_meeting_distance_l3726_372665


namespace morgan_red_pens_l3726_372618

def total_pens : ℕ := 168
def blue_pens : ℕ := 45
def black_pens : ℕ := 58

def red_pens : ℕ := total_pens - (blue_pens + black_pens)

theorem morgan_red_pens : red_pens = 65 := by sorry

end morgan_red_pens_l3726_372618


namespace class_average_weight_l3726_372644

theorem class_average_weight (students_A : ℕ) (students_B : ℕ) 
  (avg_weight_A : ℝ) (avg_weight_B : ℝ) :
  students_A = 30 →
  students_B = 20 →
  avg_weight_A = 40 →
  avg_weight_B = 35 →
  let total_students := students_A + students_B
  let total_weight := students_A * avg_weight_A + students_B * avg_weight_B
  (total_weight / total_students : ℝ) = 38 := by
  sorry

end class_average_weight_l3726_372644


namespace travel_options_count_l3726_372675

/-- The number of train services from location A to location B -/
def num_train_services : ℕ := 3

/-- The number of ferry services from location B to location C -/
def num_ferry_services : ℕ := 2

/-- The total number of travel options from location A to location C -/
def total_travel_options : ℕ := num_train_services * num_ferry_services

theorem travel_options_count : total_travel_options = 6 := by
  sorry

end travel_options_count_l3726_372675


namespace khali_snow_volume_l3726_372656

/-- Calculates the total volume of snow to be shoveled given sidewalk dimensions and snow depths -/
def total_snow_volume (length width initial_depth additional_depth : ℚ) : ℚ :=
  length * width * (initial_depth + additional_depth)

/-- Proves that the total snow volume for Khali's sidewalk is 90 cubic feet -/
theorem khali_snow_volume :
  let length : ℚ := 30
  let width : ℚ := 3
  let initial_depth : ℚ := 3/4
  let additional_depth : ℚ := 1/4
  total_snow_volume length width initial_depth additional_depth = 90 := by
  sorry

end khali_snow_volume_l3726_372656


namespace star_property_l3726_372671

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.one
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.four
  | Element.two, Element.three => Element.three
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.one
  | Element.three, Element.three => Element.four
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.four

theorem star_property : 
  star (star Element.three Element.two) (star Element.two Element.one) = Element.four := by
  sorry

end star_property_l3726_372671


namespace shelly_money_l3726_372694

/-- Calculates the total amount of money Shelly has given the number of $10 and $5 bills -/
def total_money (ten_dollar_bills : ℕ) (five_dollar_bills : ℕ) : ℕ :=
  10 * ten_dollar_bills + 5 * five_dollar_bills

/-- Proves that Shelly has $130 given the conditions -/
theorem shelly_money : 
  let ten_dollar_bills : ℕ := 10
  let five_dollar_bills : ℕ := ten_dollar_bills - 4
  total_money ten_dollar_bills five_dollar_bills = 130 := by
sorry

end shelly_money_l3726_372694


namespace max_voters_is_five_l3726_372617

/-- Represents a movie rating system where:
    - Scores are integers from 0 to 10
    - The rating is the sum of scores divided by the number of voters
    - At moment T, the rating is an integer
    - After moment T, each new vote decreases the rating by one unit -/
structure MovieRating where
  scores : List ℤ
  rating_at_T : ℤ

/-- The maximum number of viewers who could have voted after moment T
    while maintaining the property that each new vote decreases the rating by one unit -/
def max_voters_after_T (mr : MovieRating) : ℕ :=
  sorry

/-- All scores are between 0 and 10 -/
axiom scores_range (mr : MovieRating) : ∀ s ∈ mr.scores, 0 ≤ s ∧ s ≤ 10

/-- The rating at moment T is the sum of scores divided by the number of voters -/
axiom rating_calculation (mr : MovieRating) :
  mr.rating_at_T = (mr.scores.sum / mr.scores.length : ℤ)

/-- After moment T, each new vote decreases the rating by exactly one unit -/
axiom rating_decrease (mr : MovieRating) (new_score : ℤ) :
  let new_rating := ((mr.scores.sum + new_score) / (mr.scores.length + 1) : ℤ)
  new_rating = mr.rating_at_T - 1

/-- The maximum number of viewers who could have voted after moment T is 5 -/
theorem max_voters_is_five (mr : MovieRating) :
  max_voters_after_T mr = 5 :=
sorry

end max_voters_is_five_l3726_372617


namespace intersection_A_complement_B_l3726_372670

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := {x : ℝ | x^2 - 2*x < 0}

def B : Set ℝ := {x : ℝ | x ≥ 1}

theorem intersection_A_complement_B : A ∩ (U \ B) = Ioo 0 1 := by sorry

end intersection_A_complement_B_l3726_372670


namespace bills_initial_money_l3726_372651

def total_initial_money : ℕ := 42
def num_pizzas : ℕ := 3
def pizza_cost : ℕ := 11
def bill_final_money : ℕ := 39

theorem bills_initial_money :
  let frank_spent := num_pizzas * pizza_cost
  let frank_leftover := total_initial_money - frank_spent
  let bill_initial := bill_final_money - frank_leftover
  bill_initial = 30 := by
sorry

end bills_initial_money_l3726_372651


namespace f_always_above_g_iff_m_less_than_5_l3726_372642

/-- The function f(x) = |x-2| -/
def f (x : ℝ) : ℝ := |x - 2|

/-- The function g(x) = -|x+3| + m -/
def g (x m : ℝ) : ℝ := -|x + 3| + m

/-- Theorem stating that f(x) > g(x) for all x if and only if m < 5 -/
theorem f_always_above_g_iff_m_less_than_5 :
  (∀ x : ℝ, f x > g x m) ↔ m < 5 := by sorry

end f_always_above_g_iff_m_less_than_5_l3726_372642


namespace grants_age_l3726_372610

theorem grants_age (hospital_age : ℕ) (grant_age : ℕ) : hospital_age = 40 →
  grant_age + 5 = (2 / 3) * (hospital_age + 5) →
  grant_age = 25 := by
  sorry

end grants_age_l3726_372610


namespace triangle_area_l3726_372646

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  b = 2 →
  c = 2 * Real.sqrt 2 →
  C = π / 4 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 + 1 :=
by sorry

end triangle_area_l3726_372646


namespace group_size_l3726_372621

theorem group_size (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : 
  average_increase = 6 → old_weight = 45 → new_weight = 93 → 
  (new_weight - old_weight) / average_increase = 8 := by
  sorry

end group_size_l3726_372621


namespace bumper_car_line_count_l3726_372645

/-- Calculates the final number of people in line for bumper cars after several changes --/
def final_line_count (initial : ℕ) (left1 left2 left3 joined1 joined2 joined3 : ℕ) : ℕ :=
  initial - left1 + joined1 - left2 + joined2 - left3 + joined3

/-- Theorem stating the final number of people in line for the given scenario --/
theorem bumper_car_line_count : 
  final_line_count 31 15 8 7 12 18 25 = 56 := by
  sorry

end bumper_car_line_count_l3726_372645


namespace count_integers_satisfying_inequality_l3726_372634

theorem count_integers_satisfying_inequality :
  (Finset.filter (fun n : ℤ => (n - 3) * (n + 2) * (n + 6) < 0)
    (Finset.Icc (-11 : ℤ) 11)).card = 12 := by
  sorry

end count_integers_satisfying_inequality_l3726_372634


namespace tan_product_equals_two_l3726_372673

theorem tan_product_equals_two (α β : Real) 
  (h1 : Real.sin α = 2 * Real.sin β) 
  (h2 : Real.sin (α + β) * Real.tan (α - β) = 1) : 
  Real.tan α * Real.tan β = 2 := by
  sorry

end tan_product_equals_two_l3726_372673


namespace problem_solution_l3726_372659

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

theorem problem_solution :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) ∧
  (∀ k : ℝ, {x : ℝ | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1} ⊆ A → k > 1) :=
by sorry

end problem_solution_l3726_372659


namespace max_NPM_value_l3726_372666

theorem max_NPM_value : 
  ∀ M : ℕ, 
  1 ≤ M ∧ M ≤ 9 →
  let MM := 10 * M + M
  let NPM := MM * M
  100 ≤ NPM ∧ NPM < 1000 →
  (∀ N P : ℕ, NPM = 100 * N + 10 * P + M → N < 10 ∧ P < 10) →
  NPM ≤ 891 :=
by sorry

end max_NPM_value_l3726_372666


namespace percy_christmas_money_l3726_372631

/-- The amount of money Percy received at Christmas -/
def christmas_money : ℝ :=
  let playstation_cost : ℝ := 500
  let birthday_money : ℝ := 200
  let game_price : ℝ := 7.5
  let games_sold : ℕ := 20
  playstation_cost - birthday_money - (game_price * games_sold)

theorem percy_christmas_money :
  christmas_money = 150 := by
  sorry

end percy_christmas_money_l3726_372631


namespace ellipse_properties_l3726_372678

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Properties of a specific ellipse -/
theorem ellipse_properties (C : Ellipse) (P : Point) :
  P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1 →  -- P is on the ellipse
  P.x = 1 →                           -- P's x-coordinate is 1
  P.y = Real.sqrt 2 / 2 →             -- P's y-coordinate is √2/2
  (∃ F₁ F₂ : Point, |P.x - F₁.x| + |P.y - F₁.y| + |P.x - F₂.x| + |P.y - F₂.y| = 2 * Real.sqrt 2) →  -- Distance sum to foci is 2√2
  (C.a^2 = 2 ∧ C.b^2 = 1) ∧           -- Standard equation of C is x²/2 + y² = 1
  (∃ (A B O : Point) (l : Set Point),
    O = ⟨0, 0⟩ ∧                      -- O is the origin
    F₂ ∈ l ∧ A ∈ l ∧ B ∈ l ∧          -- l passes through F₂, A, and B
    A.x^2 / C.a^2 + A.y^2 / C.b^2 = 1 ∧  -- A is on the ellipse
    B.x^2 / C.a^2 + B.y^2 / C.b^2 = 1 ∧  -- B is on the ellipse
    (∀ A' B' : Point,
      A' ∈ l → B' ∈ l →
      A'.x^2 / C.a^2 + A'.y^2 / C.b^2 = 1 →
      B'.x^2 / C.a^2 + B'.y^2 / C.b^2 = 1 →
      abs ((A.x - O.x) * (B.y - O.y) - (B.x - O.x) * (A.y - O.y)) / 2 ≥
      abs ((A'.x - O.x) * (B'.y - O.y) - (B'.x - O.x) * (A'.y - O.y)) / 2) ∧
    abs ((A.x - O.x) * (B.y - O.y) - (B.x - O.x) * (A.y - O.y)) / 2 = Real.sqrt 2 / 2) -- Max area of AOB is √2/2
  := by sorry

end ellipse_properties_l3726_372678


namespace congruence_mod_10_l3726_372658

theorem congruence_mod_10 : ∃ C : ℤ, (1 + C * (2^20 - 1)) % 10 = 2011 % 10 := by
  sorry

end congruence_mod_10_l3726_372658


namespace unequal_grandchildren_probability_l3726_372664

-- Define the number of grandchildren
def n : ℕ := 12

-- Define the probability of a child being male or female
def p : ℚ := 1/2

-- Define the probability of having an equal number of grandsons and granddaughters
def prob_equal : ℚ := (n.choose (n/2)) / (2^n)

-- Theorem statement
theorem unequal_grandchildren_probability :
  1 - prob_equal = 793/1024 := by sorry

end unequal_grandchildren_probability_l3726_372664


namespace sum_f_negative_l3726_372638

/-- A monotonically decreasing odd function. -/
def MonoDecreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x > f y) ∧ (∀ x, f (-x) = -f x)

theorem sum_f_negative
  (f : ℝ → ℝ)
  (h_f : MonoDecreasingOddFunction f)
  (x₁ x₂ x₃ : ℝ)
  (h₁₂ : x₁ + x₂ > 0)
  (h₂₃ : x₂ + x₃ > 0)
  (h₃₁ : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end sum_f_negative_l3726_372638


namespace z_in_third_quadrant_l3726_372663

def z : ℂ := (-8 + Complex.I) * Complex.I

theorem z_in_third_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 :=
sorry

end z_in_third_quadrant_l3726_372663


namespace reflection_line_equation_l3726_372689

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The reflection of a point over a line -/
def reflect (p : Point) (l : Line) : Point :=
  sorry

/-- The line of reflection given three points and their reflections -/
def reflectionLine (p q r p' q' r' : Point) : Line :=
  sorry

/-- Theorem stating that the line of reflection for the given points has the equation y = (3/5)x + 3/5 -/
theorem reflection_line_equation :
  let p := Point.mk (-2) 1
  let q := Point.mk 3 5
  let r := Point.mk 6 3
  let p' := Point.mk (-4) (-1)
  let q' := Point.mk 1 1
  let r' := Point.mk 4 (-1)
  let l := reflectionLine p q r p' q' r'
  l.slope = 3/5 ∧ l.intercept = 3/5 := by
  sorry

end reflection_line_equation_l3726_372689


namespace planted_fraction_for_specific_plot_l3726_372649

/-- Represents a right triangle plot with an unplanted square at the right angle --/
structure PlotWithUnplantedSquare where
  leg1 : ℝ
  leg2 : ℝ
  unplanted_square_side : ℝ
  shortest_distance_to_hypotenuse : ℝ

/-- Calculates the fraction of the plot that is planted --/
def planted_fraction (plot : PlotWithUnplantedSquare) : ℝ := by sorry

/-- Theorem stating the planted fraction for the given plot dimensions --/
theorem planted_fraction_for_specific_plot :
  let plot : PlotWithUnplantedSquare := {
    leg1 := 5,
    leg2 := 12,
    unplanted_square_side := 3 * 7 / 5,
    shortest_distance_to_hypotenuse := 3
  }
  planted_fraction plot = 412 / 1000 := by sorry

end planted_fraction_for_specific_plot_l3726_372649


namespace rectangle_length_decrease_l3726_372654

theorem rectangle_length_decrease (b : ℝ) (x : ℝ) : 
  2 * b = 33.333333333333336 →
  (2 * b - x) * (b + 4) = 2 * b^2 + 75 →
  x = 2.833333333333336 := by sorry

end rectangle_length_decrease_l3726_372654


namespace rays_grocery_bill_l3726_372616

def calculate_total_bill (hamburger_price : ℚ) (cracker_price : ℚ) (vegetable_price : ℚ) 
  (vegetable_quantity : ℕ) (cheese_price : ℚ) (chicken_price : ℚ) (cereal_price : ℚ) 
  (rewards_discount : ℚ) (meat_cheese_discount : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let discounted_hamburger := hamburger_price * (1 - rewards_discount)
  let discounted_crackers := cracker_price * (1 - rewards_discount)
  let discounted_vegetables := vegetable_price * (1 - rewards_discount) * vegetable_quantity
  let discounted_cheese := cheese_price * (1 - meat_cheese_discount)
  let discounted_chicken := chicken_price * (1 - meat_cheese_discount)
  let subtotal := discounted_hamburger + discounted_crackers + discounted_vegetables + 
                  discounted_cheese + discounted_chicken + cereal_price
  let total := subtotal * (1 + sales_tax_rate)
  total

theorem rays_grocery_bill : 
  calculate_total_bill 5 (7/2) 2 4 (7/2) (13/2) 4 (1/10) (1/20) (7/100) = (3035/100) := by
  sorry

end rays_grocery_bill_l3726_372616


namespace expected_rolls_in_non_leap_year_l3726_372672

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieOutcome
  | One
  | Two
  | Three
  | Four
  | Five
  | Six
  | Seven
  | Eight

/-- The probability of each outcome on a fair 8-sided die -/
def dieProbability : DieOutcome → ℚ
  | DieOutcome.One => 1/8
  | DieOutcome.Two => 1/8
  | DieOutcome.Three => 1/8
  | DieOutcome.Four => 1/8
  | DieOutcome.Five => 1/8
  | DieOutcome.Six => 1/8
  | DieOutcome.Seven => 1/8
  | DieOutcome.Eight => 1/8

/-- The expected number of rolls on a single day -/
noncomputable def expectedRollsPerDay : ℚ := 8/7

/-- The number of days in a non-leap year -/
def daysInNonLeapYear : ℕ := 365

/-- The theorem to prove -/
theorem expected_rolls_in_non_leap_year :
  (expectedRollsPerDay * daysInNonLeapYear : ℚ) = 417.14 := by
  sorry


end expected_rolls_in_non_leap_year_l3726_372672


namespace permutations_of_seven_distinct_objects_l3726_372636

theorem permutations_of_seven_distinct_objects : Nat.factorial 7 = 5040 := by
  sorry

end permutations_of_seven_distinct_objects_l3726_372636


namespace trigonometric_simplification_l3726_372691

theorem trigonometric_simplification (A : ℝ) (h : 0 < A ∧ A < π / 2) :
  (2 + 2 * (Real.cos A / Real.sin A) - 3 * (1 / Real.sin A)) *
  (3 + 2 * (Real.sin A / Real.cos A) + 1 / Real.cos A) = 11 := by
  sorry

end trigonometric_simplification_l3726_372691


namespace paint_house_theorem_l3726_372696

/-- Represents the time taken to paint a house given the number of people -/
def paint_time (people : ℝ) (hours : ℝ) : Prop :=
  people * hours = 5 * 10

theorem paint_house_theorem :
  paint_time 5 10 → paint_time 4 12.5 :=
by
  sorry

end paint_house_theorem_l3726_372696


namespace gcd_of_product_3000_not_15_l3726_372625

theorem gcd_of_product_3000_not_15 (a b c : ℕ+) : 
  a * b * c = 3000 → Nat.gcd (a:ℕ) (Nat.gcd (b:ℕ) (c:ℕ)) ≠ 15 := by
sorry

end gcd_of_product_3000_not_15_l3726_372625
