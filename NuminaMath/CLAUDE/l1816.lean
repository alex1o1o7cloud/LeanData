import Mathlib

namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l1816_181647

theorem parabola_point_x_coordinate 
  (P : ℝ × ℝ) 
  (h1 : (P.2)^2 = 4 * P.1) 
  (h2 : Real.sqrt ((P.1 - 1)^2 + P.2^2) = 10) : 
  P.1 = 9 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l1816_181647


namespace NUMINAMATH_CALUDE_convex_set_enclosure_l1816_181630

-- Define a convex set in 2D space
variable (Φ : Set (ℝ × ℝ))

-- Define the property of being convex
def IsConvex (S : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property of being centrally symmetric
def IsCentrallySymmetric (S : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property of one set enclosing another
def Encloses (S T : Set (ℝ × ℝ)) : Prop := sorry

-- Define the area of a set
noncomputable def Area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Define a triangle
def IsTriangle (S : Set (ℝ × ℝ)) : Prop := sorry

-- The main theorem
theorem convex_set_enclosure (h : IsConvex Φ) : 
  ∃ S : Set (ℝ × ℝ), 
    IsConvex S ∧ 
    IsCentrallySymmetric S ∧ 
    Encloses S Φ ∧ 
    Area S ≤ 2 * Area Φ ∧
    (IsTriangle Φ → Area S ≥ 2 * Area Φ) := by
  sorry

end NUMINAMATH_CALUDE_convex_set_enclosure_l1816_181630


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l1816_181673

-- Equation 1
theorem equation_one_solution (x : ℝ) : 
  (1 / 3) * x^2 = 2 ↔ x = Real.sqrt 6 ∨ x = -Real.sqrt 6 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) : 
  8 * (x - 1)^3 = -(27 / 8) ↔ x = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l1816_181673


namespace NUMINAMATH_CALUDE_pepperoni_coverage_l1816_181689

/-- Represents a circular pizza with pepperoni toppings -/
structure PepperoniPizza where
  pizza_diameter : ℝ
  pepperoni_count : ℕ
  pepperoni_across_diameter : ℕ

/-- Calculates the fraction of pizza covered by pepperoni -/
def fraction_covered (p : PepperoniPizza) : ℚ :=
  sorry

/-- Theorem stating the fraction of pizza covered by pepperoni -/
theorem pepperoni_coverage (p : PepperoniPizza) 
  (h1 : p.pizza_diameter = 18)
  (h2 : p.pepperoni_across_diameter = 9)
  (h3 : p.pepperoni_count = 40) : 
  fraction_covered p = 40 / 81 := by
  sorry

end NUMINAMATH_CALUDE_pepperoni_coverage_l1816_181689


namespace NUMINAMATH_CALUDE_quadratic_function_value_l1816_181654

-- Define the quadratic function f(x) = x² - ax + b
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 - a*x + b

-- State the theorem
theorem quadratic_function_value (a b : ℝ) :
  f a b 1 = -1 → f a b 2 = 2 → f a b (-4) = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l1816_181654


namespace NUMINAMATH_CALUDE_original_list_size_l1816_181685

/-- Given a list of integers, if appending 25 increases the mean by 3,
    and then appending -4 decreases the mean by 1.5,
    prove that the original list contained 4 integers. -/
theorem original_list_size (l : List Int) : 
  (((l.sum + 25) / (l.length + 1) : ℚ) = (l.sum / l.length : ℚ) + 3) →
  (((l.sum + 21) / (l.length + 2) : ℚ) = (l.sum / l.length : ℚ) + 1.5) →
  l.length = 4 := by
sorry


end NUMINAMATH_CALUDE_original_list_size_l1816_181685


namespace NUMINAMATH_CALUDE_carrots_thrown_out_l1816_181679

theorem carrots_thrown_out (initial_carrots : ℕ) (additional_carrots : ℕ) (remaining_carrots : ℕ) : 
  initial_carrots = 48 →
  additional_carrots = 15 →
  remaining_carrots = 52 →
  initial_carrots + additional_carrots - remaining_carrots = 11 := by
sorry

end NUMINAMATH_CALUDE_carrots_thrown_out_l1816_181679


namespace NUMINAMATH_CALUDE_cookie_problem_l1816_181627

theorem cookie_problem (frank mike millie : ℕ) : 
  frank = (mike / 2) - 3 →
  mike = 3 * millie →
  frank = 3 →
  millie = 4 := by
sorry

end NUMINAMATH_CALUDE_cookie_problem_l1816_181627


namespace NUMINAMATH_CALUDE_fraction_difference_equals_nine_twentieths_l1816_181670

theorem fraction_difference_equals_nine_twentieths :
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_nine_twentieths_l1816_181670


namespace NUMINAMATH_CALUDE_polynomial_square_b_value_l1816_181626

theorem polynomial_square_b_value (a b : ℚ) :
  (∃ p q : ℚ, ∀ x : ℚ, x^4 + 3*x^3 + x^2 + a*x + b = (x^2 + p*x + q)^2) →
  b = 25/64 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_square_b_value_l1816_181626


namespace NUMINAMATH_CALUDE_dispatch_plans_count_l1816_181675

theorem dispatch_plans_count : ∀ (n m k : ℕ),
  n = 6 → m = 4 → k = 2 →
  (Nat.choose n k) * (n - k) * (n - k - 1) = 180 :=
by sorry

end NUMINAMATH_CALUDE_dispatch_plans_count_l1816_181675


namespace NUMINAMATH_CALUDE_apple_cost_18_pounds_l1816_181677

/-- The cost of apples given a rate and a quantity -/
def apple_cost (rate_dollars : ℚ) (rate_pounds : ℚ) (quantity : ℚ) : ℚ :=
  (rate_dollars / rate_pounds) * quantity

/-- Theorem: The cost of 18 pounds of apples at a rate of 5 dollars per 6 pounds is 15 dollars -/
theorem apple_cost_18_pounds : apple_cost 5 6 18 = 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_18_pounds_l1816_181677


namespace NUMINAMATH_CALUDE_divided_right_triangle_area_ratio_l1816_181611

/-- A right triangle divided by lines parallel to its legs through a point on its hypotenuse -/
structure DividedRightTriangle where
  /-- The area of the square formed by the division -/
  square_area : ℝ
  /-- The area of the first small right triangle -/
  small_triangle1_area : ℝ
  /-- The area of the second small right triangle -/
  small_triangle2_area : ℝ
  /-- The square_area is positive -/
  square_area_pos : 0 < square_area

/-- The theorem stating the relationship between the areas -/
theorem divided_right_triangle_area_ratio
  (t : DividedRightTriangle)
  (m : ℝ)
  (h : t.small_triangle1_area = m * t.square_area) :
  t.small_triangle2_area = (1 / (4 * m)) * t.square_area :=
by sorry

end NUMINAMATH_CALUDE_divided_right_triangle_area_ratio_l1816_181611


namespace NUMINAMATH_CALUDE_william_final_napkins_l1816_181667

def initial_napkins : ℕ := 15
def olivia_napkins : ℕ := 10
def amelia_napkins : ℕ := 2 * olivia_napkins

theorem william_final_napkins :
  initial_napkins + olivia_napkins + amelia_napkins = 45 :=
by sorry

end NUMINAMATH_CALUDE_william_final_napkins_l1816_181667


namespace NUMINAMATH_CALUDE_f_expression_l1816_181683

-- Define the function f
def f : ℝ → ℝ := fun x => sorry

-- State the theorem
theorem f_expression : 
  (∀ x : ℝ, x ≥ 0 → f (Real.sqrt x + 1) = x + 3) →
  (∀ x : ℝ, x ≥ 0 → f (x + 1) = x^2 + 3) :=
by sorry

end NUMINAMATH_CALUDE_f_expression_l1816_181683


namespace NUMINAMATH_CALUDE_number_problem_l1816_181676

theorem number_problem (x : ℝ) : 50 + 5 * 12 / (x / 3) = 51 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1816_181676


namespace NUMINAMATH_CALUDE_fair_hair_percentage_l1816_181694

/-- Proves that if 32% of employees are women with fair hair and 40% of fair-haired employees are women, then 80% of employees have fair hair. -/
theorem fair_hair_percentage (total_employees : ℝ) (women_fair_hair : ℝ) (fair_hair : ℝ)
  (h1 : women_fair_hair = 0.32 * total_employees)
  (h2 : women_fair_hair = 0.40 * fair_hair) :
  fair_hair / total_employees = 0.80 := by
sorry

end NUMINAMATH_CALUDE_fair_hair_percentage_l1816_181694


namespace NUMINAMATH_CALUDE_seven_power_plus_one_prime_factors_l1816_181620

theorem seven_power_plus_one_prime_factors (n : ℕ) :
  ∃ (primes : Finset ℕ), 
    (∀ p ∈ primes, Nat.Prime p) ∧ 
    (primes.card ≥ 2 * n + 3) ∧ 
    ((primes.prod id) = 7^(7^n) + 1) :=
sorry

end NUMINAMATH_CALUDE_seven_power_plus_one_prime_factors_l1816_181620


namespace NUMINAMATH_CALUDE_incorrect_proposition_statement_l1816_181698

theorem incorrect_proposition_statement : 
  ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_proposition_statement_l1816_181698


namespace NUMINAMATH_CALUDE_election_votes_calculation_l1816_181614

theorem election_votes_calculation (winning_percentage : ℚ) (majority : ℕ) (total_votes : ℕ) : 
  winning_percentage = 60 / 100 →
  majority = 1300 →
  winning_percentage * total_votes = (total_votes / 2 + majority : ℚ) →
  total_votes = 6500 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l1816_181614


namespace NUMINAMATH_CALUDE_remainder_problem_l1816_181672

theorem remainder_problem (x : ℤ) : x % 9 = 2 → x % 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1816_181672


namespace NUMINAMATH_CALUDE_gcd_of_container_volumes_l1816_181693

theorem gcd_of_container_volumes : Nat.gcd 496 (Nat.gcd 403 (Nat.gcd 713 (Nat.gcd 824 1171))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_container_volumes_l1816_181693


namespace NUMINAMATH_CALUDE_investment_growth_l1816_181692

/-- The present value of an investment -/
def present_value : ℝ := 217474.41

/-- The future value of the investment -/
def future_value : ℝ := 600000

/-- The annual interest rate -/
def interest_rate : ℝ := 0.07

/-- The number of years for the investment -/
def years : ℕ := 15

/-- Theorem stating that the present value invested at the given interest rate
    for the specified number of years will result in the future value -/
theorem investment_growth (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ 
  |future_value - present_value * (1 + interest_rate) ^ years| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_growth_l1816_181692


namespace NUMINAMATH_CALUDE_distance_between_points_l1816_181641

/-- The distance between points (0,12) and (9,0) is 15 -/
theorem distance_between_points : Real.sqrt ((9 - 0)^2 + (0 - 12)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1816_181641


namespace NUMINAMATH_CALUDE_two_common_tangents_l1816_181658

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Definition of circle C₂ -/
def C₂ (x y r : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = r^2

/-- Theorem stating the condition for exactly two common tangent lines -/
theorem two_common_tangents (r : ℝ) :
  (r > 0) →
  (∃ (x y : ℝ), C₁ x y ∧ C₂ x y r) ↔ (Real.sqrt 5 - 2 < r ∧ r < Real.sqrt 5 + 2) :=
sorry

end NUMINAMATH_CALUDE_two_common_tangents_l1816_181658


namespace NUMINAMATH_CALUDE_marble_fraction_after_change_l1816_181680

theorem marble_fraction_after_change (total : ℚ) (h : total > 0) :
  let initial_blue := (2 / 3 : ℚ) * total
  let initial_red := total - initial_blue
  let new_red := 3 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_marble_fraction_after_change_l1816_181680


namespace NUMINAMATH_CALUDE_max_cookies_buyable_l1816_181634

theorem max_cookies_buyable (total_money : ℚ) (pack_price : ℚ) (cookies_per_pack : ℕ) : 
  total_money = 20.75 ∧ pack_price = 1.75 ∧ cookies_per_pack = 2 →
  ⌊total_money / pack_price⌋ * cookies_per_pack = 22 := by
sorry

end NUMINAMATH_CALUDE_max_cookies_buyable_l1816_181634


namespace NUMINAMATH_CALUDE_absolute_value_sum_lower_bound_l1816_181628

theorem absolute_value_sum_lower_bound :
  (∀ x : ℝ, |x + 2| + |x - 1| ≥ 3) ∧
  (∀ ε > 0, ∃ x : ℝ, |x + 2| + |x - 1| < 3 + ε) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_sum_lower_bound_l1816_181628


namespace NUMINAMATH_CALUDE_no_primes_in_factorial_range_l1816_181653

theorem no_primes_in_factorial_range (n : ℕ) (h : n > 1) :
  ∀ k, n! + 1 < k ∧ k < n! + n → ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_factorial_range_l1816_181653


namespace NUMINAMATH_CALUDE_ampersand_composition_l1816_181660

-- Define the & operation
def ampersand_right (y : ℤ) : ℤ := 9 - y

-- Define the & operation
def ampersand_left (y : ℤ) : ℤ := y - 9

-- Theorem to prove
theorem ampersand_composition : ampersand_left (ampersand_right 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_composition_l1816_181660


namespace NUMINAMATH_CALUDE_overall_profit_calculation_l1816_181668

/-- Calculates the overall profit from selling a refrigerator and a mobile phone -/
theorem overall_profit_calculation (refrigerator_cost mobile_cost : ℕ)
  (refrigerator_loss_percent mobile_profit_percent : ℚ)
  (h1 : refrigerator_cost = 15000)
  (h2 : mobile_cost = 8000)
  (h3 : refrigerator_loss_percent = 4 / 100)
  (h4 : mobile_profit_percent = 10 / 100) :
  (refrigerator_cost * (1 - refrigerator_loss_percent) +
   mobile_cost * (1 + mobile_profit_percent) -
   (refrigerator_cost + mobile_cost)).floor = 200 :=
by sorry

end NUMINAMATH_CALUDE_overall_profit_calculation_l1816_181668


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_ratio_bound_l1816_181635

theorem binomial_coefficient_sum_ratio_bound (n : ℕ+) :
  let a := 2^(n : ℕ)
  let b := 4^(n : ℕ)
  (b / a) + (a / b) ≥ (5 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_ratio_bound_l1816_181635


namespace NUMINAMATH_CALUDE_petyas_number_l1816_181639

theorem petyas_number (x : ℝ) : x - x / 10 = 19.71 → x = 21.9 := by
  sorry

end NUMINAMATH_CALUDE_petyas_number_l1816_181639


namespace NUMINAMATH_CALUDE_trapezoid_area_between_triangles_l1816_181643

/-- Given two concentric equilateral triangles with areas 25 and 4 square units respectively,
    prove that the area of one of the four congruent trapezoids formed between them is 5.25 square units. -/
theorem trapezoid_area_between_triangles
  (outer_area : ℝ) (inner_area : ℝ) (num_trapezoids : ℕ)
  (h_outer : outer_area = 25)
  (h_inner : inner_area = 4)
  (h_num : num_trapezoids = 4) :
  (outer_area - inner_area) / num_trapezoids = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_between_triangles_l1816_181643


namespace NUMINAMATH_CALUDE_min_relevant_number_l1816_181625

def A (n : ℕ) := Finset.range (2*n + 1) \ {0}

def is_relevant_number (n m : ℕ) : Prop :=
  n ≥ 2 ∧ m ≥ 4 ∧
  ∀ (P : Finset ℕ), P ⊆ A n → P.card = m →
    ∃ (a b c d : ℕ), a ∈ P ∧ b ∈ P ∧ c ∈ P ∧ d ∈ P ∧ a + b + c + d = 4*n + 1

theorem min_relevant_number (n : ℕ) :
  n ≥ 2 → (∃ (m : ℕ), is_relevant_number n m) →
  ∃ (m : ℕ), is_relevant_number n m ∧ ∀ (k : ℕ), is_relevant_number n k → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_min_relevant_number_l1816_181625


namespace NUMINAMATH_CALUDE_n_good_lower_bound_two_is_seven_good_l1816_181699

/-- A tournament between n players where each player plays against every other player once --/
structure Tournament (n : ℕ) where
  result : Fin n → Fin n → Bool
  irreflexive : ∀ i, result i i = false
  antisymmetric : ∀ i j, result i j = !result j i

/-- A number k is n-good if there exists a tournament where for any k players, 
    there is another player who has lost to all of them --/
def is_n_good (n k : ℕ) : Prop :=
  ∃ t : Tournament n, ∀ (s : Finset (Fin n)) (hs : s.card = k),
    ∃ p : Fin n, p ∉ s ∧ ∀ q ∈ s, t.result q p = true

/-- The main theorem: For any n-good number k, n ≥ 2^(k+1) - 1 --/
theorem n_good_lower_bound (n k : ℕ) (h : is_n_good n k) : n ≥ 2^(k+1) - 1 :=
  sorry

/-- The smallest n for which 2 is n-good is 7 --/
theorem two_is_seven_good : 
  (is_n_good 7 2) ∧ (∀ m < 7, ¬ is_n_good m 2) :=
  sorry

end NUMINAMATH_CALUDE_n_good_lower_bound_two_is_seven_good_l1816_181699


namespace NUMINAMATH_CALUDE_similar_right_triangles_l1816_181663

theorem similar_right_triangles (y : ℝ) : 
  y > 0 →  -- ensure y is positive
  (16 : ℝ) / y = 12 / 9 → 
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_similar_right_triangles_l1816_181663


namespace NUMINAMATH_CALUDE_remainder_x_105_divided_by_x_plus_1_4_l1816_181623

theorem remainder_x_105_divided_by_x_plus_1_4 (x : ℤ) :
  x^105 ≡ 195300*x^3 + 580440*x^2 + 576085*x + 189944 [ZMOD (x + 1)^4] := by
  sorry

end NUMINAMATH_CALUDE_remainder_x_105_divided_by_x_plus_1_4_l1816_181623


namespace NUMINAMATH_CALUDE_cycle_price_proof_l1816_181638

/-- Proves that a cycle sold at a 5% loss for 1330 had an original price of 1400 -/
theorem cycle_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1330)
  (h2 : loss_percentage = 5) : 
  ∃ original_price : ℝ, 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1400 := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_proof_l1816_181638


namespace NUMINAMATH_CALUDE_buffalo_count_is_two_l1816_181615

/-- Represents the number of animals seen on each day of Erica's safari --/
structure SafariCount where
  saturday : ℕ
  sunday_leopards : ℕ
  sunday_buffaloes : ℕ
  monday : ℕ

/-- The total number of animals seen during the safari --/
def total_animals : ℕ := 20

/-- The actual count of animals seen on each day --/
def safari_count : SafariCount where
  saturday := 5  -- 3 lions + 2 elephants
  sunday_leopards := 5
  sunday_buffaloes := 2  -- This is what we want to prove
  monday := 8  -- 5 rhinos + 3 warthogs

theorem buffalo_count_is_two :
  safari_count.sunday_buffaloes = 2 :=
by
  sorry

#check buffalo_count_is_two

end NUMINAMATH_CALUDE_buffalo_count_is_two_l1816_181615


namespace NUMINAMATH_CALUDE_enclosed_area_theorem_l1816_181608

noncomputable def g (x : ℝ) : ℝ := 2 - Real.sqrt (1 - (2*x/3)^2)

def domain : Set ℝ := Set.Icc (-3/2) (3/2)

theorem enclosed_area_theorem (A : ℝ) :
  A = 2 * (π * (3/2)^2 / 2 - ∫ x in (Set.Icc 0 (3/2)), g x) :=
sorry

end NUMINAMATH_CALUDE_enclosed_area_theorem_l1816_181608


namespace NUMINAMATH_CALUDE_square_area_doubled_l1816_181601

theorem square_area_doubled (a : ℝ) (ha : a > 0) :
  (Real.sqrt 2 * a)^2 = 2 * a^2 := by sorry

end NUMINAMATH_CALUDE_square_area_doubled_l1816_181601


namespace NUMINAMATH_CALUDE_expression_simplification_l1816_181648

theorem expression_simplification (a b c : ℝ) :
  3 / 4 * (6 * a^2 - 12 * a) - 8 / 5 * (3 * b^2 + 15 * b) + (2 * c^2 - 6 * c) / 6 =
  (9/2) * a^2 - 9 * a - (24/5) * b^2 - 24 * b + (1/3) * c^2 - c :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1816_181648


namespace NUMINAMATH_CALUDE_trapezoid_construction_possible_l1816_181645

/-- Represents a trapezoid with sides a, b, c, d and diagonals d₁, d₂ -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  d₁ : ℝ
  d₂ : ℝ
  h_parallel : c = d
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ d₁ > 0 ∧ d₂ > 0
  h_inequality₁ : d₁ - d₂ < a + b
  h_inequality₂ : a + b < d₁ + d₂

/-- A trapezoid can be constructed given parallel sides and diagonals satisfying certain conditions -/
theorem trapezoid_construction_possible (a b c d d₁ d₂ : ℝ) 
  (h_parallel : c = d)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ d₁ > 0 ∧ d₂ > 0)
  (h_inequality₁ : d₁ - d₂ < a + b)
  (h_inequality₂ : a + b < d₁ + d₂) :
  ∃ t : Trapezoid, t.a = a ∧ t.b = b ∧ t.c = c ∧ t.d = d ∧ t.d₁ = d₁ ∧ t.d₂ = d₂ :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_construction_possible_l1816_181645


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1816_181618

theorem smaller_number_problem (x y : ℝ) : 
  y = 3 * x → x + y = 124 → x = 31 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1816_181618


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l1816_181681

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l1816_181681


namespace NUMINAMATH_CALUDE_second_number_value_l1816_181697

theorem second_number_value (A B C : ℝ) 
  (sum_eq : A + B + C = 157.5)
  (ratio_AB : A / B = 3.5 / 4.25)
  (ratio_BC : B / C = 7.5 / 11.25)
  (diff_AC : A - C = 12.75) :
  B = 18.75 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l1816_181697


namespace NUMINAMATH_CALUDE_airplane_seats_l1816_181695

theorem airplane_seats : ∃ (s : ℝ), 
  (30 : ℝ) + 0.2 * s + 0.75 * s = s ∧ s = 600 := by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_l1816_181695


namespace NUMINAMATH_CALUDE_range_of_f_l1816_181678

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem statement
theorem range_of_f :
  ∃ (a b : ℝ), a = -2 ∧ b = 2 ∧
  (∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ f x = y) ↔ a ≤ y ∧ y ≤ b) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1816_181678


namespace NUMINAMATH_CALUDE_log_meaningful_range_l1816_181652

/-- The range of real number a for which log_(a-1)(5-a) is meaningful -/
def meaningful_log_range : Set ℝ :=
  {a : ℝ | a ∈ Set.Ioo 1 2 ∪ Set.Ioo 2 5}

theorem log_meaningful_range :
  ∀ a : ℝ, (∃ x : ℝ, (a - 1) ^ x = 5 - a) ↔ a ∈ meaningful_log_range := by
  sorry

end NUMINAMATH_CALUDE_log_meaningful_range_l1816_181652


namespace NUMINAMATH_CALUDE_lcm_14_21_35_l1816_181671

theorem lcm_14_21_35 : Nat.lcm 14 (Nat.lcm 21 35) = 210 := by sorry

end NUMINAMATH_CALUDE_lcm_14_21_35_l1816_181671


namespace NUMINAMATH_CALUDE_leftHandedWomenPercentage_l1816_181624

/-- Represents the population of Smithtown -/
structure Population where
  rightHanded : ℕ
  leftHanded : ℕ
  men : ℕ
  women : ℕ

/-- Conditions for a valid Smithtown population -/
def isValidPopulation (p : Population) : Prop :=
  p.rightHanded = 3 * p.leftHanded ∧
  p.men = 3 * p.women / 2 ∧
  p.rightHanded + p.leftHanded = p.men + p.women

/-- A population with maximized right-handed men -/
def hasMaximizedRightHandedMen (p : Population) : Prop :=
  p.men = p.rightHanded

/-- Theorem: In a valid Smithtown population with maximized right-handed men,
    left-handed women constitute 25% of the total population -/
theorem leftHandedWomenPercentage (p : Population) 
  (hValid : isValidPopulation p) 
  (hMax : hasMaximizedRightHandedMen p) : 
  (p.leftHanded : ℚ) / (p.rightHanded + p.leftHanded : ℚ) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_leftHandedWomenPercentage_l1816_181624


namespace NUMINAMATH_CALUDE_minimum_training_months_l1816_181633

/-- The distance of a marathon in miles -/
def marathonDistance : ℝ := 26.3

/-- The initial running distance in miles -/
def initialDistance : ℝ := 3

/-- The function that calculates the running distance after a given number of months -/
def runningDistance (months : ℕ) : ℝ :=
  initialDistance * (2 ^ months)

/-- The theorem stating that 5 months is the minimum number of months needed to run a marathon -/
theorem minimum_training_months :
  (∀ m : ℕ, m < 5 → runningDistance m < marathonDistance) ∧
  (runningDistance 5 ≥ marathonDistance) := by
  sorry

#check minimum_training_months

end NUMINAMATH_CALUDE_minimum_training_months_l1816_181633


namespace NUMINAMATH_CALUDE_camping_group_solution_l1816_181659

/-- Represents the camping group -/
structure CampingGroup where
  initialTotal : ℕ
  initialGirls : ℕ

/-- Conditions of the camping group problem -/
class CampingGroupProblem (g : CampingGroup) where
  initial_ratio : g.initialGirls = g.initialTotal / 2
  final_ratio : (g.initialGirls + 1) * 10 = 6 * (g.initialTotal - 2)

/-- The theorem stating the solution to the camping group problem -/
theorem camping_group_solution (g : CampingGroup) [CampingGroupProblem g] : 
  g.initialGirls = 11 := by
  sorry

#check camping_group_solution

end NUMINAMATH_CALUDE_camping_group_solution_l1816_181659


namespace NUMINAMATH_CALUDE_coordinate_and_vector_problem_l1816_181610

-- Define the points and vectors
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-2, 1)  -- Calculated from |OB| = √5 and x = -2
def O : ℝ × ℝ := (0, 0)

-- Define the rotation function
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)

-- Define the vector OP
def OP : ℝ × ℝ := (2, 6)  -- Calculated from |OP| = 2√10 and cos θ = √10/10

-- Define the theorem
theorem coordinate_and_vector_problem :
  let C := rotate90Clockwise (B.1 - O.1, B.2 - O.2)
  let x := ((OP.1 * B.2) - (OP.2 * B.1)) / ((A.1 * B.2) - (A.2 * B.1))
  let y := ((OP.1 * A.2) - (OP.2 * A.1)) / ((B.1 * A.2) - (B.2 * A.1))
  C = (1, 2) ∧ x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_and_vector_problem_l1816_181610


namespace NUMINAMATH_CALUDE_half_area_to_longest_side_l1816_181602

/-- Represents a parallelogram field with given dimensions and angles -/
structure ParallelogramField where
  side1 : Real
  side2 : Real
  angle1 : Real
  angle2 : Real

/-- Calculates the fraction of the area closer to the longest side of the parallelogram field -/
def fraction_to_longest_side (field : ParallelogramField) : Real :=
  sorry

/-- Theorem stating that for a parallelogram field with specific dimensions,
    the fraction of the area closer to the longest side is 1/2 -/
theorem half_area_to_longest_side :
  let field : ParallelogramField := {
    side1 := 120,
    side2 := 80,
    angle1 := π / 3,  -- 60 degrees in radians
    angle2 := 2 * π / 3  -- 120 degrees in radians
  }
  fraction_to_longest_side field = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_half_area_to_longest_side_l1816_181602


namespace NUMINAMATH_CALUDE_floor_sqrt_equality_l1816_181684

theorem floor_sqrt_equality (n : ℕ+) :
  ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ ∧
  ⌊Real.sqrt (4 * n + 3)⌋ = ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ :=
by sorry

end NUMINAMATH_CALUDE_floor_sqrt_equality_l1816_181684


namespace NUMINAMATH_CALUDE_g_of_5_l1816_181622

def g (x : ℝ) : ℝ := 3*x^5 - 15*x^4 + 30*x^3 - 45*x^2 + 24*x + 50

theorem g_of_5 : g 5 = 2795 := by sorry

end NUMINAMATH_CALUDE_g_of_5_l1816_181622


namespace NUMINAMATH_CALUDE_scaled_tile_height_l1816_181600

/-- Calculates the new height of a proportionally scaled tile -/
def new_height (original_width original_height new_width : ℚ) : ℚ :=
  (new_width / original_width) * original_height

/-- Theorem: The new height of the scaled tile is 16 inches -/
theorem scaled_tile_height :
  let original_width : ℚ := 3
  let original_height : ℚ := 4
  let new_width : ℚ := 12
  new_height original_width original_height new_width = 16 := by
sorry

end NUMINAMATH_CALUDE_scaled_tile_height_l1816_181600


namespace NUMINAMATH_CALUDE_base_10_500_equals_base_6_2152_l1816_181686

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 6 to a natural number -/
def fromBase6 (digits : List ℕ) : ℕ :=
  sorry

theorem base_10_500_equals_base_6_2152 :
  toBase6 500 = [2, 1, 5, 2] ∧ fromBase6 [2, 1, 5, 2] = 500 :=
sorry

end NUMINAMATH_CALUDE_base_10_500_equals_base_6_2152_l1816_181686


namespace NUMINAMATH_CALUDE_solve_equation_l1816_181650

theorem solve_equation : ∃ x : ℝ, 3 * x - 6 = |(-23 + 5)|^2 ∧ x = 110 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1816_181650


namespace NUMINAMATH_CALUDE_sum_of_two_equals_third_l1816_181688

theorem sum_of_two_equals_third (a b c x y : ℝ) 
  (h1 : (a + x)⁻¹ = 6)
  (h2 : (b + y)⁻¹ = 3)
  (h3 : (c + x + y)⁻¹ = 2) : 
  c = a + b := by sorry

end NUMINAMATH_CALUDE_sum_of_two_equals_third_l1816_181688


namespace NUMINAMATH_CALUDE_unique_divisible_number_l1816_181644

def number (d : Nat) : Nat := 62684400 + d * 10

theorem unique_divisible_number :
  ∃! d : Nat, d < 10 ∧ (number d).mod 8 = 0 ∧ (number d).mod 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_number_l1816_181644


namespace NUMINAMATH_CALUDE_coin_array_problem_l1816_181619

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The problem statement -/
theorem coin_array_problem : 
  ∃ (N : ℕ), triangular_sum N = 5050 ∧ sum_of_digits N = 1 :=
sorry

end NUMINAMATH_CALUDE_coin_array_problem_l1816_181619


namespace NUMINAMATH_CALUDE_triangle_special_sequence_l1816_181613

theorem triangle_special_sequence (A B C : ℝ) (a b c : ℝ) : 
  -- Angles form an arithmetic sequence
  ∃ (α d : ℝ), A = α ∧ B = α + d ∧ C = α + 2*d ∧
  -- Sum of angles is π
  A + B + C = π ∧
  -- Reciprocals of sides form an arithmetic sequence
  2 * (1/b) = 1/a + 1/c ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Conclusion: all angles are π/3
  A = π/3 ∧ B = π/3 ∧ C = π/3 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_sequence_l1816_181613


namespace NUMINAMATH_CALUDE_cell_division_after_three_hours_l1816_181687

/-- Represents the number of cells after a given number of 30-minute periods -/
def cells_after_periods (n : ℕ) : ℕ := 2^n

/-- Represents the number of 30-minute periods in a given number of hours -/
def periods_in_hours (hours : ℕ) : ℕ := 2 * hours

theorem cell_division_after_three_hours :
  cells_after_periods (periods_in_hours 3) = 64 := by
  sorry

#eval cells_after_periods (periods_in_hours 3)

end NUMINAMATH_CALUDE_cell_division_after_three_hours_l1816_181687


namespace NUMINAMATH_CALUDE_food_distribution_l1816_181662

theorem food_distribution (initial_men : ℕ) (initial_days : ℕ) (additional_men : ℝ) (remaining_days : ℕ) :
  initial_men = 760 →
  initial_days = 22 →
  additional_men = 134.11764705882354 →
  remaining_days = 17 →
  ∃ (x : ℝ),
    x = 2 ∧
    (initial_men : ℝ) * (initial_days : ℝ) = 
      (initial_men : ℝ) * x + ((initial_men : ℝ) + additional_men) * (remaining_days : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_food_distribution_l1816_181662


namespace NUMINAMATH_CALUDE_circle_area_equality_l1816_181649

theorem circle_area_equality (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  let r : Real := 1  -- Assuming unit circle for simplicity
  let sector_area := θ * r^2
  let triangle_area := (r^2 * Real.tan θ * Real.tan (2 * θ)) / 2
  let circle_area := π * r^2
  triangle_area = circle_area - sector_area ↔ 2 * θ = Real.tan θ * Real.tan (2 * θ) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_equality_l1816_181649


namespace NUMINAMATH_CALUDE_no_real_solutions_l1816_181607

theorem no_real_solutions :
  ¬∃ (z : ℝ), (3*z - 9*z + 27)^2 + 4 = -2*(abs z) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1816_181607


namespace NUMINAMATH_CALUDE_units_digit_of_p_l1816_181669

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_p (p : ℤ) : 
  (0 < units_digit p) → 
  (units_digit (p^3) - units_digit (p^2) = 0) →
  (units_digit (p + 5) = 1) →
  units_digit p = 6 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_p_l1816_181669


namespace NUMINAMATH_CALUDE_complex_number_range_l1816_181640

variable (z : ℂ) (a : ℝ)

theorem complex_number_range :
  (∃ (r : ℝ), z + 2*Complex.I = r) →
  (∃ (s : ℝ), z / (2 - Complex.I) = s) →
  (Complex.re ((z + a*Complex.I)^2) > 0) →
  (Complex.im ((z + a*Complex.I)^2) > 0) →
  2 < a ∧ a < 4 := by
sorry

end NUMINAMATH_CALUDE_complex_number_range_l1816_181640


namespace NUMINAMATH_CALUDE_cos_300_deg_l1816_181631

/-- Cosine of 300 degrees is equal to 1/2 -/
theorem cos_300_deg : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_deg_l1816_181631


namespace NUMINAMATH_CALUDE_library_books_count_l1816_181690

/-- The number of shelves in the library -/
def num_shelves : ℕ := 1780

/-- The number of books each shelf can hold -/
def books_per_shelf : ℕ := 8

/-- The total number of books in the library -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem library_books_count : total_books = 14240 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l1816_181690


namespace NUMINAMATH_CALUDE_total_books_on_shelves_l1816_181682

/-- The number of book shelves -/
def num_shelves : ℕ := 150

/-- The number of books per shelf -/
def books_per_shelf : ℕ := 15

/-- The total number of books on all shelves -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem total_books_on_shelves :
  total_books = 2250 :=
by sorry

end NUMINAMATH_CALUDE_total_books_on_shelves_l1816_181682


namespace NUMINAMATH_CALUDE_money_division_l1816_181604

theorem money_division (a b c : ℝ) : 
  a = (1/3) * (b + c) →
  b = (2/7) * (a + c) →
  a = b + 15 →
  a + b + c = 540 :=
by
  sorry

end NUMINAMATH_CALUDE_money_division_l1816_181604


namespace NUMINAMATH_CALUDE_train_tunnel_time_l1816_181655

/-- The time taken for a train to pass through a tunnel -/
theorem train_tunnel_time (train_length : ℝ) (train_speed_kmh : ℝ) (tunnel_length : ℝ) : 
  train_length = 100 →
  train_speed_kmh = 72 →
  tunnel_length = 1.1 →
  (train_length + tunnel_length * 1000) / (train_speed_kmh * 1000 / 3600) / 60 = 1 := by
  sorry

end NUMINAMATH_CALUDE_train_tunnel_time_l1816_181655


namespace NUMINAMATH_CALUDE_evaluate_expression_l1816_181603

theorem evaluate_expression (x z : ℝ) (hx : x = 5) (hz : z = 4) :
  z^2 * (z^2 - 4*x) = -64 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1816_181603


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_count_l1816_181636

/-- Calculates the number of right-handed players in a cricket team -/
def right_handed_players (total_players throwers : ℕ) : ℕ :=
  let non_throwers := total_players - throwers
  let left_handed_non_throwers := non_throwers / 3
  let right_handed_non_throwers := non_throwers - left_handed_non_throwers
  throwers + right_handed_non_throwers

theorem cricket_team_right_handed_count :
  right_handed_players 70 37 = 59 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_count_l1816_181636


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1816_181661

def S (n : ℕ) : ℝ := 3 * n^2 - 2 * n

def a (n : ℕ) : ℝ := S n - S (n-1)

theorem arithmetic_sequence_proof :
  ∃ (d : ℝ), ∀ (n : ℕ), n ≥ 1 → a n = a 1 + (n - 1) * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1816_181661


namespace NUMINAMATH_CALUDE_system_solution_l1816_181629

theorem system_solution :
  ∃ (x y : ℝ), (1/2 * x - 3/2 * y = -1) ∧ (2 * x + y = 3) ∧ (x = 1) ∧ (y = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1816_181629


namespace NUMINAMATH_CALUDE_meeting_attendees_l1816_181609

/-- The number of handshakes in a meeting where every two people shake hands. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: There were 12 people in the meeting given the conditions. -/
theorem meeting_attendees : ∃ (n : ℕ), n > 0 ∧ handshakes n = 66 ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_meeting_attendees_l1816_181609


namespace NUMINAMATH_CALUDE_power_multiplication_l1816_181665

theorem power_multiplication (n : ℕ) : n * (n^(n - 1)) = n^n :=
by
  sorry

#check power_multiplication 3000

end NUMINAMATH_CALUDE_power_multiplication_l1816_181665


namespace NUMINAMATH_CALUDE_orchid_bushes_total_park_orchid_bushes_l1816_181656

/-- The total number of orchid bushes after planting is equal to the sum of the current number of bushes and the number of bushes planted over two days. -/
theorem orchid_bushes_total (current : ℕ) (today : ℕ) (tomorrow : ℕ) :
  current + today + tomorrow = current + today + tomorrow :=
by sorry

/-- Given the specific numbers from the problem -/
theorem park_orchid_bushes :
  let current : ℕ := 47
  let today : ℕ := 37
  let tomorrow : ℕ := 25
  current + today + tomorrow = 109 :=
by sorry

end NUMINAMATH_CALUDE_orchid_bushes_total_park_orchid_bushes_l1816_181656


namespace NUMINAMATH_CALUDE_circular_fields_radius_l1816_181637

theorem circular_fields_radius (r₁ r₂ : ℝ) : 
  r₂ = 10 →
  π * r₁^2 = 0.09 * (π * r₂^2) →
  r₁ = 3 := by
sorry

end NUMINAMATH_CALUDE_circular_fields_radius_l1816_181637


namespace NUMINAMATH_CALUDE_correct_matching_probability_l1816_181657

theorem correct_matching_probability (n : ℕ) (hn : n = 4) :
  (1 : ℚ) / (n.factorial : ℚ) = 1 / 24 :=
sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l1816_181657


namespace NUMINAMATH_CALUDE_number_difference_l1816_181617

theorem number_difference (a b : ℕ) : 
  a + b = 25800 →
  ∃ k : ℕ, b = 12 * k →
  a = k →
  b - a = 21824 :=
by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1816_181617


namespace NUMINAMATH_CALUDE_room_width_proof_l1816_181612

/-- Proves that a rectangular room with given dimensions has a specific width -/
theorem room_width_proof (room_length : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_length = 19 →
  veranda_width = 2 →
  veranda_area = 140 →
  ∃ (room_width : ℝ),
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) -
    room_length * room_width = veranda_area ∧
    room_width = 12 := by
  sorry

end NUMINAMATH_CALUDE_room_width_proof_l1816_181612


namespace NUMINAMATH_CALUDE_linear_equation_and_expression_l1816_181646

theorem linear_equation_and_expression (a : ℝ) : 
  (∀ x, (a - 1) * x^(|a|) - 3 = 0 → (a - 1) * x - 3 = 0) ∧ (a - 1 ≠ 0) →
  a = -1 ∧ -4 * a^2 - 2 * (a - (2 * a^2 - a + 2)) = 8 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_and_expression_l1816_181646


namespace NUMINAMATH_CALUDE_parabola_directrix_l1816_181606

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y = (x^2 - 8*x + 12) / 16

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop :=
  y = -5/4

/-- Theorem stating that the given directrix equation is correct for the given parabola -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_d : ℝ, directrix_equation y_d ∧ 
  y_d = -5/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1816_181606


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l1816_181621

theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are sides opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given conditions
  a = Real.sqrt 3 →
  Real.sin B = 1/2 →
  C = π/6 →
  -- Conclusion
  b = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l1816_181621


namespace NUMINAMATH_CALUDE_binomial_n_n_minus_3_l1816_181616

theorem binomial_n_n_minus_3 (n : ℕ) (h : n ≥ 3) :
  Nat.choose n (n - 3) = n * (n - 1) * (n - 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_binomial_n_n_minus_3_l1816_181616


namespace NUMINAMATH_CALUDE_x_value_l1816_181605

theorem x_value (x : ℝ) : x = 150 * (1 + 0.75) → x = 262.5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1816_181605


namespace NUMINAMATH_CALUDE_binomial_8_5_l1816_181642

theorem binomial_8_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_5_l1816_181642


namespace NUMINAMATH_CALUDE_tangent_circles_diameter_intersection_l1816_181651

/-- Given three circles that are pairwise tangent, the lines connecting
    the tangency points of two circles intersect the third circle at
    the endpoints of its diameter. -/
theorem tangent_circles_diameter_intersection
  (O₁ O₂ O₃ : ℝ × ℝ) -- Centers of the three circles
  (r₁ r₂ r₃ : ℝ) -- Radii of the three circles
  (h_positive : r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) -- Radii are positive
  (h_tangent : -- Circles are pairwise tangent
    (O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2 = (r₁ + r₂)^2 ∧
    (O₂.1 - O₃.1)^2 + (O₂.2 - O₃.2)^2 = (r₂ + r₃)^2 ∧
    (O₃.1 - O₁.1)^2 + (O₃.2 - O₁.2)^2 = (r₃ + r₁)^2)
  (h_distinct : O₁ ≠ O₂ ∧ O₂ ≠ O₃ ∧ O₃ ≠ O₁) -- Centers are distinct
  : ∃ (A B C : ℝ × ℝ), -- Tangency points
    -- A is on circle 1 and 2
    ((A.1 - O₁.1)^2 + (A.2 - O₁.2)^2 = r₁^2 ∧ (A.1 - O₂.1)^2 + (A.2 - O₂.2)^2 = r₂^2) ∧
    -- B is on circle 2 and 3
    ((B.1 - O₂.1)^2 + (B.2 - O₂.2)^2 = r₂^2 ∧ (B.1 - O₃.1)^2 + (B.2 - O₃.2)^2 = r₃^2) ∧
    -- C is on circle 1 and 3
    ((C.1 - O₁.1)^2 + (C.2 - O₁.2)^2 = r₁^2 ∧ (C.1 - O₃.1)^2 + (C.2 - O₃.2)^2 = r₃^2) ∧
    -- Lines AB and AC intersect circle 3 at diameter endpoints
    ∃ (M K : ℝ × ℝ),
      (M.1 - O₃.1)^2 + (M.2 - O₃.2)^2 = r₃^2 ∧
      (K.1 - O₃.1)^2 + (K.2 - O₃.2)^2 = r₃^2 ∧
      (M.1 - K.1)^2 + (M.2 - K.2)^2 = 4 * r₃^2 ∧
      (∃ t : ℝ, M = (1 - t) • A + t • B) ∧
      (∃ s : ℝ, K = (1 - s) • A + s • C) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_diameter_intersection_l1816_181651


namespace NUMINAMATH_CALUDE_last_score_is_84_l1816_181666

def scores : List ℕ := [68, 75, 78, 84, 85, 90]

def is_valid_last_score (s : ℕ) : Prop :=
  s ∈ scores ∧
  ∀ subset : List ℕ, subset.length < 6 → subset ⊆ scores →
  (subset.sum + s) % (subset.length + 1) = 0

theorem last_score_is_84 :
  ∀ s ∈ scores, is_valid_last_score s ↔ s = 84 := by sorry

end NUMINAMATH_CALUDE_last_score_is_84_l1816_181666


namespace NUMINAMATH_CALUDE_number_problem_l1816_181691

theorem number_problem (x : ℝ) : 5 * x + 4 = 19 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1816_181691


namespace NUMINAMATH_CALUDE_machine_purchase_price_l1816_181632

/-- Represents the purchase price of the machine in rupees -/
def purchase_price : ℕ := sorry

/-- Represents the repair cost in rupees -/
def repair_cost : ℕ := 5000

/-- Represents the transportation charges in rupees -/
def transportation_charges : ℕ := 1000

/-- Represents the profit percentage -/
def profit_percentage : ℚ := 50 / 100

/-- Represents the selling price in rupees -/
def selling_price : ℕ := 27000

/-- Theorem stating that the purchase price is 12000 rupees -/
theorem machine_purchase_price : 
  purchase_price = 12000 ∧
  (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage) = selling_price :=
sorry

end NUMINAMATH_CALUDE_machine_purchase_price_l1816_181632


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1816_181674

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), x = 4 ∧ 
  (∀ (y : ℕ), (1100 + y) % 23 = 0 → y ≥ x) ∧ 
  (1100 + x) % 23 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1816_181674


namespace NUMINAMATH_CALUDE_man_son_age_difference_l1816_181664

/-- The age difference between a man and his son -/
def age_difference : ℕ → ℕ → ℕ
  | father_age, son_age => father_age - son_age

/-- Theorem stating the age difference between the man and his son -/
theorem man_son_age_difference :
  ∀ (man_age son_age : ℕ),
    son_age = 44 →
    man_age + 2 = 2 * (son_age + 2) →
    age_difference man_age son_age = 46 := by
  sorry

#check man_son_age_difference

end NUMINAMATH_CALUDE_man_son_age_difference_l1816_181664


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l1816_181696

theorem max_product_under_constraint (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (constraint1 : x * y * z + w = (x + w) * (y + w) * (z + w))
  (constraint2 : x + y + z + w = 1) : 
  x * y * z * w ≤ 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l1816_181696
