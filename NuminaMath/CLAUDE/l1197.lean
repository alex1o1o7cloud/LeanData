import Mathlib

namespace zeros_in_99999_cubed_l1197_119728

-- Define a function to count zeros in a number
def count_zeros (n : ℕ) : ℕ := sorry

-- Define a function to count digits in a number
def count_digits (n : ℕ) : ℕ := sorry

-- Define the given conditions
axiom zeros_9 : count_zeros (9^3) = 0
axiom zeros_99 : count_zeros (99^3) = 2
axiom zeros_999 : count_zeros (999^3) = 3

-- Define the pattern continuation
axiom pattern_continuation (n : ℕ) : 
  n > 999 → count_zeros (n^3) = count_digits n

-- The theorem to prove
theorem zeros_in_99999_cubed : 
  count_zeros ((99999 : ℕ)^3) = count_digits 99999 := by
  sorry

end zeros_in_99999_cubed_l1197_119728


namespace rectangle_area_with_hole_l1197_119740

theorem rectangle_area_with_hole (x : ℝ) (h : x > 1.5) :
  (x + 10) * (x + 8) - (2 * x) * (x + 1) = -x^2 + 16*x + 80 := by
  sorry

end rectangle_area_with_hole_l1197_119740


namespace nina_widget_purchase_l1197_119737

/-- The problem of determining how many widgets Nina can purchase --/
theorem nina_widget_purchase (total_money : ℚ) (reduced_price_quantity : ℕ) (price_reduction : ℚ) : 
  total_money = 27.6 →
  reduced_price_quantity = 8 →
  price_reduction = 1.15 →
  (reduced_price_quantity : ℚ) * ((total_money / (reduced_price_quantity : ℚ)) - price_reduction) = total_money →
  (total_money / (total_money / (reduced_price_quantity : ℚ))).floor = 6 :=
by sorry

end nina_widget_purchase_l1197_119737


namespace only_C_is_comprehensive_unique_comprehensive_survey_l1197_119793

/-- Represents a survey option -/
inductive SurveyOption
| A  -- Survey of the environmental awareness of the people nationwide
| B  -- Survey of the quality of mooncakes in the market during the Mid-Autumn Festival
| C  -- Survey of the weight of 40 students in a class
| D  -- Survey of the safety and quality of a certain type of fireworks and firecrackers

/-- Defines what makes a survey comprehensive -/
def isComprehensive (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.C => true
  | _ => false

/-- Theorem stating that only option C is suitable for a comprehensive survey -/
theorem only_C_is_comprehensive :
  ∀ s : SurveyOption, isComprehensive s ↔ s = SurveyOption.C :=
by sorry

/-- Corollary: There exists exactly one comprehensive survey option -/
theorem unique_comprehensive_survey :
  ∃! s : SurveyOption, isComprehensive s :=
by sorry

end only_C_is_comprehensive_unique_comprehensive_survey_l1197_119793


namespace min_value_sum_reciprocals_l1197_119796

theorem min_value_sum_reciprocals (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_two : a + b + c + d = 2) :
  (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 6 := by
  sorry

end min_value_sum_reciprocals_l1197_119796


namespace problem_solution_l1197_119735

theorem problem_solution (a b c x : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) ≠ 0)
  (h2 : a^2 / (a + b) = a^2 / (a + c) + 20)
  (h3 : b^2 / (b + c) = b^2 / (b + a) + 14)
  (h4 : c^2 / (c + a) = c^2 / (c + b) + x) :
  x = -34 := by
  sorry

end problem_solution_l1197_119735


namespace max_distance_complex_numbers_l1197_119739

theorem max_distance_complex_numbers (z : ℂ) (h : Complex.abs z = 3) :
  Complex.abs ((2 + 3*Complex.I) * z^2 - z^4) ≤ 9*Real.sqrt 13 + 81 := by
  sorry

end max_distance_complex_numbers_l1197_119739


namespace number_of_women_l1197_119721

-- Define the total number of family members
def total_members : ℕ := 15

-- Define the time it takes for a woman to complete the work
def woman_work_days : ℕ := 180

-- Define the time it takes for a man to complete the work
def man_work_days : ℕ := 120

-- Define the time it takes to complete the work with alternating schedule
def alternating_work_days : ℕ := 17

-- Define the function to calculate the number of women
def calculate_women (total : ℕ) (woman_days : ℕ) (man_days : ℕ) (alt_days : ℕ) : ℕ :=
  sorry

-- Theorem statement
theorem number_of_women : 
  calculate_women total_members woman_work_days man_work_days alternating_work_days = 3 :=
sorry

end number_of_women_l1197_119721


namespace f_properties_l1197_119799

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |2*x + a| + |x - 1/a|

theorem f_properties :
  (∀ x : ℝ, (f 1 x < x + 3) ↔ (x > -3/4 ∧ x < 3/2)) ∧
  (∀ a : ℝ, a > 0 → ∀ x : ℝ, f a x ≥ Real.sqrt 2) := by
  sorry

end f_properties_l1197_119799


namespace order_of_powers_l1197_119765

theorem order_of_powers : 3^15 < 2^30 ∧ 2^30 < 10^10 := by
  sorry

end order_of_powers_l1197_119765


namespace vector_relations_l1197_119764

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![3, -4]

def is_collinear (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

def is_perpendicular (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 0 + v 1 * w 1 = 0

theorem vector_relations :
  (∃ k : ℝ, is_collinear (fun i => k * a i - b i) (fun i => a i + b i) ∧ k = -1) ∧
  (∃ k : ℝ, is_perpendicular (fun i => k * a i - b i) (fun i => a i + b i) ∧ k = 16) := by
  sorry


end vector_relations_l1197_119764


namespace equality_proof_l1197_119713

theorem equality_proof : 2017 - 1 / 2017 = (2018 * 2016) / 2017 := by
  sorry

end equality_proof_l1197_119713


namespace positive_interval_l1197_119727

theorem positive_interval (x : ℝ) : (x + 3) * (x - 2) > 0 ↔ x < -3 ∨ x > 2 := by
  sorry

end positive_interval_l1197_119727


namespace vector_parallel_condition_l1197_119742

/-- Given two vectors a and b in ℝ², where a is parallel to (b - a), prove that the x-coordinate of a is -2. -/
theorem vector_parallel_condition (a b : ℝ × ℝ) (h : a.1 = m ∧ a.2 = 1 ∧ b.1 = 2 ∧ b.2 = -1) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a = k • (b - a)) : m = -2 := by
  sorry

end vector_parallel_condition_l1197_119742


namespace income_comparison_l1197_119794

theorem income_comparison (juan tim mart : ℝ) 
  (h1 : tim = 0.6 * juan) 
  (h2 : mart = 0.84 * juan) : 
  (mart - tim) / tim = 0.4 := by
sorry

end income_comparison_l1197_119794


namespace actual_miles_traveled_l1197_119705

/-- Represents an odometer that skips the digit 7 -/
structure FaultyOdometer where
  current_reading : Nat
  skipped_digit : Nat

/-- Calculates the number of skipped readings up to a given number -/
def count_skipped_readings (n : Nat) : Nat :=
  -- Implementation details omitted
  sorry

/-- Theorem: The actual miles traveled when the faulty odometer reads 003008 is 2194 -/
theorem actual_miles_traveled (o : FaultyOdometer) 
  (h1 : o.current_reading = 3008)
  (h2 : o.skipped_digit = 7) : 
  o.current_reading - count_skipped_readings o.current_reading = 2194 := by
  sorry

end actual_miles_traveled_l1197_119705


namespace jar_price_proportion_l1197_119791

/-- Given two cylindrical jars with diameters d₁ and d₂, heights h₁ and h₂, 
    and the price of the first jar p₁, if the price is proportional to the volume, 
    then the price of the second jar p₂ is equal to p₁ * (d₂/d₁)² * (h₂/h₁). -/
theorem jar_price_proportion (d₁ d₂ h₁ h₂ p₁ p₂ : ℝ) (h_d₁_pos : d₁ > 0) (h_d₂_pos : d₂ > 0) 
    (h_h₁_pos : h₁ > 0) (h_h₂_pos : h₂ > 0) (h_p₁_pos : p₁ > 0) :
  p₂ = p₁ * (d₂/d₁)^2 * (h₂/h₁) ↔ 
  p₂ / (π * (d₂/2)^2 * h₂) = p₁ / (π * (d₁/2)^2 * h₁) := by
sorry

end jar_price_proportion_l1197_119791


namespace only_one_proposition_is_true_l1197_119795

-- Define the basic types
def Solid : Type := Unit
def View : Type := Unit

-- Define the properties
def has_three_identical_views (s : Solid) : Prop := sorry
def is_cube (s : Solid) : Prop := sorry
def front_view_is_rectangle (s : Solid) : Prop := sorry
def top_view_is_rectangle (s : Solid) : Prop := sorry
def is_cuboid (s : Solid) : Prop := sorry
def all_views_are_rectangles (s : Solid) : Prop := sorry
def front_view_is_isosceles_trapezoid (s : Solid) : Prop := sorry
def side_view_is_isosceles_trapezoid (s : Solid) : Prop := sorry
def is_frustum (s : Solid) : Prop := sorry

-- Define the propositions
def proposition1 : Prop := ∀ s : Solid, has_three_identical_views s → is_cube s
def proposition2 : Prop := ∀ s : Solid, front_view_is_rectangle s ∧ top_view_is_rectangle s → is_cuboid s
def proposition3 : Prop := ∀ s : Solid, all_views_are_rectangles s → is_cuboid s
def proposition4 : Prop := ∀ s : Solid, front_view_is_isosceles_trapezoid s ∧ side_view_is_isosceles_trapezoid s → is_frustum s

-- Theorem statement
theorem only_one_proposition_is_true : 
  (¬proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ ¬proposition4) ∧
  (proposition1 → (¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4)) ∧
  (proposition2 → (¬proposition1 ∧ ¬proposition3 ∧ ¬proposition4)) ∧
  (proposition4 → (¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3)) :=
sorry

end only_one_proposition_is_true_l1197_119795


namespace simplify_expression_l1197_119774

theorem simplify_expression (x : ℝ) : (3*x)^5 + (4*x)*(x^4) = 247*x^5 := by
  sorry

end simplify_expression_l1197_119774


namespace unique_congruence_in_range_l1197_119712

theorem unique_congruence_in_range : ∃! n : ℕ, 3 ≤ n ∧ n ≤ 10 ∧ n % 7 = 10573 % 7 := by
  sorry

end unique_congruence_in_range_l1197_119712


namespace floor_plus_x_eq_13_4_l1197_119711

theorem floor_plus_x_eq_13_4 :
  ∃! x : ℝ, ⌊x⌋ + x = 13.4 ∧ x = 6.4 := by sorry

end floor_plus_x_eq_13_4_l1197_119711


namespace function_properties_l1197_119719

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

-- Theorem statement
theorem function_properties :
  ∀ (a b : ℝ),
  (∀ x ∈ Set.Ioo (-3) 2, f a b x > 0) ∧
  (∀ x ∈ Set.Iic (-3) ∪ Set.Ici 2, f a b x < 0) →
  (∃ (c : ℝ), ∀ x : ℝ, -3 * x^2 + 5 * x + c ≤ 0) →
  (∃ (y : ℝ), ∀ x > -1, (f (-3) 5 x - 21) / (x + 1) ≤ y ∧ 
    ∃ x₀ > -1, (f (-3) 5 x₀ - 21) / (x₀ + 1) = y) →
  (∀ x : ℝ, f a b x = -3 * x^2 + 3 * x + 18) ∧
  (∀ c : ℝ, (∀ x : ℝ, -3 * x^2 + 5 * x + c ≤ 0) → c ≤ -25/12) ∧
  (∀ x > -1, (f (-3) 5 x - 21) / (x + 1) ≤ -3 ∧ 
    ∃ x₀ > -1, (f (-3) 5 x₀ - 21) / (x₀ + 1) = -3) :=
by sorry

end function_properties_l1197_119719


namespace bedroom_wall_area_l1197_119779

/-- Calculates the total paintable wall area for multiple identical bedrooms -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (non_paintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - non_paintable_area
  num_bedrooms * paintable_area

/-- Proves that the total paintable wall area of 4 bedrooms with given dimensions is 1860 square feet -/
theorem bedroom_wall_area : total_paintable_area 4 15 12 10 75 = 1860 := by
  sorry

end bedroom_wall_area_l1197_119779


namespace tan_negative_255_degrees_l1197_119741

theorem tan_negative_255_degrees : Real.tan (-(255 * π / 180)) = Real.sqrt 3 - 2 := by
  sorry

end tan_negative_255_degrees_l1197_119741


namespace cubic_root_sum_l1197_119715

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 4*p^2 + 6*p - 3 = 0 ∧ 
  q^3 - 4*q^2 + 6*q - 3 = 0 ∧ 
  r^3 - 4*r^2 + 6*r - 3 = 0 → 
  p / (q*r + 2) + q / (p*r + 2) + r / (p*q + 2) = 4/5 := by
sorry

end cubic_root_sum_l1197_119715


namespace max_ben_cookies_proof_l1197_119709

/-- The maximum number of cookies Ben can eat when sharing with Beth -/
def max_ben_cookies : ℕ := 12

/-- The total number of cookies shared between Ben and Beth -/
def total_cookies : ℕ := 36

/-- Predicate to check if a given number of cookies for Ben is valid -/
def valid_ben_cookies (ben : ℕ) : Prop :=
  (ben + 2 * ben = total_cookies) ∨ (ben + 3 * ben = total_cookies)

theorem max_ben_cookies_proof :
  (∀ ben : ℕ, valid_ben_cookies ben → ben ≤ max_ben_cookies) ∧
  valid_ben_cookies max_ben_cookies :=
sorry

end max_ben_cookies_proof_l1197_119709


namespace same_ending_squares_l1197_119755

theorem same_ending_squares (N : ℕ) (h1 : N > 0) 
  (h2 : ∃ (a b c d e : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    N % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    (N * N) % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e) :
  N % 100000 = 69969 := by
sorry

end same_ending_squares_l1197_119755


namespace second_smallest_prime_perimeter_l1197_119768

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def is_scalene_triangle (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def prime_perimeter_triangle (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  is_scalene_triangle a b c ∧
  is_valid_triangle a b c ∧
  is_prime (a + b + c)

theorem second_smallest_prime_perimeter :
  ∃ (a b c : ℕ),
    prime_perimeter_triangle a b c ∧
    (a + b + c = 29) ∧
    (∀ (x y z : ℕ),
      prime_perimeter_triangle x y z →
      (x + y + z ≠ 23) →
      (x + y + z ≥ 29)) :=
sorry

end second_smallest_prime_perimeter_l1197_119768


namespace kindergarten_distribution_l1197_119780

def apples : ℕ := 270
def pears : ℕ := 180
def oranges : ℕ := 235

def is_valid_distribution (n : ℕ) : Prop :=
  n ≠ 0 ∧
  (apples - n * (apples / n) : ℤ) = 3 * (oranges - n * (oranges / n)) ∧
  (pears - n * (pears / n) : ℤ) = 2 * (oranges - n * (oranges / n))

theorem kindergarten_distribution :
  ∃ (n : ℕ), is_valid_distribution n ∧ n = 29 := by
  sorry

end kindergarten_distribution_l1197_119780


namespace total_expense_calculation_l1197_119778

/-- Sandy's current age -/
def sandy_age : ℕ := 34

/-- Kim's current age -/
def kim_age : ℕ := 10

/-- Alex's current age -/
def alex_age : ℕ := sandy_age / 2

/-- Sandy's monthly phone bill expense -/
def sandy_expense : ℕ := 10 * sandy_age

/-- Alex's monthly expense next month -/
def alex_expense : ℕ := 2 * sandy_expense

theorem total_expense_calculation :
  sandy_age = 34 ∧
  kim_age = 10 ∧
  alex_age = sandy_age / 2 ∧
  sandy_expense = 10 * sandy_age ∧
  alex_expense = 2 * sandy_expense ∧
  sandy_age + 2 = 3 * (kim_age + 2) →
  sandy_expense + alex_expense = 1020 := by
  sorry

end total_expense_calculation_l1197_119778


namespace unpaired_numbers_mod_6_l1197_119761

theorem unpaired_numbers_mod_6 (n : ℕ) (hn : n = 800) : 
  ¬ (∃ (f : ℕ → ℕ), 
    (∀ x ∈ Finset.range n, f (f x) = x ∧ x ≠ f x) ∧ 
    (∀ x ∈ Finset.range n, (x + f x) % 6 = 0)) := by
  sorry

end unpaired_numbers_mod_6_l1197_119761


namespace total_cookies_l1197_119759

theorem total_cookies (chris kenny glenn : ℕ) : 
  chris = kenny / 2 →
  glenn = 4 * kenny →
  glenn = 24 →
  chris + kenny + glenn = 33 := by
sorry

end total_cookies_l1197_119759


namespace factorization_1_factorization_2_factorization_3_factorization_4_factorization_5_l1197_119752

-- 1. a³ - 9a = a(a + 3)(a - 3)
theorem factorization_1 (a : ℝ) : a^3 - 9*a = a*(a + 3)*(a - 3) := by sorry

-- 2. 3x² - 6xy + x = x(3x - 6y + 1)
theorem factorization_2 (x y : ℝ) : 3*x^2 - 6*x*y + x = x*(3*x - 6*y + 1) := by sorry

-- 3. n²(m - 2) + n(2 - m) = n(m - 2)(n - 1)
theorem factorization_3 (m n : ℝ) : n^2*(m - 2) + n*(2 - m) = n*(m - 2)*(n - 1) := by sorry

-- 4. -4x² + 4xy + y² = [(2 + 2√2)x + y][(2 - 2√2)x + y]
theorem factorization_4 (x y : ℝ) : 
  -4*x^2 + 4*x*y + y^2 = ((2 + 2*Real.sqrt 2)*x + y)*((2 - 2*Real.sqrt 2)*x + y) := by sorry

-- 5. a² + 2a - 8 = (a - 2)(a + 4)
theorem factorization_5 (a : ℝ) : a^2 + 2*a - 8 = (a - 2)*(a + 4) := by sorry

end factorization_1_factorization_2_factorization_3_factorization_4_factorization_5_l1197_119752


namespace simplify_sqrt_difference_l1197_119787

theorem simplify_sqrt_difference : 
  Real.sqrt (12 + 8 * Real.sqrt 3) - Real.sqrt (12 - 8 * Real.sqrt 3) = 2 * Real.sqrt 3 + 2 * Real.sqrt 6 := by
  sorry

end simplify_sqrt_difference_l1197_119787


namespace paper_strip_dimensions_l1197_119789

theorem paper_strip_dimensions 
  (a b c : ℕ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)  -- Positive dimensions
  (h2 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43)  -- Sum of areas is 43
  : a = 1 ∧ b + c = 22 := by
  sorry

end paper_strip_dimensions_l1197_119789


namespace marathon_distance_l1197_119783

/-- Calculates the distance Tomas can run after a given number of months of training -/
def distance_after_months (initial_distance : ℕ) (months : ℕ) : ℕ :=
  initial_distance * 2^months

/-- The marathon problem -/
theorem marathon_distance (initial_distance : ℕ) (training_months : ℕ) 
  (h1 : initial_distance = 3) 
  (h2 : training_months = 5) : 
  distance_after_months initial_distance training_months = 48 := by
  sorry

#eval distance_after_months 3 5

end marathon_distance_l1197_119783


namespace quadratic_one_solution_sum_l1197_119775

theorem quadratic_one_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x, 5 * x^2 + b₁ * x + 10 * x + 15 = 0 → (b₁ + 10)^2 = 300) ∧
  (∀ x, 5 * x^2 + b₂ * x + 10 * x + 15 = 0 → (b₂ + 10)^2 = 300) ∧
  (∀ b, (∀ x, 5 * x^2 + b * x + 10 * x + 15 = 0 → (b + 10)^2 = 300) → b = b₁ ∨ b = b₂) →
  b₁ + b₂ = -20 := by
sorry

end quadratic_one_solution_sum_l1197_119775


namespace ellipse_hyperbola_same_foci_l1197_119717

/-- Given an ellipse and a hyperbola with the same foci, prove that the semi-major axis of the ellipse is 4 -/
theorem ellipse_hyperbola_same_foci (a : ℝ) : 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / 4 = 1) →  -- Ellipse equation
  (∀ x y : ℝ, x^2 / 9 - y^2 / 3 = 1) →   -- Hyperbola equation
  (a > 0) →                              -- a is positive
  (a^2 - 4 = 12) →                       -- Same foci condition
  a = 4 := by
sorry

end ellipse_hyperbola_same_foci_l1197_119717


namespace magic_card_profit_l1197_119706

/-- Calculates the profit from selling a Magic card that triples in value -/
theorem magic_card_profit (initial_cost : ℝ) : 
  initial_cost > 0 → 2 * initial_cost = (3 * initial_cost) - initial_cost := by
  sorry

#check magic_card_profit

end magic_card_profit_l1197_119706


namespace susan_strawberry_eating_l1197_119749

theorem susan_strawberry_eating (basket_capacity : ℕ) (total_picked : ℕ) (handful_size : ℕ) :
  basket_capacity = 60 →
  total_picked = 75 →
  handful_size = 5 →
  (total_picked - basket_capacity) / (total_picked / handful_size) = 1 := by
  sorry

end susan_strawberry_eating_l1197_119749


namespace total_wrappers_collected_l1197_119732

theorem total_wrappers_collected (andy_wrappers max_wrappers : ℕ) 
  (h1 : andy_wrappers = 34) 
  (h2 : max_wrappers = 15) : 
  andy_wrappers + max_wrappers = 49 := by
  sorry

end total_wrappers_collected_l1197_119732


namespace log_8_1000_equals_inverse_log_10_2_l1197_119731

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define log_10 as the natural logarithm divided by ln(10)
noncomputable def log_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_8_1000_equals_inverse_log_10_2 :
  log 8 1000 = 1 / log_10 2 := by
  sorry

end log_8_1000_equals_inverse_log_10_2_l1197_119731


namespace parallel_lines_m_equals_one_l1197_119792

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  (1 : ℝ) / (-(1 + m)) = m / (-2 : ℝ)

/-- If the lines x + (1+m)y = 2-m and mx + 2y + 8 = 0 are parallel, then m = 1 -/
theorem parallel_lines_m_equals_one :
  ∀ m : ℝ, parallel_lines m → m = 1 := by
  sorry

end parallel_lines_m_equals_one_l1197_119792


namespace daisy_rose_dogs_pool_l1197_119745

/-- The number of legs/paws in a pool with humans and dogs -/
def legs_paws_in_pool (num_humans : ℕ) (num_dogs : ℕ) : ℕ :=
  num_humans * 2 + num_dogs * 4

/-- Theorem: The number of legs/paws in the pool with Daisy, Rose, and their 5 dogs is 24 -/
theorem daisy_rose_dogs_pool : legs_paws_in_pool 2 5 = 24 := by
  sorry

end daisy_rose_dogs_pool_l1197_119745


namespace discount_relationship_l1197_119756

/-- Represents the banker's discount in Rupees -/
def bankers_discount : ℝ := 78

/-- Represents the true discount in Rupees -/
def true_discount : ℝ := 66

/-- Represents the sum due (present value) in Rupees -/
def sum_due : ℝ := 363

/-- Theorem stating the relationship between banker's discount, true discount, and sum due -/
theorem discount_relationship : 
  bankers_discount = true_discount + (true_discount^2 / sum_due) :=
by sorry

end discount_relationship_l1197_119756


namespace adams_money_l1197_119701

/-- Adam's money problem --/
theorem adams_money (initial_amount spent allowance : ℕ) :
  initial_amount = 5 →
  spent = 2 →
  allowance = 5 →
  initial_amount - spent + allowance = 8 := by
  sorry

end adams_money_l1197_119701


namespace length_A_l1197_119781

-- Define the points
def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 10)
def C : ℝ × ℝ := (3, 7)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define the condition that A' and B' are on the line y = x
axiom A'_on_line : ∃ A' : ℝ × ℝ, line_y_eq_x A'
axiom B'_on_line : ∃ B' : ℝ × ℝ, line_y_eq_x B'

-- Define the condition that AA' and BB' intersect at C
axiom AA'_BB'_intersect_at_C : 
  ∃ A' B' : ℝ × ℝ, line_y_eq_x A' ∧ line_y_eq_x B' ∧
  (∃ t₁ t₂ : ℝ, A + t₁ • (A' - A) = C ∧ B + t₂ • (B' - B) = C)

-- State the theorem
theorem length_A'B'_is_4_sqrt_2 : 
  ∃ A' B' : ℝ × ℝ, line_y_eq_x A' ∧ line_y_eq_x B' ∧
  (∃ t₁ t₂ : ℝ, A + t₁ • (A' - A) = C ∧ B + t₂ • (B' - B) = C) ∧
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 4 * Real.sqrt 2 :=
sorry

end length_A_l1197_119781


namespace second_quadrant_complex_l1197_119723

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem second_quadrant_complex :
  let z : ℂ := -1
  is_in_second_quadrant ((2 - Complex.I) * z) := by
  sorry

end second_quadrant_complex_l1197_119723


namespace corporate_event_handshakes_eq_430_l1197_119736

/-- Represents the number of handshakes at a corporate event --/
def corporate_event_handshakes : ℕ :=
  let total_people : ℕ := 40
  let group_a_size : ℕ := 15
  let group_b_size : ℕ := 20
  let group_c_size : ℕ := 5
  let group_b_knowing_a : ℕ := 5
  let group_b_knowing_none : ℕ := 15

  let handshakes_a_b : ℕ := group_a_size * group_b_knowing_none
  let handshakes_within_b : ℕ := (group_b_knowing_none * (group_b_knowing_none - 1)) / 2
  let handshakes_b_c : ℕ := group_b_size * group_c_size

  handshakes_a_b + handshakes_within_b + handshakes_b_c

/-- Theorem stating that the number of handshakes at the corporate event is 430 --/
theorem corporate_event_handshakes_eq_430 : corporate_event_handshakes = 430 := by
  sorry

end corporate_event_handshakes_eq_430_l1197_119736


namespace rosie_pies_theorem_l1197_119725

/-- Represents the number of pies that can be made from a given number of apples -/
def pies_from_apples (apples : ℕ) : ℕ :=
  3 * (apples / 12)

theorem rosie_pies_theorem :
  pies_from_apples 36 = 9 := by
  sorry

end rosie_pies_theorem_l1197_119725


namespace interest_rate_calculation_l1197_119788

/-- Given a principal amount and an interest rate, if the simple interest for 2 years is 660
    and the compound interest for 2 years is 696.30, then the interest rate is 11%. -/
theorem interest_rate_calculation (P R : ℝ) : 
  P * R * 2 / 100 = 660 →
  P * ((1 + R / 100)^2 - 1) = 696.30 →
  R = 11 := by
sorry

end interest_rate_calculation_l1197_119788


namespace gcd_from_lcm_and_ratio_l1197_119772

theorem gcd_from_lcm_and_ratio (X Y : ℕ+) : 
  Nat.lcm X Y = 180 → 
  (X : ℚ) / (Y : ℚ) = 2 / 5 → 
  Nat.gcd X Y = 18 := by
sorry

end gcd_from_lcm_and_ratio_l1197_119772


namespace polly_tweet_time_l1197_119703

/-- Represents the number of tweets per minute in different emotional states -/
structure TweetsPerMinute where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Represents the total number of tweets and the time spent in each state -/
structure TweetData where
  tweets_per_minute : TweetsPerMinute
  total_tweets : ℕ
  time_per_state : ℕ

/-- Theorem: Given Polly's tweet rates and total tweets, prove the time spent in each state -/
theorem polly_tweet_time (data : TweetData)
  (h1 : data.tweets_per_minute.happy = 18)
  (h2 : data.tweets_per_minute.hungry = 4)
  (h3 : data.tweets_per_minute.mirror = 45)
  (h4 : data.total_tweets = 1340)
  (h5 : data.time_per_state * (data.tweets_per_minute.happy + data.tweets_per_minute.hungry + data.tweets_per_minute.mirror) = data.total_tweets) :
  data.time_per_state = 20 := by
  sorry


end polly_tweet_time_l1197_119703


namespace r_value_when_n_is_3_l1197_119782

/-- Given n, s, and r where s = 3^n + 2 and r = 4^s - 2s, 
    prove that when n = 3, r = 4^29 - 58 -/
theorem r_value_when_n_is_3 (n : ℕ) (s : ℕ) (r : ℕ) 
    (h1 : s = 3^n + 2) 
    (h2 : r = 4^s - 2*s) : 
  n = 3 → r = 4^29 - 58 := by
  sorry

end r_value_when_n_is_3_l1197_119782


namespace quadratic_equations_solutions_l1197_119707

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 11 ∧ x₂ = 2 - Real.sqrt 11 ∧
    x₁^2 - 4*x₁ - 7 = 0 ∧ x₂^2 - 4*x₂ - 7 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 1/3 ∧ y₂ = -2 ∧
    3*y₁^2 + 5*y₁ - 2 = 0 ∧ 3*y₂^2 + 5*y₂ - 2 = 0) :=
by sorry

end quadratic_equations_solutions_l1197_119707


namespace initial_persons_count_l1197_119798

/-- The initial number of persons in a group where:
    1. The average weight increases by 4.5 kg when a new person joins.
    2. The person being replaced weighs 65 kg.
    3. The new person weighs 101 kg. -/
def initialPersons : ℕ := 8

theorem initial_persons_count :
  let avgWeightIncrease : ℚ := 4.5
  let replacedPersonWeight : ℕ := 65
  let newPersonWeight : ℕ := 101
  let totalWeightIncrease : ℚ := avgWeightIncrease * initialPersons
  totalWeightIncrease = (newPersonWeight - replacedPersonWeight) →
  initialPersons = 8 := by sorry

end initial_persons_count_l1197_119798


namespace oil_bill_ratio_l1197_119751

/-- The oil bill problem -/
theorem oil_bill_ratio : 
  ∀ (feb_bill jan_bill : ℚ),
  jan_bill = 180 →
  (feb_bill + 45) / jan_bill = 3 / 2 →
  feb_bill / jan_bill = 5 / 4 := by
sorry

end oil_bill_ratio_l1197_119751


namespace basketball_win_rate_l1197_119724

theorem basketball_win_rate (initial_wins initial_games remaining_games : ℕ) 
  (h1 : initial_wins = 45)
  (h2 : initial_games = 60)
  (h3 : remaining_games = 50) :
  ∃ (remaining_wins : ℕ), 
    (initial_wins + remaining_wins : ℚ) / (initial_games + remaining_games) = 3/4 ∧ 
    remaining_wins = 38 := by
  sorry

end basketball_win_rate_l1197_119724


namespace geometric_sequence_ratio_l1197_119702

/-- Given a geometric sequence {a_n} where all terms are positive and 
    (a₁, ½a₃, 2a₂) forms an arithmetic sequence, 
    prove that (a₉ + a₁₀) / (a₇ + a₈) = 3 + 2√2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →
  (a 1 + 2 * a 2 = a 3) →
  (a 9 + a 10) / (a 7 + a 8) = 3 + 2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_ratio_l1197_119702


namespace original_number_proof_l1197_119708

/-- Given a three-digit number abc and N = 3194, where N is the sum of acb, bac, bca, cab, and cba, prove that abc = 358 -/
theorem original_number_proof (a b c : ℕ) (h1 : a ≠ 0) 
  (h2 : a * 100 + b * 10 + c < 1000) 
  (h3 : 3194 = (a * 100 + c * 10 + b) + (b * 100 + a * 10 + c) + 
               (b * 100 + c * 10 + a) + (c * 100 + a * 10 + b) + 
               (c * 100 + b * 10 + a)) : 
  a * 100 + b * 10 + c = 358 := by
sorry

end original_number_proof_l1197_119708


namespace cereal_expense_per_year_l1197_119710

def boxes_per_week : ℕ := 2
def cost_per_box : ℚ := 3
def weeks_per_year : ℕ := 52

theorem cereal_expense_per_year :
  (boxes_per_week * weeks_per_year * cost_per_box : ℚ) = 312 := by
  sorry

end cereal_expense_per_year_l1197_119710


namespace sum_of_coefficients_l1197_119733

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a + a₁*(x + 2) + a₂*(x + 2)^2 + a₃*(x + 2)^3 + 
    a₄*(x + 2)^4 + a₅*(x + 2)^5 + a₆*(x + 2)^6 + a₇*(x + 2)^7 + a₈*(x + 2)^8 + 
    a₉*(x + 2)^9 + a₁₀*(x + 2)^10 + a₁₁*(x + 2)^11) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 := by
sorry

end sum_of_coefficients_l1197_119733


namespace polygon_not_covered_by_homothetic_polygons_l1197_119738

/-- A polygon in a 2D space -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  -- Add more properties as needed

/-- Homothetic transformation of a polygon -/
def homothetic_transform (p : Polygon) (center : ℝ × ℝ) (k : ℝ) : Polygon :=
  sorry

/-- Predicate to check if a point is contained in a polygon -/
def point_in_polygon (point : ℝ × ℝ) (p : Polygon) : Prop :=
  sorry

theorem polygon_not_covered_by_homothetic_polygons 
  (M : Polygon) (k : ℝ) (h : 0 < k ∧ k < 1) :
  ∃ (point : ℝ × ℝ), 
    point_in_polygon point M ∧
    ∀ (center1 center2 : ℝ × ℝ),
      ¬(point_in_polygon point (homothetic_transform M center1 k) ∨
        point_in_polygon point (homothetic_transform M center2 k)) :=
by
  sorry

end polygon_not_covered_by_homothetic_polygons_l1197_119738


namespace find_number_l1197_119700

theorem find_number : ∃ x : ℤ, x + 5 = 9 ∧ x = 4 := by sorry

end find_number_l1197_119700


namespace min_xy_value_l1197_119784

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (3 / (2 + x)) + (3 / (2 + y)) = 1) : 
  x * y ≥ 16 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (3 / (2 + x)) + (3 / (2 + y)) = 1 ∧ x * y = 16 := by
  sorry

end min_xy_value_l1197_119784


namespace large_circle_diameter_l1197_119771

theorem large_circle_diameter (r : ℝ) (h : r = 4) :
  let small_circles_radius := r
  let small_circles_count := 8
  let inner_octagon_side := 2 * small_circles_radius
  let inner_octagon_radius := inner_octagon_side / Real.sqrt 2
  let large_circle_radius := inner_octagon_radius + small_circles_radius
  large_circle_radius * 2 = 8 * Real.sqrt 2 + 8 := by
  sorry

end large_circle_diameter_l1197_119771


namespace parking_lot_revenue_l1197_119757

/-- Given a parking lot with the following properties:
  * Total spaces: 1000
  * Section 1: 320 spaces at $5 per hour
  * Section 2: 200 more spaces than Section 3 at $8 per hour
  * Section 3: Remaining spaces at $4 per hour
  Prove that Section 2 has 440 spaces and the total revenue for 5 hours is $30400 -/
theorem parking_lot_revenue 
  (total_spaces : Nat) 
  (section1_spaces : Nat) 
  (section2_price : Nat) 
  (section3_price : Nat) 
  (section1_price : Nat) 
  (hours : Nat) :
  total_spaces = 1000 →
  section1_spaces = 320 →
  section2_price = 8 →
  section3_price = 4 →
  section1_price = 5 →
  hours = 5 →
  ∃ (section2_spaces section3_spaces : Nat),
    section2_spaces = section3_spaces + 200 ∧
    section1_spaces + section2_spaces + section3_spaces = total_spaces ∧
    section2_spaces = 440 ∧
    section1_spaces * section1_price * hours + 
    section2_spaces * section2_price * hours + 
    section3_spaces * section3_price * hours = 30400 := by
  sorry


end parking_lot_revenue_l1197_119757


namespace quadratic_equation_roots_l1197_119729

theorem quadratic_equation_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x - 2 = 0 ∧ x = 2) → 
  (b = -1 ∧ ∃ y : ℝ, y^2 + b*y - 2 = 0 ∧ y = -1) :=
by sorry

end quadratic_equation_roots_l1197_119729


namespace quadratic_trinomials_sum_l1197_119754

theorem quadratic_trinomials_sum (p q : ℝ) : 
  (∃! x, 2 * x^2 + (p + q) * x + (p + q) = 0) →
  (2 + (p + q) + (p + q) = 2 ∨ 2 + (p + q) + (p + q) = 18) :=
by sorry

end quadratic_trinomials_sum_l1197_119754


namespace cakes_served_at_lunch_l1197_119746

theorem cakes_served_at_lunch (total : ℕ) (dinner : ℕ) (yesterday : ℕ) 
  (h1 : total = 14) 
  (h2 : dinner = 6) 
  (h3 : yesterday = 3) : 
  total - dinner - yesterday = 5 := by
  sorry

end cakes_served_at_lunch_l1197_119746


namespace houses_with_both_pets_l1197_119714

theorem houses_with_both_pets (total : ℕ) (dogs : ℕ) (cats : ℕ) 
  (h_total : total = 60) 
  (h_dogs : dogs = 40) 
  (h_cats : cats = 30) : 
  dogs + cats - total = 10 := by
  sorry

end houses_with_both_pets_l1197_119714


namespace tv_screen_width_l1197_119734

/-- Given a rectangular TV screen with area 21 square feet and height 7 feet, its width is 3 feet. -/
theorem tv_screen_width (area : ℝ) (height : ℝ) (width : ℝ) : 
  area = 21 → height = 7 → area = width * height → width = 3 := by
  sorry

end tv_screen_width_l1197_119734


namespace unique_m_value_l1197_119753

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

/-- The smallest positive period of a function -/
def SmallestPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  HasPeriod f p ∧ p > 0 ∧ ∀ q, HasPeriod f q ∧ q > 0 → p ≤ q

theorem unique_m_value (f : ℝ → ℝ) (m : ℝ) :
  IsOdd f →
  SmallestPositivePeriod f 4 →
  f 1 > 1 →
  f 2 = m^2 - 2*m →
  f 3 = (2*m - 5)/(m + 1) →
  m = 0 := by sorry

end unique_m_value_l1197_119753


namespace daily_earnings_of_c_l1197_119769

theorem daily_earnings_of_c (A B C : ℕ) 
  (h1 : A + B + C = 600)
  (h2 : A + C = 400)
  (h3 : B + C = 300) :
  C = 100 := by
sorry

end daily_earnings_of_c_l1197_119769


namespace sum_of_largest_and_smallest_prime_factors_of_990_l1197_119718

theorem sum_of_largest_and_smallest_prime_factors_of_990 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ 990 ∧ largest ∣ 990 ∧
    (∀ p : ℕ, p.Prime → p ∣ 990 → smallest ≤ p ∧ p ≤ largest) ∧
    smallest + largest = 13 := by
  sorry

end sum_of_largest_and_smallest_prime_factors_of_990_l1197_119718


namespace kiley_ate_two_slices_l1197_119704

/-- Represents a cheesecake with its properties and consumption -/
structure Cheesecake where
  calories_per_slice : ℕ
  total_calories : ℕ
  percent_eaten : ℚ

/-- Calculates the number of slices eaten given a Cheesecake -/
def slices_eaten (c : Cheesecake) : ℚ :=
  (c.total_calories / c.calories_per_slice : ℚ) * c.percent_eaten

/-- Theorem stating that Kiley ate 2 slices of the specified cheesecake -/
theorem kiley_ate_two_slices (c : Cheesecake) 
  (h1 : c.calories_per_slice = 350)
  (h2 : c.total_calories = 2800)
  (h3 : c.percent_eaten = 1/4) : 
  slices_eaten c = 2 := by
  sorry

end kiley_ate_two_slices_l1197_119704


namespace no_solution_iff_m_geq_three_l1197_119760

theorem no_solution_iff_m_geq_three (m : ℝ) :
  (∀ x : ℝ, ¬(x - m ≥ 0 ∧ (1/2) * x + (1/2) < 2)) ↔ m ≥ 3 :=
by sorry

end no_solution_iff_m_geq_three_l1197_119760


namespace fonzie_payment_l1197_119790

/-- Proves that Fonzie's payment for the treasure map is $7000 -/
theorem fonzie_payment (fonzie_payment : ℝ) : 
  (∀ total_payment : ℝ, 
    total_payment = fonzie_payment + 8000 + 9000 ∧ 
    9000 / total_payment = 337500 / 900000) →
  fonzie_payment = 7000 := by
sorry

end fonzie_payment_l1197_119790


namespace expected_value_is_five_l1197_119767

/-- Represents the outcome of rolling a fair 8-sided die -/
inductive DieRoll
  | one | two | three | four | five | six | seven | eight

/-- The probability of each outcome of the die roll -/
def prob : DieRoll → ℚ
  | _ => 1/8

/-- The winnings for each outcome of the die roll -/
def winnings : DieRoll → ℚ
  | DieRoll.two => 4
  | DieRoll.four => 8
  | DieRoll.six => 12
  | DieRoll.eight => 16
  | _ => 0

/-- The expected value of the winnings -/
def expected_value : ℚ :=
  (prob DieRoll.two * winnings DieRoll.two) +
  (prob DieRoll.four * winnings DieRoll.four) +
  (prob DieRoll.six * winnings DieRoll.six) +
  (prob DieRoll.eight * winnings DieRoll.eight)

theorem expected_value_is_five :
  expected_value = 5 := by sorry

end expected_value_is_five_l1197_119767


namespace main_theorem_l1197_119758

noncomputable section

variable (e : ℝ)
variable (f : ℝ → ℝ)

-- Define the conditions
def non_negative (f : ℝ → ℝ) : Prop := ∀ x ∈ Set.Icc 0 e, f x ≥ 0
def f_e_equals_e : Prop := f e = e
def superadditive (f : ℝ → ℝ) : Prop := 
  ∀ x₁ x₂, x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ + x₂ ≤ e → f (x₁ + x₂) ≥ f x₁ + f x₂

-- Define the inequality condition
def inequality_condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 e, 4 * (f x)^2 - 4 * (2 * e - a) * f x + 4 * e^2 - 4 * e * a + 1 ≥ 0

-- Main theorem
theorem main_theorem (h1 : non_negative e f) (h2 : f_e_equals_e e f) (h3 : superadditive e f) :
  (f 0 = 0) ∧
  (∀ x ∈ Set.Icc 0 e, f x ≤ e) ∧
  (∀ a : ℝ, inequality_condition e f a → a ≤ e) := by
  sorry

end

end main_theorem_l1197_119758


namespace determinant_equal_polynomial_l1197_119797

variable (x : ℝ)

def matrix : Matrix (Fin 3) (Fin 3) ℝ := λ i j =>
  match i, j with
  | 0, 0 => 2*x + 3
  | 0, 1 => x
  | 0, 2 => x
  | 1, 0 => 2*x
  | 1, 1 => 2*x + 3
  | 1, 2 => x
  | 2, 0 => 2*x
  | 2, 1 => x
  | 2, 2 => 2*x + 3

theorem determinant_equal_polynomial (x : ℝ) :
  Matrix.det (matrix x) = 2*x^3 + 27*x^2 + 27*x + 27 := by
  sorry

end determinant_equal_polynomial_l1197_119797


namespace least_value_x_minus_y_plus_z_l1197_119720

theorem least_value_x_minus_y_plus_z (x y z : ℕ+) (h : (3 : ℕ) * x.val = (4 : ℕ) * y.val ∧ (4 : ℕ) * y.val = (7 : ℕ) * z.val) :
  (x.val - y.val + z.val : ℤ) ≥ 19 ∧ ∃ (x₀ y₀ z₀ : ℕ+), (3 : ℕ) * x₀.val = (4 : ℕ) * y₀.val ∧ (4 : ℕ) * y₀.val = (7 : ℕ) * z₀.val ∧ (x₀.val - y₀.val + z₀.val : ℤ) = 19 :=
sorry

end least_value_x_minus_y_plus_z_l1197_119720


namespace find_N_l1197_119726

theorem find_N : ∃ N : ℕ, 
  (87^2 - 78^2) % N = 0 ∧ 
  45 < N ∧ 
  N < 100 ∧ 
  (N = 55 ∨ N = 99) := by
sorry

end find_N_l1197_119726


namespace simplify_fraction_product_l1197_119785

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end simplify_fraction_product_l1197_119785


namespace hanoi_theorem_l1197_119743

/-- The minimum number of moves to solve the Towers of Hanoi puzzle with n disks -/
def hanoi_moves (n : ℕ) : ℕ := 2^n - 1

/-- The minimum number of moves to solve the Towers of Hanoi puzzle with n disks
    when direct movement between pegs 1 and 3 is prohibited -/
def hanoi_moves_restricted (n : ℕ) : ℕ := 3^n - 1

/-- The minimum number of moves to solve the Towers of Hanoi puzzle with n disks
    when the smallest disk cannot be placed on peg 2 -/
def hanoi_moves_no_small_on_middle (n : ℕ) : ℕ := 2 * 3^(n-1) - 1

theorem hanoi_theorem (n : ℕ) :
  (hanoi_moves n = 2^n - 1) ∧
  (hanoi_moves_restricted n = 3^n - 1) ∧
  (hanoi_moves_no_small_on_middle n = 2 * 3^(n-1) - 1) :=
by sorry

end hanoi_theorem_l1197_119743


namespace max_m_value_l1197_119722

theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → (4 / (1 - x)) ≥ m - (1 / x)) → 
  m ≤ 9 := by
  sorry

end max_m_value_l1197_119722


namespace excess_meat_sold_proof_l1197_119716

/-- Calculates the excess meat sold beyond the original plan. -/
def excess_meat_sold (thursday_sales : ℕ) (saturday_sales : ℕ) (original_plan : ℕ) : ℕ :=
  let friday_sales := 2 * thursday_sales
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  total_sales - original_plan

/-- Proves that the excess meat sold beyond the original plan is 325 kg. -/
theorem excess_meat_sold_proof :
  excess_meat_sold 210 130 500 = 325 := by
  sorry

end excess_meat_sold_proof_l1197_119716


namespace cannot_divide_rectangle_l1197_119744

theorem cannot_divide_rectangle : ¬ ∃ (m n : ℕ), 55 = m * 5 ∧ 39 = n * 11 := by
  sorry

end cannot_divide_rectangle_l1197_119744


namespace multiply_72517_and_9999_l1197_119776

theorem multiply_72517_and_9999 : 72517 * 9999 = 725097483 := by
  sorry

end multiply_72517_and_9999_l1197_119776


namespace gcd_18_30_l1197_119770

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l1197_119770


namespace min_distance_exp_ln_l1197_119730

/-- The minimum distance between a point on y = e^x and a point on y = ln(x) -/
theorem min_distance_exp_ln : ∀ (P Q : ℝ × ℝ),
  (∃ x : ℝ, P = (x, Real.exp x)) →
  (∃ y : ℝ, Q = (Real.exp y, y)) →
  ∃ d : ℝ, d = Real.sqrt 2 ∧ ∀ P' Q', 
    (∃ x' : ℝ, P' = (x', Real.exp x')) →
    (∃ y' : ℝ, Q' = (Real.exp y', y')) →
    d ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) := by
  sorry

end min_distance_exp_ln_l1197_119730


namespace equal_interval_points_ratio_l1197_119763

theorem equal_interval_points_ratio : 
  ∀ (s S : ℝ), 
  (∃ d : ℝ, s = 9 * d ∧ S = 99 * d) → 
  S / s = 11 :=
by
  sorry

end equal_interval_points_ratio_l1197_119763


namespace algorithm_design_principle_l1197_119786

-- Define the characteristics of algorithms
def Algorithm : Type := Unit

-- Define the properties of algorithms
def is_reversible (a : Algorithm) : Prop := sorry
def can_run_endlessly (a : Algorithm) : Prop := sorry
def is_unique_for_task (a : Algorithm) : Prop := sorry
def should_be_simple_and_convenient (a : Algorithm) : Prop := sorry

-- Define the theorem
theorem algorithm_design_principle :
  ∀ (a : Algorithm),
    ¬(is_reversible a) ∧
    ¬(can_run_endlessly a) ∧
    ¬(is_unique_for_task a) ∧
    should_be_simple_and_convenient a :=
by
  sorry

end algorithm_design_principle_l1197_119786


namespace copy_machine_rate_l1197_119747

/-- Given two copy machines working together for 30 minutes to produce 2850 copies,
    with one machine producing 55 copies per minute, prove that the other machine
    produces 40 copies per minute. -/
theorem copy_machine_rate : ∀ (rate1 : ℕ),
  (30 * rate1 + 30 * 55 = 2850) → rate1 = 40 := by
  sorry

end copy_machine_rate_l1197_119747


namespace inscribed_circle_radius_l1197_119766

/-- A triangle with an inscribed circle where the area is numerically twice the perimeter -/
structure SpecialTriangle where
  -- The semiperimeter of the triangle
  s : ℝ
  -- The area of the triangle
  A : ℝ
  -- The perimeter of the triangle
  p : ℝ
  -- The radius of the inscribed circle
  r : ℝ
  -- The semiperimeter is positive
  s_pos : 0 < s
  -- The perimeter is twice the semiperimeter
  perim_eq : p = 2 * s
  -- The area is twice the perimeter
  area_eq : A = 2 * p
  -- The area formula using inradius
  area_formula : A = r * s

/-- The radius of the inscribed circle in a SpecialTriangle is 4 -/
theorem inscribed_circle_radius (t : SpecialTriangle) : t.r = 4 := by
  sorry

end inscribed_circle_radius_l1197_119766


namespace power_function_value_l1197_119777

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f 2 = Real.sqrt 2 / 2) :
  f 9 = 1 / 3 := by
  sorry

end power_function_value_l1197_119777


namespace no_relationship_between_running_and_age_probability_of_one_not_interested_l1197_119773

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ := !![15, 20; 10, 15]

-- Define the total sample size
def n : ℕ := 60

-- Define the K² formula
def K_squared (a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 90% confidence level
def critical_value : ℚ := 2.706

-- Theorem for part 1
theorem no_relationship_between_running_and_age :
  K_squared 15 20 10 15 < critical_value :=
sorry

-- Theorem for part 2
theorem probability_of_one_not_interested :
  (Nat.choose 5 3 - Nat.choose 3 3) / Nat.choose 5 3 = 3 / 5 :=
sorry

end no_relationship_between_running_and_age_probability_of_one_not_interested_l1197_119773


namespace first_time_below_397_l1197_119750

def countingOff (n : ℕ) : ℕ := n - (n / 3)

def remainingStudents (initialCount : ℕ) (rounds : ℕ) : ℕ :=
  match rounds with
  | 0 => initialCount
  | n + 1 => countingOff (remainingStudents initialCount n)

theorem first_time_below_397 (initialCount : ℕ) (h : initialCount = 2010) :
  remainingStudents initialCount 5 ≤ 397 ∧
  ∀ k < 5, remainingStudents initialCount k > 397 :=
sorry

end first_time_below_397_l1197_119750


namespace local_extrema_of_f_l1197_119748

def f (x : ℝ) : ℝ := 1 + 3*x - x^3

theorem local_extrema_of_f :
  (∃ δ₁ > 0, ∀ x ∈ Set.Ioo (-1 - δ₁) (-1 + δ₁), f x ≥ f (-1)) ∧
  (∃ δ₂ > 0, ∀ x ∈ Set.Ioo (1 - δ₂) (1 + δ₂), f x ≤ f 1) ∧
  f (-1) = -1 ∧
  f 1 = 3 :=
sorry

end local_extrema_of_f_l1197_119748


namespace sum_of_coefficients_l1197_119762

theorem sum_of_coefficients (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, a₁*(x-1)^4 + a₂*(x-1)^3 + a₃*(x-1)^2 + a₄*(x-1) + a₅ = x^4) →
  a₂ + a₃ + a₄ = 14 :=
by sorry

end sum_of_coefficients_l1197_119762
