import Mathlib

namespace cloud9_diving_total_money_l2669_266951

/-- The total money taken by Cloud 9 Diving Company -/
theorem cloud9_diving_total_money (individual_bookings group_bookings returned : ℕ) 
  (h1 : individual_bookings = 12000)
  (h2 : group_bookings = 16000)
  (h3 : returned = 1600) :
  individual_bookings + group_bookings - returned = 26400 := by
  sorry

end cloud9_diving_total_money_l2669_266951


namespace alcohol_percentage_after_dilution_l2669_266949

/-- Calculates the alcohol percentage in a mixture after adding water --/
theorem alcohol_percentage_after_dilution
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 11)
  (h2 : initial_alcohol_percentage = 42)
  (h3 : added_water = 3)
  : (initial_alcohol_percentage * initial_volume) / (initial_volume + added_water) = 33 := by
  sorry

end alcohol_percentage_after_dilution_l2669_266949


namespace min_distinct_prime_factors_l2669_266931

theorem min_distinct_prime_factors (m n : ℕ) :
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  (p ∣ (m * (n + 9) * (m + 2 * n^2 + 3))) ∧
  (q ∣ (m * (n + 9) * (m + 2 * n^2 + 3))) :=
sorry

end min_distinct_prime_factors_l2669_266931


namespace train_passing_pole_time_l2669_266902

/-- Proves that a train 150 metres long running at 54 km/hr takes 10 seconds to pass a pole. -/
theorem train_passing_pole_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (h1 : train_length = 150) 
  (h2 : train_speed_kmh = 54) : 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 10 := by
sorry

end train_passing_pole_time_l2669_266902


namespace power_fraction_equality_l2669_266974

theorem power_fraction_equality : (1 : ℚ) / ((-5^4)^2) * (-5)^9 = -5 := by sorry

end power_fraction_equality_l2669_266974


namespace min_value_sqrt_sum_squares_l2669_266909

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 5)
  (h2 : m*a + n*b = 5) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 5 := by
  sorry

end min_value_sqrt_sum_squares_l2669_266909


namespace extended_parallelepiped_volume_sum_l2669_266921

/-- Represents a rectangular parallelepiped with dimensions l, w, and h -/
structure Parallelepiped where
  l : ℝ
  w : ℝ
  h : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a parallelepiped -/
def volume_extended_parallelepiped (p : Parallelepiped) : ℝ := sorry

/-- Checks if two integers are relatively prime -/
def relatively_prime (a b : ℕ) : Prop := sorry

theorem extended_parallelepiped_volume_sum (m n p : ℕ) :
  (∃ (parallelepiped : Parallelepiped),
    parallelepiped.l = 3 ∧
    parallelepiped.w = 4 ∧
    parallelepiped.h = 5 ∧
    volume_extended_parallelepiped parallelepiped = (m + n * Real.pi) / p ∧
    m > 0 ∧ n > 0 ∧ p > 0 ∧
    relatively_prime n p) →
  m + n + p = 505 := by sorry

end extended_parallelepiped_volume_sum_l2669_266921


namespace smallest_number_in_special_triple_l2669_266961

theorem smallest_number_in_special_triple : 
  ∀ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) →  -- Three positive integers
    ((a + b + c) / 3 : ℚ) = 30 →  -- Arithmetic mean is 30
    b = 29 →  -- Median is 29
    max a (max b c) = b + 4 →  -- Median is 4 less than the largest number
    min a (min b c) = 28 :=  -- The smallest number is 28
by
  sorry

end smallest_number_in_special_triple_l2669_266961


namespace rhombus_area_fraction_l2669_266934

theorem rhombus_area_fraction (grid_size : ℕ) (rhombus_side : ℝ) :
  grid_size = 7 →
  rhombus_side = Real.sqrt 2 →
  (4 * (1/2 * rhombus_side * rhombus_side)) / ((grid_size - 1)^2) = 1/18 :=
by sorry

end rhombus_area_fraction_l2669_266934


namespace smallest_k_for_64_power_gt_4_power_17_l2669_266960

theorem smallest_k_for_64_power_gt_4_power_17 : 
  (∃ k : ℕ, 64^k > 4^17 ∧ ∀ m : ℕ, m < k → 64^m ≤ 4^17) ∧ 
  (∀ k : ℕ, 64^k > 4^17 → k ≥ 6) :=
by sorry

end smallest_k_for_64_power_gt_4_power_17_l2669_266960


namespace range_of_m_l2669_266926

theorem range_of_m (m : ℝ) : 
  (¬∀ (x : ℝ), m * x^2 - 2 * m * x + 1 > 0) → (m < 0 ∨ m ≥ 1) := by
  sorry

end range_of_m_l2669_266926


namespace stating_remaining_slices_is_four_l2669_266996

/-- Represents the number of slices in a large pizza -/
def large_pizza_slices : ℕ := 8

/-- Represents the number of slices in an extra-large pizza -/
def extra_large_pizza_slices : ℕ := 12

/-- Represents the number of slices Mary eats from the large pizza -/
def mary_large_slices : ℕ := 7

/-- Represents the number of slices Mary eats from the extra-large pizza -/
def mary_extra_large_slices : ℕ := 3

/-- Represents the number of slices John eats from the large pizza -/
def john_large_slices : ℕ := 2

/-- Represents the number of slices John eats from the extra-large pizza -/
def john_extra_large_slices : ℕ := 5

/-- 
Theorem stating that the total number of remaining slices is 4,
given the conditions of the problem.
-/
theorem remaining_slices_is_four :
  (large_pizza_slices - min mary_large_slices large_pizza_slices - min john_large_slices (large_pizza_slices - min mary_large_slices large_pizza_slices)) +
  (extra_large_pizza_slices - mary_extra_large_slices - john_extra_large_slices) = 4 := by
  sorry

end stating_remaining_slices_is_four_l2669_266996


namespace x_equation_proof_l2669_266910

theorem x_equation_proof (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 5*x + 4/x + 1/x^2 = 34)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by
  sorry

end x_equation_proof_l2669_266910


namespace shannon_bought_no_gum_l2669_266939

/-- Represents the purchase made by Shannon -/
structure Purchase where
  yogurt_pints : ℕ
  gum_packs : ℕ
  shrimp_trays : ℕ
  yogurt_price : ℚ
  shrimp_price : ℚ
  total_cost : ℚ

/-- The conditions of Shannon's purchase -/
def shannon_purchase : Purchase where
  yogurt_pints := 5
  gum_packs := 0  -- We'll prove this
  shrimp_trays := 5
  yogurt_price := 6  -- Derived from the total cost
  shrimp_price := 5
  total_cost := 55

/-- The price of gum is half the price of yogurt -/
def gum_price (p : Purchase) : ℚ := p.yogurt_price / 2

/-- The total cost of the purchase -/
def total_cost (p : Purchase) : ℚ :=
  p.yogurt_pints * p.yogurt_price +
  p.gum_packs * (gum_price p) +
  p.shrimp_trays * p.shrimp_price

/-- Theorem stating that Shannon bought 0 packs of gum -/
theorem shannon_bought_no_gum :
  shannon_purchase.gum_packs = 0 ∧
  total_cost shannon_purchase = shannon_purchase.total_cost := by
  sorry


end shannon_bought_no_gum_l2669_266939


namespace seven_digit_numbers_even_together_even_odd_together_l2669_266944

/-- The number of even digits from 1 to 9 -/
def num_even_digits : ℕ := 4

/-- The number of odd digits from 1 to 9 -/
def num_odd_digits : ℕ := 5

/-- The number of even digits to be selected -/
def num_even_selected : ℕ := 3

/-- The number of odd digits to be selected -/
def num_odd_selected : ℕ := 4

/-- The total number of digits to be selected -/
def total_selected : ℕ := num_even_selected + num_odd_selected

theorem seven_digit_numbers (n : ℕ) :
  (n = Nat.choose num_even_digits num_even_selected * 
       Nat.choose num_odd_digits num_odd_selected * 
       Nat.factorial total_selected) → 
  n = 100800 := by sorry

theorem even_together (n : ℕ) :
  (n = Nat.choose num_even_digits num_even_selected * 
       Nat.choose num_odd_digits num_odd_selected * 
       Nat.factorial (total_selected - num_even_selected + 1) * 
       Nat.factorial num_even_selected) → 
  n = 14400 := by sorry

theorem even_odd_together (n : ℕ) :
  (n = Nat.choose num_even_digits num_even_selected * 
       Nat.choose num_odd_digits num_odd_selected * 
       Nat.factorial num_even_selected * 
       Nat.factorial num_odd_selected * 
       Nat.factorial 2) → 
  n = 5760 := by sorry

end seven_digit_numbers_even_together_even_odd_together_l2669_266944


namespace composite_power_plus_four_l2669_266972

theorem composite_power_plus_four (n : ℕ) (h : n > 1) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^2020 + 4 = a * b :=
by sorry

end composite_power_plus_four_l2669_266972


namespace gcd_360_504_l2669_266946

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_l2669_266946


namespace cubic_system_solution_l2669_266978

theorem cubic_system_solution (x y z : ℝ) : 
  ((x + y)^3 = z ∧ (y + z)^3 = x ∧ (z + x)^3 = y) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = Real.sqrt 2 / 4 ∧ y = Real.sqrt 2 / 4 ∧ z = Real.sqrt 2 / 4) ∨ 
   (x = -Real.sqrt 2 / 4 ∧ y = -Real.sqrt 2 / 4 ∧ z = -Real.sqrt 2 / 4)) :=
by sorry

end cubic_system_solution_l2669_266978


namespace max_blocks_in_box_l2669_266980

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the maximum number of blocks that can fit in a box -/
def maxBlocks (box : Dimensions) (block : Dimensions) : ℕ :=
  (box.height / block.height) * (box.width / block.width) * (box.length / block.length)

/-- The box dimensions -/
def boxDim : Dimensions := ⟨8, 10, 12⟩

/-- Type A block dimensions -/
def blockADim : Dimensions := ⟨3, 2, 4⟩

/-- Type B block dimensions -/
def blockBDim : Dimensions := ⟨4, 3, 5⟩

theorem max_blocks_in_box :
  max (maxBlocks boxDim blockADim) (maxBlocks boxDim blockBDim) = 30 := by
  sorry

end max_blocks_in_box_l2669_266980


namespace arithmetic_progression_rth_term_l2669_266906

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℚ := 3 * n + 5 * n^2

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℚ := S r - S (r - 1)

theorem arithmetic_progression_rth_term (r : ℕ) (hr : r > 0) : a r = 10 * r - 2 := by
  sorry

end arithmetic_progression_rth_term_l2669_266906


namespace negative_fifty_deg_same_terminal_side_as_three_hundred_ten_deg_l2669_266945

-- Define the property of two angles having the same terminal side
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

-- State the theorem
theorem negative_fifty_deg_same_terminal_side_as_three_hundred_ten_deg :
  same_terminal_side (-50) 310 := by
  sorry

end negative_fifty_deg_same_terminal_side_as_three_hundred_ten_deg_l2669_266945


namespace opinion_change_difference_l2669_266900

theorem opinion_change_difference (initial_like initial_dislike final_like final_dislike : ℝ) :
  initial_like = 40 →
  initial_dislike = 60 →
  final_like = 80 →
  final_dislike = 20 →
  initial_like + initial_dislike = 100 →
  final_like + final_dislike = 100 →
  let min_change := |final_like - initial_like|
  let max_change := min initial_like initial_dislike + min final_like final_dislike
  max_change - min_change = 60 := by
sorry

end opinion_change_difference_l2669_266900


namespace ratio_children_to_adults_l2669_266957

def total_people : ℕ := 120
def children : ℕ := 80

theorem ratio_children_to_adults :
  (children : ℚ) / (total_people - children : ℚ) = 2 / 1 := by sorry

end ratio_children_to_adults_l2669_266957


namespace nice_polynomial_characterization_l2669_266956

def is_nice (f : ℝ → ℝ) (A B : Finset ℝ) : Prop :=
  A.card = B.card ∧ B = A.image f

def can_produce_nice (S : ℝ → ℝ) : Prop :=
  ∀ A B : Finset ℝ, A.card = B.card → ∃ f : ℝ → ℝ, is_nice f A B

def is_polynomial (f : ℝ → ℝ) : Prop := sorry

def degree (f : ℝ → ℝ) : ℕ := sorry

def leading_coefficient (f : ℝ → ℝ) : ℝ := sorry

theorem nice_polynomial_characterization (S : ℝ → ℝ) :
  (is_polynomial S ∧ can_produce_nice S) ↔
  (is_polynomial S ∧ degree S ≥ 2 ∧
   (Even (degree S) ∨ (Odd (degree S) ∧ leading_coefficient S < 0))) :=
sorry

end nice_polynomial_characterization_l2669_266956


namespace largest_solution_equation_inverse_x_12_value_l2669_266912

noncomputable def largest_x : ℝ :=
  Real.exp (- (7 / 12) * Real.log 10)

theorem largest_solution_equation (x : ℝ) (h : x = largest_x) :
  (Real.log 10) / (Real.log (10 * x^2)) + (Real.log 10) / (Real.log (100 * x^3)) = -2 :=
sorry

theorem inverse_x_12_value :
  (1 : ℝ) / largest_x^12 = 10000000 :=
sorry

end largest_solution_equation_inverse_x_12_value_l2669_266912


namespace smartphone_price_reduction_l2669_266973

theorem smartphone_price_reduction (original_price final_price : ℝ) 
  (h1 : original_price = 7500)
  (h2 : final_price = 4800)
  (h3 : ∃ (x : ℝ), final_price = original_price * (1 - x)^2 ∧ 0 < x ∧ x < 1) :
  ∃ (x : ℝ), final_price = original_price * (1 - x)^2 ∧ x = 0.2 :=
sorry

end smartphone_price_reduction_l2669_266973


namespace solution_system_equations_l2669_266997

theorem solution_system_equations :
  ∃! (x y : ℝ), 
    x + Real.sqrt (x + 2*y) - 2*y = 7/2 ∧
    x^2 + x + 2*y - 4*y^2 = 27/2 ∧
    x = 19/4 ∧ y = 17/8 := by
  sorry

end solution_system_equations_l2669_266997


namespace function_property_l2669_266936

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def hasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_property (f : ℝ → ℝ) 
  (h_even : isEven f)
  (h_period : hasPeriod f 2)
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
sorry

end function_property_l2669_266936


namespace expression_simplification_l2669_266977

theorem expression_simplification (x : ℝ) : (3*x - 4)*(2*x + 9) - (x + 6)*(3*x + 2) = 3*x^2 - x - 48 := by
  sorry

end expression_simplification_l2669_266977


namespace inequality_transformation_l2669_266928

theorem inequality_transformation (a b c : ℝ) :
  (b / (a^2 + 1) > c / (a^2 + 1)) → b > c := by sorry

end inequality_transformation_l2669_266928


namespace complex_product_theorem_l2669_266922

theorem complex_product_theorem :
  let Q : ℂ := 4 + 3 * Complex.I
  let E : ℂ := 2 * Complex.I
  let D : ℂ := 4 - 3 * Complex.I
  Q * E * D = 50 * Complex.I :=
by sorry

end complex_product_theorem_l2669_266922


namespace equation_solution_l2669_266986

theorem equation_solution : ∃ x : ℝ, (3 * x - 5 = -2 * x + 10) ∧ (x = 3) := by
  sorry

end equation_solution_l2669_266986


namespace arithmetic_sequence_properties_l2669_266919

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℚ := 3 * n

-- Define the sum of the first n terms of {a_n}
def S (n : ℕ) : ℚ := n * (3 + a n) / 2

-- Define the sequence {c_n}
def c (n : ℕ) : ℚ := 3 / (2 * S n)

-- Define the sum of the first n terms of {c_n}
def T (n : ℕ) : ℚ := n / (n + 1)

theorem arithmetic_sequence_properties :
  (a 1 = 3) ∧
  (a 3 + S 3 = 27) ∧
  (∀ n : ℕ, a n = 3 * n) ∧
  (∀ n : ℕ, T n = n / (n + 1)) :=
by sorry

end arithmetic_sequence_properties_l2669_266919


namespace sum_series_equals_three_halves_l2669_266966

/-- The sum of the series (4n-3)/3^n from n=1 to infinity equals 3/2 -/
theorem sum_series_equals_three_halves :
  (∑' n : ℕ, (4 * n - 3 : ℝ) / 3^n) = 3/2 := by
  sorry

end sum_series_equals_three_halves_l2669_266966


namespace simplify_sqrt_expression_l2669_266952

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (4 + ((x^3 - 2) / x)^2) = (Real.sqrt (x^6 - 4*x^3 + 4*x^2 + 4)) / x :=
by sorry

end simplify_sqrt_expression_l2669_266952


namespace sum_of_coefficients_cubic_factorization_l2669_266941

theorem sum_of_coefficients_cubic_factorization :
  ∀ (a b c d e : ℝ),
  (∀ x, 512 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 60 := by
sorry

end sum_of_coefficients_cubic_factorization_l2669_266941


namespace circle_O1_equation_constant_sum_of_squares_l2669_266995

-- Define the circles and points
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle_O1 (x y m : ℝ) : Prop := (x - m)^2 + y^2 = (m - 3)^2 + 4^2

-- Define the intersection point P
def P : ℝ × ℝ := (3, 4)

-- Define the line l
def line_l (x y k : ℝ) : Prop := y - 4 = k * (x - 3)

-- Define the perpendicular line l1
def line_l1 (x y k : ℝ) : Prop := y - 4 = (-1/k) * (x - 3)

-- Theorem 1
theorem circle_O1_equation (m : ℝ) : 
  (∃ B : ℝ × ℝ, circle_O1 B.1 B.2 m ∧ line_l B.1 B.2 1 ∧ (B.1 - 3)^2 + (B.2 - 4)^2 = 98) →
  (∀ x y : ℝ, circle_O1 x y m ↔ (x - 14)^2 + y^2 = 137) :=
sorry

-- Theorem 2
theorem constant_sum_of_squares (m : ℝ) :
  ∀ k : ℝ, k ≠ 0 →
  (∃ A B C D : ℝ × ℝ, 
    circle_O A.1 A.2 ∧ circle_O1 B.1 B.2 m ∧ line_l A.1 A.2 k ∧ line_l B.1 B.2 k ∧
    circle_O C.1 C.2 ∧ circle_O1 D.1 D.2 m ∧ line_l1 C.1 C.2 k ∧ line_l1 D.1 D.2 k) →
  (∃ A B C D : ℝ × ℝ, 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 + (C.1 - D.1)^2 + (C.2 - D.2)^2 = 4 * m^2) :=
sorry

end circle_O1_equation_constant_sum_of_squares_l2669_266995


namespace fred_paid_twenty_dollars_l2669_266904

/-- The amount Fred paid with at the movie theater -/
def fred_payment (ticket_price : ℚ) (num_tickets : ℕ) (movie_rental : ℚ) (change : ℚ) : ℚ :=
  ticket_price * num_tickets + movie_rental + change

/-- Theorem stating the amount Fred paid with -/
theorem fred_paid_twenty_dollars :
  fred_payment (92 / 100) 2 (679 / 100) (137 / 100) = 20 := by
  sorry

end fred_paid_twenty_dollars_l2669_266904


namespace alcohol_mixture_proof_l2669_266933

/-- Proves that mixing equal volumes of 10% and 30% alcohol solutions results in a 20% solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 200
  let y_volume : ℝ := 200
  let x_concentration : ℝ := 0.1
  let y_concentration : ℝ := 0.3
  let target_concentration : ℝ := 0.2
  x_volume * x_concentration + y_volume * y_concentration = 
    (x_volume + y_volume) * target_concentration :=
by
  sorry

#check alcohol_mixture_proof

end alcohol_mixture_proof_l2669_266933


namespace intersection_y_intercept_l2669_266905

/-- Given two lines that intersect at a specific x-coordinate, 
    prove that the y-intercept of the first line has a specific value. -/
theorem intersection_y_intercept (m : ℝ) : 
  (∃ y : ℝ, 3 * (-6.7) + y = m ∧ -0.5 * (-6.7) + y = 20) → 
  m = -3.45 := by
sorry

end intersection_y_intercept_l2669_266905


namespace expression_evaluation_l2669_266998

theorem expression_evaluation (b : ℝ) (a : ℝ) (h1 : b = 2) (h2 : a = b + 3) :
  (a^2 + b)^2 - (a^2 - b)^2 = 200 := by sorry

end expression_evaluation_l2669_266998


namespace base_conversion_1765_l2669_266992

/-- Converts a base 10 number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 6 to a natural number in base 10 -/
def fromBase6 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_1765 :
  toBase6 1765 = [1, 2, 1, 0, 1] ∧ fromBase6 [1, 2, 1, 0, 1] = 1765 := by
  sorry

end base_conversion_1765_l2669_266992


namespace min_cost_is_800_l2669_266971

/-- Represents the number of adults in the group -/
def num_adults : ℕ := 8

/-- Represents the number of children in the group -/
def num_children : ℕ := 4

/-- Represents the cost of an adult ticket in yuan -/
def adult_ticket_cost : ℕ := 100

/-- Represents the cost of a child ticket in yuan -/
def child_ticket_cost : ℕ := 50

/-- Represents the cost of a group ticket per person in yuan -/
def group_ticket_cost : ℕ := 70

/-- Represents the minimum number of people required for group tickets -/
def min_group_size : ℕ := 10

/-- Calculates the total cost of tickets given the number of group tickets and individual tickets -/
def total_cost (num_group : ℕ) (num_individual_adult : ℕ) (num_individual_child : ℕ) : ℕ :=
  num_group * group_ticket_cost + 
  num_individual_adult * adult_ticket_cost + 
  num_individual_child * child_ticket_cost

/-- Theorem stating that the minimum cost for the given group is 800 yuan -/
theorem min_cost_is_800 : 
  ∀ (num_group : ℕ) (num_individual_adult : ℕ) (num_individual_child : ℕ),
    num_group + num_individual_adult + num_individual_child = num_adults + num_children →
    num_group = 0 ∨ num_group ≥ min_group_size →
    total_cost num_group num_individual_adult num_individual_child ≥ 800 ∧
    (∃ (ng na nc : ℕ), 
      ng + na + nc = num_adults + num_children ∧
      (ng = 0 ∨ ng ≥ min_group_size) ∧
      total_cost ng na nc = 800) := by
  sorry

#check min_cost_is_800

end min_cost_is_800_l2669_266971


namespace expression_simplification_l2669_266954

theorem expression_simplification (a b : ℝ) (h1 : a = 2) (h2 : b = 3) :
  3 * a^2 * b - (a * b^2 - 2 * (2 * a^2 * b - a * b^2)) - a * b^2 = 12 := by
  sorry

end expression_simplification_l2669_266954


namespace function_properties_l2669_266901

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

def has_period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem function_properties (f : ℝ → ℝ) :
  (is_even f ∧ symmetric_about f 1 → has_period f 2) ∧
  (symmetric_about f 1 ∧ has_period f 2 → is_even f) ∧
  (is_even f ∧ has_period f 2 → symmetric_about f 1) →
  is_even f ∧ symmetric_about f 1 ∧ has_period f 2 :=
by sorry

end function_properties_l2669_266901


namespace parabola_vertex_l2669_266938

/-- The vertex of the parabola y = -2(x-3)^2 - 4 is at (3, -4) -/
theorem parabola_vertex :
  let f : ℝ → ℝ := λ x => -2 * (x - 3)^2 - 4
  ∃! p : ℝ × ℝ, p.1 = 3 ∧ p.2 = -4 ∧ ∀ x : ℝ, f x ≤ f p.1 :=
by sorry

end parabola_vertex_l2669_266938


namespace line_intersects_circle_l2669_266959

/-- The line (x-1)a + y = 1 always intersects the circle x^2 + y^2 = 3 for any real value of a -/
theorem line_intersects_circle (a : ℝ) : ∃ (x y : ℝ), 
  ((x - 1) * a + y = 1) ∧ (x^2 + y^2 = 3) := by sorry

end line_intersects_circle_l2669_266959


namespace fraction_subtraction_fraction_division_l2669_266989

-- Problem 1
theorem fraction_subtraction (x y : ℝ) (h : x + y ≠ 0) :
  (2 * x + 3 * y) / (x + y) - (x + 2 * y) / (x + y) = 1 := by
sorry

-- Problem 2
theorem fraction_division (a : ℝ) (h : a ≠ 2) :
  (a^2 - 1) / (a^2 - 4*a + 4) / ((a - 1) / (a - 2)) = (a + 1) / (a - 2) := by
sorry

end fraction_subtraction_fraction_division_l2669_266989


namespace log_sum_equals_two_l2669_266962

theorem log_sum_equals_two : Real.log 4 + Real.log 25 = 2 * Real.log 10 := by sorry

end log_sum_equals_two_l2669_266962


namespace metallic_sheet_volumes_l2669_266925

/-- Represents the dimensions of a rectangular sheet -/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a square -/
structure SquareDimensions where
  side : ℝ

/-- Calculates the volume of open box A -/
def volume_box_a (sheet : SheetDimensions) (corner_cut : SquareDimensions) : ℝ :=
  (sheet.length - 2 * corner_cut.side) * (sheet.width - 2 * corner_cut.side) * corner_cut.side

/-- Calculates the volume of open box B -/
def volume_box_b (sheet : SheetDimensions) (corner_cut : SquareDimensions) (middle_cut : SquareDimensions) : ℝ :=
  ((sheet.length - 2 * corner_cut.side) * (sheet.width - 2 * corner_cut.side) - middle_cut.side ^ 2) * corner_cut.side

theorem metallic_sheet_volumes
  (sheet : SheetDimensions)
  (corner_cut : SquareDimensions)
  (middle_cut : SquareDimensions)
  (h1 : sheet.length = 48)
  (h2 : sheet.width = 36)
  (h3 : corner_cut.side = 8)
  (h4 : middle_cut.side = 12) :
  volume_box_a sheet corner_cut = 5120 ∧ volume_box_b sheet corner_cut middle_cut = 3968 := by
  sorry

#eval volume_box_a ⟨48, 36⟩ ⟨8⟩
#eval volume_box_b ⟨48, 36⟩ ⟨8⟩ ⟨12⟩

end metallic_sheet_volumes_l2669_266925


namespace f_increasing_implies_a_range_l2669_266927

/-- Given a real number a, f is a function from ℝ to ℝ defined as f(x) = x^2 + 2(a - 1)x + 2 -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 + 2*(a - 1)*x + 2

/-- The theorem states that if f is increasing on [4, +∞), then a ≥ -3 -/
theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x y, x ≥ 4 → y ≥ 4 → x ≤ y → f a x ≤ f a y) →
  a ≥ -3 :=
sorry

end f_increasing_implies_a_range_l2669_266927


namespace quadratic_roots_sum_product_l2669_266991

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 24) →
  p + q = 99 := by
sorry

end quadratic_roots_sum_product_l2669_266991


namespace gcd_7254_156_minus_10_l2669_266940

theorem gcd_7254_156_minus_10 : Nat.gcd 7254 156 - 10 = 68 := by
  sorry

end gcd_7254_156_minus_10_l2669_266940


namespace quadratic_inequality_theorem_l2669_266915

theorem quadratic_inequality_theorem (k : ℝ) : 
  (¬∃ x : ℝ, (k^2 - 1) * x^2 + 4 * (1 - k) * x + 3 ≤ 0) → 
  (k = 1 ∨ (1 < k ∧ k < 7)) :=
by sorry

end quadratic_inequality_theorem_l2669_266915


namespace part_one_part_two_l2669_266988

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x ∈ Set.Icc 1 2, (a - 1) * x - 1 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 1 > 0

-- Theorem for part (1)
theorem part_one (a : ℝ) : (p a ∧ q a) ↔ (3/2 < a ∧ a < 2) :=
sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : (¬(¬(p a) ∧ q a) ∧ (¬(p a) ∨ q a)) ↔ (a ≤ -2 ∨ (3/2 < a ∧ a < 2)) :=
sorry

end part_one_part_two_l2669_266988


namespace hyperbola_properties_l2669_266947

/-- Given a hyperbola C with equation 9y^2 - 16x^2 = 144 -/
def hyperbola_C (x y : ℝ) : Prop := 9 * y^2 - 16 * x^2 = 144

/-- Point P -/
def point_P : ℝ × ℝ := (6, 4)

/-- Theorem stating properties of hyperbola C and a related hyperbola -/
theorem hyperbola_properties :
  ∃ (a b c : ℝ),
    /- Transverse axis length -/
    2 * a = 8 ∧
    /- Conjugate axis length -/
    2 * b = 6 ∧
    /- Foci coordinates -/
    (∀ (x y : ℝ), hyperbola_C x y → (x = 0 ∧ (y = c ∨ y = -c))) ∧
    /- Eccentricity -/
    c / a = 5 / 4 ∧
    /- New hyperbola equation -/
    (∀ (x y : ℝ), x^2 / 27 - y^2 / 48 = 1 →
      /- Same asymptotes as C -/
      (∃ (k : ℝ), k ≠ 0 ∧ 9 * y^2 - 16 * x^2 = 144 * k) ∧
      /- Passes through point P -/
      (let (px, py) := point_P; x = px ∧ y = py)) :=
sorry

end hyperbola_properties_l2669_266947


namespace number_sum_theorem_l2669_266907

theorem number_sum_theorem :
  (∀ n : ℕ, n ≥ 100 → n ≥ smallest_three_digit) ∧
  (∀ n : ℕ, n < 100 → n ≤ largest_two_digit) ∧
  (∀ n : ℕ, n < 10 ∧ n % 2 = 1 → n ≥ smallest_odd_one_digit) ∧
  (∀ n : ℕ, n < 100 ∧ n % 2 = 0 → n ≤ largest_even_two_digit) →
  smallest_three_digit + largest_two_digit = 199 ∧
  smallest_odd_one_digit + largest_even_two_digit = 99 :=
by sorry

def smallest_three_digit : ℕ := 100
def largest_two_digit : ℕ := 99
def smallest_odd_one_digit : ℕ := 1
def largest_even_two_digit : ℕ := 98

end number_sum_theorem_l2669_266907


namespace parallel_unit_vectors_l2669_266937

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def is_unit_vector (v : E) : Prop := ‖v‖ = 1

def are_parallel (v w : E) : Prop := ∃ (k : ℝ), v = k • w

theorem parallel_unit_vectors (a b : E) 
  (ha : is_unit_vector a) (hb : is_unit_vector b) (hpar : are_parallel a b) : 
  a = b ∨ a = -b := by
  sorry

end parallel_unit_vectors_l2669_266937


namespace common_factor_proof_l2669_266916

theorem common_factor_proof (x y a b : ℝ) :
  ∃ (k : ℝ), 3*x*(a - b) - 9*y*(b - a) = 3*(a - b) * k :=
by sorry

end common_factor_proof_l2669_266916


namespace ned_lost_lives_l2669_266968

/-- Proves that Ned lost 13 lives in a video game -/
theorem ned_lost_lives (initial_lives current_lives : ℕ) 
  (h1 : initial_lives = 83) 
  (h2 : current_lives = 70) : 
  initial_lives - current_lives = 13 := by
  sorry

end ned_lost_lives_l2669_266968


namespace tangent_chord_length_l2669_266920

/-- The circle with equation x^2 + y^2 - 6x - 8y + 20 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 - 8*p.2 + 20 = 0}

/-- The origin point (0, 0) -/
def Origin : ℝ × ℝ := (0, 0)

/-- A point is on the circle if it satisfies the circle equation -/
def IsOnCircle (p : ℝ × ℝ) : Prop := p ∈ Circle

/-- A line is tangent to the circle if it touches the circle at exactly one point -/
def IsTangentLine (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ × ℝ), IsOnCircle p ∧ l ∩ Circle = {p}

/-- The theorem stating that the length of the chord formed by two tangent lines
    from the origin to the circle is 4√5 -/
theorem tangent_chord_length :
  ∃ (A B : ℝ × ℝ) (OA OB : Set (ℝ × ℝ)),
    IsOnCircle A ∧ IsOnCircle B ∧
    IsTangentLine OA ∧ IsTangentLine OB ∧
    Origin ∈ OA ∧ Origin ∈ OB ∧
    A ∈ OA ∧ B ∈ OB ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 :=
sorry

end tangent_chord_length_l2669_266920


namespace max_value_of_function_l2669_266929

theorem max_value_of_function (x : ℝ) (h : x < 0) :
  2 * x + 2 / x ≤ -4 ∧ ∃ x₀, x₀ < 0 ∧ 2 * x₀ + 2 / x₀ = -4 := by
  sorry

end max_value_of_function_l2669_266929


namespace cucumber_price_l2669_266948

theorem cucumber_price (cucumber_price : ℝ) 
  (tomato_price_relation : cucumber_price * 0.8 = cucumber_price - cucumber_price * 0.2)
  (total_price : 2 * (cucumber_price * 0.8) + 3 * cucumber_price = 23) :
  cucumber_price = 5 := by
  sorry

end cucumber_price_l2669_266948


namespace value_of_c_l2669_266953

theorem value_of_c (a b c : ℝ) (h1 : a = 6) (h2 : b = 15) (h3 : 6 * 15 * c = 3) :
  (a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) ↔ c = 3 := by
sorry

end value_of_c_l2669_266953


namespace mixture_ratio_theorem_l2669_266908

/-- Represents the ratio of alcohol to water in a mixture -/
structure AlcoholRatio where
  alcohol : ℝ
  water : ℝ

/-- Calculates the ratio of alcohol to water when mixing two solutions -/
def mixSolutions (v1 v2 : ℝ) (r1 r2 : AlcoholRatio) : AlcoholRatio :=
  { alcohol := v1 * r1.alcohol + v2 * r2.alcohol,
    water := v1 * r1.water + v2 * r2.water }

theorem mixture_ratio_theorem (V p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let jar1 := AlcoholRatio.mk (p / (p + 2)) (2 / (p + 2))
  let jar2 := AlcoholRatio.mk (q / (q + 1)) (1 / (q + 1))
  let mixture := mixSolutions V (2 * V) jar1 jar2
  mixture.alcohol / mixture.water = (p * (q + 1) + 4 * q * (p + 2)) / (2 * (q + 1) + 4 * (p + 2)) := by
  sorry

#check mixture_ratio_theorem

end mixture_ratio_theorem_l2669_266908


namespace function_properties_l2669_266970

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

theorem function_properties
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : -π / 2 ≤ φ ∧ φ < π / 2)
  (h_sym : ∀ x, f ω φ (2 * π / 3 - x) = f ω φ (2 * π / 3 + x))
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) :
  ω = 2 ∧
  φ = -π / 6 ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f ω φ x ≤ Real.sqrt 3) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f ω φ x ≥ -Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc 0 (π / 2), f ω φ x = Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc 0 (π / 2), f ω φ x = -Real.sqrt 3 / 2) :=
by sorry

end function_properties_l2669_266970


namespace coefficient_equals_21_l2669_266981

-- Define the coefficient of x^2 in the expansion of (ax+1)^5(x+1)^2
def coefficient (a : ℝ) : ℝ := 10 * a^2 + 10 * a + 1

-- Theorem statement
theorem coefficient_equals_21 (a : ℝ) : 
  coefficient a = 21 ↔ a = 1 ∨ a = -2 := by
  sorry


end coefficient_equals_21_l2669_266981


namespace female_lion_weight_l2669_266990

theorem female_lion_weight (male_weight : ℚ) (weight_difference : ℚ) :
  male_weight = 145 / 4 →
  weight_difference = 47 / 10 →
  male_weight - weight_difference = 631 / 20 := by
  sorry

end female_lion_weight_l2669_266990


namespace expression_simplification_l2669_266967

theorem expression_simplification (x : ℝ) : 
  ((7 * x - 3) + 3 * x * 2) * 2 + (5 + 2 * 2) * (4 * x + 6) = 62 * x + 48 := by
  sorry

end expression_simplification_l2669_266967


namespace fixed_point_on_line_l2669_266935

theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end fixed_point_on_line_l2669_266935


namespace total_players_l2669_266932

theorem total_players (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ) : 
  kabadi = 10 → kho_kho_only = 35 → both = 5 → kabadi + kho_kho_only - both = 40 := by
  sorry

end total_players_l2669_266932


namespace block_weight_difference_l2669_266965

theorem block_weight_difference :
  let yellow_weight : ℝ := 0.6
  let green_weight : ℝ := 0.4
  yellow_weight - green_weight = 0.2 := by
sorry

end block_weight_difference_l2669_266965


namespace rectangle_max_area_l2669_266994

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) →  -- Perimeter condition
  (l * w ≤ 100) ∧         -- Area is at most 100
  (∃ l' w' : ℕ, 2 * l' + 2 * w' = 40 ∧ l' * w' = 100) -- Maximum area exists
  :=
sorry

end rectangle_max_area_l2669_266994


namespace point_in_third_quadrant_l2669_266987

-- Define the Cartesian coordinate system
def CartesianPoint := ℝ × ℝ

-- Define the quadrants
def first_quadrant (p : CartesianPoint) : Prop := p.1 > 0 ∧ p.2 > 0
def second_quadrant (p : CartesianPoint) : Prop := p.1 < 0 ∧ p.2 > 0
def third_quadrant (p : CartesianPoint) : Prop := p.1 < 0 ∧ p.2 < 0
def fourth_quadrant (p : CartesianPoint) : Prop := p.1 > 0 ∧ p.2 < 0

-- The point in question
def point : CartesianPoint := (-5, -1)

-- The theorem to prove
theorem point_in_third_quadrant : third_quadrant point := by sorry

end point_in_third_quadrant_l2669_266987


namespace square_difference_divided_by_nine_l2669_266958

theorem square_difference_divided_by_nine : (121^2 - 112^2) / 9 = 233 := by
  sorry

end square_difference_divided_by_nine_l2669_266958


namespace max_gcd_13n_plus_3_7n_plus_1_l2669_266903

theorem max_gcd_13n_plus_3_7n_plus_1 :
  (∃ (n : ℕ+), Nat.gcd (13 * n + 3) (7 * n + 1) = 8) ∧
  (∀ (n : ℕ+), Nat.gcd (13 * n + 3) (7 * n + 1) ≤ 8) := by sorry

end max_gcd_13n_plus_3_7n_plus_1_l2669_266903


namespace exists_multiple_with_sum_of_digits_equal_to_n_l2669_266923

def sumOfDigits (m : ℕ) : ℕ :=
  if m < 10 then m else m % 10 + sumOfDigits (m / 10)

theorem exists_multiple_with_sum_of_digits_equal_to_n (n : ℕ) (hn : n > 0) :
  ∃ m : ℕ, m % n = 0 ∧ sumOfDigits m = n := by
  sorry

end exists_multiple_with_sum_of_digits_equal_to_n_l2669_266923


namespace other_number_proof_l2669_266984

theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 108) (h2 : Nat.lcm a b = 27720) (h3 : a = 216) : b = 64 := by
  sorry

end other_number_proof_l2669_266984


namespace geometric_series_second_term_l2669_266911

theorem geometric_series_second_term :
  ∀ (a : ℝ) (r : ℝ) (S : ℝ),
    r = (1 : ℝ) / 4 →
    S = 40 →
    S = a / (1 - r) →
    a * r = (15 : ℝ) / 2 :=
by sorry

end geometric_series_second_term_l2669_266911


namespace complex_number_quadrant_l2669_266982

theorem complex_number_quadrant : ∃ (z : ℂ), 
  z = (2 - Complex.I) / (2 + Complex.I) ∧ 
  0 < z.re ∧ z.im < 0 :=
by sorry

end complex_number_quadrant_l2669_266982


namespace geometric_sequence_common_ratio_l2669_266924

/-- A geometric sequence with first term 1 and fourth term 1/64 has common ratio 1/4 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℚ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence condition
  a 1 = 1 →                               -- First term is 1
  a 4 = 1 / 64 →                          -- Fourth term is 1/64
  a 2 / a 1 = 1 / 4 :=                    -- Common ratio is 1/4
by sorry

end geometric_sequence_common_ratio_l2669_266924


namespace yogurt_and_clothes_cost_l2669_266975

/-- The total cost of buying a yogurt and a set of clothes -/
def total_cost (yogurt_price : ℕ) (clothes_price_multiplier : ℕ) : ℕ :=
  yogurt_price + yogurt_price * clothes_price_multiplier

/-- Theorem: The total cost of buying a yogurt priced at 120 yuan and a set of clothes
    priced at 6 times the yogurt's price is equal to 840 yuan. -/
theorem yogurt_and_clothes_cost :
  total_cost 120 6 = 840 := by
  sorry

end yogurt_and_clothes_cost_l2669_266975


namespace gcd_85_357_is_1_l2669_266979

theorem gcd_85_357_is_1 : Nat.gcd 85 357 = 1 := by
  sorry

end gcd_85_357_is_1_l2669_266979


namespace sugar_harvesting_solution_l2669_266993

/-- Represents the sugar harvesting problem with ants -/
def sugar_harvesting_problem (initial_sugar : ℝ) (harvest_rate : ℝ) (remaining_time : ℝ) : Prop :=
  ∃ (harvesting_time : ℝ),
    initial_sugar - harvest_rate * harvesting_time = harvest_rate * remaining_time ∧
    harvesting_time > 0

/-- Theorem stating the solution to the sugar harvesting problem -/
theorem sugar_harvesting_solution :
  sugar_harvesting_problem 24 4 3 →
  ∃ (harvesting_time : ℝ), harvesting_time = 3 :=
by
  sorry

end sugar_harvesting_solution_l2669_266993


namespace inscribed_circle_area_ratio_l2669_266914

theorem inscribed_circle_area_ratio (h r b : ℝ) : 
  h > 0 → r > 0 → b > 0 →
  (b + r)^2 + b^2 = h^2 →
  (2 * π * r^2) / ((b + r + h) * r) = 2 * π * r / (2 * b + r + h) := by
sorry

end inscribed_circle_area_ratio_l2669_266914


namespace final_crayon_count_l2669_266985

/-- Represents the number of crayons in a drawer after a series of actions. -/
def crayons_in_drawer (initial : ℕ) (mary_takes : ℕ) (mark_takes : ℕ) (mary_returns : ℕ) (sarah_adds : ℕ) (john_takes : ℕ) : ℕ :=
  initial - mary_takes - mark_takes + mary_returns + sarah_adds - john_takes

/-- Theorem stating that given the initial number of crayons and the actions performed, 
    the final number of crayons in the drawer is 4. -/
theorem final_crayon_count :
  crayons_in_drawer 7 3 2 1 5 4 = 4 := by
  sorry

end final_crayon_count_l2669_266985


namespace six_digit_difference_not_divisible_l2669_266983

theorem six_digit_difference_not_divisible (A B : ℕ) : 
  100 ≤ A ∧ A < 1000 → 100 ≤ B ∧ B < 1000 → A ≠ B → 
  ¬(∃ k : ℤ, 999 * (A - B) = 1976 * k) := by
  sorry

end six_digit_difference_not_divisible_l2669_266983


namespace last_digit_congruence_l2669_266913

theorem last_digit_congruence (N : ℕ) : ∃ (a b : ℕ), N = 10 * a + b ∧ b < 10 →
  (N ≡ b [ZMOD 10]) ∧ (N ≡ b [ZMOD 2]) ∧ (N ≡ b [ZMOD 5]) := by
  sorry

end last_digit_congruence_l2669_266913


namespace quadratic_inequality_ordering_l2669_266950

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_ordering (a b c : ℝ) :
  (∀ x, f a b c x > 0 ↔ x < -2 ∨ x > 4) →
  f a b c 2 < f a b c (-1) ∧ f a b c (-1) < f a b c 5 :=
sorry

end quadratic_inequality_ordering_l2669_266950


namespace trigonometric_identities_l2669_266976

theorem trigonometric_identities (α : Real) (h : Real.tan α = -3/4) :
  (Real.sin (2 * Real.pi - α) + Real.cos (5/2 * Real.pi + α)) / Real.sin (α - Real.pi/2) = -3/2 ∧
  (Real.sin α + Real.cos α) / (Real.sin α - 2 * Real.cos α) = -1/11 := by
  sorry

end trigonometric_identities_l2669_266976


namespace total_votes_l2669_266918

theorem total_votes (veggies : ℕ) (meat : ℕ) (dairy : ℕ) (plant_protein : ℕ)
  (h1 : veggies = 337)
  (h2 : meat = 335)
  (h3 : dairy = 274)
  (h4 : plant_protein = 212) :
  veggies + meat + dairy + plant_protein = 1158 :=
by sorry

end total_votes_l2669_266918


namespace opposites_sum_l2669_266917

theorem opposites_sum (a b : ℝ) : 
  (|a - 2| = -(b + 5)^2) → (a + b = -3) := by
  sorry

end opposites_sum_l2669_266917


namespace grocery_store_salary_l2669_266942

/-- Calculates the total daily salary of all employees in a grocery store -/
def total_daily_salary (owner_salary : ℕ) (manager_salary : ℕ) (cashier_salary : ℕ) 
  (clerk_salary : ℕ) (bagger_salary : ℕ) (num_owners : ℕ) (num_managers : ℕ) 
  (num_cashiers : ℕ) (num_clerks : ℕ) (num_baggers : ℕ) : ℕ :=
  owner_salary * num_owners + manager_salary * num_managers + 
  cashier_salary * num_cashiers + clerk_salary * num_clerks + 
  bagger_salary * num_baggers

theorem grocery_store_salary : 
  total_daily_salary 20 15 10 5 3 1 3 5 7 9 = 177 := by
  sorry

end grocery_store_salary_l2669_266942


namespace petya_vasya_meeting_l2669_266943

/-- The number of lanterns along the alley -/
def num_lanterns : ℕ := 100

/-- The position where Petya is observed -/
def petya_observed : ℕ := 22

/-- The position where Vasya is observed -/
def vasya_observed : ℕ := 88

/-- The function to calculate the meeting point of Petya and Vasya -/
def meeting_point (n l p v : ℕ) : ℕ :=
  ((n - 1) - (l - p)) + 1

theorem petya_vasya_meeting :
  meeting_point num_lanterns petya_observed vasya_observed 1 = 64 := by
  sorry

end petya_vasya_meeting_l2669_266943


namespace owen_final_turtles_l2669_266963

def turtles_problem (owen_initial johanna_initial owen_after_month johanna_after_month owen_final : ℕ) : Prop :=
  (johanna_initial = owen_initial - 5) ∧
  (owen_after_month = 2 * owen_initial) ∧
  (johanna_after_month = johanna_initial / 2) ∧
  (owen_final = owen_after_month + johanna_after_month)

theorem owen_final_turtles :
  ∃ (owen_initial johanna_initial owen_after_month johanna_after_month owen_final : ℕ),
    turtles_problem owen_initial johanna_initial owen_after_month johanna_after_month owen_final ∧
    owen_initial = 21 ∧
    owen_final = 50 :=
by sorry

end owen_final_turtles_l2669_266963


namespace probability_of_two_triples_l2669_266955

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (ranks : Nat)
  (cards_per_rank : Nat)
  (h_total : total_cards = ranks * cards_per_rank)

/-- Represents the specific hand we're looking for -/
structure TargetHand :=
  (total_cards : Nat)
  (sets : Nat)
  (cards_per_set : Nat)
  (h_total : total_cards = sets * cards_per_set)

def probability_of_target_hand (d : Deck) (h : TargetHand) : ℚ :=
  (d.ranks.choose h.sets) * (d.cards_per_rank.choose h.cards_per_set)^h.sets /
  d.total_cards.choose h.total_cards

theorem probability_of_two_triples (d : Deck) (h : TargetHand) :
  d.total_cards = 52 →
  d.ranks = 13 →
  d.cards_per_rank = 4 →
  h.total_cards = 6 →
  h.sets = 2 →
  h.cards_per_set = 3 →
  probability_of_target_hand d h = 13 / 106470 :=
sorry

end probability_of_two_triples_l2669_266955


namespace no_leftover_eggs_l2669_266999

/-- The number of eggs Abigail has -/
def abigail_eggs : ℕ := 28

/-- The number of eggs Beatrice has -/
def beatrice_eggs : ℕ := 53

/-- The number of eggs Carson has -/
def carson_eggs : ℕ := 19

/-- The number of eggs in each carton -/
def carton_size : ℕ := 10

/-- Theorem stating that the remainder of the total number of eggs divided by the carton size is 0 -/
theorem no_leftover_eggs : (abigail_eggs + beatrice_eggs + carson_eggs) % carton_size = 0 := by
  sorry

end no_leftover_eggs_l2669_266999


namespace extremum_of_f_l2669_266964

/-- The function f(x) = (k-1)x^2 - 2(k-1)x - k -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x^2 - 2 * (k - 1) * x - k

/-- Theorem: Extremum of f(x) when k ≠ 1 -/
theorem extremum_of_f (k : ℝ) (h : k ≠ 1) :
  (k > 1 → ∀ x, f k x ≥ -2 * k + 1) ∧
  (k < 1 → ∀ x, f k x ≤ -2 * k + 1) := by
  sorry

end extremum_of_f_l2669_266964


namespace translation_result_l2669_266930

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the translation operation
def translate (p : Point) (dx dy : ℝ) : Point :=
  (p.1 + dx, p.2 + dy)

-- Theorem statement
theorem translation_result :
  let A : Point := (-1, 4)
  let B : Point := translate A 5 3
  B = (4, 7) := by sorry

end translation_result_l2669_266930


namespace circumscribed_sphere_surface_area_l2669_266969

/-- The surface area of a sphere circumscribing a regular square pyramid -/
theorem circumscribed_sphere_surface_area
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base : base_edge = 2)
  (h_lateral : lateral_edge = Real.sqrt 3)
  : (4 : ℝ) * Real.pi * ((3 : ℝ) / 2) ^ 2 = 9 * Real.pi :=
sorry

end circumscribed_sphere_surface_area_l2669_266969
