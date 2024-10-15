import Mathlib

namespace NUMINAMATH_CALUDE_coefficient_equals_21_l306_30632

-- Define the coefficient of x^2 in the expansion of (ax+1)^5(x+1)^2
def coefficient (a : ℝ) : ℝ := 10 * a^2 + 10 * a + 1

-- Theorem statement
theorem coefficient_equals_21 (a : ℝ) : 
  coefficient a = 21 ↔ a = 1 ∨ a = -2 := by
  sorry


end NUMINAMATH_CALUDE_coefficient_equals_21_l306_30632


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l306_30634

theorem fifteenth_student_age
  (total_students : Nat)
  (average_age : ℚ)
  (group1_size group2_size : Nat)
  (group1_average group2_average : ℚ)
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_size = 7)
  (h4 : group2_size = 7)
  (h5 : group1_average = 14)
  (h6 : group2_average = 16) :
  (total_students * average_age - (group1_size * group1_average + group2_size * group2_average)) / (total_students - group1_size - group2_size) = 15 := by
  sorry


end NUMINAMATH_CALUDE_fifteenth_student_age_l306_30634


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l306_30654

/-- The lateral surface area of a cone with an equilateral triangle cross-section --/
theorem cone_lateral_surface_area (r h : Real) : 
  r^2 + h^2 = 1 →  -- Condition for equilateral triangle with side length 2
  r * h = 1/2 →    -- Condition for equilateral triangle with side length 2
  2 * π * r = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l306_30654


namespace NUMINAMATH_CALUDE_aarti_work_completion_l306_30655

/-- The number of days Aarti needs to complete one piece of work -/
def days_for_one_work : ℕ := 5

/-- The number of times the work is multiplied -/
def work_multiplier : ℕ := 3

/-- Theorem: Aarti will complete three times the work in 15 days -/
theorem aarti_work_completion :
  days_for_one_work * work_multiplier = 15 := by
  sorry

end NUMINAMATH_CALUDE_aarti_work_completion_l306_30655


namespace NUMINAMATH_CALUDE_power_of_power_l306_30665

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l306_30665


namespace NUMINAMATH_CALUDE_cube_volume_doubled_edges_l306_30618

theorem cube_volume_doubled_edges (a : ℝ) (h : a > 0) :
  (2 * a)^3 = 8 * a^3 := by sorry

end NUMINAMATH_CALUDE_cube_volume_doubled_edges_l306_30618


namespace NUMINAMATH_CALUDE_largest_possible_a_l306_30611

theorem largest_possible_a :
  ∀ (a b c d e : ℕ),
    a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    a < 3 * b →
    b < 4 * c →
    c < 5 * d →
    e = d - 10 →
    e < 105 →
    a ≤ 6824 ∧ ∃ (a' b' c' d' e' : ℕ),
      a' = 6824 ∧
      b' > 0 ∧ c' > 0 ∧ d' > 0 ∧ e' > 0 ∧
      a' < 3 * b' ∧
      b' < 4 * c' ∧
      c' < 5 * d' ∧
      e' = d' - 10 ∧
      e' < 105 :=
by
  sorry


end NUMINAMATH_CALUDE_largest_possible_a_l306_30611


namespace NUMINAMATH_CALUDE_solution_system_equations_l306_30660

theorem solution_system_equations :
  ∃! (x y : ℝ), 
    x + Real.sqrt (x + 2*y) - 2*y = 7/2 ∧
    x^2 + x + 2*y - 4*y^2 = 27/2 ∧
    x = 19/4 ∧ y = 17/8 := by
  sorry

end NUMINAMATH_CALUDE_solution_system_equations_l306_30660


namespace NUMINAMATH_CALUDE_cubic_system_solution_l306_30627

theorem cubic_system_solution (x y z : ℝ) : 
  ((x + y)^3 = z ∧ (y + z)^3 = x ∧ (z + x)^3 = y) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = Real.sqrt 2 / 4 ∧ y = Real.sqrt 2 / 4 ∧ z = Real.sqrt 2 / 4) ∨ 
   (x = -Real.sqrt 2 / 4 ∧ y = -Real.sqrt 2 / 4 ∧ z = -Real.sqrt 2 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_system_solution_l306_30627


namespace NUMINAMATH_CALUDE_rug_profit_calculation_l306_30651

/-- Calculate the profit from selling rugs -/
theorem rug_profit_calculation (cost_price selling_price number_of_rugs : ℕ) :
  let profit_per_rug := selling_price - cost_price
  let total_profit := number_of_rugs * profit_per_rug
  total_profit = number_of_rugs * (selling_price - cost_price) :=
by sorry

end NUMINAMATH_CALUDE_rug_profit_calculation_l306_30651


namespace NUMINAMATH_CALUDE_triangle_properties_l306_30669

theorem triangle_properties (A B C : Real) (a b c : Real) 
  (m_x m_y n_x n_y : Real → Real) :
  (∀ θ, m_x θ = 2 * Real.cos θ ∧ m_y θ = 1) →
  (∀ θ, n_x θ = 1 ∧ n_y θ = Real.sin (θ + Real.pi / 6)) →
  (∃ k : Real, k ≠ 0 ∧ ∀ θ, m_x θ * k = n_x θ ∧ m_y θ * k = n_y θ) →
  a = 2 * Real.sqrt 3 →
  c = 4 →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  A = Real.pi / 3 ∧ 
  b = 2 ∧ 
  1/2 * b * c * Real.sin A = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l306_30669


namespace NUMINAMATH_CALUDE_sqrt_88200_simplification_l306_30698

theorem sqrt_88200_simplification : Real.sqrt 88200 = 210 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_88200_simplification_l306_30698


namespace NUMINAMATH_CALUDE_committee_combinations_l306_30610

theorem committee_combinations : Nat.choose 8 5 = 56 := by sorry

end NUMINAMATH_CALUDE_committee_combinations_l306_30610


namespace NUMINAMATH_CALUDE_dvd_discount_l306_30621

/-- The discount on each pack of DVDs, given the original price and the discounted price for multiple packs. -/
theorem dvd_discount (original_price : ℕ) (num_packs : ℕ) (total_price : ℕ) : 
  original_price = 107 → num_packs = 93 → total_price = 93 → 
  (original_price - (total_price / num_packs) : ℕ) = 106 :=
by sorry

end NUMINAMATH_CALUDE_dvd_discount_l306_30621


namespace NUMINAMATH_CALUDE_tims_interest_rate_l306_30668

/-- Calculates the compound interest after n years -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

theorem tims_interest_rate :
  let tim_principal : ℝ := 600
  let lana_principal : ℝ := 1000
  let lana_rate : ℝ := 0.05
  let years : ℕ := 2
  ∀ tim_rate : ℝ,
    (compoundInterest tim_principal tim_rate years - tim_principal) =
    (compoundInterest lana_principal lana_rate years - lana_principal) + 23.5 →
    tim_rate = 0.1 := by
  sorry

#check tims_interest_rate

end NUMINAMATH_CALUDE_tims_interest_rate_l306_30668


namespace NUMINAMATH_CALUDE_second_bell_interval_l306_30679

def bell_intervals (x : ℕ) : List ℕ := [5, x, 11, 15]

theorem second_bell_interval (x : ℕ) :
  (∃ (k : ℕ), k > 0 ∧ k * (Nat.lcm (Nat.lcm (Nat.lcm 5 x) 11) 15) = 1320) →
  x = 8 :=
by sorry

end NUMINAMATH_CALUDE_second_bell_interval_l306_30679


namespace NUMINAMATH_CALUDE_rectangle_max_area_l306_30645

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) →  -- Perimeter condition
  (l * w ≤ 100) ∧         -- Area is at most 100
  (∃ l' w' : ℕ, 2 * l' + 2 * w' = 40 ∧ l' * w' = 100) -- Maximum area exists
  :=
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l306_30645


namespace NUMINAMATH_CALUDE_insufficient_shots_l306_30617

-- Define the number of points on the circle
def n : ℕ := 29

-- Define the number of shots
def shots : ℕ := 134

-- Function to calculate binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

-- Total number of possible triangles
def total_triangles : ℕ := 
  binomial_coefficient n 3

-- Number of triangles that can be hit by one shot
def triangles_per_shot : ℕ := n - 2

-- Maximum number of triangles that can be hit by all shots
def max_hit_triangles : ℕ := 
  shots * triangles_per_shot

-- Theorem stating that 134 shots are insufficient
theorem insufficient_shots : max_hit_triangles < total_triangles := by
  sorry


end NUMINAMATH_CALUDE_insufficient_shots_l306_30617


namespace NUMINAMATH_CALUDE_chairs_to_hall_l306_30612

/-- Calculates the total number of chairs taken to the hall given the number of students,
    chairs per trip, and number of trips. -/
def totalChairs (students : ℕ) (chairsPerTrip : ℕ) (numTrips : ℕ) : ℕ :=
  students * chairsPerTrip * numTrips

/-- Proves that 5 students, each carrying 5 chairs per trip and making 10 trips,
    will take a total of 250 chairs to the hall. -/
theorem chairs_to_hall :
  totalChairs 5 5 10 = 250 := by
  sorry

#eval totalChairs 5 5 10

end NUMINAMATH_CALUDE_chairs_to_hall_l306_30612


namespace NUMINAMATH_CALUDE_shelter_animals_count_l306_30661

theorem shelter_animals_count (cats : ℕ) (dogs : ℕ) 
  (h1 : cats = 645) (h2 : dogs = 567) : cats + dogs = 1212 := by
  sorry

end NUMINAMATH_CALUDE_shelter_animals_count_l306_30661


namespace NUMINAMATH_CALUDE_abs_z_equals_sqrt_two_l306_30693

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem abs_z_equals_sqrt_two (z : ℂ) (h : z * (1 + i) = 2 * i) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_equals_sqrt_two_l306_30693


namespace NUMINAMATH_CALUDE_expansion_properties_l306_30664

def polynomial_expansion (x : ℝ) (a : Fin 8 → ℝ) : Prop :=
  (1 - 2*x)^7 = a 0 + a 1*x + a 2*x^2 + a 3*x^3 + a 4*x^4 + a 5*x^5 + a 6*x^6 + a 7*x^7

theorem expansion_properties (a : Fin 8 → ℝ) 
  (h : ∀ x, polynomial_expansion x a) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = -2) ∧
  (a 1 + a 3 + a 5 + a 7 = -1094) ∧
  (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| = 2187) := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l306_30664


namespace NUMINAMATH_CALUDE_expression_evaluation_l306_30691

theorem expression_evaluation (b : ℝ) (a : ℝ) (h1 : b = 2) (h2 : a = b + 3) :
  (a^2 + b)^2 - (a^2 - b)^2 = 200 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l306_30691


namespace NUMINAMATH_CALUDE_exam_score_proof_l306_30602

theorem exam_score_proof (mean : ℝ) (low_score : ℝ) (std_dev_below : ℝ) (std_dev_above : ℝ) :
  mean = 88.8 →
  low_score = 86 →
  std_dev_below = 7 →
  std_dev_above = 3 →
  low_score = mean - std_dev_below * ((mean - low_score) / std_dev_below) →
  mean + std_dev_above * ((mean - low_score) / std_dev_below) = 90 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_proof_l306_30602


namespace NUMINAMATH_CALUDE_jims_taxi_additional_charge_l306_30687

/-- The additional charge for each 2/5 of a mile in Jim's taxi service -/
def additional_charge (initial_fee total_distance total_charge : ℚ) : ℚ :=
  ((total_charge - initial_fee) * 2) / (5 * total_distance)

/-- Theorem stating the additional charge for each 2/5 of a mile in Jim's taxi service -/
theorem jims_taxi_additional_charge :
  additional_charge (5/2) (36/10) (565/100) = 35/100 := by
  sorry

end NUMINAMATH_CALUDE_jims_taxi_additional_charge_l306_30687


namespace NUMINAMATH_CALUDE_shopkeeper_stock_worth_l306_30620

def item_A_profit_percentage : Real := 0.15
def item_A_loss_percentage : Real := 0.10
def item_A_profit_portion : Real := 0.25
def item_A_loss_portion : Real := 0.75

def item_B_profit_percentage : Real := 0.20
def item_B_loss_percentage : Real := 0.05
def item_B_profit_portion : Real := 0.30
def item_B_loss_portion : Real := 0.70

def item_C_profit_percentage : Real := 0.10
def item_C_loss_percentage : Real := 0.08
def item_C_profit_portion : Real := 0.40
def item_C_loss_portion : Real := 0.60

def tax_rate : Real := 0.12
def net_loss : Real := 750

def cost_price_ratio_A : Real := 2
def cost_price_ratio_B : Real := 3
def cost_price_ratio_C : Real := 4

theorem shopkeeper_stock_worth (x : Real) :
  let cost_A := cost_price_ratio_A * x
  let cost_B := cost_price_ratio_B * x
  let cost_C := cost_price_ratio_C * x
  let profit_loss_A := item_A_profit_portion * cost_A * item_A_profit_percentage - 
                       item_A_loss_portion * cost_A * item_A_loss_percentage
  let profit_loss_B := item_B_profit_portion * cost_B * item_B_profit_percentage - 
                       item_B_loss_portion * cost_B * item_B_loss_percentage
  let profit_loss_C := item_C_profit_portion * cost_C * item_C_profit_percentage - 
                       item_C_loss_portion * cost_C * item_C_loss_percentage
  let total_profit_loss := profit_loss_A + profit_loss_B + profit_loss_C
  total_profit_loss = -net_loss →
  cost_A = 46875 ∧ cost_B = 70312.5 ∧ cost_C = 93750 := by
sorry


end NUMINAMATH_CALUDE_shopkeeper_stock_worth_l306_30620


namespace NUMINAMATH_CALUDE_sum_equals_300_l306_30695

theorem sum_equals_300 : 157 + 43 + 19 + 81 = 300 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_300_l306_30695


namespace NUMINAMATH_CALUDE_coin_problem_l306_30636

def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

def total_coins : ℕ := 11
def total_value : ℕ := 118

theorem coin_problem (p n d q : ℕ) : 
  p ≥ 1 → n ≥ 1 → d ≥ 1 → q ≥ 1 →
  p + n + d + q = total_coins →
  p * penny + n * nickel + d * dime + q * quarter = total_value →
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_coin_problem_l306_30636


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l306_30629

-- Define the purchase price, repair cost, and selling price
def purchase_price : ℚ := 900
def repair_cost : ℚ := 300
def selling_price : ℚ := 1260

-- Define the total cost
def total_cost : ℚ := purchase_price + repair_cost

-- Define the gain
def gain : ℚ := selling_price - total_cost

-- Define the gain percent
def gain_percent : ℚ := (gain / total_cost) * 100

-- Theorem to prove
theorem scooter_gain_percent : gain_percent = 5 := by
  sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l306_30629


namespace NUMINAMATH_CALUDE_cone_slant_height_l306_30688

/-- The slant height of a cone given its base radius and curved surface area -/
theorem cone_slant_height (r : ℝ) (csa : ℝ) (h1 : r = 10) (h2 : csa = 628.3185307179587) :
  csa / (π * r) = 20 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l306_30688


namespace NUMINAMATH_CALUDE_fermat_numbers_coprime_l306_30670

theorem fermat_numbers_coprime (m n : ℕ) (h : m ≠ n) :
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fermat_numbers_coprime_l306_30670


namespace NUMINAMATH_CALUDE_total_visitors_three_days_l306_30625

/-- The number of visitors to Buckingham Palace on the day Rachel visited -/
def visitors_rachel_day : ℕ := 92

/-- The number of visitors to Buckingham Palace on the day before Rachel's visit -/
def visitors_previous_day : ℕ := 419

/-- The number of visitors to Buckingham Palace two days before Rachel's visit -/
def visitors_two_days_before : ℕ := 103

/-- Theorem stating that the total number of visitors over the three known days is 614 -/
theorem total_visitors_three_days : 
  visitors_rachel_day + visitors_previous_day + visitors_two_days_before = 614 := by
  sorry

end NUMINAMATH_CALUDE_total_visitors_three_days_l306_30625


namespace NUMINAMATH_CALUDE_cuboid_third_edge_length_l306_30652

/-- Given a cuboid with two edges of 4 cm each and a volume of 96 cm³, 
    prove that the length of the third edge is 6 cm. -/
theorem cuboid_third_edge_length 
  (edge1 : ℝ) (edge2 : ℝ) (volume : ℝ) (third_edge : ℝ) 
  (h1 : edge1 = 4) 
  (h2 : edge2 = 4) 
  (h3 : volume = 96) 
  (h4 : volume = edge1 * edge2 * third_edge) : 
  third_edge = 6 :=
sorry

end NUMINAMATH_CALUDE_cuboid_third_edge_length_l306_30652


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l306_30641

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than_60 (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < 60 → ¬(p ∣ n)

theorem smallest_number_with_conditions : 
  (∀ n : ℕ, n < 4087 → 
    is_prime n ∨ 
    is_square n ∨ 
    ¬(has_no_prime_factor_less_than_60 n)) ∧ 
  ¬(is_prime 4087) ∧ 
  ¬(is_square 4087) ∧ 
  has_no_prime_factor_less_than_60 4087 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l306_30641


namespace NUMINAMATH_CALUDE_dress_final_price_l306_30614

/-- The final price of a dress after discounts and taxes -/
def final_price (d : ℝ) : ℝ :=
  let discount_price := d * (1 - 0.45)
  let staff_discount_price := discount_price * (1 - 0.40)
  let employee_month_price := staff_discount_price * (1 - 0.10)
  let local_tax_price := employee_month_price * (1 + 0.08)
  local_tax_price * (1 + 0.05)

/-- Theorem stating the final price of the dress -/
theorem dress_final_price (d : ℝ) :
  final_price d = 0.3549 * d := by
  sorry

end NUMINAMATH_CALUDE_dress_final_price_l306_30614


namespace NUMINAMATH_CALUDE_abs_equation_solution_l306_30647

theorem abs_equation_solution (x : ℝ) : |-5 + x| = 3 → x = 8 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l306_30647


namespace NUMINAMATH_CALUDE_private_teacher_cost_l306_30639

/-- Calculates the amount each parent must pay for a private teacher --/
theorem private_teacher_cost 
  (former_salary : ℕ) 
  (raise_percentage : ℚ) 
  (num_kids : ℕ) 
  (h1 : former_salary = 45000)
  (h2 : raise_percentage = 1/5)
  (h3 : num_kids = 9) :
  (former_salary + former_salary * raise_percentage) / num_kids = 6000 := by
  sorry

#check private_teacher_cost

end NUMINAMATH_CALUDE_private_teacher_cost_l306_30639


namespace NUMINAMATH_CALUDE_stating_remaining_slices_is_four_l306_30659

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

end NUMINAMATH_CALUDE_stating_remaining_slices_is_four_l306_30659


namespace NUMINAMATH_CALUDE_no_leftover_eggs_l306_30692

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

end NUMINAMATH_CALUDE_no_leftover_eggs_l306_30692


namespace NUMINAMATH_CALUDE_circles_common_internal_tangent_l306_30694

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def circle_O₂ (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 = 9

-- Define the center of circle O₂
def center_O₂ : ℝ × ℝ := (3, 3)

-- Define the property of being externally tangent
def externally_tangent (O₁ O₂ : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), O₁ x y ∧ O₂ x y ∧
  ∀ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) → ¬(O₁ x' y' ∧ O₂ x' y')

-- Define the common internal tangent line
def common_internal_tangent (x y : ℝ) : Prop := 3*x + 4*y - 21 = 0

-- State the theorem
theorem circles_common_internal_tangent :
  externally_tangent circle_O₁ circle_O₂ →
  ∀ (x y : ℝ), common_internal_tangent x y ↔
    (∃ (t : ℝ), circle_O₁ (x + t) (y - (3/4)*t) ∧
               circle_O₂ (x - t) (y + (3/4)*t)) :=
sorry

end NUMINAMATH_CALUDE_circles_common_internal_tangent_l306_30694


namespace NUMINAMATH_CALUDE_temperature_range_l306_30685

theorem temperature_range (highest_temp lowest_temp t : ℝ) 
  (h_highest : highest_temp = 30)
  (h_lowest : lowest_temp = 20)
  (h_range : lowest_temp ≤ t ∧ t ≤ highest_temp) :
  20 ≤ t ∧ t ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_temperature_range_l306_30685


namespace NUMINAMATH_CALUDE_smallest_square_cover_l306_30626

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Checks if a square can be perfectly covered by rectangles of a given size -/
def canCoverSquare (s : Square) (r : Rectangle) : Prop :=
  ∃ n : ℕ, n * r.width * r.height = s.side * s.side ∧ 
    s.side % r.width = 0 ∧ s.side % r.height = 0

/-- The number of rectangles needed to cover the square -/
def numRectangles (s : Square) (r : Rectangle) : ℕ :=
  (s.side * s.side) / (r.width * r.height)

theorem smallest_square_cover :
  ∃ (s : Square) (r : Rectangle), 
    r.width = 3 ∧ r.height = 4 ∧
    canCoverSquare s r ∧
    (∀ (s' : Square), s'.side < s.side → ¬ canCoverSquare s' r) ∧
    numRectangles s r = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_cover_l306_30626


namespace NUMINAMATH_CALUDE_tiling_count_is_96_l306_30657

/-- Represents a tile with width and height -/
structure Tile :=
  (width : Nat)
  (height : Nat)

/-- Represents a rectangle with width and height -/
structure Rectangle :=
  (width : Nat)
  (height : Nat)

/-- Represents a set of tiles -/
def TileSet := List Tile

/-- Counts the number of ways to tile a rectangle with a given set of tiles -/
def tileCount (r : Rectangle) (ts : TileSet) : Nat :=
  sorry

/-- The set of tiles for our problem -/
def problemTiles : TileSet :=
  [⟨1, 1⟩, ⟨1, 2⟩, ⟨1, 3⟩, ⟨1, 4⟩, ⟨1, 5⟩]

/-- The main theorem stating that the number of tilings is 96 -/
theorem tiling_count_is_96 :
  tileCount ⟨5, 3⟩ problemTiles = 96 :=
sorry

end NUMINAMATH_CALUDE_tiling_count_is_96_l306_30657


namespace NUMINAMATH_CALUDE_expansion_has_six_nonzero_terms_l306_30686

/-- The polynomial resulting from expanding (2x^3 - 4)(3x^2 + 5x - 7) + 5 (x^4 - 3x^3 + 2x^2) -/
def expanded_polynomial (x : ℝ) : ℝ :=
  6*x^5 + 15*x^4 - 29*x^3 - 2*x^2 - 20*x + 28

/-- The coefficients of the expanded polynomial -/
def coefficients : List ℝ := [6, 15, -29, -2, -20, 28]

/-- Theorem stating that the expansion has exactly 6 nonzero terms -/
theorem expansion_has_six_nonzero_terms :
  coefficients.length = 6 ∧ coefficients.all (· ≠ 0) := by sorry

end NUMINAMATH_CALUDE_expansion_has_six_nonzero_terms_l306_30686


namespace NUMINAMATH_CALUDE_divisibility_conditions_l306_30635

theorem divisibility_conditions (a b : ℕ) : 
  (∃ k : ℤ, a^3 * b - 1 = k * (a + 1)) ∧ 
  (∃ m : ℤ, a * b^3 + 1 = m * (b - 1)) → 
  ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_conditions_l306_30635


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l306_30630

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 - 3 * Complex.I) : 
  z.im = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l306_30630


namespace NUMINAMATH_CALUDE_range_of_m_and_n_l306_30603

-- Define sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + m > 0}
def B (n : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - n ≤ 0}

-- Define point P
def P : ℝ × ℝ := (2, 3)

-- Theorem statement
theorem range_of_m_and_n (m n : ℝ) 
  (h1 : P ∈ A m) 
  (h2 : P ∉ B n) : 
  m > -1 ∧ n < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_and_n_l306_30603


namespace NUMINAMATH_CALUDE_yannas_cookies_l306_30699

/-- Yanna's cookie baking problem -/
theorem yannas_cookies
  (morning_butter_cookies : ℕ)
  (morning_biscuits : ℕ)
  (afternoon_butter_cookies : ℕ)
  (afternoon_biscuits : ℕ)
  (h1 : morning_butter_cookies = 20)
  (h2 : morning_biscuits = 40)
  (h3 : afternoon_butter_cookies = 10)
  (h4 : afternoon_biscuits = 20) :
  (morning_biscuits + afternoon_biscuits) - (morning_butter_cookies + afternoon_butter_cookies) = 30 :=
by sorry

end NUMINAMATH_CALUDE_yannas_cookies_l306_30699


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l306_30667

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -(1 / 2) := by sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l306_30667


namespace NUMINAMATH_CALUDE_min_cost_is_800_l306_30689

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

end NUMINAMATH_CALUDE_min_cost_is_800_l306_30689


namespace NUMINAMATH_CALUDE_base_conversion_1765_l306_30672

/-- Converts a base 10 number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 6 to a natural number in base 10 -/
def fromBase6 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_1765 :
  toBase6 1765 = [1, 2, 1, 0, 1] ∧ fromBase6 [1, 2, 1, 0, 1] = 1765 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1765_l306_30672


namespace NUMINAMATH_CALUDE_cades_remaining_marbles_l306_30642

/-- Proves that Cade has 79 marbles left after giving away 8 marbles from his initial 87 marbles. -/
theorem cades_remaining_marbles (initial_marbles : ℕ) (marbles_given_away : ℕ) 
  (h1 : initial_marbles = 87) 
  (h2 : marbles_given_away = 8) : 
  initial_marbles - marbles_given_away = 79 := by
  sorry

end NUMINAMATH_CALUDE_cades_remaining_marbles_l306_30642


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l306_30615

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 1 → a^2 > 1) ∧
  (∃ a, a^2 > 1 ∧ ¬(a > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l306_30615


namespace NUMINAMATH_CALUDE_lost_card_number_l306_30684

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ≤ n) : 
  ∃ (x : ℕ), x > 0 ∧ x ≤ n ∧ (n * (n + 1)) / 2 - x = 101 ∧ x = 4 := by
  sorry

#check lost_card_number

end NUMINAMATH_CALUDE_lost_card_number_l306_30684


namespace NUMINAMATH_CALUDE_exam_score_distribution_l306_30640

/-- Represents the normal distribution of exam scores -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ

/-- Represents the class and exam information -/
structure ExamInfo where
  totalStudents : ℕ
  scoreDistribution : NormalDistribution
  middleProbability : ℝ

/-- Calculates the number of students with scores above a given threshold -/
def studentsAboveThreshold (info : ExamInfo) (threshold : ℝ) : ℕ :=
  sorry

theorem exam_score_distribution (info : ExamInfo) :
  info.totalStudents = 50 ∧
  info.scoreDistribution = { μ := 110, σ := 10 } ∧
  info.middleProbability = 0.34 →
  studentsAboveThreshold info 120 = 8 :=
sorry

end NUMINAMATH_CALUDE_exam_score_distribution_l306_30640


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l306_30601

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 < 0} = Set.Ioo (-2 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l306_30601


namespace NUMINAMATH_CALUDE_composite_power_plus_four_l306_30690

theorem composite_power_plus_four (n : ℕ) (h : n > 1) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^2020 + 4 = a * b :=
by sorry

end NUMINAMATH_CALUDE_composite_power_plus_four_l306_30690


namespace NUMINAMATH_CALUDE_product_purchase_savings_l306_30653

/-- Proves that under given conditions, the product could have been purchased for 10% less -/
theorem product_purchase_savings (original_selling_price : ℝ) 
  (h1 : original_selling_price = 989.9999999999992)
  (h2 : original_selling_price = 1.1 * original_purchase_price)
  (h3 : 1.3 * reduced_purchase_price = original_selling_price + 63) :
  (original_purchase_price - reduced_purchase_price) / original_purchase_price = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_product_purchase_savings_l306_30653


namespace NUMINAMATH_CALUDE_circle_O1_equation_constant_sum_of_squares_l306_30646

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

end NUMINAMATH_CALUDE_circle_O1_equation_constant_sum_of_squares_l306_30646


namespace NUMINAMATH_CALUDE_part_one_part_two_l306_30682

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x ∈ Set.Icc 1 2, (a - 1) * x - 1 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 1 > 0

-- Theorem for part (1)
theorem part_one (a : ℝ) : (p a ∧ q a) ↔ (3/2 < a ∧ a < 2) :=
sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : (¬(¬(p a) ∧ q a) ∧ (¬(p a) ∨ q a)) ↔ (a ≤ -2 ∨ (3/2 < a ∧ a < 2)) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l306_30682


namespace NUMINAMATH_CALUDE_lateral_side_is_five_l306_30643

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ
  lateral : ℝ

/-- The property that the given dimensions form a valid isosceles trapezoid -/
def is_valid_trapezoid (t : IsoscelesTrapezoid) : Prop :=
  t.base1 > 0 ∧ t.base2 > 0 ∧ t.area > 0 ∧ t.lateral > 0 ∧
  t.area = (t.base1 + t.base2) * t.lateral / 2

/-- The theorem stating that the lateral side of the trapezoid is 5 -/
theorem lateral_side_is_five (t : IsoscelesTrapezoid)
  (h1 : t.base1 = 8)
  (h2 : t.base2 = 14)
  (h3 : t.area = 44)
  (h4 : is_valid_trapezoid t) :
  t.lateral = 5 :=
sorry

end NUMINAMATH_CALUDE_lateral_side_is_five_l306_30643


namespace NUMINAMATH_CALUDE_sues_mother_cookies_l306_30631

/-- The number of cookies Sue's mother made -/
def total_cookies (bags : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  bags * cookies_per_bag

/-- Proof that Sue's mother made 75 cookies -/
theorem sues_mother_cookies : total_cookies 25 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_sues_mother_cookies_l306_30631


namespace NUMINAMATH_CALUDE_female_lion_weight_l306_30677

theorem female_lion_weight (male_weight : ℚ) (weight_difference : ℚ) :
  male_weight = 145 / 4 →
  weight_difference = 47 / 10 →
  male_weight - weight_difference = 631 / 20 := by
  sorry

end NUMINAMATH_CALUDE_female_lion_weight_l306_30677


namespace NUMINAMATH_CALUDE_smartphone_price_reduction_l306_30649

theorem smartphone_price_reduction (original_price final_price : ℝ) 
  (h1 : original_price = 7500)
  (h2 : final_price = 4800)
  (h3 : ∃ (x : ℝ), final_price = original_price * (1 - x)^2 ∧ 0 < x ∧ x < 1) :
  ∃ (x : ℝ), final_price = original_price * (1 - x)^2 ∧ x = 0.2 :=
sorry

end NUMINAMATH_CALUDE_smartphone_price_reduction_l306_30649


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l306_30681

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

end NUMINAMATH_CALUDE_point_in_third_quadrant_l306_30681


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l306_30616

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third is determined by these two
  angle1 : ℝ
  angle2 : ℝ
  -- Condition: sum of angles is 180°
  sum_angles : angle1 + angle2 + (180 - angle1 - angle2) = 180
  -- Condition: at least two angles are equal (isosceles property)
  isosceles : angle1 = angle2 ∨ angle1 = (180 - angle1 - angle2) ∨ angle2 = (180 - angle1 - angle2)

-- Theorem statement
theorem isosceles_triangle_vertex_angle (t : IsoscelesTriangle) (h : t.angle1 = 70 ∨ t.angle2 = 70 ∨ (180 - t.angle1 - t.angle2) = 70) :
  t.angle1 = 70 ∨ t.angle2 = 70 ∨ (180 - t.angle1 - t.angle2) = 70 ∨
  t.angle1 = 40 ∨ t.angle2 = 40 ∨ (180 - t.angle1 - t.angle2) = 40 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l306_30616


namespace NUMINAMATH_CALUDE_rosie_apple_crisps_l306_30628

/-- The number of apple crisps Rosie can make with a given number of apples -/
def apple_crisps (apples : ℕ) : ℕ :=
  (3 * apples) / 12

theorem rosie_apple_crisps :
  apple_crisps 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_apple_crisps_l306_30628


namespace NUMINAMATH_CALUDE_existence_of_n_with_s_prime_divisors_l306_30658

theorem existence_of_n_with_s_prime_divisors (s : ℕ) (hs : s > 0) :
  ∃ n : ℕ, n > 0 ∧ (∃ (P : Finset Nat), P.card ≥ s ∧ 
    (∀ p ∈ P, Nat.Prime p ∧ p ∣ (2^n - 1))) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_n_with_s_prime_divisors_l306_30658


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l306_30604

theorem sphere_volume_increase (r : ℝ) (h : r > 0) :
  let new_r := r * Real.sqrt 2
  (4 / 3 * Real.pi * new_r^3) / (4 / 3 * Real.pi * r^3) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l306_30604


namespace NUMINAMATH_CALUDE_line_circle_intersection_k_l306_30697

/-- A line intersects a circle -/
structure LineCircleIntersection where
  /-- Slope of the line y = kx + 3 -/
  k : ℝ
  /-- The line intersects the circle (x-1)^2 + (y-2)^2 = 9 at two points -/
  intersects : k > 1
  /-- The distance between the two intersection points is 12√5/5 -/
  distance : ℝ
  distance_eq : distance = 12 * Real.sqrt 5 / 5

/-- The slope k of the line is 2 -/
theorem line_circle_intersection_k (lci : LineCircleIntersection) : lci.k = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_k_l306_30697


namespace NUMINAMATH_CALUDE_total_shelves_calculation_l306_30619

/-- Calculate the total number of shelves needed for coloring books and puzzle books --/
theorem total_shelves_calculation (initial_coloring : ℕ) (initial_puzzle : ℕ)
                                  (sold_coloring : ℕ) (sold_puzzle : ℕ)
                                  (coloring_per_shelf : ℕ) (puzzle_per_shelf : ℕ)
                                  (h1 : initial_coloring = 435)
                                  (h2 : initial_puzzle = 523)
                                  (h3 : sold_coloring = 218)
                                  (h4 : sold_puzzle = 304)
                                  (h5 : coloring_per_shelf = 17)
                                  (h6 : puzzle_per_shelf = 22) :
  (((initial_coloring - sold_coloring) + coloring_per_shelf - 1) / coloring_per_shelf +
   ((initial_puzzle - sold_puzzle) + puzzle_per_shelf - 1) / puzzle_per_shelf) = 23 := by
  sorry

#eval ((435 - 218) + 17 - 1) / 17 + ((523 - 304) + 22 - 1) / 22

end NUMINAMATH_CALUDE_total_shelves_calculation_l306_30619


namespace NUMINAMATH_CALUDE_dice_surface_sum_l306_30622

/-- The number of dice in the arrangement -/
def num_dice : Nat := 2012

/-- The sum of points on all faces of a single die -/
def die_sum : Nat := 21

/-- The sum of points on opposite faces of a die -/
def opposite_faces_sum : Nat := 7

/-- A value representing the number of points on one end face of the first die -/
def X : Fin 6 := sorry

/-- The sum of points on the surface of the arranged dice -/
def surface_sum : Nat := 28175 + 2 * X.val

theorem dice_surface_sum :
  surface_sum = num_dice * die_sum - (num_dice - 1) * opposite_faces_sum + 2 * X.val :=
by sorry

end NUMINAMATH_CALUDE_dice_surface_sum_l306_30622


namespace NUMINAMATH_CALUDE_fraction_subtraction_fraction_division_l306_30683

-- Problem 1
theorem fraction_subtraction (x y : ℝ) (h : x + y ≠ 0) :
  (2 * x + 3 * y) / (x + y) - (x + 2 * y) / (x + y) = 1 := by
sorry

-- Problem 2
theorem fraction_division (a : ℝ) (h : a ≠ 2) :
  (a^2 - 1) / (a^2 - 4*a + 4) / ((a - 1) / (a - 2)) = (a + 1) / (a - 2) := by
sorry

end NUMINAMATH_CALUDE_fraction_subtraction_fraction_division_l306_30683


namespace NUMINAMATH_CALUDE_count_ak_divisible_by_9_l306_30666

/-- The number obtained by writing the integers 1 to n from left to right -/
def a (n : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The count of a_k divisible by 9 for 1 ≤ k ≤ 100 -/
def countDivisibleBy9 : ℕ := sorry

theorem count_ak_divisible_by_9 : countDivisibleBy9 = 22 := by sorry

end NUMINAMATH_CALUDE_count_ak_divisible_by_9_l306_30666


namespace NUMINAMATH_CALUDE_soccer_team_games_theorem_l306_30623

/-- Represents the ratio of wins, losses, and ties for a soccer team -/
structure GameRatio :=
  (wins : ℕ)
  (losses : ℕ)
  (ties : ℕ)

/-- Calculates the total number of games played given a game ratio and number of losses -/
def totalGames (ratio : GameRatio) (numLosses : ℕ) : ℕ :=
  let gamesPerPart := numLosses / ratio.losses
  (ratio.wins + ratio.losses + ratio.ties) * gamesPerPart

/-- Theorem stating that for a team with a 4:3:1 win:loss:tie ratio and 9 losses, 
    the total number of games played is 24 -/
theorem soccer_team_games_theorem :
  let ratio : GameRatio := ⟨4, 3, 1⟩
  totalGames ratio 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_games_theorem_l306_30623


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l306_30675

theorem complex_number_quadrant : ∃ (z : ℂ), 
  z = (2 - Complex.I) / (2 + Complex.I) ∧ 
  0 < z.re ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l306_30675


namespace NUMINAMATH_CALUDE_derivative_at_negative_two_l306_30600

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem derivative_at_negative_two (h : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
  |((f (-2 + Δx) - f (-2 - Δx)) / Δx) - (-2)| < ε) : 
  deriv f (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_negative_two_l306_30600


namespace NUMINAMATH_CALUDE_sugar_harvesting_solution_l306_30673

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

end NUMINAMATH_CALUDE_sugar_harvesting_solution_l306_30673


namespace NUMINAMATH_CALUDE_gym_member_count_l306_30607

/-- Represents a gym with its pricing and revenue information -/
structure Gym where
  charge_per_half_month : ℕ
  monthly_revenue : ℕ

/-- Calculates the number of members in the gym -/
def member_count (g : Gym) : ℕ :=
  g.monthly_revenue / (2 * g.charge_per_half_month)

/-- Theorem stating that a gym with the given parameters has 300 members -/
theorem gym_member_count :
  ∃ (g : Gym), g.charge_per_half_month = 18 ∧ g.monthly_revenue = 10800 ∧ member_count g = 300 := by
  sorry

end NUMINAMATH_CALUDE_gym_member_count_l306_30607


namespace NUMINAMATH_CALUDE_inequality_proof_l306_30662

theorem inequality_proof (a b : ℝ) (h : a + b = 1) :
  Real.sqrt (1 + 5 * a^2) + 5 * Real.sqrt (2 + b^2) ≥ 9 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l306_30662


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l306_30606

-- Define the functions
def f (a b x : ℝ) : ℝ := -|x - a| + b
def g (c d x : ℝ) : ℝ := |x - c| + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) :
  (f a b 2 = 5) ∧ (f a b 8 = 3) ∧ (g c d 2 = 5) ∧ (g c d 8 = 3) →
  a + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l306_30606


namespace NUMINAMATH_CALUDE_smallest_N_proof_l306_30663

def f (n : ℕ+) : ℕ := sorry

def g (n : ℕ+) : ℕ := sorry

def N : ℕ+ := sorry

theorem smallest_N_proof : N = 44 ∧ (∀ m : ℕ+, m < N → g m < 11) ∧ g N ≥ 11 := by sorry

end NUMINAMATH_CALUDE_smallest_N_proof_l306_30663


namespace NUMINAMATH_CALUDE_sams_phone_bill_l306_30656

-- Define the constants from the problem
def base_cost : ℚ := 25
def text_cost : ℚ := 8 / 100  -- 8 cents in dollars
def extra_minute_cost : ℚ := 15 / 100  -- 15 cents in dollars
def included_hours : ℕ := 25
def texts_sent : ℕ := 150
def hours_talked : ℕ := 26

-- Define the function to calculate the total cost
def calculate_total_cost : ℚ :=
  let text_total := text_cost * texts_sent
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_minutes_total := extra_minute_cost * extra_minutes
  base_cost + text_total + extra_minutes_total

-- State the theorem
theorem sams_phone_bill : calculate_total_cost = 46 := by sorry

end NUMINAMATH_CALUDE_sams_phone_bill_l306_30656


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l306_30648

def f (x : ℝ) := x^3 - x^2 - x - 1

theorem root_exists_in_interval :
  (f 1 < 0) → (f 2 > 0) → ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l306_30648


namespace NUMINAMATH_CALUDE_expression_evaluation_l306_30638

theorem expression_evaluation (b : ℝ) (h : b = 3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / (2 * b) = 5 / 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l306_30638


namespace NUMINAMATH_CALUDE_pencil_pen_difference_l306_30680

/-- Given a ratio of pens to pencils and the total number of pencils,
    calculate the difference between pencils and pens. -/
theorem pencil_pen_difference
  (ratio_pens : ℕ)
  (ratio_pencils : ℕ)
  (total_pencils : ℕ)
  (h_ratio : ratio_pens < ratio_pencils)
  (h_total : total_pencils = 36)
  (h_ratio_pencils : total_pencils % ratio_pencils = 0) :
  total_pencils - (total_pencils / ratio_pencils * ratio_pens) = 6 :=
by sorry

end NUMINAMATH_CALUDE_pencil_pen_difference_l306_30680


namespace NUMINAMATH_CALUDE_power_fraction_equality_l306_30650

theorem power_fraction_equality : (1 : ℚ) / ((-5^4)^2) * (-5)^9 = -5 := by sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l306_30650


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_existence_l306_30644

theorem arithmetic_geometric_mean_ratio_existence :
  ∃ (a b : ℝ), 
    (a + b) / 2 = 3 * Real.sqrt (a * b) ∧
    a > b ∧ b > 0 ∧
    round (a / b) = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_existence_l306_30644


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l306_30605

/-- Given a circle with equation x^2 + y^2 + 2x - 4y - 4 = 0, 
    its center is at (-1, 2) and its radius is 3. -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧ 
    radius = 3 ∧
    ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 4 = 0 ↔ 
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l306_30605


namespace NUMINAMATH_CALUDE_cube_preserves_order_l306_30678

theorem cube_preserves_order (a b c : ℝ) (h : b > a) : b^3 > a^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l306_30678


namespace NUMINAMATH_CALUDE_ninth_term_is_negative_256_l306_30613

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℤ
  is_geometric : ∀ n : ℕ, ∃ q : ℤ, a (n + 1) = a n * q
  a2a5 : a 2 * a 5 = -32
  a3a4_sum : a 3 + a 4 = 4

/-- The 9th term of the geometric sequence is -256 -/
theorem ninth_term_is_negative_256 (seq : GeometricSequence) : seq.a 9 = -256 := by
  sorry

#check ninth_term_is_negative_256

end NUMINAMATH_CALUDE_ninth_term_is_negative_256_l306_30613


namespace NUMINAMATH_CALUDE_half_plus_six_equals_eleven_l306_30624

theorem half_plus_six_equals_eleven (n : ℝ) : (1/2 : ℝ) * n + 6 = 11 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_six_equals_eleven_l306_30624


namespace NUMINAMATH_CALUDE_gcd_8008_12012_l306_30609

theorem gcd_8008_12012 : Nat.gcd 8008 12012 = 4004 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8008_12012_l306_30609


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l306_30671

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 24) →
  p + q = 99 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l306_30671


namespace NUMINAMATH_CALUDE_regression_coefficient_correlation_same_sign_l306_30608

/-- Linear regression model -/
structure LinearRegression where
  a : ℝ
  b : ℝ
  x : ℝ → ℝ
  y : ℝ → ℝ
  equation : ∀ t, y t = a + b * x t

/-- Correlation coefficient -/
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

theorem regression_coefficient_correlation_same_sign 
  (model : LinearRegression) 
  (r : ℝ) 
  (h_r : r = correlation_coefficient model.x model.y) :
  (r > 0 ∧ model.b > 0) ∨ (r < 0 ∧ model.b < 0) ∨ (r = 0 ∧ model.b = 0) :=
sorry

end NUMINAMATH_CALUDE_regression_coefficient_correlation_same_sign_l306_30608


namespace NUMINAMATH_CALUDE_quadratic_solution_l306_30674

theorem quadratic_solution (x : ℚ) : 
  (63 * x^2 - 100 * x + 45 = 0) → 
  (63 * (5/7)^2 - 100 * (5/7) + 45 = 0) → 
  (63 * 1^2 - 100 * 1 + 45 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l306_30674


namespace NUMINAMATH_CALUDE_six_digit_difference_not_divisible_l306_30676

theorem six_digit_difference_not_divisible (A B : ℕ) : 
  100 ≤ A ∧ A < 1000 → 100 ≤ B ∧ B < 1000 → A ≠ B → 
  ¬(∃ k : ℤ, 999 * (A - B) = 1976 * k) := by
  sorry

end NUMINAMATH_CALUDE_six_digit_difference_not_divisible_l306_30676


namespace NUMINAMATH_CALUDE_hannah_stocking_stuffers_l306_30633

/-- The number of candy canes per stocking -/
def candy_canes_per_stocking : ℕ := 4

/-- The number of beanie babies per stocking -/
def beanie_babies_per_stocking : ℕ := 2

/-- The number of books per stocking -/
def books_per_stocking : ℕ := 1

/-- The number of kids Hannah has -/
def number_of_kids : ℕ := 3

/-- The total number of stocking stuffers Hannah buys -/
def total_stocking_stuffers : ℕ := 
  (candy_canes_per_stocking + beanie_babies_per_stocking + books_per_stocking) * number_of_kids

theorem hannah_stocking_stuffers : total_stocking_stuffers = 21 := by
  sorry

end NUMINAMATH_CALUDE_hannah_stocking_stuffers_l306_30633


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l306_30696

/-- A structure composed of unit cubes -/
structure CubeStructure where
  /-- The number of unit cubes in the structure -/
  num_cubes : ℕ
  /-- The number of cubes surrounding the center cube -/
  surrounding_cubes : ℕ
  /-- Assertion that there is one center cube -/
  has_center_cube : num_cubes = surrounding_cubes + 1

/-- Calculate the volume of the structure -/
def volume (s : CubeStructure) : ℕ := s.num_cubes

/-- Calculate the surface area of the structure -/
def surface_area (s : CubeStructure) : ℕ :=
  1 + (s.surrounding_cubes - 1) * 5 + 4

/-- The theorem to be proved -/
theorem volume_to_surface_area_ratio (s : CubeStructure) 
  (h1 : s.num_cubes = 10) 
  (h2 : s.surrounding_cubes = 9) : 
  (volume s : ℚ) / (surface_area s : ℚ) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l306_30696


namespace NUMINAMATH_CALUDE_equation_solution_l306_30637

theorem equation_solution : ∃ x : ℝ, (3 * x - 5 = -2 * x + 10) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l306_30637
