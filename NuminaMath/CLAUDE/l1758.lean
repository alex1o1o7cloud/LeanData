import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1758_175880

/-- Given two quadratic equations, if the roots of the first are each three less than
    the roots of the second, then the constant term of the first equation is 24/5. -/
theorem quadratic_root_relation (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 ↔ ∃ y, 5*y^2 - 4*y - 9 = 0 ∧ x = y - 3) →
  c = 24/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1758_175880


namespace NUMINAMATH_CALUDE_base_seven_divisibility_l1758_175899

theorem base_seven_divisibility (d : Nat) : 
  d ≤ 6 → (2 * 7^3 + d * 7^2 + d * 7 + 7) % 13 = 0 ↔ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_divisibility_l1758_175899


namespace NUMINAMATH_CALUDE_sector_area_l1758_175823

theorem sector_area (r : ℝ) (θ : ℝ) (h : θ = 2 * Real.pi / 3) :
  let area := (1 / 2) * r^2 * θ
  area = 3 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1758_175823


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1758_175854

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, a * x^2 + 4 * x + 1 > 0) ↔ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1758_175854


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1758_175842

theorem smallest_number_of_eggs : ∃ (n : ℕ), 
  (n > 150) ∧ 
  (∃ (c : ℕ), n = 18 * c - 7) ∧ 
  (∀ m : ℕ, (m > 150) ∧ (∃ (d : ℕ), m = 18 * d - 7) → m ≥ n) ∧ 
  n = 155 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1758_175842


namespace NUMINAMATH_CALUDE_complex_powers_sum_l1758_175817

theorem complex_powers_sum (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (π / 4)) : 
  z^12 + z⁻¹^12 = -2 ∧ z^6 + z⁻¹^6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_powers_sum_l1758_175817


namespace NUMINAMATH_CALUDE_dans_initial_money_l1758_175851

/-- Dan's initial amount of money -/
def initial_money : ℕ := sorry

/-- Cost of the candy bar -/
def candy_cost : ℕ := 6

/-- Cost of the chocolate -/
def chocolate_cost : ℕ := 3

/-- Theorem stating that Dan's initial money is equal to the total spent -/
theorem dans_initial_money :
  initial_money = candy_cost + chocolate_cost ∧ candy_cost = chocolate_cost + 3 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_money_l1758_175851


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_remainder_equals_897_l1758_175835

def f (x : ℝ) : ℝ := 5*x^8 - 3*x^7 + 2*x^6 - 9*x^4 + 3*x^3 - 7

theorem polynomial_remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ (q : ℝ → ℝ), f = fun x ↦ (x - a) * q x + f a := by sorry

theorem remainder_equals_897 :
  ∃ (q : ℝ → ℝ), f = fun x ↦ (3*x - 6) * q x + 897 := by
  have h : ∃ (q : ℝ → ℝ), f = fun x ↦ (x - 2) * q x + f 2 := polynomial_remainder_theorem f 2
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_remainder_equals_897_l1758_175835


namespace NUMINAMATH_CALUDE_exists_one_one_appended_one_l1758_175832

def is_valid_number (n : ℕ) (num : List ℕ) : Prop :=
  num.length = n ∧ ∀ d ∈ num, d = 1 ∨ d = 2 ∨ d = 3

def differs_in_all_positions (n : ℕ) (num1 num2 : List ℕ) : Prop :=
  is_valid_number n num1 ∧ is_valid_number n num2 ∧
  ∀ i, i < n → num1.get ⟨i, by sorry⟩ ≠ num2.get ⟨i, by sorry⟩

def appended_digit (n : ℕ) (num : List ℕ) (d : ℕ) : Prop :=
  is_valid_number n num ∧ (d = 1 ∨ d = 2 ∨ d = 3)

def valid_appending (n : ℕ) (append : List ℕ → ℕ) : Prop :=
  ∀ num1 num2 : List ℕ, differs_in_all_positions n num1 num2 →
    append num1 ≠ append num2

theorem exists_one_one_appended_one (n : ℕ) :
  ∃ (append : List ℕ → ℕ),
    valid_appending n append →
    ∃ (num : List ℕ),
      is_valid_number n num ∧
      (num.count 1 = 1) ∧
      (append num = 1) := by sorry

end NUMINAMATH_CALUDE_exists_one_one_appended_one_l1758_175832


namespace NUMINAMATH_CALUDE_no_real_roots_l1758_175883

theorem no_real_roots : ∀ x : ℝ, x^2 + 2*x + 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l1758_175883


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1758_175805

theorem sum_of_two_numbers (x y : ℤ) : 
  y = 2 * x - 43 →  -- First number is 43 less than twice the second
  max x y = 31 →    -- Larger number is 31
  x + y = 68 :=     -- Sum of the two numbers is 68
by sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1758_175805


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_for_specific_quadrilateral_l1758_175801

/-- Represents a quadrilateral with side lengths a, b, c, and d -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The radius of the largest inscribed circle in a quadrilateral -/
def largest_inscribed_circle_radius (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that for a quadrilateral with side lengths 10, 12, 8, and 14,
    the radius of the largest inscribed circle is √24.75 -/
theorem largest_inscribed_circle_radius_for_specific_quadrilateral :
  let q : Quadrilateral := ⟨10, 12, 8, 14⟩
  largest_inscribed_circle_radius q = Real.sqrt 24.75 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_for_specific_quadrilateral_l1758_175801


namespace NUMINAMATH_CALUDE_geometric_sum_five_quarters_l1758_175827

def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_five_quarters :
  geometric_sum (1/4) (1/4) 5 = 341/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_five_quarters_l1758_175827


namespace NUMINAMATH_CALUDE_money_ratio_to_jenna_l1758_175892

/-- Represents the financial transaction scenario with John, his uncle, and Jenna --/
def john_transaction (money_from_uncle money_to_jenna groceries_cost money_remaining : ℚ) : Prop :=
  money_from_uncle - money_to_jenna - groceries_cost = money_remaining

/-- Theorem stating the ratio of money given to Jenna to money received from uncle --/
theorem money_ratio_to_jenna :
  ∃ (money_to_jenna : ℚ),
    john_transaction 100 money_to_jenna 40 35 ∧
    money_to_jenna / 100 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_money_ratio_to_jenna_l1758_175892


namespace NUMINAMATH_CALUDE_largest_sum_of_3digit_numbers_l1758_175861

def digits : Finset Nat := {1, 2, 3, 7, 8, 9}

def is_valid_pair (a b : Nat) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧ b ≥ 100 ∧ b < 1000 ∧
  (∃ (d1 d2 d3 d4 d5 d6 : Nat),
    {d1, d2, d3, d4, d5, d6} = digits ∧
    a = 100 * d1 + 10 * d2 + d3 ∧
    b = 100 * d4 + 10 * d5 + d6)

def sum_of_pair (a b : Nat) : Nat := a + b

theorem largest_sum_of_3digit_numbers :
  (∃ (a b : Nat), is_valid_pair a b ∧
    ∀ (x y : Nat), is_valid_pair x y → sum_of_pair x y ≤ sum_of_pair a b) ∧
  (∀ (a b : Nat), is_valid_pair a b → sum_of_pair a b ≤ 1803) ∧
  (∃ (a b : Nat), is_valid_pair a b ∧ sum_of_pair a b = 1803) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_of_3digit_numbers_l1758_175861


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1758_175882

theorem cos_alpha_value (α : Real) 
  (h1 : π/4 < α) 
  (h2 : α < 3*π/4) 
  (h3 : Real.sin (α - π/4) = 4/5) : 
  Real.cos α = -Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1758_175882


namespace NUMINAMATH_CALUDE_circle_radius_decrease_l1758_175895

theorem circle_radius_decrease (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let r' := r * (1 - x)
  let A' := π * r'^2
  A' = 0.25 * A →
  x = 0.5
  := by sorry

end NUMINAMATH_CALUDE_circle_radius_decrease_l1758_175895


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1758_175803

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- Another function defined in terms of f -/
def g (k : ℝ) (x : ℝ) : ℝ := f x - k*x

/-- Theorem stating the properties of f and g -/
theorem quadratic_function_properties :
  (∀ x, f x ≥ -1) ∧ 
  (f 2 = -1) ∧ 
  (f 1 + f 4 = 3) ∧
  (∀ k, (∀ x ∈ Set.Ioo 1 4, ∃ y ∈ Set.Ioo 1 4, g k y < g k x) ↔ 
    k ∈ Set.Iic (-2) ∪ Set.Ici 4) := by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l1758_175803


namespace NUMINAMATH_CALUDE_sum_remainder_mod_11_l1758_175828

theorem sum_remainder_mod_11 : (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_11_l1758_175828


namespace NUMINAMATH_CALUDE_jane_max_tickets_l1758_175812

/-- Calculates the maximum number of concert tickets that can be purchased given a budget and pricing structure. -/
def max_tickets (budget : ℕ) (regular_price : ℕ) (discounted_price : ℕ) (discount_threshold : ℕ) : ℕ :=
  let regular_tickets := min discount_threshold (budget / regular_price)
  let remaining_budget := budget - regular_tickets * regular_price
  let discounted_tickets := remaining_budget / discounted_price
  regular_tickets + discounted_tickets

/-- Theorem stating that given the specific conditions, the maximum number of tickets Jane can buy is 8. -/
theorem jane_max_tickets :
  max_tickets 120 15 12 5 = 8 := by
  sorry

#eval max_tickets 120 15 12 5

end NUMINAMATH_CALUDE_jane_max_tickets_l1758_175812


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l1758_175836

theorem solution_of_linear_equation (a : ℝ) : 
  (∃ x y : ℝ, x = 3 ∧ y = 2 ∧ a * x + 2 * y = 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l1758_175836


namespace NUMINAMATH_CALUDE_cuboid_volume_example_l1758_175881

/-- The volume of a cuboid with given base area and height -/
def cuboid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  base_area * height

/-- Theorem: The volume of a cuboid with base area 14 cm² and height 13 cm is 182 cm³ -/
theorem cuboid_volume_example : cuboid_volume 14 13 = 182 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_example_l1758_175881


namespace NUMINAMATH_CALUDE_pens_to_classmates_l1758_175813

/-- Represents the problem of calculating the fraction of remaining pens given to classmates. -/
theorem pens_to_classmates 
  (boxes : ℕ) 
  (pens_per_box : ℕ) 
  (friend_percentage : ℚ) 
  (pens_left : ℕ) 
  (h1 : boxes = 20) 
  (h2 : pens_per_box = 5) 
  (h3 : friend_percentage = 2/5) 
  (h4 : pens_left = 45) : 
  (boxes * pens_per_box - pens_left - (friend_percentage * (boxes * pens_per_box))) / 
  ((1 - friend_percentage) * (boxes * pens_per_box)) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_pens_to_classmates_l1758_175813


namespace NUMINAMATH_CALUDE_milk_cost_l1758_175820

theorem milk_cost (banana_cost : ℝ) (tax_rate : ℝ) (total_spent : ℝ) 
  (h1 : banana_cost = 2)
  (h2 : tax_rate = 0.2)
  (h3 : total_spent = 6) :
  ∃ milk_cost : ℝ, milk_cost = 3 ∧ 
    total_spent = (milk_cost + banana_cost) * (1 + tax_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_milk_cost_l1758_175820


namespace NUMINAMATH_CALUDE_expression_value_l1758_175855

theorem expression_value (p q r : ℝ) (hp : p ≠ 2) (hq : q ≠ 3) (hr : r ≠ 4) :
  (p^2 - 4) / (4 - r^2) * (q^2 - 9) / (2 - p^2) * (r^2 - 16) / (3 - q^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1758_175855


namespace NUMINAMATH_CALUDE_cafeteria_pies_l1758_175819

/-- Given a cafeteria scenario with initial apples, apples handed out, and apples per pie,
    calculate the number of pies that can be made. -/
theorem cafeteria_pies (initial_apples handed_out apples_per_pie : ℕ) 
    (h1 : initial_apples = 62)
    (h2 : handed_out = 8)
    (h3 : apples_per_pie = 9) :
    (initial_apples - handed_out) / apples_per_pie = 6 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l1758_175819


namespace NUMINAMATH_CALUDE_function_positivity_implies_ab_bound_l1758_175821

/-- Given a function f(x) = (x - 1/x - a)(x - b), if f(x) > 0 for all x > 0, then ab > -1 -/
theorem function_positivity_implies_ab_bound (a b : ℝ) : 
  (∀ x > 0, (x - 1/x - a) * (x - b) > 0) → a * b > -1 := by
  sorry

end NUMINAMATH_CALUDE_function_positivity_implies_ab_bound_l1758_175821


namespace NUMINAMATH_CALUDE_fraction_problem_l1758_175869

theorem fraction_problem (f : ℚ) : 
  0.60 * 412.5 = f * 412.5 + 110 → f = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1758_175869


namespace NUMINAMATH_CALUDE_A_empty_A_singleton_A_at_most_one_A_element_when_zero_A_element_when_nine_eighths_l1758_175866

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

-- Theorem 1: A is empty iff a > 9/8
theorem A_empty (a : ℝ) : A a = ∅ ↔ a > 9/8 := by sorry

-- Theorem 2: A contains exactly one element iff a = 0 or a = 9/8
theorem A_singleton (a : ℝ) : (∃! x, x ∈ A a) ↔ a = 0 ∨ a = 9/8 := by sorry

-- Theorem 3: A contains at most one element iff a = 0 or a ≥ 9/8
theorem A_at_most_one (a : ℝ) : (∀ x y, x ∈ A a → y ∈ A a → x = y) ↔ a = 0 ∨ a ≥ 9/8 := by sorry

-- Additional theorems for specific elements when A is a singleton
theorem A_element_when_zero : (∀ x, x ∈ A 0 ↔ x = 2/3) := by sorry

theorem A_element_when_nine_eighths : (∀ x, x ∈ A (9/8) ↔ x = 4/3) := by sorry

end NUMINAMATH_CALUDE_A_empty_A_singleton_A_at_most_one_A_element_when_zero_A_element_when_nine_eighths_l1758_175866


namespace NUMINAMATH_CALUDE_ellipse_equation_l1758_175871

theorem ellipse_equation (e : ℝ) (h_e : e = (2/5) * Real.sqrt 5) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    e = Real.sqrt (a^2 - b^2) / a ∧
    1^2 / b^2 + 0^2 / a^2 = 1 ∧
    (∀ x y : ℝ, x^2 / b^2 + y^2 / a^2 = 1 ↔ x^2 + (1/5) * y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1758_175871


namespace NUMINAMATH_CALUDE_difference_of_squares_l1758_175837

theorem difference_of_squares (x y : ℝ) : (x + y) * (x - y) = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1758_175837


namespace NUMINAMATH_CALUDE_intersection_equality_l1758_175884

def M : Set ℤ := {-1, 0, 1}
def N (a : ℤ) : Set ℤ := {a, a^2}

theorem intersection_equality (a : ℤ) : M ∩ N a = N a ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l1758_175884


namespace NUMINAMATH_CALUDE_condition_relationship_l1758_175870

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a > 2 ∧ b > 2 → a + b > 4) ∧
  (∃ a b, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l1758_175870


namespace NUMINAMATH_CALUDE_simplify_fraction_l1758_175845

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1758_175845


namespace NUMINAMATH_CALUDE_product_of_cubic_fractions_l1758_175877

theorem product_of_cubic_fractions :
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 2) * (f 3) * (f 4) * (f 5) * (f 6) = 43 / 63 := by
sorry

end NUMINAMATH_CALUDE_product_of_cubic_fractions_l1758_175877


namespace NUMINAMATH_CALUDE_distance_between_cities_l1758_175807

def train_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem distance_between_cities (D : ℝ) : D = 330 :=
  let train1_speed : ℝ := 60
  let train1_time : ℝ := 3
  let train2_speed : ℝ := 75
  let train2_time : ℝ := 2
  let train1_distance := train_distance train1_speed train1_time
  let train2_distance := train_distance train2_speed train2_time
  have h1 : D = train1_distance + train2_distance := by sorry
  have h2 : train1_distance = 180 := by sorry
  have h3 : train2_distance = 150 := by sorry
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l1758_175807


namespace NUMINAMATH_CALUDE_square_cloth_trimming_l1758_175872

theorem square_cloth_trimming (x : ℝ) : 
  x > 0 →  -- Ensure positive length
  (x - 6) * (x - 5) = 120 → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_square_cloth_trimming_l1758_175872


namespace NUMINAMATH_CALUDE_archie_antibiotic_cost_l1758_175858

/-- The total cost of antibiotics for Archie -/
def total_cost (doses_per_day : ℕ) (days : ℕ) (cost_per_dose : ℕ) : ℕ :=
  doses_per_day * days * cost_per_dose

/-- Proof that the total cost of antibiotics for Archie is $63 -/
theorem archie_antibiotic_cost :
  total_cost 3 7 3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_archie_antibiotic_cost_l1758_175858


namespace NUMINAMATH_CALUDE_solve_system_l1758_175873

theorem solve_system (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 5*t + 6) 
  (h3 : x = 1) : 
  y = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1758_175873


namespace NUMINAMATH_CALUDE_photo_album_completion_l1758_175898

theorem photo_album_completion 
  (total_slots : ℕ) 
  (cristina_photos : ℕ) 
  (john_photos : ℕ) 
  (sarah_photos : ℕ) 
  (h1 : total_slots = 40) 
  (h2 : cristina_photos = 7) 
  (h3 : john_photos = 10) 
  (h4 : sarah_photos = 9) : 
  total_slots - (cristina_photos + john_photos + sarah_photos) = 14 := by
  sorry

end NUMINAMATH_CALUDE_photo_album_completion_l1758_175898


namespace NUMINAMATH_CALUDE_clock_angle_l1758_175878

theorem clock_angle (hour_hand_angle hour_hand_movement minute_hand_movement : ℝ) :
  hour_hand_angle = 90 →
  hour_hand_movement = 15 →
  minute_hand_movement = 180 →
  180 - hour_hand_angle - hour_hand_movement = 75 :=
by sorry

end NUMINAMATH_CALUDE_clock_angle_l1758_175878


namespace NUMINAMATH_CALUDE_largest_prime_to_test_primality_l1758_175831

theorem largest_prime_to_test_primality (n : ℕ) (h : 1100 ≤ n ∧ n ≤ 1150) :
  (∃ p : ℕ, Nat.Prime p ∧ p^2 ≤ n ∧ ∀ q, Nat.Prime q → q^2 ≤ n → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p^2 ≤ n ∧ ∀ q, Nat.Prime q → q^2 ≤ n → q ≤ p ∧ p = 31) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_to_test_primality_l1758_175831


namespace NUMINAMATH_CALUDE_driving_time_calculation_l1758_175808

/-- 
Given a trip with the following conditions:
1. The total trip duration is 15 hours
2. The time stuck in traffic is twice the driving time
This theorem proves that the driving time is 5 hours
-/
theorem driving_time_calculation (total_time : ℝ) (driving_time : ℝ) (traffic_time : ℝ) 
  (h1 : total_time = 15)
  (h2 : traffic_time = 2 * driving_time)
  (h3 : total_time = driving_time + traffic_time) :
  driving_time = 5 := by
sorry

end NUMINAMATH_CALUDE_driving_time_calculation_l1758_175808


namespace NUMINAMATH_CALUDE_total_goals_is_fifteen_l1758_175840

/-- The total number of goals scored in a soccer match -/
def total_goals (kickers_first : ℕ) : ℕ :=
  let kickers_second := 2 * kickers_first
  let spiders_first := kickers_first / 2
  let spiders_second := 2 * kickers_second
  kickers_first + kickers_second + spiders_first + spiders_second

/-- Theorem stating that the total goals scored is 15 when The Kickers score 2 goals in the first period -/
theorem total_goals_is_fifteen : total_goals 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_goals_is_fifteen_l1758_175840


namespace NUMINAMATH_CALUDE_symmetric_points_line_equation_l1758_175822

/-- Given two points are symmetric about a line, prove the equation of the line -/
theorem symmetric_points_line_equation (O A : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  O = (0, 0) → 
  A = (-4, 2) → 
  (∀ p : ℝ × ℝ, p ∈ l ↔ (2 : ℝ) * p.1 - p.2 + 5 = 0) →
  (∀ p : ℝ × ℝ, p ∈ l → dist O p = dist A p) →
  True :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_line_equation_l1758_175822


namespace NUMINAMATH_CALUDE_max_xy_given_constraint_l1758_175841

theorem max_xy_given_constraint (x y : ℝ) : 
  x > 0 → y > 0 → x + 4 * y = 1 → x * y ≤ 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_given_constraint_l1758_175841


namespace NUMINAMATH_CALUDE_shooting_events_contradictory_l1758_175857

-- Define the sample space
def Ω : Type := List Bool

-- Define the events
def at_least_one_hit (ω : Ω) : Prop := ω.any id
def three_consecutive_misses (ω : Ω) : Prop := ω = [false, false, false]

-- Define the property of being contradictory events
def contradictory (A B : Ω → Prop) : Prop :=
  (∀ ω : Ω, A ω → ¬B ω) ∧ (∀ ω : Ω, B ω → ¬A ω)

-- Theorem statement
theorem shooting_events_contradictory :
  contradictory at_least_one_hit three_consecutive_misses :=
by sorry

end NUMINAMATH_CALUDE_shooting_events_contradictory_l1758_175857


namespace NUMINAMATH_CALUDE_min_bullseyes_for_victory_l1758_175888

/-- Represents the possible scores in the archery tournament -/
inductive Score
  | bullseye : Score
  | ten : Score
  | five : Score
  | three : Score
  | zero : Score

/-- Convert a Score to its numerical value -/
def score_value : Score → Nat
  | Score.bullseye => 12
  | Score.ten => 10
  | Score.five => 5
  | Score.three => 3
  | Score.zero => 0

/-- The total number of shots in the tournament -/
def total_shots : Nat := 120

/-- The number of shots already taken -/
def shots_taken : Nat := 60

/-- Alex's lead after half the tournament -/
def alex_lead : Nat := 70

/-- Alex's minimum score per shot -/
def alex_min_score : Nat := 5

/-- The maximum possible score per shot -/
def max_score_per_shot : Nat := 12

/-- Theorem: The minimum number of consecutive bullseyes Alex needs to guarantee victory is 51 -/
theorem min_bullseyes_for_victory :
  ∀ n : Nat,
  (∀ m : Nat, m < n → 
    ∃ opponent_score : Nat,
    opponent_score ≤ (total_shots - shots_taken) * max_score_per_shot ∧
    alex_lead + n * score_value Score.bullseye + (total_shots - shots_taken - n) * alex_min_score ≤ opponent_score) ∧
  (∀ opponent_score : Nat,
   opponent_score ≤ (total_shots - shots_taken) * max_score_per_shot →
   alex_lead + n * score_value Score.bullseye + (total_shots - shots_taken - n) * alex_min_score > opponent_score) →
  n = 51 := by
  sorry

end NUMINAMATH_CALUDE_min_bullseyes_for_victory_l1758_175888


namespace NUMINAMATH_CALUDE_score_distribution_theorem_l1758_175811

/-- Represents the frequency distribution of student scores -/
structure FrequencyDistribution :=
  (f65_70 f70_75 f75_80 f80_85 f85_90 f90_95 f95_100 : ℚ)

/-- Represents the problem setup -/
structure ProblemSetup :=
  (fd : FrequencyDistribution)
  (total_students : ℕ)
  (students_80_90 : ℕ)
  (prob_male_95_100 : ℚ)
  (female_65_70 : ℕ)

/-- The main theorem to prove -/
theorem score_distribution_theorem (setup : ProblemSetup) :
  (setup.total_students * setup.fd.f95_100 = 6) ∧
  (∃ (m : ℕ), m = 2 ∧ m ≤ 6 ∧ 
    (m * (m - 1) / 30 + m * (6 - m) / 15 : ℚ) = 3/5) ∧
  (∃ (p0 p1 p2 : ℚ), p0 + p1 + p2 = 1 ∧
    p0 * 0 + p1 * 1 + p2 * 2 = 1) :=
by sorry

/-- Assumptions about the problem setup -/
axiom setup_valid (setup : ProblemSetup) :
  setup.fd.f65_70 = 1/10 ∧
  setup.fd.f70_75 = 3/20 ∧
  setup.fd.f75_80 = 1/5 ∧
  setup.fd.f80_85 = 1/5 ∧
  setup.fd.f85_90 = 3/20 ∧
  setup.fd.f90_95 = 1/10 ∧
  setup.fd.f95_100 + setup.fd.f65_70 + setup.fd.f70_75 + setup.fd.f75_80 +
    setup.fd.f80_85 + setup.fd.f85_90 + setup.fd.f90_95 = 1 ∧
  setup.students_80_90 = 21 ∧
  setup.prob_male_95_100 = 3/5 ∧
  setup.female_65_70 = 4 ∧
  setup.total_students * (setup.fd.f80_85 + setup.fd.f85_90) = setup.students_80_90

end NUMINAMATH_CALUDE_score_distribution_theorem_l1758_175811


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1758_175889

theorem more_girls_than_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) : 
  total_students = 42 →
  boy_ratio = 3 →
  girl_ratio = 4 →
  ∃ (k : ℕ), 
    (boy_ratio * k + girl_ratio * k = total_students) ∧
    (girl_ratio * k - boy_ratio * k = 6) :=
by sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1758_175889


namespace NUMINAMATH_CALUDE_movie_theatre_revenue_l1758_175806

/-- Calculates the total ticket revenue for a movie theatre session -/
theorem movie_theatre_revenue 
  (total_seats : ℕ) 
  (adult_price child_price : ℕ) 
  (num_children : ℕ) 
  (h_full : num_children ≤ total_seats) : 
  let num_adults := total_seats - num_children
  (num_adults * adult_price + num_children * child_price : ℕ) = 1124 :=
by
  sorry

#check movie_theatre_revenue 250 6 4 188

end NUMINAMATH_CALUDE_movie_theatre_revenue_l1758_175806


namespace NUMINAMATH_CALUDE_cafeteria_apples_l1758_175864

/-- The cafeteria problem -/
theorem cafeteria_apples (initial : ℕ) (used : ℕ) (bought : ℕ) :
  initial = 38 → used = 20 → bought = 28 → initial - used + bought = 46 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l1758_175864


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1758_175860

theorem algebraic_expression_equality (x y : ℝ) (h : x + 2*y = 2) : 2*x + 4*y - 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1758_175860


namespace NUMINAMATH_CALUDE_smallest_multiple_of_112_l1758_175809

theorem smallest_multiple_of_112 (n : ℕ) : (n * 14 % 112 = 0 ∧ n > 0) → n ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_112_l1758_175809


namespace NUMINAMATH_CALUDE_tangent_slopes_sum_l1758_175814

/-- Parabola P with equation y = (x-3)^2 + 2 -/
def P : ℝ → ℝ := λ x ↦ (x - 3)^2 + 2

/-- Point Q -/
def Q : ℝ × ℝ := (15, 7)

/-- The sum of the slopes of the two tangent lines from Q to P is 48 -/
theorem tangent_slopes_sum : 
  ∃ (r s : ℝ), (∀ m : ℝ, (r < m ∧ m < s) ↔ 
    ∀ x : ℝ, P x ≠ (m * (x - Q.1) + Q.2)) ∧ r + s = 48 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slopes_sum_l1758_175814


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l1758_175853

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 5

theorem tangent_slope_angle :
  let slope := (deriv f) 1
  Real.arctan slope = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l1758_175853


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l1758_175834

/-- The difference in area between two rectangles, where one rectangle's dimensions are 1 cm less
    than the other's in both length and width, is equal to the sum of the larger rectangle's
    length and width, minus 1. -/
theorem rectangle_area_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x * y - (x - 1) * (y - 1) = x + y - 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l1758_175834


namespace NUMINAMATH_CALUDE_y_value_l1758_175802

theorem y_value (y : ℕ) (h1 : ∃ k : ℕ, y = 9 * k) (h2 : y^2 > 200) (h3 : y < 30) : y = 18 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1758_175802


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1758_175839

theorem quadratic_roots_range (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + (k-2)*x + 2*k-1 = 0 ↔ x = x₁ ∨ x = x₂) →
  0 < x₁ → x₁ < 1 → 1 < x₂ → x₂ < 2 →
  1/2 < k ∧ k < 2/3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1758_175839


namespace NUMINAMATH_CALUDE_winnie_lollipop_distribution_l1758_175896

/-- The number of lollipops left after equal distribution --/
def lollipops_left (cherry wintergreen grape shrimp_cocktail friends : ℕ) : ℕ :=
  (cherry + wintergreen + grape + shrimp_cocktail) % friends

theorem winnie_lollipop_distribution :
  lollipops_left 55 134 12 265 15 = 1 := by sorry

end NUMINAMATH_CALUDE_winnie_lollipop_distribution_l1758_175896


namespace NUMINAMATH_CALUDE_rectangle_area_is_six_l1758_175804

/-- The quadratic equation representing the sides of the rectangle -/
def quadratic_equation (x : ℝ) : Prop := x^2 - 5*x + 6 = 0

/-- The roots of the quadratic equation -/
def roots : Set ℝ := {x : ℝ | quadratic_equation x}

/-- The rectangle with sides equal to the roots of the quadratic equation -/
structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  side1_root : quadratic_equation side1
  side2_root : quadratic_equation side2
  different_sides : side1 ≠ side2

/-- The area of the rectangle -/
def area (rect : Rectangle) : ℝ := rect.side1 * rect.side2

/-- Theorem: The area of the rectangle is 6 -/
theorem rectangle_area_is_six (rect : Rectangle) : area rect = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_six_l1758_175804


namespace NUMINAMATH_CALUDE_equation_solutions_l1758_175824

theorem equation_solutions :
  (∃ x : ℚ, (3 : ℚ) / 5 - (5 : ℚ) / 8 * x = (2 : ℚ) / 5 ∧ x = (8 : ℚ) / 25) ∧
  (∃ x : ℚ, 7 * (x - 2) = 8 * (x - 4) ∧ x = 18) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1758_175824


namespace NUMINAMATH_CALUDE_tom_sees_jerry_l1758_175886

/-- Represents the cat-and-mouse chase problem -/
structure ChaseSetup where
  wallSideLength : ℝ
  tomSpeed : ℝ
  jerrySpeed : ℝ
  restTime : ℝ

/-- Calculates the time when Tom first sees Jerry -/
noncomputable def timeToMeet (setup : ChaseSetup) : ℝ :=
  sorry

/-- The main theorem stating when Tom will first see Jerry -/
theorem tom_sees_jerry (setup : ChaseSetup) :
  setup.wallSideLength = 100 ∧
  setup.tomSpeed = 50 ∧
  setup.jerrySpeed = 30 ∧
  setup.restTime = 1 →
  timeToMeet setup = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_sees_jerry_l1758_175886


namespace NUMINAMATH_CALUDE_elements_beginning_with_3_l1758_175847

/-- The set of powers of 7 from 0 to 2011 -/
def T : Set ℕ := {n : ℕ | ∃ k : ℕ, 0 ≤ k ∧ k ≤ 2011 ∧ n = 7^k}

/-- The number of digits in 7^2011 -/
def digits_of_7_2011 : ℕ := 1602

/-- Function to check if a natural number begins with the digit 3 -/
def begins_with_3 (n : ℕ) : Prop := sorry

/-- The count of elements in T that begin with 3 -/
def count_begins_with_3 (S : Set ℕ) : ℕ := sorry

theorem elements_beginning_with_3 :
  count_begins_with_3 T = 45 :=
sorry

end NUMINAMATH_CALUDE_elements_beginning_with_3_l1758_175847


namespace NUMINAMATH_CALUDE_factorial_expression_equality_l1758_175818

theorem factorial_expression_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 6 - 6 * Nat.factorial 5 = 7920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_expression_equality_l1758_175818


namespace NUMINAMATH_CALUDE_tenth_prime_l1758_175894

def is_prime (n : ℕ) : Prop := sorry

def nth_prime (n : ℕ) : ℕ := sorry

theorem tenth_prime :
  (nth_prime 5 = 11) → (nth_prime 10 = 29) := by sorry

end NUMINAMATH_CALUDE_tenth_prime_l1758_175894


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1758_175893

theorem systematic_sampling_theorem (population : ℕ) (sample_size : ℕ) 
  (h1 : population = 1650) (h2 : sample_size = 35) :
  ∃ (exclude : ℕ) (segment_size : ℕ),
    exclude = population % sample_size ∧
    segment_size = (population - exclude) / sample_size ∧
    exclude = 5 ∧
    segment_size = 47 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1758_175893


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l1758_175815

/-- The number of y-intercepts of the parabola x = 3y^2 - 2y + 1 -/
theorem parabola_y_intercepts : 
  let f : ℝ → ℝ := fun y => 3 * y^2 - 2 * y + 1
  (∃ y, f y = 0) = False :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l1758_175815


namespace NUMINAMATH_CALUDE_bob_cereal_difference_l1758_175829

/-- Represents the number of sides on Bob's die -/
def dieSides : ℕ := 8

/-- Represents the threshold for eating organic cereal -/
def organicThreshold : ℕ := 5

/-- Represents the number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- Probability of eating organic cereal -/
def probOrganic : ℚ := 4 / 7

/-- Probability of eating gluten-free cereal -/
def probGlutenFree : ℚ := 3 / 7

/-- Expected difference in days between eating organic and gluten-free cereal -/
def expectedDifference : ℚ := daysInYear * (probOrganic - probGlutenFree)

theorem bob_cereal_difference :
  expectedDifference = 365 * (4/7 - 3/7) :=
sorry

end NUMINAMATH_CALUDE_bob_cereal_difference_l1758_175829


namespace NUMINAMATH_CALUDE_stratified_sampling_second_year_l1758_175885

/-- The number of students in the second year of high school -/
def second_year_students : ℕ := 750

/-- The probability of a student being selected in the stratified sampling -/
def selection_probability : ℚ := 2 / 100

/-- The number of students to be drawn from the second year -/
def students_drawn : ℕ := 15

/-- Theorem stating that the number of students drawn from the second year
    is equal to the product of the total number of second-year students
    and the selection probability -/
theorem stratified_sampling_second_year :
  (second_year_students : ℚ) * selection_probability = students_drawn := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_second_year_l1758_175885


namespace NUMINAMATH_CALUDE_game_result_l1758_175844

/-- A game between two players where the winner gains 2 points and the loser loses 1 point. -/
structure Game where
  total_games : ℕ
  games_won_by_player1 : ℕ
  final_score_player2 : ℤ

/-- Theorem stating that if player1 wins exactly 3 games and player2 has a final score of 5,
    then the total number of games played is 7. -/
theorem game_result (g : Game) 
  (h1 : g.games_won_by_player1 = 3)
  (h2 : g.final_score_player2 = 5) :
  g.total_games = 7 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1758_175844


namespace NUMINAMATH_CALUDE_ones_digit_of_triple_4567_l1758_175825

theorem ones_digit_of_triple_4567 : (3 * 4567) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_triple_4567_l1758_175825


namespace NUMINAMATH_CALUDE_fraction_expression_l1758_175833

theorem fraction_expression : 
  (3/7 + 5/8) / (5/12 + 2/9) = 531/322 := by
  sorry

end NUMINAMATH_CALUDE_fraction_expression_l1758_175833


namespace NUMINAMATH_CALUDE_factor_implies_h_value_l1758_175867

theorem factor_implies_h_value (m h : ℤ) : 
  (m - 8) ∣ (m^2 - h*m - 24) → h = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_h_value_l1758_175867


namespace NUMINAMATH_CALUDE_calculate_expression_l1758_175848

theorem calculate_expression : 
  20062006 * 2007 + 20072007 * 2008 - 2006 * 20072007 - 2007 * 20082008 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1758_175848


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1758_175849

/-- A quadratic function with vertex (3, 5) passing through (-2, -20) has a = -1 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Condition 1
  (3, 5) = (- b / (2 * a), a * (- b / (2 * a))^2 + b * (- b / (2 * a)) + c) →  -- Condition 2
  a * (-2)^2 + b * (-2) + c = -20 →  -- Condition 3
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1758_175849


namespace NUMINAMATH_CALUDE_product_increase_factor_l1758_175838

theorem product_increase_factor (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b : ℝ, (10 * a) * b = 10 * (a * b)) :=
sorry

end NUMINAMATH_CALUDE_product_increase_factor_l1758_175838


namespace NUMINAMATH_CALUDE_angle_between_polar_lines_theorem_l1758_175830

/-- The angle between two lines in polar coordinates -/
def angle_between_polar_lines (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- First line in polar coordinates: ρ(2cosθ + sinθ) = 2 -/
def line1 (ρ θ : ℝ) : Prop :=
  ρ * (2 * Real.cos θ + Real.sin θ) = 2

/-- Second line in polar coordinates: ρcosθ = 1 -/
def line2 (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ = 1

/-- Theorem stating the angle between the two lines -/
theorem angle_between_polar_lines_theorem :
  angle_between_polar_lines line1 line2 = Real.arctan (1 / 2) :=
sorry

end NUMINAMATH_CALUDE_angle_between_polar_lines_theorem_l1758_175830


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_singleton_one_l1758_175868

def A : Set ℝ := {-1, 1, 1/2, 3}

def B : Set ℝ := {y | ∃ x ∈ A, y = x^2}

theorem A_intersect_B_eq_singleton_one : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_singleton_one_l1758_175868


namespace NUMINAMATH_CALUDE_tire_sample_size_l1758_175859

theorem tire_sample_size (p : ℝ) (n : ℕ) (h1 : p = 0.015) (h2 : n = 168) :
  1 - (1 - p) ^ n > 0.92 := by
  sorry

end NUMINAMATH_CALUDE_tire_sample_size_l1758_175859


namespace NUMINAMATH_CALUDE_hyunji_pencils_l1758_175862

/-- Given an initial number of pencils, the number given away, and the number received,
    calculate the final number of pencils. -/
def final_pencils (initial given_away received : ℕ) : ℕ :=
  initial - given_away + received

/-- Theorem stating that with 20 initial pencils, giving away 7 and receiving 5
    results in 18 pencils. -/
theorem hyunji_pencils : final_pencils 20 7 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_hyunji_pencils_l1758_175862


namespace NUMINAMATH_CALUDE_rectangle_with_hole_area_formula_l1758_175852

/-- The area of a rectangle with a hole -/
def rectangle_with_hole_area (x : ℝ) : ℝ :=
  (2*x + 8) * (x + 6) - (3*x - 4) * (x + 1)

/-- Theorem: The area of the rectangle with a hole is equal to -x^2 + 21x + 52 -/
theorem rectangle_with_hole_area_formula (x : ℝ) :
  rectangle_with_hole_area x = -x^2 + 21*x + 52 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_with_hole_area_formula_l1758_175852


namespace NUMINAMATH_CALUDE_sequence_properties_l1758_175850

def sequence_sum (n : ℕ) : ℝ := n^2

def sequence_term (n : ℕ+) : ℝ := 2 * n.val - 1

def is_geometric_triple (a b c : ℝ) : Prop :=
  a * c = b^2

theorem sequence_properties :
  (∀ n : ℕ+, n > 1 →
    1 / Real.sqrt (sequence_sum (n.val - 1)) -
    1 / Real.sqrt (sequence_sum n.val) -
    1 / Real.sqrt (sequence_sum n.val * sequence_sum (n.val - 1)) = 0) →
  sequence_term 1 = 1 →
  (∀ n : ℕ+, sequence_term n = 2 * n.val - 1) ∧
  (∀ m t : ℕ+, 1 < m → m < t → t ≤ 100 →
    is_geometric_triple (1 / sequence_term 2) (1 / sequence_term m) (1 / sequence_term t) ↔
    (m = 5 ∧ t = 14) ∨ (m = 8 ∧ t = 38) ∨ (m = 11 ∧ t = 74)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1758_175850


namespace NUMINAMATH_CALUDE_gideon_future_age_l1758_175874

def century : ℕ := 100
def gideon_current_age : ℕ := 45

def marbles_given_away (total_marbles : ℕ) : ℕ :=
  (3 * total_marbles) / 4

def remaining_marbles (total_marbles : ℕ) : ℕ :=
  total_marbles - marbles_given_away total_marbles

def doubled_remaining_marbles (total_marbles : ℕ) : ℕ :=
  2 * remaining_marbles total_marbles

theorem gideon_future_age : 
  ∃ (years_from_now : ℕ), 
    gideon_current_age + years_from_now = doubled_remaining_marbles century ∧ 
    years_from_now = 5 := by
  sorry

end NUMINAMATH_CALUDE_gideon_future_age_l1758_175874


namespace NUMINAMATH_CALUDE_square_area_proof_l1758_175879

theorem square_area_proof (x : ℝ) : 
  (5 * x - 20 : ℝ) = (25 - 2 * x : ℝ) → 
  (5 * x - 20 : ℝ) > 0 → 
  ((5 * x - 20 : ℝ) * (5 * x - 20 : ℝ)) = 7225 / 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l1758_175879


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1758_175863

-- Define the arithmetic operations
def calculation1 : ℤ := 36 * 17 + 129
def calculation2 : ℤ := 320 * (300 - 294)
def calculation3 : ℤ := 25 * 5 * 4
def calculation4 : ℚ := 18.45 - 25.6 - 24.4

-- Theorem statements
theorem arithmetic_calculations :
  (calculation1 = 741) ∧
  (calculation2 = 1920) ∧
  (calculation3 = 500) ∧
  (calculation4 = -31.55) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1758_175863


namespace NUMINAMATH_CALUDE_lifeguard_swim_speed_l1758_175843

theorem lifeguard_swim_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (front_crawl_time : ℝ) 
  (breaststroke_speed : ℝ) 
  (h1 : total_distance = 500)
  (h2 : total_time = 12)
  (h3 : front_crawl_time = 8)
  (h4 : breaststroke_speed = 35)
  : ∃ front_crawl_speed : ℝ, 
    front_crawl_speed * front_crawl_time + 
    breaststroke_speed * (total_time - front_crawl_time) = total_distance ∧ 
    front_crawl_speed = 45 := by
  sorry

end NUMINAMATH_CALUDE_lifeguard_swim_speed_l1758_175843


namespace NUMINAMATH_CALUDE_max_a_for_inequality_l1758_175846

theorem max_a_for_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), |x - 2| + |x - 8| ≥ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), |x - 2| + |x - 8| ≥ b) → b ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_inequality_l1758_175846


namespace NUMINAMATH_CALUDE_translated_cosine_monotonicity_l1758_175816

open Real

theorem translated_cosine_monotonicity (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = 2 * cos (2 * x)) →
  (∀ x, g x = f (x - π / 6)) →
  (∀ x ∈ Set.Icc (2 * a) (7 * π / 6), StrictMono g) →
  a ∈ Set.Icc (π / 3) (7 * π / 12) :=
sorry

end NUMINAMATH_CALUDE_translated_cosine_monotonicity_l1758_175816


namespace NUMINAMATH_CALUDE_square_of_negative_product_l1758_175826

theorem square_of_negative_product (m n : ℝ) : (-2 * m * n)^2 = 4 * m^2 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l1758_175826


namespace NUMINAMATH_CALUDE_square_sum_ge_twice_product_l1758_175800

theorem square_sum_ge_twice_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_twice_product_l1758_175800


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1758_175810

theorem complex_equation_solution :
  ∀ z : ℂ, z / (1 + Complex.I) = Complex.I ^ 2015 + Complex.I ^ 2016 → z = -2 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1758_175810


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l1758_175897

theorem sum_remainder_mod_seven :
  (123450 + 123451 + 123452 + 123453 + 123454 + 123455) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l1758_175897


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l1758_175865

-- Define a random variable following normal distribution
def normalDistribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def probability (ξ : normalDistribution 4 σ) (event : Set ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_symmetry 
  (σ : ℝ) 
  (ξ : normalDistribution 4 σ) 
  (h : probability ξ {x | x > 8} = 0.4) : 
  probability ξ {x | x < 0} = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l1758_175865


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1758_175875

theorem system_solution_ratio (x y c d : ℝ) : 
  (4 * x - 3 * y = c) →
  (6 * y - 8 * x = d) →
  (d ≠ 0) →
  (∃ x y, (4 * x - 3 * y = c) ∧ (6 * y - 8 * x = d)) →
  c / d = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1758_175875


namespace NUMINAMATH_CALUDE_multiples_difference_squared_l1758_175891

def a : ℕ := (Finset.filter (λ x => x % 7 = 0) (Finset.range 60)).card

def b : ℕ := (Finset.filter (λ x => x % 3 = 0 ∨ x % 7 = 0) (Finset.range 60)).card

theorem multiples_difference_squared : (a - b)^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_multiples_difference_squared_l1758_175891


namespace NUMINAMATH_CALUDE_first_2500_even_integers_digits_l1758_175887

/-- The total number of digits used to write the first n positive even integers -/
def totalDigits (n : ℕ) : ℕ :=
  sorry

/-- The 2500th positive even integer -/
def evenInteger2500 : ℕ := 5000

theorem first_2500_even_integers_digits :
  totalDigits 2500 = 9449 :=
sorry

end NUMINAMATH_CALUDE_first_2500_even_integers_digits_l1758_175887


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l1758_175856

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 23 ∧ 
  correct_value = 48 ∧ 
  corrected_mean = 32.5 →
  ∃ initial_mean : ℝ, 
    initial_mean = 32 ∧ 
    n * corrected_mean = n * initial_mean + (correct_value - wrong_value) :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l1758_175856


namespace NUMINAMATH_CALUDE_g_3_equals_109_l1758_175876

def g (x : ℝ) : ℝ := 7 * x^3 - 8 * x^2 - 5 * x + 7

theorem g_3_equals_109 : g 3 = 109 := by
  sorry

end NUMINAMATH_CALUDE_g_3_equals_109_l1758_175876


namespace NUMINAMATH_CALUDE_f_min_value_l1758_175890

/-- The quadratic function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The minimum value of f(x) is 2 -/
theorem f_min_value : ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l1758_175890
