import Mathlib

namespace NUMINAMATH_CALUDE_burger_combinations_l2377_237751

/-- The number of different toppings available. -/
def num_toppings : ℕ := 10

/-- The number of choices for meat patties. -/
def patty_choices : ℕ := 4

/-- Theorem stating the total number of burger combinations. -/
theorem burger_combinations :
  (2 ^ num_toppings) * patty_choices = 4096 :=
sorry

end NUMINAMATH_CALUDE_burger_combinations_l2377_237751


namespace NUMINAMATH_CALUDE_last_three_digits_of_1973_power_46_l2377_237780

theorem last_three_digits_of_1973_power_46 :
  1973^46 % 1000 = 689 := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_1973_power_46_l2377_237780


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2377_237768

theorem quadratic_roots_property (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) → 
  (3 * b^2 + 9 * b - 21 = 0) → 
  (3*a - 4) * (6*b - 8) = 14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2377_237768


namespace NUMINAMATH_CALUDE_total_worth_of_cloth_sold_l2377_237721

/-- Calculates the total worth of cloth sold through two agents given their commission rates and amounts -/
theorem total_worth_of_cloth_sold 
  (rate_A rate_B : ℝ) 
  (commission_A commission_B : ℝ) 
  (h1 : rate_A = 0.025) 
  (h2 : rate_B = 0.03) 
  (h3 : commission_A = 21) 
  (h4 : commission_B = 27) : 
  ∃ (total_worth : ℝ), total_worth = commission_A / rate_A + commission_B / rate_B :=
sorry

end NUMINAMATH_CALUDE_total_worth_of_cloth_sold_l2377_237721


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2377_237765

theorem complex_fraction_simplification :
  (7 + 8 * Complex.I) / (3 - 4 * Complex.I) = -11/25 + 52/25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2377_237765


namespace NUMINAMATH_CALUDE_water_usage_problem_l2377_237720

/-- Calculates the water charge based on usage --/
def water_charge (usage : ℕ) : ℚ :=
  if usage ≤ 24 then 1.8 * usage
  else 1.8 * 24 + 4 * (usage - 24)

/-- Represents the water usage problem --/
theorem water_usage_problem :
  ∃ (zhang_usage wang_usage : ℕ),
    zhang_usage > 24 ∧
    wang_usage ≤ 24 ∧
    water_charge zhang_usage - water_charge wang_usage = 19.2 ∧
    zhang_usage = 27 ∧
    wang_usage = 20 :=
by
  sorry

#eval water_charge 27  -- Should output 55.2
#eval water_charge 20  -- Should output 36

end NUMINAMATH_CALUDE_water_usage_problem_l2377_237720


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l2377_237701

/-- Given two real numbers p and q satisfying pq = 16 and p + q = 8, 
    prove that p^2 + q^2 = 32. -/
theorem square_sum_given_product_and_sum (p q : ℝ) 
  (h1 : p * q = 16) (h2 : p + q = 8) : p^2 + q^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l2377_237701


namespace NUMINAMATH_CALUDE_cell_growth_proof_l2377_237745

def geometric_sequence (a₁ : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a₁ * r^(n - 1)

theorem cell_growth_proof :
  let initial_cells : ℕ := 3
  let growth_factor : ℕ := 2
  let num_terms : ℕ := 5
  geometric_sequence initial_cells growth_factor num_terms = 48 := by
  sorry

end NUMINAMATH_CALUDE_cell_growth_proof_l2377_237745


namespace NUMINAMATH_CALUDE_x_equation_solution_l2377_237722

theorem x_equation_solution (x : ℝ) (h : x + 1/x = 3) :
  x^12 - 7*x^8 + 2*x^4 = 44387*x - 15088 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_solution_l2377_237722


namespace NUMINAMATH_CALUDE_net_effect_theorem_l2377_237788

/-- Calculates the net effect on sale given price reduction, sale increase, tax, and discount -/
def net_effect_on_sale (price_reduction : ℝ) (sale_increase : ℝ) (tax : ℝ) (discount : ℝ) : ℝ :=
  let new_price_factor := 1 - price_reduction
  let new_quantity_factor := 1 + sale_increase
  let after_tax_factor := 1 + tax
  let after_discount_factor := 1 - discount
  new_price_factor * new_quantity_factor * after_tax_factor * after_discount_factor

/-- Theorem stating the net effect on sale given specific conditions -/
theorem net_effect_theorem :
  net_effect_on_sale 0.60 1.50 0.10 0.05 = 1.045 := by
  sorry

end NUMINAMATH_CALUDE_net_effect_theorem_l2377_237788


namespace NUMINAMATH_CALUDE_triangle_inequality_l2377_237750

/-- Given points P, Q, R, S on a line with PQ = a, PR = b, PS = c,
    if PQ and RS can be rotated to form a non-degenerate triangle,
    then a < c/2 and b < a + c/2 -/
theorem triangle_inequality (a b c : ℝ) 
  (h_order : 0 < a ∧ a < b ∧ b < c)
  (h_triangle : 2*b > c ∧ c > a ∧ c > b - a) :
  a < c/2 ∧ b < a + c/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2377_237750


namespace NUMINAMATH_CALUDE_y_divisibility_l2377_237716

def y : ℕ := 80 + 120 + 160 + 240 + 360 + 400 + 3600

theorem y_divisibility :
  (∃ k : ℕ, y = 5 * k) ∧
  (∃ k : ℕ, y = 10 * k) ∧
  (∃ k : ℕ, y = 20 * k) ∧
  ¬(∃ k : ℕ, y = 40 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l2377_237716


namespace NUMINAMATH_CALUDE_work_completion_time_l2377_237735

/-- Represents the work rate of one person for one hour -/
structure WorkRate where
  man : ℝ
  woman : ℝ

/-- Represents a work scenario -/
structure WorkScenario where
  men : ℕ
  women : ℕ
  hours : ℝ
  days : ℝ

/-- Calculates the total work done in a scenario -/
def totalWork (rate : WorkRate) (scenario : WorkScenario) : ℝ :=
  (scenario.men * rate.man + scenario.women * rate.woman) * scenario.hours * scenario.days

/-- The theorem to be proved -/
theorem work_completion_time 
  (rate : WorkRate)
  (scenario1 scenario2 scenario3 : WorkScenario) :
  scenario1.men = 1 ∧ 
  scenario1.women = 3 ∧ 
  scenario1.hours = 7 ∧
  scenario2.men = 4 ∧ 
  scenario2.women = 4 ∧ 
  scenario2.hours = 3 ∧ 
  scenario2.days = 7 ∧
  scenario3.men = 7 ∧ 
  scenario3.women = 0 ∧ 
  scenario3.hours = 4 ∧ 
  scenario3.days = 5.000000000000001 ∧
  totalWork rate scenario1 = totalWork rate scenario2 ∧
  totalWork rate scenario2 = totalWork rate scenario3
  →
  scenario1.days = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2377_237735


namespace NUMINAMATH_CALUDE_coinciding_white_pairs_l2377_237738

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCount where
  red : Nat
  blue : Nat
  white : Nat

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  redRed : Nat
  blueBlue : Nat
  redWhite : Nat
  whiteWhite : Nat

/-- The main theorem that proves the number of coinciding white pairs -/
theorem coinciding_white_pairs
  (initial_count : TriangleCount)
  (coinciding : CoincidingPairs)
  (h1 : initial_count.red = 2)
  (h2 : initial_count.blue = 4)
  (h3 : initial_count.white = 6)
  (h4 : coinciding.redRed = 1)
  (h5 : coinciding.blueBlue = 2)
  (h6 : coinciding.redWhite = 2)
  : coinciding.whiteWhite = 4 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_white_pairs_l2377_237738


namespace NUMINAMATH_CALUDE_equal_average_groups_product_l2377_237709

theorem equal_average_groups_product (groups : Fin 3 → List ℕ) : 
  (∀ i : Fin 3, ∀ n ∈ groups i, 1 ≤ n ∧ n ≤ 99) →
  (groups 0).sum + (groups 1).sum + (groups 2).sum = List.sum (List.range 99) →
  (groups 0).length + (groups 1).length + (groups 2).length = 99 →
  (groups 0).sum / (groups 0).length = (groups 1).sum / (groups 1).length →
  (groups 1).sum / (groups 1).length = (groups 2).sum / (groups 2).length →
  ((groups 0).sum / (groups 0).length) * ((groups 1).sum / (groups 1).length) * ((groups 2).sum / (groups 2).length) = 125000 := by
sorry

end NUMINAMATH_CALUDE_equal_average_groups_product_l2377_237709


namespace NUMINAMATH_CALUDE_account_balance_after_transactions_l2377_237728

/-- Calculates the final account balance after a series of transactions --/
def finalBalance (initialBalance : ℚ) 
  (transfer1 transfer2 transfer3 transfer4 transfer5 : ℚ)
  (serviceCharge1 serviceCharge2 serviceCharge3 serviceCharge4 serviceCharge5 : ℚ) : ℚ :=
  initialBalance - transfer1 - (transfer3 + serviceCharge3) - (transfer5 + serviceCharge5)

/-- Theorem stating the final account balance after the given transactions --/
theorem account_balance_after_transactions 
  (initialBalance : ℚ)
  (transfer1 transfer2 transfer3 transfer4 transfer5 : ℚ)
  (serviceCharge1 serviceCharge2 serviceCharge3 serviceCharge4 serviceCharge5 : ℚ)
  (h1 : initialBalance = 400)
  (h2 : transfer1 = 90)
  (h3 : transfer2 = 60)
  (h4 : transfer3 = 50)
  (h5 : transfer4 = 120)
  (h6 : transfer5 = 200)
  (h7 : serviceCharge1 = 0.02 * transfer1)
  (h8 : serviceCharge2 = 0.02 * transfer2)
  (h9 : serviceCharge3 = 0.02 * transfer3)
  (h10 : serviceCharge4 = 0.025 * transfer4)
  (h11 : serviceCharge5 = 0.03 * transfer5) :
  finalBalance initialBalance transfer1 transfer2 transfer3 transfer4 transfer5
    serviceCharge1 serviceCharge2 serviceCharge3 serviceCharge4 serviceCharge5 = 53 := by
  sorry


end NUMINAMATH_CALUDE_account_balance_after_transactions_l2377_237728


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2377_237756

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 = 2*x₁ + 1) → (x₂^2 = 2*x₂ + 1) → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2377_237756


namespace NUMINAMATH_CALUDE_first_digit_base8_is_3_l2377_237754

/-- The base 3 representation of y -/
def y_base3 : List Nat := [2, 1, 2, 0, 2, 1, 2]

/-- Convert a list of digits in base b to a natural number -/
def to_nat (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (λ d acc => d + b * acc) 0

/-- The value of y in base 10 -/
def y : Nat := to_nat y_base3 3

/-- Get the first digit of a number in base b -/
def first_digit (n : Nat) (b : Nat) : Nat :=
  n / (b ^ ((Nat.log b n) - 1))

theorem first_digit_base8_is_3 : first_digit y 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_base8_is_3_l2377_237754


namespace NUMINAMATH_CALUDE_largest_non_sum_30_and_composite_l2377_237749

/-- A function that checks if a number is composite -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

/-- The property we want to prove for 211 -/
def IsLargestNonSum30AndComposite (m : ℕ) : Prop :=
  (∀ n > m, ∃ k c : ℕ, n = 30 * k + c ∧ k > 0 ∧ IsComposite c) ∧
  (¬∃ k c : ℕ, m = 30 * k + c ∧ k > 0 ∧ IsComposite c)

/-- The main theorem -/
theorem largest_non_sum_30_and_composite :
  IsLargestNonSum30AndComposite 211 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_30_and_composite_l2377_237749


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2377_237779

theorem polynomial_simplification (s : ℝ) : 
  (2 * s^2 + 5 * s - 3) - (s^2 + 9 * s - 6) = s^2 - 4 * s + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2377_237779


namespace NUMINAMATH_CALUDE_binomial_10_4_l2377_237742

theorem binomial_10_4 : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_4_l2377_237742


namespace NUMINAMATH_CALUDE_stripe_area_on_cylinder_l2377_237760

/-- The area of a stripe on a cylindrical tank -/
theorem stripe_area_on_cylinder (diameter : Real) (stripe_width : Real) (revolutions : Real) :
  diameter = 40 ∧ stripe_width = 4 ∧ revolutions = 3 →
  stripe_width * revolutions * π * diameter = 480 * π := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylinder_l2377_237760


namespace NUMINAMATH_CALUDE_meadowbrook_impossibility_l2377_237704

theorem meadowbrook_impossibility : ¬ ∃ (h c : ℕ), 21 * h + 6 * c = 74 := by
  sorry

end NUMINAMATH_CALUDE_meadowbrook_impossibility_l2377_237704


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2377_237727

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {-1, 2, 3}

theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2377_237727


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_and_point_l2377_237776

-- Define the equation
def equation (x y : ℝ) : Prop :=
  ((x - 1)^2 + (y + 2)^2) * (x^2 - y^2) = 0

-- Define the point (1, -2)
def point : ℝ × ℝ := (1, -2)

-- Define the two lines
def line1 (x y : ℝ) : Prop := x + y = 0
def line2 (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem equation_represents_two_lines_and_point :
  ∀ x y : ℝ, equation x y ↔ (x = point.1 ∧ y = point.2) ∨ line1 x y ∨ line2 x y :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_and_point_l2377_237776


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_ratio_l2377_237781

/-- An ellipse intersecting with a line -/
structure EllipseLineIntersection where
  /-- Coefficient of x^2 in the ellipse equation -/
  m : ℝ
  /-- Coefficient of y^2 in the ellipse equation -/
  n : ℝ
  /-- x-coordinate of point M -/
  x₁ : ℝ
  /-- y-coordinate of point M -/
  y₁ : ℝ
  /-- x-coordinate of point N -/
  x₂ : ℝ
  /-- y-coordinate of point N -/
  y₂ : ℝ
  /-- Ellipse equation for point M -/
  ellipse_eq_m : m * x₁^2 + n * y₁^2 = 1
  /-- Ellipse equation for point N -/
  ellipse_eq_n : m * x₂^2 + n * y₂^2 = 1
  /-- Line equation for point M -/
  line_eq_m : x₁ + y₁ = 1
  /-- Line equation for point N -/
  line_eq_n : x₂ + y₂ = 1
  /-- Slope of OP, where P is the midpoint of MN -/
  slope_op : (y₁ + y₂) / (x₁ + x₂) = Real.sqrt 2 / 2

/-- Theorem: If the slope of OP is √2/2, then m/n = √2/2 -/
theorem ellipse_line_intersection_ratio (e : EllipseLineIntersection) : e.m / e.n = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_ratio_l2377_237781


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l2377_237767

theorem rectangle_width_decrease (L W : ℝ) (h1 : L > 0) (h2 : W > 0) :
  let new_length := 1.4 * L
  let new_width := W / 1.4
  (new_length * new_width = L * W) ∧
  (2 * new_length + 2 * new_width = 2 * L + 2 * W) →
  (W - new_width) / W = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l2377_237767


namespace NUMINAMATH_CALUDE_smallest_with_ten_factors_l2377_237792

/-- A function that returns the number of distinct positive factors of a natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly ten distinct positive factors -/
def has_ten_factors (n : ℕ) : Prop := num_factors n = 10

theorem smallest_with_ten_factors :
  ∃ (n : ℕ), has_ten_factors n ∧ ∀ m : ℕ, m < n → ¬(has_ten_factors m) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_ten_factors_l2377_237792


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l2377_237703

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 20) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 418 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l2377_237703


namespace NUMINAMATH_CALUDE_bookstore_problem_l2377_237734

theorem bookstore_problem (x y : ℕ) 
  (h1 : x + y = 5000)
  (h2 : (x - 400) / 2 - (y + 400) = 400) :
  x - y = 3000 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_problem_l2377_237734


namespace NUMINAMATH_CALUDE_dihedral_angle_eq_inclination_l2377_237719

/-- A pyramid with an isosceles triangular base and inclined lateral edges -/
structure IsoscelesPyramid where
  -- Angle between equal sides of the base triangle
  α : Real
  -- Angle of inclination of lateral edges to the base plane
  φ : Real
  -- Assumption that α and φ are valid angles
  h_α_range : 0 < α ∧ α < π
  h_φ_range : 0 < φ ∧ φ < π/2

/-- The dihedral angle at the edge connecting the apex to the vertex of angle α -/
def dihedral_angle (p : IsoscelesPyramid) : Real :=
  -- Definition of dihedral angle (to be proved equal to φ)
  sorry

/-- Theorem: The dihedral angle is equal to the inclination angle of lateral edges -/
theorem dihedral_angle_eq_inclination (p : IsoscelesPyramid) :
  dihedral_angle p = p.φ :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_eq_inclination_l2377_237719


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_side_length_l2377_237743

theorem rectangle_area_ratio_side_length (area_ratio : ℚ) (p q r : ℕ) : 
  area_ratio = 500 / 125 →
  (p * Real.sqrt q : ℝ) / r = Real.sqrt (area_ratio) →
  p + q + r = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_side_length_l2377_237743


namespace NUMINAMATH_CALUDE_escalator_length_l2377_237777

/-- The length of an escalator given specific conditions -/
theorem escalator_length : 
  ∀ (escalator_speed person_speed time : ℝ),
    escalator_speed = 10 →
    person_speed = 4 →
    time = 8 →
    (escalator_speed + person_speed) * time = 112 := by
  sorry

end NUMINAMATH_CALUDE_escalator_length_l2377_237777


namespace NUMINAMATH_CALUDE_yah_to_bah_conversion_l2377_237711

/-- Define conversion rates between bahs, rahs, and yahs -/
def bah_to_rah_rate : ℚ := 27 / 18
def rah_to_yah_rate : ℚ := 20 / 12

/-- Theorem stating the equivalence between 800 yahs and 320 bahs -/
theorem yah_to_bah_conversion : 
  ∀ (bahs rahs yahs : ℚ),
  (18 : ℚ) * bahs = (27 : ℚ) * rahs →
  (12 : ℚ) * rahs = (20 : ℚ) * yahs →
  (800 : ℚ) * yahs = (320 : ℚ) * bahs := by
  sorry

end NUMINAMATH_CALUDE_yah_to_bah_conversion_l2377_237711


namespace NUMINAMATH_CALUDE_total_toy_cost_l2377_237763

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

theorem total_toy_cost : football_cost + marbles_cost = 12.30 := by
  sorry

end NUMINAMATH_CALUDE_total_toy_cost_l2377_237763


namespace NUMINAMATH_CALUDE_smith_family_laundry_l2377_237710

/-- The number of bath towels Kylie uses in one month -/
def kylie_towels : ℕ := 3

/-- The number of bath towels Kylie's daughters use in one month -/
def daughters_towels : ℕ := 6

/-- The number of bath towels Kylie's husband uses in one month -/
def husband_towels : ℕ := 3

/-- The number of bath towels that fit in one load of laundry -/
def towels_per_load : ℕ := 4

/-- The total number of bath towels used by the Smith family in one month -/
def total_towels : ℕ := kylie_towels + daughters_towels + husband_towels

/-- The number of loads of laundry needed to clean all used towels -/
def loads_needed : ℕ := (total_towels + towels_per_load - 1) / towels_per_load

theorem smith_family_laundry : loads_needed = 3 := by
  sorry

end NUMINAMATH_CALUDE_smith_family_laundry_l2377_237710


namespace NUMINAMATH_CALUDE_function_properties_l2377_237730

noncomputable def f (a b x : ℝ) : ℝ := 6 * Real.log x - a * x^2 - 7 * x + b

theorem function_properties (a b : ℝ) :
  (∀ x, x > 0 → HasDerivAt (f a b) ((6 / x) - 2 * a * x - 7) x) →
  HasDerivAt (f a b) 0 2 →
  (a = -1 ∧
   (∀ x, 0 < x ∧ x < 3/2 → HasDerivAt (f a b) ((x - 2) * (2 * x - 3) / x) x ∧ ((x - 2) * (2 * x - 3) / x > 0)) ∧
   (∀ x, 3/2 < x ∧ x < 2 → HasDerivAt (f a b) ((x - 2) * (2 * x - 3) / x) x ∧ ((x - 2) * (2 * x - 3) / x < 0)) ∧
   (∀ x, x > 2 → HasDerivAt (f a b) ((x - 2) * (2 * x - 3) / x) x ∧ ((x - 2) * (2 * x - 3) / x > 0)) ∧
   (33/4 - 6 * Real.log (3/2) < b ∧ b < 10 - 6 * Real.log 2)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2377_237730


namespace NUMINAMATH_CALUDE_histogram_frequency_l2377_237771

theorem histogram_frequency (m : ℕ) (S1 S2 S3 : ℚ) :
  m ≥ 3 →
  S1 + S2 + S3 = (1 : ℚ) / 4 * (1 - (S1 + S2 + S3)) →
  S2 - S1 = S3 - S2 →
  S1 = (1 : ℚ) / 20 →
  (120 : ℚ) * S3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_histogram_frequency_l2377_237771


namespace NUMINAMATH_CALUDE_solution_range_l2377_237770

-- Define the inequality as a function of x and a
def inequality (x a : ℝ) : Prop :=
  3 * x - (a * x + 1) / 2 < 4 * x / 3

-- State the theorem
theorem solution_range (a : ℝ) : 
  (inequality 3 a) → a > 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_range_l2377_237770


namespace NUMINAMATH_CALUDE_smallest_cyclic_divisible_by_1989_l2377_237700

def is_cyclic_divisible_by_1989 (n : ℕ) : Prop :=
  ∀ k : ℕ, k < 10^n → ∀ i : ℕ, i < n → (k * 10^i + k / 10^(n - i)) % 1989 = 0

theorem smallest_cyclic_divisible_by_1989 :
  (∀ m < 48, ¬ is_cyclic_divisible_by_1989 m) ∧ is_cyclic_divisible_by_1989 48 :=
sorry

end NUMINAMATH_CALUDE_smallest_cyclic_divisible_by_1989_l2377_237700


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l2377_237705

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l2377_237705


namespace NUMINAMATH_CALUDE_marble_weight_difference_is_8_l2377_237731

/-- Calculates the difference in weight between red and yellow marbles -/
def marble_weight_difference (total_marbles : ℕ) (yellow_marbles : ℕ) (blue_red_ratio : ℚ) 
  (yellow_weight : ℝ) (red_yellow_weight_ratio : ℝ) : ℝ :=
  let remaining_marbles := total_marbles - yellow_marbles
  let red_marbles := (remaining_marbles : ℝ) * (1 / (1 + blue_red_ratio)) * (blue_red_ratio / (1 + blue_red_ratio))⁻¹
  let red_weight := yellow_weight * red_yellow_weight_ratio
  red_weight - yellow_weight

theorem marble_weight_difference_is_8 :
  marble_weight_difference 19 5 (3/4) 8 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_marble_weight_difference_is_8_l2377_237731


namespace NUMINAMATH_CALUDE_smallest_780_divisible_by_1125_l2377_237729

def is_composed_of_780 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 7 ∨ d = 8 ∨ d = 0

theorem smallest_780_divisible_by_1125 :
  ∀ n : ℕ, n > 0 → is_composed_of_780 n → n % 1125 = 0 → n ≥ 77778000 :=
sorry

end NUMINAMATH_CALUDE_smallest_780_divisible_by_1125_l2377_237729


namespace NUMINAMATH_CALUDE_factory_sample_theorem_l2377_237762

/-- Given a factory with total products, a sample size, and products from one workshop,
    calculate the number of products drawn from this workshop in a stratified sampling. -/
def stratifiedSampleSize (totalProducts sampleSize workshopProducts : ℕ) : ℕ :=
  (workshopProducts * sampleSize) / totalProducts

/-- Theorem stating that for the given values, the stratified sample size is 16. -/
theorem factory_sample_theorem :
  stratifiedSampleSize 2048 128 256 = 16 := by
  sorry

end NUMINAMATH_CALUDE_factory_sample_theorem_l2377_237762


namespace NUMINAMATH_CALUDE_fourth_root_of_four_powers_l2377_237795

theorem fourth_root_of_four_powers : (4^7 + 4^7 + 4^7 + 4^7 : ℝ)^(1/4) = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_four_powers_l2377_237795


namespace NUMINAMATH_CALUDE_inequality_implies_product_l2377_237794

theorem inequality_implies_product (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) 
  (h3 : 4 * Real.log x + 2 * Real.log y ≥ x^2 + 4*y - 4) : 
  x * y = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_product_l2377_237794


namespace NUMINAMATH_CALUDE_expression_simplification_l2377_237752

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (3 * a + b / 3)⁻¹ * ((3 * a)⁻¹ + (b / 3)⁻¹) = (a * b)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2377_237752


namespace NUMINAMATH_CALUDE_book_price_increase_l2377_237799

theorem book_price_increase (original_price : ℝ) : 
  original_price > 0 →
  original_price * 1.5 = 450 →
  original_price = 300 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l2377_237799


namespace NUMINAMATH_CALUDE_negation_existence_statement_l2377_237790

open Real

theorem negation_existence_statement (f : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x < 0) ↔ (∀ x : ℝ, f x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existence_statement_l2377_237790


namespace NUMINAMATH_CALUDE_inequality_proof_l2377_237737

theorem inequality_proof (x : ℝ) (h : x > 0) : x + (2016^2016)/x^2016 ≥ 2017 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2377_237737


namespace NUMINAMATH_CALUDE_assignment_count_proof_l2377_237757

/-- The number of ways to assign doctors and nurses to schools -/
def assignment_count : ℕ := 540

/-- The number of doctors -/
def num_doctors : ℕ := 3

/-- The number of nurses -/
def num_nurses : ℕ := 6

/-- The number of schools -/
def num_schools : ℕ := 3

/-- The number of doctors assigned to each school -/
def doctors_per_school : ℕ := 1

/-- The number of nurses assigned to each school -/
def nurses_per_school : ℕ := 2

theorem assignment_count_proof : 
  assignment_count = 
    (num_doctors.factorial * (num_nurses.factorial / (nurses_per_school.factorial ^ num_schools))) := by
  sorry

end NUMINAMATH_CALUDE_assignment_count_proof_l2377_237757


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2377_237793

theorem sum_of_xyz (a b : ℝ) (x y z : ℕ+) : 
  a^2 = 9/14 ∧ 
  b^2 = (3 + Real.sqrt 7)^2 / 14 ∧ 
  a < 0 ∧ 
  b > 0 ∧ 
  (a + b)^3 = (x : ℝ) * Real.sqrt y / z →
  x + y + z = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2377_237793


namespace NUMINAMATH_CALUDE_min_handshakes_30_people_l2377_237706

/-- The minimum number of handshakes in a gathering -/
def min_handshakes (n : ℕ) (k : ℕ) : ℕ := n * k / 2

/-- Theorem: In a gathering of 30 people, where each person shakes hands
    with at least three other people, the minimum possible number of handshakes is 45 -/
theorem min_handshakes_30_people :
  let n : ℕ := 30
  let k : ℕ := 3
  min_handshakes n k = 45 := by
  sorry


end NUMINAMATH_CALUDE_min_handshakes_30_people_l2377_237706


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2377_237740

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 10) : x^2 + 1/x^2 = 98 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2377_237740


namespace NUMINAMATH_CALUDE_square_root_sum_equals_ten_l2377_237783

theorem square_root_sum_equals_ten : 
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_ten_l2377_237783


namespace NUMINAMATH_CALUDE_sock_order_ratio_l2377_237702

/-- Represents the number of pairs of socks -/
structure SockOrder where
  black : ℕ
  blue : ℕ

/-- Represents the price of socks -/
structure SockPrice where
  blue : ℝ

/-- Calculates the total cost of a sock order given the prices -/
def totalCost (order : SockOrder) (price : SockPrice) : ℝ :=
  (order.black * 3 * price.blue) + (order.blue * price.blue)

/-- The theorem to be proved -/
theorem sock_order_ratio (original : SockOrder) (price : SockPrice) : 
  original.black = 5 →
  totalCost { black := original.blue, blue := original.black } price = 1.6 * totalCost original price →
  (original.black : ℝ) / original.blue = 5 / 14 := by
  sorry

#check sock_order_ratio

end NUMINAMATH_CALUDE_sock_order_ratio_l2377_237702


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l2377_237718

def i : ℂ := Complex.I

def z : ℂ := i + 2 * i^2 + 3 * i^3

theorem z_in_third_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l2377_237718


namespace NUMINAMATH_CALUDE_multiply_subtract_distribute_computation_proof_l2377_237753

theorem multiply_subtract_distribute (a b c : ℕ) : a * c - b * c = (a - b) * c := by sorry

theorem computation_proof : 72 * 808 - 22 * 808 = 40400 := by sorry

end NUMINAMATH_CALUDE_multiply_subtract_distribute_computation_proof_l2377_237753


namespace NUMINAMATH_CALUDE_investment_percentage_l2377_237764

/-- Given two investors with equal initial investments, where one investor's value quadruples and ends up with $1900 more than the other, prove that the other investor's final value is 20% of their initial investment. -/
theorem investment_percentage (initial_investment : ℝ) (jackson_final : ℝ) (brandon_final : ℝ) :
  initial_investment > 0 →
  jackson_final = 4 * initial_investment →
  jackson_final - brandon_final = 1900 →
  brandon_final / initial_investment = 0.2 := by
sorry

end NUMINAMATH_CALUDE_investment_percentage_l2377_237764


namespace NUMINAMATH_CALUDE_stamp_ratio_problem_l2377_237713

theorem stamp_ratio_problem (k a : ℕ) : 
  k > 0 ∧ a > 0 →  -- Initial numbers of stamps are positive
  (k - 12) / (a + 12) = 8 / 6 →  -- Ratio after exchange
  k - 12 = a + 12 + 32 →  -- Kaye has 32 more stamps after exchange
  k / a = 5 / 3 :=  -- Initial ratio
by sorry

end NUMINAMATH_CALUDE_stamp_ratio_problem_l2377_237713


namespace NUMINAMATH_CALUDE_cubes_remaining_after_removal_l2377_237746

/-- Represents a cube arrangement --/
structure CubeArrangement where
  width : Nat
  height : Nat
  depth : Nat

/-- Calculates the total number of cubes in an arrangement --/
def totalCubes (arrangement : CubeArrangement) : Nat :=
  arrangement.width * arrangement.height * arrangement.depth

/-- Represents the number of vertical columns removed from the front --/
def removedColumns : Nat := 6

/-- Represents the height of each removed column --/
def removedColumnHeight : Nat := 3

/-- Calculates the number of remaining cubes after removal --/
def remainingCubes (arrangement : CubeArrangement) : Nat :=
  totalCubes arrangement - (removedColumns * removedColumnHeight)

/-- The theorem to be proved --/
theorem cubes_remaining_after_removal :
  let arrangement : CubeArrangement := { width := 4, height := 4, depth := 4 }
  remainingCubes arrangement = 46 := by
  sorry

end NUMINAMATH_CALUDE_cubes_remaining_after_removal_l2377_237746


namespace NUMINAMATH_CALUDE_weakly_increasing_h_implies_b_eq_one_l2377_237741

/-- A function is weakly increasing in an interval if it's increasing and its ratio to x is decreasing in that interval --/
def WeaklyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a < x ∧ x < y ∧ y ≤ b → f x ≤ f y) ∧
  (∀ x y, a < x ∧ x < y ∧ y ≤ b → f x / x ≥ f y / y)

/-- The function h(x) = x^2 - (b-1)x + b --/
def h (b : ℝ) (x : ℝ) : ℝ := x^2 - (b-1)*x + b

theorem weakly_increasing_h_implies_b_eq_one :
  ∀ b : ℝ, WeaklyIncreasing (h b) 0 1 → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_weakly_increasing_h_implies_b_eq_one_l2377_237741


namespace NUMINAMATH_CALUDE_square_dissection_l2377_237789

/-- Given two squares with side lengths a and b respectively, prove that:
    1. The square with side a can be dissected into 3 identical squares.
    2. The square with side b can be dissected into 7 identical squares. -/
theorem square_dissection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (sa sb : ℝ),
    sa = a / Real.sqrt 3 ∧
    sb = b / Real.sqrt 7 ∧
    3 * sa ^ 2 = a ^ 2 ∧
    7 * sb ^ 2 = b ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_square_dissection_l2377_237789


namespace NUMINAMATH_CALUDE_smoothies_from_fifteen_bananas_l2377_237775

/-- The number of smoothies Caroline can make from a given number of bananas. -/
def smoothies_from_bananas (bananas : ℕ) : ℕ :=
  (9 * bananas) / 3

/-- Theorem stating that Caroline can make 45 smoothies from 15 bananas. -/
theorem smoothies_from_fifteen_bananas :
  smoothies_from_bananas 15 = 45 := by
  sorry

#eval smoothies_from_bananas 15

end NUMINAMATH_CALUDE_smoothies_from_fifteen_bananas_l2377_237775


namespace NUMINAMATH_CALUDE_discount_sales_increase_l2377_237782

/-- Calculates the percent increase in gross income given a discount and increase in sales volume -/
theorem discount_sales_increase (discount : ℝ) (sales_increase : ℝ) : 
  discount = 0.1 → sales_increase = 0.3 → 
  (1 - discount) * (1 + sales_increase) - 1 = 0.17 := by
  sorry

#check discount_sales_increase

end NUMINAMATH_CALUDE_discount_sales_increase_l2377_237782


namespace NUMINAMATH_CALUDE_half_MN_coord_l2377_237732

def OM : Fin 2 → ℝ := ![(-2), 3]
def ON : Fin 2 → ℝ := ![(-1), (-5)]

theorem half_MN_coord : 
  (1/2 : ℝ) • (ON - OM) = ![(1/2), (-4)] := by sorry

end NUMINAMATH_CALUDE_half_MN_coord_l2377_237732


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l2377_237786

theorem quadratic_root_condition (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (3 * x₁^2 - 4*(3*a-2)*x₁ + a^2 + 2*a = 0) ∧ 
    (3 * x₂^2 - 4*(3*a-2)*x₂ + a^2 + 2*a = 0) ∧ 
    (x₁ < a ∧ a < x₂)) 
  ↔ 
  (a < 0 ∨ a > 5/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l2377_237786


namespace NUMINAMATH_CALUDE_smallest_integer_in_ratio_l2377_237755

theorem smallest_integer_in_ratio (a b c : ℕ+) : 
  (a : ℝ) / b = 2 / 3 → 
  (a : ℝ) / c = 2 / 5 → 
  (b : ℝ) / c = 3 / 5 → 
  a + b + c = 90 → 
  a = 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_in_ratio_l2377_237755


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l2377_237725

theorem quadratic_equation_condition (a : ℝ) :
  (∀ x, (a - 2) * x^2 + a * x - 3 = 0 → (a - 2) ≠ 0) ↔ a ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l2377_237725


namespace NUMINAMATH_CALUDE_sin_equality_iff_side_equality_l2377_237766

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (angle_sum : A + B + C = π)

-- State the theorem
theorem sin_equality_iff_side_equality (t : Triangle) : 
  Real.sin t.A = Real.sin t.B ↔ t.a = t.b :=
sorry

end NUMINAMATH_CALUDE_sin_equality_iff_side_equality_l2377_237766


namespace NUMINAMATH_CALUDE_derivative_of_y_l2377_237785

open Real

noncomputable def y (x : ℝ) : ℝ := cos (2*x - 1) + 1 / (x^2)

theorem derivative_of_y :
  deriv y = λ x => -2 * sin (2*x - 1) - 2 / (x^3) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l2377_237785


namespace NUMINAMATH_CALUDE_square_area_74_l2377_237769

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the square
def Square (p q : Point) :=
  {s : Set Point | ∃ (a b : ℝ), s = {(x, y) | min p.1 q.1 ≤ x ∧ x ≤ max p.1 q.1 ∧ min p.2 q.2 ≤ y ∧ y ≤ max p.2 q.2}}

-- Calculate the area of the square
def area (p q : Point) : ℝ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem square_area_74 :
  let p : Point := (-2, -1)
  let q : Point := (3, 6)
  area p q = 74 := by
sorry

end NUMINAMATH_CALUDE_square_area_74_l2377_237769


namespace NUMINAMATH_CALUDE_purse_value_is_107_percent_of_dollar_l2377_237773

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes quarters : ℕ) : ℕ :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  dimes * coin_value "dime" +
  quarters * coin_value "quarter"

theorem purse_value_is_107_percent_of_dollar (pennies nickels dimes quarters : ℕ) 
  (h_pennies : pennies = 2)
  (h_nickels : nickels = 3)
  (h_dimes : dimes = 4)
  (h_quarters : quarters = 2) :
  (total_value pennies nickels dimes quarters : ℚ) / 100 = 107 / 100 := by
  sorry

end NUMINAMATH_CALUDE_purse_value_is_107_percent_of_dollar_l2377_237773


namespace NUMINAMATH_CALUDE_lizzies_group_difference_l2377_237796

theorem lizzies_group_difference (total : ℕ) (lizzies_group : ℕ) : 
  total = 91 → lizzies_group = 54 → lizzies_group > (total - lizzies_group) → 
  lizzies_group - (total - lizzies_group) = 17 := by
sorry

end NUMINAMATH_CALUDE_lizzies_group_difference_l2377_237796


namespace NUMINAMATH_CALUDE_power_three_315_mod_11_l2377_237726

theorem power_three_315_mod_11 : 3^315 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_315_mod_11_l2377_237726


namespace NUMINAMATH_CALUDE_geometric_series_solution_l2377_237724

theorem geometric_series_solution (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (6 * n - 2) / k^n = 3) : k = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_solution_l2377_237724


namespace NUMINAMATH_CALUDE_largest_four_digit_prime_product_l2377_237759

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_four_digit_prime_product :
  ∀ (n x y z : ℕ),
    n = x * y * z * (10 * x + y) →
    x < 20 ∧ y < 20 ∧ z < 20 →
    is_prime x ∧ is_prime y ∧ is_prime z →
    is_prime (10 * x + y) →
    x ≠ y ∧ x ≠ z ∧ y ≠ z →
    n ≤ 25058 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_prime_product_l2377_237759


namespace NUMINAMATH_CALUDE_decision_block_two_exits_l2377_237723

-- Define the types of program blocks
inductive ProgramBlock
  | Termination
  | InputOutput
  | Processing
  | Decision

-- Define a function to determine if a block has two exit directions
def hasTwoExitDirections (block : ProgramBlock) : Prop :=
  match block with
  | ProgramBlock.Decision => true
  | _ => false

-- Theorem statement
theorem decision_block_two_exits :
  ∀ (block : ProgramBlock),
    hasTwoExitDirections block ↔ block = ProgramBlock.Decision :=
by sorry

end NUMINAMATH_CALUDE_decision_block_two_exits_l2377_237723


namespace NUMINAMATH_CALUDE_probability_red_ball_3_2_l2377_237714

/-- Represents the probability of drawing a red ball from a box containing red and yellow balls. -/
def probability_red_ball (red_balls yellow_balls : ℕ) : ℚ :=
  red_balls / (red_balls + yellow_balls)

/-- Theorem stating that the probability of drawing a red ball from a box with 3 red balls and 2 yellow balls is 3/5. -/
theorem probability_red_ball_3_2 :
  probability_red_ball 3 2 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_3_2_l2377_237714


namespace NUMINAMATH_CALUDE_sportswear_processing_equation_l2377_237739

/-- Represents the clothing processing factory problem --/
theorem sportswear_processing_equation 
  (total_sportswear : ℕ) 
  (processed_before_tech : ℕ) 
  (efficiency_increase : ℚ) 
  (total_time : ℚ) 
  (x : ℚ) 
  (h1 : total_sportswear = 400)
  (h2 : processed_before_tech = 160)
  (h3 : efficiency_increase = 1/5)
  (h4 : total_time = 18)
  (h5 : x > 0) :
  (processed_before_tech / x) + ((total_sportswear - processed_before_tech) / ((1 + efficiency_increase) * x)) = total_time :=
sorry

end NUMINAMATH_CALUDE_sportswear_processing_equation_l2377_237739


namespace NUMINAMATH_CALUDE_sean_julie_sum_ratio_l2377_237791

/-- The sum of even integers from 2 to 600, inclusive -/
def sean_sum : ℕ := 2 * (300 * 301) / 2

/-- The sum of integers from 1 to 300, inclusive -/
def julie_sum : ℕ := (300 * 301) / 2

/-- Theorem stating that Sean's sum divided by Julie's sum equals 2 -/
theorem sean_julie_sum_ratio :
  (sean_sum : ℚ) / (julie_sum : ℚ) = 2 := by sorry

end NUMINAMATH_CALUDE_sean_julie_sum_ratio_l2377_237791


namespace NUMINAMATH_CALUDE_sequence_non_negative_l2377_237774

/-- Sequence a_n defined recursively -/
def a : ℕ → ℝ → ℝ
  | 0, a₀ => a₀
  | n + 1, a₀ => 2 * a n a₀ - n^2

/-- Theorem stating the condition for non-negativity of the sequence -/
theorem sequence_non_negative (a₀ : ℝ) :
  (∀ n : ℕ, a n a₀ ≥ 0) ↔ a₀ ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_non_negative_l2377_237774


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l2377_237707

def population_size : ℕ := 30
def sample_size : ℕ := 6

def systematic_sampling_interval (pop_size sample_size : ℕ) : ℕ :=
  pop_size / sample_size

def generate_sample (start interval : ℕ) (size : ℕ) : List ℕ :=
  List.range size |>.map (λ i => start + i * interval)

theorem correct_systematic_sample :
  let interval := systematic_sampling_interval population_size sample_size
  let sample := generate_sample 2 interval sample_size
  (interval = 5) ∧ (sample = [2, 7, 12, 17, 22, 27]) := by
  sorry

#eval systematic_sampling_interval population_size sample_size
#eval generate_sample 2 (systematic_sampling_interval population_size sample_size) sample_size

end NUMINAMATH_CALUDE_correct_systematic_sample_l2377_237707


namespace NUMINAMATH_CALUDE_probability_seven_heads_ten_coins_prove_probability_seven_heads_ten_coins_l2377_237733

/-- The probability of getting exactly 7 heads when flipping 10 fair coins -/
theorem probability_seven_heads_ten_coins : ℚ :=
  15 / 128

/-- Proof that the probability of getting exactly 7 heads when flipping 10 fair coins is 15/128 -/
theorem prove_probability_seven_heads_ten_coins :
  probability_seven_heads_ten_coins = 15 / 128 := by
  sorry

end NUMINAMATH_CALUDE_probability_seven_heads_ten_coins_prove_probability_seven_heads_ten_coins_l2377_237733


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2377_237717

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x^2 + (2 - a) * x - 2 < 0}
  (a = 0 → S = {x : ℝ | x < 1}) ∧
  (-2 < a ∧ a < 0 → S = {x : ℝ | x < 1 ∨ x > -2/a}) ∧
  (a = -2 → S = {x : ℝ | x ≠ 1}) ∧
  (a < -2 → S = {x : ℝ | x < -2/a ∨ x > 1}) ∧
  (a > 0 → S = {x : ℝ | -2/a < x ∧ x < 1}) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2377_237717


namespace NUMINAMATH_CALUDE_multiplication_problem_l2377_237736

-- Define a custom type for single digits
def Digit := { n : Nat // n < 10 }

-- Define a function to convert a two-digit number to a natural number
def twoDigitToNat (d1 d2 : Digit) : Nat := 10 * d1.val + d2.val

-- Define a function to convert a three-digit number to a natural number
def threeDigitToNat (d1 d2 d3 : Digit) : Nat := 100 * d1.val + 10 * d2.val + d3.val

-- Define a function to convert a four-digit number to a natural number
def fourDigitToNat (d1 d2 d3 d4 : Digit) : Nat := 1000 * d1.val + 100 * d2.val + 10 * d3.val + d4.val

theorem multiplication_problem (A B C E F : Digit) :
  A ≠ B → A ≠ C → A ≠ E → A ≠ F →
  B ≠ C → B ≠ E → B ≠ F →
  C ≠ E → C ≠ F →
  E ≠ F →
  Nat.Prime (twoDigitToNat E F) →
  threeDigitToNat A B C * twoDigitToNat E F = fourDigitToNat E F E F →
  A.val + B.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l2377_237736


namespace NUMINAMATH_CALUDE_lauren_change_calculation_l2377_237744

/-- Calculates the change Lauren receives after grocery shopping --/
theorem lauren_change_calculation : 
  let hamburger_meat_price : ℝ := 3.50
  let hamburger_meat_weight : ℝ := 2
  let buns_price : ℝ := 1.50
  let lettuce_price : ℝ := 1.00
  let tomato_price_per_pound : ℝ := 2.00
  let tomato_weight : ℝ := 1.5
  let pickles_price : ℝ := 2.50
  let coupon_value : ℝ := 1.00
  let paid_amount : ℝ := 20.00

  let total_cost : ℝ := 
    hamburger_meat_price * hamburger_meat_weight +
    buns_price + 
    lettuce_price + 
    tomato_price_per_pound * tomato_weight + 
    pickles_price - 
    coupon_value

  let change : ℝ := paid_amount - total_cost

  change = 6.00 := by sorry

end NUMINAMATH_CALUDE_lauren_change_calculation_l2377_237744


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l2377_237778

theorem remainder_of_large_number (N : ℕ) (h : N = 109876543210) :
  N % 180 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l2377_237778


namespace NUMINAMATH_CALUDE_square_sum_value_l2377_237715

theorem square_sum_value (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 15) : x^2 + y^2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l2377_237715


namespace NUMINAMATH_CALUDE_line_point_k_l2377_237708

/-- A line is defined by three points it passes through -/
structure Line where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- Check if a point lies on a given line -/
def lies_on_line (l : Line) (p : ℝ × ℝ) : Prop :=
  let (x1, y1) := l.p1
  let (x2, y2) := l.p2
  let (x3, y3) := l.p3
  let (x, y) := p
  (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1) ∧
  (y - y2) * (x3 - x2) = (y3 - y2) * (x - x2)

/-- The main theorem -/
theorem line_point_k (l : Line) (k : ℝ) :
  l.p1 = (-1, 1) →
  l.p2 = (2, 5) →
  l.p3 = (5, 9) →
  lies_on_line l (50, k) →
  k = 69 := by
  sorry

end NUMINAMATH_CALUDE_line_point_k_l2377_237708


namespace NUMINAMATH_CALUDE_cistern_fill_time_l2377_237798

-- Define the fill rates of pipes
def fill_rate_p : ℚ := 1 / 10
def fill_rate_q : ℚ := 1 / 15
def drain_rate_r : ℚ := 1 / 30

-- Define the time pipes p and q are open together
def initial_time : ℚ := 2

-- Define the function to calculate the remaining time to fill the cistern
def remaining_fill_time (fill_rate_p fill_rate_q drain_rate_r initial_time : ℚ) : ℚ :=
  let initial_fill := (fill_rate_p + fill_rate_q) * initial_time
  let remaining_volume := 1 - initial_fill
  let net_fill_rate := fill_rate_q - drain_rate_r
  remaining_volume / net_fill_rate

-- Theorem statement
theorem cistern_fill_time :
  remaining_fill_time fill_rate_p fill_rate_q drain_rate_r initial_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l2377_237798


namespace NUMINAMATH_CALUDE_cosine_sum_product_simplification_l2377_237712

theorem cosine_sum_product_simplification (α β : ℝ) :
  Real.cos (α + β) * Real.cos β + Real.sin (α + β) * Real.sin β = Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_product_simplification_l2377_237712


namespace NUMINAMATH_CALUDE_village_population_equality_l2377_237747

/-- The rate of population increase for Village Y -/
def rate_Y : ℕ := sorry

/-- The initial population of Village X -/
def pop_X : ℕ := 76000

/-- The initial population of Village Y -/
def pop_Y : ℕ := 42000

/-- The rate of population decrease for Village X -/
def rate_X : ℕ := 1200

/-- The number of years after which the populations will be equal -/
def years : ℕ := 17

theorem village_population_equality :
  pop_X - (rate_X * years) = pop_Y + (rate_Y * years) ∧ rate_Y = 800 := by sorry

end NUMINAMATH_CALUDE_village_population_equality_l2377_237747


namespace NUMINAMATH_CALUDE_min_angle_line_equation_l2377_237761

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The angle between two points and a center point -/
def angle (center : ℝ × ℝ) (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) : ℝ := sorry

/-- Check if a line intersects a circle at two points -/
def intersectsCircle (l : Line) (c : Circle) : Prop := sorry

/-- Check if a point lies on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- The main theorem -/
theorem min_angle_line_equation 
  (c : Circle)
  (m : ℝ × ℝ)
  (l : Line) :
  c.center = (3, 4) →
  c.radius = 5 →
  m = (1, 2) →
  pointOnLine m l →
  intersectsCircle l c →
  (∀ l' : Line, intersectsCircle l' c → angle c.center (1, 2) (3, 4) ≤ angle c.center (1, 2) (3, 4)) →
  l.slope = -1 ∧ l.yIntercept = 3 :=
sorry

end NUMINAMATH_CALUDE_min_angle_line_equation_l2377_237761


namespace NUMINAMATH_CALUDE_root_shift_polynomial_l2377_237748

theorem root_shift_polynomial (a b c : ℂ) : 
  (a^3 - 4*a^2 + 6*a - 8 = 0) ∧ 
  (b^3 - 4*b^2 + 6*b - 8 = 0) ∧ 
  (c^3 - 4*c^2 + 6*c - 8 = 0) →
  ((a + 3)^3 - 13*(a + 3)^2 + 57*(a + 3) - 89 = 0) ∧
  ((b + 3)^3 - 13*(b + 3)^2 + 57*(b + 3) - 89 = 0) ∧
  ((c + 3)^3 - 13*(c + 3)^2 + 57*(c + 3) - 89 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_polynomial_l2377_237748


namespace NUMINAMATH_CALUDE_problem_solution_l2377_237772

theorem problem_solution : ∃ x : ℚ, (x + x / 4 = 80 - 80 / 4) ∧ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2377_237772


namespace NUMINAMATH_CALUDE_water_jars_count_l2377_237758

/-- Given 35 gallons of water stored in equal numbers of quart, half-gallon, and one-gallon jars,
    the total number of water-filled jars is 60. -/
theorem water_jars_count (total_volume : ℚ) (jar_sizes : Fin 3 → ℚ) :
  total_volume = 35 →
  jar_sizes 0 = 1/4 →
  jar_sizes 1 = 1/2 →
  jar_sizes 2 = 1 →
  (∃ (x : ℕ), total_volume = x * (jar_sizes 0 + jar_sizes 1 + jar_sizes 2)) →
  (∃ (x : ℕ), 3 * x = 60) :=
by sorry

end NUMINAMATH_CALUDE_water_jars_count_l2377_237758


namespace NUMINAMATH_CALUDE_sum_b_c_is_48_l2377_237787

/-- An arithmetic sequence with six terms -/
structure ArithmeticSequence :=
  (a₁ : ℝ)
  (a₂ : ℝ)
  (a₃ : ℝ)
  (a₄ : ℝ)
  (a₅ : ℝ)
  (a₆ : ℝ)
  (is_arithmetic : ∃ d : ℝ, a₂ - a₁ = d ∧ a₃ - a₂ = d ∧ a₄ - a₃ = d ∧ a₅ - a₄ = d ∧ a₆ - a₅ = d)

/-- The sum of the third and fifth terms in the specific arithmetic sequence -/
def sum_b_c (seq : ArithmeticSequence) : ℝ := seq.a₃ + seq.a₅

/-- Theorem stating that for the given arithmetic sequence, the sum of b and c is 48 -/
theorem sum_b_c_is_48 (seq : ArithmeticSequence) 
  (h₁ : seq.a₁ = 3)
  (h₂ : seq.a₂ = 10)
  (h₃ : seq.a₄ = 24)
  (h₄ : seq.a₆ = 38) :
  sum_b_c seq = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_b_c_is_48_l2377_237787


namespace NUMINAMATH_CALUDE_inequalities_hold_l2377_237784

theorem inequalities_hold (a b c x y z : ℝ) 
  (hx : x ≤ a) (hy : y ≤ b) (hz : z ≤ c) : 
  (x * y + y * z + z * x ≤ a * b + b * c + c * a) ∧ 
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧ 
  (x * y * z ≤ a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l2377_237784


namespace NUMINAMATH_CALUDE_one_root_implies_a_range_l2377_237797

-- Define the function f(x) = 2x³ - 3x² + a
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

-- Define the property of having only one root in [-2, 2]
def has_one_root_in_interval (a : ℝ) : Prop :=
  ∃! x, x ∈ Set.Icc (-2) 2 ∧ f a x = 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a ∈ Set.Ioo (-4) 0 ∪ Set.Ioo 1 28

-- State the theorem
theorem one_root_implies_a_range :
  ∀ a : ℝ, has_one_root_in_interval a → a_range a := by
  sorry

end NUMINAMATH_CALUDE_one_root_implies_a_range_l2377_237797
