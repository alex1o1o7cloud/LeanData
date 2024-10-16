import Mathlib

namespace NUMINAMATH_CALUDE_compare_negative_fractions_l99_9948

theorem compare_negative_fractions : -2/3 < -3/5 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l99_9948


namespace NUMINAMATH_CALUDE_probability_three_specific_heads_out_of_five_probability_three_specific_heads_out_of_five_proof_l99_9906

/-- The probability of getting heads on exactly three specific coins out of five coins -/
theorem probability_three_specific_heads_out_of_five : ℝ :=
  let n_coins : ℕ := 5
  let n_specific_coins : ℕ := 3
  let p_head : ℝ := 1 / 2
  1 / 8

/-- Proof of the theorem -/
theorem probability_three_specific_heads_out_of_five_proof :
  probability_three_specific_heads_out_of_five = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_specific_heads_out_of_five_probability_three_specific_heads_out_of_five_proof_l99_9906


namespace NUMINAMATH_CALUDE_expression_simplification_l99_9997

theorem expression_simplification (a : ℝ) :
  (a - (2*a - 1) / a) / ((1 - a^2) / (a^2 + a)) = a + 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l99_9997


namespace NUMINAMATH_CALUDE_quadratic_system_sum_l99_9953

theorem quadratic_system_sum (x y r₁ s₁ r₂ s₂ : ℝ) : 
  (9 * x^2 - 27 * x - 54 = 0) →
  (4 * y^2 + 28 * y + 49 = 0) →
  ((x - r₁)^2 = s₁) →
  ((y - r₂)^2 = s₂) →
  (r₁ + s₁ + r₂ + s₂ = -11/4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_system_sum_l99_9953


namespace NUMINAMATH_CALUDE_derivative_of_f_l99_9943

-- Define the function f
def f (x : ℝ) : ℝ := 2016 * x^2

-- State the theorem
theorem derivative_of_f (x : ℝ) :
  deriv f x = 4032 * x := by sorry

-- Note: The 'deriv' function in Lean represents the derivative.

end NUMINAMATH_CALUDE_derivative_of_f_l99_9943


namespace NUMINAMATH_CALUDE_family_weight_calculation_l99_9923

/-- The total weight of a family consisting of a grandmother, her daughter, and her grandchild. -/
def total_weight (mother_weight daughter_weight grandchild_weight : ℝ) : ℝ :=
  mother_weight + daughter_weight + grandchild_weight

/-- Theorem stating the total weight of the family under given conditions. -/
theorem family_weight_calculation :
  ∀ (mother_weight daughter_weight grandchild_weight : ℝ),
    daughter_weight + grandchild_weight = 60 →
    grandchild_weight = (1 / 5) * mother_weight →
    daughter_weight = 48 →
    total_weight mother_weight daughter_weight grandchild_weight = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_family_weight_calculation_l99_9923


namespace NUMINAMATH_CALUDE_hilt_detergent_usage_l99_9965

/-- The amount of detergent Mrs. Hilt uses per pound of clothes -/
def detergent_per_pound : ℝ := 2

/-- The number of pounds of clothes to be washed -/
def pounds_of_clothes : ℝ := 9

/-- Theorem: Mrs. Hilt will use 18 ounces of detergent to wash 9 pounds of clothes -/
theorem hilt_detergent_usage : detergent_per_pound * pounds_of_clothes = 18 := by
  sorry

end NUMINAMATH_CALUDE_hilt_detergent_usage_l99_9965


namespace NUMINAMATH_CALUDE_problem_solution_l99_9921

theorem problem_solution (x : ℝ) (h1 : x > 0) (h2 : x * ↑(⌊x⌋) = 72) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l99_9921


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l99_9933

/-- Given a line segment with midpoint (10, -14) and one endpoint at (12, -6),
    the sum of the coordinates of the other endpoint is -14. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (x + 12) / 2 = 10 →  -- Midpoint x-coordinate condition
  (y - 6) / 2 = -14 →  -- Midpoint y-coordinate condition
  x + y = -14 :=
by sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l99_9933


namespace NUMINAMATH_CALUDE_star_property_l99_9927

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element
  | five : Element

-- Define the * operation
def star : Element → Element → Element
  | Element.one, x => x
  | Element.two, Element.one => Element.two
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.five
  | Element.two, Element.five => Element.one
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.five
  | Element.three, Element.four => Element.one
  | Element.three, Element.five => Element.two
  | Element.four, Element.one => Element.four
  | Element.four, Element.two => Element.five
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.two
  | Element.four, Element.five => Element.three
  | Element.five, Element.one => Element.five
  | Element.five, Element.two => Element.one
  | Element.five, Element.three => Element.two
  | Element.five, Element.four => Element.three
  | Element.five, Element.five => Element.four

theorem star_property : 
  star (star Element.three Element.five) (star Element.two Element.four) = Element.one := by
  sorry

end NUMINAMATH_CALUDE_star_property_l99_9927


namespace NUMINAMATH_CALUDE_jessica_purchases_total_cost_l99_9950

theorem jessica_purchases_total_cost :
  let cat_toy_cost : ℚ := 10.22
  let cage_cost : ℚ := 11.73
  let total_cost : ℚ := cat_toy_cost + cage_cost
  total_cost = 21.95 := by sorry

end NUMINAMATH_CALUDE_jessica_purchases_total_cost_l99_9950


namespace NUMINAMATH_CALUDE_compound_interest_proof_l99_9911

/-- Calculate compound interest and prove the total interest earned --/
theorem compound_interest_proof (P : ℝ) (r : ℝ) (n : ℕ) (h1 : P = 1000) (h2 : r = 0.1) (h3 : n = 3) :
  (P * (1 + r)^n - P) = 331 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_proof_l99_9911


namespace NUMINAMATH_CALUDE_stamp_collection_gcd_l99_9975

theorem stamp_collection_gcd : Nat.gcd (Nat.gcd 945 1260) 630 = 105 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_gcd_l99_9975


namespace NUMINAMATH_CALUDE_quadratic_roots_l99_9986

theorem quadratic_roots : ∃ x₁ x₂ : ℝ, 
  (x₁ = 2 ∧ x₂ = -1) ∧ 
  (∀ x : ℝ, x * (x - 2) = 2 - x ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l99_9986


namespace NUMINAMATH_CALUDE_max_cookies_ella_l99_9958

/-- Represents the recipe for cookies -/
structure Recipe where
  chocolate : Rat
  sugar : Rat
  eggs : Nat
  flour : Rat
  cookies : Nat

/-- Represents available ingredients -/
structure Ingredients where
  chocolate : Rat
  sugar : Rat
  eggs : Nat
  flour : Rat

/-- Calculates the maximum number of cookies that can be made -/
def maxCookies (recipe : Recipe) (ingredients : Ingredients) : Nat :=
  min
    (Nat.floor ((ingredients.chocolate / recipe.chocolate) * recipe.cookies))
    (min
      (Nat.floor ((ingredients.sugar / recipe.sugar) * recipe.cookies))
      (min
        ((ingredients.eggs / recipe.eggs) * recipe.cookies)
        (Nat.floor ((ingredients.flour / recipe.flour) * recipe.cookies))))

theorem max_cookies_ella :
  let recipe : Recipe := {
    chocolate := 1,
    sugar := 1/2,
    eggs := 1,
    flour := 1,
    cookies := 4
  }
  let ingredients : Ingredients := {
    chocolate := 4,
    sugar := 3,
    eggs := 6,
    flour := 10
  }
  maxCookies recipe ingredients = 16 := by
  sorry

#eval maxCookies
  { chocolate := 1, sugar := 1/2, eggs := 1, flour := 1, cookies := 4 }
  { chocolate := 4, sugar := 3, eggs := 6, flour := 10 }

end NUMINAMATH_CALUDE_max_cookies_ella_l99_9958


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l99_9904

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 30| + |x - 20| = |3*x - 90| :=
by
  -- The unique solution is x = 40
  use 40
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l99_9904


namespace NUMINAMATH_CALUDE_tank_capacity_l99_9913

theorem tank_capacity : ∀ (initial_fraction final_fraction added_volume capacity : ℚ),
  initial_fraction = 1 / 4 →
  final_fraction = 3 / 4 →
  added_volume = 160 →
  (final_fraction - initial_fraction) * capacity = added_volume →
  capacity = 320 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l99_9913


namespace NUMINAMATH_CALUDE_multiply_fractions_l99_9994

theorem multiply_fractions : 12 * (1 / 15) * 30 = 24 := by
  sorry

end NUMINAMATH_CALUDE_multiply_fractions_l99_9994


namespace NUMINAMATH_CALUDE_cuboid_length_problem_l99_9955

/-- The surface area of a cuboid given its length, width, and height -/
def cuboidSurfaceArea (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: The length of a cuboid with surface area 700 m², breadth 14 m, and height 7 m is 12 m -/
theorem cuboid_length_problem :
  ∃ (l : ℝ), cuboidSurfaceArea l 14 7 = 700 ∧ l = 12 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_length_problem_l99_9955


namespace NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l99_9977

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π
  h_sum_angles : A + B + C = π
  h_area : S = (1/2) * b * c * Real.sin A

-- Theorem statement
theorem angle_measure_in_special_triangle (t : Triangle) 
  (h : (t.b + t.c)^2 - t.a^2 = 4 * Real.sqrt 3 * t.S) : 
  t.A = π/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l99_9977


namespace NUMINAMATH_CALUDE_circle_equation_proof_l99_9919

/-- Given two points P and Q in a 2D plane, we define a circle with PQ as its diameter. -/
def circle_with_diameter (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {point | (point.1 - (P.1 + Q.1) / 2)^2 + (point.2 - (P.2 + Q.2) / 2)^2 = ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) / 4}

/-- The theorem states that for P(4,0) and Q(0,2), the equation of the circle with PQ as diameter is (x-2)^2 + (y-1)^2 = 5. -/
theorem circle_equation_proof :
  circle_with_diameter (4, 0) (0, 2) = {point : ℝ × ℝ | (point.1 - 2)^2 + (point.2 - 1)^2 = 5} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l99_9919


namespace NUMINAMATH_CALUDE_round_trip_fuel_efficiency_l99_9917

/-- Calculates the average fuel efficiency for a round trip given the conditions. -/
theorem round_trip_fuel_efficiency 
  (distance : ℝ) 
  (efficiency1 : ℝ) 
  (efficiency2 : ℝ) 
  (h1 : distance = 120) 
  (h2 : efficiency1 = 30) 
  (h3 : efficiency2 = 20) : 
  (2 * distance) / (distance / efficiency1 + distance / efficiency2) = 24 :=
by
  sorry

#check round_trip_fuel_efficiency

end NUMINAMATH_CALUDE_round_trip_fuel_efficiency_l99_9917


namespace NUMINAMATH_CALUDE_a_gt_b_neither_necessary_nor_sufficient_for_a_sq_gt_b_sq_l99_9944

theorem a_gt_b_neither_necessary_nor_sufficient_for_a_sq_gt_b_sq :
  ¬(∀ a b : ℝ, a > b → a^2 > b^2) ∧ 
  ¬(∀ a b : ℝ, a^2 > b^2 → a > b) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_neither_necessary_nor_sufficient_for_a_sq_gt_b_sq_l99_9944


namespace NUMINAMATH_CALUDE_sphere_radii_formula_l99_9966

/-- Given three mutually tangent spheres touched by a plane at points A, B, and C,
    where the sides of triangle ABC are a, b, and c, prove that the radii of the
    spheres (x, y, z) are given by the formulas stated. -/
theorem sphere_radii_formula (a b c x y z : ℝ) 
  (h1 : a = 2 * Real.sqrt (x * y))
  (h2 : b = 2 * Real.sqrt (y * z))
  (h3 : c = 2 * Real.sqrt (x * z))
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  x = a * c / (2 * b) ∧ 
  y = a * b / (2 * c) ∧ 
  z = b * c / (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radii_formula_l99_9966


namespace NUMINAMATH_CALUDE_cake_recipe_difference_l99_9993

theorem cake_recipe_difference (flour_required sugar_required sugar_added : ℕ) :
  flour_required = 9 →
  sugar_required = 6 →
  sugar_added = 4 →
  flour_required - (sugar_required - sugar_added) = 7 := by
sorry

end NUMINAMATH_CALUDE_cake_recipe_difference_l99_9993


namespace NUMINAMATH_CALUDE_sum_inequality_l99_9967

theorem sum_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 1) : 
  (a / (1 - a^2)) + (b / (1 - b^2)) + (c / (1 - c^2)) ≥ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l99_9967


namespace NUMINAMATH_CALUDE_equation_solution_l99_9946

theorem equation_solution :
  ∃ x : ℝ, x = 1 ∧ 2021 * x = 2022 * (x^2021)^(1/2021) - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l99_9946


namespace NUMINAMATH_CALUDE_jims_remaining_distance_l99_9985

/-- Calculates the remaining distance in a journey. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Proves that for Jim's journey, the remaining distance is 1,068 miles. -/
theorem jims_remaining_distance :
  remaining_distance 2450 1382 = 1068 := by
  sorry

end NUMINAMATH_CALUDE_jims_remaining_distance_l99_9985


namespace NUMINAMATH_CALUDE_remainder_sum_divided_by_11_l99_9945

theorem remainder_sum_divided_by_11 : 
  (2 * (8735 + 8736 + 8737 + 8738 + 8739)) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_divided_by_11_l99_9945


namespace NUMINAMATH_CALUDE_N_is_perfect_square_l99_9925

/-- Constructs the number N with n 1s, n+1 2s, ending with 25 -/
def constructN (n : ℕ) : ℕ :=
  (10^(2*n+2) + 10^(n+1)) / 9 + 25

/-- Theorem: For any positive n, the constructed N is a perfect square -/
theorem N_is_perfect_square (n : ℕ+) : ∃ m : ℕ, (constructN n) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_N_is_perfect_square_l99_9925


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l99_9961

theorem necessary_not_sufficient : 
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (3 - x) > 0)) ∧
  (∀ x : ℝ, x * (3 - x) > 0 → |x - 1| < 2) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l99_9961


namespace NUMINAMATH_CALUDE_remainder_sum_l99_9998

theorem remainder_sum (D : ℕ) (h1 : D > 0) (h2 : 242 % D = 11) (h3 : 698 % D = 18) :
  940 % D = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l99_9998


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l99_9959

theorem quadratic_form_equivalence (x : ℝ) : x^2 + 4*x + 1 = (x + 2)^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l99_9959


namespace NUMINAMATH_CALUDE_diophantine_equation_only_trivial_solution_l99_9940

theorem diophantine_equation_only_trivial_solution (x y z : ℤ) :
  3 * x^2 + 7 * y^2 = z^4 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_only_trivial_solution_l99_9940


namespace NUMINAMATH_CALUDE_A_necessary_not_sufficient_l99_9988

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define propositions A and B
def proposition_A (x : ℝ) : Prop := log10 (x^2) = 0
def proposition_B (x : ℝ) : Prop := x = 1

-- Theorem stating A is necessary but not sufficient for B
theorem A_necessary_not_sufficient :
  (∀ x : ℝ, proposition_B x → proposition_A x) ∧
  (∃ x : ℝ, proposition_A x ∧ ¬proposition_B x) :=
sorry

end NUMINAMATH_CALUDE_A_necessary_not_sufficient_l99_9988


namespace NUMINAMATH_CALUDE_first_day_student_tickets_l99_9902

/-- The number of student tickets sold on the first day -/
def student_tickets_day1 : ℕ := 3

/-- The price of a student ticket -/
def student_ticket_price : ℕ := 9

/-- The price of a senior citizen ticket -/
def senior_ticket_price : ℕ := 13

theorem first_day_student_tickets :
  student_tickets_day1 = 3 ∧
  4 * senior_ticket_price + student_tickets_day1 * student_ticket_price = 79 ∧
  12 * senior_ticket_price + 10 * student_ticket_price = 246 ∧
  student_ticket_price = 9 := by
sorry

end NUMINAMATH_CALUDE_first_day_student_tickets_l99_9902


namespace NUMINAMATH_CALUDE_jensen_family_trip_l99_9980

/-- Calculates the miles driven on city streets given the total distance on highways,
    car efficiency on highways and city streets, and total gas used. -/
theorem jensen_family_trip (highway_miles : ℝ) (highway_efficiency : ℝ) 
  (city_efficiency : ℝ) (total_gas : ℝ) (city_miles : ℝ) : 
  highway_miles = 210 →
  highway_efficiency = 35 →
  city_efficiency = 18 →
  total_gas = 9 →
  city_miles = (total_gas - highway_miles / highway_efficiency) * city_efficiency →
  city_miles = 54 := by sorry

end NUMINAMATH_CALUDE_jensen_family_trip_l99_9980


namespace NUMINAMATH_CALUDE_original_number_proof_l99_9979

theorem original_number_proof : 
  ∃ x : ℝ, 3 * (2 * (3 * x) - 9) = 90 ∧ x = 6.5 := by sorry

end NUMINAMATH_CALUDE_original_number_proof_l99_9979


namespace NUMINAMATH_CALUDE_DR_length_zero_l99_9974

/-- Rectangle ABCD with inscribed circle ω -/
structure RectangleWithCircle where
  /-- Length of the rectangle -/
  length : ℝ
  /-- Height of the rectangle -/
  height : ℝ
  /-- Center of the inscribed circle -/
  center : ℝ × ℝ
  /-- Radius of the inscribed circle -/
  radius : ℝ
  /-- Point Q where the circle intersects AB -/
  Q : ℝ × ℝ
  /-- Point D at the bottom left corner -/
  D : ℝ × ℝ
  /-- Point R where DQ intersects the circle again -/
  R : ℝ × ℝ
  /-- The rectangle has length 2 and height 1 -/
  h_dimensions : length = 2 ∧ height = 1
  /-- The circle is inscribed in the rectangle -/
  h_inscribed : center = (0, 0) ∧ radius = height / 2
  /-- Q is on the top edge of the rectangle -/
  h_Q_on_top : Q.2 = height / 2
  /-- D is at the bottom left corner -/
  h_D_position : D = (0, -height / 2)
  /-- R is on the circle -/
  h_R_on_circle : (R.1 - center.1)^2 + (R.2 - center.2)^2 = radius^2
  /-- R is on line DQ -/
  h_R_on_DQ : R.1 = D.1 ∧ R.1 = Q.1

/-- The main theorem: DR has length 0 -/
theorem DR_length_zero (rect : RectangleWithCircle) : dist rect.D rect.R = 0 :=
  sorry


end NUMINAMATH_CALUDE_DR_length_zero_l99_9974


namespace NUMINAMATH_CALUDE_work_of_adiabatic_compression_l99_9930

/-- Work of adiabatic compression -/
theorem work_of_adiabatic_compression
  (V₀ V₁ p₀ k : ℝ) 
  (h₁ : V₀ > 0)
  (h₂ : V₁ > 0)
  (h₃ : p₀ > 0)
  (h₄ : k > 1)
  (h₅ : V₁ < V₀) :
  ∃ W : ℝ, W = (p₀ * V₀ / (k - 1)) * ((V₀ / V₁) ^ (k - 1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_work_of_adiabatic_compression_l99_9930


namespace NUMINAMATH_CALUDE_line_equation_proof_l99_9931

/-- Given a line defined by (3, -4) · ((x, y) - (2, 7)) = 0, prove that its slope-intercept form y = mx + b has m = 3/4 and b = 11/2 -/
theorem line_equation_proof (x y : ℝ) : 
  (3 * (x - 2) + (-4) * (y - 7) = 0) → 
  (∃ (m b : ℝ), y = m * x + b ∧ m = 3/4 ∧ b = 11/2) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l99_9931


namespace NUMINAMATH_CALUDE_max_product_is_18000_l99_9952

def numbers : List ℕ := [10, 15, 20, 30, 40, 60]

def is_valid_arrangement (arrangement : List ℕ) : Prop :=
  arrangement.length = 6 ∧ 
  arrangement.toFinset = numbers.toFinset ∧
  ∃ (product : ℕ), 
    (arrangement[0]! * arrangement[1]! * arrangement[2]! = product) ∧
    (arrangement[1]! * arrangement[3]! * arrangement[4]! = product) ∧
    (arrangement[2]! * arrangement[4]! * arrangement[5]! = product)

theorem max_product_is_18000 :
  ∀ (arrangement : List ℕ), is_valid_arrangement arrangement →
    ∃ (product : ℕ), 
      (arrangement[0]! * arrangement[1]! * arrangement[2]! = product) ∧
      (arrangement[1]! * arrangement[3]! * arrangement[4]! = product) ∧
      (arrangement[2]! * arrangement[4]! * arrangement[5]! = product) ∧
      product ≤ 18000 :=
by sorry

end NUMINAMATH_CALUDE_max_product_is_18000_l99_9952


namespace NUMINAMATH_CALUDE_rectangle_areas_sum_l99_9914

theorem rectangle_areas_sum : 
  let width : ℕ := 3
  let lengths : List ℕ := [1, 3, 5, 7, 9].map (λ x => x^2)
  let areas : List ℕ := lengths.map (λ l => width * l)
  areas.sum = 495 := by
sorry

end NUMINAMATH_CALUDE_rectangle_areas_sum_l99_9914


namespace NUMINAMATH_CALUDE_biology_marks_l99_9995

def marks_english : ℕ := 86
def marks_mathematics : ℕ := 85
def marks_physics : ℕ := 82
def marks_chemistry : ℕ := 87
def average_marks : ℕ := 85
def total_subjects : ℕ := 5

theorem biology_marks :
  ∃ (marks_biology : ℕ),
    marks_biology = average_marks * total_subjects - (marks_english + marks_mathematics + marks_physics + marks_chemistry) ∧
    marks_biology = 85 := by
  sorry

end NUMINAMATH_CALUDE_biology_marks_l99_9995


namespace NUMINAMATH_CALUDE_folded_paper_perimeter_ratio_l99_9992

/-- Represents the dimensions of a rectangular piece of paper --/
structure PaperDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular piece of paper --/
def perimeter (p : PaperDimensions) : ℝ :=
  2 * (p.length + p.width)

/-- Represents the paper after folding and cutting --/
structure FoldedPaper where
  original : PaperDimensions
  flap : PaperDimensions
  largest_rectangle : PaperDimensions

/-- The theorem to be proved --/
theorem folded_paper_perimeter_ratio 
  (paper : FoldedPaper) 
  (h1 : paper.original.length = 6 ∧ paper.original.width = 6)
  (h2 : paper.flap.length = 3 ∧ paper.flap.width = 3)
  (h3 : paper.largest_rectangle.length = 6 ∧ paper.largest_rectangle.width = 4.5) :
  perimeter paper.flap / perimeter paper.largest_rectangle = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_perimeter_ratio_l99_9992


namespace NUMINAMATH_CALUDE_certain_number_proof_l99_9960

theorem certain_number_proof (x y z N : ℤ) : 
  x < y → y < z →
  y - x > N →
  Even x →
  Odd y →
  Odd z →
  (∀ w, w - x ≥ 13 → w ≥ z) →
  (∃ u v, u < v ∧ v < z ∧ v - u > N ∧ Even u ∧ Odd v ∧ v - x < 13) →
  N ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l99_9960


namespace NUMINAMATH_CALUDE_notebook_cost_l99_9936

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (buyers notebooks_per_student cost : Nat),
  -- Total number of students
  total_students = 36 ∧
  -- Majority of students bought notebooks
  buyers > total_students / 2 ∧
  -- Each student bought more than one notebook
  notebooks_per_student > 1 ∧
  -- Cost in cents is higher than the number of notebooks bought
  cost > notebooks_per_student ∧
  -- Total cost equation
  buyers * notebooks_per_student * cost = total_cost ∧
  -- Total cost given
  total_cost = 2079 ∧
  -- The cost of each notebook is 11 cents
  cost = 11 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l99_9936


namespace NUMINAMATH_CALUDE_tangent_segment_area_l99_9963

theorem tangent_segment_area (r : ℝ) (l : ℝ) (h_r : r = 3) (h_l : l = 6) :
  let outer_radius := (r^2 + (l/2)^2).sqrt
  (π * outer_radius^2 - π * r^2) = 9 * π := by sorry

end NUMINAMATH_CALUDE_tangent_segment_area_l99_9963


namespace NUMINAMATH_CALUDE_min_value_expression_l99_9969

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 5) :
  ((x + 1) * (2*y + 1)) / Real.sqrt (x*y) ≥ 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l99_9969


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l99_9912

theorem arithmetic_series_sum (t : ℝ) : 
  let first_term := t^2 + 3
  let num_terms := 3*t + 2
  let common_difference := 1
  let last_term := first_term + (num_terms - 1) * common_difference
  (num_terms / 2) * (first_term + last_term) = (3*t + 2) * (t^2 + 1.5*t + 3.5) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l99_9912


namespace NUMINAMATH_CALUDE_solution_set_inequality_l99_9920

theorem solution_set_inequality (x : ℝ) : 
  (x * (2 - x) > 0) ↔ (0 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l99_9920


namespace NUMINAMATH_CALUDE_otto_sharpening_cost_l99_9935

/-- Represents the cost structure for knife sharpening --/
structure Sharpening :=
  (first_knife : ℝ)
  (next_three : ℝ)
  (five_to_ten : ℝ)
  (after_ten : ℝ)

/-- Represents the knife collection --/
structure KnifeCollection :=
  (total : ℕ)
  (chefs : ℕ)
  (paring : ℕ)

/-- Calculates the total cost of sharpening knives --/
def sharpeningCost (s : Sharpening) (k : KnifeCollection) : ℝ :=
  sorry

/-- Theorem stating the total cost of sharpening Otto's knives --/
theorem otto_sharpening_cost :
  let s : Sharpening := {
    first_knife := 6.00,
    next_three := 4.50,
    five_to_ten := 3.75,
    after_ten := 3.25
  }
  let k : KnifeCollection := {
    total := 15,
    chefs := 3,
    paring := 4
  }
  let chefs_discount := 0.15
  let paring_discount := 0.10
  sharpeningCost s k - (chefs_discount * (s.first_knife + 2 * s.next_three)) - 
    (paring_discount * (2 * s.next_three + 2 * s.five_to_ten)) = 54.35 :=
by sorry

end NUMINAMATH_CALUDE_otto_sharpening_cost_l99_9935


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l99_9937

-- Define the universal set U
def U : Set ℕ := {x | x ≤ 5}

-- Define sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {0, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l99_9937


namespace NUMINAMATH_CALUDE_cost_per_gb_over_limit_l99_9900

def data_limit : ℕ := 8
def total_usage : ℕ := 20
def extra_charge : ℕ := 120

theorem cost_per_gb_over_limit : 
  extra_charge / (total_usage - data_limit) = 10 := by sorry

end NUMINAMATH_CALUDE_cost_per_gb_over_limit_l99_9900


namespace NUMINAMATH_CALUDE_set_operations_and_intersection_l99_9915

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem statement
theorem set_operations_and_intersection (a : ℝ) :
  (A ∪ B = {x | 1 ≤ x ∧ x < 10}) ∧
  ((Aᶜ) ∩ B = {x | 7 ≤ x ∧ x < 10}) ∧
  ((A ∩ C a).Nonempty → a > 1) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_intersection_l99_9915


namespace NUMINAMATH_CALUDE_sum_of_squares_bounds_l99_9989

/-- A quadrilateral inscribed in a unit square -/
structure InscribedQuadrilateral where
  w : Real
  x : Real
  y : Real
  z : Real
  w_in_range : 0 ≤ w ∧ w ≤ 1
  x_in_range : 0 ≤ x ∧ x ≤ 1
  y_in_range : 0 ≤ y ∧ y ≤ 1
  z_in_range : 0 ≤ z ∧ z ≤ 1

/-- The sum of squares of the sides of an inscribed quadrilateral -/
def sumOfSquares (q : InscribedQuadrilateral) : Real :=
  (q.w^2 + q.x^2) + ((1-q.x)^2 + q.y^2) + ((1-q.y)^2 + q.z^2) + ((1-q.z)^2 + (1-q.w)^2)

/-- Theorem: The sum of squares of the sides of a quadrilateral inscribed in a unit square is between 2 and 4 -/
theorem sum_of_squares_bounds (q : InscribedQuadrilateral) : 
  2 ≤ sumOfSquares q ∧ sumOfSquares q ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_bounds_l99_9989


namespace NUMINAMATH_CALUDE_incorrect_factorization_l99_9957

theorem incorrect_factorization (x y : ℝ) : ¬(∀ x y : ℝ, x^2 + y^2 = (x + y)^2) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_factorization_l99_9957


namespace NUMINAMATH_CALUDE_simplify_radical_product_l99_9907

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  Real.sqrt (28 * x) * Real.sqrt (15 * x) * Real.sqrt (21 * x) = 42 * x * Real.sqrt (5 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l99_9907


namespace NUMINAMATH_CALUDE_double_negation_2023_l99_9982

theorem double_negation_2023 : -(-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_double_negation_2023_l99_9982


namespace NUMINAMATH_CALUDE_t_range_l99_9981

/-- The function f(x) = |xe^x| -/
noncomputable def f (x : ℝ) : ℝ := |x * Real.exp x|

/-- The function g(x) = [f(x)]^2 - tf(x) -/
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := (f x)^2 - t * (f x)

/-- The theorem stating the range of t -/
theorem t_range (t : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    g t x₁ = -2 ∧ g t x₂ = -2 ∧ g t x₃ = -2 ∧ g t x₄ = -2) →
  t > Real.exp (-1) + 2 * Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_t_range_l99_9981


namespace NUMINAMATH_CALUDE_simplify_expression_l99_9972

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (15 * x^2) * (6 * x) * (1 / (3 * x)^2) = 10 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l99_9972


namespace NUMINAMATH_CALUDE_glee_club_gender_ratio_l99_9971

/-- Given a glee club with total members and female members, 
    prove the ratio of female to male members -/
theorem glee_club_gender_ratio (total : ℕ) (female : ℕ) 
    (h1 : total = 18) (h2 : female = 12) :
    (female : ℚ) / ((total - female) : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_glee_club_gender_ratio_l99_9971


namespace NUMINAMATH_CALUDE_square_product_of_b_values_l99_9973

theorem square_product_of_b_values : ∃ (b₁ b₂ : ℝ),
  (∀ (x y : ℝ), (y = 3 ∨ y = 8 ∨ x = 2 ∨ x = b₁ ∨ x = b₂) →
    ((x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 8) ∨ (x = b₁ ∧ y = 3) ∨ (x = b₁ ∧ y = 8) ∨
     (x = b₂ ∧ y = 3) ∨ (x = b₂ ∧ y = 8) ∨ (x = 2 ∧ 3 ≤ y ∧ y ≤ 8) ∨
     (x = b₁ ∧ 3 ≤ y ∧ y ≤ 8) ∨ (x = b₂ ∧ 3 ≤ y ∧ y ≤ 8) ∨
     (3 ≤ x ∧ x ≤ 8 ∧ y = 3) ∨ (3 ≤ x ∧ x ≤ 8 ∧ y = 8))) ∧
  b₁ * b₂ = -21 :=
by sorry

end NUMINAMATH_CALUDE_square_product_of_b_values_l99_9973


namespace NUMINAMATH_CALUDE_sunglasses_and_hats_probability_l99_9922

theorem sunglasses_and_hats_probability (total_sunglasses : ℕ) (total_hats : ℕ) 
  (prob_hat_and_sunglasses : ℚ) :
  total_sunglasses = 75 →
  total_hats = 60 →
  prob_hat_and_sunglasses = 1 / 3 →
  (prob_hat_and_sunglasses * total_hats : ℚ) / total_sunglasses = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_sunglasses_and_hats_probability_l99_9922


namespace NUMINAMATH_CALUDE_sphere_volume_implies_pi_l99_9991

theorem sphere_volume_implies_pi (D : ℝ) (h : D > 0) :
  (D^3 / 2 + 1 / 21 * D^3 / 2 = π * D^3 / 6) → π = 22 / 7 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_implies_pi_l99_9991


namespace NUMINAMATH_CALUDE_cubic_factorization_l99_9968

theorem cubic_factorization (a b c d e : ℝ) :
  (∀ x, 216 * x^3 - 27 = (a * x - b) * (c * x^2 + d * x - e)) →
  a + b + c + d + e = 72 := by
sorry

end NUMINAMATH_CALUDE_cubic_factorization_l99_9968


namespace NUMINAMATH_CALUDE_function_not_monotonic_iff_m_gt_four_l99_9987

/-- A function f(x) = mln(x+1) + x^2 - mx is not monotonic on (1, +∞) iff m > 4 -/
theorem function_not_monotonic_iff_m_gt_four (m : ℝ) :
  (∃ (x y : ℝ), 1 < x ∧ x < y ∧
    (m * Real.log (x + 1) + x^2 - m * x ≤ m * Real.log (y + 1) + y^2 - m * y ∧
     m * Real.log (y + 1) + y^2 - m * y ≤ m * Real.log (x + 1) + x^2 - m * x)) ↔
  m > 4 :=
by sorry


end NUMINAMATH_CALUDE_function_not_monotonic_iff_m_gt_four_l99_9987


namespace NUMINAMATH_CALUDE_product_of_integers_l99_9962

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 26)
  (diff_squares_eq : x^2 - y^2 = 52) :
  x * y = 168 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l99_9962


namespace NUMINAMATH_CALUDE_one_fifth_of_ten_x_plus_three_l99_9929

theorem one_fifth_of_ten_x_plus_three (x : ℝ) : (1 / 5) * (10 * x + 3) = 2 * x + 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_ten_x_plus_three_l99_9929


namespace NUMINAMATH_CALUDE_inequality_proof_l99_9941

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem inequality_proof (a s t : ℝ) (h1 : s > 0) (h2 : t > 0) 
  (h3 : 2 * s + t = a) 
  (h4 : Set.Icc (-1) 7 = {x | f a x ≤ 4}) : 
  1 / s + 8 / t ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_inequality_proof_l99_9941


namespace NUMINAMATH_CALUDE_equation_solution_l99_9956

theorem equation_solution :
  ∃ x : ℝ, 3 * x - 2 * x = 7 ∧ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l99_9956


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l99_9970

theorem purely_imaginary_z (α : ℝ) : 
  let z : ℂ := Complex.mk (Real.sin α) (-(1 - Real.cos α))
  z.re = 0 → ∃ k : ℤ, α = (2 * k + 1) * Real.pi := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l99_9970


namespace NUMINAMATH_CALUDE_roots_inside_unit_circle_iff_triangle_interior_l99_9916

/-- The region in the (a,b) plane where both roots of z^2 + az + b = 0 satisfy |z| < 1 -/
def roots_inside_unit_circle (a b : ℝ) : Prop :=
  ∀ z : ℂ, z^2 + a*z + b = 0 → Complex.abs z < 1

/-- The interior of the triangle with vertices (2, 1), (-2, 1), and (0, -1) -/
def triangle_interior (a b : ℝ) : Prop :=
  b < 1 ∧ b > a - 1 ∧ b > -a - 1 ∧ b > -1

theorem roots_inside_unit_circle_iff_triangle_interior (a b : ℝ) :
  roots_inside_unit_circle a b ↔ triangle_interior a b :=
sorry

end NUMINAMATH_CALUDE_roots_inside_unit_circle_iff_triangle_interior_l99_9916


namespace NUMINAMATH_CALUDE_pig_count_l99_9938

theorem pig_count (initial_pigs : ℕ) : initial_pigs + 86 = 150 → initial_pigs = 64 := by
  sorry

end NUMINAMATH_CALUDE_pig_count_l99_9938


namespace NUMINAMATH_CALUDE_road_travel_cost_example_l99_9924

/-- Calculates the cost of traveling two intersecting roads on a rectangular lawn. -/
def road_travel_cost (lawn_length lawn_width road_width cost_per_sqm : ℝ) : ℝ :=
  let road1_area := road_width * lawn_width
  let road2_area := road_width * lawn_length
  let intersection_area := road_width * road_width
  let total_road_area := road1_area + road2_area - intersection_area
  total_road_area * cost_per_sqm

/-- Theorem stating that the cost of traveling two intersecting roads on a specific rectangular lawn is 4500. -/
theorem road_travel_cost_example : road_travel_cost 100 60 10 3 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_road_travel_cost_example_l99_9924


namespace NUMINAMATH_CALUDE_bryce_raisins_l99_9976

theorem bryce_raisins : ∃ (bryce carter : ℕ), 
  bryce = carter + 8 ∧ 
  carter = bryce / 3 ∧ 
  bryce = 12 := by
sorry

end NUMINAMATH_CALUDE_bryce_raisins_l99_9976


namespace NUMINAMATH_CALUDE_max_min_product_l99_9903

theorem max_min_product (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 12 → 
  a * b + b * c + c * a = 30 → 
  (min (a * b) (min (b * c) (c * a))) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l99_9903


namespace NUMINAMATH_CALUDE_tan_product_theorem_l99_9905

theorem tan_product_theorem (α β : Real) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2) (h5 : α + β = π / 4) :
  (1 + Real.tan α) * (1 + Real.tan β) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_theorem_l99_9905


namespace NUMINAMATH_CALUDE_binomial_2023_1_l99_9909

theorem binomial_2023_1 : Nat.choose 2023 1 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_binomial_2023_1_l99_9909


namespace NUMINAMATH_CALUDE_kathleen_remaining_money_l99_9908

def june_savings : ℕ := 21
def july_savings : ℕ := 46
def august_savings : ℕ := 45
def school_supplies_expense : ℕ := 12
def clothes_expense : ℕ := 54
def aunt_bonus_threshold : ℕ := 125
def aunt_bonus : ℕ := 25

def total_savings : ℕ := june_savings + july_savings + august_savings
def total_expenses : ℕ := school_supplies_expense + clothes_expense

theorem kathleen_remaining_money :
  total_savings - total_expenses = 46 :=
sorry

end NUMINAMATH_CALUDE_kathleen_remaining_money_l99_9908


namespace NUMINAMATH_CALUDE_prob_third_term_four_sum_of_fraction_parts_l99_9932

/-- Set of permutations of 1,2,3,4,5,6 with restrictions -/
def T : Set (Fin 6 → Fin 6) :=
  { σ | Function.Bijective σ ∧ 
        σ 0 ≠ 0 ∧ σ 0 ≠ 1 ∧
        σ 1 ≠ 2 }

/-- The cardinality of set T -/
def T_size : ℕ := 48

/-- The number of permutations in T where the third term is 4 -/
def favorable_outcomes : ℕ := 12

/-- The probability of the third term being 4 in a randomly chosen permutation from T -/
theorem prob_third_term_four : 
  (favorable_outcomes : ℚ) / T_size = 1 / 4 :=
sorry

/-- The sum of numerator and denominator in the probability fraction -/
theorem sum_of_fraction_parts : 
  1 + 4 = 5 :=
sorry

end NUMINAMATH_CALUDE_prob_third_term_four_sum_of_fraction_parts_l99_9932


namespace NUMINAMATH_CALUDE_parallelogram_height_l99_9984

theorem parallelogram_height (area base height : ℝ) : 
  area = 480 ∧ base = 32 ∧ area = base * height → height = 15 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l99_9984


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l99_9983

theorem sqrt_product_simplification (q : ℝ) :
  Real.sqrt (15 * q) * Real.sqrt (8 * q^2) * Real.sqrt (14 * q^3) = 4 * q^3 * Real.sqrt 105 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l99_9983


namespace NUMINAMATH_CALUDE_unit_vectors_equal_magnitude_l99_9964

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors a and b
variable (a b : V)

-- State the theorem
theorem unit_vectors_equal_magnitude
  (ha : ‖a‖ = 1) -- a is a unit vector
  (hb : ‖b‖ = 1) -- b is a unit vector
  : ‖a‖ = ‖b‖ := by
  sorry

end NUMINAMATH_CALUDE_unit_vectors_equal_magnitude_l99_9964


namespace NUMINAMATH_CALUDE_candy_bar_profit_l99_9996

theorem candy_bar_profit :
  let total_bars : ℕ := 1200
  let buy_price : ℚ := 5 / 6
  let sell_price : ℚ := 2 / 3
  let cost : ℚ := total_bars * buy_price
  let revenue : ℚ := total_bars * sell_price
  let profit : ℚ := revenue - cost
  profit = -200
:= by sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l99_9996


namespace NUMINAMATH_CALUDE_square_sum_of_complex_square_l99_9951

theorem square_sum_of_complex_square (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (↑a + ↑b * Complex.I)^2 = (3 : ℂ) + 4 * Complex.I →
  a^2 + b^2 = 25 := by sorry

end NUMINAMATH_CALUDE_square_sum_of_complex_square_l99_9951


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l99_9942

-- Define an isosceles triangle with one angle of 70°
structure IsoscelesTriangle :=
  (angle1 : Real)
  (angle2 : Real)
  (angle3 : Real)
  (isIsosceles : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3))
  (has70Degree : angle1 = 70 ∨ angle2 = 70 ∨ angle3 = 70)
  (sumIs180 : angle1 + angle2 + angle3 = 180)

-- Theorem statement
theorem isosceles_triangle_angles (t : IsoscelesTriangle) :
  (t.angle1 = 55 ∧ t.angle2 = 55 ∧ t.angle3 = 70) ∨
  (t.angle1 = 55 ∧ t.angle2 = 70 ∧ t.angle3 = 55) ∨
  (t.angle1 = 70 ∧ t.angle2 = 55 ∧ t.angle3 = 55) ∨
  (t.angle1 = 70 ∧ t.angle2 = 70 ∧ t.angle3 = 40) ∨
  (t.angle1 = 70 ∧ t.angle2 = 40 ∧ t.angle3 = 70) ∨
  (t.angle1 = 40 ∧ t.angle2 = 70 ∧ t.angle3 = 70) :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angles_l99_9942


namespace NUMINAMATH_CALUDE_triangle_special_area_l99_9934

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the angles form an arithmetic sequence and a, c, 4/√3 * b form a geometric sequence,
    then the area of the triangle is √3/2 * a² -/
theorem triangle_special_area (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  B = (A + C) / 2 →
  ∃ (q : ℝ), q > 0 ∧ c = q * a ∧ 4 / Real.sqrt 3 * b = q^2 * a →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 / 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_area_l99_9934


namespace NUMINAMATH_CALUDE_custom_mul_two_neg_three_l99_9901

-- Define the custom multiplication operation
def custom_mul (a b : ℤ) : ℤ := a^2 + a*b - b^2

-- Theorem statement
theorem custom_mul_two_neg_three :
  custom_mul 2 (-3) = -11 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_two_neg_three_l99_9901


namespace NUMINAMATH_CALUDE_lucia_dance_cost_l99_9978

/-- Represents the cost and frequency of a dance class -/
structure DanceClass where
  name : String
  cost : ℚ
  frequency : ℚ

/-- Calculates the total cost of a dance class over a 4-week period -/
def classCost (c : DanceClass) : ℚ :=
  c.cost * (4 * c.frequency)

/-- Lucia's dance schedule -/
def luciaSchedule : List DanceClass := [
  { name := "Hip-hop", cost := 10.5, frequency := 3 },
  { name := "Salsa", cost := 15, frequency := 0.5 },
  { name := "Ballet", cost := 12.25, frequency := 2 },
  { name := "Jazz", cost := 8.75, frequency := 1 },
  { name := "Contemporary", cost := 10, frequency := 1/3 }
]

/-- The total cost of Lucia's dance classes for a 4-week period -/
def totalCost : ℚ := (luciaSchedule.map classCost).sum

theorem lucia_dance_cost : totalCost = 299 := by
  sorry

end NUMINAMATH_CALUDE_lucia_dance_cost_l99_9978


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l99_9947

-- Expression 1
theorem simplify_expression_1 (a b : ℤ) :
  2*a - 6*b - 3*a + 9*b = -a + 3*b := by sorry

-- Expression 2
theorem simplify_expression_2 (m n : ℤ) :
  2*(3*m^2 - m*n) - m*n + m^2 = 7*m^2 - 3*m*n := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l99_9947


namespace NUMINAMATH_CALUDE_tan_20_plus_4sin_20_equals_sqrt_3_l99_9990

theorem tan_20_plus_4sin_20_equals_sqrt_3 :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_20_plus_4sin_20_equals_sqrt_3_l99_9990


namespace NUMINAMATH_CALUDE_min_weights_theorem_l99_9999

/-- A function that calculates the sum of powers of 2 up to 2^n -/
def sumPowersOf2 (n : ℕ) : ℕ := 2^(n+1) - 1

/-- The maximum weight we need to measure -/
def maxWeight : ℕ := 100

/-- The proposition that n weights are sufficient to measure all weights up to maxWeight -/
def isSufficient (n : ℕ) : Prop := sumPowersOf2 n ≥ maxWeight

/-- The proposition that n weights are necessary to measure all weights up to maxWeight -/
def isNecessary (n : ℕ) : Prop := ∀ m : ℕ, m < n → sumPowersOf2 m < maxWeight

/-- The theorem stating that 7 is the minimum number of weights needed -/
theorem min_weights_theorem : 
  (isSufficient 7 ∧ isNecessary 7) ∧ ∀ n : ℕ, n < 7 → ¬(isSufficient n ∧ isNecessary n) :=
sorry

end NUMINAMATH_CALUDE_min_weights_theorem_l99_9999


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l99_9949

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- State the theorem
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : t.B = (t.A + t.C) / 2)  -- B is arithmetic mean of A and C
  (h2 : t.b^2 = t.a * t.c)      -- b is geometric mean of a and c
  : t.A = t.B ∧ t.B = t.C ∧ t.a = t.b ∧ t.b = t.c :=
by sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l99_9949


namespace NUMINAMATH_CALUDE_max_a_satisfies_equation_no_larger_a_satisfies_equation_max_a_is_maximum_l99_9918

/-- The coefficient of x^4 in the expansion of (1-3x+ax^2)^8 --/
def coefficient_x4 (a : ℝ) : ℝ := 28 * a^2 + 2016 * a + 5670

/-- The equation that a must satisfy --/
def equation (a : ℝ) : Prop := coefficient_x4 a = 70

/-- The maximum value of a that satisfies the equation --/
noncomputable def max_a : ℝ := -36 + Real.sqrt 1096

theorem max_a_satisfies_equation : equation max_a :=
sorry

theorem no_larger_a_satisfies_equation :
  ∀ a : ℝ, a > max_a → ¬(equation a) :=
sorry

theorem max_a_is_maximum :
  ∃ (ε : ℝ), ε > 0 ∧ (∀ δ : ℝ, 0 < δ ∧ δ < ε → ¬(equation (max_a + δ))) :=
sorry

end NUMINAMATH_CALUDE_max_a_satisfies_equation_no_larger_a_satisfies_equation_max_a_is_maximum_l99_9918


namespace NUMINAMATH_CALUDE_example_linear_equation_l99_9939

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants and at least one of a or b is non-zero. --/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y = a * x + b * y - c

/-- The equation x + 4y = 6 is a linear equation in two variables. --/
theorem example_linear_equation :
  IsLinearEquationInTwoVariables (fun x y ↦ x + 4 * y - 6) := by
  sorry

end NUMINAMATH_CALUDE_example_linear_equation_l99_9939


namespace NUMINAMATH_CALUDE_janet_oranges_l99_9926

theorem janet_oranges (sharon_oranges : ℕ) (total_oranges : ℕ) (h1 : sharon_oranges = 7) (h2 : total_oranges = 16) :
  total_oranges - sharon_oranges = 9 :=
by sorry

end NUMINAMATH_CALUDE_janet_oranges_l99_9926


namespace NUMINAMATH_CALUDE_sum_possible_constants_eq_1232_l99_9928

/-- 
Given a quadratic equation ax² + bx + c = 0 with two distinct negative integer roots,
where b = 24, this function computes the sum of all possible values for c.
-/
def sum_possible_constants : ℤ := by
  sorry

/-- The main theorem stating that the sum of all possible constant terms is 1232 -/
theorem sum_possible_constants_eq_1232 : sum_possible_constants = 1232 := by
  sorry

end NUMINAMATH_CALUDE_sum_possible_constants_eq_1232_l99_9928


namespace NUMINAMATH_CALUDE_b_value_l99_9910

theorem b_value (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 15 * b) : b = 147 := by
  sorry

end NUMINAMATH_CALUDE_b_value_l99_9910


namespace NUMINAMATH_CALUDE_max_value_constraint_l99_9954

theorem max_value_constraint (x y : ℝ) 
  (h1 : |x - y| ≤ 2) 
  (h2 : |3*x + y| ≤ 6) : 
  x^2 + y^2 ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l99_9954
