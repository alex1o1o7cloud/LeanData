import Mathlib

namespace peach_boxes_theorem_l2498_249821

/-- Given the initial number of peaches per basket, the number of baskets,
    the number of peaches eaten, and the number of peaches per smaller box,
    calculate the number of smaller boxes of peaches. -/
def number_of_boxes (peaches_per_basket : ℕ) (num_baskets : ℕ) (peaches_eaten : ℕ) (peaches_per_small_box : ℕ) : ℕ :=
  ((peaches_per_basket * num_baskets) - peaches_eaten) / peaches_per_small_box

/-- Prove that under the given conditions, the number of smaller boxes of peaches is 8. -/
theorem peach_boxes_theorem :
  number_of_boxes 25 5 5 15 = 8 := by
  sorry

end peach_boxes_theorem_l2498_249821


namespace sandy_comic_books_l2498_249831

theorem sandy_comic_books (x : ℕ) : 
  (x / 2 : ℚ) - 3 + 6 = 13 → x = 20 := by
  sorry

end sandy_comic_books_l2498_249831


namespace min_area_hyperbola_triangle_l2498_249855

/-- A point on the hyperbola xy = 1 -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  on_hyperbola : x * y = 1

/-- An isosceles right triangle on the hyperbola xy = 1 -/
structure HyperbolaTriangle where
  A : HyperbolaPoint
  B : HyperbolaPoint
  C : HyperbolaPoint
  is_right_angle : (B.x - A.x) * (C.x - A.x) + (B.y - A.y) * (C.y - A.y) = 0
  is_isosceles : (B.x - A.x)^2 + (B.y - A.y)^2 = (C.x - A.x)^2 + (C.y - A.y)^2

/-- The area of a triangle given by three points -/
def triangleArea (A B C : HyperbolaPoint) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

/-- The theorem stating the minimum area of an isosceles right triangle on the hyperbola xy = 1 -/
theorem min_area_hyperbola_triangle :
  ∀ T : HyperbolaTriangle, triangleArea T.A T.B T.C ≥ 3 * Real.sqrt 3 := by
  sorry

#check min_area_hyperbola_triangle

end min_area_hyperbola_triangle_l2498_249855


namespace angle_relation_l2498_249805

theorem angle_relation (A B C : Real) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) (h3 : 0 < C ∧ C < π/2) (h4 : A + B + C = π) :
  (Real.cos (A/2))^2 = (Real.cos (B/2))^2 + (Real.cos (C/2))^2 - 2 * Real.cos (B/2) * Real.cos (C/2) * Real.sin (A/2) := by
  sorry

end angle_relation_l2498_249805


namespace partial_fraction_decomposition_l2498_249874

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), A = 31/9 ∧ B = 5/9 ∧
  ∀ (x : ℝ), x ≠ 7 ∧ x ≠ -2 →
  (4*x^2 + 7*x + 3) / (x^2 - 5*x - 14) = A / (x - 7) + B / (x + 2) := by
  sorry

end partial_fraction_decomposition_l2498_249874


namespace sum_of_cubes_zero_l2498_249814

theorem sum_of_cubes_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum_squares : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end sum_of_cubes_zero_l2498_249814


namespace power_seven_135_mod_12_l2498_249888

theorem power_seven_135_mod_12 : 7^135 % 12 = 7 := by
  sorry

end power_seven_135_mod_12_l2498_249888


namespace partial_fraction_decomposition_l2498_249883

theorem partial_fraction_decomposition (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ 3 →
    1 / (x^3 - 3*x^2 - 13*x + 15) = A / (x - 1) + B / (x - 3) + C / ((x - 3)^2)) →
  A = 1/4 := by
sorry

end partial_fraction_decomposition_l2498_249883


namespace inequality_properties_l2498_249844

theorem inequality_properties (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧ (a^2 > b^2) ∧ (a * b > b^2) := by
  sorry

end inequality_properties_l2498_249844


namespace bus_ride_is_75_minutes_l2498_249811

/-- Calculates the bus ride duration given the total trip time, train ride duration, and walking time. -/
def bus_ride_duration (total_trip_time : ℕ) (train_ride_duration : ℕ) (walking_time : ℕ) : ℕ :=
  let total_minutes := total_trip_time * 60
  let train_minutes := train_ride_duration * 60
  let waiting_time := walking_time * 2
  total_minutes - train_minutes - waiting_time - walking_time

/-- Proves that given the specified conditions, the bus ride duration is 75 minutes. -/
theorem bus_ride_is_75_minutes :
  bus_ride_duration 8 6 15 = 75 := by
  sorry

#eval bus_ride_duration 8 6 15

end bus_ride_is_75_minutes_l2498_249811


namespace marble_selection_ways_l2498_249835

def total_marbles : ℕ := 15
def specific_colors : ℕ := 5
def marbles_to_choose : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem marble_selection_ways :
  (specific_colors * choose (total_marbles - specific_colors - 1) (marbles_to_choose - 1)) = 630 := by
  sorry

end marble_selection_ways_l2498_249835


namespace dividend_calculation_l2498_249833

theorem dividend_calculation (divisor quotient remainder dividend : ℕ) : 
  divisor = 18 → quotient = 9 → remainder = 3 → 
  dividend = divisor * quotient + remainder →
  dividend = 165 := by
sorry

end dividend_calculation_l2498_249833


namespace lcm_prime_sum_l2498_249894

theorem lcm_prime_sum (x y z : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (hz : Nat.Prime z)
  (hlcm : Nat.lcm x (Nat.lcm y z) = 210) (hord : x > y ∧ y > z) : 2 * x + y + z = 22 := by
  sorry

end lcm_prime_sum_l2498_249894


namespace money_left_calculation_l2498_249879

def initial_amount : ℝ := 200

def notebook_cost : ℝ := 4
def book_cost : ℝ := 12
def pen_cost : ℝ := 2
def sticker_pack_cost : ℝ := 6
def shoes_cost : ℝ := 40
def tshirt_cost : ℝ := 18

def notebooks_bought : ℕ := 7
def books_bought : ℕ := 2
def pens_bought : ℕ := 5
def sticker_packs_bought : ℕ := 3

def sales_tax_rate : ℝ := 0.05

def lunch_cost : ℝ := 15
def tip_amount : ℝ := 3
def transportation_cost : ℝ := 8
def charity_amount : ℝ := 10

def total_mall_purchase_cost : ℝ :=
  notebook_cost * notebooks_bought +
  book_cost * books_bought +
  pen_cost * pens_bought +
  sticker_pack_cost * sticker_packs_bought +
  shoes_cost +
  tshirt_cost

def total_mall_cost_with_tax : ℝ :=
  total_mall_purchase_cost * (1 + sales_tax_rate)

def total_expenses : ℝ :=
  total_mall_cost_with_tax +
  lunch_cost +
  tip_amount +
  transportation_cost +
  charity_amount

theorem money_left_calculation :
  initial_amount - total_expenses = 19.10 := by
  sorry

end money_left_calculation_l2498_249879


namespace unfolded_holes_count_l2498_249816

/-- Represents a square piece of paper -/
structure Paper :=
  (side : ℝ)
  (is_square : side > 0)

/-- Represents the state of the paper after folding and punching -/
structure FoldedPaper :=
  (original : Paper)
  (center_hole : Bool)
  (upper_right_hole : Bool)

/-- Counts the number of holes when the paper is unfolded -/
def count_holes (fp : FoldedPaper) : ℕ :=
  let center_holes := if fp.center_hole then 4 else 0
  let corner_holes := if fp.upper_right_hole then 4 else 0
  center_holes + corner_holes

/-- Theorem stating that the number of holes when unfolded is 8 -/
theorem unfolded_holes_count (p : Paper) : 
  ∀ (fp : FoldedPaper), 
    fp.original = p → 
    fp.center_hole = true → 
    fp.upper_right_hole = true → 
    count_holes fp = 8 :=
sorry

end unfolded_holes_count_l2498_249816


namespace present_ages_l2498_249884

/-- Represents the ages of Rahul, Deepak, and Karan -/
structure Ages where
  rahul : ℕ
  deepak : ℕ
  karan : ℕ

/-- The present age ratio between Rahul, Deepak, and Karan -/
def age_ratio (ages : Ages) : Prop :=
  4 * ages.deepak = 3 * ages.rahul ∧ 5 * ages.deepak = 3 * ages.karan

/-- In 8 years, the sum of Rahul's and Deepak's ages will equal Karan's age -/
def future_age_sum (ages : Ages) : Prop :=
  ages.rahul + ages.deepak + 16 = ages.karan

/-- Rahul's age after 6 years will be 26 years -/
def rahul_future_age (ages : Ages) : Prop :=
  ages.rahul + 6 = 26

/-- The theorem to be proved -/
theorem present_ages (ages : Ages) 
  (h1 : age_ratio ages) 
  (h2 : future_age_sum ages) 
  (h3 : rahul_future_age ages) : 
  ages.deepak = 15 ∧ ages.karan = 51 := by
  sorry

end present_ages_l2498_249884


namespace greatest_common_length_l2498_249812

theorem greatest_common_length (a b c d : ℕ) 
  (ha : a = 48) (hb : b = 60) (hc : c = 72) (hd : d = 120) : 
  Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 12 := by
  sorry

end greatest_common_length_l2498_249812


namespace parabola_c_value_l2498_249808

/-- A parabola passing through two points -/
def Parabola (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

theorem parabola_c_value :
  ∀ b c : ℝ,
  Parabola b c 1 = 6 →
  Parabola b c 5 = 10 →
  c = 10 := by
  sorry

end parabola_c_value_l2498_249808


namespace gcd_factorial_eight_nine_l2498_249800

theorem gcd_factorial_eight_nine : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := by
  sorry

end gcd_factorial_eight_nine_l2498_249800


namespace min_people_hat_glove_not_scarf_l2498_249842

theorem min_people_hat_glove_not_scarf (n : ℕ) 
  (gloves : ℕ) (hats : ℕ) (scarves : ℕ) :
  gloves = (3 * n) / 8 ∧ 
  hats = (5 * n) / 6 ∧ 
  scarves = n / 4 →
  ∃ (x : ℕ), x = hats + gloves - (n - scarves) ∧ 
  x ≥ 11 ∧ 
  (∀ (m : ℕ), m < n → 
    (3 * m) % 8 ≠ 0 ∨ (5 * m) % 6 ≠ 0 ∨ m % 4 ≠ 0) :=
by sorry

end min_people_hat_glove_not_scarf_l2498_249842


namespace quadratic_root_existence_l2498_249859

theorem quadratic_root_existence (a b c : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : a * x₁^2 + b * x₁ + c = 0)
  (h₂ : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃, (1/2 * a * x₃^2 + b * x₃ + c = 0) ∧ 
    ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) := by
  sorry

end quadratic_root_existence_l2498_249859


namespace like_terms_exponents_l2498_249847

theorem like_terms_exponents (m n : ℤ) : 
  (∃ (x y : ℝ), 2 * x^(2*m) * y^6 = -3 * x^8 * y^(2*n)) → m = 4 ∧ n = 3 :=
by sorry

end like_terms_exponents_l2498_249847


namespace apple_percentage_after_orange_removal_l2498_249841

/-- Calculates the percentage of apples in a bowl of fruit after removing oranges -/
theorem apple_percentage_after_orange_removal (initial_apples initial_oranges removed_oranges : ℕ) 
  (h1 : initial_apples = 14)
  (h2 : initial_oranges = 20)
  (h3 : removed_oranges = 14) :
  (initial_apples : ℚ) / (initial_apples + initial_oranges - removed_oranges) * 100 = 70 := by
  sorry


end apple_percentage_after_orange_removal_l2498_249841


namespace function_inequality_l2498_249839

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f
variable (h : ∀ x, x > 0 → DifferentiableAt ℝ f x)

-- Define the condition f(x)/x > f'(x)
variable (cond : ∀ x, x > 0 → (f x) / x > deriv f x)

-- Theorem statement
theorem function_inequality : 2015 * (f 2016) > 2016 * (f 2015) := by
  sorry

end function_inequality_l2498_249839


namespace peanuts_in_box_l2498_249866

/-- Given a box with an initial number of peanuts and an additional number of peanuts added,
    calculate the total number of peanuts in the box. -/
def total_peanuts (initial : Nat) (added : Nat) : Nat :=
  initial + added

/-- Theorem stating that if there are initially 4 peanuts in a box and 8 more are added,
    the total number of peanuts in the box is 12. -/
theorem peanuts_in_box : total_peanuts 4 8 = 12 := by
  sorry

end peanuts_in_box_l2498_249866


namespace tan_theta_plus_pi_fourth_l2498_249886

theorem tan_theta_plus_pi_fourth (θ : Real) 
  (h : (Real.cos (2 * θ) + 1) / (1 + 2 * Real.sin (2 * θ)) = -2/3) : 
  Real.tan (θ + π/4) = -1/3 := by
  sorry

end tan_theta_plus_pi_fourth_l2498_249886


namespace total_acorns_formula_l2498_249875

/-- The total number of acorns for Shawna, Sheila, Danny, and Ella -/
def total_acorns (x y : ℝ) : ℝ :=
  let shawna := x
  let sheila := 5.3 * x
  let danny := sheila + y
  let ella := 2 * (danny - shawna)
  shawna + sheila + danny + ella

/-- Theorem stating the total number of acorns -/
theorem total_acorns_formula (x y : ℝ) :
  total_acorns x y = 20.2 * x + 3 * y := by
  sorry

end total_acorns_formula_l2498_249875


namespace parity_of_linear_system_solution_l2498_249843

theorem parity_of_linear_system_solution (n m : ℤ) 
  (h_n_odd : Odd n) (h_m_odd : Odd m) :
  ∃ (x y : ℤ), x + 2*y = n ∧ 3*x - y = m → Odd x ∧ Even y := by
  sorry

end parity_of_linear_system_solution_l2498_249843


namespace painted_fraction_of_specific_cone_l2498_249849

/-- Represents a cone with given dimensions -/
structure Cone where
  radius : ℝ
  slant_height : ℝ

/-- Calculates the fraction of a cone's surface area covered in paint -/
def painted_fraction (c : Cone) (paint_depth : ℝ) : ℚ :=
  sorry

/-- Theorem stating the correct fraction of painted surface area for the given cone -/
theorem painted_fraction_of_specific_cone :
  let c : Cone := { radius := 3, slant_height := 5 }
  painted_fraction c 2 = 27 / 32 := by
  sorry

end painted_fraction_of_specific_cone_l2498_249849


namespace value_of_y_l2498_249861

theorem value_of_y : ∃ y : ℚ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end value_of_y_l2498_249861


namespace line_intercepts_l2498_249897

/-- A line in the 2D plane defined by the equation y = x + 3 -/
structure Line where
  slope : ℝ := 1
  y_intercept : ℝ := 3

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ × ℝ :=
  (-l.y_intercept, 0)

/-- The y-intercept of a line -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.y_intercept)

theorem line_intercepts (l : Line) :
  x_intercept l = (-3, 0) ∧ y_intercept l = (0, 3) := by
  sorry

end line_intercepts_l2498_249897


namespace trapezoid_area_is_42_5_l2498_249873

/-- A trapezoid bounded by the lines y = x + 2, y = 12, y = 7, and the y-axis -/
structure Trapezoid where
  -- Line equations
  line1 : ℝ → ℝ := λ x => x + 2
  line2 : ℝ → ℝ := λ _ => 12
  line3 : ℝ → ℝ := λ _ => 7
  y_axis : ℝ → ℝ := λ _ => 0

/-- The area of the trapezoid -/
def trapezoid_area (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that the area of the trapezoid is 42.5 square units -/
theorem trapezoid_area_is_42_5 (t : Trapezoid) : trapezoid_area t = 42.5 := by sorry

end trapezoid_area_is_42_5_l2498_249873


namespace dish_heating_rate_l2498_249889

/-- Given the initial and final temperatures of a dish and the time taken to heat it,
    calculate the heating rate in degrees per minute. -/
theorem dish_heating_rate 
  (initial_temp : ℝ) 
  (final_temp : ℝ) 
  (heating_time : ℝ) 
  (h1 : initial_temp = 20) 
  (h2 : final_temp = 100) 
  (h3 : heating_time = 16) : 
  (final_temp - initial_temp) / heating_time = 5 := by
  sorry

#check dish_heating_rate

end dish_heating_rate_l2498_249889


namespace system_solution_l2498_249813

theorem system_solution :
  ∃ (x y z : ℚ),
    (7 * x - 3 * y + 2 * z = 4) ∧
    (2 * x + 8 * y - z = 1) ∧
    (3 * x - 4 * y + 5 * z = 7) ∧
    (x = 1262 / 913) ∧
    (y = -59 / 83) :=
by sorry

end system_solution_l2498_249813


namespace paper_cutting_game_l2498_249851

theorem paper_cutting_game (n : ℕ) (pieces : ℕ) : 
  (pieces = 8 * n + 1) → (pieces = 2009) → (n = 251) := by
  sorry

end paper_cutting_game_l2498_249851


namespace open_box_volume_l2498_249850

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet -/
theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_length : ℝ)
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 3)
  : (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 3780 :=
by sorry

end open_box_volume_l2498_249850


namespace range_of_m_l2498_249810

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 1) * x + 2

-- Define the condition that the solution set is ℝ
def solution_set_is_real (m : ℝ) : Prop :=
  ∀ x, f m x > 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  solution_set_is_real m ↔ (1 ≤ m ∧ m < 9) :=
sorry

end range_of_m_l2498_249810


namespace birds_on_fence_l2498_249838

/-- The number of additional birds that joined the fence -/
def additional_birds : ℕ := sorry

/-- The initial number of birds on the fence -/
def initial_birds : ℕ := 2

/-- The number of storks that joined the fence -/
def joined_storks : ℕ := 4

theorem birds_on_fence :
  additional_birds = 5 :=
by
  have h1 : initial_birds + additional_birds = joined_storks + 3 :=
    sorry
  sorry

end birds_on_fence_l2498_249838


namespace undeclared_major_fraction_l2498_249857

theorem undeclared_major_fraction (T : ℝ) (f : ℝ) : 
  T > 0 →
  (1/2 : ℝ) * T = T - (1/2 : ℝ) * T →
  (1/2 : ℝ) * T * (1 - (1/2 : ℝ) * (1 - f)) = (45/100 : ℝ) * T →
  f = 4/5 := by sorry

end undeclared_major_fraction_l2498_249857


namespace unique_solution_system_l2498_249801

theorem unique_solution_system (s t : ℝ) : 
  15 * s + 10 * t = 270 ∧ s = 3 * t - 4 → s = 14 ∧ t = 6 := by
  sorry

end unique_solution_system_l2498_249801


namespace coeff_x3_is_42_l2498_249898

/-- First polynomial: x^5 - 4x^4 + 7x^3 - 5x^2 + 3x - 2 -/
def p1 (x : ℝ) : ℝ := x^5 - 4*x^4 + 7*x^3 - 5*x^2 + 3*x - 2

/-- Second polynomial: 3x^2 - 5x + 6 -/
def p2 (x : ℝ) : ℝ := 3*x^2 - 5*x + 6

/-- The product of the two polynomials -/
def product (x : ℝ) : ℝ := p1 x * p2 x

/-- Theorem: The coefficient of x^3 in the product of p1 and p2 is 42 -/
theorem coeff_x3_is_42 : ∃ (a b c d e f : ℝ), product x = a*x^5 + b*x^4 + 42*x^3 + d*x^2 + e*x + f :=
sorry

end coeff_x3_is_42_l2498_249898


namespace rationalize_denominator_l2498_249862

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (2 : ℝ) / (3 * Real.sqrt 7 + 2 * Real.sqrt 13) =
    (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = 6 ∧
    B = 7 ∧
    C = -4 ∧
    D = 13 ∧
    E = 11 ∧
    Int.gcd A E = 1 ∧
    Int.gcd C E = 1 ∧
    ¬∃ (k : ℤ), k > 1 ∧ k ^ 2 ∣ B ∧
    ¬∃ (k : ℤ), k > 1 ∧ k ^ 2 ∣ D :=
by sorry

end rationalize_denominator_l2498_249862


namespace pure_imaginary_condition_l2498_249876

theorem pure_imaginary_condition (a : ℝ) : 
  let i : ℂ := Complex.I
  let z : ℂ := (a + i) / (1 + i)
  (z.re = 0) → a = -1 := by sorry

end pure_imaginary_condition_l2498_249876


namespace wilted_flowers_calculation_l2498_249870

/-- 
Given:
- initial_flowers: The initial number of flowers picked
- flowers_per_bouquet: The number of flowers in each bouquet
- bouquets_made: The number of bouquets that could be made after some flowers wilted

Prove:
The number of wilted flowers is equal to the initial number of flowers minus
the product of the number of bouquets made and the number of flowers per bouquet.
-/
theorem wilted_flowers_calculation (initial_flowers flowers_per_bouquet bouquets_made : ℕ) :
  initial_flowers - (bouquets_made * flowers_per_bouquet) = 
  initial_flowers - bouquets_made * flowers_per_bouquet :=
by sorry

end wilted_flowers_calculation_l2498_249870


namespace complement_A_intersect_B_l2498_249840

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {2, 4, 5}

-- Define set B
def B : Set Nat := {x ∈ U | 2 < x ∧ x < 6}

-- Theorem statement
theorem complement_A_intersect_B : 
  (U \ A) ∩ B = {3} := by sorry

end complement_A_intersect_B_l2498_249840


namespace board_longest_piece_length_l2498_249815

/-- Given a board of length 240 cm cut into four pieces, prove that the longest piece is 120 cm -/
theorem board_longest_piece_length :
  ∀ (L M T F : ℝ),
    L + M + T + F = 240 →
    L = M + T + F →
    M = L / 2 - 10 →
    T ^ 2 = L - M →
    L = 120 := by
  sorry

end board_longest_piece_length_l2498_249815


namespace sum_of_twenty_numbers_l2498_249885

theorem sum_of_twenty_numbers : 
  let numbers : List Nat := [87, 91, 94, 88, 93, 91, 89, 87, 92, 86, 90, 92, 88, 90, 91, 86, 89, 92, 95, 88]
  numbers.sum = 1799 := by
  sorry

end sum_of_twenty_numbers_l2498_249885


namespace plane_perpendicular_condition_l2498_249819

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Plane → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_condition 
  (a b : Line) (α β γ : Plane) :
  perp α γ → para γ β → perp α β :=
by sorry

end plane_perpendicular_condition_l2498_249819


namespace number_equation_l2498_249820

theorem number_equation : ∃ x : ℝ, x / 1500 = 0.016833333333333332 ∧ x = 25.25 := by
  sorry

end number_equation_l2498_249820


namespace poem_line_growth_l2498_249836

/-- The number of months required to reach a target number of lines in a poem, 
    given the initial number of lines and the number of lines added per month. -/
def months_to_reach_target (initial_lines : ℕ) (lines_per_month : ℕ) (target_lines : ℕ) : ℕ :=
  (target_lines - initial_lines) / lines_per_month

theorem poem_line_growth : months_to_reach_target 24 3 90 = 22 := by
  sorry

end poem_line_growth_l2498_249836


namespace grassy_plot_length_l2498_249877

/-- Represents the dimensions and cost of a rectangular grassy plot with a gravel path. -/
structure GrassyPlot where
  width : ℝ  -- Width of the grassy plot in meters
  pathWidth : ℝ  -- Width of the gravel path in meters
  gravelCost : ℝ  -- Cost of gravelling in rupees
  gravelRate : ℝ  -- Cost of gravelling per square meter in rupees

/-- Calculates the length of the grassy plot given its specifications. -/
def calculateLength (plot : GrassyPlot) : ℝ :=
  -- Implementation not provided as per instructions
  sorry

/-- Theorem stating that given the specified conditions, the length of the grassy plot is 100 meters. -/
theorem grassy_plot_length 
  (plot : GrassyPlot) 
  (h1 : plot.width = 65) 
  (h2 : plot.pathWidth = 2.5) 
  (h3 : plot.gravelCost = 425) 
  (h4 : plot.gravelRate = 0.5) : 
  calculateLength plot = 100 := by
  sorry

end grassy_plot_length_l2498_249877


namespace curve_points_difference_l2498_249822

theorem curve_points_difference (e a b : ℝ) : 
  e > 0 →
  (a^2 + e^2 = 2*e*a + 1) →
  (b^2 + e^2 = 2*e*b + 1) →
  a ≠ b →
  |a - b| = 2 := by
sorry

end curve_points_difference_l2498_249822


namespace system_solution_l2498_249853

-- Define the system of equations
def system (x₁ x₂ x₃ x₄ x₅ y : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₅

-- Define the solution
def solution (x₁ x₂ x₃ x₄ x₅ y : ℝ) : Prop :=
  (y = 2 ∧ x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅) ∨
  (y ≠ 2 ∧
    ((y^2 + y - 1 ≠ 0 ∧ x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∨
     (y^2 + y - 1 = 0 ∧ (y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) ∧
      x₃ = y * x₂ - x₁ ∧
      x₄ = -y * x₂ - y * x₁ ∧
      x₅ = y * x₁ - x₂)))

-- Theorem statement
theorem system_solution (x₁ x₂ x₃ x₄ x₅ y : ℝ) :
  system x₁ x₂ x₃ x₄ x₅ y ↔ solution x₁ x₂ x₃ x₄ x₅ y := by
  sorry

end system_solution_l2498_249853


namespace victors_total_money_l2498_249893

-- Define Victor's initial amount
def initial_amount : ℕ := 10

-- Define Victor's allowance
def allowance : ℕ := 8

-- Theorem stating Victor's total money
theorem victors_total_money : initial_amount + allowance = 18 := by
  sorry

end victors_total_money_l2498_249893


namespace envelope_addressing_equation_l2498_249829

theorem envelope_addressing_equation (x : ℝ) : x > 0 → (
  let rate1 := 800 / 12  -- rate of first machine
  let rate2 := 800 / x   -- rate of second machine
  let combined_rate := 800 / 3  -- combined rate of both machines
  rate1 + rate2 = combined_rate ↔ 1/12 + 1/x = 1/3
) := by sorry

end envelope_addressing_equation_l2498_249829


namespace break_even_point_manuals_l2498_249818

/-- The break-even point for manual production -/
theorem break_even_point_manuals :
  let average_cost (Q : ℝ) := 100 + 100000 / Q
  let planned_price := 300
  ∃ Q : ℝ, Q > 0 ∧ average_cost Q = planned_price ∧ Q = 500 :=
by sorry

end break_even_point_manuals_l2498_249818


namespace friends_video_count_l2498_249865

/-- The number of videos watched by three friends. -/
def total_videos (kelsey ekon uma : ℕ) : ℕ := kelsey + ekon + uma

/-- Theorem stating the total number of videos watched by the three friends. -/
theorem friends_video_count :
  ∀ (kelsey ekon uma : ℕ),
  kelsey = 160 →
  kelsey = ekon + 43 →
  uma = ekon + 17 →
  total_videos kelsey ekon uma = 411 :=
by
  sorry

end friends_video_count_l2498_249865


namespace susan_third_turn_move_l2498_249834

/-- A board game with the following properties:
  * The game board has 48 spaces from start to finish
  * A player moves 8 spaces forward on the first turn
  * On the second turn, the player moves 2 spaces forward but then 5 spaces backward
  * After the third turn, the player needs to move 37 more spaces to win
-/
structure BoardGame where
  total_spaces : Nat
  first_turn_move : Nat
  second_turn_forward : Nat
  second_turn_backward : Nat
  spaces_left_after_third_turn : Nat

/-- The specific game Susan is playing -/
def susans_game : BoardGame :=
  { total_spaces := 48
  , first_turn_move := 8
  , second_turn_forward := 2
  , second_turn_backward := 5
  , spaces_left_after_third_turn := 37 }

/-- Calculate the number of spaces moved on the third turn -/
def third_turn_move (game : BoardGame) : Nat :=
  game.total_spaces -
  (game.first_turn_move + game.second_turn_forward - game.second_turn_backward) -
  game.spaces_left_after_third_turn

/-- Theorem: Susan moved 6 spaces on the third turn -/
theorem susan_third_turn_move :
  third_turn_move susans_game = 6 := by
  sorry

end susan_third_turn_move_l2498_249834


namespace jumping_contest_l2498_249846

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) : 
  grasshopper_jump = 25 →
  frog_jump = grasshopper_jump + 32 →
  mouse_jump = frog_jump - 26 →
  mouse_jump = 31 := by
  sorry

end jumping_contest_l2498_249846


namespace smallest_positive_multiple_of_45_l2498_249817

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l2498_249817


namespace max_candy_pieces_l2498_249807

theorem max_candy_pieces (n : ℕ) (mean : ℕ) (min_pieces : ℕ) :
  n = 25 →
  mean = 7 →
  min_pieces = 2 →
  ∃ (max_pieces : ℕ),
    max_pieces = n * mean - (n - 1) * min_pieces ∧
    max_pieces = 127 :=
by sorry

end max_candy_pieces_l2498_249807


namespace sum_of_squares_cubic_roots_l2498_249860

theorem sum_of_squares_cubic_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 5 * p - 7 = 0) →
  (3 * q^3 - 2 * q^2 + 5 * q - 7 = 0) →
  (3 * r^3 - 2 * r^2 + 5 * r - 7 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
sorry

end sum_of_squares_cubic_roots_l2498_249860


namespace gala_dinner_seating_l2498_249828

/-- The number of couples to be seated -/
def num_couples : ℕ := 6

/-- The total number of people to be seated -/
def total_people : ℕ := 2 * num_couples

/-- The number of ways to arrange the husbands -/
def husband_arrangements : ℕ := (total_people - 1).factorial

/-- The number of equivalent arrangements due to rotation and reflection -/
def equivalent_arrangements : ℕ := 2 * total_people

/-- The number of unique seating arrangements -/
def unique_arrangements : ℕ := husband_arrangements / equivalent_arrangements

theorem gala_dinner_seating :
  unique_arrangements = 5760 :=
sorry

end gala_dinner_seating_l2498_249828


namespace squirrel_difference_l2498_249804

def scotland_squirrels : ℕ := 120000
def scotland_percentage : ℚ := 3/4

theorem squirrel_difference : 
  scotland_squirrels - (scotland_squirrels / scotland_percentage - scotland_squirrels) = 80000 := by
  sorry

end squirrel_difference_l2498_249804


namespace circle_tangent_existence_l2498_249852

/-- A line in a 2D plane -/
structure Line2D where
  slope : ℝ
  intercept : ℝ

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle2D where
  center : Point2D
  radius : ℝ

/-- Check if a point is on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if a circle is tangent to a line at a point -/
def circleTangentToLineAtPoint (c : Circle2D) (l : Line2D) (p : Point2D) : Prop :=
  pointOnLine p l ∧
  (c.center.x - p.x) * l.slope + (c.center.y - p.y) = 0 ∧
  (c.center.x - p.x)^2 + (c.center.y - p.y)^2 = c.radius^2

theorem circle_tangent_existence
  (l : Line2D) (p : Point2D) (r : ℝ) 
  (h_positive : r > 0) 
  (h_on_line : pointOnLine p l) :
  ∃ (c1 c2 : Circle2D), 
    c1.radius = r ∧
    c2.radius = r ∧
    circleTangentToLineAtPoint c1 l p ∧
    circleTangentToLineAtPoint c2 l p ∧
    c1 ≠ c2 :=
  sorry

end circle_tangent_existence_l2498_249852


namespace right_triangle_hypotenuse_l2498_249890

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive side lengths
  a^2 + b^2 = c^2 ∧        -- Pythagorean theorem (right triangle)
  a + b + c = 40 ∧         -- perimeter condition
  (1/2) * a * b = 24 ∧     -- area condition
  c = 18.8 := by            -- hypotenuse length
  sorry


end right_triangle_hypotenuse_l2498_249890


namespace percentage_of_girls_l2498_249826

theorem percentage_of_girls (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 900 →
  boys = 90 →
  girls = total - boys →
  (girls : ℚ) / (total : ℚ) * 100 = 90 := by
sorry

end percentage_of_girls_l2498_249826


namespace equation_solution_l2498_249825

theorem equation_solution : ∃! x : ℝ, (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ x = -9 := by
  sorry

end equation_solution_l2498_249825


namespace line_difference_l2498_249882

/-- Represents a character in the script --/
structure Character where
  lines : ℕ

/-- Represents the script with three characters --/
structure Script where
  char1 : Character
  char2 : Character
  char3 : Character

/-- Theorem: The difference in lines between the first and second character is 8 --/
theorem line_difference (s : Script) : 
  s.char3.lines = 2 →
  s.char2.lines = 3 * s.char3.lines + 6 →
  s.char1.lines = 20 →
  s.char1.lines - s.char2.lines = 8 := by
  sorry


end line_difference_l2498_249882


namespace factor_difference_of_squares_l2498_249891

theorem factor_difference_of_squares (t : ℝ) : 4 * t^2 - 81 = (2*t - 9) * (2*t + 9) := by
  sorry

end factor_difference_of_squares_l2498_249891


namespace douglas_fir_price_l2498_249867

theorem douglas_fir_price (total_trees : ℕ) (douglas_fir_count : ℕ) (ponderosa_price : ℕ) (total_paid : ℕ) :
  total_trees = 850 →
  douglas_fir_count = 350 →
  ponderosa_price = 225 →
  total_paid = 217500 →
  ∃ (douglas_price : ℕ),
    douglas_price * douglas_fir_count + ponderosa_price * (total_trees - douglas_fir_count) = total_paid ∧
    douglas_price = 300 :=
by sorry

end douglas_fir_price_l2498_249867


namespace factor_expression_l2498_249856

theorem factor_expression (y : ℝ) : 5 * y * (y - 2) + 9 * (y - 2) = (y - 2) * (5 * y + 9) := by
  sorry

end factor_expression_l2498_249856


namespace general_term_formula_l2498_249863

/-- The sequence defined by the problem -/
def a (n : ℕ+) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 3/4
  else if n = 3 then 5/9
  else if n = 4 then 7/16
  else (2*n - 1) / (n^2)

/-- The theorem stating that the general term formula is correct -/
theorem general_term_formula (n : ℕ+) : a n = (2*n - 1) / (n^2) := by
  sorry

end general_term_formula_l2498_249863


namespace min_moves_ten_elements_l2498_249802

/-- Represents a circular arrangement of n distinct elements -/
def CircularArrangement (n : ℕ) := Fin n → ℕ

/-- A single move in the circular arrangement -/
def Move (n : ℕ) (arr : CircularArrangement n) (i j : Fin n) : CircularArrangement n :=
  sorry

/-- Checks if the arrangement is sorted in ascending order -/
def IsSorted (n : ℕ) (arr : CircularArrangement n) : Prop :=
  sorry

/-- The minimum number of moves required to sort the arrangement -/
def MinMoves (n : ℕ) (arr : CircularArrangement n) : ℕ :=
  sorry

/-- Theorem: The minimum number of moves to sort 10 distinct elements in a circle is 8 -/
theorem min_moves_ten_elements :
  ∀ (arr : CircularArrangement 10), MinMoves 10 arr = 8 :=
  sorry

end min_moves_ten_elements_l2498_249802


namespace soldiers_food_calculation_l2498_249823

/-- Given the following conditions:
    1. Soldiers on the second side are given 2 pounds less food than the first side.
    2. The first side has 4000 soldiers.
    3. The second side has 500 soldiers fewer than the first side.
    4. The total amount of food both sides are eating altogether every day is 68000 pounds.

    Prove that the amount of food each soldier on the first side needs every day is 10 pounds. -/
theorem soldiers_food_calculation (food_first : ℝ) : 
  (4000 : ℝ) * food_first + (4000 - 500) * (food_first - 2) = 68000 → food_first = 10 := by
  sorry

end soldiers_food_calculation_l2498_249823


namespace smallest_b_value_l2498_249827

theorem smallest_b_value (a b c : ℚ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : 2 * b = a + c)  -- arithmetic sequence condition
  (h4 : c^2 = a * b)    -- geometric sequence condition
  : b ≥ (1/2 : ℚ) ∧ ∃ (a' b' c' : ℚ), 
    a' < b' ∧ b' < c' ∧ 
    2 * b' = a' + c' ∧ 
    c'^2 = a' * b' ∧ 
    b' = (1/2 : ℚ) := by
  sorry

end smallest_b_value_l2498_249827


namespace female_teachers_count_l2498_249881

/-- The number of teachers in the group -/
def total_teachers : ℕ := 5

/-- The probability of selecting a female teacher -/
def prob_female : ℚ := 7/10

/-- Calculates the number of combinations of k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates the probability of selecting two male teachers given x female teachers -/
def prob_two_male (x : ℕ) : ℚ :=
  1 - (choose (total_teachers - x) 2 : ℚ) / (choose total_teachers 2 : ℚ)

theorem female_teachers_count :
  ∃ x : ℕ, x ≤ total_teachers ∧ prob_two_male x = 1 - prob_female :=
sorry

end female_teachers_count_l2498_249881


namespace order_of_variables_l2498_249809

theorem order_of_variables (a b c d : ℝ) 
  (h1 : a > b) 
  (h2 : d > c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  b < c ∧ c < a ∧ a < d :=
sorry

end order_of_variables_l2498_249809


namespace fourth_selected_id_is_16_l2498_249899

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  totalStudents : Nat
  numTickets : Nat
  selectedIDs : Fin 3 → Nat

/-- Calculates the sampling interval for a given systematic sampling -/
def samplingInterval (s : SystematicSampling) : Nat :=
  s.totalStudents / s.numTickets

/-- Checks if a given ID is part of the systematic sampling -/
def isSelectedID (s : SystematicSampling) (id : Nat) : Prop :=
  ∃ k : Fin s.numTickets, id = (s.selectedIDs 0) + k * samplingInterval s

/-- Theorem: Given the conditions, the fourth selected ID is 16 -/
theorem fourth_selected_id_is_16 (s : SystematicSampling) 
  (h1 : s.totalStudents = 54)
  (h2 : s.numTickets = 4)
  (h3 : s.selectedIDs 0 = 3)
  (h4 : s.selectedIDs 1 = 29)
  (h5 : s.selectedIDs 2 = 42)
  : ∃ id : Nat, id = 16 ∧ isSelectedID s id :=
by
  sorry

end fourth_selected_id_is_16_l2498_249899


namespace tangent_line_equation_l2498_249806

/-- The parabola -/
def parabola (x : ℝ) : ℝ := x^2

/-- The line parallel to the tangent line -/
def parallel_line (x y : ℝ) : Prop := 2*x - y + 4 = 0

/-- The proposed tangent line -/
def tangent_line (x y : ℝ) : Prop := 2*x - y - 1 = 0

/-- Theorem: The tangent line to the parabola y = x^2 that is parallel to 2x - y + 4 = 0 
    has the equation 2x - y - 1 = 0 -/
theorem tangent_line_equation : 
  ∃ (x₀ y₀ : ℝ), 
    y₀ = parabola x₀ ∧ 
    tangent_line x₀ y₀ ∧
    ∀ (x y : ℝ), parallel_line x y ↔ ∃ (k : ℝ), y = k*x + (k*2 - 4) :=
sorry

end tangent_line_equation_l2498_249806


namespace expand_and_simplify_powers_of_two_one_more_than_cube_l2498_249892

-- Part (i)
theorem expand_and_simplify (x : ℝ) : (x + 1) * (x^2 - x + 1) = x^3 + 1 := by sorry

-- Part (ii)
theorem powers_of_two_one_more_than_cube : 
  {n : ℕ | ∃ k : ℕ, 2^n = k^3 + 1} = {0, 1} := by sorry

end expand_and_simplify_powers_of_two_one_more_than_cube_l2498_249892


namespace expression_evaluation_l2498_249830

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 13) + 1 = -x^4 + 3*x^3 - 5*x^2 + 13*x + 1 := by
  sorry

end expression_evaluation_l2498_249830


namespace cube_sum_from_sum_and_square_sum_l2498_249858

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by
sorry

end cube_sum_from_sum_and_square_sum_l2498_249858


namespace function_properties_l2498_249864

/-- The function f(x) -/
def f (a b : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

theorem function_properties (a b : ℝ) :
  (∀ x : ℝ, f' a b x = f' a b (-1 - x)) →  -- f'(x) is symmetric about x = -1/2
  f' a b 1 = 0 →                           -- f'(1) = 0
  a = 3 ∧ b = -12 ∧                        -- values of a and b
  f a b (-2) = 21 ∧ f a b 1 = -6           -- extreme values
  := by sorry

end function_properties_l2498_249864


namespace bexy_bicycle_speed_l2498_249803

/-- Bexy's round trip problem -/
theorem bexy_bicycle_speed :
  -- Bexy's walking distance and time
  let bexy_walk_distance : ℝ := 5
  let bexy_walk_time : ℝ := 1

  -- Ben's total round trip time in hours
  let ben_total_time : ℝ := 160 / 60

  -- Ben's speed relative to Bexy's average speed
  let ben_speed_ratio : ℝ := 1 / 2

  -- Bexy's bicycle speed
  ∃ bexy_bike_speed : ℝ,
    -- Ben's walking time is twice Bexy's
    let ben_walk_time : ℝ := 2 * bexy_walk_time

    -- Ben's biking time
    let ben_bike_time : ℝ := ben_total_time - ben_walk_time

    -- Ben's biking speed
    let ben_bike_speed : ℝ := bexy_bike_speed * ben_speed_ratio

    -- Distance traveled equals speed times time
    ben_bike_speed * ben_bike_time = bexy_walk_distance ∧
    bexy_bike_speed = 15 / 2 := by
  sorry

end bexy_bicycle_speed_l2498_249803


namespace triangle_angle_properties_l2498_249872

theorem triangle_angle_properties (α : Real) (h1 : 0 < α) (h2 : α < π) 
  (h3 : Real.sin α + Real.cos α = 1/5) : 
  (Real.tan α = -4/3) ∧ 
  ((Real.sin (3*π/2 + α) * Real.sin (π/2 - α) * Real.tan (π - α)^3) / 
   (Real.cos (π/2 + α) * Real.cos (3*π/2 - α)) = -4/3) := by
  sorry


end triangle_angle_properties_l2498_249872


namespace polynomial_division_remainder_l2498_249854

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 + 1 = (x^2 - 4*x + 7) * q + (8*x - 62) := by sorry

end polynomial_division_remainder_l2498_249854


namespace al_investment_l2498_249824

/-- Represents the investment scenario with four participants -/
structure Investment where
  al : ℝ
  betty : ℝ
  clare : ℝ
  dave : ℝ

/-- Defines the conditions of the investment problem -/
def valid_investment (i : Investment) : Prop :=
  i.al > 0 ∧ i.betty > 0 ∧ i.clare > 0 ∧ i.dave > 0 ∧  -- Each begins with a positive amount
  i.al ≠ i.betty ∧ i.al ≠ i.clare ∧ i.al ≠ i.dave ∧    -- Each begins with a different amount
  i.betty ≠ i.clare ∧ i.betty ≠ i.dave ∧ i.clare ≠ i.dave ∧
  i.al + i.betty + i.clare + i.dave = 2000 ∧           -- Total initial investment
  (i.al - 150) + (3 * i.betty) + (3 * i.clare) + (i.dave - 50) = 2500  -- Total after one year

/-- Theorem stating that under the given conditions, Al's original portion was 450 -/
theorem al_investment (i : Investment) (h : valid_investment i) : i.al = 450 := by
  sorry


end al_investment_l2498_249824


namespace work_completion_time_l2498_249878

theorem work_completion_time (a b c : ℝ) 
  (h1 : a + b + c = 1/4)  -- Combined work rate
  (h2 : a = 1/12)         -- a's work rate
  (h3 : b = 1/18)         -- b's work rate
  : c = 1/9 :=            -- c's work rate (to be proved)
by sorry

end work_completion_time_l2498_249878


namespace f_strictly_increasing_iff_a_in_range_l2498_249887

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a^2 - 1) * Real.exp (a * x)

-- State the theorem
theorem f_strictly_increasing_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, (deriv (f a)) x > 0) ↔ (1 < a ∧ a ≤ Real.sqrt 2) :=
sorry

end f_strictly_increasing_iff_a_in_range_l2498_249887


namespace solve_tangerines_l2498_249896

def tangerines_problem (initial_eaten : ℕ) (later_eaten : ℕ) : Prop :=
  ∃ (total : ℕ), 
    (initial_eaten + later_eaten = total) ∧ 
    (total - initial_eaten - later_eaten = 0)

theorem solve_tangerines : tangerines_problem 10 6 := by
  sorry

end solve_tangerines_l2498_249896


namespace inequality_solution_l2498_249832

-- Define the function f
def f (x : ℝ) := x^2 - x - 6

-- State the theorem
theorem inequality_solution :
  (∀ x : ℝ, f x = 0 ↔ x = -2 ∨ x = 3) →
  (∀ x : ℝ, -6 * f (-x) > 0 ↔ -3 < x ∧ x < 2) :=
by sorry

end inequality_solution_l2498_249832


namespace salad_cost_l2498_249869

/-- The cost of the salad given breakfast and lunch costs -/
theorem salad_cost (muffin_cost coffee_cost soup_cost lemonade_cost : ℝ)
  (h1 : muffin_cost = 2)
  (h2 : coffee_cost = 4)
  (h3 : soup_cost = 3)
  (h4 : lemonade_cost = 0.75)
  (h5 : muffin_cost + coffee_cost + 3 = soup_cost + lemonade_cost + (muffin_cost + coffee_cost)) :
  soup_cost + lemonade_cost + 3 - (soup_cost + lemonade_cost) = 5.25 := by
  sorry

end salad_cost_l2498_249869


namespace trigonometric_identities_l2498_249848

theorem trigonometric_identities :
  (∃ x : ℝ, x = 75 * π / 180 ∧ (Real.cos x)^2 = (2 - Real.sqrt 3) / 4) ∧
  (∃ y z : ℝ, y = π / 180 ∧ z = 44 * π / 180 ∧
    Real.tan y + Real.tan z + Real.tan y * Real.tan z = 1) := by
  sorry

end trigonometric_identities_l2498_249848


namespace roots_on_circle_l2498_249880

theorem roots_on_circle : ∃ (r : ℝ), r = 2 / Real.sqrt 3 ∧
  ∀ (z : ℂ), (z + 2)^4 = 16 * z^4 →
  ∃ (c : ℂ), Complex.abs (z - c) = r :=
sorry

end roots_on_circle_l2498_249880


namespace quadratic_roots_property_l2498_249845

theorem quadratic_roots_property (m n : ℝ) : 
  (∀ x, x^2 - 3*x + 1 = 0 ↔ x = m ∨ x = n) →
  -m - n - m*n = -4 := by
sorry

end quadratic_roots_property_l2498_249845


namespace investment_ratio_from_profit_ratio_and_time_l2498_249871

/-- Given two partners p and q, proves that if the ratio of their profits is 7:10,
    p invests for 7 months, and q invests for 14 months,
    then the ratio of their investments is 7:5. -/
theorem investment_ratio_from_profit_ratio_and_time (p q : ℝ) :
  (p * 7) / (q * 14) = 7 / 10 → p / q = 7 / 5 := by
  sorry

end investment_ratio_from_profit_ratio_and_time_l2498_249871


namespace fraction_equality_l2498_249837

theorem fraction_equality (a b c : ℝ) :
  (3 * a^2 + 3 * b^2 - 5 * c^2 + 6 * a * b) / (4 * a^2 + 4 * c^2 - 6 * b^2 + 8 * a * c) =
  ((a + b + Real.sqrt (5 * c^2)) * (a + b - Real.sqrt (5 * c^2))) /
  ((2 * (a + c) + Real.sqrt (6 * b^2)) * (2 * (a + c) - Real.sqrt (6 * b^2))) :=
by sorry

end fraction_equality_l2498_249837


namespace unique_divisible_by_29_l2498_249895

/-- Converts a base 7 number of the form 34x1 to decimal --/
def base7ToDecimal (x : ℕ) : ℕ := 3 * 7^3 + 4 * 7^2 + x * 7 + 1

/-- Checks if a number is divisible by 29 --/
def isDivisibleBy29 (n : ℕ) : Prop := n % 29 = 0

theorem unique_divisible_by_29 :
  ∃! x : ℕ, x < 7 ∧ isDivisibleBy29 (base7ToDecimal x) :=
sorry

end unique_divisible_by_29_l2498_249895


namespace damaged_chair_percentage_is_40_l2498_249868

/-- Represents the number of office chairs initially -/
def initial_chairs : ℕ := 80

/-- Represents the number of legs each chair has -/
def legs_per_chair : ℕ := 5

/-- Represents the number of round tables -/
def tables : ℕ := 20

/-- Represents the number of legs each table has -/
def legs_per_table : ℕ := 3

/-- Represents the total number of legs remaining after damage -/
def remaining_legs : ℕ := 300

/-- Calculates the percentage of chairs damaged and disposed of -/
def damaged_chair_percentage : ℚ :=
  let total_initial_legs := initial_chairs * legs_per_chair + tables * legs_per_table
  let disposed_legs := total_initial_legs - remaining_legs
  let disposed_chairs := disposed_legs / legs_per_chair
  (disposed_chairs : ℚ) / initial_chairs * 100

/-- Theorem stating that the percentage of chairs damaged and disposed of is 40% -/
theorem damaged_chair_percentage_is_40 :
  damaged_chair_percentage = 40 := by sorry

end damaged_chair_percentage_is_40_l2498_249868
