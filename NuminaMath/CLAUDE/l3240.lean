import Mathlib

namespace reciprocal_of_negative_2023_l3240_324093

theorem reciprocal_of_negative_2023 : 
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end reciprocal_of_negative_2023_l3240_324093


namespace max_value_product_sum_l3240_324035

theorem max_value_product_sum (A M C : ℕ) (sum_constraint : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 ∧
  ∃ (A' M' C' : ℕ), A' + M' + C' = 15 ∧ A' * M' * C' + A' * M' + M' * C' + C' * A' = 200 :=
by sorry

end max_value_product_sum_l3240_324035


namespace parabola_point_property_l3240_324087

/-- Given a parabola y = a(x+3)^2 + c with two points (x₁, y₁) and (x₂, y₂),
    if |x₁+3| > |x₂+3|, then a(y₁-y₂) > 0 -/
theorem parabola_point_property (a c x₁ y₁ x₂ y₂ : ℝ) :
  y₁ = a * (x₁ + 3)^2 + c →
  y₂ = a * (x₂ + 3)^2 + c →
  |x₁ + 3| > |x₂ + 3| →
  a * (y₁ - y₂) > 0 := by
  sorry

end parabola_point_property_l3240_324087


namespace product_327_3_base9_l3240_324074

/-- Represents a number in base 9 --/
def Base9 := ℕ

/-- Converts a base 9 number to a natural number --/
def to_nat (x : Base9) : ℕ := sorry

/-- Converts a natural number to a base 9 number --/
def from_nat (x : ℕ) : Base9 := sorry

/-- Multiplies two base 9 numbers --/
def mul_base9 (x y : Base9) : Base9 := sorry

theorem product_327_3_base9 : 
  mul_base9 (from_nat 327) (from_nat 3) = from_nat 1083 := by sorry

end product_327_3_base9_l3240_324074


namespace fraction_value_l3240_324098

theorem fraction_value (a b c d : ℝ) 
  (ha : a = 4 * b) 
  (hb : b = 3 * c) 
  (hc : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end fraction_value_l3240_324098


namespace helga_shoe_shopping_l3240_324050

theorem helga_shoe_shopping (first_store : ℕ) (second_store : ℕ) (third_store : ℕ) :
  first_store = 7 →
  second_store = first_store + 2 →
  third_store = 0 →
  let total_first_three := first_store + second_store + third_store
  let fourth_store := 2 * total_first_three
  first_store + second_store + third_store + fourth_store = 48 :=
by sorry

end helga_shoe_shopping_l3240_324050


namespace job_completion_time_l3240_324036

/-- The number of days it takes for B to do the job alone -/
def B_days : ℕ := 30

/-- The number of days it takes for A and B to do 4 times the job together -/
def AB_days : ℕ := 72

/-- The number of days it takes for A to do the job alone -/
def A_days : ℕ := 45

theorem job_completion_time :
  (1 : ℚ) / A_days + (1 : ℚ) / B_days = 4 / AB_days :=
sorry

end job_completion_time_l3240_324036


namespace exactly_two_propositions_true_l3240_324017

-- Define the propositions
def proposition1 : Prop := ∀ x : ℝ, (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ¬(∀ x : ℝ, x^2 + x + 1 ≥ 0)
def proposition2 : Prop := (∀ x y : ℝ, x + y = 0 → (x = -y)) ↔ (∀ x y : ℝ, x = -y → x + y = 0)

-- Theorem statement
theorem exactly_two_propositions_true : 
  (proposition1 = true) ∧ (proposition2 = true) ∧
  (¬ proposition1 = false) ∧ (¬ proposition2 = false) :=
sorry

end exactly_two_propositions_true_l3240_324017


namespace remainder_of_power_sum_l3240_324034

/-- The remainder when 5^94 + 7^94 is divided by 55 is 29. -/
theorem remainder_of_power_sum : (5^94 + 7^94) % 55 = 29 := by
  sorry

end remainder_of_power_sum_l3240_324034


namespace probability_of_specific_selection_l3240_324060

/-- A bag containing balls of different colors -/
structure BagOfBalls where
  total : ℕ
  white : ℕ
  red : ℕ
  black : ℕ

/-- The probability of selecting balls with specific conditions -/
def probability_of_selection (bag : BagOfBalls) (selected : ℕ) : ℚ :=
  sorry

/-- The main theorem to be proved -/
theorem probability_of_specific_selection : 
  let bag : BagOfBalls := ⟨20, 9, 5, 6⟩
  probability_of_selection bag 10 = 7 / 92378 := by
  sorry

end probability_of_specific_selection_l3240_324060


namespace students_on_left_side_l3240_324042

theorem students_on_left_side (total : ℕ) (right : ℕ) (h1 : total = 63) (h2 : right = 27) :
  total - right = 36 := by
  sorry

end students_on_left_side_l3240_324042


namespace f_decreasing_interval_l3240_324031

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x^2 + 3*x - 4

-- Define the derivative of f(x+1)
def f'_shifted (x : ℝ) : ℝ := (x + 1)^2 + 3*(x + 1) - 4

-- Theorem statement
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo (-5 : ℝ) 0, f'_shifted x < 0 :=
sorry

end f_decreasing_interval_l3240_324031


namespace fraction_subtraction_l3240_324078

theorem fraction_subtraction : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by sorry

end fraction_subtraction_l3240_324078


namespace remainder_theorem_polynomial_remainder_l3240_324055

def f (x : ℝ) : ℝ := 5*x^4 - 12*x^3 + 3*x^2 - 8*x + 15

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := sorry

theorem polynomial_remainder : f 4 = 543 := by sorry

end remainder_theorem_polynomial_remainder_l3240_324055


namespace saree_pricing_l3240_324057

/-- Calculates the final price of a saree given the original price and discount options --/
def calculate_final_price (original_price : ℚ) : ℚ × ℚ × ℚ := by
  -- Define the discount options
  let option_a : ℚ := (original_price * (1 - 0.18) - 100) * (1 - 0.05) * (1 + 0.0325) + 50
  let option_b : ℚ := original_price * (1 - 0.25) * (1 + 0.0275) * (1 + 0.0175)
  let option_c : ℚ := original_price * (1 - 0.12) * (1 - 0.06) * (1 + 0.035) * (1 + 0.0225)
  
  exact (option_a, option_b, option_c)

/-- Theorem stating the final prices for each option --/
theorem saree_pricing (original_price : ℚ) :
  original_price = 1200 →
  let (price_a, price_b, price_c) := calculate_final_price original_price
  price_a = 917.09 ∧ price_b = 940.93 ∧ price_c = 1050.50 := by
  sorry

end saree_pricing_l3240_324057


namespace sum_of_ages_l3240_324071

theorem sum_of_ages (marie_age marco_age : ℕ) : 
  marie_age = 12 → 
  marco_age = 2 * marie_age + 1 → 
  marie_age + marco_age = 37 := by
sorry

end sum_of_ages_l3240_324071


namespace E_parity_l3240_324005

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => E (n + 2) + E (n + 1)

def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def isOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem E_parity : isEven (E 2023) ∧ isOdd (E 2024) ∧ isOdd (E 2025) := by sorry

end E_parity_l3240_324005


namespace gcd_8164_2937_l3240_324004

theorem gcd_8164_2937 : Nat.gcd 8164 2937 = 1 := by
  sorry

end gcd_8164_2937_l3240_324004


namespace jennifer_initial_oranges_l3240_324062

/-- The number of fruits Jennifer has initially and after giving some away. -/
structure FruitCount where
  initial_pears : ℕ
  initial_apples : ℕ
  initial_oranges : ℕ
  pears_left : ℕ
  apples_left : ℕ
  oranges_left : ℕ
  total_left : ℕ

/-- Theorem stating the number of oranges Jennifer had initially. -/
theorem jennifer_initial_oranges (f : FruitCount) 
  (h1 : f.initial_pears = 10)
  (h2 : f.initial_apples = 2 * f.initial_pears)
  (h3 : f.pears_left = f.initial_pears - 2)
  (h4 : f.apples_left = f.initial_apples - 2)
  (h5 : f.oranges_left = f.initial_oranges - 2)
  (h6 : f.total_left = 44)
  (h7 : f.total_left = f.pears_left + f.apples_left + f.oranges_left) :
  f.initial_oranges = 20 := by
  sorry


end jennifer_initial_oranges_l3240_324062


namespace chord_length_theorem_l3240_324020

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c2.radius - c1.radius)^2

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem chord_length_theorem (c1 c2 c3 : Circle) 
  (h1 : c1.radius = 6)
  (h2 : c2.radius = 12)
  (h3 : are_externally_tangent c1 c2)
  (h4 : is_internally_tangent c1 c3)
  (h5 : is_internally_tangent c2 c3)
  (h6 : are_collinear c1.center c2.center c3.center) :
  ∃ (chord_length : ℝ), 
    chord_length = (144 * Real.sqrt 26) / 5 ∧ 
    chord_length^2 = 4 * (c3.radius^2 - (c3.radius - c1.radius - c2.radius)^2) := by
  sorry

end chord_length_theorem_l3240_324020


namespace valid_three_digit_count_l3240_324029

/-- The count of three-digit numbers without exactly two identical adjacent digits -/
def valid_three_digit_numbers : ℕ := 738

/-- The total count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with exactly two identical adjacent digits -/
def invalid_three_digit_numbers : ℕ := 162

theorem valid_three_digit_count :
  valid_three_digit_numbers = total_three_digit_numbers - invalid_three_digit_numbers :=
by sorry

end valid_three_digit_count_l3240_324029


namespace cos420_plus_sin330_eq_zero_l3240_324099

theorem cos420_plus_sin330_eq_zero :
  Real.cos (420 * π / 180) + Real.sin (330 * π / 180) = 0 := by
  sorry

end cos420_plus_sin330_eq_zero_l3240_324099


namespace special_triangle_sides_l3240_324047

/-- A triangle with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℕ+
  b : ℕ+
  c : ℕ+
  -- The perimeter is a natural number (implied by sides being natural numbers)
  perimeter_nat : (a + b + c : ℕ) > 0
  -- The circumradius is 65/8
  circumradius_eq : (a * b * c : ℚ) / (4 * (a + b + c : ℚ)) = 65 / 8
  -- The inradius is 4
  inradius_eq : (a * b * c : ℚ) / ((a + b + c : ℚ) * (a + b + c - 2 * min a (min b c))) = 4

/-- The sides of the special triangle are (13, 14, 15) -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 13 ∧ t.b = 14 ∧ t.c = 15 := by
  sorry


end special_triangle_sides_l3240_324047


namespace percentage_of_number_l3240_324022

theorem percentage_of_number (x : ℝ) (h : x = 16) : x * 0.0025 = 0.04 := by
  sorry

end percentage_of_number_l3240_324022


namespace right_triangle_hypotenuse_l3240_324003

theorem right_triangle_hypotenuse (a b c : ℕ) : 
  a * a + b * b = c * c →  -- Pythagorean theorem
  c - b = 1575 →           -- One leg is 1575 units shorter than hypotenuse
  a < 1991 →               -- The other leg is less than 1991 units
  c = 1799 :=              -- The hypotenuse length is 1799
by sorry

end right_triangle_hypotenuse_l3240_324003


namespace xyz_equals_four_l3240_324001

theorem xyz_equals_four (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 4 := by
  sorry

end xyz_equals_four_l3240_324001


namespace cube_sum_plus_three_l3240_324045

theorem cube_sum_plus_three (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 3 = 973 := by
  sorry

end cube_sum_plus_three_l3240_324045


namespace correct_performance_calculation_l3240_324041

/-- Represents a batsman's performance in a cricket match -/
structure BatsmanPerformance where
  initialAverage : ℝ
  eleventhInningRuns : ℝ
  averageIncrease : ℝ
  teamHandicap : ℝ

/-- Calculates the new average and total team runs for a batsman -/
def calculatePerformance (performance : BatsmanPerformance) : ℝ × ℝ :=
  let newAverage := performance.initialAverage + performance.averageIncrease
  let totalBatsmanRuns := 11 * newAverage
  let totalTeamRuns := totalBatsmanRuns + performance.teamHandicap
  (newAverage, totalTeamRuns)

/-- Theorem stating the correct calculation of a batsman's performance -/
theorem correct_performance_calculation 
  (performance : BatsmanPerformance) 
  (h1 : performance.eleventhInningRuns = 85)
  (h2 : performance.averageIncrease = 5)
  (h3 : performance.teamHandicap = 75) :
  calculatePerformance performance = (35, 460) := by
  sorry

#check correct_performance_calculation

end correct_performance_calculation_l3240_324041


namespace hyperbola_and_line_l3240_324075

/-- Hyperbola with center at origin, right focus at (2,0), and distance 1 from focus to asymptote -/
structure Hyperbola where
  center : ℝ × ℝ := (0, 0)
  right_focus : ℝ × ℝ := (2, 0)
  focus_to_asymptote : ℝ := 1

/-- Line that intersects the hyperbola at two distinct points -/
structure IntersectingLine where
  k : ℝ
  b : ℝ := 2

/-- Theorem about the hyperbola and its intersecting line -/
theorem hyperbola_and_line (C : Hyperbola) (l : IntersectingLine) :
  (∀ A B : ℝ × ℝ, A ≠ B → (A.1^2/3 - A.2^2 = 1 ∧ A.2 = l.k * A.1 + l.b) →
                        (B.1^2/3 - B.2^2 = 1 ∧ B.2 = l.k * B.1 + l.b) →
                        A.1 * B.1 + A.2 * B.2 > 2) →
  (∀ x y : ℝ, x^2/3 - y^2 = 1 ↔ C.center = (0, 0) ∧ C.right_focus = (2, 0) ∧ C.focus_to_asymptote = 1) ∧
  (l.k ∈ Set.Ioo (-Real.sqrt 15 / 3) (-Real.sqrt 3 / 3) ∪ Set.Ioo (Real.sqrt 3 / 3) (Real.sqrt 15 / 3)) :=
by sorry

end hyperbola_and_line_l3240_324075


namespace max_garden_area_l3240_324038

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ := d.length * d.width

/-- Calculates the perimeter of a rectangular garden with one side against a house -/
def gardenPerimeter (d : GardenDimensions) : ℝ := d.length + 2 * d.width

/-- The available fencing length -/
def availableFencing : ℝ := 400

theorem max_garden_area :
  ∃ (d : GardenDimensions),
    gardenPerimeter d = availableFencing ∧
    ∀ (d' : GardenDimensions),
      gardenPerimeter d' = availableFencing →
      gardenArea d' ≤ gardenArea d ∧
      gardenArea d = 20000 := by
  sorry

end max_garden_area_l3240_324038


namespace book_pages_calculation_l3240_324030

theorem book_pages_calculation (pages_per_night : ℝ) (nights : ℝ) (h1 : pages_per_night = 120.0) (h2 : nights = 10.0) :
  pages_per_night * nights = 1200.0 := by
  sorry

end book_pages_calculation_l3240_324030


namespace intersection_M_N_l3240_324088

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}

def N : Set ℝ := {-3, -2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} := by sorry

end intersection_M_N_l3240_324088


namespace route_length_l3240_324086

/-- Proves that the length of a route is 125 miles given the conditions of two trains meeting. -/
theorem route_length (time_A time_B meeting_distance : ℝ) 
  (h1 : time_A = 12)
  (h2 : time_B = 8)
  (h3 : meeting_distance = 50)
  (h4 : time_A > 0)
  (h5 : time_B > 0)
  (h6 : meeting_distance > 0) :
  ∃ (route_length : ℝ),
    route_length = 125 ∧
    route_length / time_A * (meeting_distance * time_A / route_length) = meeting_distance ∧
    route_length / time_B * (meeting_distance * time_A / route_length) = route_length - meeting_distance :=
by
  sorry


end route_length_l3240_324086


namespace trailing_zeros_of_main_expression_l3240_324007

/-- The number of trailing zeros in n -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- Prime factorization of 15 -/
def fifteen : ℕ := 3 * 5

/-- Prime factorization of 28 -/
def twentyEight : ℕ := 2^2 * 7

/-- Prime factorization of 55 -/
def fiftyFive : ℕ := 5 * 11

/-- The main expression -/
def mainExpression : ℕ := fifteen^6 * twentyEight^5 * fiftyFive^7

theorem trailing_zeros_of_main_expression :
  trailingZeros mainExpression = 10 := by sorry

end trailing_zeros_of_main_expression_l3240_324007


namespace square_root_sum_l3240_324054

theorem square_root_sum (x : ℝ) 
  (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) : 
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
  sorry

end square_root_sum_l3240_324054


namespace least_number_for_divisibility_l3240_324058

theorem least_number_for_divisibility (n m : ℕ) (h : n = 1056 ∧ m = 27) :
  ∃ x : ℕ, (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 ∧ x = 24 :=
sorry

end least_number_for_divisibility_l3240_324058


namespace weight_ratio_proof_l3240_324082

/-- Proves that the ratio of weight held in each hand to body weight is 1:1 --/
theorem weight_ratio_proof (body_weight hand_weight total_weight : ℝ) 
  (hw : body_weight = 150)
  (vest_weight : ℝ)
  (hv : vest_weight = body_weight / 2)
  (ht : total_weight = 525)
  (he : total_weight = body_weight + vest_weight + 2 * hand_weight) :
  hand_weight / body_weight = 1 := by
  sorry

end weight_ratio_proof_l3240_324082


namespace intersection_points_count_l3240_324067

/-- The number of intersection points between y = Bx^2 and y^2 + 4y - 2 = x^2 + 5y -/
theorem intersection_points_count (B : ℝ) (hB : B > 0) : 
  ∃ (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ),
    (y1 = B * x1^2 ∧ y1^2 + 4*y1 - 2 = x1^2 + 5*y1) ∧
    (y2 = B * x2^2 ∧ y2^2 + 4*y2 - 2 = x2^2 + 5*y2) ∧
    (y3 = B * x3^2 ∧ y3^2 + 4*y3 - 2 = x3^2 + 5*y3) ∧
    (y4 = B * x4^2 ∧ y4^2 + 4*y4 - 2 = x4^2 + 5*y4) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x1 ≠ x4 ∨ y1 ≠ y4) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3) ∧
    (x2 ≠ x4 ∨ y2 ≠ y4) ∧
    (x3 ≠ x4 ∨ y3 ≠ y4) ∧
    ∀ (x y : ℝ), (y = B * x^2 ∧ y^2 + 4*y - 2 = x^2 + 5*y) →
      ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3) ∨ (x = x4 ∧ y = y4)) :=
by sorry

end intersection_points_count_l3240_324067


namespace parallel_line_equation_perpendicular_line_equation_l3240_324039

-- Define the lines L1 and L2
def L1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def L2 (x y : ℝ) : Prop := x - 3 * y + 8 = 0

-- Define the reference line
def ref_line (x y : ℝ) : Prop := 2 * x + y + 5 = 0

-- Define the intersection point M
def M : ℝ × ℝ := ((-2 : ℝ), (2 : ℝ))

-- Theorem for the parallel line
theorem parallel_line_equation :
  ∀ (x y : ℝ), 2 * x + y + 2 = 0 →
  (L1 (M.1) (M.2) ∧ L2 (M.1) (M.2)) ∧
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), 2 * x + y + 2 = 0 ↔ k * (2 * x + y + 5) = 0 :=
sorry

-- Theorem for the perpendicular line
theorem perpendicular_line_equation :
  ∀ (x y : ℝ), x - 2 * y + 6 = 0 →
  (L1 (M.1) (M.2) ∧ L2 (M.1) (M.2)) ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (x - 2 * y + 6 = 0 → x₁ = x ∧ y₁ = y) →
    (2 * x + y + 5 = 0 → x₂ = x ∧ y₂ = y) →
    (x₂ - x₁) * (x - x₁) + (y₂ - y₁) * (y - y₁) = 0 :=
sorry

end parallel_line_equation_perpendicular_line_equation_l3240_324039


namespace sum_of_cubes_l3240_324011

theorem sum_of_cubes (x y z : ℕ+) : 
  (x + y + z : ℕ+)^3 - x^3 - y^3 - z^3 = 378 → x + y + z = 9 := by
  sorry

end sum_of_cubes_l3240_324011


namespace c_completion_time_l3240_324091

-- Define the work rates of A, B, and C
variable (A B C : ℝ)

-- Define the conditions
def condition1 : Prop := A + B = 1 / 15
def condition2 : Prop := A + B + C = 1 / 11

-- Theorem statement
theorem c_completion_time (h1 : condition1 A B) (h2 : condition2 A B C) :
  1 / C = 41.25 := by sorry

end c_completion_time_l3240_324091


namespace blueberry_zucchini_trade_l3240_324009

/-- The number of bushes needed to obtain a specific number of zucchinis -/
def bushes_needed (total_containers_per_bush : ℕ) (containers_for_jam : ℕ) 
                  (containers_per_trade : ℕ) (zucchinis_per_trade : ℕ) 
                  (target_zucchinis : ℕ) : ℕ :=
  let usable_containers := total_containers_per_bush - containers_for_jam
  let zucchinis_per_container := zucchinis_per_trade / containers_per_trade
  let zucchinis_per_bush := usable_containers * zucchinis_per_container
  target_zucchinis / zucchinis_per_bush

/-- Theorem stating that 18 bushes are needed to obtain 72 zucchinis under given conditions -/
theorem blueberry_zucchini_trade : bushes_needed 10 2 6 3 72 = 18 := by
  sorry

end blueberry_zucchini_trade_l3240_324009


namespace two_special_numbers_exist_l3240_324018

theorem two_special_numbers_exist : ∃ (x y : ℕ), 
  x + y = 2013 ∧ 
  y = 5 * ((x / 100) + 1) ∧ 
  x > y :=
by sorry

end two_special_numbers_exist_l3240_324018


namespace litter_patrol_collection_l3240_324083

/-- The number of glass bottles picked up by the Litter Patrol -/
def glass_bottles : ℕ := 10

/-- The number of aluminum cans picked up by the Litter Patrol -/
def aluminum_cans : ℕ := 8

/-- The total number of pieces of litter is the sum of glass bottles and aluminum cans -/
def total_litter : ℕ := glass_bottles + aluminum_cans

/-- Theorem stating that the total number of pieces of litter is 18 -/
theorem litter_patrol_collection : total_litter = 18 := by
  sorry

end litter_patrol_collection_l3240_324083


namespace parallel_lines_length_l3240_324051

-- Define the parallel lines and their lengths
def AB : ℝ := 210
def CD : ℝ := 140
def EF : ℝ := 84

-- Define the parallel relation
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

-- State the theorem
theorem parallel_lines_length :
  ∀ (ab gh cd ef : ℝ → ℝ → Prop),
    parallel ab gh → parallel gh cd → parallel cd ef →
    AB = 210 → CD = 140 →
    EF = 84 := by sorry

end parallel_lines_length_l3240_324051


namespace quadratic_points_theorem_l3240_324096

/-- Quadratic function -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + 2*m*x - 3

theorem quadratic_points_theorem (m n p q : ℝ) 
  (h_m : m > 0)
  (h_A : f m (n-2) = p)
  (h_B : f m 4 = q)
  (h_C : f m n = p)
  (h_q : -3 < q)
  (h_p : q < p) :
  (m = n - 1) ∧ ((3 < n ∧ n < 4) ∨ n > 6) := by
  sorry

end quadratic_points_theorem_l3240_324096


namespace solve_equation_l3240_324089

theorem solve_equation (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 8 → y = 40 / 3 := by
sorry

end solve_equation_l3240_324089


namespace dilation_example_l3240_324079

/-- Dilation of a complex number -/
def dilation (center scale : ℂ) (z : ℂ) : ℂ :=
  center + scale * (z - center)

theorem dilation_example : 
  dilation (2 - 3*I) (-2) (-1 + 2*I) = 8 - 13*I :=
by sorry

end dilation_example_l3240_324079


namespace factorial_sum_equality_l3240_324073

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end factorial_sum_equality_l3240_324073


namespace point_p_coordinates_l3240_324064

/-- A point in the fourth quadrant with specific distances from axes -/
structure PointP where
  x : ℝ
  y : ℝ
  in_fourth_quadrant : x > 0 ∧ y < 0
  distance_to_x_axis : |y| = 1
  distance_to_y_axis : |x| = 2

/-- The coordinates of point P are (2, -1) -/
theorem point_p_coordinates (p : PointP) : p.x = 2 ∧ p.y = -1 := by
  sorry

end point_p_coordinates_l3240_324064


namespace fourth_quarter_points_l3240_324061

def winning_team_points (q1 q2 q3 q4 : ℕ) : Prop :=
  q1 = 20 ∧ q2 = q1 + 10 ∧ q3 = q2 + 20 ∧ q1 + q2 + q3 + q4 = 80

theorem fourth_quarter_points :
  ∃ q1 q2 q3 q4 : ℕ,
    winning_team_points q1 q2 q3 q4 ∧ q4 = 30 := by
  sorry

end fourth_quarter_points_l3240_324061


namespace initial_overs_is_ten_l3240_324012

/-- Represents a cricket game scenario --/
structure CricketGame where
  target : ℕ
  initialRunRate : ℚ
  remainingOvers : ℕ
  requiredRunRate : ℚ

/-- Calculates the number of overs played initially in a cricket game --/
def initialOvers (game : CricketGame) : ℚ :=
  (game.target - game.remainingOvers * game.requiredRunRate) / game.initialRunRate

/-- Theorem stating that the number of overs played initially is 10 --/
theorem initial_overs_is_ten (game : CricketGame) 
  (h1 : game.target = 282)
  (h2 : game.initialRunRate = 16/5)
  (h3 : game.remainingOvers = 50)
  (h4 : game.requiredRunRate = 5)
  : initialOvers game = 10 := by
  sorry

#eval initialOvers { target := 282, initialRunRate := 16/5, remainingOvers := 50, requiredRunRate := 5 }

end initial_overs_is_ten_l3240_324012


namespace equation_implies_m_equals_zero_l3240_324070

theorem equation_implies_m_equals_zero (m n : ℝ) :
  21 * (m + n) + 21 = 21 * (-m + n) + 21 → m = 0 := by
sorry

end equation_implies_m_equals_zero_l3240_324070


namespace max_acute_triangles_formula_l3240_324044

/-- Represents a line with marked points -/
structure MarkedLine where
  points : Finset ℝ
  distinct : points.card = 50

/-- The maximum number of acute-angled triangles formed by points on two parallel lines -/
def max_acute_triangles (a b : MarkedLine) : ℕ :=
  (50^3 - 50) / 3

/-- Theorem stating the maximum number of acute-angled triangles -/
theorem max_acute_triangles_formula (a b : MarkedLine) (h : a.points ∩ b.points = ∅) :
  max_acute_triangles a b = 41650 :=
sorry

end max_acute_triangles_formula_l3240_324044


namespace part1_solution_set_part2_minimum_value_l3240_324043

-- Define the function f
def f (m n x : ℝ) : ℝ := |x - m| + |x - n|

-- Part 1
theorem part1_solution_set (x : ℝ) :
  (f 2 (-5) x > 9) ↔ (x < -6 ∨ x > 3) := by sorry

-- Part 2
theorem part2_minimum_value (a : ℝ) (h : a ≠ 0) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x : ℝ), f a (-1/a) x ≥ min := by sorry

end part1_solution_set_part2_minimum_value_l3240_324043


namespace inequality_solution_range_l3240_324065

theorem inequality_solution_range (m : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + m| > 3) ↔ (m > 2 ∨ m < -4) :=
sorry

end inequality_solution_range_l3240_324065


namespace roger_candies_left_l3240_324053

/-- The number of candies Roger has left after giving some away -/
def candies_left (initial : ℕ) (given_to_stephanie : ℕ) (given_to_john : ℕ) (given_to_emily : ℕ) : ℕ :=
  initial - (given_to_stephanie + given_to_john + given_to_emily)

/-- Theorem stating that Roger has 262 candies left -/
theorem roger_candies_left :
  candies_left 350 45 25 18 = 262 := by
  sorry

end roger_candies_left_l3240_324053


namespace f_comp_three_roots_l3240_324077

/-- The function f(x) = x^2 + 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Theorem stating that f(f(x)) has exactly 3 distinct real roots iff c = 1 - √13 -/
theorem f_comp_three_roots :
  ∀ c : ℝ, (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    f_comp c r₁ = 0 ∧ f_comp c r₂ = 0 ∧ f_comp c r₃ = 0) ↔ 
  c = 1 - Real.sqrt 13 :=
sorry

end f_comp_three_roots_l3240_324077


namespace intersection_equals_interval_l3240_324006

def S : Set ℝ := {x | (x - 2) * (x + 3) > 0}

def T : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x)}

theorem intersection_equals_interval : S ∩ T = Set.Ioo 2 3 ∪ Set.singleton 3 := by
  sorry

end intersection_equals_interval_l3240_324006


namespace custom_mult_seven_three_l3240_324081

/-- Custom multiplication operation -/
def custom_mult (a b : ℤ) : ℤ := 4*a + 5*b - a*b + 1

/-- Theorem stating that 7 * 3 = 23 under the custom multiplication -/
theorem custom_mult_seven_three : custom_mult 7 3 = 23 := by
  sorry

end custom_mult_seven_three_l3240_324081


namespace num_valid_selections_l3240_324023

/-- Represents the set of volunteers --/
inductive Volunteer
| A
| B
| C
| D
| E

/-- Represents the set of roles --/
inductive Role
| Translator
| TourGuide
| Etiquette
| Driver

/-- Predicate to check if a volunteer can take on a role --/
def canTakeRole (v : Volunteer) (r : Role) : Prop :=
  match v, r with
  | Volunteer.A, Role.Driver => False
  | Volunteer.B, Role.Driver => False
  | _, _ => True

/-- A selection is a function from Role to Volunteer --/
def Selection := Role → Volunteer

/-- Predicate to check if a selection is valid --/
def validSelection (s : Selection) : Prop :=
  (∀ r : Role, canTakeRole (s r) r) ∧
  (∀ v : Volunteer, ∃! r : Role, s r = v)

/-- The number of valid selections --/
def numValidSelections : ℕ := sorry

theorem num_valid_selections :
  numValidSelections = 72 := by sorry

end num_valid_selections_l3240_324023


namespace job_completion_time_l3240_324068

theorem job_completion_time (job : ℝ) (a_time : ℝ) (b_efficiency : ℝ) :
  job > 0 ∧ a_time = 15 ∧ b_efficiency = 1.8 →
  (job / (job / a_time * b_efficiency)) = 25 / 6 := by
  sorry

end job_completion_time_l3240_324068


namespace min_value_theorem_l3240_324097

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2) :
  x^2 / (2*y) + 4*y^2 / x ≥ 2 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2*y = 2 ∧ x^2 / (2*y) + 4*y^2 / x = 2 := by
  sorry

end min_value_theorem_l3240_324097


namespace sum_of_seven_step_palindromes_l3240_324072

/-- Reverses a natural number -/
def reverseNum (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Bool := sorry

/-- Performs one step of reversing and adding -/
def reverseAndAdd (n : ℕ) : ℕ := n + reverseNum n

/-- Checks if a number becomes a palindrome after exactly k steps -/
def isPalindromeAfterKSteps (n : ℕ) (k : ℕ) : Bool := sorry

/-- The set of three-digit numbers that become palindromes after exactly 7 steps -/
def sevenStepPalindromes : Finset ℕ := sorry

theorem sum_of_seven_step_palindromes :
  Finset.sum sevenStepPalindromes id = 1160 := by sorry

end sum_of_seven_step_palindromes_l3240_324072


namespace homework_problems_per_page_l3240_324027

theorem homework_problems_per_page 
  (total_problems : ℕ) 
  (solved_percentage : ℚ) 
  (remaining_pages : ℕ) 
  (h1 : total_problems = 550) 
  (h2 : solved_percentage = 65 / 100) 
  (h3 : remaining_pages = 3) : 
  (total_problems - Int.floor (solved_percentage * total_problems)) / remaining_pages = 64 := by
  sorry

end homework_problems_per_page_l3240_324027


namespace unique_solution_iff_a_nonpositive_l3240_324056

/-- The system of equations has at most one real solution if and only if a ≤ 0 -/
theorem unique_solution_iff_a_nonpositive (a : ℝ) :
  (∃! x y z : ℝ, x^4 = y*z - x^2 + a ∧ y^4 = z*x - y^2 + a ∧ z^4 = x*y - z^2 + a) ↔ a ≤ 0 := by
  sorry

end unique_solution_iff_a_nonpositive_l3240_324056


namespace shaded_region_perimeter_l3240_324063

/-- The perimeter of the shaded region formed by the segments where three identical touching circles intersect is equal to the circumference of one circle. -/
theorem shaded_region_perimeter (circle_circumference : ℝ) (segment_angle : ℝ) : 
  circle_circumference > 0 →
  segment_angle = 120 →
  (3 * (segment_angle / 360) * circle_circumference) = circle_circumference :=
by sorry

end shaded_region_perimeter_l3240_324063


namespace binomial_600_600_l3240_324002

theorem binomial_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end binomial_600_600_l3240_324002


namespace lisa_flight_time_l3240_324014

/-- 
Given that Lisa flew 500 miles at a speed of 45 miles per hour, 
prove that the time Lisa flew is equal to 500 miles divided by 45 miles per hour.
-/
theorem lisa_flight_time : 
  let distance : ℝ := 500  -- Distance in miles
  let speed : ℝ := 45      -- Speed in miles per hour
  let time : ℝ := distance / speed
  time = 500 / 45 := by sorry

end lisa_flight_time_l3240_324014


namespace excess_hour_cost_correct_l3240_324046

/-- The cost per hour in excess of 2 hours for a parking garage -/
def excess_hour_cost : ℝ := 1.75

/-- The cost to park for up to 2 hours -/
def initial_cost : ℝ := 15

/-- The average cost per hour to park for 9 hours -/
def average_cost_9_hours : ℝ := 3.0277777777777777

/-- Theorem stating that the excess hour cost is correct given the initial cost and average cost -/
theorem excess_hour_cost_correct : 
  (initial_cost + 7 * excess_hour_cost) / 9 = average_cost_9_hours :=
by sorry

end excess_hour_cost_correct_l3240_324046


namespace correct_travel_distance_l3240_324015

/-- The distance traveled by Gavril on the electric train -/
def travel_distance : ℝ := 257

/-- The time it takes for the smartphone to fully discharge while watching videos -/
def video_discharge_time : ℝ := 3

/-- The time it takes for the smartphone to fully discharge while playing Tetris -/
def tetris_discharge_time : ℝ := 5

/-- The speed of the train for the first half of the journey -/
def speed_first_half : ℝ := 80

/-- The speed of the train for the second half of the journey -/
def speed_second_half : ℝ := 60

/-- Theorem stating that given the conditions, the travel distance is correct -/
theorem correct_travel_distance :
  let total_time := (video_discharge_time * tetris_discharge_time) / (video_discharge_time / 2 + tetris_discharge_time / 2)
  travel_distance = total_time * (speed_first_half / 2 + speed_second_half / 2) :=
by sorry

end correct_travel_distance_l3240_324015


namespace difference_of_squares_601_599_l3240_324090

theorem difference_of_squares_601_599 : 601^2 - 599^2 = 2400 := by
  sorry

end difference_of_squares_601_599_l3240_324090


namespace last_two_digits_of_7_pow_2018_l3240_324028

theorem last_two_digits_of_7_pow_2018 : 7^2018 % 100 = 49 := by
  sorry

end last_two_digits_of_7_pow_2018_l3240_324028


namespace blue_pill_cost_l3240_324094

/-- Represents the cost of pills for Alice's medication --/
structure PillCosts where
  red : ℝ
  blue : ℝ
  yellow : ℝ

/-- The conditions of Alice's medication costs --/
def medication_conditions (costs : PillCosts) : Prop :=
  costs.blue = costs.red + 3 ∧
  costs.yellow = 2 * costs.red - 2 ∧
  21 * (costs.red + costs.blue + costs.yellow) = 924

/-- Theorem stating the cost of the blue pill --/
theorem blue_pill_cost (costs : PillCosts) :
  medication_conditions costs → costs.blue = 13.75 := by
  sorry


end blue_pill_cost_l3240_324094


namespace bridge_length_l3240_324076

/-- The length of a bridge given train parameters --/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 160)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  train_length + (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 215 :=
by sorry

end bridge_length_l3240_324076


namespace paper_stack_height_l3240_324085

/-- Given a ream of paper with 400 sheets that is 4 cm thick,
    prove that a stack of 6 cm will contain 600 sheets. -/
theorem paper_stack_height (sheets_per_ream : ℕ) (ream_thickness : ℝ) 
  (stack_height : ℝ) (h1 : sheets_per_ream = 400) (h2 : ream_thickness = 4) 
  (h3 : stack_height = 6) : 
  (stack_height / ream_thickness) * sheets_per_ream = 600 :=
sorry

end paper_stack_height_l3240_324085


namespace grid_path_theorem_l3240_324010

/-- Represents a closed path on a grid that is not self-intersecting -/
structure GridPath (m n : ℕ) where
  -- Add necessary fields to represent the path

/-- Counts the number of points on the path where it does not turn -/
def count_no_turn_points (p : GridPath m n) : ℕ := sorry

/-- Counts the number of squares that the path goes through two non-adjacent sides -/
def count_two_side_squares (p : GridPath m n) : ℕ := sorry

/-- Counts the number of squares with no side in the path -/
def count_empty_squares (p : GridPath m n) : ℕ := sorry

theorem grid_path_theorem {m n : ℕ} (hm : m ≥ 4) (hn : n ≥ 4) (p : GridPath m n) :
  count_no_turn_points p = count_two_side_squares p - count_empty_squares p + m + n - 1 := by
  sorry

end grid_path_theorem_l3240_324010


namespace problem_solution_l3240_324095

def f (x : ℝ) := abs (2*x - 1) - abs (2*x - 2)

theorem problem_solution :
  (∃ k : ℝ, ∀ x : ℝ, f x ≤ k) ∧
  ({x : ℝ | f x ≥ x} = {x : ℝ | x ≤ -1 ∨ x = 1}) ∧
  (∀ x : ℝ, f x ≤ 1) ∧
  (¬∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 2*b = 1 ∧ 2/a + 1/b = 4 - 1/(a*b)) := by
  sorry

end problem_solution_l3240_324095


namespace S_intersect_T_characterization_l3240_324026

def S : Set ℝ := {x | |x| < 5}
def T : Set ℝ := {x | x^2 + 4*x - 21 < 0}

theorem S_intersect_T_characterization :
  S ∩ T = {x : ℝ | -5 < x ∧ x < 3} := by sorry

end S_intersect_T_characterization_l3240_324026


namespace f_2006_equals_1_l3240_324025

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_2006_equals_1 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period (fun x ↦ f (3*x + 1)) 3)
  (h_f_1 : f 1 = -1) :
  f 2006 = 1 := by
sorry

end f_2006_equals_1_l3240_324025


namespace flour_for_nine_biscuits_l3240_324052

/-- The amount of flour needed to make a certain number of biscuits -/
def flour_needed (num_biscuits : ℕ) : ℝ :=
  sorry

theorem flour_for_nine_biscuits :
  let members : ℕ := 18
  let biscuits_per_member : ℕ := 2
  let total_flour : ℝ := 5
  flour_needed (members * biscuits_per_member) = total_flour →
  flour_needed 9 = 1.25 :=
by sorry

end flour_for_nine_biscuits_l3240_324052


namespace third_term_of_specific_arithmetic_sequence_l3240_324092

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem third_term_of_specific_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a1 : a 1 = 1) 
  (h_a2 : a 2 = 2) : 
  a 3 = 3 := by
sorry

end third_term_of_specific_arithmetic_sequence_l3240_324092


namespace solve_for_m_l3240_324013

theorem solve_for_m (Q t h m : ℝ) (hQ : Q > 0) (ht : t > 0) (hh : h ≥ 0) :
  Q = t / (1 + Real.sqrt h)^m ↔ m = Real.log (t / Q) / Real.log (1 + Real.sqrt h) :=
sorry

end solve_for_m_l3240_324013


namespace diamond_calculation_l3240_324040

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement
theorem diamond_calculation :
  (diamond (diamond 3 4) 2) - (diamond 3 (diamond 4 2)) = -13/28 := by
  sorry

end diamond_calculation_l3240_324040


namespace leahs_coins_value_l3240_324016

/-- Represents the number of coins Leah has -/
def total_coins : ℕ := 15

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents Leah's coin collection -/
structure CoinCollection where
  nickels : ℕ
  pennies : ℕ

/-- The conditions of Leah's coin collection -/
def valid_collection (c : CoinCollection) : Prop :=
  c.nickels + c.pennies = total_coins ∧
  c.nickels + 1 = c.pennies

/-- The total value of a coin collection in cents -/
def collection_value (c : CoinCollection) : ℕ :=
  c.nickels * nickel_value + c.pennies * penny_value

/-- The main theorem stating that Leah's coins are worth 43 cents -/
theorem leahs_coins_value (c : CoinCollection) :
  valid_collection c → collection_value c = 43 := by
  sorry

end leahs_coins_value_l3240_324016


namespace sin_45_degrees_l3240_324032

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end sin_45_degrees_l3240_324032


namespace meeting_point_relationship_l3240_324048

/-- Represents the scenario of two vehicles meeting on a road --/
structure MeetingScenario where
  S : ℝ  -- Distance between village and city
  x : ℝ  -- Speed of the truck
  y : ℝ  -- Speed of the car
  t : ℝ  -- Time taken to meet under normal conditions
  t1 : ℝ  -- Time taken to meet if truck leaves 45 minutes earlier
  t2 : ℝ  -- Time taken to meet if car leaves 20 minutes earlier

/-- The theorem stating the relationship between the meeting points --/
theorem meeting_point_relationship (scenario : MeetingScenario) :
  scenario.t = scenario.S / (scenario.x + scenario.y) →
  scenario.t1 = (scenario.S - 0.75 * scenario.x) / (scenario.x + scenario.y) →
  scenario.t2 = (scenario.S - scenario.y / 3) / (scenario.x + scenario.y) →
  0.75 * scenario.x + (scenario.S - 0.75 * scenario.x) * scenario.x / (scenario.x + scenario.y) - scenario.S * scenario.x / (scenario.x + scenario.y) = 18 →
  scenario.S * scenario.x / (scenario.x + scenario.y) - (scenario.S - scenario.y / 3) * scenario.x / (scenario.x + scenario.y) = 8 :=
by sorry

end meeting_point_relationship_l3240_324048


namespace square_diff_product_plus_square_equals_five_l3240_324059

theorem square_diff_product_plus_square_equals_five 
  (a b : ℝ) (ha : a = Real.sqrt 2 + 1) (hb : b = Real.sqrt 2 - 1) : 
  a^2 - a*b + b^2 = 5 := by sorry

end square_diff_product_plus_square_equals_five_l3240_324059


namespace min_area_square_on_parabola_l3240_324033

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a parabola y = x^2 -/
def OnParabola (p : Point) : Prop :=
  p.y = p.x^2

/-- Defines a square with three vertices on a parabola -/
structure SquareOnParabola where
  A : Point
  B : Point
  C : Point
  onParabola : OnParabola A ∧ OnParabola B ∧ OnParabola C
  isSquare : (A.x - B.x)^2 + (A.y - B.y)^2 = (B.x - C.x)^2 + (B.y - C.y)^2

/-- The area of a square given its side length -/
def SquareArea (sideLength : ℝ) : ℝ :=
  sideLength^2

/-- Theorem: The minimum area of a square with three vertices on the parabola y = x^2 is 2 -/
theorem min_area_square_on_parabola :
  ∀ s : SquareOnParabola, SquareArea (Real.sqrt ((s.A.x - s.B.x)^2 + (s.A.y - s.B.y)^2)) ≥ 2 :=
by sorry

end min_area_square_on_parabola_l3240_324033


namespace expression_factorization_l3240_324000

theorem expression_factorization (a b c x : ℝ) :
  (x - a)^2 * (b - c) + (x - b)^2 * (c - a) + (x - c)^2 * (a - b) = -(a - b) * (b - c) * (c - a) := by
  sorry

end expression_factorization_l3240_324000


namespace polynomial_symmetry_l3240_324049

/-- Given a polynomial g(x) = ax^2 + bx^3 + cx + d where g(-3) = 2, prove that g(3) = 0 -/
theorem polynomial_symmetry (a b c d : ℝ) (g : ℝ → ℝ) 
  (h1 : ∀ x, g x = a * x^2 + b * x^3 + c * x + d)
  (h2 : g (-3) = 2) : 
  g 3 = 0 := by sorry

end polynomial_symmetry_l3240_324049


namespace min_trios_l3240_324024

/-- Represents a group of people in a meeting -/
structure Meeting :=
  (people : Finset Nat)
  (handshakes : Set (Nat × Nat))
  (size_eq : people.card = 5)

/-- Defines a trio in the meeting -/
def is_trio (m : Meeting) (a b c : Nat) : Prop :=
  (a ∈ m.people ∧ b ∈ m.people ∧ c ∈ m.people) ∧
  ((⟨a, b⟩ ∈ m.handshakes ∧ ⟨b, c⟩ ∈ m.handshakes) ∨
   (⟨a, b⟩ ∉ m.handshakes ∧ ⟨b, c⟩ ∉ m.handshakes))

/-- Counts the number of unique trios in the meeting -/
def count_trios (m : Meeting) : Nat :=
  (m.people.powerset.filter (fun s => s.card = 3)).card

/-- The main theorem stating the minimum number of trios -/
theorem min_trios (m : Meeting) : 
  ∃ (handshakes : Set (Nat × Nat)), count_trios { people := m.people, handshakes := handshakes, size_eq := m.size_eq } = 10 ∧ 
  ∀ (other_handshakes : Set (Nat × Nat)), count_trios { people := m.people, handshakes := other_handshakes, size_eq := m.size_eq } ≥ 10 :=
sorry

end min_trios_l3240_324024


namespace person_b_work_days_l3240_324084

/-- Given that person A can complete a work in 30 days, and together with person B
    they complete 2/9 of the work in 4 days, prove that person B can complete
    the work alone in 45 days. -/
theorem person_b_work_days (a_days : ℕ) (combined_work : ℚ) (combined_days : ℕ) :
  a_days = 30 →
  combined_work = 2 / 9 →
  combined_days = 4 →
  ∃ b_days : ℕ,
    b_days = 45 ∧
    combined_work = combined_days * (1 / a_days + 1 / b_days) :=
by sorry

end person_b_work_days_l3240_324084


namespace rainy_day_probability_l3240_324069

theorem rainy_day_probability (A B : Set ℝ) (P : Set ℝ → ℝ) 
  (hA : P A = 0.06)
  (hB : P B = 0.08)
  (hAB : P (A ∩ B) = 0.02) :
  P B / P A = 1/3 :=
sorry

end rainy_day_probability_l3240_324069


namespace sum_of_digits_up_to_5000_l3240_324008

def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def sumOfDigitsUpTo (n : ℕ) : ℕ :=
  (List.range n).map sumOfDigits |>.sum

theorem sum_of_digits_up_to_5000 : 
  sumOfDigitsUpTo 5000 = 167450 := by
  sorry

end sum_of_digits_up_to_5000_l3240_324008


namespace nine_five_dollar_bills_equal_45_l3240_324019

/-- Calculates the total amount of money given the number of five-dollar bills -/
def total_money (num_bills : ℕ) : ℕ := 5 * num_bills

/-- Theorem stating that 9 five-dollar bills equal $45 -/
theorem nine_five_dollar_bills_equal_45 :
  total_money 9 = 45 := by sorry

end nine_five_dollar_bills_equal_45_l3240_324019


namespace ratio_of_powers_l3240_324037

theorem ratio_of_powers (p q : ℝ) (n : ℕ) (h1 : n > 1) (h2 : p^n / q^n = 7) :
  (p^n + q^n) / (p^n - q^n) = 4/3 := by
  sorry

end ratio_of_powers_l3240_324037


namespace line_equation_correct_l3240_324080

/-- A line in 2D space -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Check if a point (x, y) is on the line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y - l.point.2 = l.slope * (x - l.point.1)

/-- The specific line l with slope 2 passing through (2, -1) -/
def l : Line :=
  { slope := 2
  , point := (2, -1) }

/-- Theorem: The equation 2x - y - 5 = 0 represents the line l -/
theorem line_equation_correct :
  ∀ x y : ℝ, l.contains x y ↔ 2 * x - y - 5 = 0 := by
  sorry

end line_equation_correct_l3240_324080


namespace selection_probabilities_l3240_324066

/-- Represents the probabilities of passing selections for a student -/
structure StudentProb where
  first : ℝ  -- Probability of passing first selection
  second : ℝ  -- Probability of passing second selection

/-- Given probabilities for students A, B, and C, prove the required probabilities -/
theorem selection_probabilities (a b c : StudentProb)
  (ha_first : a.first = 0.5) (ha_second : a.second = 0.6)
  (hb_first : b.first = 0.6) (hb_second : b.second = 0.5)
  (hc_first : c.first = 0.4) (hc_second : c.second = 0.5) :
  (a.first * (1 - b.first) = 0.2) ∧
  (a.first * a.second * (1 - b.first * b.second) * (1 - c.first * c.second) +
   (1 - a.first * a.second) * b.first * b.second * (1 - c.first * c.second) +
   (1 - a.first * a.second) * (1 - b.first * b.second) * c.first * c.second = 217 / 500) :=
by sorry


end selection_probabilities_l3240_324066


namespace jaeho_received_most_notebooks_l3240_324021

def notebooks_given : ℕ := 30
def jaehyuk_notebooks : ℕ := 12
def kyunghwan_notebooks : ℕ := 3
def jaeho_notebooks : ℕ := 15

theorem jaeho_received_most_notebooks :
  jaeho_notebooks > jaehyuk_notebooks ∧ jaeho_notebooks > kyunghwan_notebooks :=
by sorry

end jaeho_received_most_notebooks_l3240_324021
