import Mathlib

namespace binomial_expansion_coefficients_l2568_256850

theorem binomial_expansion_coefficients :
  ∃ (b₁ b₂ b₃ b₄ : ℝ),
    (∀ x : ℝ, x^4 = (x+1)^4 + b₁*(x+1)^3 + b₂*(x+1)^2 + b₃*(x+1) + b₄) ∧
    b₁ = -4 ∧ b₂ = 6 ∧ b₃ = -4 ∧ b₄ = 1 := by
  sorry

end binomial_expansion_coefficients_l2568_256850


namespace rectangle_to_square_l2568_256865

theorem rectangle_to_square (k : ℕ) (h1 : k > 7) :
  (∃ (n : ℕ), k * (k - 7) = n^2) → (∃ (n : ℕ), k * (k - 7) = n^2 ∧ n = 24) :=
by sorry

end rectangle_to_square_l2568_256865


namespace num_quadrilaterals_is_495_l2568_256816

/-- The number of ways to choose 4 points from 12 distinct points on a circle's circumference to form convex quadrilaterals -/
def num_quadrilaterals : ℕ := Nat.choose 12 4

/-- Theorem stating that the number of quadrilaterals is 495 -/
theorem num_quadrilaterals_is_495 : num_quadrilaterals = 495 := by
  sorry

end num_quadrilaterals_is_495_l2568_256816


namespace second_box_clay_capacity_l2568_256829

/-- Represents the dimensions and clay capacity of a box -/
structure Box where
  height : ℝ
  width : ℝ
  length : ℝ
  clayCapacity : ℝ

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℝ := b.height * b.width * b.length

/-- Theorem stating the clay capacity of the second box -/
theorem second_box_clay_capacity 
  (box1 : Box)
  (box2 : Box)
  (h1 : box1.height = 4)
  (h2 : box1.width = 3)
  (h3 : box1.length = 7)
  (h4 : box1.clayCapacity = 84)
  (h5 : box2.height = box1.height / 2)
  (h6 : box2.width = box1.width * 4)
  (h7 : box2.length = box1.length)
  (h8 : boxVolume box1 * box1.clayCapacity = boxVolume box2 * box2.clayCapacity) :
  box2.clayCapacity = 168 := by
  sorry


end second_box_clay_capacity_l2568_256829


namespace state_tax_rate_is_4_percent_l2568_256856

/-- Calculates the state tax rate given the following conditions:
  * The taxpayer was a resident for 9 months out of the year
  * The taxpayer's taxable income for the year
  * The prorated tax amount paid for the time of residency
-/
def calculate_state_tax_rate (months_resident : ℕ) (taxable_income : ℚ) (tax_paid : ℚ) : ℚ :=
  let full_year_months : ℕ := 12
  let residence_ratio : ℚ := months_resident / full_year_months
  let full_year_tax : ℚ := tax_paid / residence_ratio
  (full_year_tax / taxable_income) * 100

theorem state_tax_rate_is_4_percent :
  let months_resident : ℕ := 9
  let taxable_income : ℚ := 42500
  let tax_paid : ℚ := 1275
  calculate_state_tax_rate months_resident taxable_income tax_paid = 4 := by
  sorry

end state_tax_rate_is_4_percent_l2568_256856


namespace desert_area_changes_l2568_256808

/-- Represents the desert area problem -/
structure DesertArea where
  initial_area : ℝ  -- Initial desert area in 1997
  annual_increase : ℝ  -- Annual increase in desert area
  afforestation_rate : ℝ  -- Annual reduction due to afforestation measures

/-- Calculates the desert area after a given number of years without afforestation -/
def area_after_years (d : DesertArea) (years : ℕ) : ℝ :=
  d.initial_area + d.annual_increase * years

/-- Calculates the desert area after a given number of years with afforestation -/
def area_with_afforestation (d : DesertArea) (years : ℕ) : ℝ :=
  d.initial_area + d.annual_increase * years - d.afforestation_rate * years

/-- Main theorem about desert area changes -/
theorem desert_area_changes (d : DesertArea) 
    (h1 : d.initial_area = 9e5)
    (h2 : d.annual_increase = 2000)
    (h3 : d.afforestation_rate = 8000) :
    area_after_years d 23 = 9.46e5 ∧ 
    (∃ (y : ℕ), y ≤ 19 ∧ area_with_afforestation d y < 8e5 ∧ 
                ∀ (z : ℕ), z < y → area_with_afforestation d z ≥ 8e5) :=
  sorry


end desert_area_changes_l2568_256808


namespace mice_problem_l2568_256880

theorem mice_problem (x : ℕ) : 
  (x / 2 : ℕ) * 2 = x ∧ 
  ((x - x / 2) / 3 : ℕ) * 3 = x - x / 2 ∧
  (((x - x / 2) - (x - x / 2) / 3) / 4 : ℕ) * 4 = (x - x / 2) - (x - x / 2) / 3 ∧
  ((((x - x / 2) - (x - x / 2) / 3) - ((x - x / 2) - (x - x / 2) / 3) / 4) / 5 : ℕ) * 5 = 
    ((x - x / 2) - (x - x / 2) / 3) - ((x - x / 2) - (x - x / 2) / 3) / 4 ∧
  ((((x - x / 2) - (x - x / 2) / 3) - ((x - x / 2) - (x - x / 2) / 3) / 4) - 
    ((((x - x / 2) - (x - x / 2) / 3) - ((x - x / 2) - (x - x / 2) / 3) / 4) / 5)) = 
    (x - x / 2) / 3 + 2 →
  x = 60 := by
sorry

end mice_problem_l2568_256880


namespace A_disjoint_B_iff_l2568_256878

/-- The set A defined by the quadratic inequality -/
def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}

/-- The set B defined by the linear inequalities involving m -/
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

/-- Theorem stating the condition for A and B to be disjoint -/
theorem A_disjoint_B_iff (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end A_disjoint_B_iff_l2568_256878


namespace m_range_l2568_256889

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x, x^2 + m*x + 1 ≠ 0

def q (m : ℝ) : Prop := m > 2

-- Define the condition that either p or q is true, but not both
def condition (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬(p m ∧ q m)

-- State the theorem
theorem m_range (m : ℝ) : condition m → ((-2 < m ∧ m < 2) ∨ m > 2) := by
  sorry

end m_range_l2568_256889


namespace yellow_balls_count_l2568_256867

theorem yellow_balls_count (white_balls : ℕ) (total_balls : ℕ) 
  (h1 : white_balls = 4)
  (h2 : (white_balls : ℚ) / total_balls = 2 / 3) :
  total_balls - white_balls = 2 := by
  sorry

end yellow_balls_count_l2568_256867


namespace binary_to_quaternary_conversion_l2568_256890

theorem binary_to_quaternary_conversion : 
  (1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 2^0) = 
  (1 * 4^2 + 3 * 4^1 + 0 * 4^0) := by
  sorry

end binary_to_quaternary_conversion_l2568_256890


namespace mean_equals_n_l2568_256882

theorem mean_equals_n (n : ℝ) : 
  (17 + 98 + 39 + 54 + n) / 5 = n → n = 52 := by
  sorry

end mean_equals_n_l2568_256882


namespace sum_of_tens_and_units_digits_of_9_pow_2004_l2568_256840

/-- The sum of the tens digit and the units digit in the decimal representation of 9^2004 is 7. -/
theorem sum_of_tens_and_units_digits_of_9_pow_2004 : ∃ n : ℕ, 9^2004 = 100 * n + 61 :=
sorry

end sum_of_tens_and_units_digits_of_9_pow_2004_l2568_256840


namespace coin_grid_probability_l2568_256848

/-- Represents a square grid -/
structure Grid where
  size : ℕ  -- number of squares on each side
  square_size : ℝ  -- side length of each square
  
/-- Represents a circular coin -/
structure Coin where
  diameter : ℝ

/-- Calculates the probability of a coin landing in a winning position on a grid -/
def winning_probability (g : Grid) (c : Coin) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem coin_grid_probability :
  let g : Grid := { size := 5, square_size := 10 }
  let c : Coin := { diameter := 8 }
  winning_probability g c = 25 / 441 := by
  sorry

end coin_grid_probability_l2568_256848


namespace largest_consecutive_sum_35_l2568_256844

def sum_of_consecutive_integers (start : ℕ) (count : ℕ) : ℕ :=
  count * (2 * start + count - 1) / 2

theorem largest_consecutive_sum_35 :
  (∃ (start : ℕ), sum_of_consecutive_integers start 7 = 35) ∧
  (∀ (start count : ℕ), count > 7 → sum_of_consecutive_integers start count ≠ 35) :=
sorry

end largest_consecutive_sum_35_l2568_256844


namespace farmers_income_2010_l2568_256818

/-- Farmers' income in a given year -/
structure FarmerIncome where
  wage : ℝ
  other : ℝ

/-- Calculate farmers' income after n years -/
def futureIncome (initial : FarmerIncome) (n : ℕ) : ℝ :=
  initial.wage * (1 + 0.06) ^ n + (initial.other + n * 320)

theorem farmers_income_2010 :
  let initial : FarmerIncome := { wage := 3600, other := 2700 }
  let income2010 := futureIncome initial 5
  8800 ≤ income2010 ∧ income2010 < 9200 := by
  sorry

end farmers_income_2010_l2568_256818


namespace tangent_line_is_correct_l2568_256842

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

/-- The point on the curve -/
def point : ℝ × ℝ := (-1, -3)

/-- The proposed tangent line equation -/
def tangent_line (x y : ℝ) : Prop := 3*x + y + 6 = 0

theorem tangent_line_is_correct : 
  tangent_line point.1 point.2 ∧ 
  (∀ x : ℝ, tangent_line x (f x) → x = point.1) ∧
  f' point.1 = 3 :=
sorry

end tangent_line_is_correct_l2568_256842


namespace chip_thickness_comparison_l2568_256819

theorem chip_thickness_comparison : 
  let a : ℝ := (1/3) * Real.sin (1/2)
  let b : ℝ := (1/2) * Real.sin (1/3)
  let c : ℝ := (1/3) * Real.cos (7/8)
  c > b ∧ b > a := by sorry

end chip_thickness_comparison_l2568_256819


namespace cone_base_diameter_l2568_256849

/-- A cone with surface area 3π and lateral surface unfolding to a semicircle has base diameter 2 -/
theorem cone_base_diameter (r : ℝ) (l : ℝ) : 
  r > 0 → l > 0 → 
  (π * r^2 + π * r * l = 3 * π) → 
  (π * l = 2 * π * r) → 
  (2 * r = 2) := by
  sorry

end cone_base_diameter_l2568_256849


namespace point_on_extension_line_l2568_256833

/-- Given two points in a 2D plane and a third point on their extension line,
    prove that the third point has specific coordinates. -/
theorem point_on_extension_line (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (2, -1) →
  P₂ = (0, 5) →
  (∃ t : ℝ, t > 1 ∧ P = P₁ + t • (P₂ - P₁)) →
  ‖P - P₁‖ = 2 * ‖P - P₂‖ →
  P = (-2, 11) := by
  sorry


end point_on_extension_line_l2568_256833


namespace crazy_silly_school_difference_l2568_256846

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 15

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 14

/-- Theorem: The difference between the number of books and movies in the 'crazy silly school' series is 1 -/
theorem crazy_silly_school_difference : num_books - num_movies = 1 := by
  sorry

end crazy_silly_school_difference_l2568_256846


namespace angle_DAB_is_54_degrees_l2568_256837

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- A pentagon defined by five points -/
structure Pentagon :=
  (B : Point) (C : Point) (D : Point) (E : Point) (G : Point)

/-- The measure of an angle in degrees -/
def angle_measure (p q r : Point) : ℝ := sorry

/-- The distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Checks if a triangle is isosceles -/
def is_isosceles (t : Triangle) : Prop :=
  distance t.C t.A = distance t.C t.B

/-- Checks if a pentagon is regular -/
def is_regular_pentagon (p : Pentagon) : Prop := sorry

/-- Theorem: In an isosceles triangle with a regular pentagon constructed on one side,
    the angle DAB measures 54 degrees -/
theorem angle_DAB_is_54_degrees 
  (t : Triangle) 
  (p : Pentagon) 
  (h1 : is_isosceles t) 
  (h2 : is_regular_pentagon p)
  (h3 : p.B = t.B ∧ p.C = t.C)
  (D : Point) 
  : angle_measure D t.A t.B = 54 := by sorry

end angle_DAB_is_54_degrees_l2568_256837


namespace accuracy_of_rounded_number_l2568_256811

def is_accurate_to_hundreds_place (n : ℕ) : Prop :=
  n % 1000 ≠ 0 ∧ n % 100 = 0

theorem accuracy_of_rounded_number :
  ∀ (n : ℕ), 
    (31500 ≤ n ∧ n < 32500) →
    is_accurate_to_hundreds_place n :=
by
  sorry

end accuracy_of_rounded_number_l2568_256811


namespace gcd_of_three_numbers_l2568_256855

theorem gcd_of_three_numbers : Nat.gcd 17420 (Nat.gcd 23826 36654) = 2 := by
  sorry

end gcd_of_three_numbers_l2568_256855


namespace chocolate_division_l2568_256884

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_given : ℕ) :
  total_chocolate = 72 / 7 →
  num_piles = 6 →
  piles_given = 2 →
  piles_given * (total_chocolate / num_piles) = 24 / 7 := by
  sorry

end chocolate_division_l2568_256884


namespace shells_added_correct_l2568_256822

/-- The amount of shells added to a bucket -/
def shells_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that the difference between the final and initial amounts
    equals the amount of shells added -/
theorem shells_added_correct (initial final : ℕ) (h : final ≥ initial) :
  shells_added initial final = final - initial :=
by
  sorry

end shells_added_correct_l2568_256822


namespace divisibility_and_primality_l2568_256832

def ten_eights_base_nine : ℕ := 8 * 9^9 + 8 * 9^8 + 8 * 9^7 + 8 * 9^6 + 8 * 9^5 + 8 * 9^4 + 8 * 9^3 + 8 * 9^2 + 8 * 9^1 + 8 * 9^0

def twelve_eights_base_nine : ℕ := 8 * 9^11 + 8 * 9^10 + 8 * 9^9 + 8 * 9^8 + 8 * 9^7 + 8 * 9^6 + 8 * 9^5 + 8 * 9^4 + 8 * 9^3 + 8 * 9^2 + 8 * 9^1 + 8 * 9^0

def divisor1 : ℕ := 9^4 - 9^3 + 9^2 - 9 + 1
def divisor2 : ℕ := 9^4 - 9^2 + 1

theorem divisibility_and_primality :
  (ten_eights_base_nine % divisor1 = 0) ∧
  (twelve_eights_base_nine % divisor2 = 0) ∧
  (¬ Nat.Prime divisor1) ∧
  (Nat.Prime divisor2) :=
by sorry

end divisibility_and_primality_l2568_256832


namespace employment_calculation_l2568_256843

/-- The percentage of employed people in the population of town X -/
def employed_percentage : ℝ := sorry

/-- The percentage of the population that are employed males -/
def employed_males_percentage : ℝ := 15

/-- The percentage of employed people who are females -/
def employed_females_percentage : ℝ := 75

theorem employment_calculation :
  employed_percentage = 60 :=
by
  sorry

end employment_calculation_l2568_256843


namespace inequality_proofs_l2568_256847

theorem inequality_proofs :
  (∀ x : ℝ, 4*x - 2 < 1 - 2*x → x < 1/2) ∧
  (∀ x : ℝ, 3 - 2*x ≥ x - 6 ∧ (3*x + 1)/2 < 2*x → 1 < x ∧ x ≤ 3) :=
by sorry

end inequality_proofs_l2568_256847


namespace peach_count_l2568_256859

theorem peach_count (initial : ℕ) (picked : ℕ) (total : ℕ) : 
  initial = 34 → picked = 52 → total = initial + picked → total = 86 := by
sorry

end peach_count_l2568_256859


namespace kamals_chemistry_marks_l2568_256870

theorem kamals_chemistry_marks 
  (english_marks : ℕ)
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (h1 : english_marks = 96)
  (h2 : math_marks = 65)
  (h3 : physics_marks = 82)
  (h4 : biology_marks = 85)
  (h5 : average_marks = 79)
  (h6 : num_subjects = 5)
  : ∃ (chemistry_marks : ℕ), 
    (english_marks + math_marks + physics_marks + biology_marks + chemistry_marks) / num_subjects = average_marks ∧ 
    chemistry_marks = 67 :=
by sorry

end kamals_chemistry_marks_l2568_256870


namespace apps_deleted_l2568_256869

theorem apps_deleted (initial_apps final_apps : ℝ) 
  (h1 : initial_apps = 300.5)
  (h2 : final_apps = 129.5) :
  initial_apps - final_apps = 171 := by
  sorry

end apps_deleted_l2568_256869


namespace perimeter_is_24_l2568_256885

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/36 + y^2/25 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- State that A and B are on the ellipse
axiom A_on_ellipse : ellipse A.1 A.2
axiom B_on_ellipse : ellipse B.1 B.2

-- State that A, B, and F₁ are collinear
axiom A_B_F₁_collinear : ∃ (t : ℝ), A = F₁ + t • (B - F₁) ∨ B = F₁ + t • (A - F₁)

-- Define the perimeter of triangle ABF₂
def perimeter_ABF₂ : ℝ := sorry

-- Theorem to prove
theorem perimeter_is_24 : perimeter_ABF₂ = 24 := by sorry

end perimeter_is_24_l2568_256885


namespace parallel_planes_normal_vectors_l2568_256860

/-- Given two planes α and β with normal vectors, prove that if they are parallel, then k = 4 -/
theorem parallel_planes_normal_vectors (k : ℝ) : 
  let n_alpha : ℝ × ℝ × ℝ := (1, 2, -2)
  let n_beta : ℝ × ℝ × ℝ := (-2, -4, k)
  (∃ (c : ℝ), c ≠ 0 ∧ n_alpha = c • n_beta) → k = 4 := by
  sorry

end parallel_planes_normal_vectors_l2568_256860


namespace derivative_f_derivative_g_l2568_256817

noncomputable section

open Real

-- Function 1
def f (x : ℝ) : ℝ := (1 / Real.sqrt x) * cos x

-- Function 2
def g (x : ℝ) : ℝ := 5 * x^10 * sin x - 2 * Real.sqrt x * cos x - 9

-- Theorem for the derivative of function 1
theorem derivative_f (x : ℝ) (hx : x > 0) :
  deriv f x = -(cos x + 2 * x * sin x) / (2 * x * Real.sqrt x) :=
sorry

-- Theorem for the derivative of function 2
theorem derivative_g (x : ℝ) (hx : x > 0) :
  deriv g x = 50 * x^9 * sin x + 5 * x^10 * cos x - (Real.sqrt x * cos x) / x + 2 * Real.sqrt x * sin x :=
sorry

end derivative_f_derivative_g_l2568_256817


namespace shaded_quadrilateral_area_l2568_256824

theorem shaded_quadrilateral_area (s : ℝ) (a b : ℝ) : 
  s = 20 → a = 15 → b = 20 →
  s^2 - (1/2 * a * b) - (1/2 * (s * b / (a^2 + b^2).sqrt) * (s * a / (a^2 + b^2).sqrt)) = 154 :=
by sorry

end shaded_quadrilateral_area_l2568_256824


namespace vector_ratio_bounds_l2568_256881

theorem vector_ratio_bounds (a b : ℝ × ℝ) 
  (h1 : ‖a + b‖ = 3)
  (h2 : ‖a - b‖ = 2) :
  (2 / 5 : ℝ) ≤ ‖a‖ / (a.1 * b.1 + a.2 * b.2) ∧ 
  ‖a‖ / (a.1 * b.1 + a.2 * b.2) ≤ 2 :=
sorry

end vector_ratio_bounds_l2568_256881


namespace max_value_f_times_g_l2568_256887

noncomputable def f (x : ℝ) : ℝ := 3 - x

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (2 * x + 5)

def is_non_negative (x : ℝ) : Prop := x ≥ 0

theorem max_value_f_times_g :
  ∃ (M : ℝ), M = 2 * Real.sqrt 3 - 1 ∧
  (∀ (x : ℝ), is_non_negative x →
    (f x * g x = min (f x) (g x)) →
    f x * g x ≤ M) ∧
  (∃ (x : ℝ), is_non_negative x ∧
    (f x * g x = min (f x) (g x)) ∧
    f x * g x = M) :=
sorry

end max_value_f_times_g_l2568_256887


namespace alloy_mixture_theorem_l2568_256836

/-- The percentage of chromium in the first alloy -/
def chromium_percent_1 : ℝ := 12

/-- The percentage of chromium in the second alloy -/
def chromium_percent_2 : ℝ := 8

/-- The amount of the first alloy used (in kg) -/
def amount_1 : ℝ := 20

/-- The percentage of chromium in the new alloy -/
def new_chromium_percent : ℝ := 9.454545454545453

/-- The amount of the second alloy used (in kg) -/
def amount_2 : ℝ := 35

theorem alloy_mixture_theorem :
  chromium_percent_1 * amount_1 / 100 + chromium_percent_2 * amount_2 / 100 =
  new_chromium_percent * (amount_1 + amount_2) / 100 := by
  sorry

end alloy_mixture_theorem_l2568_256836


namespace total_cost_calculation_l2568_256841

def calculate_total_cost (tv_price sound_price warranty_price install_price : ℝ)
  (tv_discount1 tv_discount2 sound_discount warranty_discount : ℝ)
  (tv_sound_tax warranty_install_tax : ℝ) : ℝ :=
  let tv_after_discounts := tv_price * (1 - tv_discount1) * (1 - tv_discount2)
  let sound_after_discount := sound_price * (1 - sound_discount)
  let warranty_after_discount := warranty_price * (1 - warranty_discount)
  let tv_with_tax := tv_after_discounts * (1 + tv_sound_tax)
  let sound_with_tax := sound_after_discount * (1 + tv_sound_tax)
  let warranty_with_tax := warranty_after_discount * (1 + warranty_install_tax)
  let install_with_tax := install_price * (1 + warranty_install_tax)
  tv_with_tax + sound_with_tax + warranty_with_tax + install_with_tax

theorem total_cost_calculation :
  calculate_total_cost 600 400 100 150 0.1 0.15 0.2 0.3 0.08 0.05 = 1072.32 := by
  sorry

end total_cost_calculation_l2568_256841


namespace barbaras_coin_collection_l2568_256845

/-- The total number of coins Barbara has -/
def total_coins : ℕ := 18

/-- The number of type A coins Barbara has -/
def type_A_coins : ℕ := 12

/-- The value of 8 type A coins in dollars -/
def value_8_type_A : ℕ := 24

/-- The value of 6 type B coins in dollars -/
def value_6_type_B : ℕ := 21

/-- The total worth of Barbara's entire collection in dollars -/
def total_worth : ℕ := 57

theorem barbaras_coin_collection :
  total_coins = type_A_coins + (total_coins - type_A_coins) ∧
  value_8_type_A / 8 * type_A_coins + value_6_type_B / 6 * (total_coins - type_A_coins) = total_worth :=
by sorry

end barbaras_coin_collection_l2568_256845


namespace chain_rule_with_local_injectivity_l2568_256838

/-- Given two differentiable functions f and g, with f having a local injectivity property,
    prove that their composition is differentiable and satisfies the chain rule. -/
theorem chain_rule_with_local_injectivity 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (x₀ : ℝ) 
  (hf : DifferentiableAt ℝ f x₀)
  (hg : DifferentiableAt ℝ g (f x₀))
  (hU : ∃ U : Set ℝ, IsOpen U ∧ x₀ ∈ U ∧ ∀ x ∈ U, x ≠ x₀ → f x ≠ f x₀) :
  DifferentiableAt ℝ (g ∘ f) x₀ ∧ 
  deriv (g ∘ f) x₀ = deriv g (f x₀) * deriv f x₀ :=
by sorry

end chain_rule_with_local_injectivity_l2568_256838


namespace parabola_through_points_point_not_on_parabola_l2568_256825

/-- A parabola of the form y = ax² + bx passing through (1, 3) and (-1, -1) -/
def Parabola (a b : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x

theorem parabola_through_points (a b : ℝ) :
  Parabola a b 1 = 3 ∧ Parabola a b (-1) = -1 → Parabola a b = λ x => x^2 + 2*x :=
sorry

theorem point_not_on_parabola :
  Parabola 1 2 2 ≠ 6 :=
sorry

end parabola_through_points_point_not_on_parabola_l2568_256825


namespace total_lives_calculation_l2568_256873

theorem total_lives_calculation (initial_players : ℕ) (new_players : ℕ) (lives_per_player : ℕ) : 
  initial_players = 16 → new_players = 4 → lives_per_player = 10 →
  (initial_players + new_players) * lives_per_player = 200 := by
sorry

end total_lives_calculation_l2568_256873


namespace fraction_inequality_l2568_256866

theorem fraction_inequality (a b m : ℝ) (h1 : b > a) (h2 : m > 0) :
  b / a > (b + m) / (a + m) := by
  sorry

end fraction_inequality_l2568_256866


namespace unique_divisor_square_equality_l2568_256877

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := (Nat.divisors n.val).card

/-- Theorem: The only positive integer n that satisfies n = [d(n)]^2 is 1 -/
theorem unique_divisor_square_equality :
  ∀ n : ℕ+, n.val = (num_divisors n)^2 → n = 1 := by
  sorry

end unique_divisor_square_equality_l2568_256877


namespace lcm_16_24_45_l2568_256883

theorem lcm_16_24_45 : Nat.lcm (Nat.lcm 16 24) 45 = 720 := by
  sorry

end lcm_16_24_45_l2568_256883


namespace hawks_score_l2568_256831

def total_points : ℕ := 82
def margin : ℕ := 22

theorem hawks_score (eagles_score hawks_score : ℕ) 
  (h1 : eagles_score + hawks_score = total_points)
  (h2 : eagles_score = hawks_score + margin) : 
  hawks_score = 30 := by
  sorry

end hawks_score_l2568_256831


namespace abc_sum_zero_product_nonpositive_l2568_256894

theorem abc_sum_zero_product_nonpositive (a b c : ℝ) (h : a + b + c = 0) :
  (∀ x ≤ 0, ∃ a b c : ℝ, a + b + c = 0 ∧ a * b + a * c + b * c = x) ∧
  (∀ a b c : ℝ, a + b + c = 0 → a * b + a * c + b * c ≤ 0) := by
  sorry

end abc_sum_zero_product_nonpositive_l2568_256894


namespace output_after_five_years_l2568_256801

/-- The output value after n years of growth at a given rate -/
def output_after_n_years (initial_value : ℝ) (growth_rate : ℝ) (n : ℕ) : ℝ :=
  initial_value * (1 + growth_rate) ^ n

/-- Theorem: The output value after 5 years with 10% annual growth -/
theorem output_after_five_years (a : ℝ) :
  output_after_n_years a 0.1 5 = 1.1^5 * a := by
  sorry

end output_after_five_years_l2568_256801


namespace largest_number_with_conditions_l2568_256804

theorem largest_number_with_conditions : ∃ n : ℕ, n = 93 ∧
  n < 100 ∧
  n % 8 = 5 ∧
  n % 3 = 0 ∧
  ∀ m : ℕ, m < 100 → m % 8 = 5 → m % 3 = 0 → m ≤ n :=
by
  sorry

end largest_number_with_conditions_l2568_256804


namespace impossibility_proof_l2568_256802

def Square := Fin 4 → ℕ

def initial_state : Square := fun i => if i = 0 then 1 else 0

def S (state : Square) : ℤ :=
  state 0 - state 1 + state 2 - state 3

def is_valid_move (before after : Square) : Prop :=
  ∃ (i : Fin 4) (k : ℕ), 
    after i + k = before i ∧
    after ((i + 1) % 4) = before ((i + 1) % 4) + k ∧
    after ((i + 3) % 4) = before ((i + 3) % 4) + k ∧
    (∀ j, j ≠ i ∧ j ≠ (i + 1) % 4 ∧ j ≠ (i + 3) % 4 → after j = before j)

def reachable (start goal : Square) : Prop :=
  ∃ (n : ℕ) (path : Fin (n + 1) → Square),
    path 0 = start ∧
    path n = goal ∧
    ∀ i : Fin n, is_valid_move (path i) (path (i + 1))

def target_state : Square := fun i => 
  if i = 0 then 1
  else if i = 1 then 9
  else if i = 2 then 8
  else 9

theorem impossibility_proof :
  ¬(reachable initial_state target_state) :=
sorry

end impossibility_proof_l2568_256802


namespace least_valid_tree_count_l2568_256857

def is_valid_tree_count (n : ℕ) : Prop :=
  n ≥ 100 ∧ n % 7 = 0 ∧ n % 6 = 0 ∧ n % 4 = 0

theorem least_valid_tree_count :
  ∃ (n : ℕ), is_valid_tree_count n ∧ ∀ m < n, ¬is_valid_tree_count m :=
by sorry

end least_valid_tree_count_l2568_256857


namespace min_value_expression_l2568_256830

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  b / (3 * a) + 3 / b ≥ 5 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ b₀ / (3 * a₀) + 3 / b₀ = 5 :=
by sorry

end min_value_expression_l2568_256830


namespace largest_odd_five_digit_has_2_in_hundreds_place_l2568_256823

def Digits : Finset Nat := {1, 2, 3, 5, 8}

def is_odd (n : Nat) : Prop := n % 2 = 1

def is_five_digit (n : Nat) : Prop := 10000 ≤ n ∧ n < 100000

def uses_all_digits (n : Nat) (digits : Finset Nat) : Prop :=
  (Finset.card digits = 5) ∧
  (∀ d ∈ digits, ∃ i : Nat, (n / (10^i)) % 10 = d) ∧
  (∀ i : Nat, i < 5 → (n / (10^i)) % 10 ∈ digits)

def largest_odd_five_digit (n : Nat) : Prop :=
  is_odd n ∧
  is_five_digit n ∧
  uses_all_digits n Digits ∧
  ∀ m : Nat, is_odd m ∧ is_five_digit m ∧ uses_all_digits m Digits → m ≤ n

theorem largest_odd_five_digit_has_2_in_hundreds_place :
  ∃ n : Nat, largest_odd_five_digit n ∧ (n / 100) % 10 = 2 := by
  sorry

end largest_odd_five_digit_has_2_in_hundreds_place_l2568_256823


namespace value_of_m_l2568_256810

theorem value_of_m (m : ℕ) : 
  (((1 : ℚ) ^ m) / (5 ^ m)) * (((1 : ℚ) ^ 16) / (4 ^ 16)) = 1 / (2 * (10 ^ 31)) → 
  m = 31 := by
sorry

end value_of_m_l2568_256810


namespace jack_morning_emails_l2568_256812

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 10

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 7

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 17

/-- Theorem stating that Jack received 10 emails in the morning -/
theorem jack_morning_emails : 
  morning_emails = afternoon_emails + 3 := by sorry

end jack_morning_emails_l2568_256812


namespace arithmetic_sequence_sum_l2568_256871

/-- Given an arithmetic sequence with first term 5, second term 12, and last term 40,
    the sum of the two terms immediately preceding 40 is 59. -/
theorem arithmetic_sequence_sum (a : ℕ → ℕ) : 
  a 0 = 5 → a 1 = 12 → 
  (∃ n : ℕ, a n = 40 ∧ ∀ k < n, a k < 40) →
  (∀ i j k : ℕ, i < j → j < k → a j - a i = a k - a j) →
  (∃ m : ℕ, a m + a (m + 1) = 59 ∧ a (m + 2) = 40) :=
by sorry

end arithmetic_sequence_sum_l2568_256871


namespace natural_number_representation_l2568_256892

theorem natural_number_representation (A : ℕ) :
  ∃ n : ℕ, A = 3 * n ∨ A = 3 * n + 1 ∨ A = 3 * n + 2 := by
  sorry

end natural_number_representation_l2568_256892


namespace laundry_dishes_time_difference_l2568_256888

theorem laundry_dishes_time_difference 
  (dawn_dish_time andy_laundry_time : ℕ) 
  (h1 : dawn_dish_time = 20) 
  (h2 : andy_laundry_time = 46) : 
  andy_laundry_time - 2 * dawn_dish_time = 6 := by
  sorry

end laundry_dishes_time_difference_l2568_256888


namespace adult_tickets_sold_l2568_256893

theorem adult_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_receipts ∧
    adult_tickets = 40 := by
  sorry

end adult_tickets_sold_l2568_256893


namespace ratio_HB_JD_l2568_256876

-- Define the points
variable (A B C D E F G H J : ℝ × ℝ)

-- Define the conditions
axiom on_line : ∃ (t : ℝ), B = A + t • (F - A) ∧
                           C = A + (t + 1) • (F - A) ∧
                           D = A + (t + 3) • (F - A) ∧
                           E = A + (t + 4) • (F - A) ∧
                           F = A + (t + 5) • (F - A)

axiom segment_lengths : 
  dist A B = 1 ∧ dist B C = 2 ∧ dist C D = 1 ∧ dist D E = 2 ∧ dist E F = 1

axiom G_not_on_line : ∀ (t : ℝ), G ≠ A + t • (F - A)

axiom H_on_GD : ∃ (t : ℝ), H = G + t • (D - G)

axiom J_on_GE : ∃ (t : ℝ), J = G + t • (E - G)

axiom parallel_lines : 
  (H.2 - B.2) / (H.1 - B.1) = (J.2 - D.2) / (J.1 - D.1) ∧
  (J.2 - D.2) / (J.1 - D.1) = (G.2 - A.2) / (G.1 - A.1)

-- Theorem to prove
theorem ratio_HB_JD : dist H B / dist J D = 5 / 4 :=
sorry

end ratio_HB_JD_l2568_256876


namespace pen_distribution_l2568_256827

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem pen_distribution (num_pencils : ℕ) (num_pens : ℕ) 
  (h1 : num_pencils = 1203)
  (h2 : is_prime num_pencils)
  (h3 : ∀ (students : ℕ), students > 1 → ¬(num_pencils % students = 0 ∧ num_pens % students = 0)) :
  ∃ (n : ℕ), num_pens = n :=
sorry

end pen_distribution_l2568_256827


namespace expression_evaluation_l2568_256899

theorem expression_evaluation (m : ℝ) (h : m = 1) : 
  (1 - 1 / (m - 2)) / ((m^2 - 6*m + 9) / (m - 2)) = -1/2 := by
  sorry

end expression_evaluation_l2568_256899


namespace participants_with_three_points_l2568_256891

/-- Represents the number of participants in a tennis tournament with a specific score -/
def participantsWithScore (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Represents the total number of participants in the tournament -/
def totalParticipants (n : ℕ) : ℕ := 2^n + 4

/-- Theorem stating the number of participants with exactly 3 points at the end of the tournament -/
theorem participants_with_three_points (n : ℕ) (h : n > 4) :
  ∃ (winner : ℕ), winner = participantsWithScore n 3 + 1 ∧
  winner ≤ totalParticipants n :=
by sorry

end participants_with_three_points_l2568_256891


namespace souvenir_relationship_l2568_256879

/-- Represents the number of souvenirs of each type -/
structure SouvenirCount where
  x : ℕ  -- 20 cents souvenirs
  y : ℕ  -- 25 cents souvenirs
  z : ℕ  -- 35 cents souvenirs

/-- Conditions of the souvenir distribution problem -/
def SouvenirProblem (s : SouvenirCount) : Prop :=
  s.x + s.y + s.z = 2000 ∧
  20 * s.x + 25 * s.y + 35 * s.z = 52000

/-- Theorem stating the relationship between 25 cents and 35 cents souvenirs -/
theorem souvenir_relationship (s : SouvenirCount) 
  (h : SouvenirProblem s) : 5 * s.y + 15 * s.z = 12000 := by
  sorry

end souvenir_relationship_l2568_256879


namespace product_of_real_parts_l2568_256868

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z : ℂ) : Prop :=
  z^2 + 2*z = 10 - 2*i

-- Define the roots of the quadratic equation
noncomputable def roots : Set ℂ :=
  {z : ℂ | quadratic_equation z}

-- State the theorem
theorem product_of_real_parts :
  ∃ (z₁ z₂ : ℂ), z₁ ∈ roots ∧ z₂ ∈ roots ∧ 
  (z₁.re * z₂.re : ℝ) = -10.25 :=
sorry

end product_of_real_parts_l2568_256868


namespace cake_price_calculation_l2568_256898

theorem cake_price_calculation (smoothie_price : ℝ) (smoothie_count : ℕ) (cake_count : ℕ) (total_revenue : ℝ) :
  smoothie_price = 3 →
  smoothie_count = 40 →
  cake_count = 18 →
  total_revenue = 156 →
  ∃ (cake_price : ℝ), cake_price = 2 ∧ smoothie_price * smoothie_count + cake_price * cake_count = total_revenue :=
by
  sorry

#check cake_price_calculation

end cake_price_calculation_l2568_256898


namespace total_harvest_earnings_l2568_256828

/-- Lewis's weekly earnings during the harvest -/
def weekly_earnings : ℕ := 2

/-- Duration of the harvest in weeks -/
def harvest_duration : ℕ := 89

/-- Theorem stating the total earnings for the harvest -/
theorem total_harvest_earnings :
  weekly_earnings * harvest_duration = 178 := by sorry

end total_harvest_earnings_l2568_256828


namespace triangle_angle_measure_l2568_256809

theorem triangle_angle_measure (A B C : ℝ) : 
  -- Triangle ABC
  A + B + C = 180 →
  -- Angle C is triple angle B
  C = 3 * B →
  -- Angle B is 15°
  B = 15 →
  -- Then angle A is 120°
  A = 120 := by
sorry

end triangle_angle_measure_l2568_256809


namespace equation_solution_l2568_256853

theorem equation_solution : ∃! x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by
  sorry

end equation_solution_l2568_256853


namespace parallel_line_point_l2568_256839

/-- Given two points on a line and another line it's parallel to, prove the x-coordinate of the second point. -/
theorem parallel_line_point (j : ℝ) : 
  (∃ (m b : ℝ), (2 : ℝ) + 3 * m = -6 ∧ 
                 (19 : ℝ) - (-3) = m * (j - 4)) → 
  j = -29 := by
sorry

end parallel_line_point_l2568_256839


namespace regular_bottle_is_16_oz_l2568_256863

/-- Represents Jon's drinking habits and fluid intake --/
structure DrinkingHabits where
  awake_hours : ℕ := 16
  drinking_interval : ℕ := 4
  larger_bottles_per_day : ℕ := 2
  larger_bottle_size_factor : ℚ := 1.25
  weekly_fluid_intake : ℕ := 728

/-- Calculates the size of Jon's regular water bottle in ounces --/
def regular_bottle_size (h : DrinkingHabits) : ℚ :=
  h.weekly_fluid_intake / (7 * (h.awake_hours / h.drinking_interval + h.larger_bottles_per_day * h.larger_bottle_size_factor))

/-- Theorem stating that Jon's regular water bottle size is 16 ounces --/
theorem regular_bottle_is_16_oz (h : DrinkingHabits) : regular_bottle_size h = 16 := by
  sorry

end regular_bottle_is_16_oz_l2568_256863


namespace solution_set_for_a_eq_2_area_implies_a_eq_1_l2568_256897

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2| + |x - a|

-- Theorem for the first part of the problem
theorem solution_set_for_a_eq_2 :
  {x : ℝ | f 2 x > 6} = {x : ℝ | x > 3 ∨ x < -3} := by sorry

-- Theorem for the second part of the problem
theorem area_implies_a_eq_1 (a : ℝ) (h : a > 0) :
  (∫ x in {x | f a x ≤ 5}, (5 - f a x)) = 8 → a = 1 := by sorry

end solution_set_for_a_eq_2_area_implies_a_eq_1_l2568_256897


namespace equation_solution_l2568_256813

theorem equation_solution : 
  ∃ x : ℚ, (x^2 + x + 1) / (x + 1) = x + 3 ∧ x = -2/3 := by
  sorry

end equation_solution_l2568_256813


namespace solve_apple_dealer_problem_l2568_256896

/-- Represents the apple dealer problem -/
def apple_dealer_problem (cost_per_bushel : ℚ) (apples_per_bushel : ℕ) (profit : ℚ) (apples_sold : ℕ) : Prop :=
  let cost_per_apple : ℚ := cost_per_bushel / apples_per_bushel
  let total_cost : ℚ := cost_per_apple * apples_sold
  let total_revenue : ℚ := total_cost + profit
  let price_per_apple : ℚ := total_revenue / apples_sold
  price_per_apple = 40 / 100

/-- Theorem stating the solution to the apple dealer problem -/
theorem solve_apple_dealer_problem :
  apple_dealer_problem 12 48 15 100 := by
  sorry

end solve_apple_dealer_problem_l2568_256896


namespace digit_57_is_5_l2568_256861

/-- The decimal expansion of 21/22 has a repeating pattern of "54" -/
def repeating_pattern : ℕ → ℕ
  | n => if n % 2 = 0 then 4 else 5

/-- The 57th digit after the decimal point in the expansion of 21/22 -/
def digit_57 : ℕ := repeating_pattern 56

theorem digit_57_is_5 : digit_57 = 5 := by
  sorry

end digit_57_is_5_l2568_256861


namespace lakeside_club_overlap_l2568_256874

/-- The number of students in both the theater and robotics clubs at Lakeside High School -/
def students_in_both_clubs (total_students theater_members robotics_members either_or_both : ℕ) : ℕ :=
  theater_members + robotics_members - either_or_both

/-- Theorem: Given the conditions from Lakeside High School, 
    the number of students in both the theater and robotics clubs is 25 -/
theorem lakeside_club_overlap : 
  let total_students : ℕ := 250
  let theater_members : ℕ := 85
  let robotics_members : ℕ := 120
  let either_or_both : ℕ := 180
  students_in_both_clubs total_students theater_members robotics_members either_or_both = 25 := by
  sorry

end lakeside_club_overlap_l2568_256874


namespace train_length_l2568_256805

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 60 * (1000 / 3600) →
  crossing_time = 18.598512119030477 →
  bridge_length = 200 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.001 := by
  sorry

end train_length_l2568_256805


namespace binomial_coefficient_20_10_l2568_256820

theorem binomial_coefficient_20_10 
  (h1 : Nat.choose 18 8 = 43758)
  (h2 : Nat.choose 18 9 = 48620)
  (h3 : Nat.choose 18 10 = 43758) : 
  Nat.choose 20 10 = 184756 := by
  sorry

end binomial_coefficient_20_10_l2568_256820


namespace wall_construction_boys_l2568_256886

/-- The number of boys who can construct the wall in 6 days -/
def num_boys : ℕ := 24

/-- The number of days it takes B boys or 24 girls to construct the wall -/
def days_boys_or_girls : ℕ := 6

/-- The number of days it takes B boys and 12 girls to construct the wall -/
def days_boys_and_girls : ℕ := 4

/-- The number of girls that can construct the wall in the same time as B boys -/
def equivalent_girls : ℕ := 24

theorem wall_construction_boys (B : ℕ) :
  (B * days_boys_or_girls = equivalent_girls * days_boys_or_girls) →
  ((B + 12 * equivalent_girls) * days_boys_and_girls = equivalent_girls * days_boys_or_girls) →
  B = num_boys :=
by sorry

end wall_construction_boys_l2568_256886


namespace perpendicular_line_l2568_256815

/-- Given a line L1 with equation 3x + 2y - 5 = 0 and a point P(1, -2),
    we define a line L2 with equation 2x - 3y - 8 = 0.
    This theorem states that L2 passes through P and is perpendicular to L1. -/
theorem perpendicular_line (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x + 2 * y - 5 = 0
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 2 * x - 3 * y - 8 = 0
  let P : ℝ × ℝ := (1, -2)
  (L2 P.1 P.2) ∧ 
  (∀ (x1 y1 x2 y2 : ℝ), L1 x1 y1 → L1 x2 y2 → L2 x1 y1 → L2 x2 y2 → 
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) * 
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) = 
    ((x2 - x1) * (y2 - y1) - (y2 - y1) * (x2 - x1)) * 
    ((x2 - x1) * (y2 - y1) - (y2 - y1) * (x2 - x1))) :=
by
  sorry


end perpendicular_line_l2568_256815


namespace perimeter_ABCDEFG_l2568_256806

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point2D) : ℝ := sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (m p1 p2 : Point2D) : Prop := sorry

/-- Calculate the perimeter of a polygon given by a list of points -/
def perimeter (points : List Point2D) : ℝ := sorry

/-- The main theorem -/
theorem perimeter_ABCDEFG :
  ∀ (A B C D E F G : Point2D),
    isEquilateral ⟨A, B, C⟩ →
    isEquilateral ⟨A, D, E⟩ →
    isEquilateral ⟨E, F, G⟩ →
    isMidpoint D A C →
    isMidpoint G A E →
    distance A B = 6 →
    perimeter [A, B, C, D, E, F, G] = 22.5 := by
  sorry

end perimeter_ABCDEFG_l2568_256806


namespace centroid_is_unique_interior_point_l2568_256852

/-- A point in the integer lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Predicate to check if a point is inside a triangle -/
def IsInside (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Predicate to check if a point is on the boundary of a triangle -/
def IsOnBoundary (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- The centroid of a triangle -/
def Centroid (t : LatticeTriangle) : LatticePoint := sorry

/-- Main theorem -/
theorem centroid_is_unique_interior_point (t : LatticeTriangle) 
  (h1 : ∀ p : LatticePoint, IsOnBoundary p t → p = t.A ∨ p = t.B ∨ p = t.C)
  (h2 : ∃! p : LatticePoint, IsInside p t) :
  ∃ p : LatticePoint, IsInside p t ∧ p = Centroid t := by
  sorry

end centroid_is_unique_interior_point_l2568_256852


namespace train_length_l2568_256895

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 10 → speed * time * (1000 / 3600) = 250 :=
by sorry

end train_length_l2568_256895


namespace pattern_equality_l2568_256872

theorem pattern_equality (n : ℕ+) : n * (n + 2) + 1 = (n + 1)^2 := by
  sorry

end pattern_equality_l2568_256872


namespace triangle_inequality_possible_third_side_length_l2568_256854

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

theorem possible_third_side_length : 
  ∃ (x : ℝ), x > 0 ∧ 3 + 6 > x ∧ 6 + x > 3 ∧ x + 3 > 6 ∧ x = 6 :=
by sorry

end triangle_inequality_possible_third_side_length_l2568_256854


namespace running_contest_average_distance_l2568_256821

/-- The average distance run by two people given their individual distances -/
def average_distance (d1 d2 : ℕ) : ℚ :=
  (d1 + d2) / 2

theorem running_contest_average_distance :
  let block_length : ℕ := 200
  let johnny_laps : ℕ := 4
  let mickey_laps : ℕ := johnny_laps / 2
  let johnny_distance : ℕ := johnny_laps * block_length
  let mickey_distance : ℕ := mickey_laps * block_length
  average_distance johnny_distance mickey_distance = 600 := by
sorry

end running_contest_average_distance_l2568_256821


namespace books_read_by_three_l2568_256834

/-- The number of different books read by three people given their individual book counts and overlap -/
def total_different_books (tony_books dean_books breanna_books tony_dean_overlap all_overlap : ℕ) : ℕ :=
  tony_books + dean_books + breanna_books - tony_dean_overlap - 2 * all_overlap

/-- Theorem stating the total number of different books read by Tony, Dean, and Breanna -/
theorem books_read_by_three :
  total_different_books 23 12 17 3 1 = 47 := by
  sorry

end books_read_by_three_l2568_256834


namespace min_pizzas_to_earn_back_car_cost_l2568_256803

/-- The cost of the car John bought -/
def car_cost : ℕ := 6500

/-- The amount John receives for each pizza delivered -/
def income_per_pizza : ℕ := 12

/-- The amount John spends on gas for each pizza delivered -/
def gas_cost_per_pizza : ℕ := 4

/-- The amount John spends on maintenance for each pizza delivered -/
def maintenance_cost_per_pizza : ℕ := 1

/-- The minimum whole number of pizzas John must deliver to earn back the car cost -/
def min_pizzas : ℕ := 929

theorem min_pizzas_to_earn_back_car_cost :
  ∀ n : ℕ, n ≥ min_pizzas →
    n * (income_per_pizza - gas_cost_per_pizza - maintenance_cost_per_pizza) ≥ car_cost ∧
    ∀ m : ℕ, m < min_pizzas →
      m * (income_per_pizza - gas_cost_per_pizza - maintenance_cost_per_pizza) < car_cost :=
by sorry

end min_pizzas_to_earn_back_car_cost_l2568_256803


namespace last_number_proof_l2568_256864

theorem last_number_proof (a b c d : ℝ) 
  (h1 : (a + b + c) / 3 = 6)
  (h2 : a + d = 13)
  (h3 : (b + c + d) / 3 = 3) : 
  d = 2 := by
sorry

end last_number_proof_l2568_256864


namespace solution_set_for_m_eq_1_m_range_for_inequality_l2568_256814

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m| + |2*x - 1|

-- Part 1
theorem solution_set_for_m_eq_1 :
  {x : ℝ | f 1 x ≥ 3} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} :=
sorry

-- Part 2
theorem m_range_for_inequality (m : ℝ) :
  (0 < m ∧ m < 1/4) →
  (∀ x ∈ Set.Icc m (2*m), (1/2) * (f m x) ≤ |x + 1|) →
  m ∈ Set.Ioo 0 (1/4) :=
sorry

end solution_set_for_m_eq_1_m_range_for_inequality_l2568_256814


namespace bus_intersection_percentages_l2568_256862

theorem bus_intersection_percentages : 
  let first_intersection_entrants : ℝ := 12
  let second_intersection_entrants : ℝ := 18
  let third_intersection_entrants : ℝ := 15
  (0.3 * first_intersection_entrants + 
   0.5 * second_intersection_entrants + 
   0.2 * third_intersection_entrants) = 15.6 := by
  sorry

end bus_intersection_percentages_l2568_256862


namespace jerry_weekly_earnings_l2568_256875

/-- Jerry's weekly earnings calculation --/
theorem jerry_weekly_earnings
  (rate_per_task : ℕ)
  (hours_per_task : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (h1 : rate_per_task = 40)
  (h2 : hours_per_task = 2)
  (h3 : hours_per_day = 10)
  (h4 : days_per_week = 7) :
  (rate_per_task * (hours_per_day / hours_per_task) * days_per_week : ℕ) = 1400 := by
  sorry

end jerry_weekly_earnings_l2568_256875


namespace arithmetic_calculation_l2568_256858

theorem arithmetic_calculation : 4 * 6 * 8 - 10 / 2 = 187 := by
  sorry

end arithmetic_calculation_l2568_256858


namespace roof_metal_bars_l2568_256800

/-- The number of sets of metal bars needed for the roof -/
def num_sets : ℕ := 2

/-- The number of metal bars in each set -/
def bars_per_set : ℕ := 7

/-- The total number of metal bars needed for the roof -/
def total_bars : ℕ := num_sets * bars_per_set

theorem roof_metal_bars : total_bars = 14 := by
  sorry

end roof_metal_bars_l2568_256800


namespace circle_problem_l2568_256807

-- Define the circles and points
def larger_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 15^2}
def smaller_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 10^2}
def P : ℝ × ℝ := (9, 12)
def S (k : ℝ) : ℝ × ℝ := (0, k)

-- Define the theorem
theorem circle_problem (k : ℝ) :
  P ∈ larger_circle ∧
  S k ∈ smaller_circle ∧
  (∀ p ∈ larger_circle, ∃ q ∈ smaller_circle, ‖p - q‖ = 5) →
  k = 10 := by
  sorry

end circle_problem_l2568_256807


namespace least_positive_angle_theorem_l2568_256851

theorem least_positive_angle_theorem : 
  ∃ θ : Real, θ > 0 ∧ θ = 15 * π / 180 ∧
  (∀ φ : Real, φ > 0 ∧ Real.cos (15 * π / 180) = Real.sin (45 * π / 180) + Real.sin φ → θ ≤ φ) :=
sorry

end least_positive_angle_theorem_l2568_256851


namespace calculate_swimming_speed_triathlete_swimming_speed_l2568_256835

/-- Calculates the swimming speed given the total distance, running speed, and average speed -/
theorem calculate_swimming_speed 
  (total_distance : ℝ) 
  (running_distance : ℝ) 
  (running_speed : ℝ) 
  (average_speed : ℝ) : ℝ :=
  let swimming_distance := total_distance - running_distance
  let total_time := total_distance / average_speed
  let running_time := running_distance / running_speed
  let swimming_time := total_time - running_time
  swimming_distance / swimming_time

/-- Proves that the swimming speed is 6 miles per hour given the problem conditions -/
theorem triathlete_swimming_speed : 
  calculate_swimming_speed 6 3 10 7.5 = 6 := by
  sorry

end calculate_swimming_speed_triathlete_swimming_speed_l2568_256835


namespace dog_bones_problem_l2568_256826

theorem dog_bones_problem (initial_bones : ℕ) : 
  initial_bones + 8 = 23 → initial_bones = 15 := by
  sorry

end dog_bones_problem_l2568_256826
